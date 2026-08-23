#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_path="${1:-${FORTAI_MODEL:-}}"
if [[ -z "$model_path" ]]; then
    echo "usage: benchmark/tournament_qwen35_cpu.sh MODEL.gguf [repeats] [token_id] [timing_steps] [context] [oracle_steps] [oracle_top_k] [oracle_tolerance]" >&2
    exit 2
fi
if [[ ! -f "$model_path" ]]; then
    echo "model not found: $model_path" >&2
    exit 2
fi
model_name=$(basename "$model_path")
case "$model_name" in
    Qwen3.5-0.8B-Q8_0.gguf|Qwen3.5-2B-Q8_0.gguf|Qwen3.5-4B-Q8_0.gguf) ;;
    *)
        echo "tournament model scope is limited to Qwen3.5 0.8B/2B/4B Q8_0: $model_name" >&2
        exit 2
        ;;
esac

repeats="${2:-${FORTAI_BENCH_REPEATS:-5}}"
token_id="${3:-${FORTAI_TOKEN_ID:-9419}}"
timing_steps="${4:-${FORTAI_BENCH_STEPS:-64}}"
context="${5:-${FORTAI_CONTEXT:-128}}"
oracle_steps="${6:-${FORTAI_LOGIT_STEPS:-8}}"
oracle_top_k="${7:-${FORTAI_LOGIT_TOP_K:-32}}"
oracle_tolerance="${8:-${FORTAI_LOGIT_TOLERANCE:-1.0e-2}}"
thread_list="${FORTAI_THREAD_LIST:-1 2 4 8 16 32}"
native_flags="${FORTAI_NATIVE_FLAGS:--O2 -march=native -mtune=native -funroll-loops -fopenmp -fno-fast-math -ffp-contract=off -fno-math-errno -flto}"
port="${LLAMA_PORT:-18081}"

if [[ ! "$repeats" =~ ^[0-9]+$ || "$repeats" -lt 5 ]]; then
    echo "tournament requires at least five repeats per thread: $repeats" >&2
    exit 2
fi
for value in "$token_id" "$timing_steps" "$context" "$oracle_steps" "$oracle_top_k"; do
    if [[ ! "$value" =~ ^[0-9]+$ || "$value" -le 0 ]]; then
        echo "tournament integer argument must be positive: $value" >&2
        exit 2
    fi
done
if [[ "${FORTAI_ALLOW_EXISTING_LLAMA_SERVER:-0}" == 1 ]]; then
    echo 'tournament refuses shared-service timing' >&2
    exit 2
fi
if pgrep -x llama-server >/dev/null 2>&1; then
    echo 'an existing llama-server prevents isolated tournament timing' >&2
    pgrep -a -x llama-server >&2 || true
    exit 2
fi
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${port}$"; then
    echo "tournament port is already in use: $port" >&2
    exit 2
fi

declare -A seen_threads=()
for threads in $thread_list; do
    if [[ ! "$threads" =~ ^[0-9]+$ || "$threads" -le 0 ]]; then
        echo "thread counts must be unique positive integers: $threads" >&2
        exit 2
    fi
    if [[ -n "${seen_threads[$threads]:-}" ]]; then
        echo "duplicate thread count: $threads" >&2
        exit 2
    fi
    seen_threads[$threads]=1
done

result_dir="$root_dir/benchmark/results"
log_dir="$root_dir/benchmark/logs"
mkdir -p "$result_dir" "$log_dir"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
base=$(basename "$model_path" .gguf)
index_file=$(mktemp)
trap 'rm -f "$index_file"' EXIT

for threads in $thread_list; do
    log_file="$log_dir/tournament_${base}_${threads}_${stamp}.log"
    OMP_NUM_THREADS="$threads" OMP_PROC_BIND="${OMP_PROC_BIND:-spread}" \
        OMP_PLACES="${OMP_PLACES:-cores}" FORTAI_NATIVE_FLAGS="$native_flags" \
        LLAMA_PORT="$port" "$root_dir/benchmark/repeat_compare_qwen35_cpu.sh" \
        "$model_path" "$repeats" "$token_id" "$timing_steps" "$context" >"$log_file" 2>&1
    summary_file=$(sed -n 's/^summary=//p' "$log_file" | tail -n 1)
    if [[ -z "$summary_file" || ! -f "$summary_file" ]]; then
        echo "tournament did not produce a summary for OMP_NUM_THREADS=$threads" >&2
        exit 1
    fi
    printf '%s\n' "$summary_file" >>"$index_file"
    echo "thread $threads/$thread_list complete: $summary_file"
done

best_thread=$(python3 - "$index_file" <<'PY'
import json
import sys
from pathlib import Path

paths = [line.strip() for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
items = [json.loads(Path(path).read_text()) for path in paths]
best = max(items, key=lambda item: float(item["fortai_matched_forward_steps_per_second"]["median"]))
print(best["omp_num_threads"])
PY
)

oracle_log="$log_dir/tournament_oracle_${base}_${stamp}.log"
oracle_file="$result_dir/tournament_oracle_${base}_${stamp}.json"
OMP_NUM_THREADS="$best_thread" OMP_PROC_BIND="${OMP_PROC_BIND:-spread}" \
    OMP_PLACES="${OMP_PLACES:-cores}" FORTAI_NATIVE_FLAGS="$native_flags" \
    FORTAI_CORRECTNESS_FLAGS="$native_flags" LLAMA_PORT="$port" \
    "$root_dir/benchmark/check_qwen35_cpu_logits.sh" "$model_path" "$token_id" \
    "$oracle_steps" "$context" "$oracle_top_k" "$oracle_tolerance" >"$oracle_log" 2>&1
oracle_payload=$(grep -E '^\{' "$oracle_log" | tail -n 1)
if [[ -z "$oracle_payload" ]]; then
    echo 'tournament behavioral oracle did not produce JSON' >&2
    exit 1
fi
printf '%s\n' "$oracle_payload" >"$oracle_file"

summary_file="$result_dir/tournament_${base}_${stamp}.json"
mapfile -t summary_paths <"$index_file"
python3 "$root_dir/benchmark/finalize_qwen35_cpu_tournament.py" \
    "${summary_paths[@]}" --oracle "$oracle_file" --output "$summary_file"
printf 'summary=%s\noracle=%s\n' "$summary_file" "$oracle_file"
