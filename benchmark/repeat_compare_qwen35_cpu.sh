#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_path="${1:-${FORTAI_MODEL:-}}"
if [[ -z "$model_path" ]]; then
    echo "usage: benchmark/repeat_compare_qwen35_cpu.sh MODEL.gguf [repeats] [token_id] [steps] [context]" >&2
    exit 2
fi
if [[ ! -f "$model_path" ]]; then
    echo "model not found: $model_path" >&2
    exit 2
fi

repeats="${2:-${FORTAI_BENCH_REPEATS:-5}}"
token_id="${3:-${FORTAI_TOKEN_ID:-9419}}"
steps="${4:-${FORTAI_BENCH_STEPS:-8}}"
context="${5:-${FORTAI_CONTEXT:-128}}"
port="${LLAMA_PORT:-18081}"
if [[ ! "$repeats" =~ ^[0-9]+$ || "$repeats" -lt 1 ]]; then
    echo "repeats must be a positive integer: $repeats" >&2
    exit 2
fi
result_dir="$root_dir/benchmark/results"
log_dir="$root_dir/benchmark/logs"
mkdir -p "$result_dir" "$log_dir"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
base=$(basename "$model_path" .gguf)
summary_file="$result_dir/repeat_${base}_${stamp}.json"
index_file=$(mktemp)
trap 'rm -f "$index_file"' EXIT

assert_no_server() {
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${port}$"; then
        echo "a server is still listening on port $port after run $1" >&2
        exit 1
    fi
}

for run in $(seq 1 "$repeats"); do
    log_file="$log_dir/repeat_${base}_r${run}_${stamp}.log"
    "$root_dir/benchmark/compare_qwen35_cpu_llama.sh" "$model_path" \
        "$token_id" "$steps" "$context" >"$log_file" 2>&1
    assert_no_server "$run"
    result_file=$(sed -n 's/^result=//p' "$log_file" | tail -n 1)
    if [[ -z "$result_file" || ! -f "$result_file" ]]; then
        echo "comparison run $run did not produce a result" >&2
        exit 1
    fi
    printf '%s\n' "$result_file" >>"$index_file"
    echo "run $run/$repeats done: $result_file"
done

python3 - "$index_file" "$summary_file" "$repeats" <<'PY'
import json
import statistics
import sys
from pathlib import Path

paths = [line.strip() for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
results = [json.loads(Path(path).read_text()) for path in paths]

def stats(values):
    return {
        "n": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
        "values": values,
    }

fortai = [float(item["fortai"]["tokens_per_second"]) for item in results]
llama = [float(item["llama_cpp"]["timings"]["predicted_per_second"]) for item in results]
first = results[0]
summary = {
    "fortai_commit": first["fortai_commit"],
    "compiler": first["compiler"],
    "cuda_visible_devices": first.get("cuda_visible_devices", "unset"),
    "omp_num_threads": first["omp_num_threads"],
    "model": first["model"],
    "model_sha256": first["model_sha256"],
    "token_id": first["token_id"],
    "steps": first["steps"],
    "context": first["context"],
    "repeats": int(sys.argv[3]),
    "fortai_tokens_per_second": stats(fortai),
    "llama_cpp_tokens_per_second": stats(llama),
    "result_files": paths,
}
Path(sys.argv[2]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps({"fortai_tokens_per_second": summary["fortai_tokens_per_second"],
                  "llama_cpp_tokens_per_second": summary["llama_cpp_tokens_per_second"]},
                 sort_keys=True))
PY
printf 'summary=%s\n' "$summary_file"
