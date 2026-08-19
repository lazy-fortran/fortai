#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_path="${1:-${FORTAI_MODEL:-}}"
if [[ -z "$model_path" ]]; then
    echo "usage: benchmark/profile_qwen35_cpu_both.sh MODEL.gguf [token_id] [steps] [context]" >&2
    exit 2
fi
if [[ ! -f "$model_path" ]]; then
    echo "model not found: $model_path" >&2
    exit 2
fi
if ! command -v perf >/dev/null 2>&1; then
    echo "perf is required for profiling" >&2
    exit 2
fi

token_id="${2:-${FORTAI_TOKEN_ID:-9419}}"
steps="${3:-${FORTAI_PROFILE_STEPS:-128}}"
context="${4:-${FORTAI_CONTEXT:-128}}"
threads="${OMP_NUM_THREADS:-$(nproc)}"
frequency="${FORTAI_PERF_FREQUENCY:-499}"
perf_mmap="${FORTAI_PERF_MMAP:-64}"
perf_call_graph="${FORTAI_PERF_CALL_GRAPH:-dwarf,32}"
fortai_delay_ms="${FORTAI_FORTAI_DELAY_MS:-0}"
llama_delay_ms="${FORTAI_LLAMA_DELAY_MS:-0}"
llama_server="${LLAMA_SERVER:-/home/ert/.local/bin/llama-server}"
llama_library_dir="${LLAMA_LIBRARY_DIR:-/home/ert/.local/llama.cpp-b10430-cuda}"
port="${FORTAI_PROFILE_PORT:-18083}"
base=$(basename "$model_path" .gguf)
stamp=$(date -u +%Y%m%dT%H%M%SZ)
profile_dir="$root_dir/benchmark/profiles/${base}_both_${stamp}"
mkdir -p "$profile_dir"

if [[ ! -x "$llama_server" ]]; then
    echo "llama-server not found: $llama_server" >&2
    exit 2
fi
if pgrep -x llama-server >/dev/null 2>&1; then
    if [[ "${FORTAI_ALLOW_EXISTING_LLAMA_SERVER:-0}" == 1 ]]; then
        echo "allowing existing llama-server; this profile is intentionally under shared-service conditions" >&2
        pgrep -a -x llama-server >&2 || true
    else
        echo "an existing llama-server is running; stop it before profiling" >&2
        pgrep -a -x llama-server >&2 || true
        exit 2
    fi
fi
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${port}$|:$((port + 1))$"; then
    echo "profile ports are already in use: $port and $((port + 1))" >&2
    exit 2
fi

export OMP_NUM_THREADS="$threads"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export CUDA_VISIBLE_DEVICES=""
native_flags="${FORTAI_NATIVE_FLAGS:--O3 -march=native -mtune=native -funroll-loops -fopenmp -ffast-math -fno-math-errno -flto}"
events="task-clock,context-switches,cpu-migrations,cycles,instructions,branches,branch-misses,cache-references,cache-misses"

{
    "$root_dir/tools/worktree_digest.sh"
    printf 'fortai_commit=%s\n' "$(git -C "$root_dir" rev-parse HEAD)"
    printf 'model=%s\n' "$model_path"
    printf 'model_sha256=%s\n' "$(sha256sum "$model_path" | awk '{print $1}')"
    printf 'compiler=%s\n' "$(gfortran --version | head -n 1)"
    printf 'build_flags=%s\n' "$native_flags"
    printf 'omp_num_threads=%s\n' "$OMP_NUM_THREADS"
    printf 'omp_proc_bind=%s\n' "$OMP_PROC_BIND"
    printf 'omp_places=%s\n' "$OMP_PLACES"
    printf 'fortai_perf_delay_ms=%s\n' "$fortai_delay_ms"
    printf 'llama_perf_delay_ms=%s\n' "$llama_delay_ms"
    printf 'perf_mmap=%s\n' "$perf_mmap"
    printf 'perf_call_graph=%s\n' "$perf_call_graph"
    printf 'llama_server=%s\n' "$llama_server"
    printf 'llama_server_sha256=%s\n' "$(sha256sum "$llama_server" | awk '{print $1}')"
    printf 'llama_library_dir=%s\n' "$llama_library_dir"
} >"$profile_dir/provenance.txt"

(cd "$root_dir" && fo build --flag "$native_flags") >"$profile_dir/fortai-build.log"
run=(fo exec --no-build --cwd "$root_dir" fortai_cpu_run "$model_path" "$token_id" "$steps" "$context")

env OMP_NUM_THREADS="$OMP_NUM_THREADS" OMP_PROC_BIND="$OMP_PROC_BIND" OMP_PLACES="$OMP_PLACES" \
    perf stat -D "$fortai_delay_ms" -x, -e "$events" -o "$profile_dir/fortai-perf-stat.csv" -- \
    "${run[@]}" >"$profile_dir/fortai-stat-run.log" 2>&1

env OMP_NUM_THREADS="$OMP_NUM_THREADS" OMP_PROC_BIND="$OMP_PROC_BIND" OMP_PLACES="$OMP_PLACES" \
    perf record -D "$fortai_delay_ms" -F "$frequency" -m "$perf_mmap" \
        --call-graph "$perf_call_graph" -o "$profile_dir/fortai-perf.data" -- \
    "${run[@]}" >"$profile_dir/fortai-record-run.log" 2>&1

active_perf_pid=""
active_llama_pid=""
cleanup() {
    if [[ -n "$active_llama_pid" ]] && kill -0 "$active_llama_pid" 2>/dev/null; then
        kill -TERM "$active_llama_pid" 2>/dev/null || true
    fi
    if [[ -n "$active_perf_pid" ]] && kill -0 "$active_perf_pid" 2>/dev/null; then
        kill -INT "$active_perf_pid" 2>/dev/null || true
    fi
    if [[ -n "$active_llama_pid" ]]; then
        wait "$active_llama_pid" 2>/dev/null || true
    fi
    if [[ -n "$active_perf_pid" ]]; then
        wait "$active_perf_pid" 2>/dev/null || true
    fi
    active_perf_pid=""
    active_llama_pid=""
}
trap cleanup EXIT INT TERM

run_llama_profile() {
    local mode="$1"
    local profile_port="$2"
    local response="$profile_dir/llama-${mode}-response.json"
    local server_log="$profile_dir/llama-${mode}-server.log"
    local output_file="$profile_dir/llama-${mode}-perf.data"
    local stat_file="$profile_dir/llama-${mode}-perf-stat.csv"
    local -a command

    if [[ "$mode" == stat ]]; then
        command=(perf stat -D "$llama_delay_ms" -x, -e "$events" -o "$stat_file" -- "$llama_server")
    else
        command=(perf record -D "$llama_delay_ms" -F "$frequency" -m "$perf_mmap" \
            --call-graph "$perf_call_graph" -o "$output_file" -- "$llama_server")
    fi
    if [[ -n "$llama_library_dir" ]]; then
        export LD_LIBRARY_PATH="$llama_library_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi

    env OMP_NUM_THREADS="$OMP_NUM_THREADS" OMP_PROC_BIND="$OMP_PROC_BIND" \
        OMP_PLACES="$OMP_PLACES" CUDA_VISIBLE_DEVICES="" \
        "${command[@]}" -m "$model_path" --host 127.0.0.1 --port "$profile_port" \
        -c "$context" -ngl 0 --device none --no-op-offload \
        -t "$threads" -tb "$threads" --no-webui >"$server_log" 2>&1 &
    active_perf_pid=$!

    for attempt in $(seq 1 120); do
        active_llama_pid=$(pgrep -P "$active_perf_pid" -x llama-server | head -n 1 || true)
        if curl -fsS "http://127.0.0.1:${profile_port}/health" >/dev/null 2>&1; then
            break
        fi
        if ! kill -0 "$active_perf_pid" 2>/dev/null; then
            echo "llama profiler exited during startup ($mode)" >&2
            return 1
        fi
        sleep 1
        if [[ "$attempt" == 120 ]]; then
            echo "llama-server did not become healthy ($mode)" >&2
            return 1
        fi
    done

    curl -fsS "http://127.0.0.1:${profile_port}/completion" \
        -H 'Content-Type: application/json' \
        --data-binary "$(TOKEN_ID="$token_id" STEPS="$steps" python3 - <<'PY'
import json
import os
print(json.dumps({
    "prompt": [int(os.environ["TOKEN_ID"])],
    "n_predict": int(os.environ["STEPS"]),
    "temperature": 0.0,
    "seed": 42,
}))
PY
)" >"$response"

    cleanup
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${profile_port}$"; then
        echo "llama cleanup failed on port $profile_port" >&2
        return 1
    fi
}

run_llama_profile stat "$port"
run_llama_profile record "$((port + 1))"

perf report --stdio --no-children --sort symbol,dso -i "$profile_dir/fortai-perf.data" \
    >"$profile_dir/fortai-perf-report.txt"
perf report --stdio --no-children --sort symbol,dso -i "$profile_dir/llama-record-perf.data" \
    >"$profile_dir/llama-perf-report.txt"
perf report --stdio --children --sort symbol,dso -i "$profile_dir/fortai-perf.data" \
    >"$profile_dir/fortai-perf-report-children.txt"
perf report --stdio --children --sort symbol,dso -i "$profile_dir/llama-record-perf.data" \
    >"$profile_dir/llama-perf-report-children.txt"
perf script -i "$profile_dir/fortai-perf.data" >"$profile_dir/fortai-perf-script.txt"
perf script -i "$profile_dir/llama-record-perf.data" >"$profile_dir/llama-perf-script.txt"
objdump -d -Mintel "$root_dir/build/fo/bin/fortai_cpu_run" \
    >"$profile_dir/fortai-objdump.txt"
objdump -d -Mintel "$llama_server" >"$profile_dir/llama-server-objdump.txt"
llama_cpu_library=$(find "$llama_library_dir" -maxdepth 1 -type f \
    -name 'libggml-cpu.so*' | sort | tail -n 1 || true)
if [[ -n "$llama_cpu_library" ]]; then
    objdump -d -Mintel "$llama_cpu_library" \
        >"$profile_dir/llama-ggml-cpu-objdump.txt"
fi

for report in "$profile_dir/fortai-perf-report.txt" "$profile_dir/llama-perf-report.txt"; do
    if rg -q '^# Total Lost Samples: [1-9]' "$report"; then
        echo "profile contains lost samples: $report" >&2
        exit 1
    fi
done

printf 'profile_dir=%s\nfortai_report=%s\nllama_report=%s\n' \
    "$profile_dir" "$profile_dir/fortai-perf-report.txt" \
    "$profile_dir/llama-perf-report.txt"
