#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_path="${1:-${FORTAI_MODEL:-}}"
if [[ -z "$model_path" ]]; then
    echo "usage: benchmark/compare_qwen35_cpu_llama.sh MODEL.gguf [token_id] [steps] [context]" >&2
    exit 2
fi
if [[ ! -f "$model_path" ]]; then
    echo "model not found: $model_path" >&2
    exit 2
fi

token_id="${2:-${FORTAI_TOKEN_ID:-9419}}"
steps="${3:-${FORTAI_BENCH_STEPS:-8}}"
context="${4:-${FORTAI_CONTEXT:-128}}"
threads="${OMP_NUM_THREADS:-$(nproc)}"
llama_server="${LLAMA_SERVER:-/home/ert/.local/bin/llama-server}"
llama_library_dir="${LLAMA_LIBRARY_DIR:-/home/ert/.local/llama.cpp-b10430-cuda}"
port="${LLAMA_PORT:-18081}"
result_dir="$root_dir/benchmark/results"
log_dir="$root_dir/benchmark/logs"
mkdir -p "$result_dir" "$log_dir"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
base=$(basename "$model_path" .gguf)
fortai_log="$log_dir/compare_fortai_${base}_${stamp}.log"
llama_log="$log_dir/compare_llama_${base}_${stamp}.log"
result_file="$result_dir/compare_${base}_${stamp}.json"

export OMP_NUM_THREADS="$threads"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-cores}"

if [[ ! -x "$llama_server" ]]; then
    echo "llama-server not found: $llama_server" >&2
    exit 2
fi
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${port}$"; then
    echo "comparison port is already in use: $port" >&2
    exit 2
fi

# Run FortAI before starting llama.cpp so the measurements do not compete.
fpm run --target fortai_cpu_run --profile native --link-flag '-fopenmp -flto' -- \
    "$model_path" "$token_id" "$steps" "$context" >"$fortai_log"

cleanup() {
    if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

request=$(env TOKEN_ID="$token_id" STEPS="$steps" python3 - <<'PY'
import json
import os
print(json.dumps({
    "prompt": [int(os.environ["TOKEN_ID"])],
    "n_predict": int(os.environ["STEPS"]),
    "temperature": 0.0,
    "seed": 42,
}))
PY
)
if [[ -n "$llama_library_dir" ]]; then
    export LD_LIBRARY_PATH="$llama_library_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
"$llama_server" -m "$model_path" --host 127.0.0.1 --port "$port" \
    -c "$context" -ngl 0 --device none --no-op-offload \
    -t "$threads" -tb "$threads" --no-webui >"$llama_log" 2>&1 &
server_pid=$!

for attempt in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "llama-server exited during startup" >&2
        exit 1
    fi
    sleep 1
    if [[ "$attempt" == 120 ]]; then
        echo "llama-server did not become healthy" >&2
        exit 1
    fi
done

curl -fsS "http://127.0.0.1:${port}/completion" \
    -H 'Content-Type: application/json' --data-binary "$request" >"$result_file.llama"

MODEL_PATH="$model_path" TOKEN_ID="$token_id" STEPS="$steps" CONTEXT="$context" \
COMMIT="$(git -C "$root_dir" rev-parse HEAD)" OMP_NUM_THREADS="$threads" \
COMPILER="$(gfortran --version | head -n 1)" FORTAI_LOG="$fortai_log" \
LLAMA_RESULT="$result_file.llama" RESULT_FILE="$result_file" \
python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

runner = {}
for line in Path(os.environ["FORTAI_LOG"]).read_text().splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        runner[key] = value.strip()
result = {
    "fortai_commit": os.environ["COMMIT"],
    "compiler": os.environ["COMPILER"],
    "omp_num_threads": int(os.environ["OMP_NUM_THREADS"]),
    "model": os.environ["MODEL_PATH"],
    "model_sha256": hashlib.sha256(Path(os.environ["MODEL_PATH"]).read_bytes()).hexdigest(),
    "token_id": int(os.environ["TOKEN_ID"]),
    "steps": int(os.environ["STEPS"]),
    "context": int(os.environ["CONTEXT"]),
    "fortai": runner,
    "llama_cpp": json.loads(Path(os.environ["LLAMA_RESULT"]).read_text()),
}
Path(os.environ["RESULT_FILE"]).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, sort_keys=True))
PY
printf 'result=%s\nllama_log=%s\nfortai_log=%s\n' "$result_file" "$llama_log" "$fortai_log"
