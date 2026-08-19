#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
result_dir="$root_dir/benchmark/results"
log_dir="$root_dir/benchmark/logs"
mkdir -p "$result_dir" "$log_dir"

model_path="${FORTAI_LLAMA_MODEL:-/mnt/storage/slopcode/models/unsloth_Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf}"
llama_server="${LLAMA_SERVER:-/home/ert/.local/bin/llama-server}"
port="${LLAMA_PORT:-18081}"
context="${LLAMA_CONTEXT:-4096}"
batch="${LLAMA_BATCH:-512}"
ubatch="${LLAMA_UBATCH:-128}"
gpu_layers="${LLAMA_GPU_LAYERS:-99}"
prompt="${FORTAI_BENCH_PROMPT:-Write one short sentence about Fortran.}"
max_tokens="${FORTAI_BENCH_MAX_TOKENS:-32}"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log_file="$log_dir/llama_cpp_${stamp}.log"
result_file="$result_dir/llama_cpp_${stamp}.json"
dry_run=false
if [[ "${1:-}" == '--dry-run' ]]; then
    dry_run=true
fi

if [[ "$dry_run" == true ]]; then
    printf '%q ' "$llama_server" -m "$model_path" --host 127.0.0.1 \
        --port "$port" -c "$context" -b "$batch" -ub "$ubatch" \
        -ngl "$gpu_layers" --no-context-shift --metrics
    printf '\n'
    exit 0
fi

if [[ ! -x "$llama_server" ]]; then
    echo "llama-server not found: $llama_server" >&2
    exit 2
fi
if [[ ! -f "$model_path" ]]; then
    echo "model not found: $model_path" >&2
    exit 2
fi
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${port}$"; then
    echo "benchmark port is already in use: $port" >&2
    exit 2
fi

cleanup() {
    if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

"$llama_server" -m "$model_path" --host 127.0.0.1 --port "$port" \
    -c "$context" -b "$batch" -ub "$ubatch" -ngl "$gpu_layers" \
    --no-context-shift --metrics >"$log_file" 2>&1 &
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

start_ns=$(date +%s%N)
response=$(curl -fsS "http://127.0.0.1:${port}/v1/completions" \
    -H 'Content-Type: application/json' \
    --data-binary "$(env FORTAI_BENCH_PROMPT="$prompt" FORTAI_BENCH_MAX_TOKENS="$max_tokens" \
        python3 -c 'import json, os; print(json.dumps({"prompt": os.environ["FORTAI_BENCH_PROMPT"], "max_tokens": int(os.environ["FORTAI_BENCH_MAX_TOKENS"]), "temperature": 0.0}))')")
end_ns=$(date +%s%N)
elapsed=$(env START_NS="$start_ns" END_NS="$end_ns" \
    python3 -c 'import os; print((int(os.environ["END_NS"]) - int(os.environ["START_NS"])) / 1e9)')

env RESPONSE="$response" ELAPSED="$elapsed" MODEL_PATH="$model_path" \
    python3 -c 'import json, os; response=json.loads(os.environ["RESPONSE"]); response["fortai_wall_seconds"]=float(os.environ["ELAPSED"]); response["fortai_model"]=os.environ["MODEL_PATH"]; print(json.dumps(response, sort_keys=True))' \
    | tee "$result_file"
printf 'result=%s\nlog=%s\n' "$result_file" "$log_file"
