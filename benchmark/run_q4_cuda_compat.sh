#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_path="${1:-${FORTAI_Q4_MODEL:-/mnt/storage/slopcode/models/unsloth_Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf}}"
token_id="${2:-${FORTAI_TOKEN_ID:-9419}}"
steps="${3:-${FORTAI_BENCH_STEPS:-1}}"
context="${4:-${FORTAI_CONTEXT:-128}}"
port="${Q4_LLAMA_PORT:-18380}"
batch="${Q4_LLAMA_BATCH:-512}"
ubatch="${Q4_LLAMA_UBATCH:-128}"
server="${LLAMA_SERVER:-/home/ert/.local/bin/llama-server}"
if [[ ! -f "$model_path" || ! -x "$server" ]]; then
    echo "Q4 model or llama-server is unavailable" >&2
    exit 2
fi
if pgrep -x llama-server >/dev/null 2>&1; then
    echo "an existing llama-server is running; stop it before Q4 CUDA smoke" >&2
    exit 2
fi
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${port}$"; then
    echo "Q4 smoke port is already in use: $port" >&2
    exit 2
fi

log_file=$(mktemp)
response_file=$(mktemp)
cleanup() {
    if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    rm -f "$log_file" "$response_file"
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
    "$server" -m "$model_path" --host 127.0.0.1 --port "$port" \
    -c "$context" -b "$batch" -ub "$ubatch" -ngl 99 -np 1 -sm layer -ts 1,1 \
    --flash-attn on --no-webui >"$log_file" 2>&1 &
server_pid=$!
for attempt in $(seq 1 300); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        tail -n 80 "$log_file" >&2
        exit 1
    fi
    sleep 1
    if [[ "$attempt" == 300 ]]; then
        echo "Q4_K_XL CUDA server did not become healthy" >&2
        tail -n 80 "$log_file" >&2
        exit 1
    fi
done

request=$(TOKEN_ID="$token_id" STEPS="$steps" python3 - <<'PY'
import json, os
print(json.dumps({"prompt": [int(os.environ["TOKEN_ID"])],
                  "n_predict": int(os.environ["STEPS"]),
                  "temperature": 0.0, "seed": 42,
                  "post_sampling_probs": True}))
PY
)
curl -fsS "http://127.0.0.1:${port}/completion" -H 'Content-Type: application/json' \
    --data-binary "$request" >"$response_file"
python3 - "$response_file" "$model_path" <<'PY'
import json, sys
from pathlib import Path
response = json.loads(Path(sys.argv[1]).read_text())
timings = response.get("timings", {})
print(json.dumps({"backend": "llama.cpp-q4_k_xl-cuda",
                  "model": sys.argv[2],
                  "content": response.get("content", ""),
                  "tokens": response.get("tokens", []),
                  "predicted_n": timings.get("predicted_n", 0),
                  "predicted_per_second": timings.get("predicted_per_second", 0.0)},
                 sort_keys=True))
PY
printf 'llama_server_cleanup=verified\n'
