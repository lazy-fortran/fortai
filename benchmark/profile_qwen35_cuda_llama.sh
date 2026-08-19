#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_path="${1:-${FORTAI_MODEL:-}}"
if [[ -z "$model_path" || ! -f "$model_path" ]]; then
    echo "usage: benchmark/profile_qwen35_cuda_llama.sh MODEL.gguf [token_id] [steps] [context] [device]" >&2
    exit 2
fi
token_id="${2:-${FORTAI_TOKEN_ID:-9419}}"
steps="${3:-${FORTAI_PROFILE_STEPS:-16}}"
context="${4:-${FORTAI_CONTEXT:-128}}"
device="${5:-${CUDA_DEVICE:-0}}"
if ! command -v nsys >/dev/null 2>&1; then
    echo "nsys is required for CUDA model profiling" >&2
    exit 2
fi

llama_server="${LLAMA_SERVER:-/home/ert/.local/bin/llama-server}"
llama_library_dir="${LLAMA_LIBRARY_DIR:-/home/ert/.local/llama.cpp-b10430-cuda}"
port="${LLAMA_PORT:-18093}"
if [[ ! -x "$llama_server" ]]; then
    echo "llama-server not found: $llama_server" >&2
    exit 2
fi
if pgrep -x llama-server >/dev/null 2>&1; then
    echo "an existing llama-server is running; stop it before profiling" >&2
    exit 2
fi
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${port}$"; then
    echo "profiling port is already in use: $port" >&2
    exit 2
fi

base=$(basename "$model_path" .gguf)
stamp=$(date -u +%Y%m%dT%H%M%SZ)
profile_dir="$root_dir/benchmark/profiles/${base}_cuda_both_d${device}_${stamp}"
mkdir -p "$profile_dir"
"$root_dir/tools/worktree_digest.sh" >"$profile_dir/worktree.txt"
{
    printf 'commit=%s\nmodel=%s\nmodel_sha256=%s\n' \
        "$(git -C "$root_dir" rev-parse HEAD)" "$model_path" \
        "$(sha256sum "$model_path" | awk '{print $1}')"
    printf 'device=%s\nsteps=%s\ncontext=%s\nnsys=%s\n' \
        "$device" "$steps" "$context" "$(nsys --version 2>&1 | head -n 1)"
    printf 'llama_server=%s\nllama_library_dir=%s\nport=%s\n' \
        "$llama_server" "$llama_library_dir" "$port"
} >"$profile_dir/provenance.txt"

"$root_dir/tools/build_cuda_qwen35.sh" >"$profile_dir/build.log"
CUDA_VISIBLE_DEVICES="$device" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" \
    nsys profile --trace=cuda,osrt --sample=none --cpuctxsw=none \
    --force-overwrite=true -o "$profile_dir/fortai" \
    "$root_dir/build/cuda/fortai_cuda_run" "$model_path" "$token_id" "$steps" \
    "$context" "$device" >"$profile_dir/fortai.log" 2>&1

if [[ -n "$llama_library_dir" ]]; then
    export LD_LIBRARY_PATH="$llama_library_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
CUDA_VISIBLE_DEVICES="$device" nsys profile --trace=cuda,osrt --sample=none --cpuctxsw=none \
    --force-overwrite=true -o "$profile_dir/llama" \
    "$llama_server" -m "$model_path" --host 127.0.0.1 --port "$port" -c "$context" \
    -ngl 99 -t "${OMP_NUM_THREADS:-2}" -tb "${OMP_NUM_THREADS:-2}" --no-webui \
    >"$profile_dir/llama.log" 2>&1 &
profile_pid=$!

cleanup() {
    if [[ -n "${profile_pid:-}" ]] && kill -0 "$profile_pid" 2>/dev/null; then
        kill -TERM "$profile_pid" 2>/dev/null || true
        wait "$profile_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

for attempt in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$profile_pid" 2>/dev/null; then
        tail -n 100 "$profile_dir/llama.log" >&2 || true
        exit 1
    fi
    sleep 1
    if [[ "$attempt" == 120 ]]; then
        echo "profiled llama-server did not become healthy" >&2
        exit 1
    fi
done

request=$(TOKEN_ID="$token_id" STEPS="$steps" python3 - <<'PY'
import json
import os
print(json.dumps({"prompt": [int(os.environ["TOKEN_ID"])],
                  "n_predict": int(os.environ["STEPS"]),
                  "temperature": 0.0, "seed": 42}))
PY
)
curl -fsS "http://127.0.0.1:${port}/completion" -H 'Content-Type: application/json' \
    --data-binary "$request" >"$profile_dir/llama-response.json"
cleanup
trap - EXIT INT TERM

if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${port}$"; then
    echo "llama-server cleanup failed on port $port" >&2
    exit 1
fi
if pgrep -x llama-server >/dev/null 2>&1; then
    echo "llama-server cleanup failed" >&2
    exit 1
fi

for runtime in fortai llama; do
    nsys stats --report cuda_api_sum,cuda_gpu_kern_sum --format csv \
        --force-export=true "$profile_dir/${runtime}.nsys-rep" \
        >"$profile_dir/${runtime}-nsys-stats.csv" 2>&1 || true
    if [[ ! -s "$profile_dir/${runtime}.nsys-rep" ]]; then
        echo "missing ${runtime} Nsight Systems report" >&2
        exit 1
    fi
done

printf 'profile_dir=%s\nfortai_stats=%s\nllama_stats=%s\n' \
    "$profile_dir" "$profile_dir/fortai-nsys-stats.csv" "$profile_dir/llama-nsys-stats.csv"
