#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_path="${1:-${FORTAI_MODEL:-}}"
if [[ -z "$model_path" || ! -f "$model_path" ]]; then
    echo "usage: benchmark/compare_qwen35_cuda_llama.sh MODEL.gguf [token_id] [steps] [context] [device]" >&2
    exit 2
fi
token_id="${2:-${FORTAI_TOKEN_ID:-9419}}"
steps="${3:-${FORTAI_BENCH_STEPS:-64}}"
context="${4:-${FORTAI_CONTEXT:-128}}"
device="${5:-${CUDA_DEVICE:-0}}"
threads="${OMP_NUM_THREADS:-2}"
llama_server="${LLAMA_SERVER:-/home/ert/.local/bin/llama-server}"
llama_library_dir="${LLAMA_LIBRARY_DIR:-/home/ert/.local/llama.cpp-b10430-cuda}"
port="${LLAMA_PORT:-18092}"
result_dir="$root_dir/benchmark/results"
log_dir="$root_dir/benchmark/logs"
mkdir -p "$result_dir" "$log_dir"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
base=$(basename "$model_path" .gguf)
fortai_log="$log_dir/compare_cuda_fortai_${base}_${stamp}.log"
llama_log="$log_dir/compare_cuda_llama_${base}_${stamp}.log"
result_file="$result_dir/compare_cuda_${base}_d${device}_${stamp}.json"

if [[ ! -x "$llama_server" ]]; then echo "llama-server not found: $llama_server" >&2; exit 2; fi
if pgrep -x llama-server >/dev/null 2>&1; then
    echo "an existing llama-server is running; stop it before CUDA model benchmarking" >&2
    exit 2
fi
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${port}$"; then
    echo "comparison port is already in use: $port" >&2
    exit 2
fi

native_flags="${FORTAI_NATIVE_FLAGS:--O3 -march=native -mtune=native -funroll-loops -fopenmp -ffast-math -fno-math-errno -flto}"
digest_output=$("$root_dir/tools/worktree_digest.sh")
patch_digest=$(printf '%s\n' "$digest_output" | sed -n 's/^patch_digest=//p')
tree_digest=$(printf '%s\n' "$digest_output" | sed -n 's/^tracked_tree_digest=//p')
worktree_digest=$(printf '%s\n' "$digest_output" | sed -n 's/^worktree_digest=//p')
model_sha256=$(sha256sum "$model_path" | awk '{print $1}')

(cd "$root_dir" && "$root_dir/tools/build_cuda_qwen35.sh") >"$log_dir/build_cuda_qwen35_${stamp}.log"
(cd "$root_dir" && env OMP_NUM_THREADS="$threads" OMP_PROC_BIND="${OMP_PROC_BIND:-spread}" \
    OMP_PLACES="${OMP_PLACES:-cores}" "$root_dir/build/cuda/fortai_cuda_run" \
    "$model_path" "$token_id" "$steps" "$context" "$device") >"$fortai_log"

cleanup() {
    if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM
if [[ -n "$llama_library_dir" ]]; then
    export LD_LIBRARY_PATH="$llama_library_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
CUDA_VISIBLE_DEVICES="$device" "$llama_server" -m "$model_path" --host 127.0.0.1 \
    --port "$port" -c "$context" -ngl 99 -t "$threads" -tb "$threads" --no-webui \
    >"$llama_log" 2>&1 &
server_pid=$!
for attempt in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then tail -n 80 "$llama_log" >&2 || true; exit 1; fi
    sleep 1
    if [[ "$attempt" == 120 ]]; then echo "llama-server did not become healthy" >&2; exit 1; fi
done
request=$(TOKEN_ID="$token_id" STEPS="$steps" ORACLE_PROBS="${FORTAI_LLAMA_ORACLE_PROBS:-0}" python3 - <<'PY'
import json
import os
request = {"prompt": [int(os.environ["TOKEN_ID"])],
           "n_predict": int(os.environ["STEPS"]),
           "temperature": 0.0, "seed": 42}
if os.environ["ORACLE_PROBS"] == "1":
    request["n_probs"] = 1
    request["post_sampling_probs"] = True
print(json.dumps(request))
PY
)
llama_result="$result_file.llama"
curl -fsS "http://127.0.0.1:${port}/completion" -H 'Content-Type: application/json' \
    --data-binary "$request" >"$llama_result"
cleanup
trap - EXIT INT TERM
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${port}$"; then
    echo "llama-server cleanup failed on port $port" >&2
    exit 1
fi
if pgrep -x llama-server >/dev/null 2>&1; then echo "llama-server cleanup failed" >&2; exit 1; fi

MODEL_PATH="$model_path" MODEL_SHA256="$model_sha256" TOKEN_ID="$token_id" STEPS="$steps" \
CONTEXT="$context" DEVICE="$device" THREADS="$threads" COMMIT="$(git -C "$root_dir" rev-parse HEAD)" \
FORTAI_LOG="$fortai_log" LLAMA_LOG="$llama_log" LLAMA_RESULT="$llama_result" RESULT_FILE="$result_file" \
PATCH_DIGEST="$patch_digest" TREE_DIGEST="$tree_digest" WORKTREE_DIGEST="$worktree_digest" \
LLAMA_SERVER="$llama_server" LLAMA_LIBRARY_DIR="$llama_library_dir" python3 - <<'PY'
import json
import os
from pathlib import Path

values = {}
for line in Path(os.environ["FORTAI_LOG"]).read_text().splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        values[key] = value.strip()
llama = json.loads(Path(os.environ["LLAMA_RESULT"]).read_text())
timings = llama.get("timings", {})
if int(values.get("steps", -1)) != int(os.environ["STEPS"]):
    raise SystemExit("FortAI CUDA runner did not execute the requested step count")
if int(timings.get("predicted_n", -1)) != int(os.environ["STEPS"]):
    raise SystemExit("llama.cpp did not execute the requested step count")
result = {
    "scope": "qwen35_model_device_resident_cuda_q8" if values.get("device_pipeline") == "T" else "qwen35_model_host_controlled_cuda_q8",
    "production_gate": "not_promoted_cuda_graph_or_full_parity" if values.get("device_pipeline") == "T" else "not_promoted_host_transfers",
    "fortai_commit": os.environ["COMMIT"],
    "fortai_patch_digest": os.environ["PATCH_DIGEST"],
    "fortai_tracked_tree_digest": os.environ["TREE_DIGEST"],
    "fortai_worktree_digest": os.environ["WORKTREE_DIGEST"],
    "model": os.environ["MODEL_PATH"],
    "model_sha256": os.environ["MODEL_SHA256"],
    "token_id": int(os.environ["TOKEN_ID"]),
    "steps": int(os.environ["STEPS"]),
    "context": int(os.environ["CONTEXT"]),
    "cuda_device": int(os.environ["DEVICE"]),
    "omp_num_threads": int(os.environ["THREADS"]),
    "llama_server_cleanup": "verified",
    "fortai_log": os.environ["FORTAI_LOG"],
    "llama_log": os.environ["LLAMA_LOG"],
    "llama_server": os.environ["LLAMA_SERVER"],
    "llama_library_dir": os.environ["LLAMA_LIBRARY_DIR"],
    "fortai": values,
    "llama_cpp": llama,
    "fortai_over_llama": float(values["tokens_per_second"]) / float(timings["predicted_per_second"]),
}
Path(os.environ["RESULT_FILE"]).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps({"fortai_tokens_per_second": float(values["tokens_per_second"]),
                  "llama_cpp_tokens_per_second": float(timings["predicted_per_second"]),
                  "fortai_over_llama": result["fortai_over_llama"],
                  "production_gate": result["production_gate"]}, sort_keys=True))
PY
printf 'result=%s\nllama_log=%s\nfortai_log=%s\n' "$result_file" "$llama_log" "$fortai_log"
