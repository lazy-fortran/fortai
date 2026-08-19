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
# The CPU comparison must never touch CUDA even though llama-server links
# libggml-cuda.so: hide every GPU so backend init finds zero devices.
export CUDA_VISIBLE_DEVICES=""
native_flags="${FORTAI_NATIVE_FLAGS:--O3 -march=native -mtune=native -funroll-loops -fopenmp -ffast-math -fno-math-errno -flto}"
digest_output=$("$root_dir/tools/worktree_digest.sh")
patch_digest=$(printf '%s\n' "$digest_output" | sed -n 's/^patch_digest=//p')
tree_digest=$(printf '%s\n' "$digest_output" | sed -n 's/^tracked_tree_digest=//p')
worktree_digest=$(printf '%s\n' "$digest_output" | sed -n 's/^worktree_digest=//p')

if [[ ! -x "$llama_server" ]]; then
    echo "llama-server not found: $llama_server" >&2
    exit 2
fi
if pgrep -x llama-server >/dev/null 2>&1; then
    echo "an existing llama-server is running; stop it before benchmarking" >&2
    pgrep -a -x llama-server >&2 || true
    exit 2
fi
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${port}$"; then
    echo "comparison port is already in use: $port" >&2
    exit 2
fi

llama_server_sha256=$(sha256sum "$llama_server" | awk '{print $1}')
llama_library_digest=$(python3 - "$llama_library_dir" <<'PY'
import hashlib
import sys
from pathlib import Path

directory = Path(sys.argv[1])
digest = hashlib.sha256()
if directory.is_dir():
    for path in sorted(p for p in directory.iterdir() if p.is_file()):
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
print(digest.hexdigest() if digest.digest() != hashlib.sha256().digest() else "none")
PY
)
cpu_model=$(LC_ALL=C lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -n 1 || true)
llama_record="$root_dir/.provenance/records/llama.cpp.txt"

# Run FortAI before starting llama.cpp so the measurements do not compete.
(cd "$root_dir" && fo build --flag "$native_flags" && \
    env OMP_NUM_THREADS="$OMP_NUM_THREADS" OMP_PROC_BIND="$OMP_PROC_BIND" \
        OMP_PLACES="$OMP_PLACES" fo exec --no-build fortai_cpu_run \
        "$model_path" "$token_id" "$steps" "$context") >"$fortai_log"

cleanup() {
    if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

request=$(env TOKEN_ID="$token_id" STEPS="$steps" ORACLE_PROBS="${FORTAI_LLAMA_ORACLE_PROBS:-0}" python3 - <<'PY'
import json
import os
request = {
    "prompt": [int(os.environ["TOKEN_ID"])],
    "n_predict": int(os.environ["STEPS"]),
    "temperature": 0.0,
    "seed": 42,
}
if os.environ["ORACLE_PROBS"] == "1":
    request["n_probs"] = 1
    request["post_sampling_probs"] = True
print(json.dumps(request))
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

if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | grep -Eq "^[[:space:]]*${server_pid}[[:space:]]*$"; then
        echo "llama-server $server_pid appeared as a GPU compute process" >&2
        exit 1
    fi
fi

curl -fsS "http://127.0.0.1:${port}/completion" \
    -H 'Content-Type: application/json' --data-binary "$request" >"$result_file.llama"

cleanup
trap - EXIT INT TERM
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${port}$"; then
    echo "llama-server cleanup failed on port $port" >&2
    exit 1
fi

MODEL_PATH="$model_path" TOKEN_ID="$token_id" STEPS="$steps" CONTEXT="$context" \
COMMIT="$(git -C "$root_dir" rev-parse HEAD)" OMP_NUM_THREADS="$threads" \
COMPILER="$(gfortran --version | head -n 1)" FORTAI_LOG="$fortai_log" \
LLAMA_RESULT="$result_file.llama" RESULT_FILE="$result_file" \
BUILD_FLAGS="$native_flags" PATCH_DIGEST="$patch_digest" \
TRACKED_TREE_DIGEST="$tree_digest" \
WORKTREE_DIGEST="$worktree_digest" \
FORTAI_LOG_PATH="$fortai_log" LLAMA_LOG_PATH="$llama_log" \
ORACLE_PROBS="${FORTAI_LLAMA_ORACLE_PROBS:-0}" \
LLAMA_CLEANUP="verified" LLAMA_SERVER_PATH="$llama_server" \
LLAMA_SERVER_SHA256="$llama_server_sha256" LLAMA_LIBRARY_DIR="$llama_library_dir" \
LLAMA_LIBRARY_DIGEST="$llama_library_digest" CPU_MODEL="$cpu_model" \
LLAMA_RECORD="$llama_record" \
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
if int(runner.get("steps", -1)) != int(os.environ["STEPS"]):
    raise SystemExit("FortAI did not execute the requested step count")
llama_payload = json.loads(Path(os.environ["LLAMA_RESULT"]).read_text())
llama_steps = int(llama_payload.get("timings", {}).get("predicted_n", -1))
if llama_steps != int(os.environ["STEPS"]):
    raise SystemExit("llama.cpp did not execute the requested step count")
llama_provenance = {}
record_path = Path(os.environ["LLAMA_RECORD"])
if record_path.is_file():
    for line in record_path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            llama_provenance[key] = value
result = {
    "fortai_commit": os.environ["COMMIT"],
    "fortai_patch_digest": os.environ["PATCH_DIGEST"],
    "fortai_tracked_tree_digest": os.environ["TRACKED_TREE_DIGEST"],
    "fortai_worktree_digest": os.environ["WORKTREE_DIGEST"],
    "fortai_log": os.environ["FORTAI_LOG_PATH"],
    "llama_log": os.environ["LLAMA_LOG_PATH"],
    "llama_oracle_probabilities": os.environ["ORACLE_PROBS"] == "1",
    "llama_server_cleanup": os.environ["LLAMA_CLEANUP"],
    "build_flags": os.environ["BUILD_FLAGS"],
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
    "compiler": os.environ["COMPILER"],
    "cpu_model": os.environ["CPU_MODEL"],
    "omp_num_threads": int(os.environ["OMP_NUM_THREADS"]),
    "model": os.environ["MODEL_PATH"],
    "model_sha256": hashlib.sha256(Path(os.environ["MODEL_PATH"]).read_bytes()).hexdigest(),
    "token_id": int(os.environ["TOKEN_ID"]),
    "steps": int(os.environ["STEPS"]),
    "context": int(os.environ["CONTEXT"]),
    "llama_cpp_provenance": llama_provenance,
    "llama_server": os.environ["LLAMA_SERVER_PATH"],
    "llama_server_sha256": os.environ["LLAMA_SERVER_SHA256"],
    "llama_library_dir": os.environ["LLAMA_LIBRARY_DIR"],
    "llama_library_digest": os.environ["LLAMA_LIBRARY_DIGEST"],
    "fortai": runner,
    "llama_cpp": llama_payload,
}
Path(os.environ["RESULT_FILE"]).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, sort_keys=True))
PY
printf 'result=%s\nllama_log=%s\nfortai_log=%s\n' "$result_file" "$llama_log" "$fortai_log"
