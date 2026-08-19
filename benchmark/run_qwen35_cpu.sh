#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_path="${1:-${FORTAI_MODEL:-}}"
if [[ -z "$model_path" ]]; then
    echo "usage: benchmark/run_qwen35_cpu.sh MODEL.gguf [token_id] [steps] [context]" >&2
    exit 2
fi
if [[ ! -f "$model_path" ]]; then
    echo "model not found: $model_path" >&2
    exit 2
fi

token_id="${2:-${FORTAI_TOKEN_ID:-9419}}"
steps="${3:-${FORTAI_BENCH_STEPS:-8}}"
context="${4:-${FORTAI_CONTEXT:-128}}"
result_dir="$root_dir/benchmark/results"
log_dir="$root_dir/benchmark/logs"
mkdir -p "$result_dir" "$log_dir"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
base=$(basename "$model_path" .gguf)
raw_file="$log_dir/fortai_${base}_${stamp}.log"
result_file="$result_dir/fortai_${base}_${stamp}.json"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-cores}"
native_flags="${FORTAI_NATIVE_FLAGS:--O3 -march=native -mtune=native -funroll-loops -fopenmp -ffast-math -fno-math-errno -flto}"
digest_output=$("$root_dir/tools/worktree_digest.sh")
patch_digest=$(printf '%s\n' "$digest_output" | sed -n 's/^patch_digest=//p')
tree_digest=$(printf '%s\n' "$digest_output" | sed -n 's/^tracked_tree_digest=//p')
worktree_digest=$(printf '%s\n' "$digest_output" | sed -n 's/^worktree_digest=//p')

(cd "$root_dir" && fo build --flag "$native_flags" && \
    env OMP_NUM_THREADS="$OMP_NUM_THREADS" OMP_PROC_BIND="$OMP_PROC_BIND" \
        OMP_PLACES="$OMP_PLACES" fo exec --no-build fortai_cpu_run \
        "$model_path" "$token_id" "$steps" "$context") >"$raw_file"

MODEL_PATH="$model_path" TOKEN_ID="$token_id" STEPS="$steps" CONTEXT="$context" \
COMMIT="$(git -C "$root_dir" rev-parse HEAD)" OMP_NUM_THREADS="$OMP_NUM_THREADS" \
COMPILER="$(gfortran --version | head -n 1)" RAW_FILE="$raw_file" RESULT_FILE="$result_file" \
BUILD_FLAGS="$native_flags" PATCH_DIGEST="$patch_digest" \
TRACKED_TREE_DIGEST="$tree_digest" WORKTREE_DIGEST="$worktree_digest" \
python3 - <<'PY'
import json
import os
from pathlib import Path

values = {}
for line in Path(os.environ["RAW_FILE"]).read_text().splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        values[key] = value.strip()
result = {
    "fortai_commit": os.environ["COMMIT"],
    "fortai_patch_digest": os.environ["PATCH_DIGEST"],
    "fortai_tracked_tree_digest": os.environ["TRACKED_TREE_DIGEST"],
    "fortai_worktree_digest": os.environ["WORKTREE_DIGEST"],
    "build_flags": os.environ["BUILD_FLAGS"],
    "compiler": os.environ["COMPILER"],
    "omp_num_threads": int(os.environ["OMP_NUM_THREADS"]),
    "model": os.environ["MODEL_PATH"],
    "token_id": int(os.environ["TOKEN_ID"]),
    "steps": int(os.environ["STEPS"]),
    "context": int(os.environ["CONTEXT"]),
    "model_sha256": __import__("hashlib").sha256(Path(os.environ["MODEL_PATH"]).read_bytes()).hexdigest(),
    "runner": values,
}
Path(os.environ["RESULT_FILE"]).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, sort_keys=True))
PY
printf 'result=%s\nlog=%s\n' "$result_file" "$raw_file"
