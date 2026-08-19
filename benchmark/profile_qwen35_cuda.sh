#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_path="${1:-${FORTAI_MODEL:-}}"
if [[ -z "$model_path" || ! -f "$model_path" ]]; then
    echo "usage: benchmark/profile_qwen35_cuda.sh MODEL.gguf [token_id] [steps] [context] [device]" >&2
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
if pgrep -x llama-server >/dev/null 2>&1; then
    echo "an existing llama-server is running; stop it before profiling" >&2
    exit 2
fi

base=$(basename "$model_path" .gguf)
stamp=$(date -u +%Y%m%dT%H%M%SZ)
profile_dir="$root_dir/benchmark/profiles/${base}_cuda_d${device}_${stamp}"
mkdir -p "$profile_dir"
"$root_dir/tools/worktree_digest.sh" >"$profile_dir/worktree.txt"
{
    printf 'commit=%s\nmodel=%s\nmodel_sha256=%s\n' \
        "$(git -C "$root_dir" rev-parse HEAD)" "$model_path" \
        "$(sha256sum "$model_path" | awk '{print $1}')"
    printf 'device=%s\nsteps=%s\ncontext=%s\nnvcc=%s\n' \
        "$device" "$steps" "$context" \
        "$(/opt/cuda/bin/nvcc --version | tail -n 1)"
} >"$profile_dir/provenance.txt"

"$root_dir/tools/build_cuda_qwen35.sh" >"$profile_dir/build.log"
CUDA_VISIBLE_DEVICES="$device" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" \
    nsys profile --trace=cuda,osrt --sample=none --cpuctxsw=none \
    --force-overwrite=true -o "$profile_dir/fortai-model" \
    "$root_dir/build/cuda/fortai_cuda_run" "$model_path" "$token_id" "$steps" \
    "$context" "$device" >"$profile_dir/run.log" 2>&1
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum --format csv --force-export=true \
    "$profile_dir/fortai-model.nsys-rep" >"$profile_dir/nsys-stats.csv" 2>&1 || true

if ! rg -q 'q8_gemv_(one|four)_warp' "$profile_dir/nsys-stats.csv"; then
    echo "CUDA model profile contains no FortAI Q8 kernels" >&2
    exit 1
fi
printf 'profile_dir=%s\nstats=%s\n' "$profile_dir" "$profile_dir/nsys-stats.csv"
