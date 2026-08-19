#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
nvcc_bin=${NVCC:-/opt/cuda/bin/nvcc}
cuda_arch=${FORTAI_CUDA_ARCH:-sm_120}
out_dir="$root_dir/build/cuda"
mkdir -p "$out_dir"
if [[ ! -x "$nvcc_bin" ]]; then
    echo "nvcc not found: $nvcc_bin" >&2
    exit 2
fi
"$nvcc_bin" -O3 --use_fast_math -std=c++17 -arch="$cuda_arch" -lineinfo \
    "$root_dir/backend/cuda/fortai_cuda_q8_bench.cu" -o "$out_dir/fortai_cuda_q8_bench"
printf 'binary=%s\narch=%s\nnvcc=%s\n' "$out_dir/fortai_cuda_q8_bench" "$cuda_arch" "$nvcc_bin"
