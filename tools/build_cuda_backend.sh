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
    -Xcompiler=-fPIC -c "$root_dir/backend/cuda/fortai_cuda_backend.cu" \
    -o "$out_dir/fortai_cuda_backend.o"
"$nvcc_bin" -O3 --use_fast_math -std=c++17 -arch="$cuda_arch" -lineinfo \
    "$root_dir/tools/cuda_backend_smoke.cu" "$out_dir/fortai_cuda_backend.o" \
    -o "$out_dir/cuda_backend_smoke"
printf 'object=%s\nsmoke=%s\narch=%s\nnvcc=%s\n' \
    "$out_dir/fortai_cuda_backend.o" "$out_dir/cuda_backend_smoke" \
    "$cuda_arch" "$nvcc_bin"
