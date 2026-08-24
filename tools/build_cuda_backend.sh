#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
nvcc_bin=${NVCC:-/opt/cuda/bin/nvcc}
cuda_arch=${FORTAI_CUDA_ARCH:-sm_120}
out_dir="$root_dir/build/cuda"
fortran_dir="$out_dir/fortran-smoke"
ggml_prefix=${FORTAI_GGML_PREFIX:-/home/ert/.local/llama.cpp-upstream-main-650913862}
mkdir -p "$out_dir"
if [[ ! -x "$nvcc_bin" ]]; then
    echo "nvcc not found: $nvcc_bin" >&2
    exit 2
fi
if [[ ! -f "$ggml_prefix/include/ggml.h" || ! -f "$ggml_prefix/lib/libggml-cuda.so" ]]; then
    echo "GGML CUDA backend not found under $ggml_prefix; set FORTAI_GGML_PREFIX" >&2
    exit 2
fi
"$nvcc_bin" -O3 --use_fast_math -std=c++17 -arch="$cuda_arch" -lineinfo \
    -Xcompiler=-fPIC -c "$root_dir/backend/cuda/fortai_cuda_backend.cu" \
    -o "$out_dir/fortai_cuda_backend.o"
"$nvcc_bin" -O3 --use_fast_math -std=c++17 -arch="$cuda_arch" -lineinfo \
    -I"$ggml_prefix/include" -I"$root_dir/backend/cuda" -Xcompiler=-fPIC -c \
    "$root_dir/backend/cuda/fortai_cuda_q4_backend.cu" -o "$out_dir/fortai_cuda_q4_backend.o"
"$nvcc_bin" -O3 --use_fast_math -std=c++17 -arch="$cuda_arch" -lineinfo \
    "$root_dir/tools/cuda_backend_smoke.cu" "$out_dir/fortai_cuda_backend.o" \
    -o "$out_dir/cuda_backend_smoke"
mkdir -p "$fortran_dir/mod"
gfortran -O2 -J"$fortran_dir/mod" -c "$root_dir/src/fortai_status.f90" \
    -o "$fortran_dir/fortai_status.o"
gfortran -O2 -I"$fortran_dir/mod" -J"$fortran_dir/mod" \
    -c "$root_dir/src/backend/cuda/fortai_backend_cuda.f90" \
    -o "$fortran_dir/fortai_backend_cuda.o"
gfortran -O2 -I"$fortran_dir/mod" -J"$fortran_dir/mod" \
    -c "$root_dir/tools/fortran_cuda_backend_smoke.f90" \
    -o "$fortran_dir/fortran_cuda_backend_smoke.o"
"$nvcc_bin" -O3 -arch="$cuda_arch" \
    "$fortran_dir/fortai_status.o" "$fortran_dir/fortai_backend_cuda.o" \
    "$fortran_dir/fortran_cuda_backend_smoke.o" "$out_dir/fortai_cuda_backend.o" \
    "$out_dir/fortai_cuda_q4_backend.o" -L"$ggml_prefix/lib" \
    -Xlinker -rpath -Xlinker "$ggml_prefix/lib" -lggml-cuda -lggml-cpu -lggml -lggml-base \
    -lgfortran -lquadmath -o "$out_dir/fortran_cuda_backend_smoke"
printf 'object=%s\nsmoke=%s\nfortran_smoke=%s\narch=%s\nnvcc=%s\n' \
    "$out_dir/fortai_cuda_backend.o" "$out_dir/cuda_backend_smoke" \
    "$out_dir/fortran_cuda_backend_smoke" \
    "$cuda_arch" "$nvcc_bin"
