#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
nvcc_bin=${NVCC:-/opt/cuda/bin/nvcc}
cuda_arch=${FORTAI_CUDA_ARCH:-sm_120}
out_dir="$root_dir/build/cuda"
ggml_prefix=${FORTAI_GGML_PREFIX:-/home/ert/.local/llama.cpp-upstream-main-650913862}
# nvcc uses GCC 15 on this host while the default Fortran toolchain is GCC 16.
# Keep the CUDA-facing archive non-LTO so a warm fo cache cannot hand nvcc a
# GCC-version-specific bytecode stream.  FORTAI_ALLOW_LTO=1 is an explicit
# escape hatch for callers using a matching host compiler.
native_flags="${FORTAI_NATIVE_FLAGS:--O3 -march=native -mtune=native -funroll-loops -fopenmp -ffast-math -fno-math-errno -fno-lto}"
allow_lto=${FORTAI_ALLOW_LTO:-0}
link_flags="${native_flags// -flto/} -fno-plt"
log_file="$out_dir/fortai_cuda_qwen35_fo-build.log"
inputs_file="$out_dir/fortai_cuda_qwen35.inputs"

if [[ ! -x "$nvcc_bin" ]]; then
    echo "nvcc not found: $nvcc_bin" >&2
    exit 2
fi
if [[ ! -f "$ggml_prefix/include/ggml.h" || ! -f "$ggml_prefix/lib/libggml-cuda.so" ]]; then
    echo "GGML CUDA backend not found under $ggml_prefix; set FORTAI_GGML_PREFIX" >&2
    exit 2
fi
mkdir -p "$out_dir"

"$root_dir/tools/build_cuda_backend.sh" >"$out_dir/fortai_cuda_qwen35_backend-build.log"
"$nvcc_bin" -O3 -std=c++17 -arch="$cuda_arch" -lineinfo \
    -I"$ggml_prefix/include" -I"$root_dir/backend/cuda" -Xcompiler=-fPIC -c \
    "$root_dir/backend/cuda/fortai_cuda_q4_backend.cu" -o "$out_dir/fortai_cuda_q4_backend.o"
(cd "$root_dir" && FO_DEBUG_LINKS=1 fo build --flag "$link_flags") >"$log_file" 2>&1

link_line=$(rg 'fo link: .*app_fortai_cuda_run\.f90\.o ' "$log_file" | tail -n 1 | sed 's/^fo link: //' || true)
if [[ -n "$link_line" ]]; then
    library=$(printf '%s\n' "$link_line" | grep -oE '/[^ ]*/build/fo/lib/objects_[^ ]+\.a' | tail -n 1)
    object=$(printf '%s\n' "$link_line" | grep -oE '/[^ ]*/build/fo/obj/app_fortai_cuda_run\.f90\.o' | tail -n 1)
fi
# Never feed GCC-version-specific LTO bytecode to nvcc's host linker.  This
# check also covers a warm fo action-cache hit where no link line is emitted.
archive_is_lto() {
    if [[ "$allow_lto" == 1 ]]; then return 1; fi
    strings "$1" | rg 'gnu\.lto|GNU GIMPLE' >/dev/null 2>&1
}
if [[ -n "${library:-}" && -f "$library" ]] && archive_is_lto "$library"; then
    library=''
fi
if [[ -z "${library:-}" || -z "${object:-}" || ! -f "$library" || ! -f "$object" || \
    "$root_dir/src/models/qwen35/fortai_qwen35_cpu.f90" -nt "$library" ]]; then
    object="$root_dir/build/fo/obj/app_fortai_cuda_run.f90.o"
    library=$(for archive in "$root_dir"/build/fo/lib/objects_*.a; do
        if ar t "$archive" 2>/dev/null | rg -q 'src_models_qwen35_fortai_qwen35_cpu' &&
            ! archive_is_lto "$archive"; then
            stat -c '%Y %n' "$archive"
        fi
    done | sort -nr | awk 'NR == 1 {print $2}')
fi
if [[ -z "$library" || -z "$object" || ! -f "$library" || ! -f "$object" ]]; then
    echo "could not resolve Fortran CUDA runner link inputs" >&2
    exit 1
fi
printf 'library=%s\nobject=%s\n' "$library" "$object" >"$inputs_file"

"$nvcc_bin" -O3 -arch="$cuda_arch" "$object" "$library" \
    "$out_dir/fortai_cuda_backend.o" "$out_dir/fortai_cuda_q4_backend.o" \
    -L"$ggml_prefix/lib" -Xlinker -rpath -Xlinker "$ggml_prefix/lib" \
    -lggml-cuda -lggml-cpu -lggml -lggml-base -lgfortran -lquadmath -lgomp \
    -o "$out_dir/fortai_cuda_run"
printf 'binary=%s\narchive=%s\nobject=%s\narch=%s\nnvcc=%s\n' \
    "$out_dir/fortai_cuda_run" "$library" "$object" "$cuda_arch" "$nvcc_bin"
