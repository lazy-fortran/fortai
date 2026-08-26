#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
nvcc_bin=${NVCC:-/opt/cuda/bin/nvcc}
cuda_arch=${FORTAI_CUDA_ARCH:-sm_120}
out_dir="$root_dir/build/cuda"
ggml_prefix=${FORTAI_GGML_PREFIX:-/home/ert/.local/llama.cpp-upstream-main-650913862}
# nvcc uses GCC 15 on this host while the default Fortran toolchain is GCC 16;
# a non-LTO archive keeps the CUDA link reproducible across those toolchains.
native_flags="${FORTAI_NATIVE_FLAGS:--O3 -march=native -mtune=native -funroll-loops -fopenmp -ffast-math -fno-math-errno -fno-lto}"
allow_lto=${FORTAI_ALLOW_LTO:-0}
link_flags="${native_flags// -flto/} -fno-plt -L${ggml_prefix}/lib -Wl,-rpath,${ggml_prefix}/lib -lggml-cpu -lggml-cuda -lggml -lggml-base -lpthread -ldl"
log_file="$out_dir/fortai_cuda_server_fo-build.log"

if [[ ! -x "$nvcc_bin" ]]; then
    echo "nvcc not found: $nvcc_bin" >&2
    exit 2
fi
mkdir -p "$out_dir"
"$root_dir/tools/build_cuda_backend.sh" >"$out_dir/fortai_cuda_server_backend-build.log"
(cd "$root_dir" && FO_DEBUG_LINKS=1 fo build --flag "$link_flags") >"$log_file" 2>&1

link_line=$(rg 'fo link: .*app_fortai_server\.f90\.o ' "$log_file" | tail -n 1 | sed 's/^fo link: //' || true)
library=''
object=''
if [[ -n "$link_line" ]]; then
    library=$(printf '%s\n' "$link_line" | grep -oE '/[^ ]*/build/fo/lib/objects_[^ ]+\.a' | tail -n 1)
    object=$(printf '%s\n' "$link_line" | grep -oE '/[^ ]*/build/fo/obj/app_fortai_server\.f90\.o' | tail -n 1)
fi
archive_is_lto() {
    if [[ "$allow_lto" == 1 ]]; then return 1; fi
    strings "$1" | rg 'gnu\.lto|GNU GIMPLE' >/dev/null 2>&1
}
if [[ -n "$library" && -f "$library" ]] && archive_is_lto "$library"; then
    library=''
fi
# A warm fo action-cache hit intentionally emits no link command.  Resolve the
# same two artifacts from the project build tree in that case; otherwise a
# correct cached build cannot be relinked with the CUDA objects.
if [[ -z "$object" ]]; then
    object="$root_dir/build/fo/obj/app_fortai_server.f90.o"
fi
if [[ -z "$library" ]]; then
    library=$(for archive in "$root_dir"/build/fo/lib/objects_*.a; do
        if ! archive_is_lto "$archive"; then
            stat -c '%Y %n' "$archive"
        fi
    done | sort -nr | head -n 1 | cut -d' ' -f2- || true)
fi
if [[ -z "$library" || -z "$object" || ! -f "$library" || ! -f "$object" ]]; then
    echo "could not resolve FortAI server CUDA inputs" >&2
    exit 1
fi

"$nvcc_bin" -O3 -arch="$cuda_arch" "$object" "$library" \
    "$out_dir/fortai_cuda_backend.o" "$out_dir/fortai_cuda_q4_backend.o" \
    -L"$ggml_prefix/lib" -Xlinker -rpath -Xlinker "$ggml_prefix/lib" \
    -lggml-cuda -lggml-cpu -lggml -lggml-base \
    -lgfortran -lquadmath -lgomp -lpthread -ldl -o "$out_dir/fortai_server"
printf 'binary=%s\narchive=%s\nobject=%s\narch=%s\nnvcc=%s\n' \
    "$out_dir/fortai_server" "$library" "$object" "$cuda_arch" "$nvcc_bin"
