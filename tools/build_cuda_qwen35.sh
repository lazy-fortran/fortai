#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
nvcc_bin=${NVCC:-/opt/cuda/bin/nvcc}
cuda_arch=${FORTAI_CUDA_ARCH:-sm_120}
out_dir="$root_dir/build/cuda"
native_flags="${FORTAI_NATIVE_FLAGS:--O3 -march=native -mtune=native -funroll-loops -fopenmp -ffast-math -fno-math-errno -flto}"
link_flags="${native_flags// -flto/} -fno-plt"
log_file="$out_dir/fortai_cuda_qwen35_fo-build.log"
inputs_file="$out_dir/fortai_cuda_qwen35.inputs"

if [[ ! -x "$nvcc_bin" ]]; then
    echo "nvcc not found: $nvcc_bin" >&2
    exit 2
fi
mkdir -p "$out_dir"

"$root_dir/tools/build_cuda_backend.sh" >"$out_dir/fortai_cuda_qwen35_backend-build.log"
(cd "$root_dir" && FO_DEBUG_LINKS=1 fo build --flag "$link_flags") >"$log_file" 2>&1

link_line=$(rg 'fo link: .*app_fortai_cuda_run\.f90\.o ' "$log_file" | tail -n 1 | sed 's/^fo link: //' || true)
if [[ -n "$link_line" ]]; then
    library=$(printf '%s\n' "$link_line" | grep -oE '/[^ ]*/build/fo/lib/objects_[^ ]+\.a' | tail -n 1)
    object=$(printf '%s\n' "$link_line" | grep -oE '/[^ ]*/build/fo/obj/app_fortai_cuda_run\.f90\.o' | tail -n 1)
elif [[ -f "$inputs_file" ]]; then
    library=$(sed -n 's/^library=//p' "$inputs_file")
    object=$(sed -n 's/^object=//p' "$inputs_file")
else
    object="$root_dir/build/fo/obj/app_fortai_cuda_run.f90.o"
    library=$(for archive in "$root_dir"/build/fo/lib/objects_*.a; do
        if ar t "$archive" 2>/dev/null | rg -q 'src_models_qwen35_fortai_qwen35_cpu'; then
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
    "$out_dir/fortai_cuda_backend.o" -lgfortran -lquadmath -lgomp \
    -o "$out_dir/fortai_cuda_run"
printf 'binary=%s\narchive=%s\nobject=%s\narch=%s\nnvcc=%s\n' \
    "$out_dir/fortai_cuda_run" "$library" "$object" "$cuda_arch" "$nvcc_bin"
