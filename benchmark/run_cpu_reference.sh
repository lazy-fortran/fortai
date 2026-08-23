#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
result_dir="$root_dir/benchmark/results"
mkdir -p "$result_dir"

rows="${1:-1024}"
cols="${2:-1024}"
repeats="${3:-20}"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
result_file="$result_dir/cpu_matvec_${stamp}.csv"
raw_file="$result_dir/cpu_matvec_${stamp}.log"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-cores}"
native_flags="${FORTAI_NATIVE_FLAGS:--O2 -march=native -mtune=native -funroll-loops -fopenmp -fno-fast-math -ffp-contract=off -fno-math-errno -flto}"

(cd "$root_dir" && fo build --flag "$native_flags" && \
    env OMP_NUM_THREADS="$OMP_NUM_THREADS" OMP_PROC_BIND="$OMP_PROC_BIND" \
        OMP_PLACES="$OMP_PLACES" fo exec --no-build fortai_bench_cpu_matvec \
        "$rows" "$cols" "$repeats") >"$raw_file"
{
    printf '# fortai_commit=%s\n' "$(git -C "$root_dir" rev-parse HEAD)"
    printf '# fortai_patch_digest=%s\n' "$(git -C "$root_dir" diff | sha256sum | awk '{print $1}')"
    printf '# build_flags=%s\n' "$native_flags"
    printf '# compiler=%s\n' "$(gfortran --version | head -n 1)"
    printf '# omp_num_threads=%s\n' "$OMP_NUM_THREADS"
    rg '^(backend,|cpu-native,)' "$raw_file"
} | tee "$result_file"

printf 'result=%s\nraw=%s\n' "$result_file" "$raw_file"
