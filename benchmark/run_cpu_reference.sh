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

fpm run --target fortai_bench_cpu_matvec --profile native \
    --link-flag '-fopenmp -flto' -- "$rows" "$cols" "$repeats" >"$raw_file"
{
    printf '# fortai_commit=%s\n' "$(git -C "$root_dir" rev-parse HEAD)"
    printf '# compiler=%s\n' "$(gfortran --version | head -n 1)"
    printf '# omp_num_threads=%s\n' "$OMP_NUM_THREADS"
    rg '^(backend,|cpu-native,)' "$raw_file"
} | tee "$result_file"

printf 'result=%s\nraw=%s\n' "$result_file" "$raw_file"
