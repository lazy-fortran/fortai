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
digest_output=$("$root_dir/tools/worktree_digest.sh")
patch_digest=$(printf '%s\n' "$digest_output" | sed -n 's/^patch_digest=//p')
tree_digest=$(printf '%s\n' "$digest_output" | sed -n 's/^tracked_tree_digest=//p')
worktree_digest=$(printf '%s\n' "$digest_output" | sed -n 's/^worktree_digest=//p')

(cd "$root_dir" && fo build --flag "$native_flags" && \
    env OMP_NUM_THREADS="$OMP_NUM_THREADS" OMP_PROC_BIND="$OMP_PROC_BIND" \
        OMP_PLACES="$OMP_PLACES" fo exec --no-build fortai_bench_cpu_matvec \
        "$rows" "$cols" "$repeats") >"$raw_file"
fortai_executable=$(readlink -f "$root_dir/build/fo/app/fortai_bench_cpu_matvec")
if [[ ! -x "$fortai_executable" ]]; then
    echo "FortAI CPU benchmark executable not found: $fortai_executable" >&2
    exit 1
fi
fortai_executable_sha256=$(sha256sum "$fortai_executable" | awk '{print $1}')
digest_after=$("$root_dir/tools/worktree_digest.sh")
if [[ "$digest_after" != "$digest_output" ]]; then
    echo 'worktree changed during CPU reference benchmark' >&2
    exit 1
fi
{
    printf '# fortai_commit=%s\n' "$(git -C "$root_dir" rev-parse HEAD)"
    printf '# fortai_patch_digest=%s\n' "$patch_digest"
    printf '# fortai_tracked_tree_digest=%s\n' "$tree_digest"
    printf '# fortai_worktree_digest=%s\n' "$worktree_digest"
    printf '# build_flags=%s\n' "$native_flags"
    printf '# compiler=%s\n' "$(gfortran --version | head -n 1)"
    printf '# omp_num_threads=%s\n' "$OMP_NUM_THREADS"
    printf '# omp_proc_bind=%s\n' "$OMP_PROC_BIND"
    printf '# omp_places=%s\n' "$OMP_PLACES"
    printf '# fortai_executable=%s\n' "$fortai_executable"
    printf '# fortai_executable_sha256=%s\n' "$fortai_executable_sha256"
    rg '^(backend,|cpu-native,)' "$raw_file"
} | tee "$result_file"

printf 'result=%s\nraw=%s\n' "$result_file" "$raw_file"
