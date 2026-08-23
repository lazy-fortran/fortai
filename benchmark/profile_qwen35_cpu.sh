#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_path="${1:-${FORTAI_MODEL:-}}"
if [[ -z "$model_path" ]]; then
    echo "usage: benchmark/profile_qwen35_cpu.sh MODEL.gguf [token_id] [steps] [context]" >&2
    exit 2
fi
if [[ ! -f "$model_path" ]]; then
    echo "model not found: $model_path" >&2
    exit 2
fi
if ! command -v perf >/dev/null 2>&1; then
    echo "perf is required for profiling" >&2
    exit 2
fi

token_id="${2:-${FORTAI_TOKEN_ID:-9419}}"
steps="${3:-${FORTAI_PROFILE_STEPS:-128}}"
context="${4:-${FORTAI_CONTEXT:-128}}"
threads="${OMP_NUM_THREADS:-$(nproc)}"
frequency="${FORTAI_PERF_FREQUENCY:-999}"
delay_ms="${FORTAI_PERF_DELAY_MS:-0}"
perf_mmap="${FORTAI_PERF_MMAP:-64}"
perf_call_graph="${FORTAI_PERF_CALL_GRAPH:-dwarf,32}"
base=$(basename "$model_path" .gguf)
stamp=$(date -u +%Y%m%dT%H%M%SZ)
profile_dir="$root_dir/benchmark/profiles/${base}_${stamp}"
mkdir -p "$profile_dir"

export OMP_NUM_THREADS="$threads"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-cores}"
native_flags="${FORTAI_NATIVE_FLAGS:--O2 -march=native -mtune=native -funroll-loops -fopenmp -fno-fast-math -ffp-contract=off -fno-math-errno -flto}"

{
    "$root_dir/tools/worktree_digest.sh"
    printf 'fortai_commit=%s\n' "$(git -C "$root_dir" rev-parse HEAD)"
    printf 'model_sha256=%s\n' "$(sha256sum "$model_path" | awk '{print $1}')"
    printf 'compiler=%s\n' "$(gfortran --version | head -n 1)"
    printf 'build_flags=%s\n' "$native_flags"
    printf 'omp_num_threads=%s\n' "$OMP_NUM_THREADS"
    printf 'omp_proc_bind=%s\n' "$OMP_PROC_BIND"
    printf 'omp_places=%s\n' "$OMP_PLACES"
    printf 'perf_mmap=%s\n' "$perf_mmap"
    printf 'perf_call_graph=%s\n' "$perf_call_graph"
    printf 'perf_delay_ms=%s\n' "$delay_ms"
} >"$profile_dir/provenance.txt"

(cd "$root_dir" && fo build --flag "$native_flags") >"$profile_dir/build.log"
run=(fo exec --no-build --cwd "$root_dir" fortai_cpu_run "$model_path" "$token_id" "$steps" "$context")
events="task-clock,context-switches,cpu-migrations,cycles,instructions,branches,branch-misses,cache-references,cache-misses"

env OMP_NUM_THREADS="$OMP_NUM_THREADS" OMP_PROC_BIND="$OMP_PROC_BIND" OMP_PLACES="$OMP_PLACES" \
    perf stat -D "$delay_ms" -x, -e "$events" -o "$profile_dir/perf-stat.csv" -- \
    "${run[@]}" >"$profile_dir/stat-run.log" 2>&1

env OMP_NUM_THREADS="$OMP_NUM_THREADS" OMP_PROC_BIND="$OMP_PROC_BIND" OMP_PLACES="$OMP_PLACES" \
    perf record -D "$delay_ms" -F "$frequency" -m "$perf_mmap" --call-graph "$perf_call_graph" \
    -o "$profile_dir/perf.data" -- \
    "${run[@]}" >"$profile_dir/record-run.log" 2>&1

perf report --stdio --no-children --sort symbol,dso -i "$profile_dir/perf.data" \
    >"$profile_dir/perf-report.txt"
perf script -i "$profile_dir/perf.data" >"$profile_dir/perf-script.txt"
objdump -d -Mintel "$root_dir/build/fo/bin/fortai_cpu_run" \
    >"$profile_dir/fortai-objdump.txt"

if rg -q '^# Total Lost Samples: [1-9]' "$profile_dir/perf-report.txt"; then
    echo "profile contains lost samples: $profile_dir/perf-report.txt" >&2
    exit 1
fi

printf 'profile_dir=%s\nstat=%s\nreport=%s\n' \
    "$profile_dir" "$profile_dir/perf-stat.csv" "$profile_dir/perf-report.txt"
