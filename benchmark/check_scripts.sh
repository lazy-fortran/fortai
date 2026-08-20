#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bash -n "$root_dir/tools/build_native.sh"
bash -n "$root_dir/benchmark/run_cpu_reference.sh"
bash -n "$root_dir/benchmark/run_llama_cpp.sh"
bash -n "$root_dir/benchmark/run_qwen35_cpu.sh"
bash -n "$root_dir/benchmark/compare_qwen35_cpu_llama.sh"
bash -n "$root_dir/benchmark/repeat_compare_qwen35_cpu.sh"
bash -n "$root_dir/benchmark/tune_qwen35_cpu.sh"
bash -n "$root_dir/benchmark/profile_qwen35_cpu.sh"
bash -n "$root_dir/benchmark/check_qwen35_cpu_trace.sh"
bash -n "$root_dir/benchmark/check_qwen35_cuda_trace.sh"
bash -n "$root_dir/benchmark/compare_cuda_q8.sh"
bash -n "$root_dir/benchmark/compare_qwen35_cuda_llama.sh"
bash -n "$root_dir/benchmark/repeat_compare_qwen35_cuda.sh"
bash -n "$root_dir/benchmark/profile_cuda_q8.sh"
bash -n "$root_dir/benchmark/profile_qwen35_cuda.sh"
bash -n "$root_dir/benchmark/profile_qwen35_cuda_llama.sh"
bash -n "$root_dir/benchmark/run_cuda_gdn_ar_bench.sh"
bash -n "$root_dir/benchmark/check_cuda_backend.sh"
bash -n "$root_dir/tools/build_cuda.sh"
bash -n "$root_dir/tools/build_llama_cuda_q8.sh"
bash -n "$root_dir/tools/build_cuda_backend.sh"
bash -n "$root_dir/tools/build_cuda_qwen35.sh"
bash -n "$root_dir/.provenance/fetch_models.sh"
"$root_dir/tools/worktree_digest.sh" >/dev/null
if rg -n '(^|[[:space:]])fpm (run|test|build)' "$root_dir/benchmark" "$root_dir/tools" >/dev/null; then
    echo 'benchmark/tools must use fo, not direct fpm commands' >&2
    exit 1
fi
"$root_dir/benchmark/run_llama_cpp.sh" --dry-run >/dev/null
"$root_dir/.provenance/fetch_models.sh" --dry-run >/dev/null
printf '%s\n' 'benchmark scripts syntax check passed'
