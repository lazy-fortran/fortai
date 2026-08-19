#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bash -n "$root_dir/tools/build_native.sh"
bash -n "$root_dir/benchmark/run_cpu_reference.sh"
bash -n "$root_dir/benchmark/run_llama_cpp.sh"
bash -n "$root_dir/benchmark/run_qwen35_cpu.sh"
bash -n "$root_dir/benchmark/compare_qwen35_cpu_llama.sh"
bash -n "$root_dir/.provenance/fetch_models.sh"
"$root_dir/benchmark/run_llama_cpp.sh" --dry-run >/dev/null
"$root_dir/.provenance/fetch_models.sh" --dry-run >/dev/null
printf '%s\n' 'benchmark scripts syntax check passed'
