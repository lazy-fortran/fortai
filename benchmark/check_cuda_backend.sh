#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
device=${CUDA_DEVICE:-0}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
result="$root_dir/benchmark/results/cuda_backend_smoke_d${device}_${stamp}.json"
mkdir -p "$root_dir/benchmark/results"
if pgrep -x llama-server >/dev/null 2>&1; then
    echo "an existing llama-server is running; stop it before CUDA backend checks" >&2
    exit 2
fi

"$root_dir/tools/build_cuda_backend.sh" >"$result.build.log"
"$root_dir/tools/worktree_digest.sh" >"$result.worktree.txt"
{
    printf 'commit=%s\nmodel_scope=resident_q8_backend_abi\n' "$(git -C "$root_dir" rev-parse HEAD)"
    printf 'device=%s\n' "$device"
    nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv,noheader
} >"$result.provenance.txt"

"$root_dir/build/cuda/cuda_backend_smoke" --device "$device" >"$result"
cat "$result"
"$root_dir/build/cuda/fortran_cuda_backend_smoke" >"$result.fortran.json"
cat "$result.fortran.json"
printf 'result=%s\n' "$result"
