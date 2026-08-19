#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
device=${CUDA_DEVICE:-0}
rows=${FORTAI_CUDA_ROWS:-4096}
width=${FORTAI_CUDA_WIDTH:-4096}
iterations=${FORTAI_CUDA_ITERATIONS:-1000}
warmup=${FORTAI_CUDA_WARMUP:-100}
seed=${FORTAI_CUDA_SEED:-0x6f727461}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
base="$root_dir/benchmark/results/cuda_q8_${rows}x${width}_d${device}_${stamp}"
mkdir -p "$root_dir/benchmark/results"
if pgrep -x llama-server >/dev/null 2>&1; then
    echo "an existing llama-server is running; stop it before GPU benchmarking" >&2
    exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required" >&2
    exit 2
fi

"$root_dir/tools/build_cuda.sh" >"$base.build-fortai.log"
"$root_dir/tools/build_llama_cuda_q8.sh" >"$base.build-llama.log"
"$root_dir/tools/worktree_digest.sh" >"$base.worktree.txt"
{
    printf 'commit=%s\nmodel_scope=resident_q8_gemv_microbenchmark\n' "$(git -C "$root_dir" rev-parse HEAD)"
    printf 'device=%s\nrows=%s\nwidth=%s\niterations=%s\nwarmup=%s\nseed=%s\n' \
        "$device" "$rows" "$width" "$iterations" "$warmup" "$seed"
    printf 'nvcc=%s\n' "$(/opt/cuda/bin/nvcc --version | tail -n 1)"
    printf 'llama_library=%s\n' "${LLAMA_LIBRARY_DIR:-/home/ert/.local/llama.cpp}"
    nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv,noheader
} >"$base.provenance.txt"

args=(--device "$device" --rows "$rows" --width "$width" --iterations "$iterations" --warmup "$warmup" --seed "$seed")
"$root_dir/build/cuda/fortai_cuda_q8_bench" "${args[@]}" >"$base.fortai.json"
llama_lib=${LLAMA_LIBRARY_DIR:-/home/ert/.local/llama.cpp}
export LD_LIBRARY_PATH="$llama_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$root_dir/build/cuda/llama_cuda_q8_bench" "${args[@]}" >"$base.llama.json"

FORTAI_JSON="$base.fortai.json" LLAMA_JSON="$base.llama.json" SUMMARY="$base.json" python3 - <<'PY'
import json
import os
from pathlib import Path

fortai = json.loads(Path(os.environ["FORTAI_JSON"]).read_text())
llama = json.loads(Path(os.environ["LLAMA_JSON"]).read_text())
if not fortai.get("correct") or not llama.get("correct"):
    raise SystemExit("CUDA correctness gate failed")
summary = {
    "scope": "resident_q8_gemv_microbenchmark",
    "fortai": fortai,
    "llama_cpp": llama,
    "fortai_over_llama": fortai["tokens_per_second"] / llama["tokens_per_second"],
    "artifacts": {"fortai": os.environ["FORTAI_JSON"], "llama_cpp": os.environ["LLAMA_JSON"]},
}
Path(os.environ["SUMMARY"]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY
printf 'summary=%s\n' "$base.json"
