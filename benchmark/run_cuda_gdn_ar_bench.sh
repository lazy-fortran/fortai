#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_file="$root_dir/backend/cuda/fortai_cuda_gdn_ar_bench.cu"
fixture="${FORTAI_GDN_FIXTURE:-$root_dir/.provenance/downloads/qwen35-0.8b/Qwen3.5-0.8B-Q8_0.gguf}"
nvcc="${NVCC:-/opt/cuda/bin/nvcc}"
device="${FORTAI_CUDA_DEVICE:-0}"
tokens="${FORTAI_GDN_TOKENS:-128}"
warmup="${FORTAI_GDN_WARMUP:-32}"
repeats="${FORTAI_GDN_REPEATS:-9}"
heads=16
head_size=128
fixture_discovered=false
fixture_sha=""

if [[ ! -x "$nvcc" ]]; then
    echo "nvcc not found: $nvcc" >&2
    exit 2
fi

gguf_python="$root_dir/.provenance/src/llama.cpp/gguf-py"
if [[ -f "$fixture" && -d "$gguf_python/gguf" ]]; then
    dimensions=$(PYTHONPATH="$gguf_python" python3 - "$fixture" <<'PY'
import sys
from gguf import GGUFReader

reader = GGUFReader(sys.argv[1], "r")
values = {field.name: field.contents() for field in reader.fields.values()}
state_size = int(values["qwen35.ssm.state_size"])
key_heads = int(values["qwen35.ssm.group_count"])
value_heads = int(values["qwen35.ssm.time_step_rank"])
inner_size = int(values["qwen35.ssm.inner_size"])
head_size = inner_size // value_heads
if inner_size % value_heads or state_size != head_size or key_heads != value_heads:
    raise SystemExit("fixture has unsupported asymmetric GDN dimensions")
print(key_heads, head_size)
PY
)
    read -r heads head_size <<<"$dimensions"
    fixture_sha=$(sha256sum "$fixture" | awk '{print $1}')
    fixture_discovered=true
fi

build_dir=$(mktemp -d "${TMPDIR:-/tmp}/fortai-gdn-ar.XXXXXX")
trap 'rm -rf "$build_dir"' EXIT
binary="$build_dir/fortai_cuda_gdn_ar_bench"
build_flags=(-O3 -std=c++17 -lineinfo -arch=native)
"$nvcc" "${build_flags[@]}" "$source_file" -o "$binary"

result_dir="$root_dir/benchmark/results"
mkdir -p "$result_dir"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
result_file="${FORTAI_GDN_RESULT:-$result_dir/cuda_gdn_ar_${stamp}.txt}"

export FORTAI_BENCH_GIT_COMMIT
export FORTAI_BENCH_PATCH_SHA
export FORTAI_BENCH_SOURCE_SHA
export FORTAI_BENCH_RUNNER_SHA
export FORTAI_BENCH_FIXTURE="$fixture"
export FORTAI_BENCH_FIXTURE_SHA="$fixture_sha"
export FORTAI_BENCH_FIXTURE_DISCOVERED="$fixture_discovered"
export FORTAI_BENCH_BUILD="${build_flags[*]}"
export FORTAI_BENCH_NVCC
export FORTAI_BENCH_UTC="$stamp"
FORTAI_BENCH_GIT_COMMIT=$(git -C "$root_dir" rev-parse HEAD)
FORTAI_BENCH_PATCH_SHA=$(git -C "$root_dir" diff --binary | sha256sum | awk '{print $1}')
FORTAI_BENCH_SOURCE_SHA=$(sha256sum "$source_file" | awk '{print $1}')
FORTAI_BENCH_RUNNER_SHA=$(sha256sum "$root_dir/benchmark/run_cuda_gdn_ar_bench.sh" | awk '{print $1}')
FORTAI_BENCH_NVCC=$($nvcc --version | tail -n 1)

"$binary" --device "$device" --heads "$heads" --head-size "$head_size" \
    --tokens "$tokens" --warmup "$warmup" --repeats "$repeats" "$@" | tee "$result_file"
printf 'result=%s\n' "$result_file"
