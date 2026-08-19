#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
llama_source=${LLAMA_SOURCE:-/home/ert/.local/src/llama.cpp}
llama_lib=${LLAMA_LIBRARY_DIR:-/home/ert/.local/llama.cpp}
out_dir="$root_dir/build/cuda"
mkdir -p "$out_dir"
if [[ ! -f "$llama_lib/libggml-cuda.so" || ! -d "$llama_source/ggml/include" ]]; then
    echo "llama.cpp CUDA installation/source not found" >&2
    exit 2
fi
g++ -O3 -std=c++17 -DNDEBUG -I"$llama_source/ggml/include" \
    "$root_dir/tools/llama_cuda_q8_bench.cpp" -L"$llama_lib" \
    -Wl,-rpath,"$llama_lib" -Wl,--no-as-needed -lggml-cuda -lggml -lggml-base \
    -o "$out_dir/llama_cuda_q8_bench"
printf 'binary=%s\nllama_source=%s\nllama_library=%s\n' "$out_dir/llama_cuda_q8_bench" "$llama_source" "$llama_lib"
