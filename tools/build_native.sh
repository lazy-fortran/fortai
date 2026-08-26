#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root_dir"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-cores}"
native_flags="${FORTAI_NATIVE_FLAGS:--O2 -march=native -mtune=native -funroll-loops -fopenmp -fno-fast-math -ffp-contract=off -fno-math-errno -flto}"
ggml_prefix=${FORTAI_GGML_PREFIX:-/home/ert/.local/llama.cpp-upstream-main-650913862}
link_flags="${native_flags} -L${ggml_prefix}/lib -Wl,-rpath,${ggml_prefix}/lib -lggml-cpu -lggml-cuda -lggml -lggml-base -lpthread -ldl"

if [[ ! -f "$ggml_prefix/include/ggml.h" || ! -f "$ggml_prefix/lib/libggml-cpu.so" ]]; then
    echo "GGML CPU backend not found under $ggml_prefix; set FORTAI_GGML_PREFIX" >&2
    exit 2
fi

fo build --flag "$link_flags"
fo test --flag "$link_flags"
