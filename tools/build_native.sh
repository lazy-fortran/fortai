#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root_dir"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-cores}"

fpm build --profile native --link-flag '-fopenmp -flto'
fpm test --profile native --link-flag '-fopenmp -flto'
