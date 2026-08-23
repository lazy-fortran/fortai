#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fixture_dir=$(mktemp -d)
log_file="$fixture_dir/guard.log"
model_path="$fixture_dir/Qwen3.8-27B-Q4_K_XL.gguf"
cleanup() {
    find "$fixture_dir" -mindepth 1 -delete
    rmdir "$fixture_dir"
}
trap cleanup EXIT

touch "$model_path"
if "$root_dir/benchmark/tournament_qwen35_cpu.sh" "$model_path" >"$log_file" 2>&1; then
    echo 'tournament accepted an out-of-scope model' >&2
    exit 1
fi
if ! grep -Fq 'tournament model scope is limited to Qwen3.5 0.8B/2B/4B Q8_0' "$log_file"; then
    echo 'tournament scope guard did not reject before execution' >&2
    cat "$log_file" >&2
    exit 1
fi
printf '%s\n' 'CPU tournament scope guard passed'
