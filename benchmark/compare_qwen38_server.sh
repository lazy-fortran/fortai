#!/usr/bin/env bash
# Matched Qwen3.8-27B server benchmark for one runtime.
#
# Usage: benchmark/compare_qwen38_server.sh fortai|llama
#
# Both sides read the same production environment file (the slopcode-llamacpp
# systemd drop-in) so model, draft, MTP depth, tensor split, batching, K/V
# types and sampler are identical.  Only the context differs: llama.cpp cannot
# allocate the 262144-token production cache.  Every request carries a unique
# nonce prefix, which defeats the shared prefix cache and keeps each prefill
# cold.  BENCH_PROMPT_SIZES and BENCH_CONTEXT are the only knobs.
set -euo pipefail

side="${1:-}"
case "$side" in
    fortai|llama) ;;
    *) echo "usage: benchmark/compare_qwen38_server.sh fortai|llama" >&2; exit 2 ;;
esac

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
env_file=/home/ert/.config/systemd/user/slopcode-llamacpp.service.d/local.conf
start_script=/home/ert/infra/slopcode-infra/scripts/server_start_llamacpp.sh
fortai_bin="$root_dir/build/cuda/fortai_server"
port=18091
gen_tokens=128
repeats=2
prompt_sizes="${BENCH_PROMPT_SIZES:-256,1024,4096,8192,12288,16384}"
# llama.cpp cannot allocate the production K/V cache; 32768 is the largest
# context that both loads and covers the sweep's longest prompt.
llama_context=32768
result_dir="$root_dir/benchmark/results"
log_dir="$root_dir/benchmark/logs"
mkdir -p "$result_dir" "$log_dir"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log_file="$log_dir/qwen38_server_${side}_${stamp}.log"
result_file="$result_dir/qwen38_server_${side}_${stamp}.json"

[[ -f "$env_file" ]] || { echo "environment file missing: $env_file" >&2; exit 2; }
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${port}$"; then
    echo "benchmark port is already in use: $port" >&2
    exit 2
fi
if pgrep -x llama-server >/dev/null 2>&1 || pgrep -x fortai_server >/dev/null 2>&1; then
    echo "another inference server is running; stop it before benchmarking" >&2
    exit 2
fi

set -a
# shellcheck disable=SC1090
. <(sed -n 's/^Environment=//p' "$env_file")
set +a
export LLAMACPP_HOST=127.0.0.1
# BENCH_CONTEXT overrides the context for either side, so FortAI can also be
# measured at the reduced context llama.cpp is able to allocate.
export LLAMACPP_CONTEXT="${BENCH_CONTEXT:-$LLAMACPP_CONTEXT}"
export LLAMACPP_PORT="$port"
export LLAMACPP_INSTANCE=bench

server_pid=''
cleanup() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        for _ in $(seq 1 60); do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [[ "$side" == fortai ]]; then
    [[ -x "$fortai_bin" ]] || { echo "fortai_server not built: $fortai_bin" >&2; exit 2; }
    context="$LLAMACPP_CONTEXT"
    "$fortai_bin" >"$log_file" 2>&1 &
    server_pid=$!
else
    [[ -x "$start_script" ]] || { echo "llama start script missing: $start_script" >&2; exit 2; }
    context="$llama_context"
    export LLAMACPP_CONTEXT="$llama_context"
    command_line=$(LLAMACPP_DRY_RUN=true bash "$start_script" | tail -n 1)
    # The dry run prints the raw binary; the llama-server wrapper is what
    # normally supplies the build-local shared libraries.
    export LD_LIBRARY_PATH="${LLAMACPP_HOME}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    printf 'command: %s\n' "$command_line" >"$log_file"
    eval "exec $command_line" >>"$log_file" 2>&1 &
    server_pid=$!
fi

for attempt in $(seq 1 900); do
    curl -fsS --max-time 5 "http://127.0.0.1:${port}/health" >/dev/null 2>&1 && break
    kill -0 "$server_pid" 2>/dev/null || { echo "$side server exited during startup" >&2; tail -20 "$log_file" >&2; exit 1; }
    sleep 1
    [[ "$attempt" == 900 ]] && { echo "$side server did not become healthy" >&2; exit 1; }
done

env SIDE="$side" PORT="$port" CONTEXT="$context" GEN_TOKENS="$gen_tokens" REPEATS="$repeats" \
    RESULT_FILE="$result_file" LOG_FILE="$log_file" PROMPT_SIZES="$prompt_sizes" \
    SERVER_PID="$server_pid" \
    python3 "$root_dir/benchmark/qwen38_server_probe.py"
printf 'result=%s\nlog=%s\n' "$result_file" "$log_file"
