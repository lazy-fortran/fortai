#!/usr/bin/env bash
# Reproducer for the long-prompt CUDA fault.
#
# Starts one native server and drives a configurable ladder of prompt sizes
# repeatedly inside that single process, so many trials share one model load.
# Exits non-zero at the first failed request and prints the server log tail.
#
#   BENCH_LADDER  comma-separated prompt sizes per trial
#   BENCH_TRIALS  how many times to repeat the ladder
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
env_file="${BENCH_ENV_FILE:-/home/ert/.config/systemd/user/slopcode-llamacpp.service.d/local.conf}"
fortai_bin="${BENCH_FORTAI_BIN:-$root_dir/build/cuda/fortai_server}"
port="${BENCH_PORT:-18095}"
ladder="${BENCH_LADDER:-256,1024,4096,16384}"
trials="${BENCH_TRIALS:-4}"
log_dir="$root_dir/benchmark/logs"
mkdir -p "$log_dir"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log_file="$log_dir/repro_long_prompt_${stamp}.log"

[[ -x "$fortai_bin" ]] || { echo "fortai_server not built: $fortai_bin" >&2; exit 2; }
if pgrep -x fortai_server >/dev/null 2>&1 || pgrep -x llama-server >/dev/null 2>&1; then
    echo "another inference server is running; stop it first" >&2
    exit 2
fi

set -a
# shellcheck disable=SC1090
. <(sed -n 's/^Environment=//p' "$env_file")
set +a
export LLAMACPP_HOST=127.0.0.1
export LLAMACPP_PORT="$port"
export LLAMACPP_CONTEXT="${BENCH_CONTEXT:-$LLAMACPP_CONTEXT}"

"$fortai_bin" >"$log_file" 2>&1 &
server_pid=$!
cleanup() {
    kill -TERM "$server_pid" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "$server_pid" 2>/dev/null || break; sleep 1; done
    kill -KILL "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for attempt in $(seq 1 900); do
    curl -fsS --max-time 5 "http://127.0.0.1:${port}/health" >/dev/null 2>&1 && break
    kill -0 "$server_pid" 2>/dev/null || { echo "server exited during startup" >&2; tail -20 "$log_file" >&2; exit 1; }
    sleep 1
    [[ "$attempt" == 900 ]] && { echo "server did not become healthy" >&2; exit 1; }
done

set +e
env PORT="$port" LADDER="$ladder" TRIALS="$trials" python3 - <<'PY'
import json, os, sys, urllib.request
port = int(os.environ["PORT"])
ladder = [int(v) for v in os.environ["LADDER"].split(",")]
trials = int(os.environ["TRIALS"])
# match the benchmark sweep: two rate requests plus one streaming request
max_tokens = 128
BASE = f"http://127.0.0.1:{port}"
WORDS = ("the quick brown fox jumps over the lazy dog while a Fortran compiler "
         "emits vectorized kernels for dense linear algebra on modern hardware ")


def post(path, payload):
    request = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(request, timeout=1800).read())


def filler(target):
    text = WORDS
    while len(post("/tokenize", {"content": text})["tokens"]) < target:
        text = text * 2
    words = text.split(" ")
    low, high = 1, len(words)
    while low < high:
        mid = (low + high) // 2
        if len(post("/tokenize", {"content": " ".join(words[:mid])})["tokens"]) >= target:
            high = mid
        else:
            low = mid + 1
    return " ".join(words[:low])


prompts = {size: filler(size) for size in sorted(set(ladder))}
n = 0
for trial in range(trials):
    for size in ladder:
        n += 1
        try:
            for repeat in range(2):
                tag = f"[repro-{trial}-{size}-{n}-{repeat}] "
                post("/v1/chat/completions",
                     {"model": "qwen",
                      "messages": [{"role": "user", "content": tag + prompts[size]}],
                      "max_tokens": max_tokens, "temperature": 0.0})
            print(f"  request {n:3d} (trial {trial}, ~{size} tok): ok", flush=True)
        except Exception as exc:
            print(f"  request {n:3d} (trial {trial}, ~{size} tok): FAILED {exc}", flush=True)
            sys.exit(1)
print(f"\nno failure in {n} requests")
PY
status=$?
set -e
echo "--- server log (batch/forward failures) ---"
grep -nE 'prompt batch failed|forward failed|without MTP' "$log_file" | tail -20 || echo "(none)"
printf 'log=%s\n' "$log_file"
exit "$status"
