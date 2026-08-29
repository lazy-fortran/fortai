#!/usr/bin/env bash
# Correctness oracle for the Qwen3.8-27B MTP path.
#
# Speculative decoding must not change what the model produces.  This runs the
# same greedy prompts twice through the native server -- once with MTP
# speculation active, once with FORTAI_NATIVE_MTP=0 so every token comes from
# the non-speculative scalar path -- and requires the two outputs to be
# identical.  The scalar path is the independent behavioral oracle here: it
# shares no draft head, no verification batch, and no rollback logic with the
# speculative path.
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
env_file="${BENCH_ENV_FILE:-/home/ert/.config/systemd/user/slopcode-llamacpp.service.d/local.conf}"
fortai_bin="${BENCH_FORTAI_BIN:-$root_dir/build/cuda/fortai_server}"
port="${BENCH_PORT:-18096}"
gen_tokens="${BENCH_GEN_TOKENS:-128}"
log_dir="$root_dir/benchmark/logs"
result_dir="$root_dir/benchmark/results"
mkdir -p "$log_dir" "$result_dir"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
result_file="$result_dir/qwen38_mtp_equivalence_${stamp}.json"

[[ -x "$fortai_bin" ]] || { echo "fortai_server not built: $fortai_bin" >&2; exit 2; }
if pgrep -x fortai_server >/dev/null 2>&1 || pgrep -x llama-server >/dev/null 2>&1; then
    echo "another inference server is running; stop it before the equivalence check" >&2
    exit 2
fi

set -a
# shellcheck disable=SC1090
. <(sed -n 's/^Environment=//p' "$env_file")
set +a
export LLAMACPP_HOST=127.0.0.1
export LLAMACPP_PORT="$port"
export LLAMACPP_CONTEXT="${BENCH_CONTEXT:-$LLAMACPP_CONTEXT}"

server_pid=''
stop_server() {
    [[ -n "$server_pid" ]] || return 0
    kill -TERM "$server_pid" 2>/dev/null || true
    for _ in $(seq 1 90); do kill -0 "$server_pid" 2>/dev/null || break; sleep 1; done
    kill -KILL "$server_pid" 2>/dev/null || true
    server_pid=''
}
trap stop_server EXIT INT TERM

run_side() {
    local label="$1" mtp="$2" out="$3"
    FORTAI_NATIVE_MTP="$mtp" "$fortai_bin" >"$log_dir/mtp_equiv_${label}_${stamp}.log" 2>&1 &
    server_pid=$!
    local i
    for i in $(seq 1 900); do
        curl -fsS --max-time 5 "http://127.0.0.1:${port}/health" >/dev/null 2>&1 && break
        kill -0 "$server_pid" 2>/dev/null || { echo "$label server exited during startup" >&2; exit 1; }
        sleep 1
        [[ "$i" == 900 ]] && { echo "$label server did not become healthy" >&2; exit 1; }
    done
    env PORT="$port" OUT="$out" GEN_TOKENS="$gen_tokens" python3 - <<'PY'
import json, os, urllib.request
port, out, gen = int(os.environ["PORT"]), os.environ["OUT"], int(os.environ["GEN_TOKENS"])
PROMPTS = [
    "Explain briefly what a Fortran module is.",
    "List three properties of the heat equation.",
    "Write a short haiku about linear algebra.",
    "What is the difference between a pointer and an allocatable in Fortran?",
    "Describe in two sentences how speculative decoding works.",
]
res = []
for prompt in PROMPTS:
    payload = json.dumps({"model": "qwen",
                          "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": gen, "temperature": 0.0}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=payload, headers={"Content-Type": "application/json"})
    body = json.loads(urllib.request.urlopen(request, timeout=1800).read())
    message = body["choices"][0]["message"]
    res.append({"prompt": prompt,
                "content": message.get("content"),
                "reasoning": message.get("reasoning_content"),
                "timings": body.get("timings")})
json.dump(res, open(out, "w"), indent=2)
PY
    stop_server
}

run_side speculative 1 "$result_dir/.mtp_on_${stamp}.json"
run_side scalar      0 "$result_dir/.mtp_off_${stamp}.json"

env ON="$result_dir/.mtp_on_${stamp}.json" OFF="$result_dir/.mtp_off_${stamp}.json" \
    RESULT="$result_file" python3 - <<'PY'
import json, os, sys
on = json.load(open(os.environ["ON"]))
off = json.load(open(os.environ["OFF"]))
cases, failures = [], 0
for a, b in zip(on, off):
    same = a["content"] == b["content"] and a["reasoning"] == b["reasoning"]
    failures += not same
    draft = (a.get("timings") or {})
    cases.append({"prompt": a["prompt"], "identical": same,
                  "speculative_draft_n": draft.get("draft_n"),
                  "speculative_draft_accepted": draft.get("draft_n_accepted"),
                  "speculative_chars": len(a["content"] or ""),
                  "scalar_chars": len(b["content"] or "")})
    print(("PASS  " if same else "FAIL  ") + a["prompt"])
    if not same:
        for field in ("content", "reasoning"):
            x, y = a[field] or "", b[field] or ""
            if x == y:
                continue
            at = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]),
                      min(len(x), len(y)))
            cases[-1][f"{field}_diverges_at_char"] = at
            cases[-1][f"{field}_lengths"] = [len(x), len(y)]
            print(f"   {field}: first difference at char {at} "
                  f"(lengths {len(x)} vs {len(y)})")
            print(f"     common tail : ...{x[max(0, at - 60):at]!r}")
            print(f"     speculative : {x[at:at + 60]!r}")
            print(f"     scalar      : {y[at:at + 60]!r}")
json.dump({"cases": cases, "failures": failures}, open(os.environ["RESULT"], "w"), indent=2)
print(f"\n{len(cases) - failures}/{len(cases)} prompts identical between the "
      f"speculative and non-speculative paths")
sys.exit(1 if failures else 0)
PY
printf 'result=%s\n' "$result_file"
