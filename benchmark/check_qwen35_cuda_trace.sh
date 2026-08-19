#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_path="${1:-${FORTAI_MODEL:-}}"
if [[ -z "$model_path" || ! -f "$model_path" ]]; then
    echo "usage: benchmark/check_qwen35_cuda_trace.sh MODEL.gguf [token_id] [steps] [context] [device]" >&2
    exit 2
fi

token_id="${2:-${FORTAI_TOKEN_ID:-9419}}"
steps="${3:-${FORTAI_TRACE_STEPS:-8}}"
context="${4:-${FORTAI_CONTEXT:-128}}"
device="${5:-${CUDA_DEVICE:-0}}"
log_file=$(mktemp)
trap 'rm -f "$log_file"' EXIT

FORTAI_TRACE_TOKENS=1 FORTAI_LLAMA_ORACLE_PROBS=1 CUDA_DEVICE="$device" \
    "$root_dir/benchmark/compare_qwen35_cuda_llama.sh" "$model_path" \
    "$token_id" "$steps" "$context" "$device" >"$log_file" 2>&1
result_file=$(sed -n 's/^result=//p' "$log_file" | tail -n 1)
if [[ -z "$result_file" || ! -f "$result_file" ]]; then
    echo 'CUDA trace comparison did not produce a result' >&2
    exit 1
fi

python3 - "$result_file" <<'PY'
import json
import re
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
fortai = {}
for line in Path(result["fortai_log"]).read_text().splitlines():
    match = re.fullmatch(r"token\[(\d+)\]=(\d+)", line)
    if match:
        fortai[int(match.group(1))] = int(match.group(2))
llama = [item["id"] for item in result["llama_cpp"].get("completion_probabilities", [])]
fortai_ids = [fortai[index] for index in sorted(fortai)]
print(json.dumps({"fortai": fortai_ids, "llama_cpp": llama}, sort_keys=True))
if fortai_ids != llama:
    for index, pair in enumerate(zip(fortai_ids, llama)):
        if pair[0] != pair[1]:
            print(f"first_token_mismatch={index} fortai={pair[0]} llama_cpp={pair[1]}", file=sys.stderr)
            break
    else:
        print(f"trace_length_mismatch fortai={len(fortai_ids)} llama_cpp={len(llama)}", file=sys.stderr)
    raise SystemExit(1)
PY
