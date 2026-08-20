#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_path="${1:-${FORTAI_MODEL:-}}"
if [[ -z "$model_path" ]]; then
    echo "usage: benchmark/repeat_compare_qwen35_cuda.sh MODEL.gguf [repeats] [token_id] [steps] [context] [device]" >&2
    exit 2
fi
if [[ ! -f "$model_path" ]]; then
    echo "model not found: $model_path" >&2
    exit 2
fi

repeats="${2:-${FORTAI_BENCH_REPEATS:-5}}"
token_id="${3:-${FORTAI_TOKEN_ID:-9419}}"
steps="${4:-${FORTAI_BENCH_STEPS:-64}}"
context="${5:-${FORTAI_CONTEXT:-128}}"
device="${6:-${CUDA_DEVICE:-0}}"
port="${LLAMA_PORT:-18092}"
if [[ ! "$repeats" =~ ^[0-9]+$ || "$repeats" -lt 1 ]]; then
    echo "repeats must be a positive integer: $repeats" >&2
    exit 2
fi

result_dir="$root_dir/benchmark/results"
log_dir="$root_dir/benchmark/logs"
mkdir -p "$result_dir" "$log_dir"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
base=$(basename "$model_path" .gguf)
summary_file="$result_dir/repeat_cuda_${base}_d${device}_${stamp}.json"
index_file=$(mktemp)
trap 'rm -f "$index_file"' EXIT

for run in $(seq 1 "$repeats"); do
    log_file="$log_dir/repeat_cuda_${base}_d${device}_r${run}_${stamp}.log"
    CUDA_DEVICE="$device" LLAMA_PORT="$port" \
        "$root_dir/benchmark/compare_qwen35_cuda_llama.sh" "$model_path" \
        "$token_id" "$steps" "$context" "$device" >"$log_file" 2>&1
    if pgrep -x llama-server >/dev/null 2>&1; then
        echo "llama-server remains after CUDA comparison run $run" >&2
        exit 1
    fi
    result_file=$(sed -n 's/^result=//p' "$log_file" | tail -n 1)
    if [[ -z "$result_file" || ! -f "$result_file" ]]; then
        echo "CUDA comparison run $run did not produce a result" >&2
        exit 1
    fi
    printf '%s\n' "$result_file" >>"$index_file"
    echo "run $run/$repeats done: $result_file"
done

python3 - "$index_file" "$summary_file" "$repeats" <<'PY'
import json
import statistics
import sys
from pathlib import Path

paths = [line.strip() for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
results = [json.loads(Path(path).read_text()) for path in paths]
if any(item.get("llama_server_cleanup") != "verified" for item in results):
    raise SystemExit("at least one CUDA comparison did not verify llama-server cleanup")
if any(int(item["fortai"].get("steps", -1)) != int(item["steps"]) for item in results):
    raise SystemExit("FortAI CUDA step count mismatch")
if any(int(item["llama_cpp"].get("timings", {}).get("predicted_n", -1)) != int(item["steps"]) for item in results):
    raise SystemExit("llama.cpp CUDA step count mismatch")

def stats(values):
    return {
        "n": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
        "values": values,
    }

fortai = [float(item["fortai"]["tokens_per_second"]) for item in results]
llama = [float(item["llama_cpp"]["timings"]["predicted_per_second"]) for item in results]
ratios = [a / b for a, b in zip(fortai, llama)]
first = results[0]
summary = {
    "fortai_commit": first["fortai_commit"],
    "fortai_patch_digest": first.get("fortai_patch_digest", ""),
    "fortai_tracked_tree_digest": first.get("fortai_tracked_tree_digest", ""),
    "fortai_worktree_digest": first.get("fortai_worktree_digest", ""),
    "cuda_device": first["cuda_device"],
    "model": first["model"],
    "model_sha256": first["model_sha256"],
    "token_id": first["token_id"],
    "steps": first["steps"],
    "context": first["context"],
    "omp_num_threads": first["omp_num_threads"],
    "repeats": int(sys.argv[3]),
    "cuda_graph_enabled": first["fortai"].get("cuda_graph_enabled", "unknown"),
    "fortai_tokens_per_second": stats(fortai),
    "llama_cpp_tokens_per_second": stats(llama),
    "fortai_over_llama": stats(ratios),
    "production_gate": first.get("production_gate", "unknown"),
    "result_files": paths,
}
Path(sys.argv[2]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps({"fortai_tokens_per_second": summary["fortai_tokens_per_second"],
                  "llama_cpp_tokens_per_second": summary["llama_cpp_tokens_per_second"],
                  "fortai_over_llama": summary["fortai_over_llama"]}, sort_keys=True))
PY
printf 'summary=%s\n' "$summary_file"
