#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_path="${1:-${FORTAI_MODEL:-}}"
if [[ -z "$model_path" ]]; then
    echo "usage: benchmark/tune_qwen35_cpu.sh MODEL.gguf [token_id] [steps] [context]" >&2
    exit 2
fi
if [[ ! -f "$model_path" ]]; then
    echo "model not found: $model_path" >&2
    exit 2
fi

token_id="${2:-${FORTAI_TOKEN_ID:-9419}}"
steps="${3:-${FORTAI_BENCH_STEPS:-8}}"
context="${4:-${FORTAI_CONTEXT:-128}}"
thread_list="${FORTAI_THREAD_LIST:-1 2 4 8 16 32}"
result_dir="$root_dir/benchmark/results"
log_dir="$root_dir/benchmark/logs"
mkdir -p "$result_dir" "$log_dir"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
base=$(basename "$model_path" .gguf)
summary_file="$result_dir/tune_${base}_${stamp}.json"
index_file=$(mktemp)
trap 'rm -f "$index_file"' EXIT

for threads in $thread_list; do
    log_file="$log_dir/tune_${base}_${threads}_${stamp}.log"
    OMP_NUM_THREADS="$threads" OMP_PROC_BIND="${OMP_PROC_BIND:-spread}" \
        OMP_PLACES="${OMP_PLACES:-cores}" \
        "$root_dir/benchmark/compare_qwen35_cpu_llama.sh" "$model_path" "$token_id" \
        "$steps" "$context" >"$log_file" 2>&1
    result_file=$(sed -n 's/^result=//p' "$log_file" | tail -n 1)
    if [[ -z "$result_file" || ! -f "$result_file" ]]; then
        echo "comparison did not produce a result for OMP_NUM_THREADS=$threads" >&2
        exit 1
    fi
    printf '%s\n' "$result_file" >>"$index_file"
    python3 - "$result_file" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
fortai = result["fortai"].get("tokens_per_second", "?")
llama = result["llama_cpp"].get("timings", {}).get("predicted_per_second", "?")
print(f"threads={result['omp_num_threads']} fortai_tok_s={fortai} llama_tok_s={llama}")
PY
done

python3 - "$index_file" "$summary_file" <<'PY'
import json
import sys
from pathlib import Path

paths = [line.strip() for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
results = [json.loads(Path(path).read_text()) for path in paths]
best_fortai = max(results, key=lambda item: float(item["fortai"]["tokens_per_second"]))
best_llama = max(results, key=lambda item: float(item["llama_cpp"]["timings"]["predicted_per_second"]))
summary = {
    "results": results,
    "best_fortai": {
        "threads": best_fortai["omp_num_threads"],
        "tokens_per_second": float(best_fortai["fortai"]["tokens_per_second"]),
    },
    "best_llama_cpp": {
        "threads": best_llama["omp_num_threads"],
        "tokens_per_second": float(best_llama["llama_cpp"]["timings"]["predicted_per_second"]),
    },
}
Path(sys.argv[2]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary["best_fortai"], sort_keys=True))
print(json.dumps(summary["best_llama_cpp"], sort_keys=True))
PY
printf 'summary=%s\n' "$summary_file"
