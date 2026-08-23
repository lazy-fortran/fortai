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
matched = result["matched_forward"]
fortai = matched["fortai_steps_per_second"]
llama = matched["llama_cpp_steps_per_second"]
print(f"threads={result['omp_num_threads']} fortai_matched_step_s={fortai} "
      f"llama_matched_step_s={llama}")
PY
done

python3 - "$index_file" "$summary_file" <<'PY'
import json
import sys
from pathlib import Path

paths = [line.strip() for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
results = [json.loads(Path(path).read_text()) for path in paths]
first = results[0]
for key in ("fortai_commit", "fortai_patch_digest", "fortai_tracked_tree_digest",
            "fortai_worktree_digest", "fortai_executable_sha256", "build_flags",
            "compiler", "cpu_model", "model_sha256", "token_id", "steps",
            "context", "omp_proc_bind", "omp_places", "llama_launcher_sha256",
            "llama_executable_sha256", "llama_loaded_libraries", "llama_version",
            "measurement_conditions", "performance_gate_eligible"):
    if any(item.get(key) != first.get(key) for item in results[1:]):
        raise SystemExit(f"mixed {key} values in thread sweep")
best_fortai = max(results,
    key=lambda item: float(item["matched_forward"]["fortai_steps_per_second"]))
best_llama = max(results,
    key=lambda item: float(item["matched_forward"]["llama_cpp_steps_per_second"]))
summary = {
    "results": results,
    "metric": "single_run_matched_forward_steps_per_second",
    "screening_only": True,
    "measurement_conditions": first["measurement_conditions"],
    "performance_gate_eligible": first["performance_gate_eligible"],
    "best_fortai": {
        "omp_num_threads": best_fortai["omp_num_threads"],
        "matched_forward_steps_per_second":
            float(best_fortai["matched_forward"]["fortai_steps_per_second"]),
    },
    "best_llama_cpp": {
        "omp_num_threads": best_llama["omp_num_threads"],
        "matched_forward_steps_per_second":
            float(best_llama["matched_forward"]["llama_cpp_steps_per_second"]),
    },
}
Path(sys.argv[2]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary["best_fortai"], sort_keys=True))
print(json.dumps(summary["best_llama_cpp"], sort_keys=True))
PY
printf 'summary=%s\n' "$summary_file"
