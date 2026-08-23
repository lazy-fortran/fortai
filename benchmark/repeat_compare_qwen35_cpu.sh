#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_path="${1:-${FORTAI_MODEL:-}}"
if [[ -z "$model_path" ]]; then
    echo "usage: benchmark/repeat_compare_qwen35_cpu.sh MODEL.gguf [repeats] [token_id] [steps] [context]" >&2
    exit 2
fi
if [[ ! -f "$model_path" ]]; then
    echo "model not found: $model_path" >&2
    exit 2
fi
model_name=$(basename "$model_path")
case "$model_name" in
    Qwen3.5-0.8B-Q8_0.gguf|Qwen3.5-2B-Q8_0.gguf|Qwen3.5-4B-Q8_0.gguf) ;;
    *)
        echo "CPU repeat model scope is limited to Qwen3.5 0.8B/2B/4B Q8_0: $model_name" >&2
        exit 2
        ;;
esac

repeats="${2:-${FORTAI_BENCH_REPEATS:-5}}"
token_id="${3:-${FORTAI_TOKEN_ID:-9419}}"
steps="${4:-${FORTAI_BENCH_STEPS:-8}}"
context="${5:-${FORTAI_CONTEXT:-128}}"
port="${LLAMA_PORT:-18081}"
if [[ ! "$repeats" =~ ^[0-9]+$ || "$repeats" -lt 1 ]]; then
    echo "repeats must be a positive integer: $repeats" >&2
    exit 2
fi
result_dir="$root_dir/benchmark/results"
log_dir="$root_dir/benchmark/logs"
mkdir -p "$result_dir" "$log_dir"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
base=$(basename "$model_path" .gguf)
summary_file="$result_dir/repeat_${base}_${stamp}.json"
index_file=$(mktemp)
trap 'rm -f "$index_file"' EXIT

assert_no_server() {
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${port}$"; then
        echo "a server is still listening on port $port after run $1" >&2
        exit 1
    fi
}

for run in $(seq 1 "$repeats"); do
    log_file="$log_dir/repeat_${base}_r${run}_${stamp}.log"
    "$root_dir/benchmark/compare_qwen35_cpu_llama.sh" "$model_path" \
        "$token_id" "$steps" "$context" >"$log_file" 2>&1
    assert_no_server "$run"
    result_file=$(sed -n 's/^result=//p' "$log_file" | tail -n 1)
    if [[ -z "$result_file" || ! -f "$result_file" ]]; then
        echo "comparison run $run did not produce a result" >&2
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
    raise SystemExit("at least one comparison did not verify llama-server cleanup")
if any(int(item["fortai"].get("steps", -1)) != int(item["steps"]) for item in results):
    raise SystemExit("FortAI step count mismatch in comparison results")
if any(int(item["llama_cpp"].get("timings", {}).get("predicted_n", -1)) != int(item["steps"]) for item in results):
    raise SystemExit("llama.cpp step count mismatch in comparison results")

first = results[0]
immutable_keys = (
    "fortai_commit", "fortai_patch_digest", "fortai_tracked_tree_digest",
    "fortai_worktree_digest", "fortai_executable_sha256", "build_flags",
    "compiler", "cpu_model", "online_cpus", "omp_num_threads", "omp_proc_bind", "omp_places",
    "cuda_visible_devices",
    "model_sha256", "token_id", "steps", "context", "llama_launcher_sha256",
    "llama_executable_sha256", "llama_loaded_libraries", "llama_version",
    "measurement_conditions", "shared_service_conditions",
    "performance_gate_eligible",
)
for key in immutable_keys:
    if any(item.get(key) != first.get(key) for item in results[1:]):
        raise SystemExit(f"mixed {key} values in repeated comparison")

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

fortai = [float(item["matched_forward"]["fortai_steps_per_second"]) for item in results]
llama = [float(item["matched_forward"]["llama_cpp_steps_per_second"]) for item in results]
summary = {
    "fortai_commit": first["fortai_commit"],
    "fortai_patch_digest": first.get("fortai_patch_digest", ""),
    "fortai_tracked_tree_digest": first.get("fortai_tracked_tree_digest", ""),
    "fortai_worktree_digest": first.get("fortai_worktree_digest", ""),
    "build_flags": first.get("build_flags", ""),
    "fortai_executable_sha256": first.get("fortai_executable_sha256", ""),
    "llama_server_cleanup": "verified",
    "temporary_llama_server_cleanup": "verified",
    "measurement_conditions": first["measurement_conditions"],
    "shared_service_conditions": first["shared_service_conditions"],
    "performance_gate_eligible": first["performance_gate_eligible"],
    "compiler": first["compiler"],
    "cpu_model": first["cpu_model"],
    "online_cpus": first["online_cpus"],
    "cuda_visible_devices": first.get("cuda_visible_devices", "unset"),
    "omp_num_threads": first["omp_num_threads"],
    "omp_proc_bind": first["omp_proc_bind"],
    "omp_places": first["omp_places"],
    "model": first["model"],
    "model_sha256": first["model_sha256"],
    "token_id": first["token_id"],
    "steps": first["steps"],
    "context": first["context"],
    "repeats": int(sys.argv[3]),
    "metric": "matched_forward_steps_per_second",
    "fortai_matched_forward_steps_per_second": stats(fortai),
    "llama_cpp_matched_forward_steps_per_second": stats(llama),
    "llama_launcher_sha256": first["llama_launcher_sha256"],
    "llama_executable_sha256": first["llama_executable_sha256"],
    "llama_loaded_libraries": first["llama_loaded_libraries"],
    "llama_version": first["llama_version"],
    "result_files": paths,
}
Path(sys.argv[2]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps({
                  "fortai_matched_forward_steps_per_second":
                      summary["fortai_matched_forward_steps_per_second"],
                  "llama_cpp_matched_forward_steps_per_second":
                      summary["llama_cpp_matched_forward_steps_per_second"]},
                 sort_keys=True))
PY
printf 'summary=%s\n' "$summary_file"
