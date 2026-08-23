#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_path="${1:-${FORTAI_MODEL:-}}"
if [[ -z "$model_path" || ! -f "$model_path" ]]; then
    echo "usage: benchmark/check_qwen35_cpu_logits.sh MODEL.gguf [token_id] [steps] [context] [top_k] [tolerance]" >&2
    exit 2
fi

token_id="${2:-${FORTAI_TOKEN_ID:-9419}}"
steps="${3:-${FORTAI_LOGIT_STEPS:-1}}"
context="${4:-${FORTAI_CONTEXT:-128}}"
top_k="${5:-${FORTAI_LOGIT_TOP_K:-32}}"
tolerance="${6:-${FORTAI_LOGIT_TOLERANCE:-1.0e-2}}"
correctness_flags="${FORTAI_CORRECTNESS_FLAGS:--O2 -fopenmp -fno-fast-math -ffp-contract=off}"
log_file=$(mktemp)
trap 'rm -f "$log_file"' EXIT

if [[ ! "$top_k" =~ ^[1-9][0-9]*$ || "$top_k" == 1 ]]; then
    echo "top_k must be an integer of at least 2: $top_k" >&2
    exit 2
fi

FORTAI_NATIVE_FLAGS="$correctness_flags" FORTAI_TRACE_TOP_LOGITS="$top_k" \
FORTAI_LLAMA_ORACLE_TOP_K="$top_k" \
    "$root_dir/benchmark/compare_qwen35_cpu_llama.sh" "$model_path" \
    "$token_id" "$steps" "$context" >"$log_file" 2>&1
result_file=$(sed -n 's/^result=//p' "$log_file" | tail -n 1)
if [[ -z "$result_file" || ! -f "$result_file" ]]; then
    echo 'logit comparison did not produce a result' >&2
    exit 1
fi

python3 - "$result_file" "$top_k" "$tolerance" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
top_k = int(sys.argv[2])
tolerance = float(sys.argv[3])
if not math.isfinite(tolerance) or tolerance < 0.0:
    raise SystemExit("tolerance must be a finite nonnegative number")
if result.get("llama_oracle_top_k") != top_k:
    raise SystemExit("result did not record the requested llama.cpp top-k oracle")

pattern = re.compile(
    r"top_logit\[(\d+),(\d+)\]=(\d+),\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)"
)
fortai = {}
for line in Path(result["fortai_log"]).read_text().splitlines():
    match = pattern.fullmatch(line)
    if match:
        step, rank, token, delta = match.groups()
        fortai[(int(step), int(rank))] = (int(token), float(delta))

oracle_steps = result["llama_cpp"].get("completion_probabilities", [])
if len(oracle_steps) != int(result["steps"]):
    raise SystemExit("llama.cpp did not return probabilities for every step")

maximum_error = 0.0
for step, oracle_step in enumerate(oracle_steps):
    oracle_top = oracle_step.get("top_logprobs", [])
    if len(oracle_top) != top_k:
        raise SystemExit(
            f"llama.cpp returned {len(oracle_top)} top logits at step {step}; expected {top_k}"
        )
    oracle_by_token = {int(item["id"]): float(item["logprob"]) for item in oracle_top}
    if len(oracle_by_token) != top_k:
        raise SystemExit(f"llama.cpp returned duplicate token IDs at step {step}")
    if any(not math.isfinite(value) for value in oracle_by_token.values()):
        raise SystemExit(f"llama.cpp returned a non-finite log probability at step {step}")
    oracle_max = max(oracle_by_token.values())
    candidate = [fortai.get((step, rank)) for rank in range(1, top_k + 1)]
    if any(item is None for item in candidate):
        raise SystemExit(f"FortAI did not emit {top_k} top logits at step {step}")
    candidate_by_token = {token: delta for token, delta in candidate}
    if len(candidate_by_token) != top_k:
        raise SystemExit(f"FortAI emitted duplicate token IDs at step {step}")
    if any(not math.isfinite(value) for value in candidate_by_token.values()):
        raise SystemExit(f"FortAI emitted a non-finite centered logit at step {step}")
    if candidate[0][0] != int(oracle_step["id"]):
        raise SystemExit(
            f"top token mismatch at step {step}: FortAI={candidate[0][0]} "
            f"llama.cpp={oracle_step['id']}"
        )
    if candidate_by_token.keys() != oracle_by_token.keys():
        missing = sorted(oracle_by_token.keys() - candidate_by_token.keys())
        extra = sorted(candidate_by_token.keys() - oracle_by_token.keys())
        raise SystemExit(
            f"top-{top_k} token set mismatch at step {step}: missing={missing} extra={extra}"
        )
    for token, candidate_delta in candidate_by_token.items():
        oracle_delta = oracle_by_token[token] - oracle_max
        error = abs(candidate_delta - oracle_delta)
        maximum_error = max(maximum_error, error)
        if error > tolerance:
            raise SystemExit(
                f"centered logit mismatch at step {step}, token {token}: "
                f"FortAI={candidate_delta:.9g} llama.cpp={oracle_delta:.9g} "
                f"abs_error={error:.9g} tolerance={tolerance:.9g}"
            )

print(json.dumps({
    "verdict": "PASS",
    "kind": "llama_cpp_centered_top_k",
    "fortai_commit": result["fortai_commit"],
    "fortai_patch_digest": result["fortai_patch_digest"],
    "fortai_tracked_tree_digest": result["fortai_tracked_tree_digest"],
    "fortai_worktree_digest": result["fortai_worktree_digest"],
    "fortai_executable_sha256": result["fortai_executable_sha256"],
    "build_flags": result["build_flags"],
    "compiler": result["compiler"],
    "omp_num_threads": result["omp_num_threads"],
    "omp_proc_bind": result["omp_proc_bind"],
    "omp_places": result["omp_places"],
    "cuda_visible_devices": result["cuda_visible_devices"],
    "measurement_conditions": result["measurement_conditions"],
    "performance_gate_eligible": result["performance_gate_eligible"],
    "shared_service_conditions": result["shared_service_conditions"],
    "llama_server_cleanup": result["llama_server_cleanup"],
    "temporary_llama_server_cleanup": result["temporary_llama_server_cleanup"],
    "model": result["model"],
    "model_sha256": result["model_sha256"],
    "token_id": int(result["token_id"]),
    "context": int(result["context"]),
    "cpu_model": result["cpu_model"],
    "online_cpus": result["online_cpus"],
    "protected_gpu_server_independent": result["protected_gpu_server_independent"],
    "protected_gpu_server_pid": result["protected_gpu_server_pid"],
    "persistent_openmp": result["persistent_openmp"],
    "llama_launcher_sha256": result["llama_launcher_sha256"],
    "llama_loaded_libraries": result["llama_loaded_libraries"],
    "llama_version": result["llama_version"],
    "llama_executable_sha256": result["llama_executable_sha256"],
    "steps": int(result["steps"]),
    "top_k": top_k,
    "maximum_centered_logit_error": maximum_error,
    "tolerance": tolerance,
}, sort_keys=True))
PY
