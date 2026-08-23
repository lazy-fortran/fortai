#!/usr/bin/env python3
"""Finalize a promotion-grade CPU thread tournament from frozen summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import tempfile
from pathlib import Path


ALLOWED_MODELS = frozenset(
    {
        "Qwen3.5-0.8B-Q8_0.gguf",
        "Qwen3.5-2B-Q8_0.gguf",
        "Qwen3.5-4B-Q8_0.gguf",
    }
)
DEFAULT_ORACLE_STEPS = 8
DEFAULT_ORACLE_TOP_K = 32
DEFAULT_ORACLE_TOLERANCE = 1.0e-2

COMMON_KEYS = (
    "fortai_commit",
    "fortai_patch_digest",
    "fortai_tracked_tree_digest",
    "fortai_worktree_digest",
    "build_flags",
    "compiler",
    "cpu_model",
    "cuda_visible_devices",
    "fortai_executable_sha256",
    "model",
    "model_sha256",
    "token_id",
    "steps",
    "context",
    "omp_proc_bind",
    "omp_places",
    "llama_launcher_sha256",
    "llama_executable_sha256",
    "llama_loaded_libraries",
    "llama_version",
    "measurement_conditions",
    "shared_service_conditions",
    "performance_gate_eligible",
    "repeats",
    "metric",
    "llama_server_cleanup",
    "temporary_llama_server_cleanup",
)


def _finite_positive(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def _stats(values: list[float]) -> dict[str, object]:
    return {
        "n": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def _close(actual: object, expected: object) -> bool:
    if actual is None or expected is None:
        return actual is expected
    return math.isclose(float(actual), float(expected), rel_tol=1.0e-12, abs_tol=1.0e-12)


def validate_summary(summary: dict[str, object], source: str) -> dict[str, object]:
    model = summary.get("model")
    if not isinstance(model, str) or Path(model).name not in ALLOWED_MODELS:
        raise ValueError(
            f"{source}: tournament model scope is limited to "
            "Qwen3.5 0.8B/2B/4B Q8_0"
        )
    if summary.get("cuda_visible_devices") != "":
        raise ValueError(f"{source}: CPU tournament must hide all CUDA devices")
    if summary.get("metric") != "matched_forward_steps_per_second":
        raise ValueError(f"{source}: unsupported tournament metric")
    if int(summary.get("steps", 0)) <= 0:
        raise ValueError(f"{source}: timing step count must be positive")
    if int(summary.get("token_id", 0)) <= 0:
        raise ValueError(f"{source}: token ID must be positive")
    if int(summary.get("context", 0)) <= 0:
        raise ValueError(f"{source}: context size must be positive")
    repeats = int(summary.get("repeats", 0))
    if repeats < 5:
        raise ValueError(f"{source}: at least five repeats are required")
    if summary.get("measurement_conditions") != "isolated":
        raise ValueError(f"{source}: tournament timings must be isolated")
    if summary.get("shared_service_conditions") is not False:
        raise ValueError(f"{source}: shared-service timing is not eligible")
    if summary.get("performance_gate_eligible") is not True:
        raise ValueError(f"{source}: performance gate is not eligible")
    if summary.get("llama_server_cleanup") != "verified":
        raise ValueError(f"{source}: llama-server cleanup was not verified")
    threads = int(summary.get("omp_num_threads", 0))
    if threads <= 0:
        raise ValueError(f"{source}: invalid OpenMP thread count")

    for key in ("fortai_matched_forward_steps_per_second", "llama_cpp_matched_forward_steps_per_second"):
        recorded = summary.get(key)
        if not isinstance(recorded, dict):
            raise ValueError(f"{source}: missing {key} statistics")
        values = [_finite_positive(value, f"{source}:{key}") for value in recorded.get("values", [])]
        if len(values) != repeats or int(recorded.get("n", -1)) != repeats:
            raise ValueError(f"{source}: {key} repeat count mismatch")
        expected = _stats(values)
        for field in ("median", "mean", "stdev", "min", "max"):
            if not _close(recorded.get(field), expected[field]):
                raise ValueError(f"{source}: tampered {key}.{field}")
    return summary


def _load_frozen(path: Path) -> tuple[dict[str, object], str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize(
    summary_paths: list[Path],
    oracle_path: Path,
    oracle_steps: int = DEFAULT_ORACLE_STEPS,
    oracle_top_k: int = DEFAULT_ORACLE_TOP_K,
    oracle_tolerance: float = DEFAULT_ORACLE_TOLERANCE,
) -> dict[str, object]:
    if len(summary_paths) < 2:
        raise ValueError("at least two thread summaries are required")
    if oracle_steps <= 0:
        raise ValueError("oracle steps must be positive")
    if oracle_top_k < 2:
        raise ValueError("oracle top-k must be at least two")
    if not math.isfinite(oracle_tolerance) or oracle_tolerance < 0.0:
        raise ValueError("oracle tolerance must be finite and nonnegative")
    frozen_summaries = [(path, *_load_frozen(path)) for path in summary_paths]
    summaries = [validate_summary(payload, str(path)) for path, payload, _ in frozen_summaries]
    summary_digests = {str(path): digest for path, _, digest in frozen_summaries}
    first = summaries[0]
    for key in COMMON_KEYS:
        if key not in first:
            raise ValueError(f"missing common provenance field: {key}")
        if any(summary.get(key) != first.get(key) for summary in summaries[1:]):
            raise ValueError(f"mixed {key} values across thread summaries")
    threads = [int(summary["omp_num_threads"]) for summary in summaries]
    if len(set(threads)) != len(threads):
        raise ValueError("thread summaries must have unique OpenMP thread counts")

    best_fortai = max(
        summaries,
        key=lambda item: float(item["fortai_matched_forward_steps_per_second"]["median"]),
    )
    best_llama = max(
        summaries,
        key=lambda item: float(item["llama_cpp_matched_forward_steps_per_second"]["median"]),
    )
    fortai_median = float(best_fortai["fortai_matched_forward_steps_per_second"]["median"])
    llama_median = float(best_llama["llama_cpp_matched_forward_steps_per_second"]["median"])
    winner = "fortai" if fortai_median >= llama_median else "llama_cpp"

    oracle, oracle_digest = _load_frozen(oracle_path)
    if oracle.get("verdict") != "PASS":
        raise ValueError("behavioral oracle did not pass")
    for key in (
        "fortai_commit",
        "fortai_patch_digest",
        "fortai_tracked_tree_digest",
        "fortai_worktree_digest",
        "build_flags",
        "fortai_executable_sha256",
        "cuda_visible_devices",
        "model",
        "model_sha256",
        "token_id",
        "context",
        "compiler",
        "cpu_model",
        "omp_proc_bind",
        "omp_places",
        "llama_launcher_sha256",
        "llama_executable_sha256",
        "llama_loaded_libraries",
        "llama_version",
        "temporary_llama_server_cleanup",
    ):
        if oracle.get(key) != first.get(key):
            raise ValueError(f"oracle does not match tournament {key}")
    if oracle.get("measurement_conditions") != "isolated":
        raise ValueError("behavioral oracle is not isolated")
    if oracle.get("shared_service_conditions") is not False:
        raise ValueError("behavioral oracle used shared-service conditions")
    if oracle.get("performance_gate_eligible") is not True:
        raise ValueError("behavioral oracle is not performance-gate eligible")
    if oracle.get("llama_server_cleanup") != "verified":
        raise ValueError("behavioral oracle did not verify cleanup")
    if oracle.get("temporary_llama_server_cleanup") != "verified":
        raise ValueError("behavioral oracle did not verify temporary cleanup")
    if int(oracle.get("omp_num_threads", 0)) != int(best_fortai["omp_num_threads"]):
        raise ValueError("behavioral oracle is not bound to the best FortAI thread")
    if int(oracle.get("steps", 0)) != oracle_steps:
        raise ValueError("behavioral oracle step count does not match tournament contract")
    if int(oracle.get("top_k", 0)) != oracle_top_k:
        raise ValueError("behavioral oracle top-k does not match tournament contract")
    if not _close(oracle.get("tolerance"), oracle_tolerance):
        raise ValueError("behavioral oracle tolerance does not match tournament contract")
    if float(oracle.get("maximum_centered_logit_error", math.inf)) > oracle_tolerance:
        raise ValueError("behavioral oracle tolerance gate failed")

    return {
        "kind": "qwen35_cpu_thread_tournament",
        "lifecycle": {
            "leaf_id": "FAI-CPU-003",
            "leaf_status": "IN_PROGRESS",
            "claim_id": "FAI-CPU-003",
            "claim_status": "OPEN",
            "parent_id": "FAI-CPU",
            "parent_status": "OPEN",
            "evidence_gate_verdict": "PASS",
            "review_verdict": "NEEDS REVIEW",
        },
        "fortai_commit": first["fortai_commit"],
        "fortai_patch_digest": first["fortai_patch_digest"],
        "fortai_tracked_tree_digest": first["fortai_tracked_tree_digest"],
        "fortai_worktree_digest": first["fortai_worktree_digest"],
        "build_flags": first["build_flags"],
        "compiler": first["compiler"],
        "cpu_model": first["cpu_model"],
        "model": first["model"],
        "model_sha256": first["model_sha256"],
        "token_id": first["token_id"],
        "steps": first["steps"],
        "context": first["context"],
        "repeats_per_thread": int(first["repeats"]),
        "threads": sorted(threads),
        "metric": "median_matched_forward_steps_per_second",
        "measurement_conditions": "isolated",
        "performance_gate_eligible": True,
        "cuda_visible_devices": first["cuda_visible_devices"],
        "oracle_contract": {
            "steps": oracle_steps,
            "top_k": oracle_top_k,
            "tolerance": oracle_tolerance,
        },
        "input_artifacts": {
            "thread_summaries": [
                {"path": str(path), "sha256": summary_digests[str(path)]}
                for path in summary_paths
            ],
            "oracle": {"path": str(oracle_path), "sha256": oracle_digest},
        },
        "best_fortai": {
            "omp_num_threads": int(best_fortai["omp_num_threads"]),
            "median_steps_per_second": fortai_median,
            "values": best_fortai["fortai_matched_forward_steps_per_second"]["values"],
        },
        "best_llama_cpp": {
            "omp_num_threads": int(best_llama["omp_num_threads"]),
            "median_steps_per_second": llama_median,
            "values": best_llama["llama_cpp_matched_forward_steps_per_second"]["values"],
        },
        "winner": winner,
        "oracle": oracle,
        "thread_summaries": [str(path) for path in summary_paths],
    }


def _fixture(thread: int = 2) -> dict[str, object]:
    values = [10.0, 11.0, 12.0, 13.0, 14.0]
    summary = {
        key: "same" for key in COMMON_KEYS
    }
    summary.update(
        {
            "repeats": 5,
            "model": "/fixtures/Qwen3.5-0.8B-Q8_0.gguf",
            "cuda_visible_devices": "",
            "steps": 64,
            "token_id": 9419,
            "context": 128,
            "omp_num_threads": thread,
            "measurement_conditions": "isolated",
            "shared_service_conditions": False,
            "performance_gate_eligible": True,
            "metric": "matched_forward_steps_per_second",
            "llama_server_cleanup": "verified",
            "temporary_llama_server_cleanup": "verified",
            "fortai_matched_forward_steps_per_second": _stats(values),
            "llama_cpp_matched_forward_steps_per_second": _stats([9.0, 10.0, 11.0, 12.0, 13.0]),
        }
    )
    return summary


def self_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first = _fixture(1)
        second = _fixture(2)
        summary_a = root / "summary_a.json"
        summary_b = root / "summary_b.json"
        summary_a.write_text(json.dumps(first))
        summary_b.write_text(json.dumps(second))
        oracle = {
            "verdict": "PASS",
            "fortai_commit": "same",
            "fortai_patch_digest": "same",
            "fortai_tracked_tree_digest": "same",
            "fortai_worktree_digest": "same",
            "build_flags": "same",
            "fortai_executable_sha256": "same",
            "cuda_visible_devices": "",
            "model": "/fixtures/Qwen3.5-0.8B-Q8_0.gguf",
            "model_sha256": "same",
            "token_id": 9419,
            "context": 128,
            "compiler": "same",
            "cpu_model": "same",
            "omp_proc_bind": "same",
            "omp_places": "same",
            "llama_launcher_sha256": "same",
            "llama_executable_sha256": "same",
            "llama_loaded_libraries": "same",
            "llama_version": "same",
            "measurement_conditions": "isolated",
            "shared_service_conditions": False,
            "performance_gate_eligible": True,
            "llama_server_cleanup": "verified",
            "temporary_llama_server_cleanup": "verified",
            "omp_num_threads": 1,
            "steps": 8,
            "top_k": 32,
            "maximum_centered_logit_error": 1.0e-7,
            "tolerance": 1.0e-2,
        }
        oracle_path = root / "oracle.json"
        oracle_path.write_text(json.dumps(oracle))
        result = finalize([summary_a, summary_b], oracle_path)
        assert result["winner"] == "fortai"
        assert result["best_fortai"]["omp_num_threads"] == 1
        assert result["lifecycle"] == {
            "leaf_id": "FAI-CPU-003",
            "leaf_status": "IN_PROGRESS",
            "claim_id": "FAI-CPU-003",
            "claim_status": "OPEN",
            "parent_id": "FAI-CPU",
            "parent_status": "OPEN",
            "evidence_gate_verdict": "PASS",
            "review_verdict": "NEEDS REVIEW",
        }
        assert result["oracle_contract"] == {
            "steps": 8,
            "top_k": 32,
            "tolerance": 1.0e-2,
        }
        assert result["input_artifacts"] == {
            "thread_summaries": [
                {"path": str(summary_a), "sha256": _sha256(summary_a)},
                {"path": str(summary_b), "sha256": _sha256(summary_b)},
            ],
            "oracle": {"path": str(oracle_path), "sha256": _sha256(oracle_path)},
        }
        try:
            finalize([summary_a], oracle_path)
        except ValueError:
            pass
        else:
            raise AssertionError("single-thread candidate was accepted")

        mixed_repeats = _fixture(2)
        mixed_repeats["repeats"] = 6
        mixed_repeats["fortai_matched_forward_steps_per_second"] = _stats(
            [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        )
        mixed_repeats["llama_cpp_matched_forward_steps_per_second"] = _stats(
            [9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
        )
        mixed_path = root / "mixed_repeats.json"
        mixed_path.write_text(json.dumps(mixed_repeats))
        try:
            finalize([summary_a, mixed_path], oracle_path)
        except ValueError:
            pass
        else:
            raise AssertionError("mixed repeat counts were accepted")

        def rejects(mutator, message):
            changed = _fixture(1)
            mutator(changed)
            path = root / f"{message}.json"
            path.write_text(json.dumps(changed))
            try:
                finalize([path], oracle_path)
            except ValueError:
                return
            raise AssertionError(f"fixture was accepted: {message}")

        rejects(lambda item: item.update({"repeats": 4}), "short")
        rejects(
            lambda item: item.update({"model": "/fixtures/Qwen3.8-27B-Q4_K_XL.gguf"}),
            "out_of_scope_model",
        )
        rejects(lambda item: item.update({"cuda_visible_devices": "0"}), "cuda_visible")
        rejects(lambda item: item.update({"metric": "wall_time"}), "metric")
        rejects(lambda item: item.update({"steps": 0}), "steps")
        rejects(lambda item: item.update({"token_id": 0}), "token")
        rejects(lambda item: item.update({"context": 0}), "context")
        rejects(lambda item: item.update({"shared_service_conditions": True}), "shared")
        rejects(lambda item: item["fortai_matched_forward_steps_per_second"].update({"median": 999.0}), "tampered")
        rejects(lambda item: item.update({"fortai_commit": "other"}), "mixed")
        bad_oracle = root / "bad_oracle.json"
        bad_oracle.write_text(json.dumps({**oracle, "omp_num_threads": 2}))
        try:
            finalize([summary_a], bad_oracle)
        except ValueError:
            pass
        else:
            raise AssertionError("mismatched oracle was accepted")
        weak_oracle = root / "weak_oracle.json"
        weak_oracle.write_text(json.dumps({**oracle, "top_k": 2}))
        try:
            finalize([summary_a], weak_oracle)
        except ValueError:
            pass
        else:
            raise AssertionError("weaker oracle contract was accepted")
        incomplete_cleanup_oracle = root / "incomplete_cleanup_oracle.json"
        incomplete_cleanup_oracle.write_text(
            json.dumps({**oracle, "temporary_llama_server_cleanup": "unverified"})
        )
        try:
            finalize([summary_a], incomplete_cleanup_oracle)
        except ValueError:
            pass
        else:
            raise AssertionError("incomplete oracle cleanup was accepted")
        runtime_oracle = root / "runtime_oracle.json"
        runtime_oracle.write_text(json.dumps({**oracle, "compiler": "other"}))
        try:
            finalize([summary_a], runtime_oracle)
        except ValueError:
            pass
        else:
            raise AssertionError("mismatched oracle runtime was accepted")
    print("CPU tournament finalizer self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="*")
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--oracle-steps", type=int, default=DEFAULT_ORACLE_STEPS)
    parser.add_argument("--oracle-top-k", type=int, default=DEFAULT_ORACLE_TOP_K)
    parser.add_argument("--oracle-tolerance", type=float, default=DEFAULT_ORACLE_TOLERANCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.summaries or args.oracle is None or args.output is None:
        parser.error("summaries, --oracle, and --output are required")
    result = finalize(
        [Path(path) for path in args.summaries],
        args.oracle,
        args.oracle_steps,
        args.oracle_top_k,
        args.oracle_tolerance,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"winner": result["winner"], "best_fortai": result["best_fortai"], "best_llama_cpp": result["best_llama_cpp"]}, sort_keys=True))


if __name__ == "__main__":
    main()
