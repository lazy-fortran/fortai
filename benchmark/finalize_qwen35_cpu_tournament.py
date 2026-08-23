#!/usr/bin/env python3
"""Finalize a promotion-grade CPU thread tournament from frozen summaries."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
from pathlib import Path


COMMON_KEYS = (
    "fortai_commit",
    "fortai_patch_digest",
    "fortai_tracked_tree_digest",
    "fortai_worktree_digest",
    "build_flags",
    "compiler",
    "cpu_model",
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


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def finalize(summary_paths: list[Path], oracle_path: Path) -> dict[str, object]:
    if not summary_paths:
        raise ValueError("at least one thread summary is required")
    summaries = [validate_summary(_load(path), str(path)) for path in summary_paths]
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

    oracle = _load(oracle_path)
    if oracle.get("verdict") != "PASS":
        raise ValueError("behavioral oracle did not pass")
    for key in (
        "fortai_commit",
        "fortai_patch_digest",
        "fortai_tracked_tree_digest",
        "fortai_worktree_digest",
        "build_flags",
        "fortai_executable_sha256",
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
    if int(oracle.get("omp_num_threads", 0)) != int(best_fortai["omp_num_threads"]):
        raise ValueError("behavioral oracle is not bound to the best FortAI thread")
    if int(oracle.get("top_k", 0)) < 2 or float(oracle.get("maximum_centered_logit_error", math.inf)) > float(oracle.get("tolerance", -1.0)):
        raise ValueError("behavioral oracle tolerance gate failed")

    return {
        "kind": "qwen35_cpu_thread_tournament",
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
            "model": "same",
            "model_sha256": "same",
            "token_id": "same",
            "context": "same",
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
            "omp_num_threads": 1,
            "top_k": 32,
            "maximum_centered_logit_error": 1.0e-7,
            "tolerance": 1.0e-2,
        }
        oracle_path = root / "oracle.json"
        oracle_path.write_text(json.dumps(oracle))
        result = finalize([summary_a, summary_b], oracle_path)
        assert result["winner"] == "fortai"
        assert result["best_fortai"]["omp_num_threads"] == 1

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
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.summaries or args.oracle is None or args.output is None:
        parser.error("summaries, --oracle, and --output are required")
    result = finalize([Path(path) for path in args.summaries], args.oracle)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"winner": result["winner"], "best_fortai": result["best_fortai"], "best_llama_cpp": result["best_llama_cpp"]}, sort_keys=True))


if __name__ == "__main__":
    main()
