#!/usr/bin/env python3
"""Validate and add matched model-evaluation metrics to a CPU result."""

import argparse
import json
import math
from pathlib import Path


def positive_number(value, label):
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def matched_forward_metrics(result):
    steps = int(result["steps"])
    if steps <= 0:
        raise ValueError("steps must be positive")

    fortai = result["fortai"]
    if int(fortai.get("steps", -1)) != steps:
        raise ValueError("FortAI step count does not match the requested count")
    fortai_seconds = positive_number(
        fortai["forward_seconds"], "FortAI forward_seconds"
    )

    timings = result["llama_cpp"]["timings"]
    prompt_n = int(timings.get("prompt_n", -1))
    cache_n = int(timings.get("cache_n", -1))
    predicted_n = int(timings.get("predicted_n", -1))
    if prompt_n != 1:
        raise ValueError(f"llama.cpp prompt_n must be 1, got {prompt_n}")
    if cache_n != 0:
        raise ValueError(f"llama.cpp cache_n must be 0, got {cache_n}")
    if predicted_n != steps:
        raise ValueError("llama.cpp predicted_n does not match the requested count")

    prompt_ms = positive_number(timings["prompt_ms"], "llama.cpp prompt_ms")
    predicted_ms = float(timings["predicted_ms"])
    if not math.isfinite(predicted_ms) or predicted_ms < 0.0:
        raise ValueError("llama.cpp predicted_ms must be finite and nonnegative")
    # FortAI's comparison runner evaluates the one-token prompt before its
    # timed loop (FORTAI_EXCLUDE_PROMPT=1), matching llama-server's
    # predicted_ms generation timing.  Keep prompt_ms as a provenance check,
    # but do not charge it to either side of the generation comparison.
    llama_seconds = predicted_ms / 1000.0
    if llama_seconds <= 0.0:
        raise ValueError("llama.cpp matched evaluation time must be positive")

    return {
        "count": steps,
        "scope": "generated_model_forwards_after_prompt",
        "fortai_seconds": fortai_seconds,
        "llama_cpp_seconds": llama_seconds,
        "fortai_steps_per_second": steps / fortai_seconds,
        "llama_cpp_steps_per_second": steps / llama_seconds,
    }


def self_test():
    fixture = {
        "steps": 8,
        "fortai": {"steps": "8", "forward_seconds": "0.4"},
        "llama_cpp": {
            "timings": {
                "cache_n": 0,
                "prompt_n": 1,
                "prompt_ms": 50.0,
                "predicted_n": 8,
                "predicted_ms": 350.0,
                # Deliberately unrelated: llama.cpp derives this from seven
                # generation evaluations, not the eight matched evaluations.
                "predicted_per_second": 20.0,
            }
        },
    }
    metrics = matched_forward_metrics(fixture)
    assert metrics["count"] == 8
    assert metrics["fortai_seconds"] == 0.4
    assert metrics["llama_cpp_seconds"] == 0.35
    assert metrics["fortai_steps_per_second"] == 20.0
    assert metrics["llama_cpp_steps_per_second"] == 8 / 0.35

    fixture["llama_cpp"]["timings"]["cache_n"] = 1
    try:
        matched_forward_metrics(fixture)
    except ValueError as error:
        assert "cache_n" in str(error)
    else:
        raise AssertionError("cached llama.cpp prompt was not rejected")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", nargs="?", type=Path)
    parser.add_argument(
        "--measurement-conditions", choices=("isolated", "shared_service")
    )
    parser.add_argument("--performance-gate-eligible", choices=("true", "false"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("CPU benchmark metric fixture passed")
        return
    if (
        args.result is None
        or args.measurement_conditions is None
        or args.performance_gate_eligible is None
    ):
        parser.error("result and measurement eligibility arguments are required")

    result = json.loads(args.result.read_text())
    result["measurement_conditions"] = args.measurement_conditions
    result["shared_service_conditions"] = (
        args.measurement_conditions == "shared_service"
    )
    result["performance_gate_eligible"] = args.performance_gate_eligible == "true"
    result["temporary_llama_server_cleanup"] = result["llama_server_cleanup"]
    result["matched_forward"] = matched_forward_metrics(result)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "matched_forward": result["matched_forward"],
                "measurement_conditions": result["measurement_conditions"],
                "performance_gate_eligible": result["performance_gate_eligible"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
