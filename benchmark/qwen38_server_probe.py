#!/usr/bin/env python3
"""Matched request suite for the Qwen3.8-27B server benchmark.

Driven by benchmark/compare_qwen38_server.sh.  Every request carries a unique
nonce as its first characters so no prefill can be served from the shared
prefix cache.  Prefill and generation rates come from the llama.cpp-compatible
``timings`` block that both servers return, so neither side is charged for HTTP
or client overhead.
"""
import json
import os
import statistics
import time
import urllib.request

PORT = int(os.environ["PORT"])
SIDE = os.environ["SIDE"]
CONTEXT = int(os.environ["CONTEXT"])
GEN_TOKENS = int(os.environ["GEN_TOKENS"])
REPEATS = int(os.environ["REPEATS"])
RESULT_FILE = os.environ["RESULT_FILE"]
BASE = f"http://127.0.0.1:{PORT}"

# Deterministic filler.  Word choice is irrelevant; only the token count is.
WORDS = ("the quick brown fox jumps over the lazy dog while a Fortran compiler "
         "emits vectorized kernels for dense linear algebra on modern hardware ")


def post(path, payload, timeout=1800):
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    body["_wall_seconds"] = time.perf_counter() - start
    return body


def token_count(text):
    return len(post("/tokenize", {"content": text})["tokens"])


def filler(target_tokens):
    """Grow the filler until it tokenizes to at least ``target_tokens``."""
    text = WORDS
    while token_count(text) < target_tokens:
        text = text * 2
    words = text.split(" ")
    low, high = 1, len(words)
    while low < high:
        mid = (low + high) // 2
        if token_count(" ".join(words[:mid])) >= target_tokens:
            high = mid
        else:
            low = mid + 1
    return " ".join(words[:low])


NONCE = [0]


def nonce():
    NONCE[0] += 1
    return f"[run-{SIDE}-{os.getpid()}-{NONCE[0]}] "


def chat(prompt, max_tokens, temperature, seed=None):
    payload = {
        "model": "qwen",
        "messages": [{"role": "user", "content": nonce() + prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if seed is not None:
        payload["seed"] = seed
    return post("/v1/chat/completions", payload)


def timings(body):
    t = body.get("timings", {})
    return {
        "prompt_n": t.get("prompt_n"),
        "prompt_per_second": t.get("prompt_per_second"),
        "predicted_n": t.get("predicted_n"),
        "predicted_per_second": t.get("predicted_per_second"),
        "wall_seconds": body["_wall_seconds"],
    }


def median_of(runs, key):
    values = [r[key] for r in runs if r.get(key)]
    return statistics.median(values) if values else None


SHORT_PROMPT = "Reply with exactly the word FORTAI_OK and nothing else."
LONG_FILLER = filler(4000)
LONG_PROMPT = LONG_FILLER + "\n\nSummarize the passage above in one sentence."

cases = []

# Prefill: one decoded token, so the rate is dominated by prompt ingestion.
for label, prompt in (("prefill_2k", filler(2000) + "\n\nAnswer with OK."),
                      ("prefill_4k", LONG_PROMPT)):
    runs = [timings(chat(prompt, 1, 0.0)) for _ in range(REPEATS)]
    cases.append({"case": label, "runs": runs,
                  "prompt_n": runs[0]["prompt_n"],
                  "median_prompt_per_second": median_of(runs, "prompt_per_second")})

# Generation: short context, greedy and production-sampled.
for label, temperature, seed in (("generation_greedy_short", 0.0, None),
                                 ("generation_sampled_short", 0.6, 1234)):
    runs = [timings(chat(SHORT_PROMPT, GEN_TOKENS, temperature, seed))
            for _ in range(REPEATS)]
    cases.append({"case": label, "runs": runs,
                  "median_predicted_per_second": median_of(runs, "predicted_per_second")})

# Generation at a grown context: the prefill is charged separately by timings.
runs = [timings(chat(LONG_PROMPT, GEN_TOKENS, 0.6, 1234)) for _ in range(REPEATS)]
cases.append({"case": "generation_sampled_4k", "runs": runs,
              "prompt_n": runs[0]["prompt_n"],
              "median_predicted_per_second": median_of(runs, "predicted_per_second")})

result = {
    "side": SIDE,
    "context": CONTEXT,
    "generation_tokens": GEN_TOKENS,
    "repeats": REPEATS,
    "cases": cases,
}
with open(RESULT_FILE, "w") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
print(json.dumps({c["case"]: {k: v for k, v in c.items() if k != "runs"}
                  for c in cases}, indent=2))
