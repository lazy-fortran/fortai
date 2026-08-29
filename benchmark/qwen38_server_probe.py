#!/usr/bin/env python3
"""Matched Qwen3.8-27B server sweep: TTFT, prefill, generation, memory.

Driven by benchmark/compare_qwen38_server.sh.  Every request carries a unique
nonce as its first characters so no prefill can be served from the shared
prefix cache.  Prefill and generation rates come from the llama.cpp-compatible
``timings`` block that both servers return, so neither side is charged for HTTP
or client overhead.  Time to first token is measured client-side from the
streaming endpoint, because that is what a caller actually waits for.
"""
import json
import os
import statistics
import subprocess
import threading
import time
import urllib.request

PORT = int(os.environ["PORT"])
SIDE = os.environ["SIDE"]
CONTEXT = int(os.environ["CONTEXT"])
GEN_TOKENS = int(os.environ["GEN_TOKENS"])
REPEATS = int(os.environ["REPEATS"])
RESULT_FILE = os.environ["RESULT_FILE"]
SERVER_PID = os.environ.get("SERVER_PID", "")
PROMPT_SIZES = [int(v) for v in os.environ["PROMPT_SIZES"].split(",")]
BASE = f"http://127.0.0.1:{PORT}"

WORDS = ("the quick brown fox jumps over the lazy dog while a Fortran compiler "
         "emits vectorized kernels for dense linear algebra on modern hardware ")


def gpu_memory_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=20).stdout
    return [int(v.strip()) for v in out.split("\n") if v.strip()]


def host_rss_mib():
    if not SERVER_PID:
        return None
    try:
        with open(f"/proc/{SERVER_PID}/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        return None
    return None


class MemorySampler(threading.Thread):
    """Track peak device and host memory while the sweep runs."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop_flag = threading.Event()
        self.peak_gpu = gpu_memory_mib()
        self.peak_rss = host_rss_mib() or 0

    def run(self):
        while not self.stop_flag.wait(0.5):
            try:
                self.peak_gpu = [max(a, b) for a, b in zip(self.peak_gpu, gpu_memory_mib())]
                self.peak_rss = max(self.peak_rss, host_rss_mib() or 0)
            except Exception:
                pass


def post(path, payload, timeout=1800):
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def token_count(text):
    return len(post("/tokenize", {"content": text})["tokens"])


def filler(target_tokens):
    """Grow then bisect the filler so it tokenizes to ~target_tokens."""
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


def body(prompt, stream):
    NONCE[0] += 1
    payload = {
        "model": "qwen",
        "messages": [{"role": "user",
                      "content": f"[run-{SIDE}-{os.getpid()}-{NONCE[0]}] " + prompt}],
        "max_tokens": GEN_TOKENS,
        "temperature": 0.0,
    }
    if stream:
        payload["stream"] = True
    return payload


def measure_rates(prompt):
    result = post("/v1/chat/completions", body(prompt, False))
    t = result.get("timings", {})
    return t.get("prompt_n"), t.get("prompt_per_second"), t.get("predicted_per_second")


def measure_ttft(prompt):
    """Wall time from request submission to the first streamed token."""
    data = json.dumps(body(prompt, True)).encode()
    request = urllib.request.Request(
        BASE + "/v1/chat/completions", data=data,
        headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                delta = json.loads(chunk)["choices"][0].get("delta", {})
            except (ValueError, KeyError, IndexError):
                continue
            if delta.get("content") or delta.get("reasoning_content"):
                return 1000.0 * (time.perf_counter() - start)
    return None


sampler = MemorySampler()
loaded_gpu = list(sampler.peak_gpu)
loaded_rss = sampler.peak_rss
sampler.start()

cases = []
for size in PROMPT_SIZES:
    prompt = filler(size) + "\n\nSummarize the passage above in one sentence."
    # A server-side failure at one prompt size must not discard the sizes that
    # did work; record it as a failed case and continue the sweep.
    try:
        runs = [measure_rates(prompt) for _ in range(REPEATS)]
        ttft = measure_ttft(prompt)
    except Exception as exc:
        cases.append({"requested_prompt_tokens": size, "error": repr(exc)})
        print(f"  prompt ~{size:6d} tok: FAILED {exc}")
        continue
    cases.append({
        "requested_prompt_tokens": size,
        "prompt_tokens": runs[0][0],
        "prefill_tokens_per_second": statistics.median(r[1] for r in runs),
        "generation_tokens_per_second": statistics.median(r[2] for r in runs),
        "ttft_ms": ttft,
    })
    print(f"  prompt {runs[0][0]:6d} tok: prefill {statistics.median(r[1] for r in runs):8.1f} tok/s  "
          f"gen {statistics.median(r[2] for r in runs):5.2f} tok/s  ttft {ttft:8.1f} ms")

sampler.stop_flag.set()
sampler.join(timeout=5)

result = {
    "side": SIDE,
    "context": CONTEXT,
    "generation_tokens": GEN_TOKENS,
    "repeats": REPEATS,
    "memory_mib": {
        "gpu_after_load": loaded_gpu,
        "gpu_peak": sampler.peak_gpu,
        "host_rss_after_load": loaded_rss,
        "host_rss_peak": sampler.peak_rss,
    },
    "cases": cases,
}
with open(RESULT_FILE, "w") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
print(f"  memory: GPU after load {loaded_gpu} MiB, peak {sampler.peak_gpu} MiB, "
      f"host RSS peak {sampler.peak_rss} MiB")
