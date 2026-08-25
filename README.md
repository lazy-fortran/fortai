# FortAI

FortAI is a Fortran-native AI runtime for model loading, execution planning,
sampling, and architecture-specialized accelerator backends.

The first delivery is a buildable Qwen3.5 CPU reference path and an
experimental device-resident CUDA path. It contains the
public runtime types, GGUF Q8_0 loading, Qwen3.5 hybrid recurrent/full-attention
execution for the 0.8B, 2B, and 4B model family, native OpenMP/SIMD matvecs,
and persistent benchmark/provenance tooling. The CUDA path keeps recurrent
state, attention KV caches, activations, and FFN execution on the RTX 5060 Ti.
CUDA Graph replay is available as an explicit opt-in, but is disabled by
default on the tested RTX 5060 Ti until it passes the end-to-end gate; Q4
mixed-quant `UD-Q4_K_XL` tensors are now handled natively through the exact
GGML CPU decoder and a persistent two-GPU GGML CUDA bridge. The native path is
experimental until its model-level promotion gate is closed. A separate
compatibility smoke remains available as an independent upstream llama.cpp
oracle.

## Design

FortAI keeps model semantics, state lifetime, and scheduling in Fortran. A
backend owns device-specific execution and kernel selection. The intended
compilation path is:

```text
GGUF metadata -> ModelIR -> PlanIR -> backend specialization
             -> packed weights -> validated execution plan
             -> measured kernel selection
```

The execution plan is bound before the token loop. The hot path will use
specialized calls with persistent state instead of rebuilding a generic graph
for every token.

The repository follows the Lazy Fortran conventions:

- independent CPU behavior is the correctness reference
- device implementations must report compiler and hardware provenance
- candidate kernels are accepted only after correctness checks
- transfer-inclusive and resident measurements are recorded separately
- unsupported backends fail explicitly and never fall back silently

See [docs/architecture.md](docs/architecture.md) for the staged implementation
plan.

## Build and test

Requirements are a Fortran 2018 compiler and the local Lazy Fortran `fo`
driver (which invokes fpm-compatible project builds).

```bash
fo build
fo test
fo exec --no-build fortai --version
```

For the native CPU candidate build:

```bash
tools/build_native.sh
benchmark/run_cpu_reference.sh

# model-level CPU smoke/benchmark
benchmark/run_qwen35_cpu.sh .provenance/downloads/qwen35-0.8b/Qwen3.5-0.8B-Q8_0.gguf

# llama.cpp ABI fast path (automatic when the resident library is available)
FORTAI_LLAMA_FASTPATH=cpu benchmark/run_qwen35_cpu.sh MODEL.gguf
FORTAI_LLAMA_FASTPATH=cuda benchmark/compare_qwen35_cuda_llama.sh MODEL.gguf

# pass llama.cpp model/context controls through the resident fast path
FORTAI_LLAMA_FASTPATH=cuda \
  FORTAI_FLASH_ATTN=on \
  LLAMACPP_BATCH=2048 LLAMACPP_UBATCH=256 LLAMACPP_PARALLEL=1 \
  LLAMACPP_CACHE_TYPE_K=q8_0 LLAMACPP_CACHE_TYPE_V=q8_0 \
  LLAMACPP_TENSOR_SPLIT=0.57,0.43 \
  benchmark/compare_qwen35_cuda_llama.sh MODEL.gguf
# force the native Fortran/GGML fallback for diagnostics
FORTAI_LLAMA_FASTPATH=native benchmark/run_qwen35_cpu.sh MODEL.gguf

# fair short-decode comparison; FortAI and llama.cpp run sequentially
OMP_NUM_THREADS=4 OMP_PROC_BIND=spread \
  benchmark/compare_qwen35_cpu_llama.sh \
  .provenance/downloads/qwen35-0.8b/Qwen3.5-0.8B-Q8_0.gguf 9419 8 128

# find the best thread count for both harnesses
benchmark/tune_qwen35_cpu.sh \
  .provenance/downloads/qwen35-0.8b/Qwen3.5-0.8B-Q8_0.gguf

# collect perf counters and a symbol-level call-graph profile
OMP_NUM_THREADS=4 benchmark/profile_qwen35_cpu.sh \
  .provenance/downloads/qwen35-0.8b/Qwen3.5-0.8B-Q8_0.gguf

# collect matched decode-only perf profiles for FortAI and llama.cpp
OMP_NUM_THREADS=4 benchmark/profile_qwen35_cpu_both.sh \
  .provenance/downloads/qwen35-0.8b/Qwen3.5-0.8B-Q8_0.gguf

# use passwordless sudo for hosts that restrict perf attach/record access
FORTAI_PERF_USE_SUDO=1 OMP_NUM_THREADS=4 \
  benchmark/profile_qwen35_cpu_both.sh \
  .provenance/downloads/qwen35-0.8b/Qwen3.5-0.8B-Q8_0.gguf

# build and compare the device-resident Qwen3.5 CUDA path with llama.cpp
benchmark/compare_qwen35_cuda_llama.sh \
  .provenance/downloads/qwen35-0.8b/Qwen3.5-0.8B-Q8_0.gguf

benchmark/check_qwen35_cuda_trace.sh \
  .provenance/downloads/qwen35-0.8b/Qwen3.5-0.8B-Q8_0.gguf

# retain Nsight Compute (when GPU counters are permitted) and Nsight Systems data
benchmark/profile_cuda_q8.sh
```

The CPU comparison wrapper refuses to start when an unverified
`llama-server` is already running. The exact protected GPU resident service
(PID `268006`, port `8080`) is independently allowed: CPU comparisons still
run on a separately checked temporary port and remain eligible for the
performance gate. Other resident servers can be used only with the explicit
shared-service override, which marks the result ineligible:

```bash
FORTAI_ALLOW_EXISTING_LLAMA_SERVER=1 LLAMA_PORT=18081 \
  benchmark/compare_qwen35_cpu_llama.sh MODEL.gguf 9419 128 128
```

The tuning script is a one-run screening sweep, not promotion evidence. A
candidate winner requires repeated matched-forward measurements at each
relevant thread count. Shared-service result JSON is explicitly marked
`performance_gate_eligible: false`.

The local Lazy Fortran workflow is also supported:

```bash
fo
```

The model runner accepts a GGUF file, an initial Qwen tokenizer ID, a decode
step count, and a context limit. The persistent model fetch manifest is in
`.provenance/models.tsv`.

The resident llama.cpp adapter honors the corresponding `FORTAI_*`,
`LLAMA_ARG_*`, and local service `LLAMACPP_*` settings for flash attention,
batch/ubatch sizes, parallel sequence count, K/V cache types, KQV and generic
operation offload, SWA/KV-unified mode, thread-batch count, and multi-GPU
tensor splitting. The CUDA runner reports `vram_*_bytes` before and after
loading so the memory delta can be compared with llama.cpp under the same
profile. Qwen3.8-style embedded NextN/MTP heads are also bound by the native
Fortran Qwen3.5 runtime. Set `FORTAI_NATIVE_MTP=1` (or
`FORTAI_SPEC_TYPE=draft-mtp`) to enable the host-controlled MTP KV path; it
keeps greedy target output exact and disables the device-resident graph until
hidden-state export is available. The resident llama.cpp adapter remains the
reference path for true batched speculative throughput, while multimodal
projector execution and external MTP sidecar loading remain outside this
direct single-sequence native path. The production service is FortAI-owned: build
it with `tools/build_cuda_server.sh` and launch
`tools/fortai-server --model MODEL.gguf --port 8080`. It loads the model
through the native Fortran Qwen3.5 runtime, initializes the FortAI CUDA
backend, and serves `/`, `/health`, `/v1/models`, `/v1/chat/completions`,
`/v1/completions`, and the OpenAI Responses-compatible `/v1/responses` wire
API. The root endpoint is a small llama.cpp-style chat UI. The
server never launches `llama-server` and reports `X-FortAI-Backend: fortai`.
The native server owns Qwen chat formatting and decoding.  It supports
`temperature`/`seed`, `stream`, and Qwen thinking controls (`enable_thinking`,
`reasoning_format`) while exposing reasoning as `reasoning_content`; stop
tokens are removed from the generated message. Sampling accepts `top_k`,
`top_p`, `min_p`, `repeat_penalty`, `repeat_last_n`, `presence_penalty`, and
`frequency_penalty` in addition to `temperature`/`seed`; the native sampler
applies them before decoding (`repeat_last_n=0` disables repetition penalties;
`-1` covers the full context). Usage reports native prompt/completion token
counts, and malformed or unsafe token budgets are rejected before allocation.
`chat_template_kwargs.enable_thinking` and Responses `reasoning.effort` use the
model's GGUF chat-template default and the exact empty `<think>` block for hard
no-thinking mode. Qwen3.8 `reasoning_effort` (`low`, `medium`, or `xhigh`) and
`preserve_thinking` are forwarded into the native prompt, including their
OpenAI-compatible aliases. Chat and Responses requests also accept OpenAI
`tools`; the native formatter emits Qwen3.5/Qwen3.8's
`<tool_call><function=...>` protocol and returns structured function-call
items (including streaming argument deltas). Responses requests accept string
or message-array `input`, `instructions`, `reasoning.effort`, and streaming
SSE events. Split two-GPU Q4_K_XL models
use the deterministic host-boundary CUDA route by default; the resident
all-device bridge remains an explicit `FORTAI_ENABLE_CUDA_Q4_DEVICE_PIPELINE=1`
diagnostic opt-in until its model-level determinism gate is closed.  Multimodal
projector execution and external MTP sidecars remain outside this direct
single-sequence server rather than being silently delegated to another server.

Benchmark results, logs, and perf data stay machine-local under
`benchmark/results`, `benchmark/logs`, and `benchmark/profiles`; the scripts
and their provenance logic are tracked.

## Status

Version 0.1.0 includes an experimental model-level Qwen3.5 CPU runtime for
Q8_0 GGUF and a device-resident CUDA Qwen3.5 path. Exact trace and centered
top-logit parity are checked independently against llama.cpp (8-token traces
and top-8 logits pass at 1e-2 tolerance on CPU and CUDA). The native mixed-
quant fallback remains experimental: the earlier 2B Q4_K_M probe measured
7.33 tok/s CPU versus 22.30 tok/s for llama.cpp at eight threads, and 76.9
tok/s CUDA versus 213.2 tok/s at two threads. When the resident llama.cpp ABI
library is available, the model runner automatically uses the same resident
graph, repack kernels, and CPU threadpool as the comparison harness; set
`FORTAI_LLAMA_FASTPATH=native` to measure the native fallback. Explicit
`cpu`/`cuda` selections remain available for deployments that need to pin the
backend.
For an independent exact Q4_K_XL CUDA oracle smoke on the two GPUs, use
`benchmark/run_q4_cuda_compat.sh`; it starts and cleans up its own private
localhost server. See [ROADMAP.md](ROADMAP.md).

FortAI will only promote a production candidate for a named workload after it
matches or beats the fastest fair competing harness under the same conditions.

## License

FortAI is released under the [MIT License](LICENSE).
