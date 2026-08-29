# FortAI

FortAI is a Fortran-native AI runtime for model loading, execution planning,
sampling, and architecture-specialized accelerator backends.

The first delivery is a buildable Qwen3.5 CPU reference path and a validated
device-resident CUDA path. It contains the
public runtime types, GGUF Q8_0 loading, Qwen3.5 hybrid recurrent/full-attention
execution for the 0.8B, 2B, and 4B model family, native OpenMP/SIMD matvecs,
and persistent benchmark/provenance tooling. The CUDA path keeps recurrent
state, attention KV caches, activations, and FFN execution on the RTX 5060 Ti;
native q8_0 K/V caches are resident and use the same FP16-scale byte layout as
the independent CPU oracle (mixed f16/q8_0 K/V is supported too).
CUDA Graph replay is available as an explicit opt-in, but is disabled by
default on the tested RTX 5060 Ti until it passes the end-to-end gate; Q4
mixed-quant `UD-Q4_K_XL` tensors are now handled natively through the exact
GGML CPU decoder and a persistent two-GPU GGML CUDA bridge. Native Qwen3.8
NextN/MTP is correctness-stable with a CUDA-resident target decode whenever
the configured KV cache fits; the small NextN verification head consumes an
explicit hidden-state handoff, with a host-boundary fallback when VRAM is
insufficient. The workload-specific speed gate remains explicit. A separate
compatibility smoke remains available as an independent upstream llama.cpp
oracle.

## Measured serving

The current Qwen3.8-27B `UD-Q4_K_XL` comparison was measured on 2026-08-29 with
CUDA 13.3 and two RTX 5060 Ti 16 GiB cards. Both servers were started from the
same production environment: MTP depth 2 (`draft-mtp`), flash attention,
`batch=2048`, `ubatch=256`, 14 threads, tensor split `0.57,0.43`, q8_0 target
K/V, q4_0 draft K/V, and the production sampler. FortAI used its production
262144-token context; llama.cpp `bb4caa7` used 8192 because it cannot allocate
the production K/V cache. Each rate is the median of three runs taken from the
server's own `timings` block, and every request carried a unique prefix so no
prefill was served from the prefix cache.

| Matched request | FortAI | llama.cpp | FortAI / llama.cpp |
|---|---:|---:|---:|
| Prefill, 2073-token prompt | 1222.7 tok/s | 1113.3 tok/s | 1.098 |
| Prefill, ~4050-token prompt | 1211.2 tok/s | 1195.6 tok/s | 1.013 |
| Generation, greedy, short context | 41.8 tok/s | 43.7 tok/s | 0.957 |
| Generation, sampled, short context | 38.1 tok/s | 49.7 tok/s | 0.767 |
| Generation, sampled, ~4050-token context | 43.3 tok/s | 45.6 tok/s | 0.949 |

Prefill is ahead. Generation is behind, and a paired Nsight profile locates the
cause precisely: **per-round cost is already at parity** -- 55.3 ms per
speculative round for FortAI against 56.1 ms for llama.cpp -- and the whole
difference is that FortAI needs about 13% more target-verification rounds
because fewer of its MTP drafts are accepted. Over four greedy prompts FortAI
accepts 277/440 drafts (63.0%) at 41.47 tok/s while llama.cpp accepts 286/421
(67.9%) at 43.39 tok/s; the 4.4% throughput gap equals the 4.2% tokens-per-round
gap. Both runtimes draft two tokens per round and use the same acceptance rule,
so the remaining work is draft-head numerics, not kernels, sampling, or
scheduling. `PLAN.md` records the full profile and the next step.

Reproduce either side with `benchmark/compare_qwen38_server.sh fortai|llama`.

At the production context FortAI occupies 15,894,315,008 bytes on GPU0 and
14,388,559,872 bytes on GPU1. llama.cpp aborts while requesting its 9 GiB GPU0
KV buffer at the same settings; its loadable 4096-token configuration occupies
about 8,948 and 9,088 MiB. These measurements support the current hardware
configuration, not a blanket claim for every context length or GPU.

### Whisper CUDA (`large-v3-turbo`)

These are end-to-end multipart HTTP latencies with four CPU threads, flash
attention, one visible RTX 5060 Ti, and deterministic decoding. “Warm” is the
steady-state median after the first request; first-request CUDA graph setup is
reported separately in the benchmark logs and intentionally not used as the
throughput comparison. Both implementations returned the same JFK
transcription; the 1-second silence case is included as a frontend/decoder
stress point.

| Audio context | Server | Warm median (ms) |
|---:|---|---:|
| 1 s silence | whisper.cpp | 79.9 |
| 1 s silence | FortAI | 81.7 |
| 11 s JFK sample | whisper.cpp | 128.2 |
| 11 s JFK sample | FortAI | 132.1 |

The native Whisper service is the production service on port `8427`; the
reference processes were private localhost servers and were stopped after the
comparison. Whisper exposes an end-to-end transcription time rather than
separate prefill/generation counters.

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

The exact llama.cpp/GGML source revision used for algorithm comparison and its
MIT notice are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Build and test

Requirements are a Fortran 2018 compiler, the local Lazy Fortran `fo`
driver, and the GGML backend libraries (set `FORTAI_GGML_PREFIX` when they
are outside the default installation).

```bash
tools/build_native.sh
```

The build script supplies the direct GGML CPU/CUDA link set to both the build
and behavioral test targets; use it for reproducible local validation.

For the native CPU candidate build:

```bash
tools/build_native.sh
benchmark/run_cpu_reference.sh

# model-level CPU smoke/benchmark
benchmark/run_qwen35_cpu.sh .provenance/downloads/qwen35-0.8b/Qwen3.5-0.8B-Q8_0.gguf

# optional llama.cpp ABI comparison path (never used by the native server)
FORTAI_LLAMA_FASTPATH=cpu benchmark/run_qwen35_cpu.sh MODEL.gguf
FORTAI_LLAMA_FASTPATH=cuda benchmark/compare_qwen35_cuda_llama.sh MODEL.gguf

# pass llama.cpp model/context controls through the comparison path
FORTAI_LLAMA_FASTPATH=cuda \
  FORTAI_FLASH_ATTN=on \
  LLAMACPP_BATCH=2048 LLAMACPP_UBATCH=256 LLAMACPP_PARALLEL=1 \
  LLAMACPP_CACHE_TYPE_K=q8_0 LLAMACPP_CACHE_TYPE_V=q8_0 \
  LLAMACPP_TENSOR_SPLIT=0.57,0.43 \
  benchmark/compare_qwen35_cuda_llama.sh MODEL.gguf
# force the native Fortran/GGML implementation
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

The native server honors the complete current `llama-server --help` command-line
and canonical `LLAMA_ARG_*` environment surface (and also accepts the
corresponding `FORTAI_*` and legacy `LLAMACPP_*` names). The launcher normalizes
those settings before model initialization, including split mode, tensor
fractions, draft/MTP cache types, reasoning controls, HTTP/UI/CORS settings,
and authentication. The independent llama.cpp comparison harness accepts the
corresponding `FORTAI_*`, `LLAMA_ARG_*`, and local service `LLAMACPP_*` settings
for flash attention, batch/ubatch sizes, parallel sequence count, K/V cache
types, KQV and generic operation offload, SWA/KV-unified mode, thread-batch
count, and multi-GPU tensor splitting. The CUDA runner reports `vram_*_bytes` before and after
loading so the memory delta can be compared with llama.cpp under the same
profile. Qwen3.8-style embedded NextN/MTP heads are also bound by the native
Fortran Qwen3.5 runtime. Set `FORTAI_NATIVE_MTP=1` (or
`FORTAI_SPEC_TYPE=draft-mtp`) to enable the native MTP path; a configured
head-only draft whose name contains `mtp` enables the same path automatically.
GGUF tensor payloads use a read-only file mapping by default (`--load-mode
auto`/`mmap`), matching llama.cpp's demand-paged ownership model and avoiding a
second heap copy during startup. `--load-mode none` (or `--no-mmap`) selects
the explicit read-copy fallback for diagnostics. After a successful
device-resident CUDA setup, file-backed pages for uploaded weights are
discarded while the CPU embedding and any host-controlled MTP tensors remain
available.
It keeps greedy target output exact and uses the CUDA-resident target pipeline
whenever the KV allocation fits, with a host-controlled NextN verification
handoff. The
independent llama.cpp remains the reference oracle for true batched speculative
throughput. FortAI accepts and validates the configured external
draft path, while a matching MTP sidecar is loaded into the target's `blk.64`
head tensors (without duplicating the target embedding/output weights) and
native speculative execution consumes that head. `/health` reports
`mtp_sidecar_active` when this merge is active. The production
service is FortAI-owned: build
it with `tools/build_cuda_server.sh` and launch
`tools/fortai-server --model MODEL.gguf --port 8080`. It loads the model
through the native Fortran Qwen3.5 runtime, initializes the FortAI CUDA
backend, and serves `/`, `/health`, `/props` (read-only unless `--props` is enabled), `/slots`, `/metrics`, `/v1/models`, `/tokenize`, `/detokenize`, `/apply-template`, `/v1/chat/completions`,
`/v1/completions`, and the OpenAI Responses-compatible `/v1/responses` wire
API. The tokenizer and template endpoints follow llama.cpp's request and
response shapes, including special-token flags, token pieces, strict token
ID validation, and optional generation-prompt suppression. The root endpoint is a small llama.cpp-style chat UI. The
server never launches `llama-server` and reports `X-FortAI-Backend: fortai`.
`--api-prefix`/`LLAMA_ARG_API_PREFIX` applies the configured base path to every
endpoint (including the embedded UI), `--path` serves a safe static tree, and
the C transport emits the configured CORS and Bearer-key policy.
The native CLI accepts the production launcher's `--chat-template-kwargs`
JSON object and applies its `enable_thinking`, `preserve_thinking`, and
`reasoning_effort` values as request defaults, matching the corresponding
llama.cpp configuration.
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
SSE events. Split two-GPU Q4_K_XL models use the deterministic resident CUDA
bridge by default when flash attention is enabled; Q8 tensors stay on the
primary CUDA context while Q4 tensors are split across the configured GPUs,
avoiding invalid cross-context matvecs. A bounded producer-stream CUDA event
ring handles the cross-context handoff on hosts without peer access. Set
`FORTAI_DISABLE_CUDA_Q4_DEVICE_PIPELINE=1` (or the compatibility spelling
`FORTAI_ENABLE_CUDA_Q4_DEVICE_PIPELINE=0`) to select the host-boundary route.
Native `draft-mtp` keeps only the NextN verification head on the host; the
target decode remains resident when the configured context fits in VRAM. If a
single large attention KV allocation cannot fit on the primary GPU, FortAI
retries the configured second GPU and records the per-layer placement. It
retains the other resident layers and crosses an explicit host boundary only
when both devices reject that layer's allocation.
Multimodal
projector paths are accepted and surfaced in `/health`; the native text runtime
reports vision/audio as unsupported and rejects non-text content parts instead
of silently dropping them. A standalone `--model-draft` is
loaded and advanced by the native Fortran service for greedy requests, with
target logits remaining authoritative; `draft-mtp` sidecars use the merged
`blk.64` NextN implementation. The server never delegates either feature to
another process.
The native service retains the exact token sequence represented by its live
KV/recurrent state between serialized requests.  A request that begins with
that sequence skips the already-evaluated prefix (including the common
OpenCode system prompt); `/health` reports `cache_reuse_supported`,
`cache_reuse_active`, and the retained token count.  The cache is single-slot
by design so it adds no second model or KV allocation; a non-prefix request
invalidates it and replays from position zero.
For native split-Q4 placement, `FORTAI_TENSOR_SPLIT=a,b` (or
`--tensor-split a,b`/`-ts a,b`) supplies normalized non-negative fractions for
the two visible GPUs; omitted values retain equal-byte balancing, and malformed
values fail model initialization instead of being ignored. The native server
also accepts llama.cpp's `-np`, `-fa`, `-fit`, `-sm`, `-mg`, `-md`, `-ctk`,
`-ctv`, and `-cram` short forms, plus `--ui`/`--no-ui` and llama's `auto`/
`all`/`none` GPU-layer values and `-t -1` automatic-thread spelling, so the
same production command line can be used on a separate FortAI port while
llama.cpp remains resident. `-c 0` uses the model's GGUF context length, just
like llama.cpp. `/health` includes the effective GGUF context size, thread
count, GPU-layer count, and main GPU alongside the other profile settings.
`split-mode
none` keeps all native tensors on `main-gpu`; `layer`, `row`, and `tensor` use
the native two-device Q4 placement policy.
On CPU, `--parallel`/`-np` greater than one creates isolated forked sequence
workers after model load; immutable GGUF pages remain shared and recurrent/KV
state is private per worker. CUDA reports `slot_mode=serialized` because CUDA
contexts are not fork-safe; its native calls remain ordered until per-slot
device ownership is implemented.
Main native K/V caches support `f32`, `f16`, and `q8_0`; q8_0 storage is
quantized with llama-compatible FP16 scales and is validated against the
independent CUDA/CPU trace oracle. With native CUDA selected, q8_0 K/V stays
device-resident, including full-context streaming softmax, whenever the
resident allocation fits; the host cache is retained for the host-boundary
fallback and the MTP verification head.

### Native Whisper large-v3-turbo

Legacy Whisper `ggml-*.bin` files are loaded and executed by FortAI's native
Fortran model, audio frontend, tokenizer, decoder, and HTTP service. The
low-level GGML tensor/device ABI is used only as a backend primitive; no
`libwhisper`, `whisper.cpp`, or subprocess is involved. The same
`fortai-server` binary selects this path automatically from the `.bin` model
suffix:

```bash
tools/build_cuda_server.sh
CUDA_VISIBLE_DEVICES=1 tools/fortai-server \
  --model /home/ert/.local/share/voxtype/models/ggml-large-v3-turbo.bin \
  --port 8427 --gpu-layers all --flash-attn on
```

`--gpu-layers 0` selects the CPU backend; `--main-gpu` and
`CUDA_VISIBLE_DEVICES` follow the same device-selection rules as the existing
server. `/`, `/ui`, `/health`, `/v1/health`, `/models`, `/v1/models`, and
`/metrics` are available, together with OpenAI-compatible
`/v1/audio/transcriptions` and `/v1/audio/translations` (direct PCM16 WAV or
`multipart/form-data` uploads). Multipart `language`, `task`, `temperature`,
`seed`, and `max_tokens` fields are honored. JSON output is escaped by the
native `string_t` path, and invalid limits, temperatures, languages, codecs,
or truncated WAV data fail explicitly. Only PCM16 WAV is currently
supported; compressed codecs are intentionally rejected rather than silently
decoded by an external library.

The model loader uses one selected CUDA device by default, matching the
production Whisper profile where the other GPU is reserved for the language
model. `memory_bytes` in `/health` and `fortai_whisper_memory_bytes` in
`/metrics` expose the native resident model, encoder, and decoder-cache
accounting. The native frontend and decoder were checked against an
independently built whisper.cpp CPU oracle on the same large-v3-turbo file;
first-token logits agree within the reference FP16/FP32 rounding envelope,
and the GPU route is exercised independently of the protected language-model
service.

Benchmark results, logs, and perf data stay machine-local under
`benchmark/results`, `benchmark/logs`, and `benchmark/profiles`; the scripts
and their provenance logic are tracked.

## Status

Version 0.1.0 includes a validated model-level Qwen3.5 CPU runtime for Q8_0
GGUF and a device-resident CUDA Qwen3.5 path. Exact trace and centered
top-logit parity are checked independently against llama.cpp (8-token traces
and top-8 logits pass at 1e-2 tolerance on CPU and CUDA). The native mixed-
quant fallback remains a correctness-first compatibility path: the earlier 2B
Q4_K_M probe measured
7.33 tok/s CPU versus 22.30 tok/s for llama.cpp at eight threads, and 76.9
tok/s CUDA versus 213.2 tok/s at two threads. Native FortAI is the default and
production path; `FORTAI_LLAMA_FASTPATH=cpu|cuda` is retained only for explicit
compatibility measurements, while `native` selects the Fortran/GGML path.
For an independent exact Q4_K_XL CUDA oracle smoke on the two GPUs, use
`benchmark/run_q4_cuda_compat.sh`; it starts and cleans up its own private
localhost server. See [ROADMAP.md](ROADMAP.md).

FortAI will only promote a production candidate for a named workload after it
matches or beats the fastest fair competing harness under the same conditions.

## License

FortAI is released under the [MIT License](LICENSE).
