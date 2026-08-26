# Tools

The planned tools will pack weights, tune validated candidates, and generate
backend bindings or execution plans. They remain separate from the runtime so
model loading and serving do not depend on a code-generation toolchain.

`fortai-server` is the FortAI-native HTTP entrypoint. It serves the OpenAI
compatible `/v1/chat/completions`, `/v1/completions`, `/v1/responses`,
`/v1/models`, `/tokenize`, `/detokenize`, `/apply-template`, `/health`, `/props` (writable with `--props`), `/slots`, `/metrics`, and the embedded `/` UI from the Fortran Qwen3.5 runtime; it never launches
`llama-server`. `tools/build_cuda_server.sh` builds the CUDA binary, while
`fo build` builds the CPU diagnostic binary. Set `FORTAI_SERVER_BIN` only when
selecting an already-built FortAI binary explicitly.

The native CLI accepts the complete current `llama-server --help` option and
alias surface, including model/GPU placement, split ratios, batching, KV and
draft/MTP settings, sampling, reasoning, multimodal, static/UI, CORS,
authentication, router, and speculative-decoding controls. Every documented
`LLAMA_ARG_*` variable is normalized before model initialization; the local
`LLAMACPP_*` names remain accepted for the existing systemd profile. Options
whose execution is not part of the native Qwen runtime are retained in the
effective configuration and exposed by `/health` rather than silently dropped.
The production launcher's `--chat-template-kwargs` option is accepted as
well; its JSON object is used as the native request default (including
`preserve_thinking` and `enable_thinking`).
`/health` also reports the effective context, threads, GPU-layer count, and
main GPU.
As in llama.cpp, `-c 0` selects the context length stored in the GGUF.
`/health` reports the effective values. Main Qwen K/V caches support `f16`,
`f32`, and `q8_0`; native
CUDA keeps q8_0 K/V resident with mixed f16/q8_0 combinations whenever the
requested context fits, while the host-boundary fallback uses the same
quantization layout on the host. The native MTP path uses the embedded NextN
head and its two-token greedy speculative step; a configured head-only
`mtp*.gguf` sidecar is merged into the target `blk.64` tensors and is reported
as `mtp_sidecar_active` by `/health`. Its target decode remains CUDA-resident
when possible; if one large attention cache cannot fit, only that layer uses
the host-boundary oracle while the remaining layers stay resident. A
standalone external draft model is loaded and advanced by the native Fortran
service for greedy requests when it is a real transformer draft; MTP sidecars
use the target's NextN path.
The ISO-C socket transport uses `--parallel` as sequence slots. On CPU, two or
more slots are isolated forked workers: immutable GGUF pages stay shared while
each worker owns its recurrent/KV/work state, so requests can execute at the
same time. CUDA contexts are not fork-safe, so CUDA currently reports
`slot_mode=serialized` and keeps native model calls ordered until per-slot
device ownership is available. `--threads-http` still controls the bounded
accept queue when the serialized transport is active.
`/v1/completions` accepts either one prompt string or a bounded string array;
array prompts are decoded natively and returned as indexed choices under the
configured `--batch-size` limit.

The chat and Responses endpoints read the thinking default from the GGUF chat
template: `enable_thinking`, `chat_template_kwargs.enable_thinking`, and
`reasoning.effort` select the hard thinking/no-thinking prompt form.
Qwen3.8's `reasoning_effort` and `preserve_thinking` controls are forwarded to
the prompt as well. OpenAI `tools` are formatted natively and Qwen3.5/Qwen3.8
XML tool calls are returned as structured function-call objects and streaming
argument events. Sampling parameters (`top_k`, `top_p`, `min_p`,
repeat/presence/frequency penalties, and `seed`) are applied by the native
Fortran decoder rather than being silently ignored. Successful responses
report native prompt/completion token usage; invalid token budgets fail before
model allocation.

Split two-GPU Q4_K_XL serving uses the deterministic resident CUDA bridge by
default when flash attention is enabled. Set
`FORTAI_DISABLE_CUDA_Q4_DEVICE_PIPELINE=1` (or the compatibility spelling
`FORTAI_ENABLE_CUDA_Q4_DEVICE_PIPELINE=0`) to select the host-boundary route.
The bridge uses a bounded producer-stream CUDA event ring for cross-stream
hand-off on hosts without CUDA peer access. `FORTAI_TENSOR_SPLIT=a,b` (also accepted as
`--tensor-split a,b`/`-ts a,b`) controls native Q4 byte placement across the two GPUs;
fractions are normalized and invalid values fail initialization. `split-mode
none` keeps native tensors on `main-gpu`; `layer`, `row`, and `tensor` select
the native two-device Q4 placement policy.
