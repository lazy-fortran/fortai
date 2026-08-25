# Tools

The planned tools will pack weights, tune validated candidates, and generate
backend bindings or execution plans. They remain separate from the runtime so
model loading and serving do not depend on a code-generation toolchain.

`fortai-server` is the FortAI-native HTTP entrypoint. It serves the OpenAI
compatible `/v1/chat/completions`, `/v1/completions`, `/v1/responses`,
`/v1/models`, `/health`, `/metrics`, and the embedded `/` UI from the Fortran Qwen3.5 runtime; it never launches
`llama-server`. `tools/build_cuda_server.sh` builds the CUDA binary, while
`fo build` builds the CPU diagnostic binary. Set `FORTAI_SERVER_BIN` only when
selecting an already-built FortAI binary explicitly.

The native CLI accepts the production llama.cpp profile controls (`--alias`,
`--parallel`, `--tensor-split`, `--model-draft`, `--spec-type`, `--flash-attn`,
K/V cache types, batch/ubatch, cache budget/reuse, sampler defaults, reasoning
budget, mmproj path/offload, and `--no-webui`) and mirrors their
`LLAMACPP_*` environment variables into `FORTAI_*`. `/health` reports the
effective values. Main Qwen K/V caches support `f16`, `f32`, and `q8_0`; native
CUDA keeps q8_0 K/V resident with mixed f16/q8_0 combinations, while the
host-controlled/MTP path uses the same quantization layout on the host. The
native MTP path uses an embedded NextN head. A standalone external draft model
is loaded and advanced by the native Fortran service for greedy requests; the
target model remains the correctness oracle.
The ISO-C socket transport uses a bounded worker queue for `--threads-http`
and `--parallel`, so multiple connections can be accepted without corrupting
the mutable recurrent/KV state.
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

Split two-GPU Q4_K_XL serving uses the deterministic host-boundary CUDA path by
default. `FORTAI_ENABLE_CUDA_Q4_DEVICE_PIPELINE=1` enables the resident
all-device bridge for diagnostics. The bridge attaches the native Q8 consumer
stream and uses rotating CUDA events for cross-stream hand-off; it is not yet
the production default because its end-to-end performance/determinism gate is
still open. `FORTAI_TENSOR_SPLIT=a,b` (also accepted as
`--tensor-split a,b`) controls native Q4 byte placement across the two GPUs;
fractions are normalized and invalid values fail initialization.
