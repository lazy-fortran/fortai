# Tools

The planned tools will pack weights, tune validated candidates, and generate
backend bindings or execution plans. They remain separate from the runtime so
model loading and serving do not depend on a code-generation toolchain.

`fortai-server` is the FortAI-native HTTP entrypoint. It serves the OpenAI
compatible `/v1/chat/completions`, `/v1/completions`, `/v1/responses`,
`/v1/models`, and `/health` endpoints from the Fortran Qwen3.5 runtime; it never launches
`llama-server`. `tools/build_cuda_server.sh` builds the CUDA binary, while
`fo build` builds the CPU diagnostic binary. Set `FORTAI_SERVER_BIN` only when
selecting an already-built FortAI binary explicitly.

The chat and Responses endpoints share the Qwen3.5 template: `enable_thinking`,
`chat_template_kwargs.enable_thinking`, and `reasoning.effort` select the hard
thinking/no-thinking prompt form. OpenAI `tools` are formatted natively and
Qwen3.5 XML tool calls are returned as structured function-call objects and
streaming argument events. Sampling parameters (`top_k`, `top_p`, `min_p`,
repeat/presence/frequency penalties, and `seed`) are applied by the native
Fortran decoder rather than being silently ignored. Successful responses
report native prompt/completion token usage; invalid token budgets fail before
model allocation.

Split two-GPU Q4_K_XL serving uses the deterministic host-boundary CUDA path by
default. `FORTAI_ENABLE_CUDA_Q4_DEVICE_PIPELINE=1` enables the resident
all-device bridge for diagnostics; it is not yet the production default because
its end-to-end determinism gate is still open.
