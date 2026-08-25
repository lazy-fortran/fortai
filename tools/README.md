# Tools

The planned tools will pack weights, tune validated candidates, and generate
backend bindings or execution plans. They remain separate from the runtime so
model loading and serving do not depend on a code-generation toolchain.

`fortai-server` is the FortAI-native HTTP entrypoint. It serves the OpenAI
compatible `/v1/chat/completions`, `/v1/completions`, `/v1/models`, and
`/health` endpoints from the Fortran Qwen3.5 runtime; it never launches
`llama-server`. `tools/build_cuda_server.sh` builds the CUDA binary, while
`fo build` builds the CPU diagnostic binary. Set `FORTAI_SERVER_BIN` only when
selecting an already-built FortAI binary explicitly.

Split two-GPU Q4_K_XL serving uses the deterministic host-boundary CUDA path by
default. `FORTAI_ENABLE_CUDA_Q4_DEVICE_PIPELINE=1` enables the resident
all-device bridge for diagnostics; it is not yet the production default because
its end-to-end determinism gate is still open.
