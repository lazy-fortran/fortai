# Tools

The planned tools will pack weights, tune validated candidates, and generate
backend bindings or execution plans. They remain separate from the runtime so
model loading and serving do not depend on a code-generation toolchain.

`fortai-server` is the production-compatible entrypoint. It forwards its
complete argument vector to the selected upstream `llama-server` without
rewriting flags or ports. Override the binary with `FORTAI_LLAMA_SERVER`,
`LLAMACPP_SERVER`, or `LLAMA_SERVER` when more than one llama.cpp build is
installed.
