# FortAI

FortAI is a Fortran-native AI runtime for model loading, execution planning,
sampling, and architecture-specialized accelerator backends.

The first delivery is a buildable Qwen3.5 CPU reference path. It contains the
public runtime types, GGUF Q8_0 loading, Qwen3.5 hybrid recurrent/full-attention
execution for the 0.8B, 2B, and 4B model family, native OpenMP/SIMD matvecs,
and persistent benchmark/provenance tooling. The first CUDA Q8 resident GEMV
kernel is now measured on RTX 5060 Ti; the complete model backends remain
staged behind explicit acceptance gates.

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

# build and compare the resident CUDA Q8 GEMV kernel with llama.cpp ggml-cuda
benchmark/compare_cuda_q8.sh

# retain Nsight Compute (when GPU counters are permitted) and Nsight Systems data
benchmark/profile_cuda_q8.sh
```

The comparison and paired-profile scripts refuse to start when another
`llama-server` is already running. If a protected production server must stay
up, an explicit override starts the temporary CPU comparison on its own port;
these results are labelled shared-service measurements and are not a fair
performance-gate result:

```bash
FORTAI_ALLOW_EXISTING_LLAMA_SERVER=1 LLAMA_PORT=18081 \
  benchmark/compare_qwen35_cpu_llama.sh MODEL.gguf 9419 128 128
```

The local Lazy Fortran workflow is also supported:

```bash
fo
```

The model runner accepts a GGUF file, an initial Qwen tokenizer ID, a decode
step count, and a context limit. The persistent model fetch manifest is in
`.provenance/models.tsv`.

Benchmark results, logs, and perf data stay machine-local under
`benchmark/results`, `benchmark/logs`, and `benchmark/profiles`; the scripts
and their provenance logic are tracked.

## Status

Version 0.1.0 includes an experimental model-level Qwen3.5 CPU runtime for
Q8_0 GGUF and a resident CUDA Q8 GEMV kernel. The CPU path is benchmarked
against llama.cpp on the 0.8B, 2B, and 4B fixtures; the CUDA kernel is matched
against llama.cpp's ggml-cuda operation on identical resident data. Neither
is promoted as a complete production model backend until the named workload
gate passes. Full CUDA Qwen3.8-27B and its required multi-GPU split remain on
the roadmap. See [ROADMAP.md](ROADMAP.md).

FortAI will only promote a production candidate for a named workload after it
matches or beats the fastest fair competing harness under the same conditions.

## License

FortAI is released under the [MIT License](LICENSE).
