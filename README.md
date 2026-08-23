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
repacking and multi-GPU 27B execution remain staged behind explicit gates.

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

# build and compare the device-resident Qwen3.5 CUDA path with llama.cpp
benchmark/compare_qwen35_cuda_llama.sh \
  .provenance/downloads/qwen35-0.8b/Qwen3.5-0.8B-Q8_0.gguf

benchmark/check_qwen35_cuda_trace.sh \
  .provenance/downloads/qwen35-0.8b/Qwen3.5-0.8B-Q8_0.gguf

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

Benchmark results, logs, and perf data stay machine-local under
`benchmark/results`, `benchmark/logs`, and `benchmark/profiles`; the scripts
and their provenance logic are tracked.

## Status

Version 0.1.0 includes an experimental model-level Qwen3.5 CPU runtime for
Q8_0 GGUF and a device-resident CUDA Qwen3.5 path. The CPU path is benchmarked
against llama.cpp on the 0.8B, 2B, and 4B fixtures; the CUDA path has matched
trace tests on 0.8B and smoke coverage on all three fixtures. Neither is
promoted until the named workload gate passes. CUDA Graph replay, Q4
repacking, and full CUDA Qwen3.8-27B with its required multi-GPU split remain
on the roadmap. See [ROADMAP.md](ROADMAP.md).

FortAI will only promote a production candidate for a named workload after it
matches or beats the fastest fair competing harness under the same conditions.

## License

FortAI is released under the [MIT License](LICENSE).
