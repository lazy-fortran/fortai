# FortAI

FortAI is a Fortran-native AI runtime for model loading, execution planning,
sampling, and architecture-specialized accelerator backends.

The first release is a small, buildable foundation. It contains the public
runtime types, a deterministic byte tokenizer, a greedy sampler, an OpenMP and
SIMD-enabled CPU reference matvec, a minimal GGUF header reader, and contracts
for model and execution plans. CUDA, Metal, MLX, HIP, SYCL, Vulkan, and TinyGPU
backends are reserved for measured implementations.

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

Requirements are a Fortran 2018 compiler and [fpm](https://fpm.fortran-lang.org/).

```bash
fpm build
fpm test
fpm run -- --version
```

For the native CPU candidate build:

```bash
tools/build_native.sh
benchmark/run_cpu_reference.sh
```

The local Lazy Fortran workflow is also supported:

```bash
fo
```

The executable currently reports the project version and exposes no model
loading command. That interface will be added with the first end-to-end model
path.

## Status

Version 0.1.0 is an API and reference-kernel seed. It is suitable for building
the contracts around GGUF, ModelIR, PlanIR, CPU execution, and independent
tests. It is not yet a model-level LLM runtime and it does not claim
accelerator or llama.cpp performance. See [ROADMAP.md](ROADMAP.md).

FortAI will only promote a production candidate for a named workload after it
matches or beats the fastest fair competing harness under the same conditions.

## License

FortAI is released under the [MIT License](LICENSE).
