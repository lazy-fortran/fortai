# Roadmap

The current delivery target is a CPU reference path with native compilation,
OpenMP parallelism, SIMD directives, and reproducible benchmark tooling.

## CPU path

| ID | Work | Status | Acceptance |
| --- | --- | --- | --- |
| FAI-CPU-001 | Qwen3.5-0.8B text-only CPU reference | open | token-by-token logits agree with an independent oracle |
| FAI-CPU-002 | Qwen3.5-4B CPU scaling path | open | same parity suite, memory and throughput recorded |
| FAI-CPU-003 | native CPU candidate tournament | open | compiler flags, thread count, and winner recorded |
| FAI-CPU-MOE-001 | CPU MoE execution for Qwen3.6-35B | deferred | expert routing, resident weights, and llama.cpp comparison |

The CPU implementation should use the available compiler features for the
machine under test, including `-O3`, `-march=native`, `-mtune=native`, OpenMP,
SIMD, loop transformation, and link-time optimization where supported. Fast
math flags belong to measured candidate builds, not the correctness oracle.

## GPU path

| ID | Work | Status | Acceptance |
| --- | --- | --- | --- |
| FAI-CUDA-001 | single-device CUDA backend | deferred | CPU parity and resident-device benchmark |
| FAI-MGPU-001 | Qwen3.8-27B multi-GPU split | deferred | two-device placement, transfer accounting, and stable generation |
| FAI-CUDA-002 | Q4 repack and fused GDN kernels | deferred | validated kernel tournament on named GPUs |
| FAI-CUDA-003 | CUDA MTP and speculative decoding | deferred | acceptance rate and end-to-end throughput |

The 27B path must include multi-GPU placement from its first production CUDA
implementation. A single-device prototype may be used for kernel debugging,
but it is not the production target.

## Performance contract

FortAI production status is scoped to a named model, quantization, hardware,
context, batch, and workload. For each such tuple, the selected FortAI
implementation must match or beat the fastest independently measured
competing harness under the same conditions. A slower candidate remains
experimental and cannot become the default execution plan. If no fair
comparison exists, the result stays unpromoted until one is available.

## Benchmark and provenance

| ID | Work | Status | Acceptance |
| --- | --- | --- | --- |
| FAI-BENCH-001 | persistent CPU reference benchmark | in progress | native build and CSV result with provenance |
| FAI-BENCH-002 | llama.cpp model benchmark harness | deferred | same model, prompt, context, output budget, and competitor gate |
| FAI-PROV-001 | source fetch and revision records | in progress | tracked fetch tooling, ignored fetched trees |

No FortAI versus llama.cpp performance claim is valid until FortAI has a
model-level forward and decode path. Kernel microbenchmarks and model-level
token benchmarks must remain separate.
