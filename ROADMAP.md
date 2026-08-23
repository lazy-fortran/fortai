# Roadmap

The current delivery target is a CPU reference path plus a measured resident
CUDA model path, with native compilation, OpenMP parallelism, SIMD directives,
and reproducible benchmark tooling.

## CPU path

| ID | Work | Status | Acceptance |
| --- | --- | --- | --- |
| FAI-CPU-001 | Qwen3.5-0.8B text-only CPU reference | experimental | model opens, independent Q8 oracle passes, and the 8-step trace matches llama.cpp |
| FAI-CPU-001B | Qwen3.5-2B CPU scaling path | experimental | model opens, runs, and benchmark metadata is recorded |
| FAI-CPU-001C | Qwen3.5-4B CPU scaling path | experimental | model opens, runs, and benchmark metadata is recorded |
| FAI-CPU-002 | CPU logits parity and tokenizer path | open | token-by-token logits agree with an independent oracle |
| FAI-CPU-003 | native CPU candidate tournament | in progress | compiler flags, thread count, and winner recorded |
| FAI-CPU-MOE-001 | CPU MoE execution for Qwen3.6-35B | deferred | expert routing, resident weights, and llama.cpp comparison |

The CPU implementation should use the available compiler features for the
machine under test, including `-O3`, `-march=native`, `-mtune=native`, OpenMP,
SIMD, loop transformation, and link-time optimization where supported. Fast
math flags belong to measured candidate builds, not the correctness oracle.

## GPU path

| ID | Work | Status | Acceptance |
| --- | --- | --- | --- |
| FAI-CUDA-001 | single-device CUDA backend | experimental, device-resident Qwen path | independent ABI oracle, model trace parity, and paired llama.cpp benchmark |
| FAI-CUDA-001A | host-controlled Qwen3.5 CUDA integration slice | experimental, measured slower | weights resident, model correctness smoke, persistent transfer/launch profile; cannot be promoted |
| FAI-CUDA-001B | device-resident Qwen3.5 recurrent, attention, KV, and FFN path | experimental, measured slower | 0.8B/2B/4B smoke, CUDA trace parity, and fewer host activation round trips |
| FAI-CUDA-001C | CUDA Graph capture and launch-plan replay | experimental, opt-in and slower on RTX 5060 Ti | matched Nsight profile and end-to-end speed at least equal to the direct resident plan and llama.cpp |
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
| FAI-BENCH-001 | persistent CPU reference benchmark | in progress | native fo build, content digest, and result with provenance |
| FAI-BENCH-002 | llama.cpp model benchmark harness | in progress | same model, token count, context, output budget, cleanup, and trace gate |
| FAI-PROV-001 | source fetch and revision records | in progress | tracked fetch tooling, ignored fetched trees |

No FortAI versus llama.cpp performance claim is valid until FortAI has a
model-level forward and decode path. Kernel microbenchmarks and model-level
token benchmarks must remain separate.

The tracked trace verifier matches the 0.8B Q8_0 fixture for the eight-token
oracle run beginning at token 9419. The centered-logit verifier matches the
initial 0.8B forward pass, but its multi-step run exposes a later state
divergence, so FAI-CPU-002 remains open. Current CPU evidence covers 0.8B, 2B,
and 4B Q8_0 model-level runs with the server stopped and sequential execution.
The CUDA evidence now includes an experimental device-resident Qwen3.5 path
for the 0.8B, 2B, and 4B fixtures. The 0.8B eight-token CUDA trace matches
llama.cpp. CUDA Graph replay was profiled as an A/B and is slower on the tested
RTX 5060 Ti, so it is opt-in while the direct resident launch plan remains the
default. The paired 64-token result remains unpromoted until it matches the
competing harness. Nsight Compute is attempted and its permission result is
retained; Nsight Systems remains the fallback timing profile when GPU
performance-counter access is unavailable.
