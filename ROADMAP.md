# Roadmap

The current delivery target is a CPU reference path plus a measured resident
CUDA model path, with native compilation, OpenMP parallelism, SIMD directives,
and reproducible benchmark tooling.

## CPU path

| ID | Work | Status | Acceptance |
| --- | --- | --- | --- |
| FAI-CPU-001 | Qwen3.5-0.8B text-only CPU reference | experimental | model opens, independent Q8 oracle passes, and the 8-step trace matches llama.cpp |
| FAI-CPU-001B | Qwen3.5-2B CPU scaling path | closed | model opens, runs, and benchmark metadata is recorded |
| FAI-CPU-001C | Qwen3.5-4B CPU scaling path | closed | model opens, runs, and benchmark metadata is recorded |
| FAI-CPU-002 | CPU logits parity and tokenizer path | closed | token-by-token logits agree with an independent oracle |
| FAI-CPU-003 | native CPU candidate tournament | in progress | compiler flags, thread count, and winner recorded |
| FAI-CPU-MOE-001 | CPU MoE execution for Qwen3.6-35B | deferred | expert routing, resident weights, and llama.cpp comparison |

The CPU implementation should use validated compiler features for the machine
under test, including `-march=native`, `-mtune=native`, OpenMP, SIMD, loop
transformation, and link-time optimization where supported. Optimization
levels and fast-math flags are tournament candidates and must pass the logits
oracle before performance measurement.

The 0.8B Q8_0 native-flags screen rejected the former `-O3 -ffast-math`
default (first reported over-tolerance error `2.30922698e-2`) and an `-O3`
no-fast-math candidate (first reported over-tolerance error
`1.16548542e-2`) against the `1.0e-2` gate. The
current `-O2 -march=native` default, with loop unrolling, LTO, no fast math,
and FP contraction disabled, passed all eight steps with maximum error
`2.431842038852494e-7`. FAI-CPU-003 still requires repeated isolated timing
evidence before a performance winner can be recorded.

The 2B and 4B CPU scaling acceptance runs were repeated at revision
`2b3b135f6fff19148b5613bca4fc294b89a25082` with the validated native flags,
Q8_0 fixtures, token `9419`, eight steps, context `128`, two OpenMP threads,
and `CUDA_VISIBLE_DEVICES=""`. The protected llama-server was not used. Both
results have matching tracked/worktree digest
`1dd461b76c6cf1f8f1daad793f0dcb40479e02eb0c1b8cb4a7b3084f7c06d162` and the
empty-patch digest
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`:

* 2B model SHA-256 `30d5d309e77d48b44325fe8eee08f4201f99687752f9a4a77fa66498457b45f4`,
  result `fortai_Qwen3.5-2B-Q8_0_20260823T143741Z.json` SHA-256
  `c6a3ecfac771ecbdb63c2b604c74c869eb02ba63de259c0ed10a89ad8ecc2cb5`,
  24 layers, eight steps, and `7.35294056` tokens/s.
* 4B model SHA-256 `c3fc7bcaf6f75b8f7ceeead9a769f5a7a9f86a8180af1cfb2b72958dcad8e028`,
  result `fortai_Qwen3.5-4B-Q8_0_20260823T143848Z.json` SHA-256
  `d305a5913442f218cd164fe76cd5c1e39ab587bde3c7470732b924167aa20bf3`,
  32 layers, eight steps, and `2.49143577` tokens/s.

These records close only the model-open/run/metadata claims. They do not
establish logits parity for the larger fixtures or a performance comparison.

```text
leaf_id: FAI-CPU-001B
leaf_status: PASS
claim_id: FAI-CPU-001B
claim_status: CLOSED
parent_id: FAI-CPU
parent_status: OPEN
evidence_gate_verdict: PASS
review_verdict: PASS
```

```text
leaf_id: FAI-CPU-001C
leaf_status: PASS
claim_id: FAI-CPU-001C
claim_status: CLOSED
parent_id: FAI-CPU
parent_status: OPEN
evidence_gate_verdict: PASS
review_verdict: PASS
```

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

The tracked trace and centered-logit verifiers match the 0.8B Q8_0 fixture for
the eight-token oracle run beginning at token 9419. The strict top-32 gate
measured a maximum centered-logit error of `2.431842038852494e-7` against its
`1.0e-2` tolerance. Current CPU evidence covers 0.8B, 2B, and 4B Q8_0
model-level runs with the server stopped and sequential execution.
The CUDA evidence now includes an experimental device-resident Qwen3.5 path
for the 0.8B, 2B, and 4B fixtures. The 0.8B eight-token CUDA trace matches
llama.cpp. CUDA Graph replay was profiled as an A/B and is slower on the tested
RTX 5060 Ti, so it is opt-in while the direct resident launch plan remains the
default. The paired 64-token result remains unpromoted until it matches the
competing harness. Nsight Compute is attempted and its permission result is
retained; Nsight Systems remains the fallback timing profile when GPU
performance-counter access is unavailable.

The immutable FAI-CPU-002 promotion evidence is revision
`efdcfed023e9878da4cea5d24b39e44f64d86bf6`, reviewed as base
`7f6dc94bdcc21f14856c62d4f4fd4bd8e7759238` plus patch SHA-256
`adeb61819468d805c3865c0bfaf7190cb1949720f4caebbe8c30aafa9a52d523`.
The verifier used Qwen3.5-0.8B Q8_0 model SHA-256
`091d8deba394f428b67aa42c100ee145fcbecac5a79621935d3655016ca737e5`,
llama.cpp b10566 executable SHA-256
`5b07556654335803a4a6f6e8f94b4bec13945e68843f28319bf69852176f935c`,
Ryzen 9 5950X, two OpenMP threads, context 128, and correctness flags
`-O2 -fopenmp -fno-fast-math -ffp-contract=off`. It requires the same top
token and top-32 token set at every step, with centered-logit error at most
`1.0e-2`; all eight steps passed. This closes only that named CPU gate, not
other prompts, contexts, quantizations, hardware paths, or performance claims.

```text
leaf_id: FAI-CPU-002
leaf_status: PASS
claim_id: FAI-CPU-002
claim_status: CLOSED
parent_id: FAI-CPU
parent_status: OPEN
evidence_gate_verdict: PASS
review_verdict: PASS
```
