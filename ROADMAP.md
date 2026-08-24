# Roadmap

The current delivery target is a CPU reference path plus a measured resident
CUDA model path, with native compilation, OpenMP parallelism, SIMD directives,
and reproducible benchmark tooling.

## CPU path

| ID | Work | Status | Acceptance |
| --- | --- | --- | --- |
| FAI-CPU-001 | Qwen3.5-0.8B text-only CPU reference | closed | model opens, independent Q8 oracle passes, and the 8-step trace matches llama.cpp |
| FAI-CPU-001B | Qwen3.5-2B CPU scaling path | closed | model opens, runs, and benchmark metadata is recorded |
| FAI-CPU-001C | Qwen3.5-4B CPU scaling path | closed | model opens, runs, and benchmark metadata is recorded |
| FAI-CPU-002 | CPU logits parity and tokenizer path | closed | token-by-token logits agree with an independent oracle |
| FAI-CPU-003 | native CPU candidate tournament | closed | independent three-model review confirms isolated repeated medians, bound logits oracles, and strict FortAI speedup over llama.cpp at every power-of-two thread level through 64 |
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
`2.431842038852494e-7`. The candidate was then checked by the full isolated
three-model thread tournament described below.

The current tournament-readiness implementation anchor is revision
`564bd69a5f4bd3751be24175a6b6afd9fe432536` (building on
`93b32de46fe3d5bc27a819b393b4235e4c4131d0`). It refuses any unverified resident `llama-server` before starting
timing, while allowing only the exact protected GPU service (PID `268006`,
port `8080`) as independent from the CPU measurements. It requires at least five repeats for
each thread candidate, restricts the tournament and direct-repeat wrappers
and finalizer to the exact 0.8B/2B/4B Q8_0 model allowlist, binds the selected
median to the exact
oracle step/top-k/tolerance contract, requires CUDA to remain hidden, validates
the matched-forward metric, positive timing steps, positive token/context
workload dimensions, at least two distinct thread candidates, and SHA-256
freezes of all summary/oracle inputs from single-read snapshots, and checks
the independent centered-logit oracle. It requires the full power-of-two thread
set `1 2 4 8 16 32 64` and rejects any level where FortAI is not strictly
faster than llama.cpp by matched-forward median; the 64-thread point is
explicitly oversubscribed on this machine's 32 online CPUs. It writes the final
lifecycle record
through a same-directory fsync-and-replace sequence so a partial output cannot
be promoted.
A successful finalizer also emits exact FAI-CPU-003/FAI-CPU lifecycle IDs and
explicit open/pending review axes without self-promoting the claim. The
three-model tournament artifacts are now recorded below; an independent Luna
review confirmed every artifact before the lifecycle promotion below.

The binding hardening in revision `a8cd84e19ddd04a6e4026bde13417e4c2dc004ba`
now validates the rank and configured dimensions of the global, FFN,
recurrent/SSM, and full-attention tensors before any Qwen3.5 layer is
allocated. The permitted 4B fixture opens and executes one CPU step through
these checks; this is fail-fast diagnostic hardening, not a logits-parity
fix.

The recurrent head-broadcast fix in revision
`a8cd84e19ddd04a6e4026bde13417e4c2dc004ba` changed the 4B strict centered-
top-32 check from a step-0 failure (`FortAI=314`, `llama.cpp=11`) to PASS.
Fresh eight-step checks now pass for all three permitted fixtures under the
same token `9419`, context `128`, top-32, and `1.0e-2` tolerance contract:

* 0.8B: maximum centered-logit error `2.431842038852494e-7`, result
  `compare_Qwen3.5-0.8B-Q8_0_20260823T163558Z.json`, SHA-256
  `97ab09c7d8a3f79d66a78dd1098903e4de5b4f910f5a7da96b89aee78216088d`.
* 2B: maximum centered-logit error `3.5649993890274345e-7`, result
  `compare_Qwen3.5-2B-Q8_0_20260823T163524Z.json`, SHA-256
  `a9d6b1e30da5dac091d80ca364485171d013a482af4712526927f2f4fdfb3703`.
* 4B: maximum centered-logit error `2.434161379127886e-7`, result
  `compare_Qwen3.5-4B-Q8_0_20260823T163416Z.json`, SHA-256
  `b187576b580bb0001f2e4996d4e37bf029b867cf13c2a74f525dde663d388843`.

All three are explicitly shared-service correctness-only evidence because the
protected server remains resident; they do not establish any performance
claim or change the FAI-CPU-003 lifecycle.

The same strict centered-top-32 oracle was revalidated at the current
`1496c239cce14687e4f2b03f72714785c540a4bb` with isolated temporary CPU
servers, CUDA hidden, empty patch digests, and verified cleanup. The maximum
centered-logit errors were `2.431842038852494e-7` (0.8B),
`3.5649993890274345e-7` (2B), and `2.434161379127886e-7` (4B), all below the
`1.0e-2` tolerance. These current runs close correctness evidence for the
three permitted small models; they do not close the strict 1–64-thread speed
gate.

A bounded Luna scaling audit on the 0.8B fixture found no safe scheduling-only
promotion. With 64 forwards at eight threads, `spread/cores` measured
`27.4207` tokens/s, `close/cores` `27.4560`, and `master/cores` `27.6817`; a
temporary `schedule(dynamic,32)` variant was not reproducibly better than
static scheduling at four or eight threads. The host reports GNU OpenMP's
default `OMP_WAIT_POLICY=PASSIVE`, so workers sleep between matvec regions
while llama.cpp keeps a hot worker pool. A persistent worker region or broader
forward fusion is a materially larger redesign and remains unpromoted until it
passes the independent logits oracle and the full seven-level tournament.

Revision `93b32de46fe3d5bc27a819b393b4235e4c4131d0` adds an opt-in
`FORTAI_ENABLE_PERSISTENT_OPENMP=1` worker region for the permitted Qwen3.5
Q8_0 CPU model shapes.
It keeps the default path unchanged, records the mode in comparison, repeat,
tournament, oracle, and perf-profile provenance, and passes the isolated
centered-top-32 oracle (maximum error `2.431842038852494e-7`). A fresh
seven-level, five-repeat 0.8B run with the mode enabled passed the strict
median comparison at 1, 2, 16, 32, and 64 threads but failed at 4 and 8
threads in that initial spread/cores run (the finalizer rejected that
historical candidate); no FAI-CPU-003 performance promotion was claimed from
it. The clean paired profile
`benchmark/profiles/Qwen3.5-0.8B-Q8_0_both_20260823T180005Z` used
`sudo -n perf`, captured zero lost samples, and recorded `q8_dot_avx2` at
`72.07%` of FortAI samples. The corresponding llama.cpp profile recorded
`84.06%` in `ggml_vec_dot_q8_0_q8_0`; host perf counters are usable. This
profile predates the unified three-model tournament and is retained as the
low-level instruction/cycle evidence.

At revision `564bd69a5f4bd3751be24175a6b6afd9fe432536`, the persistent mode
has promotion-grade artifacts for all three permitted small models. Every run
used `master/cores`, passive OpenMP waits, hidden CUDA, an isolated temporary
CPU llama.cpp server, five repeats at each required level, and verified
cleanup. Each artifact passes the strict median comparison at
`1 2 4 8 16 32 64` threads and binds to an isolated centered-top-32 oracle:

* 0.8B: [tournament](</mnt/storage/code/lazy-fortran/fortai/benchmark/results/tournament_Qwen3.5-0.8B-Q8_0_20260823T201011Z.json>), oracle maximum error `2.431842038852494e-7`, artifact SHA-256 `282cfff34ac9908791c6ae524e1dfd846f58c2eaf92aa46f21d91d895b971c5b`.
* 2B: [tournament](</mnt/storage/code/lazy-fortran/fortai/benchmark/results/tournament_Qwen3.5-2B-Q8_0_20260823T192349Z.json>), oracle maximum error `3.5649993890274345e-7`, artifact SHA-256 `87310e3c9d67f7568413eb2e7eb2573308d6965c51a3c52e7c0c10c7206fbf15`.
* 4B: [tournament](</mnt/storage/code/lazy-fortran/fortai/benchmark/results/tournament_Qwen3.5-4B-Q8_0_20260823T192834Z.json>), oracle maximum error `2.434161379127886e-7`, artifact SHA-256 `3d86c9a8a98cf17e31f758d2eb8470e9d049e16bc995df23b4ddc6043fa78f10`.

FortAI is strictly faster by median at every level for all three models. The
64-thread points are explicit oversubscription measurements on 32 online CPUs;
the tournament artifacts record the full per-level speedups and provenance.

The low-level CPU path is not missing an assembly implementation. The tracked
`src/backend/cpu/fortai_q8_dot.c` contains AVX2/F16C/FMA kernels for Q8 dot,
activation quantization, dequantization, SiLU, SiLU-product, and GDN steps;
the historical paired profile at
`benchmark/profiles/Qwen3.5-0.8B-Q8_0_both_20260819T212359Z` captured Intel-
syntax disassembly and retired counters. At two threads that profile reported
FortAI `22.970B` instructions / `14.080B` cycles with `91.70%` of samples in
`q8_dot_avx2`, versus llama.cpp `30.575B` instructions / `19.047B` cycles
with `77.88%` in `ggml_vec_dot_q8_0_q8_0`; both were about `1.6` IPC. A
fresh `llvm-mca -mcpu=znver3` analysis of the AVX2 Q8 inner block estimates a
three-cycle backend throughput before cache and memory effects. This makes
the remaining optimization target a scheduling/parallel graph problem, not a
missing scalar-to-assembly conversion. The current-runtime profiles and the
three-model tournament provide the required model-level evidence; no
theoretical optimum claim is made beyond the measured result. A
current paired capture at
`benchmark/profiles/Qwen3.5-0.8B-Q8_0_both_20260823T171834Z` (commit
`a96f153275ee3da8dbb0dc376352058fff623fe3`, `sudo -n perf`, zero lost
samples) puts `q8_dot_avx2` at `83.10%` of FortAI samples and
`ggml_vec_dot_q8_0_q8_0` at `78.07%` of llama.cpp samples. Its host was under
unrelated load, so it is assembly evidence rather than a promotion-grade
throughput result; the strict tournament remains the independent gate. The
current SIMD dispatch also checks FMA before entering the FMA-targeted Q8 dot
kernel, while the dequantization kernel no longer advertises an unnecessary
FMA requirement.

```text
leaf_id: FAI-CPU-003
leaf_status: PASS
claim_id: FAI-CPU-003
claim_status: CLOSED
parent_id: FAI-CPU
parent_status: OPEN
evidence_gate_verdict: PASS
review_verdict: PASS
```

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

Fresh CPU model-level runs at current revision
`58dcfac82775216965e4c36d0e39bc32f8cc99d6` used the same validated native
flags, token `9419`, eight steps, context `128`, two OpenMP threads, and
`CUDA_VISIBLE_DEVICES=""`; the protected llama-server was not used. They
share tracked/worktree digest
`65a54e19cafc5a56a8d80c9e8ae5ef8e3acda67a30db2684625bcb62b974b351` and the
empty-patch digest
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`:

* 0.8B model SHA-256 `091d8deba394f428b67aa42c100ee145fcbecac5a79621935d3655016ca737e5`,
  result `fortai_Qwen3.5-0.8B-Q8_0_20260823T151903Z.json` SHA-256
  `0125d816d7c352b44f0802658379c3c76fa817b3e7dc636197be9d0890369d1f`,
  24 layers, and `22.4719105` tokens/s.
* 2B model SHA-256 `30d5d309e77d48b44325fe8eee08f4201f99687752f9a4a77fa66498457b45f4`,
  result `fortai_Qwen3.5-2B-Q8_0_20260823T151739Z.json` SHA-256
  `5a6bab31b8b35c86090b4e98ee20cea15df3b46b0e5a7540be05f65dd04fddd1`,
  24 layers, and `10.1265821` tokens/s.
* 4B model SHA-256 `c3fc7bcaf6f75b8f7ceeead9a769f5a7a9f86a8180af1cfb2b72958dcad8e028`,
  result `fortai_Qwen3.5-4B-Q8_0_20260823T151808Z.json` SHA-256
  `bc62ecc91c3f35c2aa712c158a3f954395e30f8e38a006ec1ea4fb6dc943653d`,
  32 layers, and `3.76825261` tokens/s.

These freshness runs confirm the permitted model-open/run path only; they do
not establish logits parity for the larger fixtures or a fair performance
comparison.

A fresh strict 0.8B centered-top-32 logits gate at revision
`71e5df9d65ddaaf2adf6cba2b732aefcc15278c4` also passed with maximum centered
logit error `2.431842038852494e-7` against tolerance `1.0e-2`. The comparison
result `compare_Qwen3.5-0.8B-Q8_0_20260823T152028Z.json` has SHA-256
`6bc6815da8635374e55dfb0ba633729609a31a090c5fcd7c3e94b4bdec46c52e`, model
SHA-256 `091d8deba394f428b67aa42c100ee145fcbecac5a79621935d3655016ca737e5`,
and verified temporary-server cleanup. Since protected PID `268006` was
resident, this result is explicitly `measurement_conditions=shared_service`
and `performance_gate_eligible=false`; it is correctness evidence only.

The 0.8B text-only acceptance was rerun at revision
`3dd5d011c8fde305f284ba653eee4943669434e9` with Q8_0, token `9419`, eight
steps, context `128`, two OpenMP threads, and the independent llama.cpp b10566
oracle. The trace result
`compare_Qwen3.5-0.8B-Q8_0_20260823T144318Z.json` has SHA-256
`dcc2c826c61ad4587deaa04e85bab83203d8a40be6584248251559cb535d4573` and
matched token IDs `[11, 271, 40, 1044, 3133, 440, 264, 12654]`. The centered
top-32 result
`compare_Qwen3.5-0.8B-Q8_0_20260823T144301Z.json` has SHA-256
`17acddc6075cfc7b95163a69c2ff2c57dd60c25f338a49032a09343caf94841f` and
maximum centered-logit error `2.431842038852494e-7` against tolerance
`1.0e-2`. Both results identify model SHA-256
`091d8deba394f428b67aa42c100ee145fcbecac5a79621935d3655016ca737e5`, llama
executable SHA-256
`5b07556654335803a4a6f6e8f94b4bec13945e68843f28319bf69852176f935c`, and
the empty-patch digest
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
The shared-service condition is correctness-only evidence; it is not eligible
for a performance claim.

```text
leaf_id: FAI-CPU-001
leaf_status: PASS
claim_id: FAI-CPU-001
claim_status: CLOSED
parent_id: FAI-CPU
parent_status: OPEN
evidence_gate_verdict: PASS
review_verdict: PASS
```

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

The llama.cpp model harness acceptance is closed for the named 0.8B
correctness tuple. The paired artifacts above were produced from revision
`3dd5d011c8fde305f284ba653eee4943669434e9` with model SHA-256
`091d8deba394f428b67aa42c100ee145fcbecac5a79621935d3655016ca737e5`, token
`9419`, context `128`, and eight requested/generated steps. Each result
records `prompt_n=1`, `cache_n=0`, `predicted_n=8`, and
`llama_server_cleanup=verified`; the trace and centered-top-32 checks passed.
The runs are marked shared-service and performance-ineligible, so this closes
the harness contract only and makes no throughput or winner claim.

```text
leaf_id: FAI-BENCH-002
leaf_status: PASS
claim_id: FAI-BENCH-002
claim_status: CLOSED
parent_id: FAI-BENCH
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
| FAI-BENCH-001 | persistent CPU reference benchmark | closed | native fo build, content digest, and result with provenance |
| FAI-BENCH-002 | llama.cpp model benchmark harness | closed | same model, token count, context, output budget, cleanup, and trace gate |
| FAI-PROV-001 | source fetch and revision records | closed | tracked fetch tooling, ignored fetched trees |

No FortAI versus llama.cpp performance claim is valid until FortAI has a
model-level forward and decode path. Kernel microbenchmarks and model-level
token benchmarks must remain separate.

The persistent CPU reference benchmark acceptance is closed at revision
`b6bc652d17d20d8409bb7adfc761a972fb436fa8`. The fixed `256x256x20` native
`fo` run produced `cpu_matvec_20260823T145028Z.csv` with SHA-256
`b447c99cc26a8d14750d63535ec194b6f634d20db8d826469d428215105c3427`.
It records the empty-patch digest, tracked/worktree digest
`002d1794678330b495a49804520a9b9d1ecbfad5573409a9136263f49295bad2`,
validated native flags, compiler, OpenMP affinity, executable SHA-256
`ba00e1fb7b97ca3a2c86def8d54fd082af4fc183a66028d7cb43a11d2a5c5dd6`,
positive timing, and deterministic checksum. An independent recomputation
gave checksum `2364.133735294118` versus reported `2364.13374000` (relative
error `1.991e-9`). This is a kernel/reference benchmark record, not a model
performance comparison.

```text
leaf_id: FAI-BENCH-001
leaf_status: PASS
claim_id: FAI-BENCH-001
claim_status: CLOSED
parent_id: FAI-BENCH
parent_status: OPEN
evidence_gate_verdict: PASS
review_verdict: PASS
```

The tracked trace and centered-logit verifiers match the 0.8B Q8_0 fixture for
the eight-token oracle run beginning at token 9419. The strict top-32 gate
measured a maximum centered-logit error of `2.431842038852494e-7` against its
`1.0e-2` tolerance. Current CPU evidence covers 0.8B, 2B, and 4B Q8_0
model-level runs with sequential FortAI execution. The 0.8B paired oracle
run is explicitly shared-service correctness evidence and performance-
ineligible; the 2B and 4B scaling runs do not start a llama-server.
The CUDA evidence now includes an experimental device-resident Qwen3.5 path
for the 0.8B, 2B, and 4B fixtures. The 0.8B eight-token CUDA trace matches
llama.cpp. CUDA Graph replay was profiled as an A/B and is slower on the tested
RTX 5060 Ti, so it is opt-in while the direct resident launch plan remains the
default. The paired 64-token result remains unpromoted until it matches the
competing harness. Nsight Compute is attempted and its permission result is
retained; Nsight Systems remains the fallback timing profile when GPU
performance-counter access is unavailable.

The source/model provenance acceptance is closed at revision
`0fe922dbc7dd02240262e40dcede203ab44905cf`. An end-to-end local source clone
resolved the current commit into a revision record; source and model dry-runs
parsed all seven and three manifest entries respectively. The fetched source,
downloaded models, and revision records are all ignored by Git, while the
fetch scripts and manifests remain tracked. This verifies provenance behavior
without fetching or committing a model artifact.

```text
leaf_id: FAI-PROV-001
leaf_status: PASS
claim_id: FAI-PROV-001
claim_status: CLOSED
parent_id: FAI-PROV
parent_status: OPEN
evidence_gate_verdict: PASS
review_verdict: PASS
```

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
