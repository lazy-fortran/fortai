# Qwen3.8-27B production promotion

## Frozen configuration

- Hardware: two NVIDIA RTX 5060 Ti 16 GiB GPUs.
- Target: `Qwen3.8-27B-UD-Q4_K_XL.gguf`, SHA-256
  `3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e`.
- Draft: `mtp-Qwen3.8-27B-Q4_0.gguf`, native `draft-mtp`, depth 2.
- FortAI: context 262144, threads 14, tensor split `0.57,0.43`, batch
  2048, ubatch 256, FA on, fit off, target q8_0 K/V, draft q4_0 K/V.
- Sampling: temperature 0.6, top-p 0.95, top-k 20, min-p 0, seed 1234.
- Reference: llama.cpp `bb4caa7`. It cannot allocate the 262144-token
  production cache, so paired 3270/6470-token decode uses the smallest
  reference context that fits (4096/8192) while model, draft, active K/V,
  split, batching, and sampler remain identical.
- Production services remain stopped until every promotion gate passes.

## Promotion gates

| ID | Requirement | Lifecycle |
|---|---|---|
| Q27-CORRECT | Greedy MTP reproduces the non-speculative target; sampled selection, rollback, and graph reuse pass independent oracles. | `leaf_status: FAIL` (3/5 general prompts match; see Open defects); `claim_status: OPEN`; `parent_status: OPEN`; `review_verdict: PENDING`; `evidence_gate_verdict: PENDING` |
| Q27-PREFILL | Matched cold prompt ingestion is at least as fast as llama.cpp. | `leaf_status: FAIL` (holds below ~4k, 55% of llama.cpp at ~16k); `claim_status: OPEN`; `parent_status: OPEN`; `review_verdict: PENDING`; `evidence_gate_verdict: PENDING` |
| Q27-DECODE | Greedy and production-sampled MTP generation are at least as fast as llama.cpp at short and growing contexts. | `leaf_status: OPEN`; `claim_status: OPEN`; `parent_status: OPEN`; `review_verdict: PENDING`; `evidence_gate_verdict: FAIL` |
| Q27-MEMORY | Peak RAM and per-GPU VRAM are no greater than llama.cpp. | `leaf_status: PASS`; `claim_status: OPEN`; `parent_status: OPEN`; `review_verdict: PENDING`; `evidence_gate_verdict: PENDING` |
| Q27-FEATURES | Dual-GPU placement, FA, MTP, sampling, reasoning, prefix cache, and mmproj work through the native server. | `leaf_status: OPEN`; `claim_status: OPEN`; `parent_status: OPEN`; `review_verdict: PENDING`; `evidence_gate_verdict: PENDING` |
| Q27-LOAD | Model and service startup are at least as fast as llama.cpp. | `leaf_status: OPEN`; `claim_status: OPEN`; `parent_status: OPEN`; `review_verdict: PENDING`; `evidence_gate_verdict: PENDING` |
| Q27-PRODUCTION | All gates pass from one frozen commit and evidence bundle. | `leaf_status: BLOCKED`; `claim_status: OPEN`; `parent_status: OPEN`; `review_verdict: PENDING`; `evidence_gate_verdict: BLOCKED` |

## Correctness evidence

- Target verification consumes the complete shifted batch exactly once; the
  MTP sidecar receives output-normalized `h_nextn`, matching llama.cpp's
  Qwen3.5 process hook. Greedy MTP output at 3270 and 6470 tokens exactly
  matches FortAI's independent non-speculative target, including reasoning,
  answer `323`, and rejected-draft rollback.
- CUDA top-k returns only 20 `(token, logit)` pairs per row. A CPU
  `partial_sort` oracle passes exactly for 3 x 248320 randomized logits and
  deliberate equal-logit ties. Full-logit fallback remains for unsupported
  sampler shapes.
- Draft K/V uses native q4_0 with the GGML signed-extremum quantizer; target
  K/V uses q8_0. At context 262144 the resident allocation is about 15.2 GiB
  on GPU0 and 13.3 GiB on GPU1, while this llama.cpp build OOMs.
- The earlier broad sky prompt still yields a deterministic but different
  sampled stream from llama.cpp. The answer-level contract passes, but the
  broader sampled-equivalence claim remains open pending first-logit
  localization.

## Decode evidence (2026-08-29)

All rates exclude the stop token and use the same 6470-token arithmetic
payload. The paired long-context reference was rerun after the implementation
work on the same host and clocks.

| Context / mode | llama.cpp tok/s | FortAI tok/s | Status |
|---|---:|---:|---|
| short sampled arithmetic | 49.19 | 48.34 | within 1.7% noise; exact stream |
| short greedy arithmetic | 52.10 | 51.28 | within 1.6% noise; exact stream |
| 3270 sampled | 43.74 | 44.89 | FortAI 2.6% faster |
| 3270 greedy | 43.40 | 45.26 | FortAI 4.3% faster |
| 6470 greedy | 43.30 | 42.34 | FortAI 2.2% slower; noise band |
| 6470 sampled, paired | 44.87 | 40.73 | FortAI 9.2% slower; blocker |

The growing-context collapse had two independent causes:

1. The one-block-per-query-head Q8/Q4 attention kernels reloaded GQA K/V
   rows. The retained kernels share each packed K/V tile across six query
   heads, use online softmax, and partition long Q8 attention 24 ways. This
   raised the original 3270-token result from 23.50 to roughly 45 tok/s.
2. Verification graphs were keyed only by GPU segment. A graph captured below
   4096 tokens therefore replayed its four-partition launch above 4096 and
   reduced later long requests to 37--39 tok/s. Graph slots are now keyed by
   short/long attention regime for both batched verification and scalar
   decode. A deliberate short-then-long test stays at 41.8 tok/s.

The remaining sampled gap is not top-k correctness. Parallel CUB row selection
raised long sampled decode from 38.67 to 41.29 tok/s; the exact two-stage block
radix path is comparable. The 32-way attention experiment regressed to 40.14
tok/s and is removed; 24 partitions remains the measured optimum.

## Decode profile (2026-08-29)

A paired Nsight session on the warmed request (model load and graph
construction excluded) settles where the decode gap actually is, and it is not
where the previous slice assumed.

| Measure | FortAI | llama.cpp |
|---|---:|---:|
| GPU busy, both cards | 3680 ms | 3310 ms |
| Wall per speculative round | 55.3 ms | 56.1 ms |
| Tokens per round | 1.97 | 2.23 |
| MTP draft acceptance | 47.7% | 60.0% |

Per-round cost is already at parity: FortAI is marginally *faster* per round.
The entire throughput difference is that FortAI needs 13% more target
verification rounds for the same 128 tokens, because fewer of its drafts are
accepted. A greedy rerun over four prompts reproduces this without sampling
noise: FortAI accepts 277/440 (63.0%) at 41.47 tok/s, llama.cpp 286/421
(67.9%) at 43.39 tok/s. The 4.4% throughput gap equals the 4.2% tokens-per-round
gap. Both runtimes draft two tokens per round and use the identical acceptance
rule (sample the target, accept while it equals the draft), so the difference is
draft quality, not the acceptance test, the sampler, or scheduling.

Splitting FortAI's greedy trace by draft step gives 76.7% for step one and
60.6% for step two conditional on step one. Draft-one accuracy falls from 84%
after a fully accepted round to 60% after a rejection, but that correlation is
confounded -- rejections cluster at high-entropy positions where the next draft
is independently harder -- and is not on its own evidence of stale draft state.

Two structural differences from llama.cpp are recorded but are not the cause of
the gap:

- FortAI issues 543 `cudaStreamBeginCapture` + `cudaGraphInstantiate` pairs and
  1388 `cudaGraphLaunch` calls per 128-token request; llama.cpp captures once
  and issues 60 `cudaGraphExecUpdate` and 170 `cudaGraphLaunch` calls. The
  instantiate cost is 110 ms of 3593 ms (about 3%). Worth removing, but it
  cannot account for a gap that the round-cost measurement shows is absent.
- FortAI's greedy target stream diverges from llama.cpp's within about ten
  tokens on the same prompt, so the two runtimes' target logits already differ
  numerically. Draft proposals therefore cannot be diffed position-by-position
  without forcing an identical token sequence through both.

`Q27-DECODE` remains open. Closing it requires raising MTP draft acceptance to
llama.cpp's level, which needs a numerical comparison of the draft head's
logits against llama.cpp's for forced-identical inputs -- not further kernel,
sampler, or scheduling work, all of which the profile shows are already at or
past parity.

## Long-context scaling (2026-08-29)

The decode profile above used short prompts. A prompt-size sweep shows it does
not generalize: FortAI's advantage is confined to short contexts and inverts as
the prompt grows, while llama.cpp holds a flat prefill rate.

| Prompt tokens | Prefill FortAI / llama | Generation FortAI / llama | TTFT FortAI / llama |
|---:|---:|---:|---:|
| ~330 | 634 / 414 tok/s | 45.1 / 48.2 tok/s | 3.3 / 1.1 s |
| ~4 150 | 1196 / 1165 tok/s | 40.5 / 48.1 tok/s | 7.9 / 4.0 s |
| ~8 250 | 963 / 1231 tok/s | 32.8 / 46.7 tok/s | 15.6 / 7.0 s |
| ~16 450 | 659 / 1205 tok/s | 30.4 / 43.4 tok/s | 35.5 / 14.4 s |

llama.cpp's prefill is flat from 4k to 16k; FortAI's falls monotonically from
1196 to 659 tok/s, and generation from 40.5 to 30.4 tok/s. This is a larger
effect than the draft-acceptance gap and supersedes it as the priority. The
shape -- cost growing with resident context rather than with batch -- points at
attention over the grown K/V, not at projection throughput.

Batch graph slots are keyed by only two regimes,
`merge(2, 1, position_start + batch - 1 >= 4096)`, so one captured graph serves
every position above 4096. CUDA graph capture bakes scalar kernel arguments, so
any length-dependent scalar baked at capture time is reused unchanged from 4k
to 262k. That is the first thing to audit.

## Open defects

- `CUDA batched RMS norm failed` (HTTP 500) on long prompts. Reproduced once in
  three attempts on a 256/1024/4096/16384 ladder, never under
  `CUDA_LAUNCH_BLOCKING=1`; it is a race and the reported kernel is only where
  the sticky error surfaces. The dual-GPU host bridge in
  `fortai_cuda_q4_context_transfer` was audited and its ring-slot event
  ordering and `ensure_host_buffer` synchronization are sound, so the race is
  elsewhere. A reliable reproducer is required before any fix.
- Greedy MTP output matches the non-speculative scalar oracle on only 3 of 5
  general prompts; the two failures diverge inside the reasoning stream. The
  batched verification matmul and the scalar matvec reduce in different orders,
  so a near-tie can flip the greedy argmax. Not yet distinguished from a
  speculative defect -- needs a logit-level comparison at the first differing
  position. `Q27-CORRECT` was validated on an arithmetic payload and does not
  generalize.

## Next execution slice

1. Fix long-context scaling first: audit every scalar baked into a captured
   batch graph for length dependence, and key graph slots by the attention
   regime that actually changes launch geometry rather than by a single 4096
   boundary. This is the largest measured gap.
2. Build a reliable reproducer for the long-prompt race, then fix it.
3. Instrument both runtimes to dump `h_nextn` and draft-head logits for a
   forced-identical token sequence, and localize the first divergence. This is
   the remaining lever on short-context `Q27-DECODE`.
4. Replace per-step graph capture/instantiate with capture-once plus
   `cudaGraphExecUpdate`, matching llama.cpp. Expect roughly 3% of round cost;
   require the exact target and CPU top-k oracles before retention.
5. Restore an MTP-capable layout at reduced contexts. At 8192 the placement
   leaves a layer's projections split across both GPUs, so `batch_supported`
   rejects the verification batch and the server now degrades to the scalar
   oracle instead of failing the request. Matched-context decode cannot be
   compared until MTP runs there.
6. Localize the sky-stream first-logit difference, then run prefix reuse,
   reasoning-budget, mmproj, startup, and peak-memory gates from one build.
7. Remove superseded experiments, run the full `fo` and CUDA suite, obtain an
   independent review, and only then promote production.

## Retained / rejected

- Retain native q4_0 draft K/V, Q8 target K/V, grouped GQA Q8/Q4 attention,
  24-way long-context partitioning, regime-keyed CUDA graphs, compact exact
  CUDA top-k, dual-GPU batch pipeline, and device-resident MTP rollback.
- Do not reintroduce the packed Group8 kernel, single-thread top-k, projection
  padding, scheduler-stream unification, per-layer rejection replay, the
  faulting small-batch F16 active-view shortcut, or 32-way attention.

## Primary references

- llama.cpp Qwen3.5 graph and MTP contract:
  <https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp>
- llama.cpp CUDA flash-attention selectors:
  <https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn.cu>
- llama.cpp speculative implementation and documentation:
  <https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp>,
  <https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md>
- Exact speculative decoding: <https://arxiv.org/abs/2211.17192>
- CUDA graph execution model:
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs>
