# Benchmarks

Benchmarks report workload shape, compiler, thread count, latency, and a
deterministic checksum. Results are meaningful only with the model,
quantization, device, compiler, and workload shape recorded beside them.

## CPU reference

Build and run the native CPU matvec benchmark:

```bash
tools/build_native.sh
benchmark/run_cpu_reference.sh 1024 1024 20
```

The output is stored under `benchmark/results`, which is ignored. The script
uses OpenMP parallel rows and an OpenMP SIMD reduction over each row. Native
flags are candidates for measurement, not the independent correctness oracle.
The default native flags are the strongest screened set that currently passes
the named eight-step 0.8B logits gate:

```text
-O2 -march=native -mtune=native -funroll-loops -fopenmp -fno-fast-math -ffp-contract=off -fno-math-errno -flto
```

Override candidates must pass that behavioral gate before their timing can
enter a tournament.

## llama.cpp

`run_llama_cpp.sh` starts a temporary llama.cpp server on a dedicated port,
runs one completion, stores the JSON response and server log, and shuts the
server down. It refuses to use a port that is already occupied. Set
`FORTAI_LLAMA_MODEL` to select a different GGUF file. The model-level CPU
runner and fair comparison wrapper are `run_qwen35_cpu.sh` and
`compare_qwen35_cpu_llama.sh`.

For the mixed-quant Qwen3.8-27B `UD-Q4_K_XL` format, the native FortAI CUDA
runner uses the exact GGML CUDA kernels for every tensor encoding and balances
active tensors across two visible GPUs. The native CPU path uses the GGML CPU
quantized matvec backend, sharing its thread pool and caching grouped projection
graphs. Build it with:

```bash
FORTAI_GGML_PREFIX=/home/ert/.local/llama.cpp-upstream-main-650913862 \
  tools/build_cuda_qwen35.sh
```

For an independent upstream oracle, use the isolated CUDA compatibility smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1 benchmark/run_q4_cuda_compat.sh \
  /mnt/storage/slopcode/models/unsloth_Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  9419 8 128
```

`UD-Q4_K_XL` is a mixed GGUF tensor format, not a single Q4 block type. The
compatibility runner delegates to the installed upstream llama.cpp CUDA
implementation as an independent correctness oracle, on private port `18380`
by default, and verifies cleanup before returning. It does not claim native
FortAI performance or enter the FortAI performance gate. Set
`FORTAI_GGML_CPU_CACHE=0` when host memory is too constrained for the bounded
CPU plan cache; correctness is unchanged.
The dynamic ABI adapter honors `FORTAI_GGML_LIBRARY` and
`FORTAI_GGML_CPU_LIBRARY` when the GGML installation is not in the machine
local default locations.

The comparison wrapper uses the same initial token ID, context, decode budget,
CPU thread count, and Q8_0 GGUF for both programs. It always terminates the
temporary server. Results are evidence, not promotion: a slower FortAI result
is explicitly retained as experimental.

llama.cpp's generation rate omits the prompt forward because its first
generated token is free, so the comparison runner evaluates the uncached
one-token prompt before the timed loop on both sides. `matched_forward` now
reports exactly the requested generated-token forwards after that prompt;
CPU repeat and tuning summaries use this generated-only rate.

The comparison is strictly CPU-only. The installed llama-server links
libggml-cuda, so `-ngl 0 --device none` alone still lets the CUDA backend
initialize; the wrapper therefore also exports `CUDA_VISIBLE_DEVICES=""` and,
when `nvidia-smi` is available, fails if the server appears as a GPU compute
process. The result JSON records `cuda_visible_devices`, the launcher and
actual process-image digests, the reported llama.cpp version, and every loaded
`libllama`/`libggml` path and digest. `LLAMA_LIBRARY_DIR` is optional; when it
is explicitly set, the comparison fails if the process loads those libraries
from another directory. The `llama_cpp_provenance` object describes the latest
machine-local fetched source checkout; the executable, version, and loaded
library fields identify the runtime used by a comparison. Shared-service runs
record `shared_service_conditions: true` and
`performance_gate_eligible: false`; cleanup refers only to the temporary CPU
server started by the wrapper.

## Resident CUDA kernel

The reusable CUDA ABI smoke check validates persistent weight upload, a
resident-device activation/output path, the host-controlled integration path,
and the linked Fortran binding. The C++ path is checked against the same
independent CPU oracle:

```bash
CUDA_DEVICE=0 benchmark/check_cuda_backend.sh
```

The CUDA comparator measures the resident Q8 GEMV operation against the
installed llama.cpp `ggml-cuda` implementation on identical deterministic
Q8 data. It is a kernel gate, not a complete Qwen model benchmark:

```bash
CUDA_DEVICE=0 benchmark/compare_cuda_q8.sh
CUDA_DEVICE=1 FORTAI_CUDA_ROWS=2048 FORTAI_CUDA_WIDTH=8192 \
  benchmark/compare_cuda_q8.sh
CUDA_DEVICE=0 FORTAI_CUDA_ITERATIONS=100 FORTAI_CUDA_WARMUP=20 \
  benchmark/profile_cuda_q8.sh
```

The profiler retains raw Nsight Compute diagnostics and Nsight Systems
reports. `ERR_NVGPUCTRPERM` is recorded when the host denies performance
counter access; it does not invalidate the CUDA-event timing or correctness
gate.

The experimental model-level CUDA slice can be built and compared with:

```bash
downloads=/mnt/storage/code/lazy-fortran/fortai/.provenance/downloads
OMP_NUM_THREADS=2 benchmark/compare_qwen35_cuda_llama.sh \
  "$downloads/qwen35-0.8b/Qwen3.5-0.8B-Q8_0.gguf" 9419 16 128 0
benchmark/profile_qwen35_cuda.sh \
  "$downloads/qwen35-0.8b/Qwen3.5-0.8B-Q8_0.gguf" 9419 16 128 0
benchmark/profile_qwen35_cuda_llama.sh \
  "$downloads/qwen35-0.8b/Qwen3.5-0.8B-Q8_0.gguf" 9419 16 128 0

# repeat the paired CUDA comparison and retain median/variance evidence
benchmark/repeat_compare_qwen35_cuda.sh \
  "$downloads/qwen35-0.8b/Qwen3.5-0.8B-Q8_0.gguf" 5 9419 64 128 0
```

The model path uploads Q8 weights once and keeps the token embedding lookup,
activations, recurrent state, attention KV caches, FFN projections, and final
logit projection device-resident. Only the logits needed by the current host
sampler and the next-token/control values cross the boundary. The path is
host-controlled at the model-call boundary and uses a resident CUDA Graph only
when explicitly enabled:

```bash
FORTAI_ENABLE_CUDA_GRAPH=1 benchmark/compare_qwen35_cuda_llama.sh \
  "$downloads/qwen35-0.8b/Qwen3.5-0.8B-Q8_0.gguf" 9419 64 128 0
```

On the tested RTX 5060 Ti, repeated graph and direct A/B runs are within normal
timing variance of each other and llama.cpp. Graph replay is therefore retained
as an opt-in optimization while the direct resident launch plan remains the
default. Results remain marked `not_promoted_cuda_graph_or_full_parity` because
this repository records effective parity, not an unsupported winner claim.

The `_llama` profiler stores paired Nsight Systems reports and API/kernel CSV
summaries for FortAI and llama.cpp under one provenance directory. The current
paired profile shows FortAI's standalone Q8/GDN/attention sequence versus
llama.cpp's fused `mul_mat_vec_q`, `gated_delta_net_cuda`, and graph plan; that
is the actionable CUDA gap.

The standalone recurrent-kernel harness uses an independent scalar CPU oracle,
checks both state and output, and compares a four-launch decomposition with a
one-launch fused recurrence:

```bash
FORTAI_CUDA_DEVICE=0 benchmark/run_cuda_gdn_ar_bench.sh
```

It records device-only CUDA-event timing, source/worktree digests, compiler,
driver, GPU identity, and fixture metadata under `benchmark/results`.

## Repeated runs, medians, and variance

Single measurements are not evidence. `repeat_compare_qwen35_cpu.sh` runs the
comparison wrapper N times (default 5) for one model and writes a summary JSON
with median, mean, sample standard deviation, min, max, and all raw values for
both FortAI and llama.cpp matched-forward throughput. It rejects mixed
revisions, flags, affinity, workload, model, or llama.cpp runtime provenance
and records whether the host conditions are performance-gate eligible.

```bash
downloads=/mnt/storage/code/lazy-fortran/fortai/.provenance/downloads
benchmark/repeat_compare_qwen35_cpu.sh "$downloads/qwen35-0.8b/Qwen3.5-0.8B-Q8_0.gguf" 5
benchmark/repeat_compare_qwen35_cpu.sh "$downloads/qwen35-2b/Qwen3.5-2B-Q8_0.gguf" 5
benchmark/repeat_compare_qwen35_cpu.sh "$downloads/qwen35-4b/Qwen3.5-4B-Q8_0.gguf" 5
```

Summaries land in `benchmark/results/repeat_<model>_<stamp>.json` (ignored,
like all results). Fix `OMP_NUM_THREADS`, steps, and context explicitly when
citing numbers; compare medians and report the spread beside them.

The promotion-grade CPU thread tournament combines those repeated summaries
and binds the selected FortAI thread to the independent centered-logit oracle:

```bash
benchmark/tournament_qwen35_cpu.sh \
  "$downloads/qwen35-0.8b/Qwen3.5-0.8B-Q8_0.gguf" 5 9419 64 128 8 32 1.0e-2
```

It screens every power-of-two thread level through 64 (`1 2 4 8 16 32 64`),
requires at least five isolated runs per thread, recomputes medians and
spreads, rejects mixed provenance or shared-service timing, and requires
FortAI's median matched-forward rate to be strictly higher than llama.cpp's at
every screened level. On a machine with fewer than 64 online CPUs, the 64
thread point is an explicit oversubscription measurement. The finalizer records
per-level speedups, separate FortAI/llama.cpp optima, and an explicit winner.
It also checks that the winning thread, binary, flags, revision, model, and
runtime match the eight-step top-32 logits oracle. The wrapper refuses any
unverified resident llama-server; the exact protected GPU service may remain
resident because the temporary CPU server uses a separately checked port.
Other shared-service results remain screening evidence and cannot enter the
performance gate.

For the independent greedy decode gate, run:

```bash
OMP_NUM_THREADS=4 benchmark/check_qwen35_cpu_trace.sh \
  "$downloads/qwen35-0.8b/Qwen3.5-0.8B-Q8_0.gguf" 9419 8 128
```

This asks llama.cpp for top-token IDs and compares them with FortAI's optional
per-step trace. The named 0.8B Q8_0 run currently passes, but matching greedy
token IDs alone does not establish numeric logits parity.

For a numeric CPU diagnostic, compare centered top logits against llama.cpp's
pre-sampling log probabilities. Centering removes the common log-softmax
normalization, so the reported values are directly comparable logits. The
eight-step form exercises recurrent and KV-cache state across decode:

```bash
OMP_NUM_THREADS=2 benchmark/check_qwen35_cpu_logits.sh \
  "$downloads/qwen35-0.8b/Qwen3.5-0.8B-Q8_0.gguf" 9419 8 128 32 1.0e-2
```

The checker builds FortAI without fast-math, requires the same top-token and
top-k token set, and reports the maximum centered-logit error together with
both model and llama.cpp executable digests. Set `steps` to 1 when isolating
the initial forward pass. The named eight-step 0.8B Q8_0 gate passes with a
maximum centered-logit error of `2.431842038852494e-7`.

The same centered-logit checker accepts native mixed Q4/IQ tensors. For the
2B Q4_K_M fixture, use a two-GPU visibility list and a private oracle port:

```bash
CUDA_VISIBLE_DEVICES=0,1 FORTAI_CUDA_Q4_SECOND_DEVICE=1 LLAMA_PORT=18395 \
  benchmark/check_qwen35_cuda_logits.sh \
  /mnt/storage/lazy-fortran-models/qwen3.5-2b-q4/qwen3.5-2b-q4_k_m.gguf \
  9419 8 128 4 0.5 0
```

This records the native `fortai-cuda-host-q4-xl-ggml` backend and compares it
with llama.cpp's two-GPU CUDA result; the tested run passed with maximum
centered-logit error `0.1808955686`.

For instruction-level CPU work, `profile_qwen35_cpu_both.sh` records the
matched FortAI/llama.cpp `perf stat` counters (including cycles and retired
instructions), zero-loss call graphs, and Intel-syntax disassembly for both
the FortAI executable and llama.cpp's `libggml-cpu`. The profiler accepts the
same verified protected GPU resident service as the CPU comparison wrapper and
defaults to the b10566 executable/library pair used by the current oracle. On
hosts that restrict perf attach or record access, set
`FORTAI_PERF_USE_SUDO=1`; this uses `sudo -n perf`, records the runner in
provenance, and restores artifact ownership to the invoking user. Kernel
symbol warnings caused by `kernel.kptr_restrict` do not affect user-space
FortAI/llama.cpp assembly samples.

## Promotion rule

FortAI is promoted for a named model, quantization, device, context, batch, and
workload only when its validated candidate matches or beats the fastest fair
competitor measurement. A slower result is retained as evidence and remains
experimental. Kernel measurements do not substitute for model-level token
throughput.
