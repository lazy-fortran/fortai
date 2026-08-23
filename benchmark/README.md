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

The comparison wrapper uses the same initial token ID, context, decode budget,
CPU thread count, and Q8_0 GGUF for both programs. It always terminates the
temporary server. Results are evidence, not promotion: a slower FortAI result
is explicitly retained as experimental.

FortAI times every requested model forward. llama.cpp's generation rate omits
the prompt forward because its first generated token is free, so that raw rate
is not directly comparable. The result therefore validates an uncached
one-token prompt and records `matched_forward` over the prompt forward plus
the remaining generated-token forwards. CPU repeat and tuning summaries use
only this matched rate.

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

On the tested RTX 5060 Ti, the graph A/B is retained as evidence but is slower
than the direct resident launch plan, so graph replay is disabled by default.
Its results remain marked `not_promoted_cuda_graph_or_full_parity` until the
complete model-level gate reaches the competing harness.

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
It also checks that the winning thread, binary,
flags, revision, model, and runtime match the eight-step top-32 logits oracle.
The wrapper refuses to run while any llama-server is resident; shared-service
results remain screening evidence and cannot enter the performance gate.

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

## Promotion rule

FortAI is promoted for a named model, quantization, device, context, batch, and
workload only when its validated candidate matches or beats the fastest fair
competitor measurement. A slower result is retained as evidence and remains
experimental. Kernel measurements do not substitute for model-level token
throughput.
