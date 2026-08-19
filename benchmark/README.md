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

The comparison is strictly CPU-only. The installed llama-server links
libggml-cuda, so `-ngl 0 --device none` alone still lets the CUDA backend
initialize; the wrapper therefore also exports `CUDA_VISIBLE_DEVICES=""` and,
when `nvidia-smi` is available, fails if the server appears as a GPU compute
process. The result JSON records `cuda_visible_devices`.

## Repeated runs, medians, and variance

Single measurements are not evidence. `repeat_compare_qwen35_cpu.sh` runs the
comparison wrapper N times (default 5) for one model and writes a summary JSON
with median, mean, sample standard deviation, min, max, and all raw values for
both FortAI and llama.cpp decode throughput, plus the commit, compiler, thread
count, model path + SHA-256, token/steps/context, and server-cleanup checks.

```bash
downloads=/mnt/storage/code/lazy-fortran/fortai/.provenance/downloads
benchmark/repeat_compare_qwen35_cpu.sh "$downloads/qwen35-0.8b/Qwen3.5-0.8B-Q8_0.gguf" 5
benchmark/repeat_compare_qwen35_cpu.sh "$downloads/qwen35-2b/Qwen3.5-2B-Q8_0.gguf" 5
benchmark/repeat_compare_qwen35_cpu.sh "$downloads/qwen35-4b/Qwen3.5-4B-Q8_0.gguf" 5
```

Summaries land in `benchmark/results/repeat_<model>_<stamp>.json` (ignored,
like all results). Fix `OMP_NUM_THREADS`, steps, and context explicitly when
citing numbers; compare medians and report the spread beside them.

## Promotion rule

FortAI is promoted for a named model, quantization, device, context, batch, and
workload only when its validated candidate matches or beats the fastest fair
competitor measurement. A slower result is retained as evidence and remains
experimental. Kernel measurements do not substitute for model-level token
throughput.
