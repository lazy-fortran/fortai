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
`FORTAI_LLAMA_MODEL` to select a different GGUF file.

The current repository has no model-level FortAI forward or decode path, so a
token-throughput comparison is deliberately deferred. The roadmap records the
matching conditions required before comparing FortAI with llama.cpp.

## Promotion rule

FortAI is promoted for a named model, quantization, device, context, batch, and
workload only when its validated candidate matches or beats the fastest fair
competitor measurement. A slower result is retained as evidence and remains
experimental. Kernel measurements do not substitute for model-level token
throughput.
