# Architecture

FortAI is organized around a narrow runtime core and explicit backend
boundaries. The first implementation keeps each boundary small so that a model
path can be delivered before the complete backend matrix exists.

## Runtime layers

| Layer | Current responsibility |
| --- | --- |
| `fortai_tensor` | owned host tensor storage and shape metadata |
| `fortai_device` | stable device identity and backend names |
| `fortai_arena` | explicit byte accounting for planned memory ownership |
| `fortai_gguf` | GGUF header validation and metadata entry point |
| `fortai_model_ir` | model architecture and shape metadata |
| `fortai_plan_ir` | ordered execution steps and kernel names |
| `fortai_cache` | deterministic packed-weight cache keys |
| `fortai_sampler` | reference greedy token selection |
| `fortai_speculative` | accepted-prefix contract for draft verification |
| `fortai_backend_cpu` | independent CPU reference matvec |
| `backend/cuda` | resident Q8 GEMV, recurrent, attention/KV, FFN kernels, and reusable C ABI boundary |

The public `fortai` module exports the stable entry points. Model-specific
modules live below `src/models`, and device code lives below
`src/backend/<backend>` or behind a small C ABI shim when the native language
of the device requires it.

## Plan IR

A model loader will create `model_ir_t` from GGUF metadata. A model family will
lower that description into `plan_ir_t`. Backend specialization can then pick
weight layouts and kernel candidates without changing model semantics.

The execution plan records operation names, selected kernels, and verification
batch sizes. A later cache layer will persist the selected packed layout and
kernel ABI version for a particular model, device, and workload.

## Backend boundary

The initial CPU backend is a correctness reference. Future backends follow the
same boundary:

```text
Fortran runtime and plan
            |
            +-- backend ABI and residency management
            |
            +-- native kernels and vendor libraries
```

The backend owns launch geometry, residency, synchronization, and native
kernel candidates. The runtime owns model state, token sequencing, sampling,
and error propagation. Unsupported features return an explicit status.

## Delivery sequence

| Phase | Deliverable | Acceptance evidence |
| --- | --- | --- |
| 0 | tensor, GGUF, ModelIR, PlanIR, and CPU reference | build, independent tests, exact metadata checks |
| 1 | pure CPU model path | logits agree with an independent oracle |
| 2 | NVIDIA CUDA backend | resident kernel results agree with CPU and a reference runtime |
| 3 | mixed Q4/IQ GGML CPU/CUDA bridge, fused recurrent path, attention candidates, and CUDA Graph plan | independent mixed-quant oracle plus validated kernel tournament on named hardware |
| 4 | MTP verification | accepted-token rate and end-to-end throughput |
| 5 | DFlash2 integration | independent acceptance and latency measurements |
| 6 | MLX backend | Apple device correctness and memory evidence |
| 7 | direct Metal backend | measured comparison against MLX on named workloads |
| 8 | multimodal vision path | image and text parity tests |
| 9 | HIP backend | ROCm correctness and benchmark provenance |
| 10 | SYCL backend | Intel device correctness and benchmark provenance |
| 11 | Vulkan fallback | portable device correctness |
| 12 | TinyGPU NVIDIA on macOS | native integration with explicit external bridge |

The project will use the fastest validated candidate for a named model,
quantization, device, and workload class. Performance claims remain scoped to
those recorded conditions.
