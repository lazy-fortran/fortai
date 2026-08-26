# FortAI agent rules

## Native implementation requirement

- Do not wrap, embed, delegate to, or ship an existing inference library as a
  production implementation. In particular, do not use `libwhisper`,
  `whisper.cpp`, `llama.cpp`, or another external runtime as a FortAI model
  adapter.
- External projects may be inspected for algorithms, file-format details,
  independent correctness oracles, and performance comparisons only. Their
  production execution path must not be called by FortAI.
- Model semantics, state lifetime, scheduling, loading, sampling, and service
  behavior belong to FortAI and should be implemented in Fortran first. CUDA
  Fortran/ISO-C bindings and `.cu` kernels are allowed where the CUDA toolchain
  or measured performance requires them; low-level POSIX I/O, hardware/network
  interfaces, and CUDA driver/runtime code may remain C/CUDA. Non-Fortran code
  must not implement a model/runtime behind an adapter.
- Shared functionality must be implemented once in FortAI and reused by model
  families. Do not duplicate an external implementation merely to hide a
  wrapper behind a different API.

## Acceptance

- Every native model path needs an independent behavioral oracle.
- Promote a backend only after correctness, peak VRAM/RAM, and latency evidence
  on named hardware. The target is at least the independent reference speed
  and no greater peak memory for the same settings.
- Unsupported features fail explicitly; never silently delegate or fall back
  to another runtime.
