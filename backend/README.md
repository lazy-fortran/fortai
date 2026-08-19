# Backend workspace

Backend-specific code is staged here as the backend matrix grows. The runtime
boundary is defined in `docs/architecture.md` and the first executable CPU
reference lives in `src/backend/cpu`.

The CUDA directory contains the first measured resident Q8 GEMV candidate,
the reusable resident-weight C ABI, and standalone correctness/performance
benchmarks. A Fortran `iso_c_binding` wrapper and a linked Fortran smoke test
exercise that ABI. An experimental Qwen3.5 model runner can route Q8 matvecs
through the ABI, but its host-controlled activation/output transfers are
measured and explicitly not promoted as the production CUDA backend.
The other directories identify planned lanes; an empty directory does not mean
that the corresponding backend is available.
