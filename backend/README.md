# Backend workspace

Backend-specific code is staged here as the backend matrix grows. The runtime
boundary is defined in `docs/architecture.md` and the first executable CPU
reference lives in `src/backend/cpu`.

The CUDA directory contains the first measured resident Q8 GEMV candidate and
its standalone benchmark. It is not yet the complete Qwen execution backend.
The other directories identify planned lanes; an empty directory does not mean
that the corresponding backend is available.
