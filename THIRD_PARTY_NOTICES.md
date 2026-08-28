# Third-party provenance

FortAI uses the GGML CPU and CUDA backends as low-level quantization and
device primitives.  The native Fortran model, scheduler, state management,
sampling, and HTTP service remain FortAI code; FortAI does not call the
llama.cpp inference runtime or ship a llama.cpp wrapper as its production
server.

The CUDA Q4 path was checked against the local llama.cpp source tree at
commit `3b35a33ee43d7c238f352d4ca8a21d82f1f6115d` (2026-04-24).  The following
upstream techniques are used with independent FortAI implementations:

- scheduler-owned streams and event edges for split-device work;
- cached CUDA graph capture/replay with capture-safe stream selection;
- quantized MMVQ/MMQ tiling and packed activation layouts; and
- backend-side greedy `argmax`, so a decode transfers a token ID rather than
  a full vocabulary row.

No upstream source was copied into the native kernels.  The upstream GGML/
llama.cpp project is distributed under the MIT license:

```text
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
