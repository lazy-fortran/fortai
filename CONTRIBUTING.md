# Contributing

FortAI is a small project with a performance-sensitive core. Contributions
should keep the public API explicit and leave unrelated cleanup for a separate
change.

## Local checks

Run the complete local workflow before opening a pull request:

```bash
fo
```

At minimum, run:

```bash
fpm build
fpm test
```

Tests should compare the implementation with an independently constructed
behavioral oracle. A test that only checks an internal field or reproduces the
implementation is not sufficient.

## Backend contributions

Every backend candidate must document:

- device identity and driver version
- compiler and backend library versions
- model architecture, quantization, and tensor shapes
- transfer-inclusive and resident timings
- memory use
- numerical error against the CPU reference

The runtime must return an explicit unsupported status when a backend cannot
execute a requested operation. Silent host fallback obscures performance and
correctness evidence.

Production promotion requires a fair competitor measurement. Record the exact
model, quantization, device, context, batch, prompt workload, compiler, and
runtime versions. A candidate that is slower than the fastest reproducible
competitor remains experimental.

## Provenance

Original FortAI code is MIT licensed. Copied code and generated bindings need a
source, license, and version record before they enter the repository. The
`oracles` and `kernels` directories hold references and contracts, not a place
to copy external implementations without an audit.
