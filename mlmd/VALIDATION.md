# Validation record — 2026-10-14

## Executed

Host: Windows x86_64. Isolated official Rust 1.99.0
(b940084d7, stable-x86_64-pc-windows-gnu); no global PATH changes.

- `cargo check --manifest-path mlmd/Cargo.toml --workspace`: passed.
- `cargo test --manifest-path mlmd/Cargo.toml --workspace`: 8 passed, 0 failed.
  Six FFI contract tests and two shared-contract tests.
- `cargo fmt --manifest-path mlmd/Cargo.toml --all`: completed.
- `cargo build --manifest-path mlmd/Cargo.toml -p omnimd-deepmd --locked`: passed.
- The contract/API tests use synthetic Rust functions with C calling convention.
  They check repeated evaluations on one handle, one combined compute call per
  frame, metadata/model/string ownership, type order/mapping, row-vector cells,
  null nonperiodic cells, unsupported extra parameters/spin, missing/nonfinite
  outputs, and explicit backend error propagation.

libloading is pinned to 0.8.8, which uses prebuilt Windows import libraries.
0.8.9's windows-link path required an external assembler absent on this host.
The pin allowed a clean test/build using the official Rust toolchain alone;
no system C/C++ compiler, driver or security setting was changed.

## Not yet established

- Real DeepMD runtime loading and scientific parity with a public float64 model.
  No DeepMD runtime or production model was installed, loaded or evaluated.
- Internal frozen-model dtype and training units cannot be introspected using
  these C API calls. Float64 here means input/output buffers, not proof of
  float64 model weights or intermediate arithmetic.
- GPU evaluation and placement attestation are unsupported and fail explicitly.
- Legacy root-workspace build/MD integration is not verified by these tests.
  The nested MLMD workspace has no mandatory tch/chemfiles dependency.
- Native se_e2_a is delegated to B, not implemented by A.
- The declared Rust 1.82 MSRV was not separately tested with old toolchains.

## Local branch provenance

- origin/main: 761b2484feee132426183c9b66ff2b126d1f7c4a.
- origin/dev: 186817f5a85837df5f1df28e0fd2e8b2de6ecade.
  main...dev left/right counts = 13/1; final tree diff is README only.
- old origin/shawn_dev: 3440f605dde61b01e894f6532693612bc57c0aaa.
  main...shawn_dev counts = 0/1; sole tree change is a .DS_Store ignore entry.
- Shared new baseline: 8121e214e062077b88c39d0ccd861c55efaecc2c.
- No upstream remote was modified; no old shawn_dev work was imported.
