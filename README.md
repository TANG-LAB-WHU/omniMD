# omniMD: Rust MLIP molecular dynamics

This local `shawn_dev` starts from upstream main
`761b2484feee132426183c9b66ff2b126d1f7c4a`. It does not inherit the old
shawn_dev-only commit. Upstream omniMD remains alpha software under MIT.

The [mlmd workspace](mlmd) provides two CPU backends behind one fallible,
unit-explicit, float64 energy/force/virial interface:

- [DeepMD C API](mlmd/deepmd): one loaded official model handle, strict type map
  and output validation. This is FFI; double buffers do not change a model's
  internal precision.
- [Native Rust](mlmd/native): CPU float64 `se_e2_a` descriptor and energy fitting
  networks, with analytic forces and strain virial. Only the documented
  versioned model format and architecture subset are supported.

A separate JSON command-line runner drives the actual upstream `omnimd-core::System` through
`omnimd-sim::md::MlipDynamics`. The [MD runner](mlmd/engine) implements fixed-cell
NVE velocity Verlet and upstream CSVR NVT, seeded thermal initialization,
cached combined model output, streamed JSONL trajectories, periodic checkpoints,
and model-fingerprinted checkpoints with full RNG state. It rejects classical potential
mixing and rolls back the published state on a failed integration step.
This runner does not enable the full legacy Simulation/Propagator pipeline;
the engine README describes the implemented bridge and remaining integration work.

```sh
cargo build --locked --manifest-path mlmd/Cargo.toml --workspace
cargo test --locked --manifest-path mlmd/Cargo.toml --workspace
cargo run --locked --manifest-path mlmd/Cargo.toml -p omnimd-mlmd -- run request.json new-trajectory.jsonl
```

These builds have no mandatory LibTorch, chemfiles, Python or DeepMD runtime
dependency. Real C API inference requires a compatible external DeepMD runtime.
Legacy root-workspace features remain enabled by default; the new runner selects
the lightweight core/sim features. The old TorchPotential contract and infallible
GlobalPotential interface are not used for MLIP execution.

Read the [contract](mlmd/CONTRACT.md), [validation results](mlmd/VALIDATION.md),
[roadmap](ROADMAP.md), and [third-party notices](mlmd/THIRD_PARTY_NOTICES.md).
Full native training and GPU acceleration are future milestones. Short synthetic
and public-fixture tests establish software behavior, not physical model quality
or a performance improvement.

No private research data or old HAP tasks are included. Remote branches remain
unchanged; publishing requires a separate review of the exact content scope.
