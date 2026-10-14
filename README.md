# omniMD: Rust MLMD development

This local `shawn_dev` starts from upstream main
`761b2484feee132426183c9b66ff2b126d1f7c4a`. It does not inherit the old
shawn_dev-only commit. Upstream omniMD remains alpha software under MIT.

The new implementation lives in the independent Cargo workspace [mlmd](mlmd).
It deliberately has no mandatory LibTorch or chemfiles build dependency.

- **DeepMD backend (A):** Rust calls the official DeePMD-kit C API. This is FFI,
  not native Rust inference. One model handle serves repeated combined energy,
  force and virial evaluations with double-precision input/output.
- **Native backend (B):** first milestone is CPU float64 `se_e2_a` and energy
  fitting only, with an explicit versioned weight format. It is under development.
- Full native training and GPU kernels are future milestones. No performance
  improvement or new algorithm is claimed.

Read [ROADMAP](ROADMAP.md), [interface contract](mlmd/CONTRACT.md), and
[native implementation handoff](mlmd/NATIVE_HANDOFF.md).

The existing root Cargo workspace is preserved as the upstream legacy engine.
The new backend is not yet connected to its infallible GlobalPotential trait.
In particular, its old TorchPotential contract is not a DeepMD interface.

No research data, production models, or old HAP tasks are part of this project.
No remote branch is changed by this local implementation.

## Development

```sh
cargo test --manifest-path mlmd/Cargo.toml --workspace
```

The DeepMD runtime is an external dependency, licensed LGPL-3.0-or-later.
See [third-party notices](mlmd/THIRD_PARTY_NOTICES.md).
