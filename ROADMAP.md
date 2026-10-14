# Rust MLIP-MD roadmap

## Implemented locally

- Shared fallible, explicit eV/angstrom, float64 E/F/virial contract.
- Official DeepMD 3.1.3 C API binding: one model handle, combined evaluation,
  strict metadata/type/shape/finite/error checks, explicit CPU policy.
- Native CPU se_e2_a environment matrix, periodic neighbor images, smooth cutoff,
  normalization, embedding/fitting networks, versioned weights and analytic
  energy derivatives. Unsupported architectures fail explicitly.
- Actual omniMD System integration through fallible fixed-cell NVE dynamics,
  initial forces, cached E/F/virial and explicit internal-unit conversion.
- Backend-selectable CLI, streamed trajectories and model-fingerprinted restart.
- Public-model C API/Python parity and independent integration comparison;
  synthetic native invariance/gradient checks. See mlmd/VALIDATION.md for limits.

## Current model interchange work

- Strict conversion of the supported official DeepMD serialized se_e2_a subset
  into native weights, with no pickle loading or architecture guessing.
- Same-model native/C API E/F/virial comparison on generated float64 fixtures.
- Converter results will be recorded separately from the existing public
  regression model, which lacks the metadata needed for official serialization.

## Next simulation milestones

- Longer NVE and equal-duration timestep convergence on suitable public models.
- Explicit ensemble support and pressure/stress conventions before NVT/NPT.
- Typed trajectory import, optional classical coupling and constraints only
  after the corresponding fallible force/update contracts are implemented.
- Reproducible benchmarks with fixed hardware, model, precision and tolerances.

## Future training and acceleration (not implemented)

- Native training, optimizers, training checkpoints and force-loss gradients.
- Data pipelines, sampling and active learning with explicit supported formats.
- Native GPU kernels and CPU/GPU numerical comparison.
- Broader descriptors, including hybrids, only with implementation and validation.

## Publication gate

Before any push, report the exact code/document/fixture scope and validation
limits. Keep the existing remote shawn_dev until replacement is explicitly
authorized for publication. Never include private research data/models/logs.
