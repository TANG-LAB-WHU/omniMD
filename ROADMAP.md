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
- Backend-selectable run/check CLI, seeded thermal initialization, upstream CSVR
  NVT, explicit DOF, streamed trajectories, periodic checkpoints and exact RNG restart.
- Standard XYZ and bounded extxyz import with explicit units/PBC and optional labels.
- Dual-backend fixed-cell steepest descent, bounded Armijo trials, force/energy
  convergence, explicit nonconvergence and label-free structure export.
- Explicit DeepMD standard NPY import, normalized dataset JSONL, partial E/F/V
  labels, batch predictions/metrics, dataset-frame MD entry and model queries.
- Public-model C API/Python parity and independent integration comparison;
  synthetic native invariance/gradient checks. See mlmd/VALIDATION.md for limits.

## Model interchange

- Implemented strict conversion of the supported DeepMD 3.1.3 .dp/safe-YAML subset
  into native weights, with no pickle loading or architecture guessing.
- Passed same-model native/C API E/F/virial comparison on generated float64 fixtures.
- Explicit dpmodel/tensorflow geometry semantics; no implicit radius rounding.
- Implemented two audited frozen-graph profiles with complete offsets and strict
  topology checks; public and synthetic native/C API parity passed. Arbitrary .pb
  and hybrid HAP architectures remain unsupported.

## Next simulation milestones

- Longer NVE and equal-duration timestep convergence on suitable public models.
- Fallible Simulation/Propagator and output-snapshot bridge to more upstream features.
- Broader NVT validation; pressure/stress conventions and testing before NPT.
- Typed trajectory import, optional classical coupling and constraints only
  after the corresponding fallible force/update contracts are implemented.
- Reproducible benchmarks with fixed hardware, model, precision and tolerances.

## Training and acceleration

- Bounded frozen-hidden final-readout E+F fitting is integrated, including explicit
  heldout IDs and rank-aware QR. It is not complete-network training.
- Full supported tanh network E+F training is integrated: exact mixed
  parameter-coordinate AD, full-batch Adam and deterministic checkpoints.
  Normalization/cutoff/type-map/selection/atomic offsets stay fixed; small-model
  CPU bounds and independent duplicated one-side networks are explicit.
- Minibatching, broader activation support, virial loss and scalable training
  remain future work; no generic DeepMD training compatibility is claimed.
- Broader data pipelines, sampling and active learning with explicit formats.
- DeepMD C API CUDA device propagation and kernel/synchronization checks are
  integrated locally. Real model GPU parity and warmed timing remain pending;
  the published CPU baseline is unchanged.
- Native GPU kernels and CPU/GPU numerical comparison are under separate
  development/acceptance and are not integrated into this branch yet.
- Broader descriptors, including hybrids, only with implementation and validation.

## Publication gate

Before any push, report the exact code/document/fixture scope and validation
limits. Keep the existing remote shawn_dev until replacement is explicitly
authorized for publication. Never include private research data/models/logs.
