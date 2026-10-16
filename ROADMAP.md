# Rust MLIP-MD roadmap

The [DeepMD 3.1.3 feature matrix](mlmd/FEATURE_MATRIX.md) is the versioned coverage
baseline. [Stepwise usage](mlmd/GETTING_STARTED.md) connects implemented features
with executable commands; pending branch work is not counted as integrated.

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
- Model-free dataset inspection with byte-exact SHA256, provenance declarations,
  PBC/species counts and bounded E/F label coverage; complete synthetic CPU tutorial.
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
  published with actual public-model E/F/virial agreement, GPU batch
  and two-step MD evidence. The six-atom paired sample was 8.27 times slower on
  GPU including verification/synchronization overhead. CPU remains the default;
  larger-model performance is unmeasured.
- Native CUDA inference and three-fixture CPU/GPU numerical evidence are
  published behind the explicit native-cuda CLI feature. The bounded
  216-atom timing was 1.155x faster; small cases were about 8.4x slower.
  The unified CLI passed CPU routing/build checks and a short same-model NVE
  CPU/GPU comparison with exact checkpoint continuation. Broader GPU simulation
  validation and native CUDA training remain unimplemented.
- Broader descriptors, including hybrids, only with implementation and validation.

## Current local integration after 9eaf284

- Native cooperative per-atom fitting and persistent host workspace are integrated.
  Final same-window 192-atom / 20-step complete MD measured 662 ms old CUDA,
  412 ms current CUDA and 1352 ms CPU, with numerical agreement and exact same-
  environment restart. This is a bounded result, not general GPU acceleration.
- C API adapter-local homogeneous batching, explicit fparam/aparam, optional
  atomic E/virial and standalone inspection are integrated. Focused tests pass;
  new real-runtime numerical acceptance remains pending. Shared engine schemas
  remain parameter-free, total E/F/V and one combined call per frame.
- The full CPU tutorial was rerun with the integrated feature-enabled executable
  selecting CPU. Native GPU training and generic DeepMD coverage remain missing.

## Publication gate

The CPU baseline fce0af9 and CUDA release 9074817 are published on shawn_dev.
Before any further push, report its exact code/document/fixture scope and
validation limits and obtain publication authorization. Never include private
research data/models/logs.
