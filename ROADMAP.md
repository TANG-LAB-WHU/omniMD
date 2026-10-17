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
  parameter-coordinate AD, default full-batch or explicit minibatch Adam and
  deterministic checkpoints.
  Normalization/cutoff/type-map/selection/atomic offsets stay fixed; small-model
  CPU bounds and independent duplicated one-side networks are explicit. Optional
  bounded CUDA computes all neural E/F gradients and Adam; CPU inference/loss
  seeds, geometry, stable reduction and assessment remain. CUDA checkpoint v2
  binds execution identity; CPU v1 remains compatible. See [scope](mlmd/native/CUDA_FULL_TRAINING.md).
- Optional [row-strain virial supervision](mlmd/native/VIRIAL_TRAINING.md) is
  implemented for full CPU/CUDA training, with explicit weight/per-atom scale and
  full-batch checkpoint v3 on CPU/default hyperdual, or v4 on reverse CUDA;
  minibatching uses v5. Legacy E/F-only checkpoint schemas remain compatible.
- Deterministic minibatching is integrated for CPU and both CUDA executors,
  with actual-count tails and v5 sampler checkpoints. Bounded indexed v1 input is
  also integrated; sharded access, schedules,
  broader activation support and scalable training remain future work; no generic
  DeepMD training compatibility is claimed.
- CPU ensemble disagreement and explicit candidate marking are implemented with
  independent population-statistics checks. Hash-bound subset extraction is now
  implemented and verified through assessment and CPU training of preserved
  existing labels. Bound external-label preparation/import now closes the file
  return and explicit CPU retraining workflow, preserving v2 context. Calculator
  adapters, job submission and scientific validation remain separate future work.
- Broader sampling and active learning with explicit formats.
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
  validation remains future work. Optional CUDA readout feature generation is now
  integrated; QR and assessment remain CPU and the microcase was slower on CUDA.
- Broader descriptors, including hybrids, only with implementation and validation.

## Current local integration after 9eaf284

- Native cooperative per-atom fitting and persistent host workspace are integrated.
  Final same-window 192-atom / 20-step complete MD measured 662 ms old CUDA,
  412 ms current CUDA and 1352 ms CPU, with numerical agreement and exact same-
  environment restart. This is a bounded result, not general GPU acceleration.
- C API adapter-local homogeneous batching, explicit fparam/aparam, optional
  atomic E/virial, stored frame-default opt-in and standalone inspection are
  integrated with TF/PT CPU numerical acceptance. Engine dataset v2, explicit
  NPY parameter import and fixed-parameter CPU MD/checkpoint v3 are implemented.
  Shared geometry/EFV types are unchanged; one combined call per frame remains.
  TF restart is exact; PT first-call runtime roundoff prevents a bitwise promise.
- The full CPU tutorial was rerun with the integrated feature-enabled executable
  selecting CPU. CUDA full-gradient/Adam integration is now available within its
  stricter bounds; end-to-end GPU training and generic DeepMD coverage remain missing.

## Remaining substantial compatibility gaps

- Native model coverage remains bounded `se_e2_a` with restricted conversion;
  `se_e2_r`, `se_e3`/hybrid, attention/DPA, type embeddings, spin and tensor models
  are not implemented. Two audited frozen graph profiles are not a generic
  `.pb` importer; native `.pth`/JAX/Paddle import and compression are absent.
- Training supports full-batch defaults and explicit deterministic minibatches
  with fixed normalization/atomic offsets and separately documented shape/data/
  tape budgets. Bounded indexed datasets are supported; sharded datasets, schedules,
  distributed/multi-task training, tied one-side embeddings and general DeepMD
  training-file/checkpoint compatibility remain missing. Optional virial loss
  closes one loss-function gap, without removing those execution restrictions.
- Engine parameterized models are CPU C API only; native parameterized training,
  CUDA engine parameter propagation, parameterized minimization/ensembles and
  shared atomic/tensor outputs remain unsupported. New C API GPU batch/parameter
  acceptance remains unverified. Native CUDA now has bounded
  [Linux x86_64/WSL2 acceptance](mlmd/LINUX_CUDA_VALIDATION.md); other Linux
  configurations, new large-frame v9 execution and cross-platform checkpoint
  restore remain unverified.
- NPT, constraints, domain decomposition, broader classical coupling and external
  MD ecosystem adapters remain separate simulation gaps. Physical accuracy and
  production stability have not been established. The
  [physical-label pilot and diagnosis](mlmd/PHYSICAL_DATA_VALIDATION.md) demonstrate
  pipeline feasibility and a loss-weight tradeoff. Both fitted models remain
  unsuitable for scientific acceptance; the 43 diagnostic-validation frames are
  not an untouched final test.

## Delivered: optional reverse CUDA joint gradients

The [cached reverse executor](mlmd/native/CUDA_REVERSE_TRAINING.md) is integrated
and verified, replacing parameter-by-parameter differentiated forwards only when
`config.cuda_gradient="reverse"` is explicit. Hyperdual remains the default and
an independent numerical oracle. Reverse checkpoint v4 binds loss and execution
identity. Existing model/data/work limits remain, with an additional 64 MiB
planned device-buffer payload limit that excludes context/code/driver memory.

[B's receipt](mlmd/native/fixtures/cuda-reverse-training-report.json) records
all-parameter CPU/hyperdual agreement, joint E/F/V learning, exact 150+150 restart
and bounded complete-process measurements. The 466-parameter/64-atom fixture's
three-repeat medians were CPU 2963 ms, hyperdual 1349 ms and reverse 877 ms. The
112-parameter tiny replay remained faster on CPU (615 ms versus reverse 1573 ms).
[A's final short replay](mlmd/evidence/native-reverse-integrated.json) verifies the
integrated CLI, CPU parity, exact v4 continuation and refusal contracts; it adds
no performance result. CPU inference/loss seeds, geometry, stable reduction,
clipping and assessment remain. No general speedup or larger-model support follows.

## Delivered: bounded reuse of immutable training geometry

The optional [fixed geometry cache](mlmd/native/FIXED_GEOMETRY_TRAINING.md) is
implemented for both CUDA full-training gradient executors. It reuses selected
training-frame neighbors, full periodic displacements and normalized R/J for CPU
primal inference and gradient direction construction. Current-model predictions,
loss seeds and directional rows are rebuilt every step. The default remains off;
CPU-only requests reject. Exact identity changes invalidate the entire pool and
stop training. Existing numerical bounds and checkpoint schemas remain unchanged;
cache toggles are allowed on restart under the same execution identity.

The 64 MiB retained host payload cap includes identity snapshots and cache records,
not allocator overhead, original dataset or temporary construction buffers. Eager
construction fails as a whole over budget. Assessment remains uncached, and input
rows still transfer each gradient call. This cache is separate from moving-frame
MD candidate/Verlet reuse.

[B's evidence](mlmd/native/fixtures/fixed-geometry-training-report.json) includes
cache-on/off exact gradients, Adam and cross-toggle continuation for both CUDA
executors, old-checkpoint compatibility, and three complete processes per setting.
The 466-parameter/64-atom cluster off/on medians are 865.5/818.4 ms; the triclinic
periodic variant is 6946.3/956.9 ms. These bounded synthetic measurements include
cache construction and reflect avoided CPU geometry enumeration, not a general
GPU speedup or production-model scaling result.

Remaining throughput work should first measure current-model CPU inference,
geometry-input transfers and launch costs with this cache enabled. Keeping fixed
rows/types resident on device would need its own device-memory accounting, exact
identity and failure/restart checks before implementation. Sharded datasets,
larger accepted models and moving primal inference remain distinct later work;
no supported limit is raised by this cache.

## Delivered: deterministic minibatches and v5 sampler restart

Explicit `config.minibatch` supports CPU and both CUDA executors for E/F/optional-V
training. Actual short-batch normalization, epoch-local sequential/shuffled
sampling and canonical next-batch state are specified in the
[native guide](mlmd/native/MINIBATCH_TRAINING.md). Full-batch defaults and actual
v1-v4 historical continuation remain compatible. [Final integrated evidence](mlmd/evidence/native-minibatch-integrated.json)
covers the CLI, sampler reference, tail/mid-epoch/cache-toggle restart and CPU
parity. This milestone changes sampling, not supported physics or dataset capacity.

## Delivered: bounded indexed dataset access

Explicit top-level `dataset_access="indexed"` with minibatching now validates and
indexes the existing v1 JSONL file, then reads frames individually for training,
assessment and reload checks. The [indexed contract](mlmd/native/INDEXED_TRAINING.md)
sets separate bounds: 100000 frames, 16 GiB input, 256 KiB records and 64 MiB index
payload. Index/sampler/report/checkpoint metadata remains O(frame count). Default
memory limits, model architecture/64-atom bounds and the geometry-cache cap remain.
Per-line and publication-boundary whole-file hashes preserve raw dataset identity.
[Final integration](mlmd/evidence/native-indexed-integrated.json) verifies exact
CPU/CUDA state/metrics, old-memory continuation and both access/cache directions.

The separate 2051-frame, 128.19-MiB input is deliberately whitespace padded. Its
12.83-MiB Windows peak working set and 319722-byte index are boundary-test evidence,
not representative physical throughput or a universal RAM bound. The full indexed
ceiling and other-platform execution remain unvalidated. Sharding/distributed
training and unchanged per-frame model limits remain open.

## Delivered: consolidate confirmed duplicate startup validation

The full trainer reuses the parameter layout returned by its complete checkpoint
and device validation; the CLI no longer repeats that validation immediately
before entry. This removes two frame passes per CLI invocation. The public
constructor/validator and trainer entry still validate independently. No token
or cross-call cache is introduced: public state may change between invocations.
Per-read hashes and every publication whole-file verification remain unchanged.

[CPU-only final acceptance](mlmd/evidence/native-indexed-startup-integrated.json)
records fresh seven-update and historical step-6-to-7 indexed runs with exact
checkpoint contents, model bytes, initial/final metrics and sampler. Each saves
48 frame reads on the existing 24-frame input; whole-file checks remain 9 and 3.
Default-memory comparison and three pre-output refusals also pass. Nine focused
CPU checks and the locked offline release build passed; no GPU ran this tranche.

[Separate native timing evidence](mlmd/native/fixtures/indexed-startup-report.json)
uses three alternating comparisons on the deliberately whitespace-padded
2051-frame input with warm filesystem cache. Fresh/resumed complete-process
medians decrease from 2127.12/1518.39 ms to 1860.17/1261.83 ms. This bounded parsing/
IO result is not a physical-training, cold-disk or general GPU throughput claim.

Remaining costs include the initial strict scan, constructor validation on fresh
runs, trainer validation, full initial/final assessments, reload verification and
O(file bytes) publication hashes. Metadata/sampler/checkpoints remain O(N).
Sharding/prefetch, broader architectures, larger models,
distributed execution and other-platform acceptance remain separate substantial
work. No further feature is started as part of this candidate.

## Delivered: optimizer-update learning-rate schedules

Optional explicit constant/exponential staircase schedules apply to the supported
full CPU/CUDA trainers. The globally completed update count determines the next
rate; no epoch inference or mutable scheduler state is added. Explicit schedules
use v6 (v7 with tiling, v8 with GPU loss, v9 with an expanded frame ceiling)
with immutable base/schedule identity, while absent/null preserve legacy
constant behavior and v1-v5. [Final engine evidence](mlmd/evidence/native-learning-rate-integrated.json)
records exact decay-boundary continuation and historical-checkpoint compatibility.
Model/atom/workspace limits, CUDA kernels and PTX remain unchanged.

## Delivered: bounded multi-frame reverse CUDA execution

Explicit `cuda_tiling` packs independent frames into actual GPU tiles with four
reverse kernels each. Original per-frame gradient reduction order, optimizer
batch normalization, schedules and Adam boundaries remain. v7 binds the tiled
executor and capacities; storage/cache may switch on restart. The
[final engine receipt](mlmd/evidence/native-tiled-integrated.json) records exact
framewise state/metrics, storage/cache-switch continuation and refusal checks.

This milestone batches the existing supported models; it does not expand them.
Requested capacities up to 32 frames/1024 aggregate atoms are reduced by checked
work and payload budgets. That milestone retained 64 atoms per frame, 2048
parameters, width 32 and 64 neighbor slots. A later explicit frame expansion is
documented below. Explicit device buffers are capped at 64 MiB and
host tile payload at 32 MiB, with documented exclusions from total memory.

[Separate native evidence](mlmd/native/fixtures/tiled-training-report.json)
records three alternating 128-update whole-process comparisons. Synthetic tiny
and medium medians decrease 1071->551 ms and 4484->3568 ms. These results do not
establish physical accuracy, larger-architecture support or universal speedup.

## Delivered: optional tiled GPU prediction and loss preparation

Explicit `cuda_loss="gpu"` computes E/F/V predictions and detached loss seeds on
GPU, reusing forward values in directional/mixed reverse derivatives. Ten kernels
and one synchronization execute per tile. v8 binds this arithmetic mode, while
absent/CPU loss preserves historical v7 state/PTX. Geometry, stable reduction/
clipping, assessment and publication remain CPU work. See the
[contract](mlmd/native/DEVICE_LOSS_TRAINING.md) and
[final engine acceptance](mlmd/evidence/native-device-loss-integrated.json).

The earlier medium tiled CPU prediction bottleneck motivated this implementation;
its cost moved rather than disappearing. B's three alternating 128-update
whole-process synthetic medians are 604.53->756.03 ms for 112 parameters (a
regression) and 3505.78->3060.90 ms for 466 parameters (1.15x). Tiny initialization
increases from about 329 to 523 ms. On the medium case, GPU kernel/sync and upload
costs increase while CPU assessment still takes about 446 ms. Keep the option
explicit; no general speed guarantee or automatic selection follows.

## Delivered: explicit resource-bounded larger frames

`cuda_tiling.max_atoms_per_frame` accepts 64..256 with default 64. Larger ceilings
require reverse CUDA tiling and select v9; GPU-loss execution uses backend v2.
The strided gather covers every force/virial output of actual large Frames.
High-level admission checks all dataset sizes before cache/context construction;
the low-level batch checks sizes before any tile allocation/launch. Frames are
not split or truncated. Default v7/v8 identities and CPU/untiled bounds remain.

[Final engine evidence](mlmd/evidence/native-large-frame-integrated.json) verifies
four scheduled updates on 32/128/256-atom inputs, six training/two validation
frames, exact 2+2 continuation across storage/cache changes and resource refusals.
B's separate independent derivative/capacity checks and 12-update measurements
are recorded in the [native contract](mlmd/native/LARGE_FRAME_TRAINING.md).
The process observations were 1554 ms CPU-loss CUDA and 1721 ms GPU-loss CUDA;
this delivery expands admitted capacity, not demonstrated throughput.

## Delivered: packed transfers and resident GPU optimizer state

Sources `cb2b94f` and `f4ca7f3` are integrated as `9d389ca` and `8743e0a` after
engine-only diagnostic reuse `76d60c6`. Explicit reverse/tiled `cuda_loss="gpu"`
training retains parameters and Adam moments on device until original save
boundaries, with a synchronized status check each update and full validation before
publication. Packed inputs use 256-byte offsets. No kernels, FP64 formulas, model/
checkpoint formats, default device or new configuration fields changed.

The [fixed-work comparison](mlmd/native/RESIDENT_STATE_TRAINING.md) records Windows
RTX 4090 whole-process medians versus `201cdef`: public 531-parameter E/F,
1,000 updates, 1332.67->1229.95 ms (7.7% less elapsed time); medium synthetic
466-parameter E/F/V, 128 updates, 2946.89->2875.74 ms (2.4%). The medium incremental
benefit over transfer-only is unresolved. Tiny CPU 504.19 ms remains faster than
CUDA 703.56 ms. Earlier transfer-only public regression is retained; alignment
and residency contributions are not separately identified. Frequent checkpoints
reduce savings. No current Linux performance or residency acceptance is claimed.

[Integrated CLI acceptance](mlmd/evidence/native-resident-integrated.json) reproduces
the complete public run and historical v8 step-500 continuation with exact model,
checkpoint, metric and execution identity. Full all-frame export validation remains.
Native 29-test/CPU-only/v9 evidence was reused; no benchmark suite was repeated.
See [step 27](mlmd/GETTING_STARTED.md#27-use-resident-gpu-training-state-and-interpret-measured-speed).
Scientific model suitability remains unchanged.

## Physical-label diagnosis: measured loss tradeoff, model quality unsuitable

The unchanged `8167a4a` runtime completed the original physical-label fit and one
controlled comparison: same 531-parameter model, initialization, split and 1,000
updates, with only `energy_weight` changed from 1 to 0.1. Validation force RMSE
was 3.933565 initially, 4.164336 after the original fit and 3.707381 eV/angstrom
after the comparison. Validation E/atom RMSE was 0.569513/0.455950/0.523472 eV/atom.
Force improvement accompanies less energy progress; new training total-energy
RMSE worsened from initial 6.626241 to 6.650750 eV. Both models remain unsuitable.

A two-update CPU/CUDA comparison of 531 parameters and Adam state differed by
about 1.11e-16 at most over 16 frame visits. It supports limited numerical
consistency, not absence of shared defects or full 1,000-update agreement.
Normalization was unchanged and its independent effect was not tested. Clipping
settings were unchanged, but clipping frequency under the changed loss was not
measured. No implementation fix is established by this evidence.

The [recorded diagnosis](mlmd/PHYSICAL_DATA_VALIDATION.md#bounded-training-diagnosis)
retains both fits, their losses and all composition groups. The 43 index-held-out
frames are now diagnostic validation, not an untouched final test. Original
trajectory identities remain unavailable. There are no physical virial labels,
transferability or production-quality claims. This Linux default-limit/v8 case
does not validate v9; integration adds no new experiment or feature.

## Remaining end-to-end and model limits

Supported native training remains tanh se_e2_a with fixed normalization, atomic
offsets, selection, cutoffs and type map. CUDA accepts at most 2048 parameters,
width 32 and 64 selected slots. Frames default to 64 atoms; explicit reverse
tiling admits up to 256 only when unchanged resource budgets permit it. Requested
tile ceilings of 32 frames/1024 aggregate atoms can shrink under work/payload
planning. The 112-parameter fixture fits 256, while the 466-parameter GPU-loss
example has effective capacity 138 with the expanded ceiling and rejects 256.
The configured ceiling contributes conservatively to quadratic gather work even
for smaller actual frames. These bounds do not establish representative large-model use.

Data validation, geometry/scales, gradient reduction/clipping, assessment and
export remain CPU work. Indexed metadata/samplers/checkpoints are O(frame count),
and publication hashes remain O(file bytes). No sharding, prefetch/distributed
execution, generic DeepMD architecture support, learned normalization/offsets,
tied one-side embeddings or production chemical validation is added. The
[Linux acceptance](mlmd/LINUX_CUDA_VALIDATION.md) covers only the recorded
Ubuntu x86_64/WSL2 container, RTX 4090 and small synthetic inference/training
v7/v8 cases; it does not cover new large-frame v9 execution, other Linux
configurations or cross-platform restart.
The 64 MiB device/32 MiB tile budgets are explicit payloads, not total VRAM or
process memory limits.

Whole-process startup, transfers and ten-kernel tile execution are now material
costs alongside assessment and constrained occupancy. Any later throughput work
should compare matching workloads with those phases separated, retaining numerical,
restart, failure and resource guarantees. Cache-off resume timing is a different
invocation and cannot quantify cache speedup. Larger architectures and physical
quality require independent acceptance. No new feature or benchmark begins in
this integration tranche.

## Publication gate

The CPU baseline fce0af9 and CUDA release 9074817 are published on shawn_dev.
Before any further push, report its exact code/document/fixture scope and
validation limits and obtain publication authorization. Never include private
research data/models/logs.
