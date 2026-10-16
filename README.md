# omniMD: Rust MLIP molecular dynamics

omniMD is alpha software under the MIT license. This development line starts
from upstream main 761b2484feee132426183c9b66ff2b126d1f7c4a and adds a separate
CPU MLIP workspace with one fallible, explicit-unit, float64 E/F/virial contract.
It is an initial software delivery, not a physical model or a performance claim.

The [physical-label training pilot](mlmd/PHYSICAL_DATA_VALIDATION.md) completed
1,000 native GPU updates on the official DeepMD `nopbc` example (168 training /
43 index-held-out frames, 531 parameters). **Pipeline feasibility passed; the
resulting model is unsuitable for scientific acceptance or production use.**
Held-out energy/atom RMSE improved 0.569513->0.455950 eV/atom, while force RMSE
worsened 3.933565->4.164336 eV/angstrom. CPU/GPU inference agreement does not
establish label accuracy. Only 7-10-atom default-limit/v8 frames were exercised;
this result does not validate the later large-frame/v9 path.

The CPU baseline was published as fce0af9f6a49592a48c13883dfc3aa0d3eb4a28e.
The [CUDA release 9074817](https://github.com/TANG-LAB-WHU/omniMD/commit/907481708ae5f3df4228a9fb5d5b1110c39f5e23)
is now published on shawn_dev. It adds two explicitly selected inference paths
with recorded CPU/GPU numerical
agreement: [official DeepMD C API CUDA](mlmd/deepmd/CUDA_VALIDATION.md) and
[bounded native CUDA](mlmd/native/CUDA.md). The native unified CLI also passed a
short NVE/checkpoint continuation [check](mlmd/evidence/native-cuda-cli.json);
C API GPU batch evaluation and a two-step MD smoke test passed.

CPU remains the default at every system size. The six-atom C API sample measured
0.486 ms/frame on CPU versus 4.018 ms/frame on GPU, about 8.27 times slower with
kernel verification and synchronization included. Separately, the native
216-atom sample measured a CPU/GPU time ratio of 1.155, while its small cases were
about 8.4 times slower on GPU. These bounded measurements have different scopes;
they do not establish general acceleration or long-time stability. Local training extensions include CUDA readout features with CPU QR and bounded
all-neural-parameter E/F gradients and Adam on CUDA. Geometry, reduction and
assessment remain CPU work; optional tiled GPU prediction/loss preparation is
described below. Tiny training cases can be slower on CUDA. Runtime/subset requirements
are documented under [device selection](mlmd/engine/README.md#explicit-native-cuda-inference).
The [published CUDA release scope](mlmd/RELEASE.md#published-cuda-release)
records the code, documentation and compact validation evidence included.

The local candidate `06439f1` also passed [Linux native CUDA acceptance](mlmd/LINUX_CUDA_VALIDATION.md):
Ubuntu 22.04.5 x86_64 under WSL2, the official cached DeepMD CUDA image, RTX 4090,
driver 617.42 and NVRTC 12.9.86. This covers one FP64 inference fixture, 24-frame
tiled E/F/V training and exact same-runtime v8 continuation. The small v8 run
was slower than v7; no speed or broader-platform claim follows. C API execution
and cross-platform checkpoint restore were outside this acceptance. The later
v9 large-frame path below is not covered by this Linux validation.

This local branch additionally includes a conservative native CUDA neighbor
candidate cache, with final-version timing, MD and restart acceptance. It reduces
host neighbor preparation; kernels and the CPU path are unchanged. This follow-up
is **not published**. See [its scope and measurements](mlmd/RELEASE.md#local-native-cuda-cache-follow-up).

The current **local integrated follow-up** also contains cooperative native CUDA
fitting and persistent host buffers, plus standalone C API batching, explicit
frame/atom parameters, atomic outputs and model inspection. Native final-code
acceptance measured the same 192-atom, 20-step complete MD job at **412 ms** versus
**662 ms** previous CUDA and **1352 ms** CPU; the before/after whole-job ratio is
**1.61x**, separate from the **3.74x** warmed single-frame ratio for that model.
These are bounded measurements with startup/output/checkpoints included in MD.
[The final receipt](mlmd/evidence/native-cuda-integrated-fit.json) includes CPU/GPU
parity and exact restart. C API CPU batch/parameter/default/atomic paths now have real TF/PT numerical
validation; the new GPU paths remain untested. Use the
[standalone C API guide](mlmd/deepmd/BACKEND_GUIDE.md) for those options; shared
engine now supports versioned parameter datasets and fixed-parameter CPU C API MD. This integrated follow-up is **not published**.

The local native source `dc5449e` is integrated as `65a7192`: explicit CUDA-assisted
frozen-readout features, CPU QR and CPU assessment. [Its receipt](mlmd/native/fixtures/cuda-readout-report.json)
records same-model parity; the 3/4-atom case took 21.37 ms for CUDA features versus
5.15 ms CPU, plus 195.88 ms CUDA initialization. Full-network CUDA source `f4dbf62` is integrated as `02b5457`: all supported neural
E/F gradients and Adam, with CPU inference/loss seeds, geometry, reduction and
assessment. CPU remains the default; see [the bounds and checkpoint rules](mlmd/native/CUDA_FULL_TRAINING.md).
The recorded 3.46x ratio concerns a warmed 64-atom gradient stage only; the tiny
complete training case was slower. This follow-up remains local.

## Stepwise setup and current feature coverage

Start with the [complete English setup and usage guide](mlmd/GETTING_STARTED.md):

1. Build the separate `mlmd/Cargo.toml` workspace and its engine/native binaries.
2. Run `mlmd/scripts/run_cpu_tutorial.py` in a new output directory. It generates
   synthetic data and executes import, preflight, readout training, evaluation,
   CSVR NVT and exact restart with complete saved requests.
3. Choose native-v1 weights or a trusted official DeepMD C API runtime; validate
   conversion scope, type map, units and model metadata.
4. Import explicit DeepMD NPY sets or XYZ/extxyz; use `inspect-dataset` for full
   JSONL validation, byte SHA256 and E/F/V label coverage without loading a model.
5. Run batch assessment, then the supported CPU readout/full-network trainer with
   explicit validation identities and loss scales.
6. Minimize when appropriate, inspect convergence, run fixed-cell NVE/CSVR NVT,
   and continue from a matching checkpoint.
7. Select native CUDA or official C API CUDA explicitly, verify same-model E/F/V,
   and compare loading, warmed evaluations and full MD separately.
8. Follow the guide's troubleshooting table and retain hashes and completed reports.

[The versioned feature matrix](mlmd/FEATURE_MATRIX.md) compares this implementation
with official **DeepMD 3.1.3** and distinguishes native, delegated, partial and
missing functions. Broad DeepMD feature parity is not implemented. The preflight
command and executable tutorial are local follow-ups pending publication; the
[final CPU tutorial receipt](mlmd/evidence/cpu-tutorial-integrated.json) records the actual run.

## CPU inference, training and simulation capabilities

- **DeepMD C API inference:** one official model handle, one combined
  energy/force/virial call, explicit CPU selection, and strict type-map,
  coordinate, cell, unit, shape, finite-value and error checks. This is FFI;
  double input/output buffers do not change internal model precision.
- **Native Rust inference:** CPU float64 se_e2_a, periodic neighbor images,
  cutoff/normalization, embedding and fitting networks, analytic forces and
  strain virial. Only the documented native-v1 architecture subset is accepted.
- **Model conversion:** bounded DeepMD 3.1.3 .dp/safe-YAML conversion and two
  audited frozen .pb graph profiles. Unsupported models fail explicitly.
- **Parameterized CPU models:** explicit frame/atom data, opt-in model-stored frame
  defaults, v2 dataset/NPY import and fixed-context MD/checkpoint v3 through C API.
  [Stepwise parameter guide](mlmd/engine/PARAMETERS.md) records strict ordering and
  TF/PT acceptance, including PT numerical rather than bitwise restart parity.
- **Data and prediction:** plain XYZ, a documented extxyz subset and explicit
  DeepMD NPY sets; shared JSONL, optional E/F/virial labels, batch evaluation,
  error metrics and model/capability queries. No units or labels are guessed.
- **CPU model disagreement:** retained ensembles, absolute E/F/virial deviation and
  explicit force-window candidate flags; [stepwise usage](mlmd/engine/MODEL_DEVIATION.md)
  and independent synthetic replay. Disagreement is not calibrated uncertainty.
  [Hash-bound subset extraction](mlmd/engine/DATASET_SELECTION.md) preserves reviewed
  IDs and labels for assessment or externally labeled training inputs.
- **Native readout training:** frozen-hidden final linear weights (optional
  biases), exact E/F features, rank-aware QR, explicit validation splits and
  unchanged native-v1 model export.
- **Full supported-network training:** tanh embedding/hidden networks with linear
  outputs, exact mixed E/F parameter gradients, default full-batch or explicit
  minibatch Adam and deterministic checkpoints. All neural weights/biases/timesteps
  train; geometry, normalization
  and separate atomic offsets stay fixed. CPU reference-size limits apply.
- **Fixed-cell minimization:** both inference backends, bounded steepest descent,
  force plus energy-change stopping criteria, explicit failure/nonconvergence,
  and geometry-only extxyz/JSONL exports.
- **MD:** fixed-cell NVE velocity Verlet and upstream CSVR NVT, explicit masses,
  seeded thermal initialization, streamed trajectories and exact RNG restart.
  The runner uses the actual omniMD System through fallible MlipDynamics,
  caching combined E/F/virial and rolling back failed steps.

## Quickstart: no external model runtime

From the repository root with Rust/Cargo installed:

    cargo build --locked --manifest-path mlmd/Cargo.toml -p omnimd-mlmd -p omnimd-native --bins

Run the checked-in synthetic fixture (a software test, not a chemistry model).
On Linux/macOS:

    cat mlmd/native/fixtures/frame.jsonl | ./mlmd/target/debug/omnimd-native-eval mlmd/native/fixtures/synthetic-v1.json
    ./mlmd/target/debug/omnimd-mlmd capabilities

On Windows PowerShell:

    Get-Content mlmd/native/fixtures/frame.jsonl | .\mlmd\target\debug\omnimd-native-eval.exe mlmd/native/fixtures/synthetic-v1.json
    .\mlmd\target\debug\omnimd-mlmd.exe capabilities

The first command prints one combined E/F/virial JSON result. Native execution
requires no Python, DeepMD, LibTorch or chemfiles. The initial build resolves the
locked Rust dependencies; offline builds need them cached already. Windows and
Linux builds were checked with Rust 1.99; the declared lower MSRV has not been
independently verified.

For a short MD example, copy [native-nvt.json](mlmd/engine/examples/native-nvt.json),
set native_model to the absolute fixture path, then use:

    omnimd-mlmd check request.json
    omnimd-mlmd run request.json new-trajectory.jsonl

Use the executable path above if it is not on PATH. Output names must be new.
The example parameters only exercise software behavior. The [workflow](mlmd/WORKFLOW.md)
provides exact requests for import, prediction/training, minimization and MD.
Real C API inference additionally requires a trusted compatible DeepMD runtime;
see [DeepMD setup and limits](mlmd/deepmd/README.md).

## Current limitations

- NPT/barostats, cell relaxation or constraints. Optional CUDA inference is
  available within the separate scopes above; training retains CPU stages.
- Arbitrary DeepMD architectures, hybrid se_e3 or generic .pb/.pth/pickle import.
- Partial PBC, mixed-type/virtual atoms or automatic unit/type-map guessing.
- Classical-potential mixing and the full legacy Simulation/Propagator pipeline.
- Sharded/distributed or end-to-end GPU training, learned normalization or generic
  DeepMD training compatibility; duplicated one-side embeddings are not tied.
- Production validation, long-time stability guarantees or general speedup claims.

The root legacy workspace remains available with its original default features;
these do not imply equivalent support in the new MLIP runner. Training covers the
documented native tanh subset; it does not reproduce the full DeepMD training stack.

## Documentation, verification and licensing

- [Shared contract](mlmd/CONTRACT.md), [data formats](mlmd/engine/DATA.md),
  [minimization](mlmd/engine/MINIMIZATION.md), [MD runner](mlmd/engine/README.md).
- [Native models](mlmd/native/README.md), [conversion](mlmd/native/CONVERSION.md),
  [frozen graph scope](mlmd/native/FROZEN.md), [readout training](mlmd/native/TRAINING.md),
  [full supported-network training](mlmd/native/FULL_TRAINING.md).
- [Validation and its limits](mlmd/VALIDATION.md), [roadmap](ROADMAP.md),
  [publication scope and procedure](mlmd/RELEASE.md).
- [MIT license](LICENSE), [third-party notices](mlmd/THIRD_PARTY_NOTICES.md).
  DeepMD-derived topology profiles retain separate LGPL notices/licenses.

The release contains source, small synthetic fixtures, reproducible scripts,
compact validation summaries and licenses. External runtimes, public pretrained
weights, generated training data, raw operational logs and private research
inputs are excluded. No HAP research inputs or old HAP computations are used.
The CPU baseline and CUDA release are published on shawn_dev.

The local [external-label return workflow](mlmd/engine/EXTERNAL_LABELS.md) now
connects reviewed candidates to a new labeled dataset and explicit CPU retraining.
Source/return hashes and exact frame/atom/parameter identity are required;
conflicting labels reject. It preserves dataset v2 context without adding native
parameterized training, and launches no external calculator or automatic loop.

Full supported-network CPU/CUDA training also accepts optional
[virial supervision](mlmd/native/VIRIAL_TRAINING.md): all nine row-strain entries,
explicit per-atom scale/weight and loss-bound checkpoint v3 on CPU/default
hyperdual (v4 on reverse CUDA) for full-batch training; explicit minibatching
uses v5. E/F-only CPU-v1 and hyperdual CUDA-v2
continuation remain compatible. Frozen-readout fitting remains E/F-only.
The recorded tiny joint-loss full process was 619 ms CPU versus 1816 ms CUDA;
that historical hyperdual measurement does not establish a training speedup.
This extension is local and unpushed.

The local [cached reverse CUDA executor](mlmd/native/CUDA_REVERSE_TRAINING.md) is
available through explicit `config.cuda_gradient="reverse"`; hyperdual remains
the default. Full-batch reverse checkpoint v4 binds loss/executor and device/runtime/program
identity. On B's original 466-parameter/64-atom/100-step E/F/V fixture, three fresh
process medians were CPU 2963 ms, hyperdual 1349 ms and reverse 877 ms. The tiny
112-parameter case remained faster on CPU (615 ms versus reverse 1573 ms). These
bounded measurements include startup, IO/checkpoints and export; they do not
establish production scaling or general speedup. Existing bounds remain in force.

[Final integrated reverse verification](mlmd/evidence/native-reverse-integrated.json)
passed six joint-loss steps, exact 3+3 checkpoint continuation, CPU E/F/V agreement
within 4.45e-16 and zero model-reload difference. Device, loss and executor changes
reject without training output. This short check adds no timing claim.

Optional [fixed training geometry reuse](mlmd/native/FIXED_GEOMETRY_TRAINING.md)
is enabled with `config.fixed_geometry_cache=true` and an explicit CUDA device,
using either gradient executor. Default is false; CPU-only training rejects it.
The eager cache retains selected training-frame neighbors, full periodic
displacements and normalized R/J within a 64 MiB host-payload cap. Current-model
predictions and loss directions are rebuilt every step. Identity mismatch stops
training; cache on/off may toggle at restart without changing numerical state.
See [the step-by-step entrypoint](mlmd/GETTING_STARTED.md#17-reuse-fixed-training-geometry-explicitly).
Recorded three-process off/on medians are 865.5/818.4 ms for the 64-atom synthetic
cluster and 6946.3/956.9 ms for its triclinic periodic variant. The benefit is
workload-specific CPU geometry reuse, not a general GPU acceleration claim.

[Final cache integration evidence](mlmd/evidence/native-geometry-cache-integrated.json)
confirms six steps, exact 3+3 and both cache-toggle continuations, agreement with
the prior uncached checkpoint, CPU E/F/V differences below 4.45e-16 and zero
reload difference. The short replay retained 221,414 bytes for 18 training frames;
15 affected CPU-safe checks and the offline release build passed.

[Deterministic minibatching](mlmd/native/MINIBATCH_TRAINING.md) is now available
for CPU and both CUDA full-network executors through explicit
`config.minibatch={"batch_size":7,"seed":20261016,"order":"shuffle_each_epoch"}`.
Absent/null retains full-batch behavior. Short tail batches are retained and
normalized by their actual frame count; `steps` counts optimizer updates.
Without an explicit learning-rate schedule, checkpoint v5 stores the validated
next-batch sampler and supports exact
same-runtime mid-epoch restart. The [step-18 guide](mlmd/GETTING_STARTED.md#18-select-deterministic-minibatches)
and [integrated receipt](mlmd/evidence/native-minibatch-integrated.json) cover
seven updates, tail/cache-toggle continuation and legacy full-batch compatibility.
Default memory access retains the 1024-frame full-training limit. Explicit indexed
access below reads frames individually and has separate index/record/count limits.

[Indexed dataset access](mlmd/native/INDEXED_TRAINING.md) is now available through
top-level `"dataset_access":"indexed"` plus explicit `config.minibatch`. It
validates the existing v1 JSONL file, retains offsets/IDs/line hashes, reads one
frame at a time and verifies the whole file before checkpoints/publication.
Small inputs can switch memory/indexed on v5 restart exactly; memory is the default.
Indexed limits are 100000 frames, 16 GiB input, 256 KiB records and 64 MiB retained
index payload. Index/sampler/report/checkpoint memory remains O(frame count);
these are not peak-process-memory or validated production-capacity claims.
The [step-19 guide](mlmd/GETTING_STARTED.md#19-read-training-data-through-a-bounded-index)
and [final integration evidence](mlmd/evidence/native-indexed-integrated.json)
record exact CPU/CUDA state, metrics and storage/cache-switch continuation.
The separate 2051-frame, 128.19-MiB boundary case is deliberately whitespace
padded; its measured 12.83-MiB Windows peak working set is not a physical-data
throughput benchmark. That indexed milestone preserved the default 64-atom and
geometry-cache bounds; the later explicit large-frame option is described below.

The [startup validation consolidation](mlmd/native/INDEXED_TRAINING.md#startup-validation-consolidation)
removes two duplicate frame-validation passes per full-training CLI invocation.
Public validation, per-read hashes and publication whole-file checks remain.
[CPU-only final acceptance](mlmd/evidence/native-indexed-startup-integrated.json)
records fresh and historical-checkpoint continuation, default-memory equality,
48 fewer reads per 24-frame invocation, unchanged hash counts, exact checkpoint
contents/model bytes/metrics and zero reload difference. This tranche adds no GPU
runtime validation or capacity change. See [step 20](mlmd/GETTING_STARTED.md#20-interpret-startup-validation-and-io-counters)
for cumulative timing fields and the short CPU replay.

[Learning-rate schedules](mlmd/native/LEARNING_RATE.md) are available for full
neural-parameter training through optional `config.learning_rate_schedule`:
explicit constant or exponential staircase, indexed by globally completed Adam
updates. Absent/null retains constant behavior and historical checkpoint schemas;
an explicit schedule uses v6 (v7 with tiling, v8 with GPU loss, v9 with an expanded
frame ceiling) and must remain identical on resume. The
[step-21 guide](mlmd/GETTING_STARTED.md#21-select-an-optimizer-update-learning-rate-schedule)
and [final CPU engine receipt](mlmd/evidence/native-learning-rate-integrated.json)
cover boundary continuation, legacy v5 and refusal behavior. No model/atom limit
or CUDA kernel changes accompany this addition. Bounded multi-frame execution and optional GPU prediction/loss preparation are
available below. Larger model support and remaining end-to-end limits are
described in the [roadmap](ROADMAP.md#remaining-end-to-end-and-model-limits).

[Bounded multi-frame CUDA training](mlmd/native/TILED_TRAINING.md) is enabled with
explicit reverse CUDA and `config.cuda_tiling={"max_frames":16,"max_atoms":512}`.
With default CPU loss, each tile executes four gradient kernels, preserving per-frame reduction order,
optimizer batches, schedules and one Adam update per batch. Tile capacities bind
v7 restart identity with CPU loss (v8 for GPU loss); absent/null tiling retains
framewise execution and v1-v6. Limits
are 32 frames / 1024 aggregate atoms per tile, subject to work/payload planning;
the default is 64 atoms per frame, with explicit bounded expansion described
below. The 2048 neural-parameter and width-32 limits remain unchanged.
[Step 22](mlmd/GETTING_STARTED.md#22-batch-frames-in-bounded-cuda-tiles) explains
payload limits and diagnostics. [Final engine acceptance](mlmd/evidence/native-tiled-integrated.json)
records exact framewise equality and storage/cache-switch restart. Separate
synthetic whole-process medians improved 1071->551 and 4484->3568 ms; these are
bounded measurements, not physical-model or universal acceleration claims.

Optional [GPU predictions and detached loss seeds](mlmd/native/DEVICE_LOSS_TRAINING.md)
are enabled with `config.cuda_loss="gpu"` plus reverse CUDA and tiling. This path
reuses forward intermediates and binds v8 restart identity. Absent or `"cpu"`
retains the CPU-loss default and historical v7 state/PTX. Geometry, gradient
reduction/clipping and assessment remain CPU work. The
[step-23 guide](mlmd/GETTING_STARTED.md#23-select-gpu-prediction-and-loss-preparation-explicitly)
and [final receipt](mlmd/evidence/native-device-loss-integrated.json) cover CLI
forwarding, actual v7 compatibility, tolerant cross-mode comparison and exact
same-backend v8 restart. Three-repeat synthetic whole-process medians were
604.53->756.03 ms for 112 parameters (slower) and 3505.78->3060.90 ms for 466
parameters (1.15x). This option is not selected automatically; bounds remain.

[Resource-bounded larger training frames](mlmd/native/LARGE_FRAME_TRAINING.md)
use explicit `config.cuda_tiling.max_atoms_per_frame=256` with reverse CUDA.
Default/explicit 64 preserves prior v7/v8 behavior; CPU and untiled training stay
at 64. Above 64 selects v9, with GPU-loss backend v2 and complete strided force/
virial output coverage. Work and payload budgets are unchanged: the 112-parameter
fixture fits 256 atoms, but the 466-parameter example has effective capacity 138
and rejects 256. This is not a blanket 256-atom guarantee or a speed claim.
[Step 25](mlmd/GETTING_STARTED.md#25-opt-in-to-resource-bounded-larger-training-frames)
and the [final engine receipt](mlmd/evidence/native-large-frame-integrated.json)
record real 256-atom frames, exact v9 continuation and refusals. B's separate
12-update process observations were CPU-loss CUDA 1554 ms and GPU-loss CUDA
1721 ms; both still retain CPU geometry/reduction/assessment. New v9 Linux
execution remains unverified.
