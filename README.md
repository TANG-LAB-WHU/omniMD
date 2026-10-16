# omniMD: Rust MLIP molecular dynamics

omniMD is alpha software under the MIT license. This development line starts
from upstream main 761b2484feee132426183c9b66ff2b126d1f7c4a and adds a separate
CPU MLIP workspace with one fallible, explicit-unit, float64 E/F/virial contract.
It is an initial software delivery, not a physical model or a performance claim.

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
all-neural-parameter E/F gradients and Adam on CUDA. CPU inference, geometry,
reduction and assessment remain; tiny training cases were slower on CUDA. Runtime/subset requirements
are documented under [device selection](mlmd/engine/README.md#explicit-native-cuda-inference).
The [published CUDA release scope](mlmd/RELEASE.md#published-cuda-release)
records the code, documentation and compact validation evidence included.

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
  outputs, exact mixed E/F parameter gradients, full-batch Adam and deterministic
  checkpoints. All neural weights/biases/timesteps train; geometry, normalization
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
- Minibatch/end-to-end GPU training, learned normalization or generic
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
hyperdual (v4 on reverse CUDA). E/F-only CPU-v1 and hyperdual CUDA-v2
continuation remain compatible. Frozen-readout fitting remains E/F-only.
The recorded tiny joint-loss full process was 619 ms CPU versus 1816 ms CUDA;
that historical hyperdual measurement does not establish a training speedup.
This extension is local and unpushed.

The local [cached reverse CUDA executor](mlmd/native/CUDA_REVERSE_TRAINING.md) is
available through explicit `config.cuda_gradient="reverse"`; hyperdual remains
the default. Reverse checkpoint v4 binds loss/executor and device/runtime/program
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
