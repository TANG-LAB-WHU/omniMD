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
they do not establish general acceleration or long-time stability. **GPU training
is not implemented**; native training remains CPU-only. Runtime/subset requirements
are documented under [device selection](mlmd/engine/README.md#explicit-native-cuda-inference).
The [published CUDA release scope](mlmd/RELEASE.md#published-cuda-release)
records the code, documentation and compact validation evidence included.

This local branch additionally includes a conservative native CUDA neighbor
candidate cache, with final-version timing, MD and restart acceptance. It reduces
host neighbor preparation; kernels and the CPU path are unchanged. This follow-up
is **not published**. See [its scope and measurements](mlmd/RELEASE.md#local-native-cuda-cache-follow-up).

## Available in the accepted CPU release

- **DeepMD C API inference:** one official model handle, one combined
  energy/force/virial call, explicit CPU selection, and strict type-map,
  coordinate, cell, unit, shape, finite-value and error checks. This is FFI;
  double input/output buffers do not change internal model precision.
- **Native Rust inference:** CPU float64 se_e2_a, periodic neighbor images,
  cutoff/normalization, embedding and fitting networks, analytic forces and
  strain virial. Only the documented native-v1 architecture subset is accepted.
- **Model conversion:** bounded DeepMD 3.1.3 .dp/safe-YAML conversion and two
  audited frozen .pb graph profiles. Unsupported models fail explicitly.
- **Data and prediction:** plain XYZ, a documented extxyz subset and explicit
  DeepMD NPY sets; shared JSONL, optional E/F/virial labels, batch evaluation,
  error metrics and model/capability queries. No units or labels are guessed.
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

## Not included in the accepted CPU release

- GPU execution/acceleration, NPT/barostats, cell relaxation or constraints.
- Arbitrary DeepMD architectures, hybrid se_e3 or generic .pb/.pth/pickle import.
- Partial PBC, mixed-type/virtual atoms or automatic unit/type-map guessing.
- Classical-potential mixing and the full legacy Simulation/Propagator pipeline.
- Minibatch/GPU training, virial-loss training, learned normalization or generic
  DeepMD training compatibility; duplicated one-side embeddings are not tied.
- Production validation, long-time stability guarantees or demonstrated speedup.

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
