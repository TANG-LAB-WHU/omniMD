# Stepwise setup and use

This guide covers the MLIP workspace in `mlmd/`, tested at integrated runtime
9b8d184 (native source b6b7aa7 and C API source 1aee39c). Read the
[versioned feature matrix](FEATURE_MATRIX.md) before selecting a model. Native
Rust and official DeepMD FFI are distinct inference paths. CPU is the default;
CUDA is explicitly selected. Training defaults to CPU; optional CUDA readout
features keep QR/assessment on CPU, while full-network GPU training is unsupported.

All commands below run from the repository root. `omnimd-mlmd` means the binary
built in step 1, not a globally installed program. Use its full path unless you
have explicitly added its directory to your current shell's PATH. JSON examples
use absolute paths; replace `/absolute/...` with your own existing files. On
Windows use `D:/path/file.json` or escaped `D:\path\file.json` inside JSON.
Always choose new output names: existing outputs are never overwritten.

## 1. Build the standalone MLIP workspace

Install Git and Rust/Cargo using the [official Rust installer](https://rust-lang.org/tools/install/)
and the linker required by your target. Windows MSVC requires the corresponding
C++ build tools; the recorded Windows acceptance used an isolated GNU toolchain.
Rust 1.99 was tested; the manifest's lower declared MSRV has not been verified.
No Python, TensorFlow, DeepMD, LibTorch or CUDA is needed for native CPU inference.
Python 3 is needed only for the optional tutorial and model conversion scripts.

For a new checkout (choose an unused directory):

```sh
git clone --branch shawn_dev https://github.com/TANG-LAB-WHU/omniMD.git omniMD
cd omniMD
git rev-parse HEAD
```

The published branch may precede this local tranche. Check that
`mlmd/scripts/run_cpu_tutorial.py` exists and `omnimd-mlmd --help` includes
`inspect-dataset` before using the new tutorial. See [release status](RELEASE.md).
Existing contributors should keep their current worktree rather than cloning or
resetting over existing work.

```sh
rustc --version
cargo --version
cargo build --release --locked --manifest-path mlmd/Cargo.toml -p omnimd-mlmd -p omnimd-native --bin omnimd-mlmd --bin omnimd-native-eval --bin omnimd-native-train
```

Use `--offline` as well when the locked dependencies are already cached.
The initial normal build can download crates. This nested workspace avoids the
legacy workspace's optional Torch/chemfiles requirements. Windows binaries have
`.exe`; Linux binaries do not. Preserve separate binaries when comparing versions.

On the already provisioned isolated Windows setup, activate it for this shell
only before building (the script installs nothing):

```powershell
. .\mlmd\scripts\windows-env.ps1 -ToolchainRoot C:/path/to/omnimd-toolchain
```

That helper expects the documented isolated GNU layout; it is unnecessary for a
normal working Rust installation. Do not point it at a random existing directory.

## 2. Run a complete CPU example with no external model

Use a new work directory outside the checkout. The script creates original
synthetic teacher weights, evaluates labels with Rust, imports extxyz, inspects
the dataset, trains a final readout, evaluates the exported model, runs ten
CSVR steps and checks exact continuation from step five. It saves every request,
step output, model, report and checkpoint. It does not install or download files.
The three binaries from step 1 must be built; the trainer must sit beside the
engine binary. All model options explicitly select CPU.

Windows PowerShell:

```powershell
python mlmd/scripts/run_cpu_tutorial.py --engine mlmd/target/release/omnimd-mlmd.exe --eval-exe mlmd/target/release/omnimd-native-eval.exe --work D:/omnimd-runs/tutorial-01
```

Linux:

```sh
python3 mlmd/scripts/run_cpu_tutorial.py --engine mlmd/target/release/omnimd-mlmd --eval-exe mlmd/target/release/omnimd-native-eval --work /absolute/omnimd-runs/tutorial-01
```

Expected final message: `Passed.` The receipt records 48 frames, 36 training
and 12 explicit validation IDs, dataset/executable hashes, complete label
preservation and exact resumed snapshots/checkpoint. The
[final integrated receipt](evidence/cpu-tutorial-integrated.json) records held-out E/atom RMSE
6.72e-16 eV and force RMSE 3.17e-17 eV/A for this manufactured readout problem.
These values demonstrate software consistency, not chemical accuracy. The small
time step, temperature and system in this recipe are demonstration settings.

To repeat one saved request manually, choose a different output path:

```sh
omnimd-mlmd model-info /absolute/tutorial-01/model-info.request.json new-info.json
omnimd-mlmd inspect-dataset /absolute/tutorial-01/dataset.jsonl new-inspection.json
omnimd-mlmd batch-eval /absolute/tutorial-01/batch.request.json new-predictions.jsonl
omnimd-mlmd check /absolute/tutorial-01/md.request.json
```

The saved training request already names an existing trained directory after the
first run. Change `output_directory` before running it again.

## 3. Choose and connect a model

### A. Native Rust weights

Start with a native-v1 model or convert a supported DeepMD model. Native CPU
loading itself has no external model-runtime dependency.

For the bounded DeepMD 3.1.3 interchange format:

```sh
python mlmd/native/tools/convert_deepmd.py /absolute/model.dp /absolute/new-native.json --units EvAngstrom --geometry tensorflow
```

Use an existing Python environment with `h5py` for `.dp` or `PyYAML` for safe YAML.
For a source using dpmodel geometry, select `--geometry dpmodel`. The TensorFlow
mode reproduces float32 operator cutoff attributes while keeping weights and
arithmetic f64; choose according to the source backend. Review
[conversion restrictions](native/CONVERSION.md), including normalization, offset,
architecture and version checks. This is not a generic `.pth`/pickle importer.

For one of the two audited frozen TensorFlow graph profiles:

```sh
python mlmd/native/tools/export_frozen.py /absolute/model.pb /absolute/new-native.json --units EvAngstrom
```

This separate tool needs existing TensorFlow/protobuf/NumPy for decoding; it does
not execute the supplied graph. [Exact profile matching](native/FROZEN.md) is
required. An unsupported graph is an error; missing training metadata is never
filled by guessing. Retain source/output hashes and compare E/F/V on identical
frames against the source runtime before using a new conversion.

Create `model-info.request.json`:

```json
{"options":{"native_model":"/absolute/new-native.json","device":"cpu"}}
```

```sh
omnimd-mlmd capabilities
omnimd-mlmd model-info model-info.request.json new-model-info.json
```

`capabilities` does not load a runtime. `model-info` loads the chosen model and
validates its metadata; it does not test physical accuracy or infer training units.

### B. Official DeepMD runtime through C API

1. Obtain a trusted matching official runtime for your OS/architecture, backend
   and model. Use the pinned [3.1.3 installation guidance](https://docs.deepmodeling.com/projects/deepmd/en/v3.1.3/install/easy-install.html)
   or [precompiled C library instructions](https://docs.deepmodeling.com/projects/deepmd/en/v3.1.3/install/install-from-c-library.html).
   Record its version and file hash. The precompiled library documented there is
   a Linux distribution, not a Windows DLL. Installing only a Python module is
   not evidence that a loadable matching C library is available.
2. For a downloaded official Linux archive, extract it into a dedicated runtime
   directory, inspect its `lib` contents and dependencies, then use the actual
   absolute path to `libdeepmd_c.so`. Keep matching TensorFlow/CUDA dependencies
   with that runtime. A Windows build needs matching official DLLs; a Linux `.so`
   cannot be loaded by a Windows executable. Run Linux binaries inside the same
   Linux/container environment as their libraries.
3. Build the optional standalone C API evaluator if desired:

```sh
cargo build --release --locked --manifest-path mlmd/Cargo.toml -p omnimd-deepmd --bin omnimd-deepmd
```

4. Use this `options` object in model-info, batch, minimize or MD requests:

```json
{
  "options": {
    "library": "/absolute/runtime/lib/libdeepmd_c.so",
    "model": "/absolute/model.pb",
    "device": "cpu",
    "type_map": ["O", "H"],
    "units": "EvAngstrom"
  }
}
```

Replace the ordered type map with the model's actual chemical symbols. Explicit
units assert the model's training convention. The wrapper cannot discover that
convention or upgrade float32 weights by passing double buffers. Spin remains
unsupported. Engine requests still use the parameter-free shared E/F/V contract;
parameterized models need the adapter-local interface described next. See
[C API runtime/error contract](deepmd/README.md).

### C. Standalone C API batches, parameters and atomic outputs

The integrated `omnimd-deepmd` executable additionally exposes explicit homogeneous
batches, parameter arrays and optional atomic energy/virial outputs. Its 21 focused
Rust tests pass; new real-runtime numerical validation remains pending. These
options do not extend `omnimd-mlmd` MD/dataset schemas or native inference.

Create `options.cpu.json` containing the C API `options` object from above
**without** its outer `options` key, then query dimensions:

```sh
omnimd-deepmd inspect options.cpu.json new-inspection.json
```

The output reports `model_details.frame_parameter_dim` and `atom_parameter_dim`.
For ordinary unparameterized models, the standalone evaluation request contains
`options`, an array of shared `frames`, `batch_size:2` and `atomic:true`. Each
consecutive batch must share atom count, ordered species and periodicity; cells
and coordinates may differ. `batch_size:1` preserves serial behavior.

For a model reporting two frame and one atomic parameter, provide one explicit
object per frame, for example three atoms:

```json
"parameters": [{"fparam":[300.0,1.0],"aparam":[[0.1],[0.2],[0.3]]}]
```

This is a member of the full request, with exactly as many objects as `frames`.
Values here illustrate shape only; their units/meaning must match the model.
Nonzero dimensions require explicit values, with no broadcasting or guessed
zeros. Requested `atomic_evaluations` use input atom order and the official
runtime's decomposition; they are not independently measurable atomic energies.
Use the [complete standalone requests](deepmd/BACKEND_GUIDE.md) and read
[the current validation boundary](deepmd/BACKEND_VALIDATION.md). Standalone C API
output publication requires a filesystem supporting hard links (NTFS/ext4 are
examples); existing outputs are preserved.

## 4. Import and inspect your data

For DeepMD standard NPY, create `import.request.json` with explicit set order:

```json
{"system":"/absolute/deepmd-system","sets":["set.000","set.001"],"units":"EvAngstrom","pbc":"Periodic"}
```

```sh
omnimd-mlmd import-deepmd import.request.json new-dataset.jsonl
```

The supported subset requires `type_map.raw`, `type.raw`, C-order float64
coordinates and periodic boxes; E/F/V labels are optional. `NonPeriodic` requires
the explicit `nopbc` convention and no conflicting boxes. Mixed-type arrays,
parameters, object/pickle arrays and unsupported shapes/dtypes fail. No automatic
set selection, physical unit conversion or splitting occurs.

For labeled extxyz, create `import.request.json`:

```json
{"input":"/absolute/structures.extxyz","format":"Extxyz","type_map":["O","H"],"units":"EvAngstrom","pbc":"NonPeriodic"}
```

```sh
omnimd-mlmd import-xyz import.request.json new-dataset.jsonl
omnimd-mlmd inspect-dataset /absolute/new-dataset.jsonl new-inspection.json
```

Use `format:"Xyz"` for plain species/position XYZ. Periodic plain XYZ needs an
explicit row-vector `cell` in the request; periodic extxyz needs `Lattice`.
Only all-axis periodicity or no periodicity is supported. See
[the exact extxyz/NPY/JSONL layouts](engine/DATA.md) before converting tensors;
virial is not stress and no stress-to-virial conversion is inferred.

Inspection succeeds only after the complete header/frame/completed stream passes
validation. It hashes those exact bytes in the same pass, reports frame/atom and
PBC counts plus independent E/F/V coverage, and lists at most 32 missing E/F IDs.
A valid unlabeled dataset succeeds with `all_frames_have_energy_and_forces:false`;
it can be used for prediction, not this trainer. Source hashes are retained
provenance declarations, not verified source files. Label coverage alone does not
validate model type-map agreement, trainer size/rank/split limits or label quality.

## 5. Evaluate and train

Create `batch.request.json` using either backend's complete `options`:

```json
{"options":{"native_model":"/absolute/native.json","device":"cpu"},"dataset":"/absolute/dataset.jsonl"}
```

```sh
omnimd-mlmd batch-eval batch.request.json new-predictions.jsonl
```

The current engine streams one frame per combined evaluation and retains the
model. The last `completed` record contains metrics over actual supplied labels;
missing labels never become zeros. A failed process or missing completion is not
successful assessment. The integrated adapter-level multi-frame call is available
through `omnimd-deepmd`; the engine command does not yet pack frames into that API.

For a readout fit, adapt the complete `train.request.json` generated in step 2.
Keep both E and F labels, choose held-out systems/trajectory IDs deliberately,
and set physically appropriate loss scales. Then:

```sh
omnimd-mlmd train-native train.request.json
```

The wrapper runs the adjacent CPU `omnimd-native-train`. Output `model.json`
uses the same inference schema; `report.json` contains rank, held-out identities,
hashes, initial/final metrics and serialization checks. Readout training keeps
hidden parameters fixed by default: [full contract](native/TRAINING.md).

Full supported tanh-network training uses `omnimd-native-full-train-v1`; all
embedding/fitting weights, biases and timesteps train with exact E/F gradients.
Use the complete request in [FULL_TRAINING.md](native/FULL_TRAINING.md). It needs
an initialized supported model, fixed normalization and explicit scales; it is
not the official DeepMD `input.json` interface. To reproduce its existing tiny
full-network learning and exact checkpoint acceptance:

```sh
python mlmd/native/tools/replay_full_training.py --eval-exe /absolute/omnimd-native-eval --train-exe /absolute/omnimd-native-train --output-dir /absolute/new-full-training-run
```

Append `.exe` to executable paths on Windows. Full-network resume requests replace
`initial_model` with `checkpoint`, name a new output directory, retain the dataset
and fixed configuration, and specify additional steps; see the training document.
Training checkpoints are distinct from MD checkpoints. GPU, minibatch,
distributed and virial-loss training remain future work. Optional virial labels
are assessed, not included in the native loss.

## 6. Minimize, run MD and restart

The generated `md.request.json` already contains explicit masses, time units,
initial velocities, ensemble, frame reference and cadence. Adapt those values
for your real model/system rather than treating tutorial values as defaults.

```sh
omnimd-mlmd check md.request.json
omnimd-mlmd run md.request.json new-trajectory.jsonl
```

`check` performs an actual initial evaluation on the selected device. `run`
uses NVE (`{"kind":"nve"}`) or CSVR NVT, one combined E/F/V per step after
initialization, and explicit eV/angstrom/fs/u dynamics units. Mass order must
match atom order. The final checkpoint is `new-trajectory.jsonl.checkpoint.json`;
interval checkpoints include `.step-N`. Validate the terminal completion record.

A resume request uses the same model and timestep, with no fresh frame, masses,
velocities or thermal initialization:

```json
{"options":{"native_model":"/absolute/native.json","device":"cpu"},"restart":"/absolute/new-trajectory.jsonl.checkpoint.json","timestep_fs":0.001,"steps":10,"sample_every":1}
```

```sh
omnimd-mlmd run resume.request.json new-resumed.jsonl
```

`steps` is additional work. The checkpoint restores ensemble, COM policy and
full thermostat RNG state and verifies model SHA256. Exact continuation is
validated within the same binary/backend/environment; it is not guaranteed
across hardware, compiler or runtime changes.

For fixed-cell minimization use the complete request in
[WORKFLOW.md](WORKFLOW.md#4-minimize-at-fixed-cell) and run:

```sh
omnimd-mlmd minimize minimize.request.json new-minimization.jsonl
```

Inspect convergence and both force/energy criteria, not just the output geometry.
A limit or failed line search returns nonzero with an explicit reason. Exported
extxyz and dataset files contain geometry only; they are not labeled training
data. NPT, cell relaxation and constraints are not part of this runner.

## 7. Enable CUDA deliberately

First establish CPU E/F/V and dataset correctness. CUDA availability and supported
model shapes depend on the chosen path. Stop after a device/context/launch failure
and inspect it; a new CPU request is an explicit fallback. No automatic retry,
driver reset or change to global system settings is performed.

For native CUDA, use an existing compatible NVIDIA driver and CUDA Toolkit NVRTC.
The recorded Windows runtime is CUDA 12.8; the current loader uses the documented
CUDA 12 DLL layout. For installation on a new host follow the
[official NVIDIA Windows guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).
Native Rust build/link does not itself require CUDA libraries; actual execution
loads the Driver API and NVRTC. An already installed toolkit can be selected for
the current shell with `CUDA_PATH`; an explicit absolute `OMNIMD_NVRTC_LIBRARY`
selects its NVRTC library. See [runtime lookup and supported bounds](native/CUDA.md).

```sh
cargo build --release --locked --manifest-path mlmd/Cargo.toml -p omnimd-mlmd --features native-cuda --bin omnimd-mlmd
```

Change only native options to
`{"native_model":"/absolute/native.json","device":"cuda:0"}` and choose new
outputs for a paired batch/MD comparison. `cuda:N` selects an entry from an
inherited visibility mask; the dedicated worker exposes that entry as ordinal
zero. `model-info` with CUDA loads a context/model, so use `capabilities` when no
GPU interaction is intended. Rebuilding only the engine keeps the CPU trainer
beside it. CUDA readout feature generation additionally requires an explicit
training request device; full-network GPU training remains unsupported.

For official C API CUDA, no native-cuda feature is needed. Use a matching official
TensorFlow `.pb` runtime and add actual absolute CUDA runtime/CUPTI paths:

```json
{"library":"/absolute/runtime/lib/libdeepmd_c.so","model":"/absolute/model.pb","device":"cuda:0","type_map":["O","H"],"units":"EvAngstrom","cuda":{"runtime":"/absolute/runtime/lib/libcudart.so.12","cupti":"/absolute/runtime/lib/libcupti.so.12"}}
```

This is the `options` object. The adapter requires observed direct kernels and
synchronization during evaluation. A model-info result can still be marked
pending; it does not prove GPU computation. Different backend/runtime versions
need their own acceptance. [Recorded C API CUDA evidence](deepmd/CUDA_VALIDATION.md)
is Linux DeepMD 3.1.0, not Windows runtime acceptance or a 3.1.3 GPU claim.

Measure loading, first evaluation, warmed evaluation and complete MD separately.
Keep identical frames, precision, output/checkpoint cadence and hardware. Current
native cooperative-kernel integration measured a complete 20-step, 192-atom MD
job at 412 ms CUDA versus 662 ms previous CUDA and 1352 ms CPU. This is a bounded
same-window result including startup and IO; the earlier cache-only short run
was slower than CPU. Small C API GPU models were also slower. See [measured release scopes](RELEASE.md); GPU presence is not a speedup.

## 8. Diagnose failures and preserve provenance

| Symptom | Action |
| --- | --- |
| Cargo tries to build Torch/chemfiles | Use `--manifest-path mlmd/Cargo.toml`, not the root legacy workspace. |
| Offline dependency unavailable | Populate the locked cache using a normal authorized build; do not silently change dependency versions. |
| Trainer executable missing | Build `omnimd-native-train` in the same profile/target directory as the engine. |
| Relative model/dataset/library path rejected | Use absolute paths in engine requests; native training paths resolve relative to its request file. |
| Existing output error | Select a new file/directory; do not delete unrelated data to rerun. |
| Inspection fails at completion | Check truncation, consecutive IDs/indices, final count, trailing records and newline; do not manufacture missing labels. |
| All E/F labels present but training rejects | Check ordered type map, model activation/size bounds, split, rank and positive loss scales. |
| Conversion rejects graph/schema | Compare exact supported formats/profiles; do not rename files or remove guards. Use compatible official runtime inference where supported. |
| Dynamic library not found / ABI error | Match OS, architecture, DeepMD backend and dependent libraries; a Python import alone does not validate C API loading. |
| Native CUDA feature disabled | Build with `--features native-cuda`, or explicitly choose CPU. |
| CUDA/CUPTI/launch/synchronization error | Preserve the error and runtime paths; stop the failed run. Check exclusive device use and compatible libraries before a new attempt. |
| Nonconverged minimization | Read its stated stopping reason and criteria; an exported geometry is not proof of a minimum. |
| Restart model mismatch | Use the exact model bytes and compatible checkpoint; changing trained weights starts a new trajectory. |

Retain the Git commit, commands, binary/model/dataset hashes, runtime versions,
request files, completed metrics and checkpoint. Generated data, weights and raw
logs stay in the external work directory. Compact committed receipts record
software acceptance without distributing those inputs. No private research model
or data is needed for this tutorial.

## 9. Compare models on the same dataset

Use [the model-deviation workflow](engine/MODEL_DEVIATION.md) to load 2..16 models
once and stream absolute E/F/virial population deviations on CPU. Start with an
inspected dataset, explicit member IDs and identical ordered type maps/units.
Run without a force window to inspect disagreements, then choose a justified
half-open max-force interval and review candidate IDs. The completed output binds
them to the consumed dataset SHA256. A candidate can enter existing MD or
minimization through its source `dataset_frame` index; no labeling or training is
triggered. The linked replay exercises independent Python statistical oracles,
label independence, invalid-input refusal and a two-step candidate MD job.

## 10. Optional CUDA-assisted frozen-readout features

Build the sibling `omnimd-native-train` with `--features cuda` and add
`"device": "cuda:0"` to a readout request only when intentionally selecting that
device. `train-native REQUEST.json` forwards the request to that executable.
The engine's own build feature does not prove which features its sibling trainer
contains. Keep `"device": "cpu"` (the default) for the normal tutorial.

Read [the precise training-device scope](native/TRAINING.md) first: CUDA computes
E/F/virial basis columns, while reference predictions, QR and assessment run on
CPU. Hidden weights stay frozen. Full-network CUDA requests reject before input
or output work. The recorded 3/4-atom feature microcase was slower on CUDA,
including 21.37 ms feature time versus 5.15 ms CPU and separate 195.88 ms CUDA
initialization. There is no full-network GPU training or speedup guarantee.

## 11. Extract reviewed candidates for reuse

Follow [dataset selection](engine/DATASET_SELECTION.md): take the completed
model-deviation dataset hash and reviewed IDs, create a strict selection request,
and run `select-dataset REQUEST.json NEW_DIRECTORY`. The new bundle retains
frame IDs and label values, gives them contiguous local indices, and records
source/output hashes and index mapping. Inspect it before assessment or fitting.
The linked replay takes six candidates through assessment and uses three already-
labeled records for CPU readout fitting; it refuses missing-label training.
External labeling remains a separate step for unlabeled candidates.

## 12. Import and simulate parameterized CPU C API models

Follow [the parameter guide](engine/PARAMETERS.md). Inspect runtime `df`/`da` and
stored-default availability, declare ordered parameter names/units, and opt into
dataset v2 using `parameter_layout` in the NPY import request. Supply exact
float64 frame and atomic arrays; no broadcasting or semantic inference occurs.
`batch-eval` then propagates each frame's conditions through one combined call.

A v2 `dataset_frame` binds its conditions for the entire new MD run. A direct
frame can instead supply versioned `fixed_parameters`, including ordered atom
species. Checkpoint v3 preserves that context, rejecting changes on restart.
Model-default policy must be explicitly selected; atomic rows remain explicit.
The TF/PT CPU replay includes same-species parameter-row permutation and every
moving-frame comparison with standalone C API. TF restart is exact; PT has
measured runtime first-call roundoff and numerical restart parity. Native,
engine CUDA, parameterized ensembles/minimization/training remain unsupported.
