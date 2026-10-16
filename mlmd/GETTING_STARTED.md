# Stepwise setup and use

This guide covers the MLIP workspace in `mlmd/`, tested at integrated runtime
9b8d184 (native source b6b7aa7 and C API source 1aee39c). Read the
[versioned feature matrix](FEATURE_MATRIX.md) before selecting a model. Native
Rust and official DeepMD FFI are distinct inference paths. CPU is the default;
CUDA is explicitly selected. Training defaults to CPU; optional CUDA readout
features keep QR/assessment on CPU. Bounded full-network CUDA gradients and Adam
retain CPU inference, geometry, reduction and assessment; see step 13.

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
Training checkpoints are distinct from MD checkpoints. Explicit deterministic
minibatching is described in step 18 and bounded indexed access in step 19;
sharded/distributed training remains future work. Optional full-network virial supervision is now
explicitly available on CPU/CUDA (step 15); readout fitting still uses E/F only.

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
training request device. Full-network CUDA gradients/Adam use the same explicit
selection with stricter shape bounds; see step 13.

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
CPU. Hidden weights stay frozen. The separate full-network route is described
in step 13. The recorded 3/4-atom feature microcase was slower on CUDA,
including 21.37 ms feature time versus 5.15 ms CPU and separate 195.88 ms CUDA
initialization. This measurement does not establish a training speedup.

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

## 13. Train the supported full network with explicit CUDA

Read [the CUDA full-training contract](native/CUDA_FULL_TRAINING.md) and confirm
its joint bounds before choosing a device: at most 2048 neural parameters,
width/descriptor width 32, 64 neighbor slots, 64 atoms/frame and the stated work
budget. Use tanh embedding/hidden layers and a linear output. Both E and F labels
are required. Normalization, geometry settings and atomic offsets remain fixed;
virial is assessed by default and can be explicitly enabled in the full-network
loss as described in step 15.

1. Build the sibling trainer explicitly with CUDA support:

   ```sh
   cargo build --release --locked --offline --manifest-path mlmd/Cargo.toml -p omnimd-native --features cuda --bins
   ```

2. Start with the complete full-training request in
   [FULL_TRAINING.md](native/FULL_TRAINING.md). Keep its dataset/model and explicit
   split, scales and optimizer settings; add `"device":"cuda:0"`, choose a new
   output directory, then run `omnimd-mlmd train-native REQUEST.json`.
3. Verify `report.json` reports `cuda_hyperdual_f64_v1`, nonzero parameter changes,
   completed optimizer steps, held-out metrics and reload checks. CUDA computes
   the neural gradients and Adam; CPU inference/loss seeds, geometry, reductions
   and assessment still run. A CUDA error stops the job; there is no fallback.
4. Resume using exactly one `checkpoint` instead of `initial_model`, the same
   device/dataset/fixed configuration, a new output directory and an additional
   step count. CUDA checkpoint v2 records backend, visible device, reported
   device information, driver API version and actual PTX hash. CPU v1 remains
   compatible. Changing backend intentionally starts a new optimizer from the
   exported native-v1 model, rather than editing checkpoint identity.

B's original synthetic 300-step replay and exact 150+150 continuation are recorded
in [its evidence](native/fixtures/cuda-full-training-report.json). Small fits were
slower on CUDA. The separate 3.46x result covers a warmed 64-atom full-gradient
stage, excluding initialization, Adam and checkpoint IO; it is not an end-to-end
training speedup. Linux CUDA, large-model scaling and chemical accuracy remain
unverified. Use an authorized GPU window for any replay.

The final integrated [CLI receipt](evidence/native-cuda-full-integrated.json)
checks six steps and exact 3+3 continuation through `train-native`, zero export
reload E/F/V differences and rejected CPU continuation of a CUDA checkpoint.
To repeat only this short acceptance with already generated original synthetic
inputs from B's replay (choose a new output directory and reserve the GPU):

```sh
python mlmd/scripts/validate_cuda_full_integration.py --engine /absolute/omnimd-mlmd --evaluator /absolute/omnimd-native-eval --fixtures /absolute/original-full-training-replay --work /new/integration-output
```

The fixture directory must contain `initial.json`, `dataset.jsonl`,
`frames.jsonl` and `continuous.request.json` from the linked full-training replay.
The sibling CUDA trainer must be beside the engine. The script never retries a
failed command. This short run verifies integration, not convergence, general
performance or cross-runtime reproducibility.

## 14. Return external labels and retrain explicitly

After reviewing model-deviation candidates and selecting a subset, follow the
[external label guide](engine/EXTERNAL_LABELS.md): inspect the immediate subset
hash, run `prepare-labels`, adapt external results into its exact return schema,
then bind the return-file hash and run `import-labels` into a new directory.
The returned geometry and parameter context must match exactly. Missing/duplicate
frames, shape/unit errors and conflicting existing labels fail. Existing virial
is retained if the return omits it. Source order and v2 parameters are preserved.

Inspect the labeled output before pointing a CPU training request at it. Dataset
v2 preservation does not add parameterized native training; use the documented
CPU C API route for parameterized assessment/MD. The synthetic acceptance closes
candidate selection, label import and v1 CPU retraining using saved software
fixture predictions, without launching an external calculation. Calculator
accuracy, convergence and provenance authenticity remain the user's scientific
validation responsibilities; no automatic job submission or active learning loop
is introduced.

## 15. Enable full-network virial supervision explicitly

Start from a full-network request in step 5 or 13, inspect label coverage, and
add this object inside `config`:

```json
"virial_loss": {"weight": 1.0, "scale_ev_per_atom": 0.1}
```

These example scales are not a physical recommendation. Choose scales and weights
for your declared labels. Both fields must be finite and positive; omit the
object or use `null` to retain E/F-only training. Every training/validation frame
then needs all nine finite `labels.virial` entries in eV, `MinusDEpsilonRow`.
E/F labels remain mandatory even if their loss weights are zero. This option
applies only to the full trainer, not the frozen-readout QR request.

Run the same `omnimd-mlmd train-native REQUEST.json` command, choosing CPU or an
explicit CUDA device with the separately built trainer. Read
[VIRIAL_TRAINING.md](native/VIRIAL_TRAINING.md) for the exact formula: each frame's
virial is normalized per physical atom, all nine components contribute, and no
volume/stress conversion or implicit symmetrization occurs. Inspect the separate
E/F, virial and total normalized objectives and virial-per-atom RMSE in the report.
Antisymmetric target components cannot be learned by this invariant energy model.

Full-batch virial supervision with CPU or default hyperdual writes checkpoint v3
with the immutable
`energy-force-row-virial-per-atom-nine-v1` loss contract and configured weight/scale.
Resume with the same dataset, split, objective and backend into a new output
directory, specifying additional steps. Changing/enabling/disabling the loss on
resume rejects. To change the objective intentionally, start a new optimizer
from exported `model.json`. Absent/null virial loss retains old E/F CPU-v1 and
CUDA-v2 checkpoint behavior. Existing CUDA execution identity checks still apply.

For a short final-binary integration replay, reserve the GPU and add
`--virial-loss` to the step-13 `validate_cuda_full_integration.py` command. It runs
six CUDA steps, exact 3+3 continuation, six CPU reference steps, reload/parity
checks and a changed-loss rejection. Optional `--legacy-ef-work` points to an old
six-step E/F integration directory to verify its CUDA-v2 step-3 continuation.
The script stops on failures without retry. B's separate 300-step evidence is in
[its report](native/fixtures/virial-training-report.json); this short replay does
not repeat gradient acceptance or establish physical accuracy or acceleration.

The [final integrated receipt](evidence/native-virial-integrated.json) confirms
this short v3 replay, CPU/CUDA exported E/F/V differences at most 4.44e-16 /
2.22e-16 / 2.22e-16, zero serialization/reload difference, rejected changed loss
and exact continuation of the previously saved E/F-only CUDA-v2 checkpoint.
These fixture results do not imply cross-platform/runtime exact reproducibility.

## 16. Select cached reverse CUDA gradients explicitly

Build the CUDA-enabled sibling trainer as in step 13. Use the same full-training
request and explicit `"device":"cuda:0"`, adding inside `config`:

```json
"cuda_gradient": "reverse"
```

Omitting this field preserves the original hyperdual executor. CPU remains the
default device; requesting reverse on CPU is an error. The same E/F and optional
virial loss configuration applies. Keep model, dataset, scales, split and all
existing shape/work limits unchanged. The reverse planner additionally checks
64 MiB of device-buffer payload; this excludes context/code/driver allocations
and does not bound total GPU memory.

Run `omnimd-mlmd train-native REQUEST.json` into a new output directory. Verify
`execution.backend="cuda_reverse_f64_v1"` and full-batch checkpoint v4. This records an
explicit E/F or E/F/V loss contract plus executor/device/driver/compiled-PTX
identity. Resume with the same configuration and additional steps; switching
executors on a nonzero-step checkpoint rejects. An intentional switch requires
starting a new optimizer from exported native-v1 weights. CPU v1, hyperdual v2
and hyperdual-virial v3 remain supported under their existing rules.

Read [the reverse execution contract](native/CUDA_REVERSE_TRAINING.md) before
interpreting performance. Neural forward intermediates are cached and reversed;
CPU inference/loss seeds, geometry, atom/frame reduction and assessment remain.
Reports expose phase totals, but these phases are not a whole-process benchmark.
B's 466-parameter/64-atom fixture measured median complete processes of 2963 ms
CPU, 1349 ms hyperdual and 877 ms reverse across three rotated-order repeats per
backend. The tiny 112-parameter case measured 615 ms CPU versus 1573 ms reverse.
The reference workload and output/checkpoint cadence matter; no universal
acceleration, larger accepted model or physical accuracy follows from these cases.

For only the short final-binary replay, add `--cuda-gradient reverse --virial-loss`
to `validate_cuda_full_integration.py` in step 13, reserve the GPU and choose a
new work directory. It checks six steps versus 3+3 exact continuation, a CPU
reference, export/reload and refused loss/device/executor changes. Independent
gradient and 300-step evidence remain in [B's report](native/fixtures/cuda-reverse-training-report.json).

The [final integrated reverse receipt](evidence/native-reverse-integrated.json)
records exact v4 model/moment continuation, six-step CPU/reverse parameter
agreement within 1.12e-16 and exported E/F/V differences of 4.44e-16 / 3.89e-16 /
4.44e-16. All 112 parameters changed and reload differences were zero. Loss,
device and executor changes rejected without output; this establishes no new
performance result or cross-runtime exact reproducibility.

## 17. Reuse fixed training geometry explicitly

Use the full-training request and CUDA-enabled sibling trainer from step 13,
with explicit `"device":"cuda:0"`. Either hyperdual or reverse gradients work;
add inside `config`:

```json
"fixed_geometry_cache": true
```

The default is false (omitted from serialized defaults). CPU-only training rejects
this option. Run the existing `omnimd-mlmd train-native REQUEST.json` command into
a new output directory. Dataset/split checks and eager cache construction happen
before CUDA context creation and optimizer updates. Only selected training frames
are retained; inspect `report.json`'s `fixed_geometry_cache` fields for frame/slot
count, retained payload, 64 MiB policy limit, construction time, hits and audit
geometry digest. The cap excludes allocator overhead, the original dataset,
current neural buffers and temporary construction buffers; it is not peak RAM.
Over-budget or construction failure rejects rather than silently falling back.

Reuse checks exact coordinate/cell bits, ordered types, units, type map, descriptor,
cutoffs, selection and normalization. Any mismatch invalidates the pool and stops;
recreate the trainer/cache to use changed geometry. It is not a moving-frame MD
neighbor cache. Full periodic displacements and normalized R/J are reused, while
current neural predictions, loss cotangents and directional rows are recomputed.
Assessment stays uncached and gradient input rows still upload each step.

Resume may enable or disable this runtime cache with the same numerical backend,
dataset, split, objective and execution identity. Cache arrays/stats are not saved
in checkpoints; enabled caches rebuild on restart. Existing checkpoint schemas
remain unchanged. Exact comparisons exclude only this runtime flag and requested
additional-step count. Do not combine a cache toggle with an executor/loss change.

The [native contract and measurements](native/FIXED_GEOMETRY_TRAINING.md) record
three alternating fresh-process runs per setting on 466-parameter, 64-atom,
eight-frame E/F/V fixtures for 100 updates. Median off/on processes are 865.5 /
818.4 ms for a cluster and 6946.3 / 956.9 ms for its triclinic periodic variant.
Both retain 1,290,058 bytes for six training frames. The periodic result reflects
avoided repeated CPU image enumeration in this bounded synthetic case. It does
not establish production scaling, physical accuracy or general GPU acceleration.

For a short integrated replay, extend the step-16 command with
`--fixed-geometry-cache --legacy-uncached-work PREVIOUS_REVERSE_WORK`, where the
previous directory contains the accepted uncached six-step reverse E/F/V replay
and its step-3 checkpoint. Reserve the GPU and choose a new work directory. The
script checks six cached steps, cached 3+3 continuation, on-to-off continuation,
old off-to-on continuation, CPU parity and unchanged refusal contracts. It stops
on failures without retry and does not rerun the native performance benchmarks.

The [final cache integration receipt](evidence/native-geometry-cache-integrated.json)
confirms reverse-executor six-step and 3+3 equality, on-to-off equality, and exact
off-to-on continuation from the prior `94427ef` checkpoint. Comparisons exclude
only requested step count and the cache flag. It retained 221,414 bytes for
18 frames, recording 108 hits over six steps and 54 over the three resumed steps.
CPU/reverse E/F/V differences were 4.44e-16 / 3.89e-16 / 4.44e-16, reload zero.
Native evidence separately covers both executors; this final short replay adds
no performance measurement or broader architecture/data support.

## 18. Select deterministic minibatches

Start a new full-network optimizer using the request from step 5 (CPU) or step 13
(explicit CUDA), adding this optional object inside `config`:

```json
"minibatch": {
  "batch_size": 7,
  "seed": 20261016,
  "order": "shuffle_each_epoch"
}
```

The example is for exercising the interface, not a physical training recommendation.
All fields are required: batch size is 1..1024, seed is an unsigned 64-bit integer
independent of the train/validation split seed, and order is `sequential` or
`shuffle_each_epoch`. Absent/null preserves full-batch defaults and serialization.
The option applies to the full neural-parameter trainer, not readout QR. On CUDA,
both gradient executors work and the fixed-geometry cache remains optional.

Use `omnimd-mlmd train-native REQUEST.json` with a new output directory. `steps`
is additional Adam updates, not epochs. Each epoch visits every selected training
frame once. With 18 training frames and batch size 7, sizes are 7,7,4; the tail is
neither dropped nor filled from the next epoch. Gradients use the actual batch
count, then clipping and one Adam update. Learning rate is constant by default;
step 21 adds an explicit update-indexed schedule.
Complete-partition metrics remain separate from the sampled training objective;
validation frames never enter batches. Smaller batches do not promise full-batch
convergence, equivalent epoch weighting or a throughput improvement.

Inspect checkpoint `omnimd-native-full-checkpoint-v5` (v6 with an explicit
learning-rate schedule, v7 with CUDA tiling) and `report.minibatch`:
start/end sampler, processed frame count, partial-batch count and size histogram.
Sampler state describes the next batch and includes seed/order, epoch/cursor,
permutation and derived RNG state. Resume from that checkpoint with the same
batch settings, data/split, loss and numerical backend, specifying additional
updates. Missing or inconsistent sampler state and changed batch size/seed/order
reject. Cache may toggle; its arrays rebuild, while the numerical sampler and
optimizer state remain exact within the recorded runtime identity.

Old full-batch v1-v4 checkpoints keep their original route. Do not enable
minibatching when resuming one; initialize a new optimizer from its exported
native model to change training semantics. The same model/shape/work bounds and
default 1024-frame in-memory dataset limit apply; step 19 provides explicit indexed
access with separate limits. The optional cache still prepares the
whole selected training pool and retains its own 64 MiB host-payload bound.
Minibatch selection alone changes neither storage mode nor learning-rate policy.

Read [the complete sampler/loss contract](native/MINIBATCH_TRAINING.md) for the
actual-count formula, portable integer shuffle and canonical next-batch state.
[Final engine acceptance](evidence/native-minibatch-integrated.json) records seven
updates over 18 training / six validation frames: 43 processed frames, two tails,
exact continuation at updates 2/3/4, both cache-toggle directions, independent
sampler agreement and exact prior-v4 full-batch continuation. CPU/reverse maximum
parameter/moment difference is 6.66e-16; E/F/V differences are 8.88e-16 / 6.66e-16 /
8.88e-16, with zero native export/reload difference. Five changed/corrupt sampler
requests reject before output. These are synthetic software checks.

For the short replay, use the final feature-enabled engine and evaluator, reserve
the GPU and provide the prior accepted six-step full-batch reverse E/F/V directory
and a new output directory:

```text
python mlmd/scripts/validate_minibatch_integration.py --engine ENGINE_EXE --evaluator EVAL_EXE --baseline-work PREVIOUS_FULL_BATCH_WORK --work NEW_WORK
```

It stops on failure without retry. The native 31-update and both-executor evidence
remain separate; this replay does not repeat their performance benchmarks.

## 19. Read training data through a bounded index

Start from the full-network minibatch request in step 18, on CPU or explicit CUDA.
Add this field at the request top level, alongside `dataset` and `device`:

```json
"dataset_access": "indexed"
```

Run `omnimd-mlmd train-native REQUEST.json` into a new output directory. Indexed
access requires explicit `config.minibatch`; it is not accepted by readout QR and
has no silent memory fallback. Omit the field or use `"memory"` for the historical
reader. The existing `omnimd-dataset-jsonl-v1` coordinates, labels, IDs, units and
source hashes are unchanged; this does not add native training for dataset v2.

The indexed scanner validates all input before training, retaining frame offsets,
lengths, IDs, atom counts and hashes of exact raw lines. Training, complete-partition
assessment and reload verification then decode one frame per read. Sampling and
gradient reduction order stay unchanged. No disk sidecar, prefetch worker or new
coordinate format is introduced.

Review [all parser/resource bounds](native/INDEXED_TRAINING.md) before choosing
this mode. Indexed input is a regular file up to 16 GiB and 100000 frames, with
256 KiB records, 256-byte IDs, at most 64 atoms/frame and a 64 MiB retained index
payload budget. Default memory full training still limits input to 128 MiB and
1024 frames. Architecture, AD/workspace, per-frame CUDA and geometry-cache limits
are unchanged. The indexed ceiling itself has not been benchmarked in full.
Index, sampler, partitions, report identities and checkpoints remain O(frame
count); a bounded index is not an O(1)-memory optimizer or a total RAM guarantee.

Inspect `report.dataset_access` for mode, frame/file count, index payload, record
buffer capacity, maximum decoded-frame payload, frame reads and IO/integrity
counts/times. Payload excludes allocator overhead, parser temporaries, construction
scratch and model/optimizer/report/checkpoint workspaces. Do not interpret
`frames_decoded_per_read:1` as a count of every live allocation.

Each read checks exact line SHA256, then frame index/ID/atom count. Before every
checkpoint, after final metrics and before CLI model/report publication, the
whole file must still match its original SHA256. This costs O(file bytes) per
verification; frequent checkpoints can increase IO substantially. Windows retains
a handle denying write/delete sharing. Other platforms' portable checks are not
an atomic snapshot against concurrent mutation; immutable input is required, and
Linux/macOS execution is not validated here. Failure invalidates the reader and
stops the invocation; earlier completed checkpoints remain available.

Storage mode is runtime-only: v5/v6/v7 checkpoints do not embed the index or change
schema when storage mode changes. Resume may switch memory/indexed with the identical raw input, numerical
backend, split, objective and sampler settings, provided the input satisfies the
destination mode's bounds. Even whitespace changes alter dataset identity. The
optional CUDA geometry cache can toggle at the same time, but still prepares the
entire selected training pool within its independent 64 MiB payload cap. A pool
that does not fit rejects before CUDA initialization; disable caching explicitly.
There is no partial/evicting geometry cache in this implementation.

[Final integrated acceptance](evidence/native-indexed-integrated.json) reuses the
24-frame synthetic minibatch fixture. Seven indexed CPU/CUDA updates match the
prior memory model, Adam, sampler, all four initial/final partition metrics and
exported model bytes exactly. Indexed-to-memory with cache off and old memory-to-
indexed with cache on resume exactly. The index retains 19936 bytes; the run has
nine whole-file verifications and zero export/reload difference. Three invalid
mode/config/hash requests reject before output. These short checks add no capacity
or performance benchmark.

[B's separate boundary evidence](native/fixtures/indexed-training-report.json)
uses 2051 frames deliberately padded with whitespace to 128.19 MiB, crosses both
old limits and resumes exactly. Its measured Windows peak working set is 12.83 MiB
and retained index is 319722 bytes. This exercises reader limits and memory
accounting; it is not representative physical-data throughput or a universal
memory bound. The full 100000-frame/16-GiB ceiling remains unmeasured.

For only the short engine integration replay, reserve the GPU, provide the prior
accepted seven-update memory minibatch directory (including its CPU reference)
and choose a new output directory:

```text
python mlmd/scripts/validate_indexed_integration.py --engine ENGINE_EXE --baseline-work PREVIOUS_MINIBATCH_WORK --work NEW_WORK
```

It stops on failure without retry and does not regenerate the padded capacity case.

## 20. Interpret startup validation and IO counters

Requests and checkpoints are unchanged. The trainer now retains the parameter
layout from its own complete validation, and the CLI avoids repeating that
validation immediately before entry. Public constructor and library validation
remain independent; there is no validation cache across calls. Indexed startup
still scans the file and checks frames. Every accessed line hash and every
checkpoint/publication whole-file hash remains required.

Full-training `report.json` now includes `startup_timing`:

| Field | Boundary |
| --- | --- |
| `dataset_open_ms` | Initial strict reader/index scan |
| `state_preparation_ms` | Initial model parsing and constructor validation, or checkpoint parsing on resume |
| `before_trainer_ms` | Cumulative CLI time through trainer entry; includes the preceding phases |
| `before_first_update_ms` | `before_trainer_ms + timing.startup_ms`; excludes process launch |

`timing.startup_ms` spans trainer validation, initialization and initial complete
train/validation assessments, ending before the first update. Existing
`timing.initialization_ms` ends before those assessments. These phases overlap;
do not sum all timing fields. Indexed-only `timing.startup_frame_reads` is the
cumulative logical read counter at the first-update boundary, including any
previous caller reads but excluding index construction. Timings/counters are
diagnostics, not checkpoint identity, physical accuracy or total memory measures.

[Final CPU integration](evidence/native-indexed-startup-integrated.json) reuses the
existing 24-frame input and saved pre-change outputs. Fresh seven-update frame
reads fall 211 to 163, with nine whole-file checks. Historical step-6-to-7 reads
fall 151 to 103, with three whole-file checks. Startup read counts are 72 and 48.
Checkpoint contents, model bytes, all four partition metrics and sampler match;
reload E/F/V differences are zero. Default memory agrees exactly. Zero updates,
CPU geometry cache and malformed resumed Adam moments reject before output.

Run this short acceptance with the prior `indexed-integrated-01` directory
containing `cpu-indexed` and `final-engine-cpu` outputs, and a new output directory:

```text
python mlmd/scripts/validate_indexed_startup_integration.py --engine ENGINE_EXE --baseline-work PREVIOUS_INDEXED_WORK --work NEW_WORK
```

This replay explicitly selects CPU and hides CUDA devices. It runs no GPU or
performance benchmark. The [separate native comparison](native/INDEXED_TRAINING.md#startup-validation-consolidation)
uses three alternating repeats with warm filesystem cache on 2051 deliberately
whitespace-padded frames; its timings cannot establish production throughput.

## 21. Select an optimizer-update learning-rate schedule

Start from the full-network request used above. Keep the positive finite base
`config.learning_rate` and optionally add `learning_rate_schedule` inside `config`:

```json
{
  "learning_rate": 0.005,
  "learning_rate_schedule": {
    "kind": "exponential",
    "decay_steps": 2,
    "decay_rate": 0.5
  }
}
```

Run `omnimd-mlmd train-native REQUEST.json` into a new output directory. The
engine forwards the request to its sibling native trainer. CPU is the default;
the same host float64 rate feeds either CUDA Adam executor when explicitly
selected. Readout QR does not accept an iterative learning-rate schedule.

For `k` globally completed optimizer updates, the next rate is
`learning_rate * decay_rate^floor(k/decay_steps)`. The example uses
0.005, 0.005, 0.0025, 0.0025, 0.00125, 0.00125, 0.000625 for updates 1..7.
Update 3 first uses the lower rate. `k` counts successful optimizer updates,
including short minibatches; it is neither frame nor epoch count and does not
reset on resume. Additional requested `steps` and Adam bias correction retain
their existing meanings. This is a native formula, not inferred DeepMD defaults.

Use `{"kind":"constant"}` for an explicit constant schedule. Omit the field or
set it to null to retain historical constant behavior and v1-v5 serialization.
Every explicit schedule, including constant, uses checkpoint v6 unless CUDA
tiling selects v7 (step 22). Resume requires
exact base rate, schedule presence/kind/interval/rate, together with existing
loss/data/backend/sampler identity. Runtime storage/cache choices retain their
previous compatibility rules. Changing or adding a schedule requires starting a
new optimizer from exported `model.json`; do not rewrite an old checkpoint.

`decay_steps` is an integer in 1..1000000000; finite `decay_rate` is in (0,1].
The complete requested used-rate interval must stay positive and finite in f64;
underflow to zero rejects before CUDA or output, with no automatic floor. Unknown
kinds or fields reject. There is no warmup, continuous/cosine/plateau scheduling,
scheduled loss weight, convergence guarantee or cross-platform bitwise guarantee.

Inspect `report.learning_rate` for the contract, effective schedule, base rate,
one-based first/last updates and first/last used rates in that invocation. A
resume from step 2 starts with update 3 and its decayed rate. No unbounded trace
or independent mutable scheduler state is stored.

[Final engine acceptance](evidence/native-learning-rate-integrated.json) uses the
existing 24-frame CPU fixture: seven indexed minibatch updates, exact continuation
at steps 2 and 3, identical exported model bytes/final metrics, and zero reload
E/F/V differences. Explicit constant matches historical v5 numerical state and
all four metrics; null resumes actual old v5 exactly. Invalid rate and changed/
removed resume schedules reject before output. Three CPU routing tests and the
offline release build pass. B's broader CPU/CUDA acceptance remains separate.

```text
python mlmd/scripts/validate_learning_rate_integration.py --engine ENGINE_EXE --baseline-work PREVIOUS_INDEXED_STARTUP_WORK --work NEW_WORK
```

The baseline is the accepted `indexed-startup-integrated-01` directory, including
`fresh` outputs. The short replay explicitly selects CPU, hides CUDA devices and
uses a new output directory. It runs no GPU or performance benchmark.

## 22. Batch frames in bounded CUDA tiles

Use the CUDA-enabled sibling trainer and an explicit `"device":"cuda:0"`.
In the full-network request's `config`, add:

```json
{
  "cuda_gradient": "reverse",
  "cuda_tiling": {"max_frames": 16, "max_atoms": 512}
}
```

Run `omnimd-mlmd train-native REQUEST.json` into a new output directory. Tiling
requires reverse CUDA; CPU or hyperdual requests reject. Omit or set `cuda_tiling`
to null to retain historical framewise behavior. Both full-batch and minibatch
training work; indexed access still requires minibatching. These capacities group
GPU execution, not optimizer samples. Each original batch retains its true frame
count, short tail, sampler order, schedule and single Adam update.

Frames are packed in order, never mixed geometrically or split. A tile uploads
packed inputs, launches four reverse-gradient kernels, synchronizes and downloads
atomic derivatives. Stable host reduction retains atom-within-frame then original
frame order, without tile subtotals. CPU prediction/loss seeds, geometry, reduction,
clipping, metrics and serialization remain. A failed tile prevents the entire
batch's Adam/sampler commit and stops training; prior checkpoints remain usable.

Requested limits are 1..32 frames and 1..1024 aggregate atoms per tile. The planner
may lower effective atom capacity to fit the 50-million-term work policy, 64 MiB
explicit device buffers and 32 MiB host tile payload. A single frame that cannot
fit rejects before CUDA; no partial frame or silent data truncation is allowed.
These payload limits exclude context/module/local memory, CPU workspaces, allocator,
index/cache/model/report overhead; they are not total VRAM/RSS limits. Supported
architecture remains at most 2048 parameters, width 32, 64 neighbor slots and
**64 atoms per frame**. See [the full policy](native/TILED_TRAINING.md).

Tiling selects checkpoint `omnimd-native-full-checkpoint-v7`, including when a
schedule is present, and executor `cuda_tiled_reverse_f64_v1`. Tile capacities
are immutable along with existing data/model/loss/sampler/schedule/runtime identity.
Storage and geometry-cache flags may change on resume. To change capacities or
switch to framewise execution, start a new optimizer explicitly from the exported
model. Same-runtime exact restart does not imply cross-platform/driver equality.

Inspect `report.cuda_tiling` for requested/effective capacities and planned budgets.
`cuda_gradient_profile` reports frames, tiles, four gradient kernels per tile,
synchronizations, maximum actual occupancy and observed explicit buffer payloads.
Launch counts exclude Adam. Phase timings and external whole-process timing have
different boundaries; do not treat payload counters as process memory.

[Final engine acceptance](evidence/native-tiled-integrated.json) uses the existing
24-frame fixture with 3/4 atoms per frame, seven scheduled updates and minibatches
7,7,4,7,7,4,7. Requested capacities are three frames / 16 atoms; actual occupancy
reaches three frames / 11 atoms. The 43 processed frames use 19 tiles and 76
gradient kernels, versus 172 kernels framewise. Model/Adam/sampler and all four
partition metrics match framewise exactly after excluding executor metadata;
exported model bytes match. Step-2 resume switches indexed to memory and cache on
to off, with exact final state except additional steps/cache flag. Reload E/F/V
differences are zero. CPU tiling, changed capacity and framewise resume reject.

For this short replay only, reserve the GPU and supply the accepted
`learning-rate-integrated-01` directory with its existing synthetic inputs:

```text
python mlmd/scripts/validate_tiled_integration.py --engine ENGINE_EXE --baseline-work PREVIOUS_LEARNING_RATE_WORK --work NEW_WORK
```

It stops on failure without retry. B's larger boundary/gradient tests and
three-repeat performance comparisons remain separate and are not repeated here.
The documented whole-process synthetic ratios do not establish physical accuracy,
production scaling or speed relative to DeepMD.
