# MLIP dynamics and data CLI

This is a **separate JSON runner**, backed by the actual upstream
`omnimd-core::System` and the new fallible `omnimd-sim::md::MlipDynamics`
adapter. It reuses omniMD's units, particle/cell storage, Gaussian velocity
initialization, COM removal and CSVR thermostat. It does not run the legacy
`Simulation/MolecularDynamics/Propagator` pipeline or all its input/output,
control, constraint or ensemble features. The legacy force/propagation traits
are infallible and cannot safely propagate model errors or share cached E/F/V.

The minimal bridge now present is a velocity-only upstream `Thermostat` hook
inside the transactional MLIP step. A later full Simulation bridge needs fallible
propagation and snapshot/output interfaces that consume the cached evaluation;
adapting the old `System::forces()` path without those changes is not sufficient.

## Build and commands

```sh
cargo build --locked --manifest-path mlmd/Cargo.toml -p omnimd-mlmd
omnimd-mlmd --help
omnimd-mlmd check request.json
omnimd-mlmd run request.json new-trajectory.jsonl
```

The binary is under `mlmd/target/debug` (`.exe` on Windows).
`check` validates the configuration and performs the initial model evaluation
without writing a trajectory. The older two-path invocation is also accepted.

Both backends use the same run schema. Model/library paths must be absolute.
Windows JSON uses escaped backslashes. Output paths must be new; existing files
are never replaced. DeepMD inference uses a fresh device-isolated worker and one
trusted official model handle. CPU requests disable GPU visibility. Native CPU
inference needs no DeepMD, Python, GPU or LibTorch installation.

## Explicit native CUDA inference

CPU remains the default for native options, including small systems. No atom
count heuristic automatically enables GPU execution. The optional unified CLI
build is:

    cargo build --release --locked --manifest-path mlmd/Cargo.toml -p omnimd-mlmd --features native-cuda --bin omnimd-mlmd

For model-info, batch-eval, minimize, check or run, select the same model with:

    {"native_model":"/absolute/native-model.json","device":"cuda:0"}

This is the options object within the existing request. Omit device or use cpu
for the ordinary CPU path. Legacy {"Cuda":0} device spelling is also accepted.
A native CUDA request without the build feature fails explicitly before a worker
or CUDA runtime is started. CUDA failures are errors; CPU fallback is an explicit
new request, not an automatic retry. The train-native command defaults to CPU;
optional CUDA readout features and bounded full-network gradients/Adam require a
CUDA-enabled sibling trainer and explicit request device.

Both inference backends share the fresh-process selector: cuda:N selects index
N from an inherited CUDA_VISIBLE_DEVICES list, or system CUDA index N when no
mask is set. The child exposes just that selected device and the backend uses
visible ordinal zero. Disabled/invalid masks and mismatched worker markers fail.
See NVIDIA's [device enumeration rules](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-visible-devices).
Parent process visibility is not mutated. Native CUDA uses the installed NVIDIA
Driver API and NVRTC, not the C API backend's cuda.runtime/cuda.cupti options;
see [native CUDA runtime and bounds](../native/CUDA.md). Even model-info with a
native CUDA request creates its context/compiled model; it is not a GPU-free query.

B's standalone native CUDA evaluator passed E/F/virial parity on three fixture
sets. The 216-atom case measured 60.385 ms CPU versus 52.288 ms CUDA (1.155x), while
two small cases were about 8.4x slower. Timings include CPU geometry, transfers,
kernels and final reduction, exclude initialization, and were affected by desktop
graphics load. This is inference only: CPU PBC/neighbor geometry and final E/F/V
reduction remain. The separate bounded training route also retains CPU stages;
no general speedup is claimed.

The unified CLI passed an eight-step native CUDA NVE comparison with CPU on the
original four-atom triclinic tanh fixture, plus continuation from step four.
Maximum E/F/virial differences were 0 / 6.66e-16 / 8.88e-16; positions and
velocities matched exactly. All resumed snapshots and final checkpoint state
matched the continuous CUDA run exactly after JSON decoding. See the compact
[acceptance evidence](../evidence/native-cuda-cli.json). This debug-build check
exercises routing, MD and restart, not performance or long-time stability.

Reproduce only in a coordinated GPU window with the feature-enabled binary and
existing official CUDA runtime (the output directory must be new):

    python mlmd/scripts/validate_native_cuda_cli.py --binary /path/omnimd-mlmd --work /new/native-cli-replay --run-gpu

The separate DeepMD C API CUDA path also has recorded public-model numerical
agreement, verified kernel launches, four-frame batch evaluation and a two-step
MD smoke test; see [its report](../deepmd/CUDA_VALIDATION.md). The six-atom C API
sample was about 8.27 times slower on GPU including CUPTI/synchronization. This
runtime-based measurement is separate from the native benchmark above.

The static capabilities command lists both paths, their evidence and runtime
requirements without loading a model or probing GPU availability on this host.
Bounded CUDA full-gradient/Adam training is a later local extension. These inference changes are published on shawn_dev
as 907481708ae5f3df4228a9fb5d5b1110c39f5e23.

## Native training and optional GPU loss

`omnimd-mlmd train-native REQUEST.json` forwards to the adjacent native trainer.
Build both binaries when selecting native CUDA training:

```sh
cargo build --release --locked --offline --manifest-path mlmd/Cargo.toml -p omnimd-mlmd -p omnimd-native --features omnimd-mlmd/native-cuda --bins
```

CPU training is the default. A valid full-network request with explicit CUDA may
select `config.cuda_gradient="reverse"`, `config.cuda_tiling` and optional
`config.cuda_loss="gpu"`. That last option computes tiled E/F/V and detached seeds
on GPU, reuses primal intermediates and selects v8. Absent or `"cpu"` retains the
historical CPU-loss path (v7 with tiling). Geometry, reduction/clipping and assessment remain
CPU work; changing loss mode requires a new optimizer, not a checkpoint edit.

Use [step 23](../GETTING_STARTED.md#23-select-gpu-prediction-and-loss-preparation-explicitly)
and [the native contract](../native/DEVICE_LOSS_TRAINING.md) for required fields,
limits and exact restart rules. The final [engine receipt](../evidence/native-device-loss-integrated.json)
verifies historical v7 and opt-in v8. Synthetic whole-process medians regressed
605->756 ms for 112 parameters and improved 3506->3061 ms for 466 parameters;
keep this option explicit. These measurements do not establish general speed.

The unchanged `06439f1` engine and sibling trainer also passed the recorded
[Linux native CUDA check](../LINUX_CUDA_VALIDATION.md): Ubuntu 22.04.5/WSL2
x86_64, RTX 4090, driver 617.42/NVRTC 12.9.86, one inference fixture and 24-frame
tiled v7/v8 training with exact same-runtime v8 continuation. C API execution,
other Linux configurations and cross-platform checkpoint restore were outside
that check. Use [step 24](../GETTING_STARTED.md#24-use-the-recorded-linux-native-cuda-configuration)
for the exact environment and limits; no performance advantage is established.
That Linux check does not cover the subsequent large-frame v9 path.

Explicit `config.cuda_tiling.max_atoms_per_frame=64..256` now admits larger frames
under unchanged work/payload limits. Default 64 preserves historical v7/v8;
CPU/untiled training stays 64. Above 64 uses v9, with GPU-loss backend v2 and complete
strided force/virial gather. The 112-parameter fixture fits 256; the 466-parameter
example has effective capacity 138 and rejects 256. Use
[step 25](../GETTING_STARTED.md#25-opt-in-to-resource-bounded-larger-training-frames)
and the [final receipt](../evidence/native-large-frame-integrated.json) for actual
256-atom forwarding, exact 2+2 restart and refusals. Capacity expansion does not
promise speed; new v9 Linux execution remains unverified.

## Dataset workflows

See [DATA.md](DATA.md) for explicit DeepMD NPY import, batch predictions and
E/F/virial error statistics, model/capability queries, and dataset-frame MD input.

See [MINIMIZATION.md](MINIMIZATION.md) for fixed-cell relaxation and explicit
convergence results. The [end-to-end workflow](../WORKFLOW.md) connects model and
structure conversion, batch assessment, bounded readout fitting, minimization and MD.

## Configuration and units

Copy [examples/native-nvt.json](examples/native-nvt.json) and set its model path
to the checked-in `mlmd/native/fixtures/synthetic-v1.json`. This is a synthetic
software fixture, not a physical production model.

| Field | Meaning |
| --- | --- |
| `dynamics_units` | `"EvAngstromFsU"`; only supported system (also the legacy default) |
| `frame` | Shared Frame: angstrom coordinates, row lattice vectors, `EvAngstrom` units |
| `masses` | Per-atom masses in u, positive and finite |
| `velocities` | Per-atom Cartesian velocities in angstrom/fs |
| `initialize_velocities` | Alternative to explicit velocities: positive `temperature_kelvin`, required `seed`, and `rescale_to_temperature` boolean |
| `remove_center_of_mass` | Remove mass-weighted translation once, use 3N-3 DOF; default false (3N DOF) |
| `timestep_fs` | Positive finite timestep in fs |
| `ensemble` | `{"kind":"nve"}` (default), or CSVR settings below |
| `steps` | Additional steps in this invocation |
| `sample_every` | Positive trajectory sampling interval in steps |
| `checkpoint_every` | Optional positive checkpoint interval; final checkpoint is always written |
| `restart` | Checkpoint path, exclusive with fresh frame/masses/velocity initialization |

A finite cell is periodic in all three directions; null is nonperiodic.
Coordinates in output are unwrapped. Energy, force and virial use eV,
eV/angstrom and eV. Virial is not stress or pressure. No rotation is removed
during the new initialization path. One-atom COM removal is rejected.

Gaussian initialization reuses upstream BoltzmannVelocities and its seeded
XorShift stream. Rescaling enforces the initial kinetic temperature exactly;
it removes the initial canonical kinetic-energy fluctuation, so equilibrate
before collecting samples. With rescaling disabled, the instantaneous initial
temperature fluctuates. Explicit velocities are not temperature-rescaled.

For NVT use:

```json
"ensemble": {
  "kind": "nvt_csvr",
  "temperature_kelvin": 300,
  "tau_fs": 100,
  "seed": 42
}
```

This calls the upstream CSVR algorithm after each velocity-Verlet step. Initial
kinetic energy must be nonzero. The wrapper requires finite positive temperature
and `tau_fs >= timestep_fs`, matching the upstream API's coupling restriction.
CSVR draws use a seeded ChaCha8 stream, separate from initialization.

The upstream CSVR noise was corrected to use an independent chi-square variate
for every DOF parity, a signed rescaling factor and a stable sum-of-squares form.
References: [Bussi et al., Appendix A](https://arxiv.org/pdf/0803.4060) and
[sign convention, Eq. A6d](https://arxiv.org/pdf/0901.0779).
See the validation record for small statistical and oscillator tests; this is
not broad production-ensemble validation.

For the C API backend replace the native `options` object with:

```json
{
  "library": "/opt/deepmd-kit/lib/libdeepmd_c.so",
  "model": "/absolute/public-model.pb",
  "device": "Cpu",
  "type_map": ["O","H"],
  "units": "EvAngstrom"
}
```

The native options are `{"native_model":"/absolute/model.json"}`.
For supported DeepMD .dp/YAML conversion, use the separate
[converter](../native/CONVERSION.md) with explicit units and source geometry;
conversion options are not inferred by the MD runner.

## Trajectory and restart

JSONL v2 contains a header, initial/sampled/final frames and a final `completed`
record. Each frame includes cached E/F/virial, kinetic/total energy, temperature,
velocities, time, thermostat heat delivered to the system and effective energy
`E_total - Q_thermostat`. Each step evaluates the model once; snapshots do not
trigger another model call.

For NVT, total energy fluctuates due to heat exchange. The NVE drift field is
null and the effective-energy drift is reported separately. Neither a reasonable
temperature nor a small short-run drift establishes physical model validity.

The final checkpoint is `OUTPUT.jsonl.checkpoint.json`; interval checkpoints
are `OUTPUT.jsonl.step-N.checkpoint.json`, where N is the absolute step. Intervals
are measured from the start of each invocation. Writes use new files only.
A failed/interrupted run exits nonzero and lacks a completed record; a partial
checkpoint write must not be used.

To resume, supply `options`, `restart`, the same `timestep_fs`, `steps` and
`sample_every`. Omit fresh frame/masses/velocities/initialization. The saved
ensemble and COM policy are restored automatically; an explicitly different
policy is rejected. Example:

```json
{
  "options": {"native_model": "/absolute/model.json"},
  "restart": "/absolute/previous.jsonl.checkpoint.json",
  "timestep_fs": 0.01,
  "steps": 100,
  "sample_every": 10
}
```

Checkpoint v2 stores coordinates/cell/species, masses, velocities, step,
timestep, model-file SHA-256, COM/DOF policy, ensemble, accumulated heat and full
ChaCha8 RNG state. A seed alone is insufficient for stochastic continuation.
Old v1 NVE checkpoints remain readable. E/F/virial are recomputed on load;
JSON uses float64 round-trip parsing. Exact replay is validated with the same
binary/backend/environment, not promised across compilers, hardware or runtimes.

The current System is exposed read-only to protect cached model outputs.
Failed numerical/model/thermostat updates leave the published System unchanged;
the CSVR wrapper advances its RNG only after a successful dynamics step.

MPI, NPT/barostats, constraints, classical force-field mixing and legacy
trajectory plugins are not enabled by this CLI. Explicit CUDA inference and the
`train-native` wrapper (CPU default; optional CUDA readout features or full gradients/Adam) are available within their documented scopes.

For executable setup, model integration, dataset inspection and a complete CPU
tutorial, see [GETTING_STARTED.md](../GETTING_STARTED.md). The
[versioned matrix](../FEATURE_MATRIX.md) distinguishes implemented, delegated and
missing DeepMD functions.


## Current integrated adapter and CUDA follow-up

The local integrated runtime 9b8d184 includes native cooperative fitting and
persistent host buffers. The [final receipt](../evidence/native-cuda-integrated-fit.json)
measures complete 20-step NVE separately from warmed evaluations and verifies
exact checkpoint continuation. Older measurements above describe their original
release snapshots; see [release scopes](../RELEASE.md) for before/after details.

C API multi-frame batching and atomic outputs remain in the separate standalone
CLI. Engine `batch-eval` retains one call per frame; dataset v2 now propagates
CPU C API frame/atom parameters and explicit stored-default policy. Fixed-parameter
MD uses the same context for every evaluation and checkpoint v3. See
[parameter usage](PARAMETERS.md) and [the standalone guide](../deepmd/BACKEND_GUIDE.md).
Real TF/PT CPU comparisons passed. TF fixed-context restart is bitwise exact;
PT exhibits external-runtime first-call roundoff and has numerical restart parity.
No new C API GPU parameter/batch acceptance is claimed.

## Ensemble comparison and explicit subset extraction

[model-deviation](MODEL_DEVIATION.md) retains 2..16 models on CPU and streams
absolute population E/F/virial deviations with optional candidate marking.
[select-dataset](DATASET_SELECTION.md) takes explicit IDs and an expected source
SHA256, validates the entire input, then creates a new shared-format subset and
receipt. It preserves scientific values and missing labels, with source-order
IDs and explicit old/new indices. The executable replay connects these commands
to inspection, assessment and CPU readout training without generating labels.

Local source `dc5449e` additionally supplies explicit CUDA frozen-readout E/F/V
features through the sibling trainer. QR and assessment remain CPU and the
recorded microcase was slower than CPU. Local source `f4dbf62` adds bounded
[full-network CUDA gradients/Adam](../native/CUDA_FULL_TRAINING.md), retaining CPU
inference, loss seeds, geometry, reduction and assessment. See
[training-device rules](../native/TRAINING.md). Engine compilation alone does not
prove its sibling trainer was built with CUDA support.

## External label return

`prepare-labels REQUEST.json NEW_DIRECTORY` exports exact frame identities from
an explicitly hash-bound dataset. `import-labels REQUEST.json NEW_DIRECTORY`
accepts a complete hash-bound return with explicit units, geometry/context and
E/F labels, preserving source order and v2 parameters. Duplicate/missing frames,
changed identities and conflicting existing labels reject before output creation.
See [the complete return-file contract and replay](EXTERNAL_LABELS.md). Neither
command starts a calculator, model or GPU. Existing training is a separate command.

## Optional virial loss in the sibling full trainer

The engine forwards `config.virial_loss` unchanged for full-network requests;
see [the complete loss/checkpoint contract](../native/VIRIAL_TRAINING.md).
Explicit positive weight and eV/atom scale enable all-nine-component row-strain
virial supervision on CPU or bounded CUDA, with checkpoint v3 for CPU/hyperdual
and v4 for explicit reverse execution. The readout QR
mode remains E/F-only. Absent/null V retains E/F CPU-v1 and CUDA-v2 behavior.
The [final integration receipt](../evidence/native-virial-integrated.json) records
short joint training, exact continuation, CPU parity and legacy CUDA-v2 resume.
