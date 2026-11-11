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
omnimd-mlmd run-replicas replicas.json new-directory
```

The binary is under `mlmd/target/debug` (`.exe` on Windows).
`check` validates the configuration and performs the initial model evaluation
without writing a trajectory. The older two-path invocation is also accepted.

Both backends use the same run schema. Model/library paths must be absolute.
Windows JSON uses escaped backslashes. Output paths must be new; existing files
are never replaced. DeepMD inference uses a fresh device-isolated worker and one
trusted official model handle. CPU requests disable GPU visibility. Native CPU
inference needs no DeepMD, Python, GPU or LibTorch installation.

## The omnimd-gpu se_e2_a runtime

A second, independent GPU path runs native se_e2_a models of any network size
entirely on the device (see [guide step 28](../GETTING_STARTED.md#28-run-se_e2_a-on-the-gpu-runtime-and-compute-dataset-statistics)).
Build the engine with it:

    cargo build --release --locked --manifest-path mlmd/Cargo.toml -p omnimd-mlmd --features native-gpu --bin omnimd-mlmd

and select it in the options object of a `run`, `check`, `minimize`,
`batch-eval`, `model-info` or `model-deviation` request by adding `gpu_runtime`
to a native model on `cuda:N`:

    {"native_model":"/absolute/native-model.json","device":"cuda:0",
     "gpu_runtime":{"precision":"float32","compress_min_distance":0.5}}

- `gpu_runtime: {}` runs the networks in float64. `"precision":"float32"` runs
  them in float32; geometry, forces and virial stay float64.
- `compress_min_distance` (angstrom) tabulates the embedding networks for
  neighbor distances from that value to the cutoff, with `table_step` (default
  0.01) in the normalized radial input. The tables continue at ten times the
  step down to a fifth of that distance; a frame with an even closer pair is
  evaluated with the exact networks, so the value affects speed only.
- Without `gpu_runtime`, `cuda:N` keeps its earlier meaning (the bounded
  `native-cuda` path below). `gpu_runtime` with `device: cpu`, or in a build
  without `native-gpu`, is rejected before any worker starts.
- The reported model metadata has `backend: native-gpu-se-e2-a` and the network
  precision. Unlike the CPU path, a center with more neighbors than `sel` is
  not an error: neighbors are selected like DeepMD-kit 3.
- `model-deviation` accepts the GPU runtime when every member uses the same
  `cuda:N` with `gpu_runtime`; mixed devices are rejected. Members of one
  architecture with the same options are evaluated together, all members and
  a batch of frames in one device pass
  ([details](MODEL_DEVIATION.md#6-on-the-gpu-runtime-all-members-in-one-pass)).
- `batch-eval` on the GPU runtime evaluates the dataset a batch of frames at
  a time (up to 64 frames or 8192 atoms in one device pass); the predictions
  and their order are the same as frame by frame. A frame that cannot be
  evaluated fails the run with its dataset index and ID.
- The forces a center exerts on its neighbors are accumulated with atomic
  additions, whose order varies from run to run. Each evaluation is right to
  rounding (float64 forces within 1e-12 eV/angstrom of DeepMD-kit on the 83
  frames of the benchmark summary), but a trajectory is not reproduced bit for
  bit: two runs of the same request, or a run and its restart, drift apart and
  then agree only statistically. Exact replay is a property of the CPU path.

Each evaluation still passes one frame from the host and returns energy, forces
and virial to it, so very large systems pay a per-step host cost that the
standalone evaluator's `--bench` figure does not include.

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
limits and restart rules. The current GPU-loss route automatically retains parameters
and Adam moments on device until the original save boundaries; each update still
checks synchronized status, and complete snapshots are validated before saving.
No new request option is needed. CPU and CPU-loss defaults remain unchanged.

[Step 27](../GETTING_STARTED.md#27-use-resident-gpu-training-state-and-interpret-measured-speed)
records the combined transfer/residency/alignment changes: whole-process medians
versus `201cdef` fell 7.7% for the public 531-parameter workload and 2.4% for medium
synthetic work. Medium incremental benefit over transfer-only is unresolved; tiny
CPU remains faster (504.19 versus 703.56 ms GPU). Frequent checkpoints reduce savings.
The [integrated receipt](../evidence/native-resident-integrated.json) verifies actual
CLI forwarding, unchanged full-workload outputs and historical v8 resume. These
Windows-only results do not establish general speed or scientific model accuracy.

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

The [physical-label pilot and diagnosis](../PHYSICAL_DATA_VALIDATION.md#bounded-training-diagnosis)
show a loss-weight tradeoff, not scientific acceptance. Changing only energy
weight 1->0.1 improved validation force RMSE 4.164336->3.707381 eV/angstrom but
worsened E/atom RMSE 0.455950->0.523472 eV/atom. Training total-E RMSE rose above
initialization, 6.626241->6.650750 eV. Both models remain unsuitable, and the
43 diagnostic-validation frames are not an untouched final test. Two-step
CPU/GPU optimizer agreement does not prove full-trajectory or defect-free
execution. This 7-10-atom v8 case provides no v9 validation. See
[step 26](../GETTING_STARTED.md#26-interpret-the-physical-label-pilot-and-controlled-diagnosis).

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
| `ensemble` | `{"kind":"nve"}` (default), or the CSVR (NVT) or constant-pressure (NPT) settings below |
| `steps` | Additional steps in this invocation |
| `sample_every` | Positive trajectory sampling interval in steps |
| `deviation` | Optional: the other members of the model's ensemble and a force window; every sampled frame then carries their [model deviation](MODEL_DEVIATION.md#7-along-a-run-the-members-watch-the-sampled-frames) |
| `checkpoint_every` | Optional positive checkpoint interval; final checkpoint is always written |
| `restart` | Checkpoint path, exclusive with fresh frame/masses/velocity initialization |

A finite cell is periodic in all three directions; null is nonperiodic.
Coordinates in output are unwrapped. Energy, force and virial use eV,
eV/angstrom and eV. Virial is not stress or pressure; frames of periodic systems
carry the pressure separately (see below). No rotation is removed
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

For constant pressure (NPT) use:

```json
"ensemble": {
  "kind": "npt_csvr_crescale",
  "temperature_kelvin": 300,
  "tau_fs": 100,
  "seed": 42,
  "pressure_bar": 1.0,
  "barostat_tau_fs": 500,
  "compressibility_per_bar": 1e-6,
  "coupling": "triclinic",
  "barostat_every": 10
}
```

The thermostat is the CSVR thermostat above. The barostat is stochastic cell
rescaling ([Bernetti and Bussi, J. Chem. Phys. 153, 114107 (2020)](https://doi.org/10.1063/5.0020514);
fully flexible cell: [Del Tatto et al., Appl. Sci. 12, 1139 (2022)](https://doi.org/10.3390/app12031139)),
a first-order barostat with the noise that gives the volume fluctuations of the
isothermal-isobaric ensemble. It needs a periodic cell and nonzero kinetic
energy. [examples/native-npt.json](examples/native-npt.json) is a complete
request for the synthetic fixture.

| Field | Meaning |
| --- | --- |
| `pressure_bar` | Target pressure; any finite value |
| `barostat_tau_fs` | Relaxation time of the cell, exact when `compressibility_per_bar` is the real compressibility; at least `barostat_every * timestep_fs` |
| `compressibility_per_bar` | Assumed isothermal compressibility (1/bulk modulus). It only sets the time scale: twice the real value halves the relaxation time. Water is about 4.5e-5, oxides and phosphates about 1e-6 |
| `coupling` | `isotropic` (default): one scale factor for all moving axes, the cell keeps its shape. `semi_isotropic`: one factor for x and y and another for z, for a layer or an interface normal to z. `anisotropic`: independent factors along the moving axes, no shear. `triclinic`: all six strain components. With all axes moving these correspond to `iso`, `x y z` with `couple xy`, `aniso` and `tri` of LAMMPS `fix npt` |
| `axes` | The Cartesian axes along which the cell may change, as their letters (default `"xyz"`). `"xy"` holds z fixed: the setting for a slab with vacuum above it, where the pressure along z means nothing. `"z"` changes only the height. `semi_isotropic` and `triclinic` need all three |
| `barostat_every` | Steps between cell moves (default 10). Each move costs one model evaluation, so 10 adds 10 percent |

Every `barostat_every` steps, after the thermostat, the cell, the positions
and the velocities are deformed together by `exp(strain)`, with

```text
strain = -g (P0 - P) + sqrt(2 kT g / V) * noise,   g = compressibility * barostat_every * timestep / (3 barostat_tau)
```

where `P` is the instantaneous pressure tensor
`(sum m v v^T + virial) / V`. Positions and lattice vectors are multiplied by
the deformation and velocities by its inverse transpose. The diagonal
couplings give every group of axes that share a scale factor the mean of its
diagonal strain entries, with the noise of that mean: all moving axes for
`isotropic`, x and y for `semi_isotropic`, every axis on its own for
`anisotropic`. `triclinic` uses the full matrix with the rotation removed, so
the cell does not tumble. Along an axis left out of `axes` the cell vectors,
positions and velocities keep their components. The model is then evaluated at the new geometry, and the move is
accepted or refused by a Metropolis-Hastings test on the energy and pressure
found there. With that test the barostat samples the ensemble exactly for any
`barostat_tau_fs`; settings that are too aggressive show up as refused moves
(`barostat_refused` in the `completed` record), not as a wrong volume
distribution. A move whose strain exceeds 5 percent stops the run with an
error: lengthen `barostat_tau_fs`, lower `compressibility_per_bar`, or relax
the structure first.

What was checked ([`src/barostat.rs`](src/barostat.rs), [`tests/npt.rs`](tests/npt.rs)):
for an ideal gas whose cell has a shape energy, the sampled volume has the
mean `(N + 1) kT / P` and the variance of the exact distribution for the four
couplings with all axes moving, and so has the area times the fixed height
with `axes: "xy"` and the height times the fixed area with `axes: "z"`; the
strain fluctuations satisfy the virial balance of the ensemble along the
directions a setting samples (between x and y, in shear, between the layer
and its normal for `semi_isotropic`); a move and its reverse have opposite
acceptance ratios and the move statistics follow the equation above for
eight combinations of coupling and axes; eleven single-line errors introduced
into the first version of the barostat and eight into the code for groups of
axes and fixed axes each fail these tests. The cell move maps cell, positions
and velocities as stated, a refused or failed move leaves the state
untouched, and the native model's virial equals minus the strain derivative
of its energy. With forces, on a 32-atom Lennard-Jones crystal run through
the engine, the diagonal average of `V (P - P0)` is `-kT` to within 0.3 kT for
the isotropic, anisotropic and triclinic couplings over four million steps
each. The ignored test of `tests/npt.rs` repeats this in a release build with
two million steps for all four couplings and for cells with only x and y or
only z moving: the average over each group of axes that moves together is
`-kT` within 2.5 standard errors, which are 0.3 to 0.7 kT (kT / V is about 7 bar
there, so this resolves the mean pressure to a few bar). On a
trained five-element model, 400 ps with a fully flexible cell
give the mean volume of a LAMMPS Nose-Hoover run of the same model to 0.01
percent, within its error of 0.03 percent
([benchmark summary](../bench/results/SUMMARY.md)). This is not a validation
of any model's equation of state.

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

Frames of periodic systems also carry `volume_angstrom3`, `pressure_bar` and
`pressure_tensor_bar`: the instantaneous pressure `(sum m v v^T + virial) / V`
of that frame, in every ensemble.

A constant-pressure run writes JSONL v4: the cell differs from frame to frame,
frames add `barostat_work_ev` (the kinetic plus potential energy the accepted
cell moves delivered), the effective energy is
`E_total - Q_thermostat - W_barostat`, and the `completed` record adds
`barostat_work_ev`, `barostat_moves` and `barostat_refused`.

For NVT and NPT, total energy fluctuates due to heat and work exchange. The NVE drift field is
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
Old v1 NVE checkpoints remain readable. A constant-pressure run writes checkpoint
v4, which adds the accumulated barostat work and the counts of tried and refused
cell moves; cell moves fall on absolute step numbers, so a restart keeps their
schedule. E/F/virial are recomputed on load;
JSON uses float64 round-trip parsing. Exact replay is validated with the same
binary/backend/environment, not promised across compilers, hardware or runtimes.

The current System is exposed read-only to protect cached model outputs.
Failed numerical/model/thermostat updates leave the published System unchanged;
the CSVR wrapper advances its RNG only after a successful dynamics step.

MPI, constraints, surface-tension pressure coupling, classical force-field mixing and legacy
trajectory plugins are not enabled by this CLI. Explicit CUDA inference and the
`train-native` wrapper (CPU default; optional CUDA readout features or full gradients/Adam) are available within their documented scopes.

For executable setup, model integration, dataset inspection and a complete CPU
tutorial, see [GETTING_STARTED.md](../GETTING_STARTED.md). The
[versioned matrix](../FEATURE_MATRIX.md) distinguishes implemented, delegated and
missing DeepMD functions.

## Several runs at once

`omnimd-mlmd run-replicas REPLICAS.json NEW_DIRECTORY` advances several
independent systems under one model in step. Every step asks the model for all
of them in one batch, and once more for the systems whose barostat moves the
cell at that step. On the [GPU runtime](#the-omnimd-gpu-se_e2_a-runtime) a
batch is one pass over the device, so several small systems cost little more
than one; other backends evaluate the systems one after the other and gain
nothing.

```json
{
  "schema": "omnimd-replicas-v1",
  "options": {"native_model": "/absolute/model.json", "device": "cuda:0",
              "gpu_runtime": {"precision": "float32", "compress_min_distance": 0.5}},
  "timestep_fs": 0.5,
  "steps": 100000,
  "sample_every": 50,
  "checkpoint_every": 20000,
  "replicas": [
    {"name": "bulk-300K",
     "frame": {"positions": [[0.0, 0.0, 0.0]], "species": ["Ca"], "cell": null, "units": "EvAngstrom"},
     "masses": [40.078],
     "initialize_velocities": {"temperature_kelvin": 300.0, "seed": 1, "rescale_to_temperature": true},
     "remove_center_of_mass": false,
     "ensemble": {"kind": "nvt_csvr", "temperature_kelvin": 300.0, "tau_fs": 50.0, "seed": 11}},
    {"name": "bulk-500K", "restart": "/absolute/earlier/bulk-500K.jsonl.checkpoint.json"}
  ]
}
```

- A replica takes the fields of a `run` request that describe its system:
  `frame` or `dataset_frame`, `masses`, `velocities` or
  `initialize_velocities`, `remove_center_of_mass` and `ensemble`, or `restart`
  alone. The systems need not be alike: sizes, cells, boundaries and ensembles
  may all differ. `options`, `timestep_fs`, `steps`, `sample_every`,
  `checkpoint_every` and `deviation` are shared. A run takes 1 to 1024 replicas; frame and
  atom parameters are not carried.
- `name` (1 to 64 characters of `[A-Za-z0-9_.-]`, not starting with a dot,
  distinct without regard to case) names the files: the trajectory
  `NEW_DIRECTORY/NAME.jsonl` and its checkpoints
  `NAME.jsonl.checkpoint.json` and `NAME.jsonl.step-N.checkpoint.json`. Each
  trajectory is what `run` writes for that system alone, with
  `replica: {name, index, of}` added to its header, so `import-trajectory`
  and a later `run` or `run-replicas` with `restart` read it as usual.
- On the CPU the trajectories and checkpoints are identical to those of the
  single runs, cell moves and random streams included. On the GPU runtime
  they agree as two runs of one request do (see the note on atomic additions
  above).
- `NEW_DIRECTORY/replicas.json` is written last and lists the trajectories
  and checkpoints, with the wall time of the steps. A directory without it
  holds incomplete trajectories. The directory must be new.
- A step is all or nothing: if it fails for one replica (a model error, a
  nonfinite energy, a thermostat or barostat failure), the run stops and no
  replica has taken that step. The error names the replica.

Both `run` and `run-replicas` print how fast their steps went on standard
error (trajectory output included, loading the model left out).


## Current integrated adapter and CUDA follow-up

The local integrated runtime 9b8d184 includes native cooperative fitting and
persistent host buffers. The [final receipt](../evidence/native-cuda-integrated-fit.json)
measures complete 20-step NVE separately from warmed evaluations and verifies
exact checkpoint continuation. Older measurements above describe their original
release snapshots; see [release scopes](../RELEASE.md) for before/after details.

C API multi-frame batching and atomic outputs remain in the separate standalone
CLI. Engine `batch-eval` retains one call per frame for the C API and the CPU
(the GPU runtime takes a batch of frames per pass); dataset v2 now propagates
CPU C API frame/atom parameters and explicit stored-default policy. Fixed-parameter
MD uses the same context for every evaluation and checkpoint v3. See
[parameter usage](PARAMETERS.md) and [the standalone guide](../deepmd/BACKEND_GUIDE.md).
Real TF/PT CPU comparisons passed. TF fixed-context restart is bitwise exact;
PT exhibits external-runtime first-call roundoff and has numerical restart parity.
No new C API GPU parameter/batch acceptance is claimed.

## Ensemble comparison and explicit subset extraction

[model-deviation](MODEL_DEVIATION.md) retains 2..16 models on the CPU or on
one GPU and streams absolute population E/F/virial deviations with optional
candidate marking.
`import-trajectory` turns the sampled frames of a finished run into the
unlabeled dataset it reads, so a run, its deviations and the selection of
candidates chain without leaving the CLI. `merge-datasets MERGE.json
NEW_DATASET.jsonl` with `{"schema": "omnimd-dataset-merge-v1", "datasets":
[...]}` concatenates 2 to 256 complete datasets that share type map, units and
virial convention: frames, labels and IDs are copied unchanged in the order
given, indices are renumbered, IDs must be distinct across the inputs, and the
SHA-256 of every input is recorded. The [workflow](../WORKFLOW.md#6-an-active-learning-iteration-on-the-gpu-runtime)
puts these commands in the order of one active-learning round.
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
