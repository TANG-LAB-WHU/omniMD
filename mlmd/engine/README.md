# CPU MLIP dynamics CLI

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
are never replaced. Real DeepMD inference uses a fresh worker with GPU visibility
disabled and one trusted official model handle. Native inference needs no DeepMD,
Python, GPU or LibTorch installation.

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

GPU, MPI, NPT/barostats, constraints, classical force-field mixing, legacy
trajectory plugins and native training are not enabled by this CLI.
