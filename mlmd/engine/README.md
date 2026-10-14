# Fallible MLIP molecular dynamics

`omnimd-mlmd` runs fixed-cell NVE velocity Verlet on the actual omniMD
`System` through `omnimd-sim::md::MlipDynamics<P: Potential>`. The initial
forces are evaluated before the first half kick; each subsequent step uses
one combined model evaluation. The model is loaded once per process.

Input coordinates/cell are in angstrom, velocity in angstrom/fs, mass in u,
and timestep in fs. Energy is eV, force eV/angstrom and virial eV.
The core stores lattice vectors as columns; conversion transposes them once.
Finite cells are fully periodic; null is nonperiodic. Positions are unwrapped.
The inherited core unit constants are used explicitly at the force/energy
boundary; they are not silently replaced.

## Run either backend

```sh
cargo run --locked --manifest-path mlmd/Cargo.toml -p omnimd-mlmd -- request.json new-trajectory.jsonl
```

Both the trajectory path and its appended `.checkpoint.json` path must be new.
Example using the checked-in synthetic native model (replace the absolute path):

```json
{
  "options": {"native_model": "/absolute/omniMD/mlmd/native/fixtures/synthetic-v1.json"},
  "frame": {
    "positions": [[0.2,0.3,0.4],[1.1,0.5,0.7],[0.6,1.4,0.9],[1.7,1.3,1.1]],
    "species": ["O","H","H","O"],
    "cell": [[3.7,0,0],[1.2,3.4,0],[0.6,0.5,3.8]],
    "units": "EvAngstrom"
  },
  "masses": [15.999,1.008,1.008,15.999],
  "velocities": [[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
  "timestep_fs": 0.01,
  "steps": 20,
  "sample_every": 10
}
```

For DeepMD, replace `options` with:

```json
{
  "library": "/opt/deepmd-kit/lib/libdeepmd_c.so",
  "model": "/absolute/public-model.pb",
  "device": "Cpu",
  "type_map": ["O","H"],
  "units": "EvAngstrom"
}
```

Only use a trusted official library with the supported C ABI. The CLI launches
a fresh child with GPU visibility disabled. Native inference needs no DeepMD
installation. Model/library paths must be absolute. Windows JSON paths use
escaped backslashes.

## Restart

Replace `frame`, `masses`, and `velocities` with the single key
`"restart": "/absolute/previous.jsonl.checkpoint.json"`. Keep `options`
and the same `timestep_fs`; `steps` is the number of additional steps.
The checkpoint contains positions, cell, species, velocities, masses, step,
timestep, schema version and model-file SHA-256. Schema/model/timestep changes
fail before integration. E/F/virial are recomputed after loading the checkpoint.
JSON parsing enables float64 round-trip precision.

Trajectory JSONL contains a header, initial/sampled/final frames and a final
`completed` record. Every frame includes cached E/F/virial, kinetic/total energy,
velocities, time and the shared Frame. A failed run exits nonzero and has no
completed record. The final checkpoint is written only after successful steps;
an interrupted or failed write must not be treated as a valid checkpoint.

## Scope

The library exposes the current System read-only to prevent stale cached forces.
A failed model call or nonfinite update leaves its published System and cached
evaluation unchanged. This does not rewind internal state of an arbitrary custom
Potential implementation. Existing classical/global potentials are rejected.

Thermostats, barostats, constraints, force-field mixing, legacy Simulation output
plugins, and training are not implemented by this runner. Synthetic fixture
trajectories demonstrate execution, not physical validity.
