# Minimal end-to-end local workflow

The same explicit units and Frame representation connect structure conversion,
prediction or bounded training, fixed-cell relaxation and MD. This page uses
native inference after model conversion; C API model options can replace native
options for prediction, minimization and MD.

## 1. Build and choose a supported model

    cargo build --locked --manifest-path mlmd/Cargo.toml -p omnimd-mlmd -p omnimd-native --bins
    omnimd-mlmd capabilities
    omnimd-mlmd model-info model.json new-model-info.json

Executables are under mlmd/target/debug. model.json contains:

    {"options":{"native_model":"/absolute/native-model.json"}}

Use an already supported native-v1 model, or explicitly convert the supported
DeepMD .dp/safe-YAML subset using [CONVERSION.md](native/CONVERSION.md).
The bounded .pb route is documented separately in [FROZEN.md](native/FROZEN.md);
only two audited graph profiles are supported. Runtime/model units are an
operator assertion. The checked-in synthetic model tests software, not chemistry.

## 2. Import structures and optional reference labels

    omnimd-mlmd import-xyz import.json new-data.jsonl

    {
      "input": "/absolute/structures.extxyz",
      "format": "Extxyz",
      "units": "EvAngstrom",
      "pbc": "Periodic",
      "type_map": ["O", "H"]
    }

Plain Xyz and explicit DeepMD NPY set import are also available; see
[DATA.md](engine/DATA.md). No reference labels, split or unit conversion are
inferred. Unknown physical fields or unsupported PBC fail explicitly.

## 3. Predict; optionally fit the last readout

    omnimd-mlmd batch-eval evaluate.json new-predictions.jsonl

    {
      "options":{"native_model":"/absolute/native-model.json"},
      "dataset":"/absolute/new-data.jsonl"
    }

The final record reports label coverage and E, E/atom, force-component and
virial-component MAE/RMSE/max residuals. Unlabeled structures still get predictions.

Training is an explicit separate command, requires E and F labels on every frame,
and currently fits only the final linear readout. Descriptor/embedding and hidden
layers remain frozen. Create a request following [TRAINING.md](native/TRAINING.md):

    {
      "schema":"omnimd-native-readout-train-v1",
      "initial_model":"/absolute/native-model.json",
      "dataset":"/absolute/new-data.jsonl",
      "output_directory":"/absolute/new-trained-directory",
      "config":{
        "split":{"mode":"explicit","validation_ids":["xyz/18","xyz/19"]},
        "fit_bias":false,
        "energy_weight":1.0,
        "force_weight":1.0,
        "energy_scale_ev_per_atom":1.0,
        "force_scale_ev_per_angstrom":0.01,
        "ridge_inverse_ev2":0.0,
        "rank_rtol":1e-12
      }
    }

    omnimd-mlmd train-native train.json

The wrapper runs the adjacent omnimd-native-train executable unchanged.
Choose validation IDs appropriate to your own dataset; the example IDs are
placeholders. Inspect training/validation metrics and rank in report.json.
Validation labels never enter the fit. Then use the saved model.json as
native_model for subsequent commands. Full-network training remains separate,
ongoing work and is not claimed by this readout command.

## 4. Minimize at fixed cell

    omnimd-mlmd minimize minimize.json new-minimum.jsonl

    {
      "options":{"native_model":"/absolute/new-trained-directory/model.json"},
      "dataset_frame":{"dataset":"/absolute/new-data.jsonl","index":0},
      "minimization":{
        "force_tolerance_ev_per_angstrom":0.001,
        "energy_tolerance_ev":0.000001,
        "initial_step_angstrom2_per_ev":0.02,
        "max_displacement_angstrom":0.05,
        "max_backtracks":32,
        "max_iterations":500,
        "armijo":0.0001
      }
    }

Check the completed record's status, converged flag and both criteria. A
nonconverged run returns nonzero even though it exports the last accepted
geometry. A small example is not assurance of a physical/global minimum.
[MINIMIZATION.md](engine/MINIMIZATION.md) explains all outcomes.

Geometry-only outputs are new-minimum.jsonl.final.extxyz and
new-minimum.jsonl.final.dataset.jsonl. They carry no inferred training labels.

## 5. Initialize MD from the exported structure

    omnimd-mlmd check md.json
    omnimd-mlmd run md.json new-trajectory.jsonl

For an illustrative three-atom O/H/H structure, md.json is:

    {
      "options":{"native_model":"/absolute/new-trained-directory/model.json"},
      "dataset_frame":{"dataset":"/absolute/new-minimum.jsonl.final.dataset.jsonl","index":0},
      "masses":[15.999,1.008,1.008],
      "dynamics_units":"EvAngstromFsU",
      "initialize_velocities":{"temperature_kelvin":300,"seed":42,"rescale_to_temperature":true},
      "remove_center_of_mass":true,
      "ensemble":{"kind":"nve"},
      "timestep_fs":0.01,
      "steps":100,
      "sample_every":10,
      "checkpoint_every":50
    }

Mass count and order must match the actual frame; no masses are guessed.
The timestep, temperature and stopping tolerances above are illustrative input,
not validated production settings. Check model suitability and equilibration
for the actual scientific problem. For seeded NVT and restart see
[engine README](engine/README.md).

## Reproducible software evidence

The stdlib/NumPy script validate_structure_workflow.py in mlmd/scripts accepts
existing synthetic teacher data, initial model and public regression model paths.
It performs actual import, prediction, explicit-heldout readout fitting, both
minimization backends, geometry export/reimport and short MD continuation.

The public regression model converged under the demo thresholds. The separately
fitted synthetic teacher hit the declared 300-step limit; its exported last
accepted frame was intentionally used for a five-step IO/integration check.
That check does not represent equilibration or a converged physical structure.
See [VALIDATION.md](VALIDATION.md) and the committed evidence for exact numbers.