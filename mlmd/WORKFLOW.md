# Minimal end-to-end local workflow

For installation and a fully executable synthetic CPU walkthrough, start with
[GETTING_STARTED.md](GETTING_STARTED.md). [FEATURE_MATRIX.md](FEATURE_MATRIX.md)
pins the official DeepMD comparison version and records implementation gaps.

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

## 3. Predict; optionally train the native model

    omnimd-mlmd batch-eval evaluate.json new-predictions.jsonl

    {
      "options":{"native_model":"/absolute/native-model.json"},
      "dataset":"/absolute/new-data.jsonl"
    }

The final record reports label coverage and E, E/atom, force-component and
virial-component MAE/RMSE/max residuals. Unlabeled structures still get predictions.

Training is an explicit separate command, requires E and F labels on every frame,
and offers two strict modes. The readout mode below fits only the final linear
readout with descriptor/embedding and hidden layers frozen; see
[TRAINING.md](native/TRAINING.md):

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
native_model for subsequent commands.

For all neural weights/biases/timesteps in the supported tanh network, use the
complete omnimd-native-full-train-v1 request in
[FULL_TRAINING.md](native/FULL_TRAINING.md) with the same train-native command.
That mode uses exact mixed E/F gradients and Adam, with checkpoint restart.
Normalization, geometry/type metadata and atomic offsets remain fixed; the
document lists bounds, independent one-side copies and unsupported activations.
Both training modes export the same native-v1 model for the steps below.

CPU stays the default. With a CUDA-enabled sibling trainer, full training also
supports reverse tiling and optional `config.cuda_loss="gpu"` for GPU E/F/V
predictions and detached seeds. GPU loss requires reverse+tiling and binds v8;
default CPU loss with tiling keeps v7, and geometry/reduction/assessment remain CPU work.
Follow [step 23](GETTING_STARTED.md#23-select-gpu-prediction-and-loss-preparation-explicitly)
for the complete selection/restart rules. Select this per workload: 128-update
synthetic medians were 605->756 ms (112 parameters, slower) and 3506->3061 ms
(466 parameters, 1.15x). No architecture or per-frame limit is raised.

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
## Final native CUDA cache acceptance

This optional inference-only follow-up is local, based on published a74804d.
CPU stays the default. Optional bounded CUDA gradients/Adam now retain CPU
inference, geometry, reduction and assessment; see [CUDA training](native/CUDA_FULL_TRAINING.md). The recorded run uses
existing original synthetic inputs and the already converted public regression
model. It does not use research data or an equilibrated-water accuracy dataset.

Preserve the baseline a74804d release CLI and CUDA checker in a separate output
location before building this candidate. Build both versions with the same
Rust/toolchain and release profile. The measured baseline checker adds only
host timing instrumentation; its hash is recorded. No CUDA library, driver or
system change is needed. Candidate build:

    cargo build --release --locked --offline --manifest-path mlmd/Cargo.toml -p omnimd-mlmd -p omnimd-native --features omnimd-mlmd/native-cuda,omnimd-native/cuda --bin omnimd-mlmd --bin omnimd-native-cuda-check

Prepare the existing inputs using native/tools/replay_cuda.py without --run-gpu
and native/tools/replay_cuda_infra.py, as described in [CUDA_INFRA.md](native/CUDA_INFRA.md).
Use the same 216-atom medium model/frame and 192-atom public-model water frame.
Then, only in an explicitly coordinated exclusive GPU window, run once:

```sh
python mlmd/scripts/validate_cuda_infra.py \
  --baseline-engine /saved/a74804d/omnimd-mlmd \
  --baseline-check /saved/a74804d/omnimd-native-cuda-check \
  --engine /current/mlmd/target/release/omnimd-mlmd \
  --check /current/mlmd/target/release/omnimd-native-cuda-check \
  --public-model /prepared/public-native.json \
  --public-frames /prepared/public-water192.jsonl \
  --medium-model /prepared/medium-model.json \
  --medium-frames /prepared/medium-frames.jsonl \
  --work /new/acceptance-directory --run-gpu
```

Append .exe on Windows. The script permits a directory containing the preserved
baseline binaries but requires new report/request/output names. It records input
and binary hashes; two five-repeat timing pairs; 20-step old-CUDA/current-CPU/
current-CUDA MD with matching output/checkpoint cadence; exact 10-step restart;
and a seven-frame moving/rebuild check in one persistent potential. It stops on
first process failure/timeout with no retry or GPU reset. An owned timed-out
process tree is stopped. Normal completion leaves no worker running.

[Final receipt](evidence/native-cuda-infra-final.json) and [release measurements](RELEASE.md#local-native-cuda-cache-follow-up)
separate CUDA construction, first evaluation, warm evaluation and complete MD.
A large old/new CUDA warm ratio must not be relabeled GPU-over-CPU acceleration.
The CPU algorithm is unchanged and has no candidate cache. In the recorded run,
final CUDA's complete short MD still took longer than CPU because initialization
is material. Changed cells, wrapping or large displacements can require rebuilds;
independent-frame workloads need not obtain the same warm-cache benefit.


## Integrated cooperative CUDA acceptance

The later local integration combines native source b6b7aa7, C API source 1aee39c
and A's preflight/tutorial. Exact tested runtime: 9b8d184. The new compact
[GPU receipt](evidence/native-cuda-integrated-fit.json) and
[CPU tutorial receipt](evidence/cpu-tutorial-integrated.json) supersede no historical
receipts: they identify their own binaries and inputs. C API new numerical
validation remains separate and pending; this check exercises native CUDA only.

Build and preserve the engine and CUDA checker from 9eaf284 before building the
integrated candidate with identical release settings. Use the build and input
preparation above, then run the same acceptance script with the actual provenance
arguments (paths must name the newly built/preserved binaries and existing inputs):

```sh
python mlmd/scripts/validate_cuda_infra.py \
  --baseline-engine /saved/9eaf284/omnimd-mlmd \
  --baseline-check /saved/9eaf284/omnimd-native-cuda-check \
  --engine /integrated/mlmd/target/release/omnimd-mlmd \
  --check /integrated/mlmd/target/release/omnimd-native-cuda-check \
  --public-model /prepared/public-native.json \
  --public-frames /prepared/public-water192.jsonl \
  --medium-model /prepared/medium-model.json \
  --medium-frames /prepared/medium-frames.jsonl \
  --baseline-commit 9eaf284050257dae5ffeb53210987d7c35d391a5 \
  --source-commit 9b8d1847f892e32751c73d3262e0245d0efb4cdd \
  --native-source-commit b6b7aa7bc239702dafdefb559d758efd4c357eaa \
  --work /new/integrated-acceptance --run-gpu
```

Only run in a coordinated GPU window. The recorded invocation completed without
errors/retries and released its window at 2026-10-16 07:50:26 UTC. Complete MD
throughput is 20 steps divided by total process wall time, including startup,
model loading/compilation, JSON output and checkpoint fsync. Warmed five-repeat
means exclude those startup costs. Normal desktop graphics remain uncontrolled;
these figures do not establish a scaling curve or production performance.

## External label return after candidate selection

Use [EXTERNAL_LABELS.md](engine/EXTERNAL_LABELS.md) for the exact preparation and
return contract. Select a hash-bound subset, export its frame identities, obtain
external results separately, then import complete E/F labels into a new dataset.
The merge fills missing labels and rejects conflicts, changed geometry/parameters,
duplicates and mismatched hashes. Inspect before explicitly retraining. The
[CPU replay](evidence/external-labels.json) closes these file/CLI steps on saved
synthetic labels, while preserving v2 context in a separate IO check. It does not
launch an external calculator or make the native trainer parameter-aware.
