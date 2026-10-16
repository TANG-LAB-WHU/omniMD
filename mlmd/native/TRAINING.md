# Native final-readout training

This document covers the `omnimd-native-readout-train-v1` mode. For full supported
network training, see [FULL_TRAINING.md](FULL_TRAINING.md).

In readout mode, `omnimd-native-train` fits only the final linear weights of each species-specific
fitting network. Descriptor/embedding weights, all hidden layers, normalization,
cutoffs, selection and separate atomic offsets stay frozen. Final fitting biases
also stay frozen unless `fit_bias=true`. This is **last-layer training, not
full-network DeepMD training**. Output is the same native float64 JSON v1 consumed
by `NativePotential` and `omnimd-native-eval`.

## Run and input contract

```sh
cargo build --locked --offline --manifest-path mlmd/native/Cargo.toml --bins
omnimd-native-train train.request.json
```

The request schema is strict; paths are relative to the request file unless
absolute. The output directory must be new and its parent must already exist.
The CLI writes `model.json` and `report.json` after fitting and verifying a
native-v1 serialization/reload. Example:

```json
{
  "schema": "omnimd-native-readout-train-v1",
  "device": "cpu",
  "initial_model": "initial.json",
  "dataset": "dataset.jsonl",
  "output_directory": "trained",
  "config": {
    "split": {"mode": "seeded", "seed": 20261015, "validation_fraction": 0.25},
    "fit_bias": false,
    "energy_weight": 1.0,
    "force_weight": 1.0,
    "energy_scale_ev_per_atom": 1.0,
    "force_scale_ev_per_angstrom": 0.01,
    "ridge_inverse_ev2": 0.0,
    "rank_rtol": 1e-12
  }
}
```

The trainer reuses A's `omnimd-dataset-jsonl-v1` exactly: one header, consecutive
`{"kind":"frame","data":{"index":...,"id":...,"frame":Frame,"labels":...}}`
records, then mandatory `completed` with frame count and `source_sha256`.
Header fields are `schema`, ordered `type_map`, `units:"EvAngstrom"`,
`virial_convention:"MinusDEpsilonRow"`, `source`, and `selected_sets`. No second
position/species/cell representation is introduced. See the integration tree's
`mlmd/engine/DATA.md`; native currently parses this wire contract independently
to avoid a native-to-engine dependency cycle.

For this bounded trainer every frame must have both energy and all force labels.
Missing labels are errors, never zeros. Virial is optional and used only for
assessment. Unknown fields/types/units, duplicate IDs, wrong shapes, nonfinite
labels, nonconsecutive indices, missing completion and trailing records fail.
Every record requires its terminating newline. The ordered dataset/model type
maps must agree. Bounds: 128 MiB dataset, 16 MiB record, 10000 frames, 256 fitted
coefficients and four million dense design entries including possible ridge rows.

A seeded split uses SplitMix64 and unbiased Fisher-Yates, rounds the requested
validation count and bounds it to `[1, frames-1]`. Both sets retain input order
and are recorded as original index/ID pairs. Alternatively use
`"split":{"mode":"explicit","validation_ids":["set.001/0","set.001/1"]}`.
No validation labels enter the solve or parameter selection. Random frame
splits do not guarantee independence of correlated trajectories; explicit IDs
allow a separately chosen system/trajectory holdout. The report preserves the
header, source SHA map, actual dataset JSONL SHA, request/model hashes, config,
split identities, rank and initial/final metrics.

## Explicit CPU or CUDA features

`device` defaults to `"cpu"`. With a `--features cuda` build, select an explicit
`"cuda:N"` visible device ordinal, for example `"cuda:0"`. There is no `auto`
selection or silent CPU fallback. The request schema and engine forwarding are
otherwise unchanged. Rust callers can use `training::train_with_device` and
`TrainingDevice`; the existing `train` function still selects CPU.

CUDA computes the exact one-hot readout E/F/virial columns with the existing
FP64 descriptor, fitting and coordinate-pullback kernels. A private zero-head
potential keeps model weights and workspace resident; each basis transition
clears the previous coefficient and sets the next coefficient. Atomic offsets
and frozen final biases are tracked separately, so large energy offsets never
enter a feature column. Native-v1 validation requires a final linear layer with
no residual or timestep. CPU still prepares geometry and reduces forces/virial,
predicts the initial model, solves the weighted QR update and independently
assesses the exported model. This is **GPU-assisted frozen-readout training**;
it does not implement CUDA gradients or Adam for the full supported network.

The CUDA architecture bounds in [CUDA.md](CUDA.md) also apply, including only
tanh/linear activations and at most 2048 atoms per frame. Dataset shapes, labels,
type map, split, dense design bound, model architecture and atom bound are
checked before creating the GPU context. Geometry/neighbor errors remain
fallible during feature evaluation. Any CUDA upload/evaluation error aborts the
fit and poisons that private basis; no partial trained model is published.
Unsupported CUDA full-network requests fail before dataset/checkpoint IO or
output creation. The original CPU Adam checkpoint schema is unchanged.

The direct QR solve has no iterative optimizer state or resume checkpoint. Run
it again from the same initial model, dataset, split and configuration; a saved
fitted model can initialize a later run, but that creates a new ridge anchor
and is not an optimizer resume. Same-device repeat exported identical model
bytes in the check below. Cross-device/driver bitwise identity is not promised;
small floating-point differences can also change pivot decisions near rank
tolerance. Reports preserve the existing rank policy and add `execution` with
the requested device, feature backend/device, CPU solver/assessment and separate
initialization, feature, reference, QR and assessment wall times. These segments
exclude file IO, design assembly and export; they are not total process timing.

For an explicitly authorized GPU window, reproduce the focused checks:

```sh
cargo test --release --locked --offline --manifest-path mlmd/native/Cargo.toml \
  --features cuda --test training_device -- --ignored --nocapture --test-threads=1
python mlmd/native/tools/replay_cuda_readout.py \
  --eval-exe /path/to/omnimd-native-eval \
  --train-exe /path/to/omnimd-native-train --device cuda:0 \
  --output-dir /new/output/directory
```

On Windows RTX 4090 (2026-10-16), both fixed-bias and fitted-bias feature tests
passed across nonperiodic/triclinic cells, changing composition/count,
repeated frames and large fixed offsets. Maximum column E/F/virial differences
were `0 / 1.11e-15 / 1.11e-15`. The real CLI used 32 original synthetic frames,
24 explicit training IDs and eight held-out IDs, fitting all eight readout
weights/biases. Saved CPU/CUDA model predictions differed by at most
`8.88e-16 / 1.78e-15 / 2.22e-15`; serialization/reload error was zero. Repeated
CUDA fits produced identical model bytes and all frozen parameters stayed
unchanged. [The compact report](fixtures/cuda-readout-report.json) includes
commands, hashes, ranks, metrics and exact execution scope.

This tiny 3/4-atom case was **slower on CUDA**: feature extraction took 21.37 ms
versus CPU 5.15 ms; CUDA initialization was another 195.88 ms. This is an
implementation/parity check, not a speed or chemical-accuracy claim. Full GPU
training, feature checkpointing, virial-loss optimization and representative
large-system training performance remain unimplemented or unvalidated.

## Exact features and objective

For fixed hidden features `h_i` and type-specific readouts `w_t,b_t`:

```text
E = sum_t [w_t . sum_{i:type_i=t} h_i + b_t N_t] + E_fixed
F_q = -sum_t w_t . sum_{i:type_i=t} d(h_i)/d(r_q)
```

A feature-only copy clears final weights and additive offsets, activates one
readout coefficient at a time, and evaluates the existing analytic native
potential. Thus each column contains exact energy, force and virial features;
there is no finite-difference parameter derivative and no subtraction of large
energy offsets. Bias force/virial columns are zero. Only one mutable basis model
is kept. Feature extraction costs one ordinary evaluation per fitted coefficient
per training frame; it is intended for small readouts, not a throughput claim.

For `M` training frames, `N_f` atoms, scales `s_E` (eV/atom) and `s_F` (eV/A):

```text
L_data = (1/M) sum_f [
  w_E ((E_f - E_ref,f)/(N_f s_E))^2
  + w_F/(3N_f) sum_q ((F_fq - F_ref,fq)/s_F)^2
]
L = L_data + lambda sum_j (theta_j - theta_initial,j)^2
```

The E/F data loss is dimensionless. Weights are nonnegative with at least one
positive; scales are positive. `lambda=ridge_inverse_ev2` is an optional penalty
on final coefficient updates in the native eV convention, not on the initial
coefficients themselves. Both rows and targets receive the same normalization;
ridge rows are appended after the `1/sqrt(M)` data normalization.

Column-pivoted Householder QR solves for the update without forming normal
equations. `rank_rtol` applies relative to the largest initial column norm.
Reports distinguish data rank from ridge-augmented solve rank. Unidentified free
pivoted updates are zero: this is a basic solution, not a minimum-norm claim.
Fixed-composition energies do not identify all species biases, hence the default
frozen biases. An underdetermined fit is visible in the reported rank.

Physical metrics include energy RMSE (eV), per-atom energy RMSE (eV/atom), pooled
force-component RMSE (eV/A), and optional pooled virial-component RMSE (eV).
The force term in `L_data` is averaged per frame; pooled force RMSE differs when
frame atom counts vary. `ridge_penalty` is reported separately from `L_data`.

## Reproduce the actual small training run

```sh
python mlmd/native/tools/replay_readout_training.py \
  --eval-exe /path/to/omnimd-native-eval \
  --train-exe /path/to/omnimd-native-train \
  --output-dir /new/output/directory
```

This stdlib-only recipe creates original synthetic teacher/readout weights and
48 distinct three/four-atom nonperiodic/triclinic frames (data seed 20261014).
The Rust trainer uses split seed 20261015: 36 training and 12 validation frames.
All 10 readout coefficients are identifiable. It then independently evaluates
the saved model with the normal inference CLI and checks all frozen parameters
remain exactly unchanged. JSON `float_roundtrip` parsing preserves stored
float64 values across export; default parsing had exposed a one-ULP drift.

| Held-out metric | Initial | Fitted |
| --- | ---: | ---: |
| Energy RMSE (eV/atom) | 0.1293704981 | 4.0455948871e-16 |
| Force RMSE (eV/A) | 0.0106064914 | 1.4395814190e-17 |
| Dimensionless E/F objective | 1.0997425122 | 2.1355246224e-30 |

Training objective fell from 0.6930168689 to 1.3672756013e-30. Saved-model maximum
E/F/virial errors against the teacher were 2.16e-15 / 9.71e-17 / 5.55e-17.
Serialize/reload differences were exactly zero. These are manufactured
teacher-recovery results, not physical accuracy or generalization claims.
[The committed report](fixtures/readout-training-report.json) records evidence.

`cargo test --locked --offline --manifest-path mlmd/native/Cargo.toml` passes ten
tests: the seven existing native numerical tests, one rank/ridge/normalization
test, one analytic-feature/coordinate-difference test, and one real-fit test
that also perturbs all validation labels and proves fitted weights unchanged.
A's unlabeled example record was rejected without creating output. After enabling
exact float parsing, the existing four-frame public C API comparison also stayed
within its 1e-12 absolute+relative tolerance. No dependencies were downloaded;
sha2 0.10 was reused from the already approved shared project's offline cache.
