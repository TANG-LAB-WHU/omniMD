# Native full supported-network training

`omnimd-native-train` now has a second strict request schema,
`omnimd-native-full-train-v1`. It trains all embedding and fitting matrices,
biases and present timesteps jointly against energy/force labels and optional
[virial supervision](VIRIAL_TRAINING.md) using
CPU float64 Rust. It preserves the existing readout request and inference model
formats. The engine's `train-native REQUEST.json` wrapper can forward either
request unchanged to this executable. The optional request field `device`
defaults to `"cpu"`; a CUDA-enabled build also accepts explicit `"cuda:N"` for
bounded all-neural-parameter GPU gradients and Adam. See
[CUDA full training](CUDA_FULL_TRAINING.md) for its CPU/GPU boundary, tighter
resource limits and checkpoint identity. E/F-only CPU v1 and CUDA v2 checkpoints
remain compatible; enabled virial loss uses v3, opt-in CUDA reverse execution
uses v4, and explicit [minibatching](MINIBATCH_TRAINING.md) uses v5.

## Supported scope

The model must pass native-v1 validation and use tanh in every embedding/hidden
layer, with linear fitting outputs. Explicit none/identity/concat2 residuals
and branch timesteps are supported. There is no production finite-difference
parameter differentiation. Other activations fail explicitly even if inference
supports them. All native per-center/per-neighbor network copies are independent
parameters: duplicated one-side embeddings are **not tied** during training.

The type map, normalization mean/std, cutoff radii, neighbor capacities,
axis dimension, architecture, and separate `atom_energy_bias` remain fixed.
Supply an initialized model and explicit positive E/F loss scales; this trainer
does not infer normalization statistics, randomize initial weights, or estimate
atomic offsets. Final fitting biases do train. Inference remains the existing
shared `Potential` E/F/virial implementation, including normalized padding,
quintic switching, full periodic-image enumeration and strict neighbor overflow.

Current CPU bounds: at most 10000 neural scalar parameters, 128 total neighbor
slots, 64 atoms per frame, 1024 frames, and two million AD nodes per frame.
The tape bound can reject a model below the other limits. Existing dataset
limits of 128 MiB/file and 16 MiB/record still apply. This is a bounded reference
trainer, not a claim of production scaling. No new dependency was introduced.

## Request and reproducibility

```json
{
  "schema": "omnimd-native-full-train-v1",
  "device": "cpu",
  "initial_model": "initial.json",
  "dataset": "dataset.jsonl",
  "output_directory": "trained",
  "config": {
    "split": {"mode": "seeded", "seed": 20261015, "validation_fraction": 0.25},
    "steps": 300,
    "energy_weight": 1.0,
    "force_weight": 1.0,
    "energy_scale_ev_per_atom": 0.1,
    "force_scale_ev_per_angstrom": 0.1,
    "learning_rate": 0.005,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "gradient_clip_norm": 10.0,
    "checkpoint_every": 150
  }
}
```

Run `omnimd-native-train REQUEST.json`. Paths are relative to the request file;
output must be a new directory with an existing parent. The exact shared
`omnimd-dataset-jsonl-v1` header/frame/completed contract is reused. Every frame
must have E and F labels; missing values are errors, not zeros. Virial labels
are assessment-only by default. Adding `config.virial_loss` enables the
[nine-component virial loss](VIRIAL_TRAINING.md) and requires virial labels on
every frame. See [TRAINING.md](TRAINING.md) for the wire
contract and strict parsing details.

Seeded selection uses the existing SplitMix64/Fisher-Yates split, rounded
validation count clamped to `[1, frames-1]`, and stable original dataset order.
Alternatively use `{"mode":"explicit","validation_ids":["set.000/3"]}`.
By default there is no optimizer RNG or reshuffling: the split seed only
controls validation selection. Optional `config.minibatch` adds a separate explicit
seed, order policy and exact sampler checkpoint; see [MINIBATCH_TRAINING.md](MINIBATCH_TRAINING.md). Validation labels never enter an update, model
selection, or schedule. Without minibatching, every requested step is a complete training-set update;
the exported model is the last iterate. Training reports retain exact dataset
byte SHA-256, source SHA map, original index/ID of both splits, loss scales,
parameter-layout hash, parameter group changes, and initial/final E/F/virial
metrics. For resumed runs, initial metrics refer to the start of that invocation.

For training frames `f=1..M`, atom count `N_f`, energy/atom scale `sE` and force
scale `sF`, the loss has **no one-half factor**:

```text
L = (1/M) sum_f [wE ((E_f-E*_f)/(N_f sE))^2
                 + wF/(3 N_f) sum_q ((F_fq-F*_fq)/sF)^2]
```

E/F weights are finite/nonnegative, with at least one positive unless virial
supervision is enabled. Both E/F scales are
finite/positive even when their weight is zero. `gradient_clip_norm:null`
disables clipping. Otherwise the whole accumulated gradient is scaled to the
specified Euclidean norm before standard Adam moment updates. Learning rate is
constant; moments are bias-corrected using the global step, with update divisor
`sqrt(v_hat)+epsilon`. There is no ridge penalty or weight decay. Optional
virial supervision adds a separately normalized term; see
[VIRIAL_TRAINING.md](VIRIAL_TRAINING.md) for its exact formula and v3 restart rules.

## Exact E/F mixed derivatives

With detached loss seeds `a=2*wE*(E-E*)/(M*N^2*sE^2)` and
`v_q=2*wF*(F_q-F*_q)/(M*3*N*sF^2)`, use `F=-grad_r E`:

```text
grad_theta L = a grad_theta E - grad_theta (v . grad_r E)
```

Each normalized environment row has an existing analytic displacement Jacobian
`J`. Its coordinate-directional tangent is `J*(v_neighbor-v_center)`; padding
has zero tangent, and periodic self-image tangents cancel. Cell/image choices
are fixed while differentiating parameters. The Rust tape propagates dual
(value,tangent) nodes through embedding networks, contraction and fitting
networks, then reverse-differentiates `a*E - E_tangent` with shared parameter
leaves. Activation derivatives remain differentiable graph expressions; tanh
therefore contributes its second derivative correctly. Residual and timestep
paths participate at both levels. No full Hessian is materialized and no
coordinate second derivative of the environment is needed for these parameter
mixed derivatives. The AD primal is checked against ordinary native energy.

## Checkpoint/restart

Outputs are `model.json`, `report.json`, `checkpoint.json`, and
`checkpoint-step-N.json` at the requested global interval and final step.
Checkpoint files are written to a new temporary file, synced and renamed;
existing targets are never overwritten. A later failed run may leave its output
directory and earlier valid checkpoints. Filesystem sync does not promise
recovery from every power-loss scenario.

To continue, replace `initial_model` with
`"checkpoint":"trained/checkpoint.json"`, choose a new output directory,
and set `config.steps` to the number of **additional** updates. Exactly one of
`initial_model` or `checkpoint` is required. All other configuration fields,
including checkpoint interval and split specification, must match exactly, except
for the runtime-only CUDA fixed geometry cache flag. Explicit minibatching also
requires unchanged batch size, seed and order policy.

For E/F-only runs, CPU checkpoint schema `omnimd-native-full-checkpoint-v1` stores the last model,
original anchor model, first/second moments, global step, configuration,
dataset SHA, parameter-layout SHA and split indices. Restart checks both model
schemas, shapes, finite moments/nonnegative second moments, split, exact dataset
bytes and immutable settings. Restoring all neural parameters from the anchor
must reproduce it exactly, protecting every frozen model field. Float64 JSON
round trips use serde_json's `float_roundtrip` feature. Determinism is verified
within the same toolchain/platform; cross-platform bitwise equality is not
claimed. Checkpoints are not cryptographically signed.

## Acceptance and replay

```sh
cargo test --locked --offline --manifest-path mlmd/native/Cargo.toml
cargo build --release --locked --offline --manifest-path mlmd/native/Cargo.toml --bins
python mlmd/native/tools/replay_full_training.py \
  --eval-exe mlmd/native/target/release/omnimd-native-eval \
  --train-exe mlmd/native/target/release/omnimd-native-train \
  --output-dir mlmd/native/fixtures/generated-deepmd/full-training-replay
```

Append `.exe` to Windows binary paths. The original tiny fixture
`fixtures/full-tanh-v1.json` contains 112 scalar neural parameters. It uses two
types, pair-dependent embeddings, nonzero radial means, two embedding layers,
two fitting hidden layers, both residual forms, timesteps and scalar outputs.
The replay perturbs **all** teacher neural parameters, generates 24 frames
(including periodic/nonperiodic and 3/4-atom cases), and creates E/F/virial labels
with the ordinary native evaluator. The teacher is synthetic, not a chemical
potential. Data seed is 20261014 and split seed is 20261015 (18 train, 6 validation).

Windows GNU Rust 1.99.0 offline results after 300 Adam steps:

| Metric | Initial | Final |
| --- | ---: | ---: |
| Training normalized E/F loss | 8.83036148 | 0.000530212 |
| Validation energy/atom RMSE (eV) | 0.0603337 | 0.000447666 |
| Validation force-component RMSE (eV/angstrom) | 0.317807 | 0.00679365 |
| Validation virial-component RMSE (eV) | 0.407247 | 0.0109499 |

All 56 embedding, 48 fitting-hidden and 8 fitting-output parameters changed.
The serialized 150+150 restart exactly matched continuous 300-step model and
moments. Native model serialization/reload E/F/virial differences were all zero.
`fixtures/full-training-report.json` records metrics and hashes; generated data,
weights, checkpoints and raw outputs stay in the ignored replay directory.

The focused Rust checks compare every parameter against central differences
of the independent ordinary native E/F loss for E-only, F-only and combined
losses, on periodic and nonperiodic configurations; check zero/one-valued
parameter leaves and final-bias zero force gradients; verify two Adam steps
against explicit formulas; verify serialized restart equality and incompatible
restart rejection; and verify validation-label isolation and unsupported
activation rejection. Finite differences are exclusively a tiny test oracle.
Existing inference E/F/strain-virial, symmetry and boundary checks remain intact.

This establishes implementation correctness for this bounded subset and
learning on an original synthetic teacher. It does not establish physical
accuracy, chemical generalization, production MD stability, full DeepMD/HAP
support, a new optimization algorithm or any speed advantage.

For explicit minibatching, [indexed dataset access](INDEXED_TRAINING.md) can read
larger datasets one frame at a time. It preserves the existing memory-mode limits
and numerical/checkpoint semantics; its own parsing/index bounds and the unchanged
geometry payload cap are documented separately.
