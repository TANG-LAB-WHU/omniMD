# Training se_e2_a on the GPU runtime

`omnimd-native-gpu-train` trains a native se_e2_a model of any network size on
one NVIDIA GPU, with DeePMD-kit's `ener` loss, learning-rate schedule and Adam
settings. It is separate from the bounded CPU/CUDA trainers behind
`omnimd-mlmd train-native`, which are limited to small networks.

Build (same runtime requirements as [guide step 28](../GETTING_STARTED.md#28-run-se_e2_a-on-the-gpu-runtime-and-compute-dataset-statistics)):

```sh
cargo build --release --locked --manifest-path mlmd/Cargo.toml -p omnimd-native --features gpu --bin omnimd-native-gpu-train
```

## Request

```text
omnimd-native-gpu-train REQUEST.json [REQUEST.json ...]
```

```json
{
  "model": {
    "type_map": ["Ca", "P", "O", "H", "C"],
    "descriptor": {"type": "se_e2_a", "sel": "auto", "rcut": 6.0, "rcut_smth": 0.5,
                   "neuron": [25, 50, 100], "axis_neuron": 16, "resnet_dt": false,
                   "type_one_side": false, "seed": 11},
    "fitting_net": {"neuron": [240, 240, 240], "resnet_dt": true, "seed": 11}
  },
  "dataset": "/absolute/train.jsonl",
  "output": "/absolute/trained.native.json",
  "precision": "float64",
  "learning_rate": {"start_lr": 1e-3, "stop_lr": 3e-8, "decay_steps": 5000},
  "loss": {"start_pref_e": 0.02, "limit_pref_e": 1.0, "start_pref_f": 1000.0,
           "limit_pref_f": 1.0, "start_pref_v": 0.0, "limit_pref_v": 0.02},
  "training": {"numb_steps": 300000, "batch_size": 1, "seed": 11,
               "disp_freq": 1000, "disp_file": "/absolute/lcurve.out",
               "save_freq": 20000}
}
```

- `model` is either the `model` section of a DeePMD input script, as above, or
  the path of an existing native model to continue from (for example one
  converted with `tools/convert_deepmd.py`).
- `dataset` is an omnimd-dataset-jsonl-v1 file (`omnimd-mlmd import-deepmd`, or
  `bench/hap_prepare.py` for DeePMD npy systems). Every frame needs energy and
  force labels; virial labels are used where present.
- `learning_rate`, `loss` and `training` take the DeePMD keys shown, with
  DeePMD's defaults for the loss prefactors and `decay_steps`. Unknown keys are
  rejected.
- `precision` is the network precision, `float64` (default) or `float32`.
  Geometry, forces, virial and the loss are float64 either way.
- `training.validation_data` (optional), `{"dataset": "/absolute/val.jsonl",
  "numb_btch": 1}`: frames that are never trained on. Whenever a line of the
  learning curve is written, `numb_btch` of them are scored with the current
  parameters, in a shuffled order of their own. Training is the same with and
  without them.
- `training.save_freq` (optional) writes a snapshot every that many steps:
  `<output>.step-N`, the model, and `<output>.step-N.adam`, the optimizer state.
- `output`, `disp_file` and snapshot files must not exist; nothing is overwritten.

Several requests on one command line train at the same time, each on a stream
of its own on the same device: for example the members of an ensemble, which
differ in their seeds and output names. Each prints its own summary line, and
the models are those of the requests run one by one (not bit for bit: the
order of atomic additions on the device varies). A request that fails is named
and makes the exit status nonzero; the others still finish.

To continue a stopped run, give the snapshot as `model` and its `.adam` file as
`training.optimizer_state`, keep the other settings, and choose new `output` and
`disp_file` names. The run resumes at the snapshot's step with the same
schedule and Adam state. Its frame order is drawn afresh, so it is not
identical to the uninterrupted run.

## What happens

**A fresh model.** With a `model` section the networks are drawn like
DeePMD-kit's PyTorch backend does: weights normal with standard deviation
`1/sqrt(inputs + outputs)`, biases standard normal, timesteps normal around 0.1.
A hidden layer is residual when it keeps or doubles its width. The random stream
is omniMD's own, so a seed does not reproduce DeePMD's numbers. From the dataset
the trainer computes, with DeePMD's definitions ([`stats`](src/stats.rs)):

- `sel`, when given as `"auto"` or `"auto:FACTOR"`, from the neighbor statistics;
- the descriptor normalization `davg`/`dstd`, over all frames;
- the per-type energy bias, by minimum-norm least squares.

DeePMD samples a few batches per system for these statistics; using every frame
gives slightly different numbers.

**Loss.** For a frame of `N` atoms, with prefactors `p_e`, `p_f`, `p_v`:

```text
L = p_e (E - E*)^2 / N + p_f mean((F - F*)^2) + p_v mean((V - V*)^2) / N
```

and the batch loss is the mean over its frames. Each prefactor moves from its
`start_` to its `limit_` value in proportion to the learning rate:
`p = limit + (start - limit) * lr / start_lr`. A frame without a virial label
contributes no virial term.

**Learning rate.** Stepped exponential decay as DeePMD's `LearningRateExp`:
`lr(step) = max(start_lr * r^floor(step / decay_steps), stop_lr)` with
`r = (stop_lr / start_lr)^(decay_steps / numb_steps)`. A `decay_steps` no shorter
than the run is replaced as DeePMD does. The values are pinned to DeePMD-kit
3.1.3 in the unit tests.

**Optimizer.** Adam with `torch.optim.Adam`'s defaults (betas 0.9 and 0.999,
epsilon 1e-8, no weight decay), one update per batch. Trainable are all weights,
biases and timesteps of the embedding and fitting networks; normalization and
energy bias stay fixed.

**Gradient.** Forces and virial are derivatives of the energy, so the gradient
of their loss terms needs second-order information. With `a = dL/dE`,
`c = dL/dF` and `w = dL/dV` held fixed, the loss gradient equals the gradient of
`a*E - E'`, where `E'` is the directional derivative of the energy when every
neighbor vector `d = r_j - r_i` moves along `c_j - c_i + w^T d`. One forward
pass carries that tangent through the environment matrix, the embedding nets,
the descriptor and the fitting nets, and one reverse pass through both gives
the gradient ([`gpu_train.rs`](src/gpu_train.rs)). All embedding nets are
evaluated as one set and all fitting nets as another (`omnimd_gpu::TrainSet`):
every row names its net, one kernel does the forward pass with tangent, a
second the reverse pass, and the parameter gradients are one matrix product per
net and layer. A step therefore issues about a hundred device calls whatever
the number of types.

**Frame order.** Frames are visited in shuffled epochs from `training.seed`.
With `batch_size` 1 this samples systems in proportion to their frame counts,
like DeePMD's default `prob_sys_size`.

## Output

- `output`: the trained native model, usable by `omnimd-native-gpu-eval`,
  `omnimd-native-eval` and the engine.
- `disp_file`: one line every `disp_freq` steps with the step, the energy RMSE
  per atom, the force RMSE and the virial RMSE per atom of that batch, and the
  learning rate. These are DeePMD's `rmse_e_trn`, `rmse_f_trn` and `rmse_v_trn`.
  With `validation_data` each of the three is preceded by the same quantity
  over the validation frames of that line (`rmse_e_val` and so on); the header
  line names the columns.
- Standard output: one JSON object with the step count, time and steps per
  second.

## Limits

- One GPU, one frame at a time: a frame and its recorded intermediates must fit
  in device memory (about 1 GB per request in float64 for the model and the
  frames of up to 224 atoms below; memory grows with atoms times `sum(sel)`).
- se_e2_a with one embedding net per type pair (`type_one_side: false`), tanh
  activations and an energy fitting net. No hybrid descriptors, type embedding,
  fparam/aparam, atomic energies or force-only prefactors (`pref_pf`, `pref_gf`).
- Without `save_freq` a run that stops leaves no model.
- No fine-tuning controls (frozen layers, bias adjustment).
- Each step still issues about a hundred device calls and returns to the host
  for the loss, so small frames are bound by that overhead in float32.

## Speed

Steps per second on an RTX 4090 for the model and frames of the full run below
(2,664,455 parameters; frames of 54 to 224 atoms; one frame per step):

| | float32 | float64 |
|---|---:|---:|
| one request | 284 | 92 |
| two requests at once, in total | 330 | 122 |
| four requests at once, in total | 347 | 152 |

- DeePMD-kit 3.1.3 was recorded at 6.5 steps/s for this model on this machine
  (PyTorch backend, float64).
- In float32 the time goes to the device calls of a step and the host work
  between them. In float64 it goes mostly to arithmetic: this card does
  double-precision matrix products about twenty times slower than single
  (0.37 ms against 0.017 ms for the first fitting layer over 224 atoms).
- Requests trained at once share the card through separate streams and gain
  about 20 percent in total in float32 and 60 percent in float64 with four of
  them. Separate processes do not add up: two or four of them reach less in
  total than one alone.
- Before the embedding and fitting nets were evaluated as sets, a step issued
  about eleven hundred device calls and reached 118 steps/s in float32 and 58
  in float64; the full run below was made with that version.

## How it was checked

- The trainer's prediction and loss equal the inference path
  (`tests/gpu_train.rs`).
- The gradient matches central finite differences of the loss for the energy,
  force and virial terms separately and together, on every 37th of about 7,000
  parameters of a two-type model with both residual kinds and timesteps.
- Batches average their frames; Adam matches a host implementation of
  `torch.optim.Adam`'s rule; a student network fits a teacher's labels in both
  precisions.
- A run continued from an exported model and its optimizer state reaches the
  same parameters as the uninterrupted run (to 1e-11); without the optimizer
  state it does not.
- Removing the second-derivative term, transposing the virial direction or
  dropping either descriptor adjoint term makes the gradient test fail.
- The network sets give the outputs, tangents, parameter gradients and input
  adjoints of the same networks evaluated one by one with separate matrix
  products (to 1e-12 in float64), for one input and for many, for narrow and
  wide layers, with both residual kinds and timesteps (`omnimd-gpu`,
  `tests/device.rs`). Eight single-line errors introduced into their kernels
  each fail that test.
- Two requests trained in one process give the models of the same requests
  trained alone to 1e-8, and a request with validation frames gives the model
  of the same request without them (`tests/gpu_train_cli.rs`).

## A full run on real data

The iteration-0 training set of the HAP CO2-mineralization pipeline as its
DeePMD models used it (196 DFT-labelled frames in 8 systems of 54 to 224
atoms, 152 of them with virial labels), with that pipeline's DeePMD-kit
settings: type map Ca P O H C, `sel` [44, 16, 60, 72, 20], cutoff 6.0,
embedding [25, 50, 100], axis 16, fitting [240, 240, 240] with timesteps
(2,664,455 parameters), float64, learning rate 1e-3 to 3e-8 with `decay_steps`
5000, prefactors 0.02 to 1 (energy), 1000 to 1 (force) and 0 to 0.02 (virial),
300,000 steps of one frame. omniMD started from fresh networks (seed 11) and
computed normalization and energy bias from the 196 frames.

| | omniMD, one run | DeePMD-kit 3.1.3, the pipeline's four models |
|---|---:|---:|
| Time for 300,000 steps | 88 min (57 steps/s) | about 12.8 h each (6.5 steps/s) |
| Sealed test, force RMSE (eV/Å) | 0.0719 | 0.0699 to 0.0711 |
| Sealed test, energy RMSE (meV/atom) | 1.27 | 1.64 to 1.74 |

The sealed test is the pipeline's 83 held-out frames, scored with
`omnimd-native-gpu-test`. Per system (force RMSE in eV/Å, energy RMSE in
meV/atom; DeePMD: smallest and largest of the four models):

| system (frames, atoms) | force, omniMD | force, DeePMD | energy, omniMD | energy, DeePMD |
|---|---:|---:|---:|---:|
| CaCO3_20 (9, 160) | 0.0543 | 0.0519 to 0.0529 | 0.45 | 0.38 to 0.71 |
| calcite (12, 80) | 0.0426 | 0.0391 to 0.0401 | 0.48 | 0.32 to 0.58 |
| HAP(001) + CO2 (9, 91) | 0.0868 | 0.0838 to 0.0849 | 0.98 | 0.76 to 1.10 |
| HAP(001) + water + CO2 (9, 169) | 0.1090 | 0.1062 to 0.1090 | 1.57 | 1.52 to 1.75 |
| HAP bulk (12, 88) | 0.0445 | 0.0430 to 0.0452 | 0.49 | 0.46 to 0.70 |
| lime (12, 54) | 0.0147 | 0.0200 to 0.0214 | 0.72 | 1.19 to 1.73 |
| monetite (9, 224) | 0.0860 | 0.0809 to 0.0846 | 0.42 | 0.40 to 0.77 |
| portlandite (11, 90) | 0.0386 | 0.0376 to 0.0397 | 2.81 | 3.82 to 4.00 |

How to read this:

- The omniMD model is as accurate as the DeePMD models trained on the same
  frames with the same settings: its force error is 1 to 3 percent above
  theirs overall, its energy error about a quarter below. This is one omniMD
  run against four DeePMD seeds; the DeePMD seeds differ among themselves by
  about 2 percent in force and 6 percent in energy.
- The two trainers are not expected to give the same numbers: the random
  streams differ, omniMD takes the statistics from every frame, and its
  frame order is its own.
- The DeePMD errors are the pipeline's recorded sealed-test results and the
  DeePMD rate is the pipeline's record on this machine; neither was re-run
  here. For the pipeline's later models, omniMD's test command reproduces
  such records to their printed precision.
- The omniMD time includes about 8 minutes in which the GPU was shared with
  another run, and the host was compiling meanwhile. Alone, the trainer
  measured 57 steps/s in float64 and 118 in float32 on these frames.
- The pipeline marks its models as not production-qualified. The comparison
  says the two trainers reach the same quality on this data, nothing about
  the quality of either model for science.

Numbers and learning curve:
[`bench/results/hap-train-omnimd-rtx4090-2026-11-10.json`](../bench/results/hap-train-omnimd-rtx4090-2026-11-10.json),
[`.lcurve.out`](../bench/results/hap-train-omnimd-rtx4090-2026-11-10.lcurve.out).
