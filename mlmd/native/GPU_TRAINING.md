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
  Geometry, forces, virial and the loss are float64 either way, and so are
  the parameters the optimizer updates: float32 networks compute with their
  rounded copy (see Optimizer below).
- `model.fitting_net.atom_ener` (optional, a DeePMD key) gives the energy
  bias of every type, in the order of `type_map`, instead of the fit to the
  training energies.
- `training.validation_data` (optional), `{"dataset": "/absolute/val.jsonl",
  "numb_btch": 1}`: frames that are never trained on. Whenever a line of the
  learning curve is written, `numb_btch` of them are scored with the current
  parameters, in a shuffled order of their own. Training is the same with and
  without them.
- `training.save_freq` (optional) writes a snapshot every that many steps:
  `<output>.step-N`, the model, and `<output>.step-N.adam`, the optimizer state.
- `training.profile_stages` (optional, default false) times every stage of a
  step and adds `stages_ms_per_step` to the summary. The device is then
  synchronized after each stage, so the run is slower and the stages add up to
  more than a normal step; use it to see where the time goes, not for a run.
- `output`, `disp_file` and snapshot files must not exist; nothing is overwritten.

Several requests on one command line train at the same time, each on a stream
of its own on the same device: for example the members of an ensemble, which
differ in their seeds and output names. This is a convenience more than a
gain: together they are little faster than one after the other (see Speed).
Each prints its own summary line, and the models are those of the requests
run one by one (not bit for bit: the order of atomic additions on the device
varies). A request that fails is named and makes the exit status nonzero; the
others still finish. The requests of one command share one device context,
though, so an error of the device itself (not of one request's input) stops
them all: give long runs a `save_freq`.

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

A term is part of the loss only when both of its prefactors are nonzero. This
is the rule of DeePMD-kit's PyTorch backend (`deepmd/pt/loss/ener.py` in 3.1.3:
`has_v = start_pref_v != 0.0 and limit_pref_v != 0.0`), so the common setting
`"start_pref_v": 0, "limit_pref_v": 0.02` trains without the virial there and
here; the trainer says so when it starts. (DeePMD's TensorFlow backend keeps
such a term.) To bring a term in gradually, start it from a small nonzero
prefactor. A term that is off is still reported in the learning curve.

**Learning rate.** Stepped exponential decay as DeePMD's `LearningRateExp`:
`lr(step) = max(start_lr * r^floor(step / decay_steps), stop_lr)` with
`r = (stop_lr / start_lr)^(decay_steps / numb_steps)`. A `decay_steps` no shorter
than the run is replaced as DeePMD does. The values are pinned to DeePMD-kit
3.1.3 in the unit tests.

**Optimizer.** Adam with `torch.optim.Adam`'s defaults (betas 0.9 and 0.999,
epsilon 1e-8, no weight decay), one update per batch. Trainable are all weights,
biases and timesteps of the embedding and fitting networks; normalization and
energy bias stay fixed.

With float32 networks the parameters are still kept in float64: each step is
formed in float32 and added to the float64 values, and the networks get those
rounded. A schedule ends at learning rates of the order of the spacing of
float32 numbers near the parameters (a step of 3e-8 against a spacing of 6e-8
at 1), and steps that small would otherwise be rounded away. On the model of
the full run below, repeating the last third of one float32 run gave a
sealed-test energy RMSE of 1.84 meV/atom with float32 parameters, and 1.61
both with float64 networks and with float32 networks over float64 parameters;
the force RMSE was the same in all three.

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
second the reverse pass, and a third the parameter gradients of all nets and
layers. A step therefore issues about thirty device calls whatever the number
of types.

**Empty neighbor slots.** `sel` reserves slots for the largest neighborhood
in the data, so most slots of a typical frame are empty (about two thirds in
the frames of the full run below). An empty slot still enters the descriptor,
through the normalized zero of its environment row, but all empty slots of
one pair of center and neighbor type have the same embedding input. The
trainer therefore gives the embedding nets one row per neighbor and a single
row per pair of types for all its empty slots; the adjoints of those slots
are summed into that row before the reverse pass. The result is the same as
with a row for every slot.

**Frames seen before.** A training set is visited many times, and a frame's
sorted atoms, neighbor list and embedding rows do not depend on the
parameters. The trainer keeps them on the device for every frame it has
seen, up to 1 GiB (about 1.4 MB for a frame of 224 atoms with 212 slots per
atom), and recognizes a frame by comparing all its numbers. Frames beyond
that are prepared again at every visit.

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
  second, and with `profile_stages` the mean milliseconds per step of every
  stage (neighbors, environment, embedding, descriptor, fitting, forces, loss,
  tangent, the two parameter-gradient passes, descriptor reverse, optimizer).

## Limits

- One GPU, one frame at a time: a frame and its recorded intermediates must fit
  in device memory. A float32 request took about 0.6 GB for the model and the
  171 frames of the full run below, their cache included; the intermediates
  grow with the number of atoms times their neighbors.
- se_e2_a with one embedding net per type pair (`type_one_side: false`), tanh
  activations and an energy fitting net. No hybrid descriptors, type embedding,
  fparam/aparam, atomic energies or force-only prefactors (`pref_pf`, `pref_gf`).
- Without `save_freq` a run that stops leaves no model.
- No fine-tuning controls (frozen layers, bias adjustment).
- Each step still issues about thirty device calls and returns to the host
  for the loss (and, at a frame's first visit, for its neighbor counts), so
  small frames are bound by that overhead in float32.
- Two runs of the same request agree to rounding, not bit for bit: sums on
  the device are formed in an order that varies.

## Speed

Steps per second on an RTX 4090 for the model of the full run below (2,664,455
parameters) on the 196 frames of that pipeline's iteration-0 directory (54 to
224 atoms; one frame per step; 20,000 steps in float32 and 6,000 in float64,
nothing else on the card):

| | float32 | float64 |
|---|---:|---:|
| one request | 604 | 289 |
| two requests at once, in total | 648 | 321 |
| four requests at once, in total | 715 | 352 |

- DeePMD-kit 3.1.3 (PyTorch backend, float64) trained this model at 12.3
  steps/s on this machine when it had the machine to itself, and at 6.5 steps/s
  each with two models at once. These are the pipeline's records; they were
  not re-run here.
- Where the time of one float32 request goes, in milliseconds per step with
  every stage timed by `profile_stages` (1.87 in total): the two fitting passes
  0.46 and 0.39, the reverse pass of the embedding nets with their gradients
  0.20, the descriptor reverse pass 0.17, optimizer 0.16, the embedding nets
  0.11, forces 0.11, tangent 0.08, descriptor 0.07, loss 0.07. In float64 (3.69
  in total) the passes through the networks take 0.38 to 0.92 each: this card
  does double-precision arithmetic far more slowly than single.
- Requests trained at once share the card through separate streams and run
  at equal speed. That gains little in total: 7 percent for two and 18
  percent for four in float32, 11 and 22 percent in float64. Separate
  processes did not add up when that was measured with an earlier version:
  two or four reached less in total than one alone.
- How it got here, for one request on the same model and frames (float32 and
  float64): 118 and 58 with one matrix product per network and layer, about
  eleven hundred device calls per step; 284 and 92 with the networks
  evaluated as sets; 402 and 191 with the gradient kernel; 567 and 297
  without rows for empty neighbor slots; 661 and 294 with the frame cache;
  604 and 289 with float64 parameters under float32 networks (the optimizer
  then touches more memory and the next step starts colder, about a tenth of
  a step) and float32 dot products in partial sums (which gain about as
  much).

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
  wide layers, with both residual kinds and timesteps, for layers smaller
  and larger than a piece of the gradient kernel and for more rows than one
  of its jobs holds (`omnimd-gpu`, `tests/device.rs`). Eight single-line
  errors introduced into the forward and reverse kernels and nine into the
  gradient kernel and its host code each fail that test.
- With slots that are never full, always full and partly full, the trainer
  gives the prediction of the inference path, which has a row for every slot,
  and a gradient that matches finite differences. Of ten single-line errors
  introduced into the handling of empty slots eight fail the tests; the other
  two only cost time (rows for slots that could share one).
- A frame seen before gives the same loss and gradient as a frame prepared
  afresh, and a frame that differs in one coordinate is not taken for it.
- A loss term with one zero prefactor is off, as in DeePMD-kit's PyTorch
  backend (`bin/omnimd-native-gpu-train.rs`, unit test).
- Two requests trained in one process give the models of the same requests
  trained alone to 1e-8, and a request with validation frames gives the model
  of the same request without them (`tests/gpu_train_cli.rs`).

## A full run on real data

The frames the HAP CO2-mineralization pipeline's iteration-0 DeePMD models
were trained on: 171 DFT-labelled frames in 8 systems of 54 to 224 atoms.
(That pipeline's iteration-0 training directory now holds 196 frames. 25 of
its 30 lime frames were labelled only after that training had started, as
the times of their label files show, and are left out here; the five that
existed are kept.) Settings as in the pipeline's DeePMD-kit input: type map
Ca P O H C, `sel` [44, 16, 60, 72, 20], cutoff 6.0, embedding [25, 50, 100],
axis 16, fitting [240, 240, 240] with timesteps (2,664,455 parameters),
learning rate 1e-3 to 3e-8 with `decay_steps` 5000, prefactors 0.02 to 1
(energy) and 1000 to 1 (force), a virial prefactor of 0 to 0.02, which leaves
the virial term off in DeePMD-kit's PyTorch backend and here, and 300,000
steps of one frame. omniMD trained four models with the seeds 11 to 14 at
once, from fresh networks, with normalization and energy bias computed from
the 171 frames, with float32 networks (DeePMD: float64).

| | omniMD, four models | DeePMD-kit 3.1.3, the pipeline's four models |
|---|---:|---:|
| Time for the four | 30 min at once | 30.8 h one after the other (12.3 steps/s) |
| Sealed test, force RMSE (eV/Å) | 0.0698 to 0.0723 | 0.0699 to 0.0711 |
| Sealed test, energy RMSE (meV/atom) | 1.64 to 1.78 | 1.64 to 1.74 |

The sealed test is the pipeline's 83 held-out frames, scored with
`omnimd-native-gpu-test`. Per system (force RMSE in eV/Å, energy RMSE in
meV/atom; smallest and largest of the four models):

| system (frames, atoms) | force, omniMD | force, DeePMD | energy, omniMD | energy, DeePMD |
|---|---:|---:|---:|---:|
| CaCO3_20 (9, 160) | 0.0518 to 0.0533 | 0.0519 to 0.0529 | 0.36 to 0.51 | 0.38 to 0.70 |
| calcite (12, 80) | 0.0393 to 0.0411 | 0.0391 to 0.0401 | 0.39 to 0.55 | 0.32 to 0.58 |
| HAP(001) + CO2 (9, 91) | 0.0849 to 0.0854 | 0.0838 to 0.0849 | 0.69 to 1.23 | 0.76 to 1.10 |
| HAP(001) + water + CO2 (9, 169) | 0.1063 to 0.1118 | 0.1062 to 0.1090 | 1.36 to 1.95 | 1.52 to 1.75 |
| HAP bulk (12, 88) | 0.0434 to 0.0444 | 0.0430 to 0.0452 | 0.36 to 0.62 | 0.46 to 0.70 |
| lime (12, 54) | 0.0183 to 0.0252 | 0.0200 to 0.0214 | 1.09 to 1.86 | 1.19 to 1.73 |
| monetite (9, 224) | 0.0831 to 0.0864 | 0.0809 to 0.0846 | 0.42 to 0.61 | 0.40 to 0.77 |
| portlandite (11, 90) | 0.0367 to 0.0386 | 0.0376 to 0.0397 | 3.83 to 4.14 | 3.82 to 4.00 |
| **all** (83, 113 on average) | 0.0698 to 0.0723 | 0.0699 to 0.0711 | 1.64 to 1.78 | 1.64 to 1.74 |

**The same exploration with both ensembles.** The pipeline's sixteen
iteration-0 exploration runs (NVT at 300 and 500 K, 20,000 steps of 0.5 fs, a
frame every 50 steps) repeated by `omnimd-mlmd` with the first model of an
ensemble (float32, compressed), then the force deviation among its four
models on the 401 frames (float64), classified with the pipeline's window of
0.1 to 0.3 eV/Å ([`bench/hap_explore.py`](../bench/hap_explore.py)). Fractions
of frames below, inside and above the window, and the mean over the frames
of the largest deviation in a frame (eV/Å). "DeePMD models, omniMD" runs the
pipeline's own four models, converted, through the same omniMD commands;
"recorded" is the pipeline's LAMMPS run with them:

| run | omniMD models: accurate / candidate / failed | DeePMD models, omniMD: accurate / candidate / failed | recorded: accurate / candidate / failed | omniMD models: mean | DeePMD models, omniMD: mean | recorded: mean |
|---|---:|---:|---:|---:|---:|---:|
| CaCO3 20 T300 | 1.000 / 0.000 / 0.000 | 1.000 / 0.000 / 0.000 | 1.000 / 0.000 / 0.000 | 0.030 | 0.030 | 0.029 |
| CaCO3 20 T500 | 0.993 / 0.007 / 0.000 | 0.978 / 0.022 / 0.000 | 0.985 / 0.015 / 0.000 | 0.053 | 0.056 | 0.052 |
| calcite CaCO3 T300 | 1.000 / 0.000 / 0.000 | 1.000 / 0.000 / 0.000 | 1.000 / 0.000 / 0.000 | 0.027 | 0.026 | 0.027 |
| calcite CaCO3 T500 | 0.998 / 0.002 / 0.000 | 0.993 / 0.007 / 0.000 | 0.998 / 0.002 / 0.000 | 0.046 | 0.044 | 0.045 |
| hap001 co2 T300 | 0.030 / 0.140 / 0.830 | 0.047 / 0.077 / 0.875 | 0.045 / 0.257 / 0.698 | 1.2 | 8.7 | 0.669 |
| hap001 co2 T500 | 0.035 / 0.249 / 0.716 | 0.005 / 0.284 / 0.711 | 0.042 / 0.414 / 0.544 | 0.914 | 0.596 | 0.517 |
| hap001 water co2 T300 | 0.000 / 0.052 / 0.948 | 0.000 / 0.065 / 0.935 | 0.000 / 0.077 / 0.923 | 45.0 | 25.2 | 21.6 |
| hap001 water co2 T500 | 0.000 / 0.025 / 0.975 | 0.000 / 0.042 / 0.958 | 0.000 / 0.017 / 0.983 | 44.2 | 31.3 | 31.2 |
| hap bulk T300 | 1.000 / 0.000 / 0.000 | 1.000 / 0.000 / 0.000 | 1.000 / 0.000 / 0.000 | 0.033 | 0.035 | 0.035 |
| hap bulk T500 | 0.975 / 0.025 / 0.000 | 0.990 / 0.010 / 0.000 | 1.000 / 0.000 / 0.000 | 0.049 | 0.051 | 0.050 |
| lime CaO T300 | 1.000 / 0.000 / 0.000 | 1.000 / 0.000 / 0.000 | 1.000 / 0.000 / 0.000 | 0.019 | 0.016 | 0.015 |
| lime CaO T500 | 1.000 / 0.000 / 0.000 | 0.995 / 0.005 / 0.000 | 0.993 / 0.007 / 0.000 | 0.031 | 0.028 | 0.028 |
| monetite CaHPO4 T300 | 0.628 / 0.372 / 0.000 | 0.721 / 0.279 / 0.000 | 0.718 / 0.282 / 0.000 | 0.099 | 0.095 | 0.096 |
| monetite CaHPO4 T500 | 0.007 / 0.045 / 0.948 | 0.027 / 0.888 / 0.085 | 0.005 / 0.025 / 0.970 | 33.2 | 1.6 | 29.5 |
| portlandite CaOH2 T300 | 1.000 / 0.000 / 0.000 | 1.000 / 0.000 / 0.000 | 1.000 / 0.000 / 0.000 | 0.028 | 0.028 | 0.028 |
| portlandite CaOH2 T500 | 0.955 / 0.045 / 0.000 | 0.990 / 0.010 / 0.000 | 0.985 / 0.015 / 0.000 | 0.052 | 0.045 | 0.053 |

How to read this:

- On the frames and with the loss DeePMD used, omniMD's four models have the
  sealed-test errors of DeePMD's four: the ranges of the two sets of four
  seeds overlap in every system, lime included, where both saw five frames.
  omniMD's seeds spread a little wider (3.6 against 1.8 percent in the
  overall force RMSE).
- The two ensembles also explore alike. In the eleven runs where the
  pipeline's record has most frames below the window, the fraction below it
  differs between omniMD's models and the record by at most 0.09 (monetite at
  300 K, where the window's lower edge lies in the middle of the
  distribution) and by 0.03 or less elsewhere, and the mean deviation by
  -6 to +25 percent (4 percent on average; the largest is lime, with its
  five training frames). In the five runs where the models fail (the
  HAP(001) interfaces and monetite at 500 K) all three leave the window in
  most frames; how far they run away is chaotic and not comparable.
- The exploration chain itself is the same instrument as the pipeline's:
  with DeePMD's own models it gives the recorded fractions to 0.01 and the
  recorded mean deviations to -14 to +9 percent (-0.6 on average) in those
  eleven runs, from different trajectories. `omnimd-native-gpu-test` gives
  the pipeline's recorded test errors of those models to every printed digit
  ([`hap-test-deepmd-iter0-by-omnimd-...json`](../bench/results/hap-test-deepmd-iter0-by-omnimd-rtx4090-2026-11-10.json)).
- The two trainers are not expected to give the same numbers: the random
  streams differ, omniMD takes the statistics from every frame, and its frame
  order is its own.
- Which frames DeePMD saw is reconstructed, not recorded: the pipeline's
  notes say the lime labels were redone and entered from iteration 1, the
  label files of five lime frames predate the training, and its validation
  set had no lime. That the other systems' frames are unchanged since then is
  an assumption. With no lime frame at all, omniMD's models miss lime by 0.6
  to 2 eV/atom; with all thirty, they beat DeePMD's there (force RMSE 0.013
  to 0.014 against 0.020 to 0.021) and are up to 4 percent behind elsewhere,
  which is what first showed that the training sets differed.
- DeePMD's time is the pipeline's record on this machine (file times of the
  four runs; 12.3 steps/s per its notes) and was not re-run here.
- The pipeline marks its models as not production-qualified. The comparison
  says the two trainers reach the same quality on this data, nothing about
  the quality of either ensemble for science.

Numbers:
[`hap-train-ensemble-float32-omnimd-...json`](../bench/results/hap-train-ensemble-float32-omnimd-rtx4090-2026-11-10.json)
(sealed test),
[`hap-explore-iter0-omnimd-float32-...json`](../bench/results/hap-explore-iter0-omnimd-float32-rtx4090-2026-11-10.json)
and
[`hap-explore-iter0-deepmd-models-by-omnimd-...json`](../bench/results/hap-explore-iter0-deepmd-models-by-omnimd-rtx4090-2026-11-10.json)
(exploration). Earlier runs on other frames or with another loss are listed
in the [benchmark summary](../bench/results/SUMMARY.md).
