# omniMD: Rust MLIP molecular dynamics

omniMD is alpha software under the MIT license. This development line starts
from upstream main 761b2484feee132426183c9b66ff2b126d1f7c4a and adds a separate
CPU MLIP workspace with one fallible, explicit-unit, float64 E/F/virial contract.
It is an initial software delivery, not a physical model or a performance claim.

The [physical-label pilot and controlled diagnosis](mlmd/PHYSICAL_DATA_VALIDATION.md)
used official DeepMD `nopbc` labels, 168 training / 43 diagnostic-validation frames
and a 531-parameter model. **Pipeline feasibility passed; both trained models
remain unsuitable for scientific acceptance or production use.** The original
1,000-update fit worsened validation force RMSE from 3.933565 to 4.164336 eV/angstrom.
Changing only `energy_weight` from 1 to 0.1 reduced it to 3.707381, but validation
energy/atom RMSE was 0.523472 versus 0.455950 eV/atom for the original fit
(initial 0.569513 eV/atom).
New training total-energy RMSE also worsened versus initialization:
6.626241->6.650750 eV. This demonstrates a loss-weight tradeoff. The 43 frames
have been used for diagnosis; there is no untouched final test. Only 7-10-atom
v8 frames were exercised, and two-step CPU/GPU optimizer consistency does not
prove agreement over all 1,000 updates or validate the new v9 path.

The latest local [CUDA training optimization](mlmd/GETTING_STARTED.md#27-use-resident-gpu-training-state-and-interpret-measured-speed)
retains optimizer state on the device between existing checkpoints. On the recorded
Windows/RTX 4090 workloads, combined changes reduced complete-process median time
by 7.7% for the public 531-parameter E/F fit and 2.4% for a 466-parameter synthetic
case versus `201cdef`. The 112-parameter tiny case remains faster on CPU. CPU stays
the default; these results do not establish physical accuracy or general speedup.

The CPU baseline was published as fce0af9f6a49592a48c13883dfc3aa0d3eb4a28e.
The [CUDA release 9074817](https://github.com/TANG-LAB-WHU/omniMD/commit/907481708ae5f3df4228a9fb5d5b1110c39f5e23)
is now published on shawn_dev. It adds two explicitly selected inference paths
with recorded CPU/GPU numerical
agreement: [official DeepMD C API CUDA](mlmd/deepmd/CUDA_VALIDATION.md) and
[bounded native CUDA](mlmd/native/CUDA.md). The native unified CLI also passed a
short NVE/checkpoint continuation [check](mlmd/evidence/native-cuda-cli.json);
C API GPU batch evaluation and a two-step MD smoke test passed.

CPU remains the default at every system size. The six-atom C API sample measured
0.486 ms/frame on CPU versus 4.018 ms/frame on GPU, about 8.27 times slower with
kernel verification and synchronization included. Separately, the native
216-atom sample measured a CPU/GPU time ratio of 1.155, while its small cases were
about 8.4 times slower on GPU. These bounded measurements have different scopes;
they do not establish general acceleration or long-time stability. Local training extensions include CUDA readout features with CPU QR and bounded
all-neural-parameter E/F gradients and Adam on CUDA. Geometry, reduction and
assessment remain CPU work; optional tiled GPU prediction/loss preparation is
described below. Tiny training cases can be slower on CUDA. Runtime/subset requirements
are documented under [device selection](mlmd/engine/README.md#explicit-native-cuda-inference).
The [published CUDA release scope](mlmd/RELEASE.md#published-cuda-release)
records the code, documentation and compact validation evidence included.

The local candidate `06439f1` also passed [Linux native CUDA acceptance](mlmd/LINUX_CUDA_VALIDATION.md):
Ubuntu 22.04.5 x86_64 under WSL2, the official cached DeepMD CUDA image, RTX 4090,
driver 617.42 and NVRTC 12.9.86. This covers one FP64 inference fixture, 24-frame
tiled E/F/V training and exact same-runtime v8 continuation. The small v8 run
was slower than v7; no speed or broader-platform claim follows. C API execution
and cross-platform checkpoint restore were outside this acceptance. The later
v9 large-frame path below is not covered by this Linux validation.

This local branch additionally includes a conservative native CUDA neighbor
candidate cache, with final-version timing, MD and restart acceptance. It reduces
host neighbor preparation; kernels and the CPU path are unchanged. This follow-up
is **not published**. See [its scope and measurements](mlmd/RELEASE.md#local-native-cuda-cache-follow-up).

The current **local integrated follow-up** also contains cooperative native CUDA
fitting and persistent host buffers, plus standalone C API batching, explicit
frame/atom parameters, atomic outputs and model inspection. Native final-code
acceptance measured the same 192-atom, 20-step complete MD job at **412 ms** versus
**662 ms** previous CUDA and **1352 ms** CPU; the before/after whole-job ratio is
**1.61x**, separate from the **3.74x** warmed single-frame ratio for that model.
These are bounded measurements with startup/output/checkpoints included in MD.
[The final receipt](mlmd/evidence/native-cuda-integrated-fit.json) includes CPU/GPU
parity and exact restart. C API CPU batch/parameter/default/atomic paths now have real TF/PT numerical
validation; the new GPU paths remain untested. Use the
[standalone C API guide](mlmd/deepmd/BACKEND_GUIDE.md) for those options; shared
engine now supports versioned parameter datasets and fixed-parameter CPU C API MD. This integrated follow-up is **not published**.

The local native source `dc5449e` is integrated as `65a7192`: explicit CUDA-assisted
frozen-readout features, CPU QR and CPU assessment. [Its receipt](mlmd/native/fixtures/cuda-readout-report.json)
records same-model parity; the 3/4-atom case took 21.37 ms for CUDA features versus
5.15 ms CPU, plus 195.88 ms CUDA initialization. Full-network CUDA source `f4dbf62` is integrated as `02b5457`: all supported neural
E/F gradients and Adam, with CPU inference/loss seeds, geometry, reduction and
assessment. CPU remains the default; see [the bounds and checkpoint rules](mlmd/native/CUDA_FULL_TRAINING.md).
The recorded 3.46x ratio concerns a warmed 64-atom gradient stage only; the tiny
complete training case was slower. This follow-up remains local.

## GPU se_e2_a inference, compression and dataset statistics

The `gpu` feature of `omnimd-native` evaluates se_e2_a energy, forces and virial
entirely on one NVIDIA GPU through the `omnimd-gpu` crate (cudarc, NVRTC kernels
with a disk cache, cuBLAS). It has no network-size limits, selects neighbors like
DeepMD-kit 3, and splits large frames into chunks sized from free device memory.
Networks run in float64 or float32; geometry, reductions, forces and virial stay
float64. `--compress` replaces the embedding networks by piecewise quintic tables
(DeepMD's `dp compress`; here in pieces of uneven length, each within a tolerance
of its network) and falls back to the exact networks for any frame that leaves
the table range. The kernels are laid out by what was measured to cost
time on the card (host calls, double precision divisions, loads from global
memory); the numbers and the rules drawn from them are in
[`mlmd/gpu/README.md`](mlmd/gpu/README.md).

Measured on one RTX 4090 with the water se_e2_a benchmark model (sel [46, 92],
embedding [25, 50, 100], fitting [240, 240, 240]). The weights are untrained, so
these numbers show throughput and same-model agreement, not physical accuracy.
Milliseconds per energy/force/virial evaluation, the least of five samples of
each engine (three for the other descriptors below), the two taking turns in
one session on 2026-11-22
([record](mlmd/bench/results/omnimd-deepmd-alternated-rtx4090-2026-11-22.json),
with the medians; [method](mlmd/bench/README.md#both-engines-in-one-session)).
DeepMD-kit 3.1.3 is LAMMPS `pair_style deepmd` (PyTorch backend) in Docker
Desktop's Linux VM on the same card, with the atoms held in place; "tables" are
`--compress 0.5` in omniMD and `dp compress` in DeepMD-kit:

| Atoms | Precision | DeepMD-kit | omniMD | DeepMD-kit, tables | omniMD, tables |
| ---: | --- | ---: | ---: | ---: | ---: |
| 192 | float64 | 10.9 | 1.24 | 8.38 | 0.721 |
| 192 | float32 | 8.46 | 0.243 | 7.21 | 0.184 |
| 1,536 | float64 | 39.7 | 8.38 | 20.7 | 4.54 |
| 1,536 | float32 | 11.0 | 1.26 | 8.05 | 0.655 |
| 12,288 | float64 | 277 | 65.6 | 115 | 34.8 |
| 12,288 | float32 | 59.1 | 7.68 | 23.9 | 3.08 |
| 98,304 | float64 | out of memory | 522 | out of memory | 275 |
| 98,304 | float32 | out of memory | 60.3 | out of memory | 24.6 |

- DeepMD-kit takes 4.2 to 35 times as long as omniMD with the exact networks
  and 3.3 to 39 times as long with tables. At 192 atoms most of a LAMMPS step
  is a cost that does not grow with the atoms, so the ratios there say little
  about the arithmetic; at 12,288 atoms they are 4.2 and 7.7 with the networks
  in float64 and float32, and 3.3 and 7.8 with tables.
- At 98,304 atoms every DeepMD-kit run (se_e2_a in float64 and float32 and the
  hybrid below in float32, each with and without tables) stopped at its first
  evaluation with "CUDA out of memory. Tried to allocate 115.96 GiB" from
  PyTorch: the PyTorch backend on one MPI rank, as run here.
- Agreement with DeepMD-kit CPU float64 at 192 and 1,536 atoms: the float64
  energy is identical with and without tables, and forces differ by at most
  2e-17 eV/angstrom with the networks and 2e-14 with tables; float32 forces are
  within 9e-9 eV/angstrom.
- The two sides are not timed alike. With the atoms held (a step of 1e-6 fs:
  the untrained model has no repulsion and heats in dynamics) LAMMPS never
  rebuilds its neighbor list, while every omniMD evaluation bins the atoms on
  the host, searches their neighbors on the device and downloads its result.
  LAMMPS keeps its default skin of 2 angstrom (neighbors to 8). DeepMD-kit's
  float32 is a model built in float32; omniMD runs the networks of the float64
  model in float32, with geometry, reductions, forces and virial in float64.
  DeepMD-kit's tables have uniform steps, omniMD's are pieces of uneven length
  within a tolerance of each network. omniMD's host side was held to the
  performance cores; Docker's VM was not. Until 2026-11-22 this table had
  DeepMD-kit's times from NVE dynamics at 0.5 fs and omniMD's of 2026-11-10.
- Later the same day the host's share of a large frame shrank: the results
  come back through page-locked memory kept between evaluations, and the
  sums of each frame are formed on the device, so only the forces and those
  sums are copied back
  ([summary](mlmd/bench/results/SUMMARY.md#the-hosts-share-of-a-large-frame-2026-11-22)).
  With tables in float32, alternated with the build of the table above in
  one session, an evaluation of 12,288 atoms took 2.76 ms against 3.07, and
  one of 98,304 atoms 22.1 ms against 24.5
  ([record](mlmd/bench/results/host-path-rtx4090-2026-11-22.json)); the
  table keeps the times of its own session.
- That evening the force kernel of se_e2_a began to give each neighbor slot
  a group of 8 lanes in float32 and of 16 in float64 instead of a warp, four
  or two slots a warp at once
  ([summary](mlmd/bench/results/SUMMARY.md#the-force-kernel-of-se_e2_a-with-a-group-of-lanes-a-slot-2026-11-22)).
  Alternated with the build before in one session, its stage took 35% to
  38% less at 12,288 and 98,304 atoms; an evaluation with tables in float32
  took 2.50 ms against 2.77 at 12,288 atoms and 20.0 ms against 22.2 at
  98,304, in float64 32.4 ms against 34.6 and 257 against 274
  ([record](mlmd/bench/results/force-lanes-rtx4090-2026-11-22.json)). With
  the exact networks in float32 the stage took 2.7% and 5.4% more at 1,536
  and 12,288 atoms, an evaluation 0.1% and 0.3%.
- On 2026-11-23 the two smaller follow-ups of that change were timed
  against it, by rules written before the timings, and neither was kept:
  16 lanes for the exact networks in float32, in a kernel of their own,
  took 2.3% and 5.5% off their stage at 1,536 and 12,288 atoms but added
  1.3% to an evaluation of one 88-atom HAP frame, and the sums across the
  warps just past what the device keeps at once added 2.0% at 1,120
  blocks. The trainer's force kernel takes 16 lanes in float32 since then,
  by a rule of its own, with no gain measured. Before them an evaluation of
  98,304 atoms was taken apart kernel by kernel: the neighbor search kernel
  takes 13% of it, and the card idles about 3.2 ms (15%) between
  evaluations
  ([summary](mlmd/bench/results/SUMMARY.md#where-a-large-frame-goes-and-two-follow-ups-of-the-lanes-not-kept-2026-11-23)).
- The three-body descriptor se_e3 with the same cutoff and `sel` and embedding
  [10, 20, 40] costs 1.93 ms at 192 atoms and 13.4 ms at 1,536 in float32
  (DeepMD-kit: 27.5 and 209 ms), 9.42 and 70.1 ms in float64, and 833 ms at
  98,304 atoms in float32. DeepMD-kit takes 88.5 ms at 192 atoms in float64;
  at 1,536 its 20 untimed steps took 208 s and the run gave no result within
  its 1,500 s, the memory in use on the card reaching 23,355 of 24,564 MiB.
  omniMD's float64 result equals DeepMD-kit's at 192 atoms within 1e-14 eV per
  atom and 6e-17 eV/angstrom. Both sides evaluate the embedding networks
  themselves there (omniMD once for each pair of neighbors of an atom,
  DeepMD-kit for every pair of its slots, 4.2 times as many in this water):
  the compressed model that DeepMD-kit 3.1.3 makes of this one did not
  reproduce the uncompressed model (see the summary), so it gave no baseline.
  omniMD has tables for these networks since 2026-11-18, in pieces of uneven
  length that are each within a tolerance of the network (a table of one
  step is either wrong by percent for a trained network of this descriptor,
  or of gigabytes): 0.415 ms at 192 atoms and 2.34 ms at 1,536 in float32,
  1.13 and 6.70 ms in float64, 143 ms at 98,304 atoms in float32, with
  float64 forces within 7e-14 eV/angstrom of DeepMD-kit's uncompressed
  model at 192 atoms.
- A hybrid of two descriptors, se_e2_a to 4.0 angstrom (`sel` [16, 32]) and
  se_e2_r to 6.0 angstrom ([46, 92]), costs 0.299 ms at 192 atoms and 9.03 ms
  at 12,288 in float32, and 0.228 and 5.03 ms with embedding tables;
  DeepMD-kit takes 11.0 and 72.2 ms, with tables 9.07 and 32.3 ms. At 98,304
  atoms omniMD takes 70.8 ms (39.6 ms with tables), where DeepMD-kit runs out
  of memory as above. omniMD's float64 energy equals DeepMD-kit's at 192 and
  1,536 atoms to the last digit, and its forces within 1e-16 eV/angstrom.
  This hybrid is no cheaper than the plain se_e2_a model, which took 0.243
  and 7.68 ms in omniMD and 8.46 and 59.1 ms in DeepMD-kit in the same
  session. A hybrid of se_e2_a with an se_e3 descriptor to 4.0 angstrom costs
  0.447 ms at 192 atoms and 2.73 ms at 1,536 in float32 (DeepMD-kit: 12.3 and
  35.8 ms), 0.259 and 0.974 ms with tables, and 18.7 ms at 12,288 atoms (5.13
  with tables), where DeepMD-kit's held runs of 2026-11-16 gave no result
  within 300 s.
- Scope: one GPU, se_e2_a or se_e2_r with one embedding net per type pair or
  per neighbor type, se_e3, or a hybrid of these descriptors. The engine can
  select it for dynamics, minimization, batch evaluation and model deviation
  (`gpu_runtime` in the model options). It has a trainer of its own (below); the
  bounded `train-native` trainers do not use it.
  [Method and raw results](mlmd/bench/README.md), [summary](mlmd/bench/results/SUMMARY.md).

On a trained five-element se_e2_a ensemble (the HAP pipeline's iteration-4
models, 83 DFT-labelled test frames of 54 to 224 atoms), float64 omniMD matches
DeepMD-kit to 6e-13 eV/angstrom in forces and reproduces the recorded test errors.
Those systems are bound by per-step latency: NVT dynamics through `omnimd-mlmd`
takes 0.7 to 1.5 ms per step in float64 (0.2 to 0.3 ms in compressed float32)
against 13.6 to 15.7 ms for LAMMPS `pair_style deepmd` with the same models and
a neighbor list rebuilt every step. A pass over the device can carry more than
one frame and more than one model: sixteen copies of the 88-atom system are
evaluated in 1.4 ms in compressed float32 (0.41 of sixteen separate
evaluations), and the four models of the ensemble on one frame in 0.5 ms
against 0.8 ms one by one, which is what `model-deviation` and `batch-eval`
now do. `run-replicas` advances several runs of one model together the same
way: thirty-two replicas of the 88-atom system take 71 microseconds per step
each against 215 alone, with the trajectory and checkpoints each would write
alone. A run can carry the other members of its model's ensemble along
(`deviation`): every sampled frame is then written with their model deviation
and a candidate mark, for 6 to 8% more time at one sampled frame in fifty
steps. `collect-candidates` then writes the candidate frames of all runs as
one dataset, with a limit per run, and the share of the frames the members
agree on. An exploration with the pipeline's protocol, eight systems at two
temperatures for 40,000 steps each with the four models compared every 50
steps, is one `run-replicas` of 70 seconds and one `collect-candidates` of 2
seconds: 9,198 run-steps per second against 58 to 78 in the LAMMPS records
of the same models. The ensemble is not a production-qualified potential; see the
[summary](mlmd/bench/results/SUMMARY.md) for the protocol.

`omnimd-native-gpu-train` trains se_e2_a, se_e2_r and se_e3 models of any
size, and hybrids of them, on the
GPU with DeepMD-kit's `ener` loss, learning-rate schedule and Adam settings,
from a DeepMD `model` section or an existing native model, with one embedding
net per pair of types or per neighbor type (`type_one_side`), with snapshots
and restart. On the
frames and with the settings of that pipeline's iteration-0 DeepMD training (171
frames, 300,000 steps per model), four models train at once in 30 minutes with
float32 networks, against 30.8 hours recorded for the pipeline's four DeepMD-kit
models one after the other on the same machine, and they score like those on
the pipeline's 83-frame sealed test: force RMSE 0.0698 to 0.0723 eV/angstrom
(DeepMD 0.0699 to 0.0711), energy RMSE 1.64 to 1.78 meV/atom (DeepMD 1.64 to
1.74). The radial descriptor se_e2_r trains at about 1,240 steps per second
with float32 networks on those frames (se_e2_a 544), where DeepMD-kit's PyTorch
backend makes 41 on the same request; after 20,000 steps of it four seeds of
each trainer score alike on the sealed test (force RMSE 0.304 to 0.352
against 0.338 to 0.347 eV/angstrom). On this data the radial descriptor is no
substitute for se_e2_a: the full schedule brings it to 0.15 to 0.18, where
se_e2_a reaches 0.07. A hybrid of se_e2_a and se_e2_r descriptors trains the
same way, each descriptor with its own cutoffs, selection and nets. Given
DeepMD-kit's initial model and the frames in the order DeepMD-kit took them,
omniMD arrives at DeepMD-kit's trained model (4,000 steps with energy and
force terms in float64: its parameters end 4e-10 away on average). It
trains at 471 steps per second with float32 networks and 220 with float64
ones against DeepMD-kit's 27.8 in float64 on the same request. On those frames an angular
descriptor to 4 angstrom with a radial one to 6 reaches a force RMSE of
0.072 to 0.075 eV/angstrom after the full schedule: close to se_e2_a to
6 angstrom (0.070 to 0.072), well below the angular descriptor to 4
angstrom alone (0.095), and no cheaper to train or to evaluate than
se_e2_a to 6. The three-body descriptor se_e3 trains the same way, alone or
in a hybrid, and again omniMD arrives at DeepMD-kit's model from its initial
model and frame order: 9e-11 away on average after 4,000 steps with se_e3
alone, and 3e-8 to 5e-8 with the hybrid of the HAP project's earlier models
(se_e2_a with se_e3), a run that amplifies any change of its initial model
two thousand times. On the same frames se_e3 to 4 angstrom trains at 1,293
steps per second with float32 networks and 355 with float64 ones where
DeepMD-kit makes 19.3, the pipeline's se_e2_a with that descriptor beside
it at 414 and 160 against 7.0, and the earlier hybrid at 1,486 and 647
against 7.75. After the full schedule the three-body descriptor beside the
pipeline's se_e2_a lowers the force RMSE on the sealed test from 0.070 to
0.072 eV/angstrom to 0.064 to 0.066 (four seeds each, every one of them
below every one without it), for 23 percent more time to train and 1.6 to
2 times the time to evaluate; alone to 4 angstrom it reaches 0.13.
[Contract, checks and the full comparison](mlmd/native/GPU_TRAINING.md).

`omnimd-native-stat` computes DeepMD-style dataset statistics: `dp neighbor-stat`,
the se_e2_a input statistics `davg`/`dstd` and the per-type energy bias. They
match DeepMD-kit 3.1.3 to 1e-8 on the recorded fixture; the GPU trainer applies
them when it starts from a DeepMD `model` section. Commands are in [guide step 28](mlmd/GETTING_STARTED.md#28-run-se_e2_a-on-the-gpu-runtime-and-compute-dataset-statistics).

## Stepwise setup and current feature coverage

Start with the [complete English setup and usage guide](mlmd/GETTING_STARTED.md):

1. Build the separate `mlmd/Cargo.toml` workspace and its engine/native binaries.
2. Run `mlmd/scripts/run_cpu_tutorial.py` in a new output directory. It generates
   synthetic data and executes import, preflight, readout training, evaluation,
   CSVR NVT and exact restart with complete saved requests.
3. Choose native-v1 weights or a trusted official DeepMD C API runtime; validate
   conversion scope, type map, units and model metadata.
4. Import explicit DeepMD NPY sets or XYZ/extxyz; use `inspect-dataset` for full
   JSONL validation, byte SHA256 and E/F/V label coverage without loading a model.
5. Run batch assessment, then the supported CPU readout/full-network trainer with
   explicit validation identities and loss scales.
6. Minimize when appropriate, inspect convergence, run NVE, CSVR NVT or
   constant-pressure NPT, and continue from a matching checkpoint.
7. Select native CUDA or official C API CUDA explicitly, verify same-model E/F/V,
   and compare loading, warmed evaluations and full MD separately.
8. Follow the guide's troubleshooting table and retain hashes and completed reports.

[The versioned feature matrix](mlmd/FEATURE_MATRIX.md) compares this implementation
with official **DeepMD 3.1.3** and distinguishes native, delegated, partial and
missing functions. Broad DeepMD feature parity is not implemented. The preflight
command and executable tutorial are local follow-ups pending publication; the
[final CPU tutorial receipt](mlmd/evidence/cpu-tutorial-integrated.json) records the actual run.

## CPU inference, training and simulation capabilities

- **DeepMD C API inference:** one official model handle, one combined
  energy/force/virial call, explicit CPU selection, and strict type-map,
  coordinate, cell, unit, shape, finite-value and error checks. This is FFI;
  double input/output buffers do not change internal model precision.
- **Native Rust inference:** CPU float64 se_e2_a, periodic neighbor images,
  cutoff/normalization, embedding and fitting networks, analytic forces and
  strain virial. Only the documented native-v1 architecture subset is accepted.
- **Model conversion:** bounded DeepMD 3.1.3 .dp/safe-YAML conversion and two
  audited frozen .pb graph profiles. Unsupported models fail explicitly.
- **Parameterized CPU models:** explicit frame/atom data, opt-in model-stored frame
  defaults, v2 dataset/NPY import and fixed-context MD/checkpoint v3 through C API.
  [Stepwise parameter guide](mlmd/engine/PARAMETERS.md) records strict ordering and
  TF/PT acceptance, including PT numerical rather than bitwise restart parity.
- **Data and prediction:** plain XYZ, a documented extxyz subset and explicit
  DeepMD NPY sets; shared JSONL, optional E/F/virial labels, batch evaluation,
  error metrics and model/capability queries. No units or labels are guessed.
- **Model disagreement:** retained ensembles on the CPU or on one GPU, absolute
  E/F/virial deviation and explicit force-window candidate flags;
  [stepwise usage](mlmd/engine/MODEL_DEVIATION.md) and independent synthetic
  replay. On the GPU runtime the members of an ensemble are evaluated together,
  a batch of frames per device pass. Disagreement is not calibrated uncertainty.
  [Hash-bound subset extraction](mlmd/engine/DATASET_SELECTION.md) preserves reviewed
  IDs and labels for assessment or externally labeled training inputs.
- **Native readout training:** frozen-hidden final linear weights (optional
  biases), exact E/F features, rank-aware QR, explicit validation splits and
  unchanged native-v1 model export.
- **Full supported-network training:** tanh embedding/hidden networks with linear
  outputs, exact mixed E/F parameter gradients, default full-batch or explicit
  minibatch Adam and deterministic checkpoints. All neural weights/biases/timesteps
  train; geometry, normalization
  and separate atomic offsets stay fixed. CPU reference-size limits apply.
- **Fixed-cell minimization:** both inference backends, bounded steepest descent,
  force plus energy-change stopping criteria, explicit failure/nonconvergence,
  and geometry-only extxyz/JSONL exports.
- **MD:** NVE velocity Verlet, upstream CSVR NVT and constant-pressure NPT
  (stochastic cell rescaling with isotropic, layer, per-axis or fully flexible
  cell, optionally with fixed axes, and a Metropolis acceptance test; see the
  [engine README](mlmd/engine/README.md#configuration-and-units)), explicit masses,
  seeded thermal initialization, streamed trajectories with pressure and volume,
  and exact RNG restart.
  The runner uses the actual omniMD System through fallible MlipDynamics,
  caching combined E/F/virial and rolling back failed steps.

## Quickstart: no external model runtime

From the repository root with Rust/Cargo installed:

    cargo build --locked --manifest-path mlmd/Cargo.toml -p omnimd-mlmd -p omnimd-native --bins

Run the checked-in synthetic fixture (a software test, not a chemistry model).
On Linux/macOS:

    cat mlmd/native/fixtures/frame.jsonl | ./mlmd/target/debug/omnimd-native-eval mlmd/native/fixtures/synthetic-v1.json
    ./mlmd/target/debug/omnimd-mlmd capabilities

On Windows PowerShell:

    Get-Content mlmd/native/fixtures/frame.jsonl | .\mlmd\target\debug\omnimd-native-eval.exe mlmd/native/fixtures/synthetic-v1.json
    .\mlmd\target\debug\omnimd-mlmd.exe capabilities

The first command prints one combined E/F/virial JSON result. Native execution
requires no Python, DeepMD, LibTorch or chemfiles. The initial build resolves the
locked Rust dependencies; offline builds need them cached already. Windows and
Linux builds were checked with Rust 1.99; the declared lower MSRV has not been
independently verified.

For a short MD example, copy [native-nvt.json](mlmd/engine/examples/native-nvt.json),
set native_model to the absolute fixture path, then use:

    omnimd-mlmd check request.json
    omnimd-mlmd run request.json new-trajectory.jsonl

Use the executable path above if it is not on PATH. Output names must be new.
The example parameters only exercise software behavior. The [workflow](mlmd/WORKFLOW.md)
provides exact requests for import, prediction/training, minimization and MD.
Real C API inference additionally requires a trusted compatible DeepMD runtime;
see [DeepMD setup and limits](mlmd/deepmd/README.md).

## Current limitations

- Cell relaxation or constraints; NPT has isotropic, semi-isotropic, per-axis
  and fully flexible coupling and fixed axes, but no surface-tension coupling. Optional
  CUDA inference is available within the separate scopes above; training
  retains CPU stages.
- The GPU trainer (se_e2_a, se_e2_r, se_e3 and hybrids of them) takes one frame per step on one GPU
  and has no fine-tuning controls; a frame of an se_e3 descriptor, whose rows go with the square of
  an atom's neighbors, must fit the device in one piece. The bounded `train-native` trainers do not
  use the GPU runtime, and the engine passes each frame through the host.
- Arbitrary DeepMD architectures (attention, type embedding) or generic
  .pb/.pth/pickle import.
- Partial PBC, mixed-type/virtual atoms or automatic unit/type-map guessing.
- Classical-potential mixing and the full legacy Simulation/Propagator pipeline.
- Sharded/distributed or end-to-end GPU training, learned normalization or generic
  DeepMD training compatibility; the bounded trainers do not tie duplicated
  one-side embeddings (the GPU trainer does).
- Production validation, long-time stability guarantees or general speedup claims.

The root legacy workspace remains available with its original default features;
these do not imply equivalent support in the new MLIP runner. Training covers the
documented native tanh subset; it does not reproduce the full DeepMD training stack.

## Documentation, verification and licensing

- [Shared contract](mlmd/CONTRACT.md), [data formats](mlmd/engine/DATA.md),
  [minimization](mlmd/engine/MINIMIZATION.md), [MD runner](mlmd/engine/README.md).
- [Native models](mlmd/native/README.md), [conversion](mlmd/native/CONVERSION.md),
  [frozen graph scope](mlmd/native/FROZEN.md), [readout training](mlmd/native/TRAINING.md),
  [full supported-network training](mlmd/native/FULL_TRAINING.md).
- [Validation and its limits](mlmd/VALIDATION.md), [roadmap](ROADMAP.md),
  [publication scope and procedure](mlmd/RELEASE.md).
- [MIT license](LICENSE), [third-party notices](mlmd/THIRD_PARTY_NOTICES.md).
  DeepMD-derived topology profiles retain separate LGPL notices/licenses.

The release contains source, small synthetic fixtures, reproducible scripts,
compact validation summaries and licenses. External runtimes, public pretrained
weights, generated training data, raw operational logs and private research
inputs are excluded. No HAP research inputs or old HAP computations are used.
The CPU baseline and CUDA release are published on shawn_dev.

The local [external-label return workflow](mlmd/engine/EXTERNAL_LABELS.md) now
connects reviewed candidates to a new labeled dataset and explicit CPU retraining.
Source/return hashes and exact frame/atom/parameter identity are required;
conflicting labels reject. It preserves dataset v2 context without adding native
parameterized training, and launches no external calculator or automatic loop.

Full supported-network CPU/CUDA training also accepts optional
[virial supervision](mlmd/native/VIRIAL_TRAINING.md): all nine row-strain entries,
explicit per-atom scale/weight and loss-bound checkpoint v3 on CPU/default
hyperdual (v4 on reverse CUDA) for full-batch training; explicit minibatching
uses v5. E/F-only CPU-v1 and hyperdual CUDA-v2
continuation remain compatible. Frozen-readout fitting remains E/F-only.
The recorded tiny joint-loss full process was 619 ms CPU versus 1816 ms CUDA;
that historical hyperdual measurement does not establish a training speedup.
This extension is local and unpushed.

The local [cached reverse CUDA executor](mlmd/native/CUDA_REVERSE_TRAINING.md) is
available through explicit `config.cuda_gradient="reverse"`; hyperdual remains
the default. Full-batch reverse checkpoint v4 binds loss/executor and device/runtime/program
identity. On B's original 466-parameter/64-atom/100-step E/F/V fixture, three fresh
process medians were CPU 2963 ms, hyperdual 1349 ms and reverse 877 ms. The tiny
112-parameter case remained faster on CPU (615 ms versus reverse 1573 ms). These
bounded measurements include startup, IO/checkpoints and export; they do not
establish production scaling or general speedup. Existing bounds remain in force.

[Final integrated reverse verification](mlmd/evidence/native-reverse-integrated.json)
passed six joint-loss steps, exact 3+3 checkpoint continuation, CPU E/F/V agreement
within 4.45e-16 and zero model-reload difference. Device, loss and executor changes
reject without training output. This short check adds no timing claim.

Optional [fixed training geometry reuse](mlmd/native/FIXED_GEOMETRY_TRAINING.md)
is enabled with `config.fixed_geometry_cache=true` and an explicit CUDA device,
using either gradient executor. Default is false; CPU-only training rejects it.
The eager cache retains selected training-frame neighbors, full periodic
displacements and normalized R/J within a 64 MiB host-payload cap. Current-model
predictions and loss directions are rebuilt every step. Identity mismatch stops
training; cache on/off may toggle at restart without changing numerical state.
See [the step-by-step entrypoint](mlmd/GETTING_STARTED.md#17-reuse-fixed-training-geometry-explicitly).
Recorded three-process off/on medians are 865.5/818.4 ms for the 64-atom synthetic
cluster and 6946.3/956.9 ms for its triclinic periodic variant. The benefit is
workload-specific CPU geometry reuse, not a general GPU acceleration claim.

[Final cache integration evidence](mlmd/evidence/native-geometry-cache-integrated.json)
confirms six steps, exact 3+3 and both cache-toggle continuations, agreement with
the prior uncached checkpoint, CPU E/F/V differences below 4.45e-16 and zero
reload difference. The short replay retained 221,414 bytes for 18 training frames;
15 affected CPU-safe checks and the offline release build passed.

[Deterministic minibatching](mlmd/native/MINIBATCH_TRAINING.md) is now available
for CPU and both CUDA full-network executors through explicit
`config.minibatch={"batch_size":7,"seed":20261016,"order":"shuffle_each_epoch"}`.
Absent/null retains full-batch behavior. Short tail batches are retained and
normalized by their actual frame count; `steps` counts optimizer updates.
Without an explicit learning-rate schedule, checkpoint v5 stores the validated
next-batch sampler and supports exact
same-runtime mid-epoch restart. The [step-18 guide](mlmd/GETTING_STARTED.md#18-select-deterministic-minibatches)
and [integrated receipt](mlmd/evidence/native-minibatch-integrated.json) cover
seven updates, tail/cache-toggle continuation and legacy full-batch compatibility.
Default memory access retains the 1024-frame full-training limit. Explicit indexed
access below reads frames individually and has separate index/record/count limits.

[Indexed dataset access](mlmd/native/INDEXED_TRAINING.md) is now available through
top-level `"dataset_access":"indexed"` plus explicit `config.minibatch`. It
validates the existing v1 JSONL file, retains offsets/IDs/line hashes, reads one
frame at a time and verifies the whole file before checkpoints/publication.
Small inputs can switch memory/indexed on v5 restart exactly; memory is the default.
Indexed limits are 100000 frames, 16 GiB input, 256 KiB records and 64 MiB retained
index payload. Index/sampler/report/checkpoint memory remains O(frame count);
these are not peak-process-memory or validated production-capacity claims.
The [step-19 guide](mlmd/GETTING_STARTED.md#19-read-training-data-through-a-bounded-index)
and [final integration evidence](mlmd/evidence/native-indexed-integrated.json)
record exact CPU/CUDA state, metrics and storage/cache-switch continuation.
The separate 2051-frame, 128.19-MiB boundary case is deliberately whitespace
padded; its measured 12.83-MiB Windows peak working set is not a physical-data
throughput benchmark. That indexed milestone preserved the default 64-atom and
geometry-cache bounds; the later explicit large-frame option is described below.

The [startup validation consolidation](mlmd/native/INDEXED_TRAINING.md#startup-validation-consolidation)
removes two duplicate frame-validation passes per full-training CLI invocation.
Public validation, per-read hashes and publication whole-file checks remain.
[CPU-only final acceptance](mlmd/evidence/native-indexed-startup-integrated.json)
records fresh and historical-checkpoint continuation, default-memory equality,
48 fewer reads per 24-frame invocation, unchanged hash counts, exact checkpoint
contents/model bytes/metrics and zero reload difference. This tranche adds no GPU
runtime validation or capacity change. See [step 20](mlmd/GETTING_STARTED.md#20-interpret-startup-validation-and-io-counters)
for cumulative timing fields and the short CPU replay.

[Learning-rate schedules](mlmd/native/LEARNING_RATE.md) are available for full
neural-parameter training through optional `config.learning_rate_schedule`:
explicit constant or exponential staircase, indexed by globally completed Adam
updates. Absent/null retains constant behavior and historical checkpoint schemas;
an explicit schedule uses v6 (v7 with tiling, v8 with GPU loss, v9 with an expanded
frame ceiling) and must remain identical on resume. The
[step-21 guide](mlmd/GETTING_STARTED.md#21-select-an-optimizer-update-learning-rate-schedule)
and [final CPU engine receipt](mlmd/evidence/native-learning-rate-integrated.json)
cover boundary continuation, legacy v5 and refusal behavior. No model/atom limit
or CUDA kernel changes accompany this addition. Bounded multi-frame execution and optional GPU prediction/loss preparation are
available below. Larger model support and remaining end-to-end limits are
described in the [roadmap](ROADMAP.md#remaining-end-to-end-and-model-limits).

[Bounded multi-frame CUDA training](mlmd/native/TILED_TRAINING.md) is enabled with
explicit reverse CUDA and `config.cuda_tiling={"max_frames":16,"max_atoms":512}`.
With default CPU loss, each tile executes four gradient kernels, preserving per-frame reduction order,
optimizer batches, schedules and one Adam update per batch. Tile capacities bind
v7 restart identity with CPU loss (v8 for GPU loss); absent/null tiling retains
framewise execution and v1-v6. Limits
are 32 frames / 1024 aggregate atoms per tile, subject to work/payload planning;
the default is 64 atoms per frame, with explicit bounded expansion described
below. The 2048 neural-parameter and width-32 limits remain unchanged.
[Step 22](mlmd/GETTING_STARTED.md#22-batch-frames-in-bounded-cuda-tiles) explains
payload limits and diagnostics. [Final engine acceptance](mlmd/evidence/native-tiled-integrated.json)
records exact framewise equality and storage/cache-switch restart. Separate
synthetic whole-process medians improved 1071->551 and 4484->3568 ms; these are
bounded measurements, not physical-model or universal acceleration claims.

Optional [GPU predictions and detached loss seeds](mlmd/native/DEVICE_LOSS_TRAINING.md)
are enabled with `config.cuda_loss="gpu"` plus reverse CUDA and tiling. This path
reuses forward intermediates and binds v8 restart identity. Absent or `"cpu"`
retains the CPU-loss default and historical v7 state/PTX. Geometry, gradient
reduction/clipping and assessment remain CPU work. The
[step-23 guide](mlmd/GETTING_STARTED.md#23-select-gpu-prediction-and-loss-preparation-explicitly)
and [final receipt](mlmd/evidence/native-device-loss-integrated.json) cover CLI
forwarding, actual v7 compatibility, tolerant cross-mode comparison and exact
same-backend v8 restart. Historical CPU-loss/GPU-loss measurements remain in that
receipt. The [latest resident-state comparison](mlmd/native/RESIDENT_STATE_TRAINING.md)
measures the combined transfer, alignment and optimizer-state changes against
`201cdef`: 1332.67->1229.95 ms public E/F and 2946.89->2875.74 ms medium synthetic.
The tiny native CPU reference is faster, 504.19 versus 703.56 ms CUDA. GPU loss
remains explicit; state residency activates automatically within that path and
preserves checkpoint cadence. Frequent saves reduce its benefit; bounds remain.

[Resource-bounded larger training frames](mlmd/native/LARGE_FRAME_TRAINING.md)
use explicit `config.cuda_tiling.max_atoms_per_frame=256` with reverse CUDA.
Default/explicit 64 preserves prior v7/v8 behavior; CPU and untiled training stay
at 64. Above 64 selects v9, with GPU-loss backend v2 and complete strided force/
virial output coverage. Work and payload budgets are unchanged: the 112-parameter
fixture fits 256 atoms, but the 466-parameter example has effective capacity 138
and rejects 256. This is not a blanket 256-atom guarantee or a speed claim.
[Step 25](mlmd/GETTING_STARTED.md#25-opt-in-to-resource-bounded-larger-training-frames)
and the [final engine receipt](mlmd/evidence/native-large-frame-integrated.json)
record real 256-atom frames, exact v9 continuation and refusals. B's separate
12-update process observations were CPU-loss CUDA 1554 ms and GPU-loss CUDA
1721 ms; both still retain CPU geometry/reduction/assessment. New v9 Linux
execution remains unverified.
