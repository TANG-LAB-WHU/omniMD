# Several runs advanced together, and a lighter step, 2026-11-11

Two changes to molecular dynamics through `omnimd-mlmd`, both on the host.

**Replicas.** `run-replicas` advances several independent systems of one
model in step: one model evaluation per step for all of them, which the GPU
runtime answers in one pass over the device (the batches of the section
below), and one more for the systems whose cell moves at that step. Each
replica has its own ensemble and writes the trajectory and checkpoints it
writes alone; on the CPU they are identical to those of the single runs.

**The step itself.** A step used to copy the whole system, names and
molecules included, and to build the frame handed to the model five times
over; it now works on the system in place and takes the step back if any
part of it fails. The host's share of a step, measured with a model that
costs nothing (`tests/step_cost.rs`):

| atoms | NVE step, before | now | NVT step, before | now |
|---:|---:|---:|---:|---:|
| 88 | 13.4 µs | 1.0 µs | 20.7 µs | 1.75 µs |
| 224 | 35.7 | 2.5 | 50.7 | 3.9 |
| 1,024 | 165 | 11.6 | 222 | 16.3 |

Trajectories and checkpoints of the CPU path are the same bytes as before
(NVE, NVT and NPT runs of 400 steps compared between the two builds).

**Microseconds per step and replica** (`hap_replicas.py`: NVT at 300 K, CSVR
thermostat, 0.5 fs, sampled every 50 steps, model 0 of the HAP ensemble;
replicas of one system with different seeds; the engine's own count of its
stepping time, least of two runs; raw numbers in
`hap-replicas-rtx4090-2026-11-11.csv`). Float32 networks with compressed
embedding:

| system (atoms) | plain run, before | plain run | 2 replicas | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HAP bulk (88) | 235 | 215 | 144 | 102 | 104 | 90 | 71 |
| HAP(001) + water + CO2 (169) | 320 | 285 | 208 | 197 | 200 | 146 | 121 |
| monetite (224) | 361 | 320 | 271 | 242 | 208 | 171 | 138 |

Float64 networks:

| system (atoms) | plain run, before | plain run | 2 replicas | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HAP bulk (88) | 764 | 744 | 653 | 551 | 534 | 525 | 522 |
| HAP(001) + water + CO2 (169) | 1310 | 1272 | 1063 | 1039 | 1014 | 988 | 973 |
| monetite (224) | 1587 | 1544 | 1578 | 1447 | 1392 | 1372 | 1355 |

- A plain run is 8 to 11% faster in compressed float32 (20 to 41 µs less a
  step) and about 3% faster in float64, from the lighter step alone. ("Before"
  is the previous build timed by the difference of two runs of different
  length; that build did not report its stepping time.)
- Replicas in compressed float32: sixteen 88-atom systems advance at 90 µs
  per step each and thirty-two at 71 µs, against 215 µs alone, that is 2.4
  and 3.0 times the simulated time per day from one card (19,400 ns per day
  in all for thirty-two replicas at 0.5 fs, against 200 for one run). For
  169 and 224 atoms the gain is 2.0 and 1.9 times at sixteen replicas and 2.4
  and 2.3 times at thirty-two.
- The engine adds some 5 to 15 µs per replica and step to the evaluation of
  the batch (integrator, building frames, sampled output), so the table follows
  the batch timings below; the plateau between 300 and 1,400 atoms per pass
  is the fused fitting kernel, as there.
- Float64 gains 1.1 to 1.4 times, as in the batch timings.
- GPU replicas against CPU replicas of the same request (three systems, NVE,
  NPT with cell moves, NVT without a cell; test `gpu_runtime.rs`): sampled
  frames within 1e-9 over 40 steps in float64.

# Batches of frames and committees of models, 2026-11-11

A system of a hundred atoms leaves most of the card idle: its evaluation is a
dozen launches of a few microseconds that wait for each other. The GPU
se_e2_a path now takes more work per launch in two ways (layout in
[`gpu/README.md`](../../gpu/README.md)). A batch of frames is one list of
atoms on the device, with a neighbor search that keeps the frames apart; and
models of one architecture, such as an ensemble kept for model deviation, are
a second dimension of every launch, with one neighbor search for all of them.

**Microseconds per frame and model** in process (`hap_batch.py`, which runs
`omnimd-native-gpu-eval --bench-batches` over copies of one frame; least of
three runs; raw numbers in `hap-batch-rtx4090-2026-11-11.csv`). HAP models,
float32 networks with compressed embedding:

| system (atoms) | models | 1 frame | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HAP bulk (88) | 1 | 206 | 134 | 96 | 98 | 85 | 65 | 56 |
| | 4 | 125 | 97 | 87 | 84 | 76 | 60 | 49 |
| HAP(001) + water + CO2 (169) | 1 | 272 | 196 | 190 | 183 | 134 | 110 | 99 |
| | 4 | 199 | 174 | 161 | 164 | 122 | 98 | 88 |
| monetite (224) | 1 | 306 | 259 | 231 | 193 | 158 | 130 | 130 |
| | 4 | 233 | 221 | 208 | 176 | 139 | 123 | 119 |

Float64 networks:

| system (atoms) | models | 1 frame | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HAP bulk (88) | 1 | 733 | 643 | 542 | 521 | 521 | 515 | 504 |
| | 4 | 539 | 522 | 517 | 511 | 499 | 498 | 496 |
| HAP(001) + water + CO2 (169) | 1 | 1249 | 1045 | 1023 | 986 | 958 | 948 | 941 |
| | 4 | 1006 | 970 | 952 | 939 | 931 | 927 | 926 |
| monetite (224) | 1 | 1522 | 1563 | 1428 | 1372 | 1351 | 1337 | 1329 |
| | 4 | 1409 | 1356 | 1347 | 1325 | 1315 | 1314 | 1313 |

- **Batches** (one model, float32 compressed): a frame in a batch of 16
  costs 0.41 of a frame alone for 88 atoms, 0.49 for 169 and 0.52 for 224;
  in a batch of 64, 0.27, 0.36 and 0.42. Sixteen copies of the 88-atom
  system are evaluated in 1.36 ms, against 3.3 ms one after the other.
- **Committees**: the four models on one frame, which is what model deviation
  along a trajectory asks for, cost 0.50 ms for 88 atoms against 0.82 ms one
  by one (1.65 times faster), 0.80 against 1.09 ms for 169 atoms and 0.93
  against 1.22 ms for 224.
- **Float64** gains 1.15 to 1.45 times: its arithmetic already keeps the
  double precision units of the card busy for a single frame, so only the
  launches and the shared neighbor search are saved.
- Where the gain ends. With more atoms in a pass the launches no longer
  count and the stages are bound by what they compute: for 8 frames of 88
  atoms the fused fitting kernel is 44% of the pass (its layers after the
  first read their weights a row at a time), the force kernel 20%, the
  descriptor 14%. Above 256 atoms per type and model the fitting nets go
  layer by layer through cuBLAS, which costs 0.68 ms in launches for five
  types whatever the rows and little more for more rows; the two paths
  cross near 1,400 atoms as before, and between some 300 and 1,400 atoms a
  pass gains little per frame (the plateau at 2 to 8 frames of 169 atoms).
  That range is the next thing to work on.
- **Agreement.** The four models as one committee over the 83 HAP test
  frames in batches of 16 (`hap_batch.py --check`), each member against
  DeePMD's own evaluation with that model: float64 within 1.8e-13 eV per
  atom and 5.6e-13 eV/Å, float64 compressed 8.4e-10 and 3.9e-6, float32
  3.1e-7 and 8.5e-6, float32 compressed 4.5e-7 and 1.0e-5. The float64
  numbers are those of the single model; the float32 ones differ in the
  last digits because batches of that size take the cuBLAS fitting path. A
  batch gives each frame the neighbors it gets alone, bit for bit (tested
  against each system searched alone).

**Model deviation through the engine** (`hap_deviation.py`:
`omnimd-mlmd model-deviation`, the four models on `cuda:0`, the 83 frames of
54 to 224 atoms; time per frame from two runs over the frames once and twelve
times, which takes out loading; this build against the one before, which
evaluated member by member and frame by frame):

| mode | before, ms per frame | now | frames per second |
|---|---:|---:|---:|
| float64 | 3.79 | 2.42 | 1.57 times |
| float32, compressed | 1.17 | 0.47 | 2.48 times |

The deviations of the two builds agree to 3e-15 eV/Å in float64 (1e-6 in
compressed float32) and with those computed from DeePMD's reference forces to
1e-14 eV/Å (4e-6). A whole run over the 83 frames takes 1.7 s in float64 and
4.9 s compressed: loading four models, and for the compressed mode building
their embedding tables (about 0.8 s per model, on one thread).

`batch-eval` reads a dataset the same way, a batch of frames per pass.

# Fitting nets with the first layer in tiles, 2026-11-11

After the rework below the fitting nets were 40 to 60% of a small system's
evaluation, bound by reading their weights from global memory a row at a
time. For rows grouped by network (`GradientNetwork::forward_gradient_grouped`)
the first layer, 1600 to 240 here and three quarters of the arithmetic, now
goes through shared memory in tiles that serve all rows of a chunk; the other
layers stay row by row. Layout and measurements are in
[`gpu/README.md`](../../gpu/README.md).

**Microseconds per evaluation** in process (`omnimd-native-gpu-eval --bench
2500`), same HAP model and frames, the previous build and this one alternated
four times and the best run of each kept:

| system (atoms) | fp32 | fp32, tiles | fp32 compressed | fp32 compressed, tiles |
|---|---:|---:|---:|---:|
| HAP bulk (88) | 269 | 230 | 244 | 205 |
| HAP(001) + water + CO2 (169) | 473 | 368 | 402 | 301 |
| monetite (224) | 552 | 421 | 477 | 356 |

- Float32 evaluations are 1.17 to 1.34 times faster. Float64 is unchanged:
  there the arithmetic is the bound and the tiles measured slower (88 atoms,
  compressed: 457 to 503), so they stay off.
- The fitting kernels alone, 1600 -> 240 -> 240 -> 240 -> 1 nets, per call
  (`fitting_costs` probe): 88 rows in four type runs 125 to 81 µs, 224 rows
  270 to 149, 1024 rows of one type 757 to 498.
- Agreement with DeePMD on the 83 HAP test frames (`hap_check.py`): float32
  within 2.4e-7 eV per atom and 7.5e-6 eV/Å as before; float64 unchanged.
- Why best runs: the card is shared with the desktop (an animated wallpaper
  takes up to a fifth of it), single runs of one build differ by 10 to 25%
  and the best of four by about 2% (the float64 modes, whose code did not
  change, came out within 2% of their previous value). Whole-process
  timings of molecular dynamics through the engine scatter too much under
  that load to resolve this gain, so the table of steps below stands as the
  last clean record of those.
- The trainer, measured again on the idle card for the rework below (it
  shares those kernels; it does not use the tiled ones): float32 597 to 622,
  626 to 649 and 688 to 700 steps/s for 1, 2 and 4 requests at once; float64
  unchanged within the scatter (best of two alternated rounds 240 and 282,
  304 and 308, 333 and 336).

# Evaluation after the call and double-precision rework, 2026-11-10 (r3)

The GPU se_e2_a path was measured piece by piece (`mlmd/gpu/tests/costs.rs`,
results and consequences in [`gpu/README.md`](../../gpu/README.md)) and
rebuilt around what that showed: a host call costs 5 to 10 microseconds, a
division in double precision as much as sixty multiplications, and most
neighbor slots are empty. Same models, frames and protocols as the tables
further down; the previous build was run again in the same session.

**ms per MD step**, NVT at 300 K through `omnimd-mlmd`
(`hap-omnimd-rtx4090-2026-11-10-r2.csv`), LAMMPS from the record below:

| system (atoms) | LAMMPS `pair_style deepmd` | fp64 | fp64 compressed | fp32 | fp32 compressed |
|---|---:|---:|---:|---:|---:|
| HAP bulk (88), previous build | | 1.60 | 0.84 | 0.58 | 0.47 |
| HAP bulk (88) | 13.6 | 0.80 | 0.49 | 0.31 | 0.29 |
| HAP(001) + water + CO2 (169), previous build | | 2.70 | 1.35 | 0.83 | 0.77 |
| HAP(001) + water + CO2 (169) | 15.7 | 1.38 | 0.84 | 0.45 | 0.42 |
| monetite (224), previous build | | 3.10 | 1.26 | 0.86 | 0.75 |
| monetite (224) | 14.6 | 1.64 | 0.88 | 0.53 | 0.48 |

(The previous build's monetite row is the morning's record: its rerun
overlapped another load on the card.)

**ms per evaluation**, water (`omnimd-gpu-rtx4090-2026-11-10-r3.csv`):

| atoms | fp64 | fp64 compressed | fp32 | fp32 compressed |
|---:|---:|---:|---:|---:|
| 192, previous build | 1.97 | 1.04 | 0.69 | 0.65 |
| 192 | 1.47 | 0.83 | 0.47 | 0.39 |
| 1,536, previous build | 13.1 | 6.28 | 2.43 | 1.92 |
| 1,536 | 9.21 | 5.10 | 1.53 | 1.22 |
| 12,288, previous build | 99.6 | 46.1 | 16.7 | 12.4 |
| 12,288 | 70.1 | 38.7 | 9.59 | 7.24 |
| 98,304, previous build | 798 | 367 | 132 | 100 |
| 98,304 | 564 | 304 | 76.6 | 57.4 |

- Molecular dynamics of the HAP systems is 1.4 to 2.1 times faster per step
  than a day earlier, and 9 to 17 times (float64) or 30 to 47 times
  (compressed float32) faster than the LAMMPS record. Large systems gain 1.2
  to 1.7 times.
- Agreement is unchanged. All 83 HAP test frames, four models, four modes
  against DeePMD (`hap_check.py`): float64 within 1.8e-13 eV per atom and
  5.6e-13 eV/Å, float32 within 2.4e-7 and 7.5e-6, compressed float64 within
  8.4e-10 and 3.9e-6. Water against DeePMD float64: energy identical, forces
  within 1.8e-17 eV/Å (8.5e-16 compressed). The four-model deviation matches
  the one from DeePMD's forces as before.
- What changed, in the order of what it gave for 88 atoms (compressed
  float32, 0.53 ms per evaluation before):
  - One upload, one download and kept buffers per evaluation instead of
    about thirty device calls: 0.38 ms. The neighbor search alone had made
    fifteen.
  - The neighbor search ranks candidates without running its tie-breaking
    arithmetic for every pair (the compiler had made the branch
    unconditional), scans in single precision before measuring candidates in
    double, and reads positions in cell order: 0.33 ms.
  - No divisions or square roots in double precision per neighbor slot, and
    no work for empty slots (65% of the slots in these systems): 0.265 ms.
- Where the time goes now, 88 atoms, compressed float32 (extra time of
  running one stage twice): fitting nets 0.11 ms, forces 0.03, neighbor
  search 0.03, descriptor 0.025, embedding tables 0.015, environment 0.01.
  The fitting kernel is bound by loading its weights from global memory
  (about 20 ns per weight line and warp); see the section above for what
  was done about it.
  At 98,304 atoms (57 ms): forces 16.3 ms, descriptor 12.6, embedding tables
  10.0, fitting 8.6, neighbor search 5.6.
- The trainer shares the reworked environment, descriptor and force kernels:
  2 to 4% faster in float32, unchanged in float64 (numbers in the section
  above). Its time is in the fitting passes.

# Training on the HAP iteration-0 data, 2026-11-10

`omnimd-native-gpu-train` against the HAP pipeline's four iteration-0 DeePMD
models, on the frames those were trained on (171 frames in 8 systems: the
pipeline's iteration-0 training directory without the 25 lime frames that
were labelled after that training had started), with the same settings
(se_e2_a, `sel` [44, 16, 60, 72, 20], embedding [25, 50, 100], fitting
[240, 240, 240], 2,664,455 parameters, learning rate 1e-3 to 3e-8, 300,000
steps of one frame) and the same loss (energy and force terms; the virial
prefactors 0 to 0.02 leave that term off in DeePMD-kit's PyTorch backend and
in omniMD). Four omniMD models with seeds 11 to 14, trained at once with
float32 networks over float64 parameters; DeePMD trained in float64. Scored
on the pipeline's 83-frame sealed test with `omnimd-native-gpu-test`.

| | omniMD, four models | DeePMD-kit 3.1.3, four models |
|---|---:|---:|
| time for the four | 30 min at once | 30.8 h one after the other |
| sealed test, force RMSE (eV/Å) | 0.0698 to 0.0723 | 0.0699 to 0.0711 |
| sealed test, energy RMSE (meV/atom) | 1.64 to 1.78 | 1.64 to 1.74 |

- Same quality: the ranges of the two sets of four seeds overlap in every
  system, lime included (per-system table in
  [`native/GPU_TRAINING.md`](../../native/GPU_TRAINING.md)). omniMD's seeds
  spread a little wider in the overall force RMSE (3.6 against 1.8 percent).
- About sixty times less time for the ensemble. DeePMD's is the pipeline's
  record on this machine (file times of the four runs; 12.3 steps/s for one
  model alone, 6.5 each for two at once), not re-run here.
- Same exploration. The pipeline's sixteen iteration-0 exploration runs
  (20,000 steps each, a frame every 50) were repeated with omniMD's models
  and, as a control, with DeePMD's own models converted, both through
  `omnimd-mlmd` (`hap_explore.py`). In the eleven runs where the pipeline's
  LAMMPS record has most frames below the 0.1 eV/Å edge of the window, the
  fraction below it differs from the record by at most 0.09 with omniMD's
  models (0.03 without monetite at 300 K, where the edge lies in the middle
  of the distribution) and by at most 0.01 with DeePMD's models; the mean
  deviation differs by -6 to +25 percent and by -14 to +9 percent. In the
  other five runs (the HAP(001) interfaces, monetite at 500 K) the models
  fail in all three. The sixteen runs with their four-model deviations took
  5.1 minutes, 0.58 to 1.21 ms per MD step.
- The same instrument: `omnimd-native-gpu-test` gives the pipeline's recorded
  sealed-test errors of the converted DeePMD models to every printed digit
  (`hap-test-deepmd-iter0-by-omnimd-rtx4090-2026-11-10.json`).
- Which frames DeePMD saw is reconstructed from the pipeline's notes, the
  times of the label files (five lime frames predate the training) and the
  missing lime validation system; that the other systems' frames are
  unchanged since then is an assumption.

Raw numbers: `hap-train-ensemble-float32-omnimd-rtx4090-2026-11-10.json`
(sealed test), `hap-explore-iter0-omnimd-float32-rtx4090-2026-11-10.json` and
`hap-explore-iter0-deepmd-models-by-omnimd-rtx4090-2026-11-10.json`
(exploration).

**Runs on other frames or with another loss, kept as records.** They are
not comparisons with the pipeline's models:

| files (`hap-train-ensemble-float32-…`, `hap-explore-iter0-omnimd-float32-…`) | frames | loss | parameters | force RMSE | energy RMSE | lime force RMSE |
|---|---|---|---|---:|---:|---:|
| `with-lime-virial` | 196, all 30 lime | virial ramped in | float32 | 0.0711 to 0.0742 | 1.47 to 1.58 | 0.0125 to 0.0140 |
| `with-lime` | 196, all 30 lime | energy, force | float32 | 0.0712 to 0.0735 | 1.76 to 1.84 | 0.0129 to 0.0144 |
| `without-lime` | 166, no lime | energy, force | float64 | 0.0771 to 0.0938 | 242 to 763 | 0.1256 to 0.2307 |
| (the comparison above) | 171, 5 lime | energy, force | float64 | 0.0698 to 0.0723 | 1.64 to 1.78 | 0.0183 to 0.0252 |

- The pipeline's models score 0.020 to 0.021 on lime: between the runs with
  thirty lime frames and with none, which is how the difference in the
  training sets was found. Without lime frames the other systems are scored
  like the pipeline's; only lime fails, and with it the overall numbers.
- With float32 parameters the energy error is higher than with float64
  parameters under the same float32 networks: steps at the end of the
  schedule are below the spacing of float32 numbers. One member, last third
  repeated: 1.84 meV/atom with float32 parameters, 1.61 with float64
  networks, 1.61 with float32 networks over float64 parameters.
- `hap-train-omnimd-rtx4090-2026-11-10.json` is one float64 model of the
  first version of the trainer on the 196 frames with the virial ramped in,
  in 88 minutes (force RMSE 0.0719, energy 1.27 meV/atom).

**Trainer speed** on this model and the 196 frames (steps per second; 20,000
steps in float32, 6,000 in float64, nothing else on the card):

| requests at once | float32, each | float32, total | float64, each | float64, total |
|---:|---:|---:|---:|---:|
| 1, first version | 118 | 118 | 58 | 58 |
| 1 | 604 | 604 | 289 | 289 |
| 2 | 324 | 648 | 160 | 321 |
| 4 | 179 | 715 | 88 | 352 |

- From the first version to this one: networks evaluated as sets (284 and 92
  for one request), parameter gradients in one kernel (402 and 191), no
  embedding rows for empty neighbor slots, of which these frames have about
  two thirds (567 and 297), neighbor lists kept per frame (661 and 294),
  float64 parameters under float32 networks and float32 dot products in
  partial sums (604 and 289).
- A float32 step takes 1.87 ms with every stage timed, of which the two
  fitting passes take 0.85; requests trained at once gain 7 to 22 percent in
  total. Separate trainer processes did not add up with the first version
  (two at once 24.8 steps/s each, four 12.3 each, against 57.9 for one).
- The training set can be produced by omniMD alone: `import-deepmd` for each
  system with `id_prefix` and the common `type_map`, then `merge-datasets`,
  give the 196 frames of the directory in the order of the preparation
  script, with no numeric difference.

# Constant pressure against the pipeline's LAMMPS run, 2026-11-10

HAP bulk (88 atoms), 300 K, 1.01325 bar, fully flexible cell, 0.5 fs, 100,000
steps, sampled every 50 steps, driven by the iteration-4 model 0. LAMMPS: the
pipeline's own exploration run (`fix npt temp 300 300 0.05 tri 1.01325 1.01325
0.5`, Nose-Hoover chains; model 0 drives and all four TensorFlow models are
evaluated every 50 steps), read from its `log.lammps` and `traj.dump`.
omniMD: `npt_csvr_crescale` with the same temperature, pressure and time
constants (CSVR thermostat 50 fs,
stochastic cell rescaling 500 fs, assumed compressibility 1e-6 per bar,
`triclinic` coupling, a cell move every 10 steps), started from the same
`conf.lmp` (`hap_npt.py`, `hap-npt-omnimd-rtx4090-2026-11-10-*.json`). The
algorithms differ, so the trajectories do; means over the last 40 ps with
standard errors from eight block averages:

| quantity | LAMMPS (recorded) | omniMD float64 | omniMD float64-compressed | omniMD float32 | omniMD float32-compressed |
|---|---:|---:|---:|---:|---:|
| volume (Å³) | 1080.18 ± 0.13 | 1080.82 ± 1.07 | 1080.57 ± 1.11 | 1080.66 ± 1.11 | 1080.70 ± 1.00 |
| potential energy (eV) | -45045.160 ± 0.007 | -45045.174 ± 0.028 | -45045.167 ± 0.025 | -45045.172 ± 0.023 | -45045.171 ± 0.026 |
| temperature (K) | 302.0 ± 0.9 | 300.1 ± 1.5 | 300.0 ± 1.7 | 300.2 ± 1.8 | 300.4 ± 1.5 |
| a (Å) | 9.469 ± 0.000 | 9.485 ± 0.012 | 9.484 ± 0.012 | 9.484 ± 0.012 | 9.485 ± 0.012 |
| b (Å) | 9.471 ± 0.001 | 9.467 ± 0.009 | 9.466 ± 0.009 | 9.467 ± 0.009 | 9.467 ± 0.009 |
| c (Å) | 13.909 ± 0.001 | 13.922 ± 0.019 | 13.921 ± 0.019 | 13.922 ± 0.019 | 13.922 ± 0.019 |
| alpha (°) | 89.97 ± 0.02 | 90.02 ± 0.16 | 90.04 ± 0.15 | 90.05 ± 0.15 | 90.03 ± 0.15 |
| beta (°) | 90.00 ± 0.02 | 89.87 ± 0.11 | 89.86 ± 0.11 | 89.83 ± 0.12 | 89.85 ± 0.11 |
| gamma (°) | 119.99 ± 0.00 | 120.14 ± 0.12 | 120.15 ± 0.11 | 120.15 ± 0.12 | 120.16 ± 0.12 |
| volume std (Å³) | 7.00 | 7.10 | 7.16 | 7.17 | 7.09 |
| pressure (bar) | -1 ± 12 | -607 ± 753 | -585 ± 753 | -612 ± 772 | -591 ± 762 |
| refused cell moves | | 10 of 10000 | 9 of 10000 | 8 of 10000 | 8 of 10000 |
| ms per step | 45.9 | 1.68 | 0.90 | 0.60 | 0.56 |

- Every mean of every mode agrees with the LAMMPS run within 1.4 combined
  standard errors, and the volume fluctuation is the same (7.0 to 7.2 against
  7.0 Å³). The omniMD errors are larger because a first-order barostat lets
  the volume diffuse where Nose-Hoover makes it oscillate: over a time t the
  mean pressure of such a barostat is uncertain by about
  sqrt(2 kT tau / (compressibility V t)), here 1000 bar for 40 ps. The mean
  pressure is therefore a weak check on this time scale and the volume the
  meaningful one.
- About one cell move in a thousand is refused by the acceptance test.
- Each cell move costs one extra evaluation, so NPT with a move every 10
  steps costs 10 to 15 percent more per step than NVT (float64 1.68 against
  1.50 ms; float32 compressed 0.56 against 0.49 ms).
- The recorded LAMMPS time, 45.9 ms per step, comes from the pipeline's
  exploration batch, in which several runs shared the card, so it is not a
  like-for-like speed comparison; alone, LAMMPS takes 13.6 ms per NVT step for
  this system (table below).
- Two runs of the same request do not repeat bit for bit on the GPU runtime
  (the order of atomic force additions varies): a second float64 run gave
  1080.76 ± 1.02 Å³. The four modes above share the seed of thermostat and
  barostat, so their volumes move together; three further seeds of 50 ps
  (float32 compressed) gave 1081.65 ± 0.82, 1080.79 ± 1.43 and 1080.85 ±
  0.87 Å³, and a 400 ps run with a fifth seed 1080.08 ± 0.28 Å³ (LAMMPS:
  1080.18 ± 0.13) with a mean pressure of -17 ± 197 bar, potential energy
  -45045.167 ± 0.004 eV, a = 9.462 ± 0.006, b = 9.474 ± 0.007, c = 13.919 ±
  0.008 Å and gamma = 120.02 ± 0.04 degrees
  (`hap-npt-omnimd-rtx4090-2026-11-10-float32-compressed-400ps.json`).
- The effective energy (total minus thermostat heat minus barostat work)
  drifts by at most 0.009 eV over the 50 ps in every mode.
- The exploration chain on the same system (`hap_npt.py --deviation`,
  `hap-explore-omnimd-rtx4090-2026-11-10.json`): the 2,001 sampled frames of
  a float64 run went through `import-trajectory` and `model-deviation` with
  the four models on the GPU runtime. The mean over frames of the largest
  force deviation is 0.0338 ± 0.0002 eV/Å (the LAMMPS run's `model_devi.out`:
  0.0333 ± 0.0002) and of the mean force deviation 0.01603 ± 0.00004
  (0.01595 ± 0.00005). With the pipeline's window of 0.1 to 0.3 eV/Å both
  count all 2,001 frames as accurate (largest value 0.078 against 0.066).
  Import and deviation of the 2,001 frames took 14.5 s.

# HAP pipeline ensemble: agreement and exploration-style speed, 2026-11-10

Models and frames from the HAP CO2-mineralization pipeline v2 on this machine
(iteration 4: four se_e2_a models, type map Ca P O H C, sel [44, 16, 60, 72, 20],
embedding [25, 50, 100], fitting [240, 240, 240], float64; sealed test of 83
DFT-labelled frames in 8 systems, 54 to 224 atoms). The pipeline marks these
models as not production-qualified; the numbers below are agreement and speed
only. Prepared with `hap_prepare.py`, which reads the pipeline tree and writes
only to `cases/hap/`.

**Agreement with DeePMD-kit 3.1.3** (`hap_check.py`; DeepPot on CPU, all 83
frames, largest difference over the four models):

| omniMD mode | energy per atom (eV) | force (eV/Å) | virial (eV) |
|---|---:|---:|---:|
| float64 | 1.8e-13 | 5.6e-13 | 1.1e-12 |
| float64, compressed | 8.4e-10 | 3.9e-6 | 2.0e-5 |
| float32 | 2.4e-7 | 7.5e-6 | 7.9e-5 |
| float32, compressed | 2.8e-7 | 8.7e-6 | 7.2e-5 |

- The sealed-test errors against the DFT labels come out the same from omniMD
  as from DeePMD (largest shift 6e-14 in float64, 2.4e-7 in float32
  compressed, in eV/atom or eV/Å), and both reproduce the pipeline's recorded
  `sealed_test.json` to its printed precision (3.6e-7 relative).
- No frame exceeds `sel`, and no frame leaves the table range at
  `--compress 0.5`.
- Compression costs more accuracy here than on the untrained water model
  (4e-6 against 9e-16 eV/Å) because trained embedding nets vary faster. It
  remains four orders of magnitude below the models' own test error
  (0.014 to 0.098 eV/Å per system and model).
- Model deviation (`hap_md.py`): `omnimd-mlmd model-deviation` with the four
  models on the GPU runtime gives max/min/mean force deviations within 1e-14
  eV/Å of those computed from DeePMD's reference forces. The whole run over
  83 frames takes about 2 s, including loading four models.

**Speed, ms per MD step** (NVT at 300 K, 0.5 fs). DeePMD: LAMMPS
`pair_style deepmd` with the pipeline's protocol and models (TensorFlow
graphs, skin 0 with a rebuild every step, 2 OpenMP threads;
`hap-lammps-rtx4090-2026-11-10.csv`). omniMD: `omnimd-mlmd run` on the GPU
runtime with the CSVR thermostat, sampled every 50 steps
(`hap-omnimd-rtx4090-2026-11-10.csv`):

| system (atoms) | DeePMD 1 model | DeePMD 4 models | omniMD fp64 | fp64 compressed | fp32 | fp32 compressed |
|---|---:|---:|---:|---:|---:|---:|
| HAP bulk (88) | 13.6 | 12.8 | 1.50 | 0.78 | 0.52 | 0.49 |
| HAP(001) + water + CO2 (169) | 15.7 | 16.5 | 2.57 | 1.19 | 0.80 | 0.69 |
| monetite (224) | 14.6 | 15.5 | 3.10 | 1.26 | 0.86 | 0.75 |

- omniMD float64 gives DeePMD's forces to rounding and is 4.7 to 9 times
  faster per step; compressed float64 is 11 to 17 times faster and compressed
  float32 19 to 28 times.
- "4 models" is the pipeline's exploration setting: model 0 drives the
  dynamics and all four are evaluated every 50 steps. omniMD runs the
  deviation as a separate command over the sampled frames (about 25 ms per
  frame including model loading, so under 1 ms per MD step at this sampling).
- The pipeline's own 100,000-step exploration logs show 52 ms per step for the
  169-atom system (three jobs sharing the GPU, 87 min per run).
- The thermostats differ (Nose-Hoover in LAMMPS, CSVR in omniMD); this does
  not affect the cost per step. The pipeline's 300 K bulk runs are NPT; see
  the constant-pressure comparison above.
- These systems are bound by per-step latency, not arithmetic. What made
  them fast here: one launch for all 25 embedding nets and one fused
  forward-and-backward kernel for all five fitting nets (169 atoms, float32
  compressed: 1.41 to 0.67 ms per evaluation).

# Water benchmark rerun with the single-launch kernels, 2026-11-10 (r2)

`omnimd-gpu-rtx4090-2026-11-10-r2.csv`, same protocol as the table below.

| atoms | fp64 | fp64 compressed | fp32 | fp32 compressed |
|---:|---:|---:|---:|---:|
| 192 | 2.07 | 1.05 | 0.70 | 0.63 |
| 1,536 | 12.4 | 5.88 | 2.38 | 1.81 |
| 12,288 | 95.7 | 44.2 | 16.0 | 11.9 |
| 98,304 | 763 | 352 | 128 | 94.8 |

The machine ran about 10% slower during this run than during the morning run
below: the morning build, rebuilt and alternated with the new one, also takes
11.6 ms at 12,288 atoms in float32 compressed (10.4 in the morning). At equal
conditions the new build is faster at 192 atoms (float64 2.77 to 2.07 ms,
compressed 1.65 to 1.05) and within 1.5% elsewhere. Agreement with the DeePMD
reference is unchanged.

# Fused embedding, compression and DeePMD compressed baseline, 2026-11-10

Same card, model and frames as below; ms per energy/force/virial evaluation.
DeePMD rows are LAMMPS `pair_style deepmd` (uncompressed from 2026-11-09,
compressed from `deepmd-3.1.3-lammps-compressed-2026-11-10.csv`, models from
`compress_deepmd.py`). omniMD rows are `omnimd-gpu-rtx4090-2026-11-10.csv`:
10 to 100 in-process evaluations each; "compressed" is `--compress 0.5`
(piecewise quintic embedding tables, step 0.01).

| atoms | precision | DeePMD | omniMD | speed-up | DeePMD compressed | omniMD compressed | speed-up |
|---:|---|---:|---:|---:|---:|---:|---:|
| 192 | fp32 | 10.4 | 0.70 | 14.9× | 7.58 | 0.69 | 11.0× |
| 1,536 | fp32 | 13.1 | 2.18 | 6.0× | 8.94 | 1.64 | 5.5× |
| 12,288 | fp32 | 66.1 | 14.2 | 4.7× | 32.5 | 10.4 | 3.1× |
| 98,304 | fp32 | error | 118 | — | error | 84.9 | — |
| 192 | fp64 | 11.4 | 2.77 | 4.1× | 8.57 | 1.65 | 5.2× |
| 1,536 | fp64 | 41.9 | 11.8 | 3.5× | 20.2 | 6.12 | 3.3× |
| 12,288 | fp64 | 293 | 90.4 | 3.2× | 118 | 40.0 | 2.9× |
| 98,304 | fp64 | error | 699 | — | error | 329 | — |

- Changes since 2026-11-09:
  - The embedding net runs as one fused kernel (12,288 atoms fp32: embedding
    stage 11.0 → 5.3 ms).
  - Optional compression replaces it by tables (5.3 → 1.5 ms; fp64 53 → 7 ms).
  - Neighbor selection follows DeePMD-kit 3.
- Accuracy against DeepMD-kit CPU float64 `DeepPot` (192 and 1,536 atoms):
  omniMD fp64 energy is identical with and without compression; the largest
  force difference is 2e-17 eV/Å exact and 9e-16 eV/Å compressed. fp32 forces
  are within 6e-9 eV/Å in both modes. DeePMD's own compressed accuracy was not
  measured.
- Remaining cost at 12,288 atoms, fp32 compressed: force/virial kernel 4.0 ms,
  descriptor 1.9, embedding tables 1.5, neighbor list 1.3, fitting 0.9.
- LAMMPS rebuilds its neighbor list every ~13 steps; omniMD rebuilds on every
  evaluation. At 98,304 atoms LAMMPS stops with a TorchScript runtime error in
  every mode (probably out of memory; not verified).

# GPU se_e2_a and LAMMPS baseline, 2026-11-09 (same card, model and frames)

ms per energy/force/virial evaluation. DeePMD LAMMPS rows are `pair_style deepmd`
NVE loop time per step (`deepmd-3.1.3-lammps-2026-11-09.csv`); omniMD rows are the
in-process GPU evaluator, `omnimd-native-gpu-eval --precision P --bench K`
(`omnimd-gpu-rtx4090-2026-11-09.csv`, stage times and errors in the `.json`).
omniMD float32 runs the networks in float32; geometry, the B accumulation,
forces and virial stay float64.

| atoms | DeePMD LAMMPS fp64 | omniMD GPU fp64 | speed-up | DeePMD LAMMPS fp32 | omniMD GPU fp32 | speed-up |
|---:|---:|---:|---:|---:|---:|---:|
| 192 | 11.4 | 3.52 | 3.2× | 10.4 | 1.01 | 10.3× |
| 1,536 | 41.9 | 15.7 | 2.7× | 13.1 | 2.86 | 4.6× |
| 12,288 | 293 | 117 | 2.5× | 66.1 | 20.4 | 3.2× |
| 98,304 | error | 929 | — | error | 175 | — |

- Accuracy against DeepMD-kit CPU float64 `DeepPot` (192 and 1,536 atoms): omniMD
  float64 energy is identical and the largest force difference is 2e-17 eV/Å. In
  float32 the force error is at most 5.4e-9 eV/Å. At 12,288 and 98,304 atoms,
  float32 is within 8.8e-9 eV/Å of omniMD float64.
- LAMMPS rebuilds its neighbor list only every ~13 steps (2 Å skin), while omniMD
  rebuilds on every evaluation. The comparison therefore slightly favors DeePMD.
- At 98,304 atoms LAMMPS stopped with a TorchScript runtime error in both
  precisions. This is probably out of memory, but that has not been verified.
- Milestone M2 (12k atoms fp32 ≥ 1.0× DeePMD) is met: 3.2×.

# W1 baseline, 2026-10-29 (RTX 4090 24 GB, Windows 11 host)

Water `se_e2_a` (sel [46, 92], embedding [25, 50, 100], fitting [240, 240, 240]),
same parameters in both engines. Raw rows: `deepmd-3.1.3-rtx4090-2026-10-29.csv`,
`omnimd-rtx4090-2026-10-29.csv`. Times are ms per energy/force/virial evaluation.

| atoms | DeePMD GPU fp64 | DeePMD GPU fp32 | DeePMD CPU fp64 (8 thr) | DeePMD CPU fp32 (8 thr) | omniMD CPU fp64 (1 thr) | omniMD CUDA fp64 |
|---:|---:|---:|---:|---:|---:|---:|
| 192 | 39.9 | 36.7 | 128 | 79.5 | 299 | rejected |
| 1,536 | 68.8 | 39.0 | 1,111 | 691 | 4,599 | rejected |
| 12,288 | OOM | not run | n/a | n/a | n/a | rejected |
| 98,304 | OOM | not run | n/a | n/a | n/a | rejected |

DeePMD is the Python `DeepPot` route (PyTorch backend, DeepMD-kit 3.1.3,
`ghcr.io/deepmodeling/deepmd-kit:3.1.3_cuda129`). GPU peak allocator memory was
128 MB at 192 atoms and 1.97 GB at 1,536 atoms. CPU cases above 1,536 atoms were
not run by design.

## Findings

1. **omniMD CUDA cannot run a production-size model.** The CUDA subset requires
   embedding width ≤ 64 and descriptor width ≤ 128; this model has 100 and
   1,600. There is no GPU comparison point yet. Plan item I-3 (batched GEMM, no
   width limits) in W5–W7 has to close this gap before M2.
2. **omniMD CPU is 2.3× (192 atoms) to 4.1× (1,536 atoms) slower than DeePMD
   CPU** in float64. omniMD runs single-threaded against DeePMD's 8 threads, so the
   I-7 target (≤ DeePMD CPU at equal threads) looks reachable at 192 atoms. The
   ratio grows with size, which fits the brute-force neighbor search (I-1).
3. **DeePMD's Python route stops at 12,288 atoms.** `DeepPot` ran out of GPU memory
   even at batch size 1. The large-system baseline must come from LAMMPS
   `pair_style deepmd`, which uses its own neighbor list.

## Not measured yet

- **LAMMPS `pair_style deepmd`:** every case failed in this run because the image
  runs as root and Open MPI refuses that. The script also handed LAMMPS a relative
  log path. Both are fixed; see the 2026-11-09 table above.
- **DeePMD fp32 at 12,288 and 98,304 atoms:** the Docker Desktop VM went down
  during the fp32 12,288-atom DeepPot case and the run ended there.
- **Compressed DeePMD models** (`dp compress`), which the W1 row of the plan lists,
  were not built yet.
- **Rerun command once Docker is back:**
  `bench_deepmd.py cases results/deepmd-3.1.3-lammps-<tag>.csv --skip-deeppot`.

omniMD timings come from the CLI. They subtract the fastest one-frame run from
the fastest K-frame run (3 repeats) and include per-frame JSON parsing.
