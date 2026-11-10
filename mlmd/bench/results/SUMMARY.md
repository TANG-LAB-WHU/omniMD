# Training on the HAP iteration-0 data, 2026-11-10

`omnimd-native-gpu-train` from fresh networks on the training set the HAP
pipeline's iteration-0 DeePMD models used (196 frames in 8 systems, 152 with
virial labels), with the pipeline's DeePMD settings (se_e2_a, `sel`
[44, 16, 60, 72, 20], embedding [25, 50, 100], fitting [240, 240, 240],
2,664,455 parameters, float64; learning rate 1e-3 to 3e-8, 300,000 steps of
one frame). Scored on the pipeline's 83-frame sealed test with
`omnimd-native-gpu-test`. Raw numbers:
`hap-train-omnimd-rtx4090-2026-11-10.json` and `.lcurve.out`.

| | omniMD, one run | DeePMD-kit 3.1.3, four models |
|---|---:|---:|
| 300,000 steps | 88 min (57 steps/s) | about 12.8 h each (6.5 steps/s) |
| sealed test, force RMSE (eV/Å) | 0.0719 | 0.0699 to 0.0711 |
| sealed test, energy RMSE (meV/atom) | 1.27 | 1.64 to 1.74 |

- Same quality: the force error is 1 to 3 percent above the DeePMD models',
  the energy error about a quarter below; the DeePMD seeds spread by 2 and 6
  percent among themselves. Per system the force errors are inside the DeePMD
  range or up to 7 percent above it, except lime, where omniMD's is a quarter
  lower.
- About 8.8 times faster per step. The DeePMD errors and rate are the
  pipeline's own records on this machine, not re-run here; the per-system
  table is in [`native/GPU_TRAINING.md`](../../native/GPU_TRAINING.md).
- The omniMD run shared the GPU for about 8 of its 88 minutes. Alone the
  trainer measured 57 steps/s in float64 and 118 in float32.
- The step time is set by about a thousand small device launches, not by
  arithmetic, so it barely depends on the frame size here.
- Several trainer processes on one card do not add up: two at once run at
  24.8 steps/s each and four at 12.3 each (49 in total, against 57.9 for one
  alone; 3,000 steps of this model in float64). An ensemble is therefore
  trained one model after the other for now.

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
