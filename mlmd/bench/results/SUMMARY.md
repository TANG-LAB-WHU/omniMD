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
