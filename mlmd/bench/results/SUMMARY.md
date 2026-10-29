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
  runs as root and Open MPI refuses that. `bench_deepmd.py` now sets
  `OMPI_ALLOW_RUN_AS_ROOT`; this has not been rerun yet.
- **DeePMD fp32 at 12,288 and 98,304 atoms:** the Docker Desktop VM went down
  during the fp32 12,288-atom DeepPot case and the run ended there.
- **Compressed DeePMD models** (`dp compress`), which the W1 row of the plan lists,
  were not built yet.
- **Rerun command once Docker is back:**
  `bench_deepmd.py cases results/deepmd-3.1.3-lammps-<tag>.csv --skip-deeppot`.

omniMD timings come from the CLI. They subtract the fastest one-frame run from
the fastest K-frame run (3 repeats) and include per-frame JSON parsing.
