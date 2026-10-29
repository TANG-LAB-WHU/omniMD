# Throughput benchmark: omniMD vs DeePMD-kit

Same model, same frames, same GPU. This is the W1 baseline for the omniMD plan
(`D:\project\omniMD_PLAN_2026-10-29.md`, milestone M2 compares against it).

## Case

- Model: official water `se_e2_a` example architecture (rcut 6.0 Å, sel [46, 92],
  embedding [25, 50, 100], axis_neuron 16, fitting [240, 240, 240]) with untrained,
  seeded weights. Throughput does not depend on weight values; every engine loads
  the same parameters (`.dp` → `.pth` for DeePMD, `.dp` → native JSON for omniMD).
  Built in float64 and float32.
- Frames: 64-molecule water boxes at 0.997 g/cm³, random orientations, replicated
  to 192, 1,536, 12,288 and 98,304 atoms.

## Run

```bash
# 1. Model and frames (DeepMD-kit 3.1.3 image)
docker run --rm -v $PWD/mlmd/bench:/work -w /work ghcr.io/deepmodeling/deepmd-kit:3.1.3_cuda129 python make_water_case.py cases
# 2. Native model for omniMD
docker run --rm -v $PWD/mlmd:/m -w /m ghcr.io/deepmodeling/deepmd-kit:3.1.3_cuda129 python native/tools/convert_deepmd.py bench/cases/water-se_e2_a-float64.dp bench/cases/water-se_e2_a-float64.native.json --units EvAngstrom --geometry dpmodel
# 3. DeepMD timings: Python DeepPot (GPU and 8-thread CPU) and LAMMPS pair_style deepmd (GPU)
docker run --rm --gpus all -v $PWD/mlmd/bench:/work -w /work ghcr.io/deepmodeling/deepmd-kit:3.1.3_cuda129 python bench_deepmd.py cases results/deepmd-<tag>.csv
# 4. omniMD timings (host; release build with --features cuda)
python mlmd/bench/bench_omnimd.py mlmd/bench/cases <target>/release mlmd/bench/results/omnimd-<tag>.csv
```

Run steps 3 and 4 one after the other, never at the same time, so they don't
compete for the GPU. `cases/` is generated and git-ignored. Step 3 accepts
`--skip-deeppot` or `--skip-lammps` to rerun one route into a separate CSV.

The Python DeepPot route runs out of GPU memory from 12,288 atoms on a 24 GB card,
and on the 2026-10-29 Windows run the Docker Desktop VM went down during the
float32 attempt at that size. Use the LAMMPS route for the large cases.

## Metrics

`ms_per_eval` is one energy/force/virial evaluation; `us_per_atom_step` divides it
by the atom count. LAMMPS rows also report `ns_per_day` from NVE at 0.5 fs.
`peak_mem_mb` is the PyTorch allocator peak (DeepPot GPU rows only). Failed cases
are kept as rows with `status` = `oom`, `rejected`, `timeout` or `error`.

The omniMD CLI timing subtracts the fastest one-frame run from the fastest K-frame
run (three repeats each), which removes
start-up and model loading but still includes per-frame JSON parsing. That favors
DeepMD slightly at large sizes; the in-process Rust harness planned for W5 removes it.

## Results

See `results/` and the summary table in `results/SUMMARY.md`.
