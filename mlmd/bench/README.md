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
# 5. DeePMD reference E/F/V (CPU float64; DeepPot needs too much memory past 1,536 atoms)
docker run --rm -v $PWD/mlmd/bench:/work -w /work ghcr.io/deepmodeling/deepmd-kit:3.1.3_cuda129 python deepmd_reference.py cases water-se_e2_a-float64.pth 192 1536 > mlmd/bench/cases/deepmd-ref-float64.jsonl
# 6. omniMD GPU se_e2_a (host; cargo build --release -p omnimd-native --features gpu --bins)
python mlmd/bench/bench_omnimd_gpu.py mlmd/bench/cases <target>/release mlmd/bench/results/omnimd-gpu-<tag>.csv --reference mlmd/bench/cases/deepmd-ref-float64.jsonl
```

Step 6 also writes `omnimd-gpu-<tag>.json` with per-stage times and the energy,
force and virial errors against the DeePMD reference and against omniMD float64.
It times the exact embedding networks and the compressed mode (`--compress 0.5`).

For a compressed DeePMD baseline, run `compress_deepmd.py cases` in the image
(the `dp --pt compress` steps with the minimum neighbor distance set to 0.5 Å,
because the untrained models do not store one), then step 3 with
`--skip-deeppot --compressed`.

## A real ensemble: the HAP pipeline models

`hap_*.py` repeat the comparison on a trained DeePMD se_e2_a ensemble and its
DeePMD npy test systems (written for the HAP pipeline tree, which is only read):

```bash
# DeePMD image: convert the models, write frames, DeePMD reference E/F/V
docker run --rm -v $PWD/mlmd:/m -v <pipeline>:/hap:ro -w /m <image> python bench/hap_prepare.py /hap/iter-004/train /hap/sealed-test/data/train bench/cases/hap
# Host: agreement of every model and mode with DeePMD, and with recorded test errors
python mlmd/bench/hap_check.py mlmd/bench/cases/hap <target>/release report.json --recorded <pipeline>/iter-004/train/sealed_test.json
# Host: 4-model deviation check and NVT timing through omnimd-mlmd (--features native-gpu)
python mlmd/bench/hap_md.py mlmd/bench/cases/hap <target>/release mlmd/bench/results/hap-omnimd-<tag>.csv <system> ...
# DeePMD image: LAMMPS with the pipeline's exploration protocol
docker run --rm --gpus all -v $PWD/mlmd:/m -v <pipeline>/iter-004/explore:<path the models were written for>:ro -w /m <image> python bench/hap_lammps.py <that path> bench/cases/hap bench/results/hap-lammps-<tag>.csv <run> ...
# Host: constant pressure through omnimd-mlmd against a finished LAMMPS fix npt run of the pipeline
python mlmd/bench/hap_npt.py <pipeline>/iter-004/explore/<run>_npt mlmd/bench/cases/hap <target>/release mlmd/bench/results/hap-npt-omnimd-<tag>.json --mode float64
# DeePMD image, then host: write the training systems as a dataset (no models), train from scratch, score on the test frames
docker run --rm -v $PWD/mlmd:/m -v <pipeline>:/hap:ro -w /m <image> python bench/hap_prepare.py /hap/iter-000/train /hap/<training systems> bench/cases/hap-train --models
omnimd-native-gpu-train request.json
omnimd-native-gpu-test trained.native.json mlmd/bench/cases/hap/dataset.jsonl > report.json
# Host: score an ensemble trained that way against the pipeline's recorded test errors
python mlmd/bench/hap_train_score.py mlmd/bench/cases/hap/dataset.jsonl <pipeline>/iter-000/train/sealed_test.json <target>/release mlmd/bench/results/hap-train-ensemble-<tag>.json model-0.native.json model-1.native.json ...
# Host: repeat finished exploration runs of the pipeline with an ensemble of native models (manifest.json and model-K.native.json in a case directory)
python mlmd/bench/hap_explore.py <pipeline>/iter-000/explore <case directory> <target>/release mlmd/bench/results/hap-explore-<tag>.json <run> ...
```

`hap_npt.py` reads the LAMMPS run's `conf.lmp`, `in.lmp`, `log.lammps` and
`traj.dump` and starts no LAMMPS itself; `hap_explore.py` reads `conf.lmp`,
`in.lmp` and `model_devi.out` of every run and compares the fractions of
frames below, inside and above the force-deviation window. The training request is the one in
[`native/GPU_TRAINING.md`](../native/GPU_TRAINING.md).

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
