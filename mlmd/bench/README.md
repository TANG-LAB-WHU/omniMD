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

The same six steps time the radial descriptor: `make_water_case.py cases/se-r
--descriptor se_e2_r` builds an se_e2_r model of the same sizes and the same
boxes, and every later step takes `cases/se-r` and `water-se_e2_r-float64`
where it has `cases` and `water-se_e2_a-float64` (step 4 does not apply: the
CUDA subset has se_e2_a only). `--descriptor se_e3` with `cases/se-e3` and
`water-se_e3-float64` does the same for the three-body descriptor, with the
embedding sizes DeePMD-kit's input has by default for it ([10, 20, 40]);
DeePMD's reference for it is taken at 192 atoms only, and DeePMD-kit 3.1.3's
compressed model of it was not usable as a baseline (it does not reproduce
the uncompressed model: see the summary), so DeePMD-kit is timed with its
nets only there. omniMD has tables for se_e3 since 2026-11-18.

`--descriptor hybrid` with `cases/hybrid` and `water-hybrid-float64` times a
hybrid of two descriptors, angular near the atom and radial further out: an
se_e2_a descriptor with a cutoff of 4.0 Å and `sel` [16, 32] and an se_e2_r
descriptor with the cutoff and `sel` of the other models, both with the
embedding sizes [25, 50, 100]. `--descriptor hybrid_e3` with
`cases/hybrid_e3` and `water-hybrid_e3-float64` times the se_e2_a descriptor
of the first model together with an se_e3 descriptor of cutoff 4.0 Å, `sel`
[16, 32] and embedding sizes [10, 20, 40] (DeePMD-kit with its nets only,
as for se_e3). The
sizes of these two models are this recipe's own choice, not an example of
DeePMD-kit's; the script stops if a descriptor has fewer slots than an atom
of the box has neighbors within its cutoff.

For a compressed DeePMD baseline, run `compress_deepmd.py cases` in the image
(the `dp --pt compress` steps with the minimum neighbor distance set to 0.5 Å,
because the untrained models do not store one), then step 3 with
`--skip-deeppot --compressed`. Before it is timed, compare the compressed
model with the uncompressed one on a frame (`deepmd.infer.DeepPot` on both):
DeePMD-kit 3.1.3's tables for se_e2_a and se_e2_r reproduce the nets, alone
and in a hybrid, its tables for se_e3 do not (see the summary). omniMD's
own tables for se_e3 are held against its exact nets, which equal
DeePMD-kit's uncompressed model.

The models are untrained and have no repulsion, so the LAMMPS run of step 3
is dynamics in which the atoms fall into each other sooner or later; its log
in `cases/.../lammps` shows the temperature. `bench_deepmd.py --held` takes a
time step of 1e-6 fs, so that every step evaluates the frame of the case
(LAMMPS then never rebuilds its neighbor list), `--max-atoms N` leaves the
larger cases out, and a run without a result after `--timeout` seconds (600)
is stopped and recorded: a run that needs more memory than the card has is
not refused on Windows, it takes minutes a step. The hybrid tables of the
summary use the held times; the tables of the other descriptors have the
NVE times, which a held run of 2026-11-16 confirmed. The top-level README
has held times since 2026-11-22, from the driver below.

## Both engines in one session

`bench_alternated.py` times omniMD and DeePMD-kit in one session,
alternated; the tables of the top-level README come from it:

```bash
# Host: the cases of steps 1, 2 and 5 in cases, cases/se-e3, cases/hybrid and cases/hybrid_e3, the compressed
# DeePMD-kit models, the image, and a release build of omnimd-native-gpu-eval; it times what steps 3 and 6 time, in turns
python mlmd/bench/bench_alternated.py <target>/release mlmd/bench/results/<name>.json --busy-check <command> --work <raw outputs>
python mlmd/bench/alternated_report.py mlmd/bench/results/<name>.json
```

A round has three slots, omniMD, DeePMD-kit and omniMD again, in reverse
order every other round; omniMD's repeat is the control, what the machine
adds between two runs of the same thing. An omniMD sample is one process of
`omnimd-native-gpu-eval --bench K`, K chosen so that the K evaluations take
about two seconds; a DeePMD-kit sample is one held LAMMPS run in the image
(`bench_deepmd.py --skip-deeppot --held`, which `--precisions`,
`--min-atoms` and `--max-atoms` narrow to the cases a slot needs). Of each
case the least sample is kept and the median recorded beside it. Before
every slot the driver waits until no container runs and the command of
`--busy-check` prints nothing (it should list the other users of the card;
on Windows a container's GPU work does not show as a process of its own),
checks the card again after the slot, and keeps the slot's clocks, memory,
temperature and load from `nvidia-smi`. It copies every LAMMPS log and
records what it says (steps, neighbor-list builds, cutoff, temperature, out
of memory), runs the evaluator from a copy whose hash it records, runs the
image by its ID (it never pulls), checks, untimed, the agreement of every
model with DeePMD-kit's CPU float64 result, and by default ends with
DeePMD-kit's single runs (`once`: 98,304 atoms, se_e3 in float64 at
1,536). Start it through an affinity
wrapper to hold omniMD's host side to chosen cores; Docker's virtual
machine is not held by that. `alternated_report.py` prints the README's
se_e2_a table, the controls and spreads of every case, the single
DeePMD-kit runs and the agreement.

## A real ensemble: the HAP pipeline models

`hap_*.py` repeat the comparison on a trained DeePMD se_e2_a ensemble and its
DeePMD npy test systems (written for the HAP pipeline tree, which is only read):

```bash
# DeePMD image: convert the models, write frames, DeePMD reference E/F/V
docker run --rm -v $PWD/mlmd:/m -v <pipeline>:/hap:ro -w /m <image> python bench/hap_prepare.py /hap/iter-004/train /hap/sealed-test/data/train bench/cases/hap
# Host: agreement of every model and mode with DeePMD, and with recorded test errors
python mlmd/bench/hap_check.py mlmd/bench/cases/hap <target>/release report.json --recorded <pipeline>/iter-004/train/sealed_test.json
# Host: batches of frames and all models as one committee; timings, and the committee against DeePMD
python mlmd/bench/hap_batch.py mlmd/bench/cases/hap <target>/release mlmd/bench/results/hap-batch-<tag>.csv --check report.json
# Host: model deviation through omnimd-mlmd timed per frame, optionally against another build
python mlmd/bench/hap_deviation.py mlmd/bench/cases/hap <target>/release mlmd/bench/results/hap-deviation-<tag>.json [--other <other target>/release]
# Host: replicas of a system advanced together (run-replicas) against the plain run
python mlmd/bench/hap_replicas.py mlmd/bench/cases/hap <target>/release mlmd/bench/results/hap-replicas-<tag>.csv <system> ... [--other <other target>/release]
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
# DeePMD image: train DeePMD-kit on the same request and frames (the dataset written back as npy systems), convert its model for the same test command
docker run --rm -v $PWD/mlmd:/m -w /m <image> python bench/dataset_to_deepmd.py bench/cases/hap-train/dataset.jsonl bench/cases/hap-train/systems
docker run --rm --gpus all -v $PWD/mlmd:/m -w /m <image> python bench/deepmd_train.py request.json bench/cases/hap-train/systems <new output directory>
# Host: groups of models scored side by side by the one test command (omniMD's seeds, DeePMD-kit's converted models, ...)
python mlmd/bench/score_groups.py mlmd/bench/cases/hap/dataset.jsonl <target>/release mlmd/bench/results/<name>.json omniMD=a.native.json,b.native.json DeePMD-kit=<output directory>/model.native.json
# DeePMD image, then host: replay that DeePMD-kit run in omniMD, frame for frame (the run with SAMPLER_RECORD=1, and its initial model from one step at a learning rate of 1e-12)
docker run --rm --gpus all -e SAMPLER_RECORD=1 -v $PWD/mlmd:/m -w /m <image> python bench/deepmd_train.py request.json bench/cases/hap-train/systems <run directory>
python mlmd/bench/deepmd_replay.py <run directory> mlmd/bench/cases/hap-train/dataset.jsonl <initial run directory>/model.native.json <target>/release <new output directory> --test mlmd/bench/cases/hap/dataset.jsonl
# Host: repeat finished exploration runs of the pipeline with an ensemble of native models (manifest.json and model-K.native.json in a case directory)
python mlmd/bench/hap_explore.py <pipeline>/iter-000/explore <case directory> <target>/release mlmd/bench/results/hap-explore-<tag>.json <run> ...
# Host: an exploration of every system of the case at 300 and 500 K as one replica run, then its candidates (new output directory)
python mlmd/bench/hap_explore_all.py mlmd/bench/cases/hap <target>/release <output directory> --result mlmd/bench/results/hap-explore-all-<tag>.json [--npt-temperatures 300]
```

`hap_npt.py` reads the LAMMPS run's `conf.lmp`, `in.lmp`, `log.lammps` and
`traj.dump` and starts no LAMMPS itself; `hap_explore.py` reads `conf.lmp`,
`in.lmp` and `model_devi.out` of every run and compares the fractions of
frames below, inside and above the force-deviation window. The training request is the one in
[`native/GPU_TRAINING.md`](../native/GPU_TRAINING.md). `deepmd_train.py`
copies the request's `model`, `learning_rate` and `loss` sections and its
steps, batch size, seed and display interval into a DeePMD input script, runs
`dp --pt train` and `freeze`, and writes the trainer's own time per batch and
the model converted for omniMD; the request's `dataset` and `output` paths
are the host's and are not read there. The request may ask for a hybrid:
its `precision` then goes into every descriptor of the list. It may ask
for se_e3, alone or in a hybrid, as well.

`deepmd_replay.py` takes the random numbers out of a comparison of the two
trainers. DeePMD-kit writes the system and the frame of every step when it
runs with `SAMPLER_RECORD=1`; omniMD then starts from DeePMD-kit's initial
model and is given those frames in that order, with the run's schedule and
loss, and the script reports how far its parameters end from DeePMD-kit's
and the errors of both models on a test set. A run of more than 10,000
steps goes in parts, each continued from the snapshot of the one before
(the trainer loads at most 10,000 frames). Run it with nothing else on the
card: the trainer of a part is stopped by force once its snapshot is
written, and a DeePMD-kit training in Docker that was running beside one
such stop ended with a device error.

Run steps 3 and 4 one after the other, never at the same time, so they don't
compete for the GPU. `cases/` is generated and git-ignored. Step 3 accepts
`--skip-deeppot` or `--skip-lammps` to rerun one route into a separate CSV.

The Python DeepPot route runs out of GPU memory from 12,288 atoms on a 24 GB card,
and on the 2026-10-29 Windows run the Docker Desktop VM went down during the
float32 attempt at that size. Use the LAMMPS route for the large cases.

## Metrics

`ms_per_eval` is one energy/force/virial evaluation; `us_per_atom_step` divides it
by the atom count. A LAMMPS row's `ms_per_eval` is the loop time of its timed
run divided by the steps, the integrator included (`Pair` takes over 99% of
the loop in a held run), and its `ns_per_day` follows its time step (0.5 fs,
or next to nothing held).
`peak_mem_mb` is the PyTorch allocator peak (DeepPot GPU rows only). Failed cases
are kept as rows with `status` = `oom`, `rejected`, `timeout` or `error`.

`bench_omnimd.py` (step 4) subtracts the fastest one-frame run from the fastest
K-frame run (three repeats each), which removes start-up and model loading but
still includes per-frame JSON parsing; that favors DeepMD slightly at large
sizes. The GPU evaluator of step 6 and of `bench_alternated.py` times in its
own process instead (`--bench K`: the mean of K evaluations of one frame after
two untimed ones). Each of its evaluations bins the atoms on the host and
searches their neighbors on the device anew, runs the kernels and downloads
the energy, forces and virial, where a held LAMMPS run never rebuilds its
neighbor list.

## Results

See `results/` and the summary table in `results/SUMMARY.md`.
