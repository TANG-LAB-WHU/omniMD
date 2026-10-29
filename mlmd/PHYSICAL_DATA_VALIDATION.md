# Physical E/F data training smoke test

Candidate `8167a4a3f3ff8d67f53c22f911f3ff31f053fa5d` completed one bounded
native CUDA fit against published quantum-chemical energy/force labels on
2026-10-16. The pipeline, export and CPU/GPU numerical comparison passed.
**The original fit worsened validation force error; this model is not suitable
for production.** Its original result is retained below. A subsequent
[bounded diagnosis](#bounded-training-diagnosis) made one controlled loss-weight
change; all 43 held-out frames are validation, not an untouched final test.
The [compact receipt](evidence/physical-nopbc-training.json) retains provenance,
source hashes, exact split indices, configuration, metrics and execution evidence.

| Partition / metric | Initialized model | After 1,000 updates |
| --- | ---: | ---: |
| Training energy RMSE, eV/atom | 0.775821 | 0.593012 |
| Training force RMSE, eV/angstrom | 6.970691 | 7.170289 |
| Held-out energy RMSE, eV/atom | 0.569513 | 0.455950 |
| Held-out force RMSE, eV/angstrom | 3.933565 | 4.164336 |

Energy/atom RMSE is the square root of the frame mean of `(E_error/N_atoms)^2`.
Force RMSE pools all force components across atoms. Independent native CPU
evaluation of the initial/final exports reproduced the training report metrics
within `1e-10`. The energy improvement does not establish force accuracy.

## Dataset identity and split

Source: official DeepMD `v3.1.0`, commit
`b494a0d5413d5f1895202c82f20477300b716bf2`,
[`examples/nopbc/data`](https://github.com/deepmodeling/deepmd-kit/tree/b494a0d5413d5f1895202c82f20477300b716bf2/examples/nopbc/data).
Its [README](https://github.com/deepmodeling/deepmd-kit/blob/b494a0d5413d5f1895202c82f20477300b716bf2/examples/nopbc/README.md)
identifies the subset as coming from Zeng et al., *Nature Communications* 11,
5713 (2020), [DOI:10.1038/s41467-020-19497-z](https://doi.org/10.1038/s41467-020-19497-z).
The published final E/F labels were calculated with Gaussian 16 at
MN15/6-31G**; initial geometry collection involved ReaxFF and active learning.
These are published quantum-chemical labels, not newly computed or neural-teacher
labels. The calculations themselves were not independently recomputed here.

Original data attribution: Jinzhe Zeng and Tong Zhu, *Dataset for methane
combustion* (2020), [version 3](https://doi.org/10.6084/m9.figshare.12973055.v3),
licensed **CC BY 4.0**. The official DeepMD repository distributes its materials
under **LGPL-3.0-or-later**. Both attributions are retained. The full original
research dataset contains 578,731 structures; this run used only the small
official example described below. No source arrays are redistributed in this commit.

| Composition group | Source frames | Atoms/frame | Train | Held out | Train index interval | Held-out interval |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| C1H4O2 | 1,562 | 7 | 96 | 24 | `[0,1093)` | `[1250,1562)` |
| C3H3O4 | 485 | 10 | 64 | 16 | `[0,339)` | `[388,485)` |
| C4H3O1 | 15 | 8 | 8 | 3 | `[0,10)` | `[12,15)` |
| Total | 2,062 | 7–10 | 168 | 43 | | |

Within each half-open interval `[lo,hi)`, select `k` indices using
`lo + floor(i*(hi-lo-1)/(k-1))`, for `i=0..k-1`. Intervals were fixed before model
evaluation: training ends at `floor(7*N/10)`; validation starts at
`N-floor(2*N/10)`. Frame IDs retain `COMPOSITION/set.000/SOURCE_INDEX`.
Training and validation IDs are disjoint. No validation label or statistic enters
the baseline, initialization, updates, schedule or continuation decision.

Species-aware sorted pair-distance screening found no equal quantized fingerprint
at `1e-4` angstrom, nor maximum sorted-distance difference at or below that
threshold, among 3,352 cross-partition same-composition pairs. The minimum observed
maximum distance difference was 0.101150 angstrom. Original trajectory identities
and ordering semantics are unavailable; this is **index-held-out validation within
the example**, not independent-trajectory or chemical-transferability validation.
The index gap and geometric screen do not prove statistical independence.

All 21 downloaded files (400,761 bytes including license/configuration) matched
the pinned official Git blob identities. NPY loading disabled pickle. The source
arrays are float32 and were promoted exactly to FP64, preserving atom order,
absolute coordinates and every E/F value. Units are angstrom, total eV/frame and
eV/angstrom under the [official data convention](https://docs.deepmodeling.com/projects/deepmd/en/v3.1.0/data/system.html).
The `nopbc` marker becomes `cell:null`. No virial labels exist; virial loss was
disabled. The model's computed virial was compared only for CPU/GPU parity.

## Frozen model and resource plan

The pre-run plan was saved before any model computation. It selected native-v1
FP64 `se_e2_a`, ordered types `[C,H,O]`, nine independent pair embeddings
`1 -> 4 tanh`, and three fitting networks `8 -> 8 tanh -> 8 tanh -> 1 linear`:
**531 neural parameters**. Axis width is 2; descriptor width is 8; `sel=[4,4,4]`,
cutoff 6 angstrom, smooth cutoff 1 angstrom; no residuals or branch timesteps.
Actual neighbor counts never exceeded `[3,4,4]`.

Python `random.Random(20261016)` initializes matrices with Xavier uniform values
in fixed center/neighbor/layer/input/output order; neural biases start at zero.
Environment normalization is fixed at mean zero/std one. Fixed atomic energy
offsets solve the three composition equations against training-only
`mean(E_label - E_initial_NN)`, evaluated on CPU. Offsets for C/H/O are
`-1029.3460205372046 / -14.575719399331478 / -2047.1684064867834` eV.
These are regression baselines, not isolated-atom reference energies.

The explicit request uses CUDA reverse derivatives and GPU prediction/loss,
batch size 8, shuffled epochs with seed 20261016, E/F weights 1, scales
0.1 eV/atom and 1 eV/angstrom, Adam learning rate 0.001, betas 0.9/0.999,
epsilon `1e-8`, clip norm 10, and rate decay by 0.5 every 250 updates.
Tiles allow 8 frames / 80 atoms, with fixed geometry caching and checkpoints
every 100 updates. The default 64-atom frame ceiling remains selected.

The predeclared sequence was 100 updates, then 900 updates from its checkpoint
only if `9*pilot_seconds + 60` fit the remaining 20-minute budget with 90 seconds
reserved. This decision used timing alone. Both stages completed. This is v8
continuation; no extra uninterrupted-versus-resumed equality run was performed.
This physical replay does **not** validate the conditional 256-atom/v9 path.

The runtime planner confirmed 17,607,360 estimated scalar terms at tile capacity
(limit 50,000,000), 1,597,072 bytes planned explicit device payload and 569,904
bytes host tile payload. Actual peaks were 1,538,008 / 489,576 bytes, and the
training geometry cache retained 3,500,387 bytes. Payloads are not total GPU/RSS
limits. CUDA computes prediction, detached loss seeds, mixed derivatives and Adam;
CPU retains geometry, normalization scalars, stable reduction, clipping and assessment.

## Runtime and measured execution

The clean candidate was built in an isolated read-only source mount using the
cached official `deepmodeling/deepmd-kit:3.1.0_cuda129` image, ID
`sha256:23d23b922459eb49b23f9d4977b6e453fd17feb2ee03a280c70fb7f4a6452e8c`.
Runtime: Ubuntu 22.04.5 / WSL2 x86_64, RTX 4090, driver 617.42 (Driver API 13040),
NVRTC 12.9.86, existing Rust/Cargo 1.99.0 and conda-forge GCC 13.3.0-2.
Offline locked release compilation took 36.85 seconds. No package installation,
driver/security change, HAP/private data access or new quantum calculation occurred.

The initial physical seven-atom CPU/GPU probe passed. Training recorded 8,000
GPU-predicted frame visits, 1,000 tiles/synchronizations and 10,000 gradient-path
kernel launches across 1,000 updates. These are application counters; they exclude
Adam launches and are not hardware-profiler counts. The explicit backend was
`cuda_tiled_device_loss_f64_v1`; there was no fallback.

The 100-update process took 1.420652 seconds; the resumed 900-update process took
1.564180 seconds. The full bounded sequence, including probe and output comparison,
took 4.477439 seconds (4.504780 including CPU baseline preparation). These are
single observations including startup/JIT, transfers, assessment and checkpoint
I/O, not steady-state training benchmarks or CPU/GPU speedup measurements.

The trained model's three held-out slice frames had maximum CPU/GPU absolute
differences E `0`, F `4.440892098500626e-16`, virial `8.881784197001252e-16`.
Every component passed `1e-8 + 1e-10*abs(CPU)`. This compares inference, not CPU
and GPU optimizer trajectories. Export reload E/F/V differences were zero.
No GPU error or retry occurred. Container `omnimd-physical-8167a4a` was stopped
and the GPU window released immediately afterward.

## Bounded training diagnosis

The follow-up used the same compiled candidate and existing data, with one short
CPU/GPU optimizer comparison and **one** controlled training comparison. No native
code, labels, architecture, normalization, split or initialization changed.
The [diagnosis receipt](evidence/physical-nopbc-diagnosis.json) records the plan
saved before execution, actual request differences, full metrics, hashes and logs.

The evidence supports a loss tradeoff in this small training setup. No inspected
data-mapping, energy-baseline or short CPU/CUDA optimizer defect was found, so
no implementation fix is justified by this experiment. It does not prove that
model capacity or training duration explains all remaining error, exclude shared
CPU/CUDA defects, or establish agreement through every one of the 1,000 updates.

### Initialization, mapping and loss diagnosis

All 211 frames retain the official atom order and C/H/O mapping, including H-first
source rows. The fixed atomic baseline uses only the 168 training labels and
initial CPU predictions. Each composition's mean training energy residual after
the offset is below `1e-9` eV. Fixed offsets cannot change forces.

Pure Python arithmetic on the 168 training geometries found no broad tanh
saturation. Initial/original-final descriptor absolute-value P95 was
`0.0145877 / 0.0400383`; maximum first fitting-layer preactivation was
`0.112312 / 0.249695`, and second-layer maximum was `0.0572682 / 0.639892`.
Neither fitting layer had activations above 0.95 in absolute value. Only 0.1166%
of original-final embedding activations exceeded 0.95 (padding included).
The fixed mean-zero/std-one normalization produces small fitting inputs, but
its independent effect on training was not tested. All 531 neural parameters
changed in both fits; fixed model metadata and baselines stayed identical.

The original objective is
`mean_frames[(dE/N/0.1)^2] + mean_frames[mean_components(dF^2)]`.
From initialization to the original 1,000-step model, its energy/force terms were:

| Partition | Initial energy / force | Original final energy / force |
| --- | ---: | ---: |
| Training | 60.1898 / 46.0598 | 35.1663 / 48.7387 |
| Validation | 32.4345 / 14.6913 | 20.7890 / 16.4497 |

The total objective can improve while force error worsens, even on training
data. Force loss uses equal frame weighting; the reported force RMSE pools atom
components. Both conventions were independently reconciled. Saved intermediate
checkpoints do not contain an error curve, so no onset step is inferred.

### Two-update CPU/CUDA state comparison

Both backends started from the same initial model and used the same first two
shuffled batches of eight training frames. CPU set `device=cpu` and removed only
`cuda_gradient`, `cuda_loss`, `cuda_tiling` and `fixed_geometry_cache`; both used
two updates and the original E/F weights. All other numerical settings matched.
CPU uses reverse AD over coordinate-directional tangents; CUDA uses device loss/reverse
derivatives and Adam. No checkpoint was resumed across backends.

The preregistered per-element tolerance was
`abs(a-b) <= 2e-10*(1+max(abs(a),abs(b)))`:

| Compared array | Elements | Maximum absolute difference |
| --- | ---: | ---: |
| Neural parameters | 531 | 2.77556e-17 |
| Adam first moments | 531 | 1.11022e-16 |
| Adam second moments | 531 | 6.93889e-18 |

Every value was finite. Step, sampler, partitions, data/layout hashes, initial
model and frozen fields matched exactly. CPU v6 and CUDA v8 checkpoint schemas
are intentionally different. This checks 16 frame visits and two updates only.

### Single controlled loss-weight comparison

The sole numerical configuration change was `energy_weight: 1.0 -> 0.1`.
The rationale used the training loss decomposition: at that weight, the old
final model scores worse than initialization (`52.2554` versus `52.0787`), so
the old energy-for-force tradeoff is no longer rewarded. This was a hypothesis,
not a guarantee of better force accuracy.

The comparison restarted from the identical `initial.json` with a fresh optimizer
and the identical 100-update plus 900-update continuation. E/F scales, force
weight, batch/seed, Adam, clipping, learning-rate schedule, tile and cache settings
remained fixed. Paths changed only to separate outputs and their checkpoints.
Clipping remains at 10; changing the energy weight can also change when clipping
is active. Neither clipping frequency nor per-step gradient balance was measured.

| Partition / metric | Initial | Original weight 1, step 1000 | Weight 0.1, step 1000 |
| --- | ---: | ---: | ---: |
| Training E/atom RMSE, eV | 0.775821 | 0.593012 | 0.765865 |
| Training total E RMSE, eV | 6.626241 | 5.099892 | 6.650750 |
| Training force RMSE, eV/angstrom | 6.970691 | 7.170289 | 6.619143 |
| Validation E/atom RMSE, eV | 0.569513 | 0.455950 | 0.523472 |
| Validation total E RMSE, eV | 4.878643 | 3.947965 | 4.451817 |
| Validation force RMSE, eV/angstrom | 3.933565 | 4.164336 | 3.707381 |

Compared with initialization, controlled force RMSE improved by 5.04% on training
and 5.75% on validation. Energy improvement was smaller than the original fit;
training total E RMSE actually worsened by 0.37%. Each cell below is
`E/atom RMSE (eV) / force RMSE (eV/angstrom)`; no composition was omitted:

| Partition / composition | Initial | Original weight 1 | Weight 0.1 |
| --- | ---: | ---: | ---: |
| Training C1H4O2 | 0.741510 / 4.782089 | 0.551635 / 4.968613 | 0.698176 / 4.696897 |
| Training C3H3O4 | 0.858108 / 7.474768 | 0.667514 / 7.710302 | 0.890345 / 7.034156 |
| Training C4H3O1 | 0.388610 / 15.679827 | 0.410753 / 15.864367 | 0.355543 / 14.681568 |
| Validation C1H4O2 | 0.540499 / 2.569158 | 0.402533 / 2.704303 | 0.509124 / 2.510929 |
| Validation C3H3O4 | 0.640125 / 4.149817 | 0.521140 / 4.405649 | 0.576455 / 3.913755 |
| Validation C4H3O1 | 0.355602 / 8.119471 | 0.484806 / 8.588005 | 0.285899 / 7.438997 |

C3H3O4 training energy worsened versus initialization. Its ten-atom frames
carry more weight in total-E squared error than seven/eight-atom frames, which
explains why aggregate E/atom and total E move differently. The new objective
decreased from `52.0787 -> 47.5852` on training and `17.9348 -> 15.8195` on
validation; total objectives with different weights must not be compared directly.
The receipt also evaluates all three models under each common objective.

The controlled fit changed all 72 embedding, 432 hidden-fitting and 27 output
parameters. Their L2 displacements from initialization were respectively
`2.341564 / 5.805360 / 1.542246`, versus `2.790761 / 6.059254 / 1.425379` in the
original fit. Its final model SHA256 is
`9f8bf5036b8f168f1b944e2bd7a3b17fda4e4f2de39895e59d1db21f5806d1ea`.

Independent CPU evaluation reproduced all new final metrics within `1e-10`.
Three validation slice frames had maximum CPU/GPU E/F/V differences
`0 / 8.88178e-16 / 4.44089e-16`; export reload differences were zero. CUDA
execution was explicit, with no fallback, error or retry. The comparison's fit
processes took `0.433565 + 1.402735` seconds. The diagnostic runner took
2.859038 seconds; combined with the original runner and CPU baseline, recorded
wall time was 7.363818 seconds, below the 1,200-second compute budget. These
single measurements include startup/JIT/cache, synchronization and I/O; they
are not a speedup comparison. Compilation, static analysis and documentation
are outside that recorded model-run wall time.

The container was stopped immediately after completion and the GPU window
released. No further fit or parameter search followed. This remains one seed,
three compositions and index-held-out **validation** used for analysis; there
is no untouched final test. Errors remain large, especially C4H3O1 forces, and
the controlled model is not ready for production. No wider architecture or
chemical-transferability claim follows from this experiment.

Exact requests, models, checkpoints and logs are in the existing local artifact
root under `diagnosis-01/`. The local `run_diagnosis.py` and
`diagnostic-static-analysis.json` retain the recipe/arithmetic; hashes are in
the tracked receipt. A replay needs those task-local files and a fresh output
directory with the same cached inputs and binaries. Never clear attempt markers
or reuse attempted outputs; this diagnosis does not rerun automatically.

## Replaying the saved experiment

Only this guide and compact evidence are tracked. Exact source arrays, initialized
and trained weights, checkpoints, raw logs and helpers remain in the task-local
directory `C:/Users/Shawn/Documents/Codex/2026-10-13/task-4/physical-run-8167a4a`.
The receipt records the hashes of `fetch_official_nopbc.py`, `prepare_physical.py`,
`run_physical.py`, all source arrays and outputs. These helpers are required to
replay the saved recipe; they are not installed or included in the evidence commit.

Use a fresh output directory. Copy those three helpers there and copy the saved
`official-source` directory unchanged. If that data cache is unavailable, the
fetch helper obtains only the same pinned official files and verifies their Git
blob hashes; run it once from the fresh directory. It executes no remote code.
Preserve an isolated source checkout at the exact candidate. Mount it as `/src`
read-only and the new output directory as `/work` in a fresh offline container
from the image above. Use the cached compiler/registry mounts from
[the Linux runtime recipe](LINUX_CUDA_VALIDATION.md#bounded-reproduction), replacing
the candidate, output and container names. Never reuse an attempted output directory.

Build and execute, sequentially, inside that container:

```bash
set -eu
export CARGO_HOME=/work/cargo CARGO_TARGET_DIR=/work/target
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/opt/deepmd-kit/bin/x86_64-conda-linux-gnu-cc
export PATH=/toolchain/bin:/opt/deepmd-kit/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cargo build --offline --locked --release --manifest-path /src/mlmd/Cargo.toml \
  -p omnimd-native -p omnimd-mlmd \
  --features omnimd-native/cuda,omnimd-mlmd/native-cuda --bins

CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 /opt/deepmd-kit/bin/python \
  /work/prepare_physical.py --eval-exe /work/target/release/omnimd-native-eval

OMNIMD_NVRTC_LIBRARY=/opt/deepmd-kit/lib/libnvrtc.so.12 \
LD_LIBRARY_PATH=/opt/deepmd-kit/lib CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  /opt/deepmd-kit/bin/python /work/run_physical.py --execute
```

Review the generated preparation/resource report before the last command.
Stop the container on completion or failure. The runner uses exclusive attempt
markers, a combined 1,200-second budget and per-command timeouts; a failure stops
the sequence without retry. Do not clear markers to repeat an uncertain GPU run.
The `physical-acceptance.json` file is published only after completion. Assess the
retained held-out metrics as reported, including force regression; no chemical
accuracy threshold is implied by successful pipeline completion.
