# Validation record: 2026-10-14

Later physical-data evidence: [public nonperiodic E/F training, 2026-10-16](PHYSICAL_DATA_VALIDATION.md)
covers exact candidate `8167a4a`: 168 training / 43 held-out frames, 1,000 CUDA
updates and numerical CPU/GPU inference parity. The original force error worsened.
A [single controlled loss-weight comparison](PHYSICAL_DATA_VALIDATION.md#bounded-training-diagnosis)
and two-update CPU/GPU optimizer check support a loss tradeoff in this small
training setup. The new force error improved but remains large; energy results
are mixed. The 43-frame holdout is validation used for diagnosis, not a final test.

Later platform evidence: [Linux native CUDA validation, 2026-10-16](LINUX_CUDA_VALIDATION.md)
covers exact candidate `06439f1`, including tiled GPU E/F/V training and exact
same-runtime checkpoint resume. The sections below retain their dated scope.

## Builds and targeted tests

- Isolated official Rust 1.99.0: Windows GNU workspace build and 19 tests passed
  (6 C-API boundary tests, 2 contract tests, 4 MD/restart tests, 7 native tests).
- Linux x86_64 workspace build passed inside the official DeepMD CPU container.
- No mandatory tch/chemfiles dependencies were built for the MLMD workspace.
  The full legacy root workspace and old Rust 1.82 MSRV were not tested.
- Existing legacy float-constant deprecation warnings remain.
- Mock FFI tests verify single model load, one combined call per frame, ownership,
  ordered type mapping, row cells, missing/nonfinite output and backend failures.
- Native synthetic tests cover analytic forces/all nine strain derivatives,
  normalization/padding, cutoff, invariance, skew-cell images and invalid formats.
- MD tests check initial forces and units, one evaluation per step, conservation,
  failure rollback, cell orientation, rejection of potential mixing and restart.

## Real official C API and MD

Runtime: DeepMD 3.1.3, TensorFlow 2.19.1, CPU image
`ghcr.io/deepmodeling/deepmd-kit:3.1.3_cpu`, pinned manifest digest
`sha256:7953cbd4b48798e936bae520d44a12ae747c3eb45d9aca8ac75668edc95240a2`.
No GPU devices are passed to the container; worker visibility is disabled.
The C library is `/opt/deepmd-kit/lib/libdeepmd_c.so`.

The public [official regression fixture](https://github.com/deepmodeling/deepmd-kit/blob/v3.1.3/source/tests/infer/deeppot.pbtxt)
was checksum-verified, converted to protobuf and inspected: floating Const nodes
and coordinate input are float64. Its type map is O,H. No production suitability
is assumed. Models/runtimes/raw logs are outside Git.

[`evidence/deepmd-cpu-3.1.3.json`](evidence/deepmd-cpu-3.1.3.json) records checksums
and measured results:

| Comparison | Maximum absolute difference |
| --- | ---: |
| Total energy vs official Python | 0 eV |
| Force vs official Python | 0 eV/angstrom |
| Virial vs official Python | 6.661338147750939e-16 eV |
| One force component vs central energy difference | 6.65607585714767e-9 eV/angstrom |
| Off-diagonal virial vs central strain difference | 3.462445885560328e-9 eV |

Four frames cover nonperiodic, cubic, triclinic and permuted-atom inputs.
Wrong ordered map, missing model and invalid protobuf fail explicitly.

Actual omniMD NVE: 6 atoms, 0.01 fs, 100 steps (1 fs). Maximum total-energy
drift is 7.740936780464835e-9 eV. Against a separate Python velocity-Verlet loop
using official DeepMD forces, final positions agree exactly and velocities differ
by at most 4.336808689942018e-19 angstrom/fs. Final E/F/virial also agree.

A prior 0.1 fs, 100-step (10 fs) run exceeded the chosen 1e-3 eV drift criterion
(maximum 0.02642816732270603 eV). Its final trajectory nevertheless matched the
independent Python integrator. The shorter 1 fs result does not establish
equal-duration timestep convergence or long-time physical stability.

After adding checkpoint and backend selection, a 50+50-step real-model restart
was compared with the existing continuous run. Position, velocity, E/F/virial
differences were all zero. Native synthetic-model CLI dynamics completed 20 steps
at 0.01 fs, with drift 2.220446049250313e-16 eV. See
[`evidence/md-restart-native.json`](evidence/md-restart-native.json).
These small fixture checks make no physical-model or speed claims.

## Reproduce the real checks

Build the three binaries with the nested workspace lockfile. In a compatible
CPU runtime, use:

```sh
python mlmd/scripts/validate_deepmd.py \
  --work /absolute/new-validation-dir \
  --library /opt/deepmd-kit/lib/libdeepmd_c.so \
  --eval-bin /absolute/omnimd-deepmd \
  --md-bin /absolute/omnimd-mlmd \
  --timestep-fs 0.01
```

The script downloads only the pinned public fixture (or accepts `--pbtxt`),
compares E/F/virial, performs finite differences and checks the actual MD path.
`validate_md_smoke.py --help` documents the focused restart/native CLI check.
Outputs require new paths and stay outside the source tree.

## Thermal CLI and CSVR validation

The command entry is a separate JSON runner. The implementation uses upstream
System and the fallible sim::md::MlipDynamics adapter, with upstream Gaussian
initialization, COM removal and CSVR. It does not enable the full legacy
Simulation/Propagator pipeline. See [engine scope and bridge](engine/README.md).

Four additional thermal tests passed (23 unique MLMD tests in total; the prior
19 workspace tests and the targeted NVE rerun also passed). The updated CLI built
on Windows GNU and Linux. Tests exercise reproducible initial velocities,
mass-weighted COM removal and 3N-3 DOF, signed/independent CSVR noise, stochastic
checkpoint continuation, and a small anisotropic harmonic oscillator.

[Measured thermal statistics](evidence/thermal-statistics.json):
independent one-step CSVR trials used DOF 1,2,3,6 with 25,000 samples each.
Mean/analytic ratios ranged 0.9902-0.9984 and conditional-variance ratios
0.9849-1.0128. Negative velocity rescaling was observed. The 3D unequal-frequency
oscillator used 100,000 inexpensive synthetic steps at 0.1 fs, discarded 10,000
and sampled every 10 steps. Kinetic mean/variance ratios were 1.0071/1.0038;
coordinate second-moment ratios were 0.8714/0.9280/1.1349. Maximum effective-energy
drift was 1.0032e-5 eV. These are finite-sample smoke checks with correlated
configurations, not a general ergodicity or production-ensemble certification.

Actual user CLI checks:

- [Native, Windows](evidence/native-nvt-cli.json): 100 seeded NVT steps at 0.01 fs,
  COM removed (9 DOF); 50+50 continuation produced an identical final frame.
  Periodic checkpoint verified; ensemble change, unknown units and overwrite
  rejected. Maximum effective-energy drift 1.6964e-13 eV.
- [Official C API, Linux](evidence/capi-nvt-cli.json): 40 seeded NVT steps at
  0.001 fs on the existing public regression model, COM removed (15 DOF).
  20+20 continuation produced identical final frame **and checkpoint/RNG state**.
  Periodic checkpoint and zero-kinetic rejection verified; effective-energy drift
  1.7053e-12 eV. This is a short CLI integration check, not a thermalized production
  trajectory.

NVT reports thermostat heat and effective energy separately from fluctuating
total energy. These checks do not establish GPU/MPI/NPT or legacy output-plugin
support. The shared validation container remains available to B; the new Linux
CLI was built into a separate target directory.

## Supported DeepMD conversion

The native converter commit 10ee322 is integrated. It accepts the explicitly
documented DeepMD 3.1.3 .dp/HDF5 or safe YAML subset, with mandatory units and
dpmodel/tensorflow geometry mode. B's
[same-generated-model evidence](native/fixtures/deepmd-3.1.3-report.json)
records three frames with maximum E/F/virial differences
4.4409e-16 / 5.0307e-17 / 3.4694e-17, eight focused rejection cases and identical
.dp/YAML conversion outputs. These generated weights are untrained; this result
does not validate arbitrary pretrained models. See [conversion scope](native/CONVERSION.md).


## Frozen graphs and dataset workflow

B's native-only commit 39b2ea6 is integrated as 1498d40. The two audited graph
profiles preserve both energy offset sites and effective TensorFlow cutoff
attributes; their provenance and licenses are retained. Public four-frame
E/F/virial differences are 2.2737e-13 / 4.2188e-15 / 6.6613e-15; generated-model
differences remain below 4.45e-16 / 5.04e-17 / 3.48e-17. Seven focused rejections
passed. See [scope](native/FROZEN.md) and [B's evidence](native/fixtures/frozen-graph-report.json).
No public pretrained weights were committed, and no generic graph support is claimed.

Four engine data tests passed on Windows (27 unique MLMD tests cumulatively,
not a repeated full workspace run). They cover symbol/order/tensor preservation,
independent metric arithmetic, PBC/dtype/shape/nonfinite/overflow/unknown-field
rejections and exact JSONL completion. Windows and Linux CLIs compiled.

[Data workflow evidence](evidence/dataset-workflow.json) uses actual NumPy arrays,
the same public model and deliberately offset synthetic labels. Two labeled
periodic frames plus one unlabeled frame exercise coverage/counts and independent
NumPy MAE/RMSE calculations. Nonperiodic unlabeled input, source-file digests,
both model-info backends, capabilities, wrong units and incomplete-output
handling also passed. Batch native/C API maximum differences were E 2.2737e-13,
F 4.4409e-15 and virial 7.3275e-15 in contract units.

The same imported triclinic frame initialized both MD backends for 20 steps at
0.01 fs (0.2 fs total). Across all snapshots: coordinates matched exactly;
maximum velocity difference 5.2042e-18 angstrom/fs, E 2.2737e-13 eV,
F 6.5503e-15 eV/angstrom, virial 7.5495e-15 eV. Maximum absolute NVE energy drift
was 3.4538e-10 eV for C API and 3.4515e-10 eV for native. This is a short
integration/parity check, not chemical accuracy, long-time stability or a speed claim.

Replay with the existing public C API request/evaluation and an exported native
model (all output paths must be new):

    python mlmd/scripts/validate_dataset.py --binary /absolute/omnimd-mlmd \
      --work /absolute/new-work --public-request /absolute/eval.request.json \
      --public-evaluation /absolute/eval.json --native-model /absolute/native.json

## XYZ, minimization and integrated readout training

B's native-only f941992 is integrated as 26e1d10. It fits final linear readouts
using exact analytic E/F basis features and rank-aware pivoted QR, retaining
descriptor/hidden parameters and, by default, biases. Its 48-frame manufactured
36/12 experiment and ten native tests are documented in
[native/TRAINING.md](native/TRAINING.md). Full supported-network mixed-derivative
training is subsequent work, not part of this integrated milestone.

Six targeted engine tests passed: four minimization tests (including legacy
harmonic behavior after sharing the step-size policy) and two XYZ tests.
They check force-plus-energy convergence, displacement limits, backtracking,
rollback on model errors, stationary/iteration/stagnation outcomes, exact
lattice/virial ordering, label-free export and malformed input rejection.
These are focused checks, not a repeated full repository test run.

The actual [structure workflow](evidence/structure-workflow.json) uses 24
explicitly selected periodic synthetic teacher frames imported from extxyz.
Six named IDs are held out by the training request. Across the 24 assessment
frames, E/atom RMSE fell from 0.12954023 to 2.4476e-16 eV/atom and force RMSE from
0.01154521 to 1.0924e-17 eV/angstrom. These are manufactured teacher recovery
and IO checks, not physical accuracy measurements or a new generalization claim.

Both C API and native public-model minimizers accepted 51 moves (52 evaluations)
and satisfied the selected force 0.01 eV/angstrom and energy-change 1e-5 eV
criteria. Maximum final force was 0.00308753 eV/angstrom and last energy change
about -9.91528e-6 eV. Energy fell from about -931.05341046 to -939.35875305 eV.
Final backend maximum E/F/virial differences were
2.2737e-13 / 7.5218e-15 / 4.7956e-15 in contract units. This validates agreement
and the stopping implementation for a public regression model, not chemistry
or a global minimum.

The fitted synthetic teacher's separate relaxation reduced energy from
-0.76765602 to -3.70429566 eV but hit the explicit 300-step limit with maximum
force 0.03937108 eV/angstrom. It correctly returned nonzero and
converged=false, status=max_iterations. The last accepted geometry exported
without reference labels, reimported exactly and entered a five-step MD IO test;
energy drift was 4.4409e-16 eV. This intentionally tests continuation from an
explicitly nonconverged geometry and is not an equilibration result.

The integrated trainer, imports and relaxation share dataset JSONL v1 without
changing the shared Frame/Potential contract. Windows and Linux binaries build.
The full end-to-end run precedes only a whitespace-tolerant scalar-parser polish
and a more accurate numerical-error status label; its binary hash is retained
in evidence. The final sources are separately rebuilt and targeted tests pass.

## Final CPU delivery acceptance

B full-training commit f78c146 was integrated as 40b7319 without changes to
native code or dependencies. The final affected mlmd workspace passed 39 tests
on Windows, including all 13 native tests, C API contract checks, data imports,
minimization, NVE/CSVR and checkpoint tests. Windows debug and Linux release
binaries built offline with the existing Rust 1.99 toolchains; the README native
fixture command also passed. The legacy root workspace was not rerun.

The single final [release replay](evidence/release-workflow.json) uses
scripts/validate_release.py and B's existing full-training recipe. It trains all
112 neural parameters on 24 original synthetic frames (18 train/6 validation),
then checks the serialized 150+150 restart against 300 continuous Adam updates.
Models and moments match exactly. Export/reload E/F/virial differences are zero.
Training objective fell from 8.83036148 to 0.00053021182. Held-out E/atom RMSE fell
from 0.0603337 to 0.000447666 eV, and force-component RMSE from 0.317807 to
0.00679365 eV/angstrom. Virial is assessed, not optimized. These are synthetic
teacher-learning checks, not chemical accuracy or generic DeepMD training.

The unified train-native command resumed the checkpoint for one additional
update to global step 301. The resulting model passed extxyz import of 12
periodic frames, batch prediction, fixed-cell minimization, geometry-only export
and exact reimport, then five MD steps. This synthetic minimization exhausted
its explicit 50-step budget (maximum force 0.691402 eV/angstrom) and correctly
returned nonzero; the subsequent MD is only an IO/integration check, not an
equilibration claim. Its energy drift was 3.8858e-16 eV over those five tiny steps.

The unchanged public regression model again converged in both C API and native
minimizers in 51 accepted steps. Final maximum E/F/virial backend differences
were 2.2737e-13 / 7.5218e-15 / 4.7956e-15. The report records the exact release
binary hashes; no generated weights, datasets, checkpoints or raw logs are
committed. All replay outputs remain in the task-local runtime directory.

Reproduce on a prepared runtime with existing public fixture paths:

    python mlmd/scripts/validate_release.py --binary /path/omnimd-mlmd --native-eval /path/omnimd-native-eval --native-train /path/omnimd-native-train --public-request /path/eval.request.json --public-native-model /path/public-native.json --work /new/replay-directory

This CPU candidate does not include CUDA implementation or GPU validation.
The separately authorized CUDA work starts from this stable baseline.

## Local native CUDA integration

Native implementation 1f572de and measured evidence aa24526 are integrated as
234173c and 2e8b06b. The [native CUDA report](native/fixtures/cuda-validation-report.json)
records the owner's actual GPU run on three original/public fixture sets. Maximum
E/F/virial differences across those cases were 3.55e-15 / 4.44e-16 / 8.88e-16.
The 216-atom, 28258-parameter case measured 60.385 ms CPU and 52.288 ms CUDA
(1.155x CPU/CUDA ratio); the small cases were about 8.4x slower. Timing includes
CPU geometry, transfers and reductions, excludes initialization, and has the
reported desktop-graphics-load limitation. Native CUDA performs inference,
not parameter training. See native/CUDA.md for exact subset/platform bounds.

The unified CLI now accepts native options with explicit device=cpu or cuda:N,
behind the engine native-cuda feature. Default selection remains CPU at every
system size. Feature-disabled CUDA requests fail explicitly. Both backends use
the same isolated child device mapping; the native constructor receives visible
ordinal zero only after validating the worker mapping. Shared Frame/E/F/virial
formats are unchanged, and no dependency was added.

Two new CPU-only routing tests passed in both default and native-cuda builds;
the feature-enabled Windows CLI compiled and its static capabilities report
confirmed the feature. The tests neither initialize CUDA nor invoke NVRTC.
In one subsequent exclusive GPU window, A used the existing Windows CUDA 12.8
runtime and the feature-enabled debug CLI at f912b2d. The [CLI replay report](evidence/native-cuda-cli.json)
records one CPU eight-step NVE run, one CUDA eight-step run and one CUDA four-step
continuation from the step-four checkpoint. All use the same checked-in original
112-parameter tanh model and four-atom triclinic frame, dt=0.001 fs. Headers
confirm native-se-e2-a-v1 versus native-cuda-se-e2-a-v1 and the same model SHA.
Across nine matched snapshots, maximum absolute E/F/virial differences were
0 / 6.6613e-16 / 8.8818e-16; positions and velocities matched exactly. Every
resumed snapshot and final checkpoint state equaled the continuous CUDA output
after JSON decoding. Both runs had maximum energy drift 7.2564e-13 eV. This is a
short CLI/restart integration check, not a timestep/stability or timing study.

All three sequential processes exited successfully. No error or retry occurred;
no omnimd-mlmd process remained at 05:04:49 UTC and the GPU window was released.
Only the compact report and reproducer are tracked; raw requests, trajectories,
checkpoints and logs stay in the task-local runtime. The owner's GPU benchmark
was not rerun. DeepMD C API was not invoked by this native replay; its separate
completed validation follows. The CUDA integration was subsequently published
on shawn_dev as 907481708ae5f3df4228a9fb5d5b1110c39f5e23.

## Local C API CUDA integration

Validation commit f839cef was imported as 03965e7 using only its three exact
validation files: the [report](deepmd/CUDA_VALIDATION.md), compact
[receipt](deepmd/validation/cuda-public-six-atom.json), and saved-output comparator.
Operational-only parent 46d913b was not imported into candidate ancestry; its hash
in the receipt identifies C's actual source HEAD at runtime. Rust implementation
remains bf6f2cf as already integrated by 3f0b26a.

C's authorized run used the official DeepMD 3.1.0 / TensorFlow 2.18.0 CUDA image
on RTX 4090 and the public six-atom O/H model. Four frames (nonperiodic,
orthorhombic, triclinic and permuted triclinic) had maximum absolute CPU/GPU
E/F/virial differences of 1.1369e-13 eV / 1.5543e-15 eV/angstrom / 2.4425e-15 eV.
The evaluator recorded 24,863 successful direct CUDA kernel launches over
92 evaluations and synchronized the observed contexts. This does not imply all
TensorFlow operations ran on GPU. GPU batch completed four frames and GPU MD
completed two steps plus a checkpoint. These are bounded integration checks,
not chemical-accuracy or long-time stability tests.

The paired 80-evaluation steady sample measured 0.485547 ms/frame CPU versus
4.017700 ms/frame GPU: GPU was 8.27458 times slower. It includes validation,
packing, transfers, CUPTI kernel checks, synchronization and output validation;
it excludes process startup, JSON serialization, loading and warmup. This is one
small sample, not a statistical benchmark or evidence for larger systems. The
container stopped and the exclusive GPU window was released at 06:13:51.533 UTC.

A integrated the saved evidence without another GPU run. The strict comparator
passed on the saved CPU/GPU outputs under python -O, exactly matching all ten
compared report fields (errors, kernel counts, timing conditions and ratios).
The Windows debug unified CLI built offline with native-cuda. Its static
capabilities confirmed both CPU defaults, the compiled native CUDA feature,
GPU training=false and the three existing evidence paths. This query performed
no GPU probe. The recorded runtime hashes remain those of C's measured binaries,
not a claim that the newly built integrated binary underwent another GPU acceptance.

## Limits and local provenance

The C API guarantees double buffers, not all models' internal precision or
training units. CPU remains the default; explicit CUDA is bounded by the local
acceptance above. Native inference accepts only its documented se_e2_a subset.
Bounded tanh full-network training is integrated on CPU; GPU training, generic
DeepMD training and hybrid se_e3 remain unsupported. Both native and C API CUDA
have the specified local fixture validation above, with different runtime bounds.
The supported safe-serialization converter and generated-model comparison are integrated.
The two audited frozen-graph profiles are integrated; the public regression .pb
lacks training serialization metadata, so generic export is not claimed.

Upstream main was `761b2484feee132426183c9b66ff2b126d1f7c4a`;
dev `186817f5a85837df5f1df28e0fd2e8b2de6ecade`
(main...dev 13/1, final diff README only);
old shawn_dev `3440f605dde61b01e894f6532693612bc57c0aaa`
(main...shawn_dev 0/1, only .DS_Store ignore).
New shared baseline is `8121e214e062077b88c39d0ccd861c55efaecc2c`.
No old shawn_dev-specific work or private HAP inputs were imported. The accepted
CPU tree was published as fce0af9; the CUDA release was published as 9074817.
Toolchains/runtimes were installed only in task-local locations
and the isolated CPU container; no global PATH, driver or security changes.
