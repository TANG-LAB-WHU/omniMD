# Release scope

The accepted CPU baseline fce0af9f6a49592a48c13883dfc3aa0d3eb4a28e was published on
shawn_dev. Following separate user authorization, the CUDA release
[907481708ae5f3df4228a9fb5d5b1110c39f5e23](https://github.com/TANG-LAB-WHU/omniMD/commit/907481708ae5f3df4228a9fb5d5b1110c39f5e23)
was published to the same branch with an ordinary non-force push. Its remote
hash was verified after publication.

## Local native CUDA cache follow-up

The published base is a74804d8cbba86c1a2563dbbca6444059e740903 (CUDA release plus
status-only documentation). Native commit b6f36dd was integrated unchanged as
bdf6048. This follow-up remains local: ten native implementation/profile/test/doc
files, the final acceptance script and compact report, and README/release/workflow
notes. No engine/shared-contract changes, dependency additions or GPU training.
CPU remains the default. Generated models, frames, trajectories and logs are untracked.

The [final-version receipt](evidence/native-cuda-infra-final.json) includes the
conservative coordinate-scale guard. The two cases each used five warmed
same-frame evaluations, with identical model/frame hashes for old and new paths:

| Case | Published CUDA ms/eval | Final CUDA ms/eval | Final-build CPU ms/eval | Old/new CUDA ratio |
| --- | ---: | ---: | ---: | ---: |
| Original 216-atom synthetic | 45.442 | 2.342 | 57.312 | 19.40 |
| Public model, original 192-atom water lattice | 46.572 | 4.237 | 54.090 | 10.99 |

The ratios in the last column compare the old and new CUDA paths, not GPU versus
CPU. Final-build CPU uses its unchanged uncached neighbor algorithm. Its elapsed
ratio to final CUDA was 24.47 / 12.77 for these two cases, which is not a pure
CPU/GPU kernel comparison. This change removes repeated host neighbor searches;
CUDA kernels are unchanged. Timings are means of one five-evaluation batch, not
statistical benchmarks. Desktop graphics were not stopped or controlled.

CUDA construction (including runtime compilation/model upload) was old/new
915/943 ms for the synthetic case and 731/720 ms for water. Final first evaluation
cost 102.2 / 62.8 ms, including initial candidate construction and runtime warm-up;
those first-evaluation timings were not captured by the old checker. These are
per-process initialization observations, not machine-cold timings.

For the same 192-atom, 20-step NVE job (0.1 fs; per-step output; checkpoint at 10),
whole-process time was 2.192 s old CUDA, 1.450 s final CUDA and 1.310 s CPU.
Thus the whole-job old/new CUDA ratio was only 1.51, and final CUDA remained
about 10.7% slower than CPU for this short job. These times include startup,
loading/compilation, serialization and checkpoint fsync. The 10-step resume took
1.249 s including its own initialization; it is not a steady-step timing.

All old/new CUDA trajectory fields matched exactly. CPU/final-CUDA maximum
E/F/virial differences were 0 / 1.44e-15 / 1.78e-14; position/velocity differences
were 5.55e-17 / 1.39e-17. The resumed snapshots and final decoded checkpoint state
matched continuous final CUDA exactly. Normal-scale timing reused one candidate
build; five moving MD frames reused it, translation by 0.25 Angstrom rebuilt,
and the subsequent 0.01 Angstrom motion reused the new list. Each matched fresh
CPU evaluation. The focused CPU cache test also passed the large-coordinate
fallback, cutoff-entry, threshold, cell/type/wrapping and error cases.

The one acceptance window ended at 2026-10-14T07:53:21.7983377Z with no remaining
validation process, CUDA errors or retries. This is bounded software acceptance,
not chemical accuracy or long-time stability. Reproduction is in the
[workflow](WORKFLOW.md#final-native-cuda-cache-acceptance); the earlier native
report explicitly retains its pre-final-guard timing provenance.

## Published CUDA release

At commit 9074817, the delta from the published CPU commit is limited to 32 files:

- README.md and ROADMAP.md: current capabilities, measured limits and status.
- mlmd/VALIDATION.md and this release scope: acceptance and exact boundaries.
- Nine mlmd/deepmd files: CUDA/CUPTI runtime verifier, explicit worker device
  selection, combined-evaluation integration, standalone CLI, focused checks,
  documentation, saved-output comparator and public-fixture receipt.
- Four mlmd/engine files: native-cuda feature, shared device routing, static
  capabilities and CLI documentation. CPU remains the default for every size.
- Thirteen mlmd/native files: optional Driver/NVRTC inference implementation,
  CUDA kernels/evaluator/readiness command, feature manifest, docs/notices,
  reproducible benchmark script and compact readiness/numerical reports.
- mlmd/evidence/native-cuda-cli.json and mlmd/scripts/validate_native_cuda_cli.py:
  the short native MD/checkpoint acceptance summary and reproducer.

There are no new dependency or lockfile changes, shared data-contract changes,
root simulation-code changes, pretrained weights or generated training datasets
in this delta. No HAP inputs/results, drivers, libraries, containers, checkpoints
or raw logs are added. Existing native training remains CPU-only; GPU training
is not implemented.

Both inference paths have actual fixture-based CPU/GPU numerical agreement.
Native CUDA additionally has short unified NVE/restart acceptance; official
C API CUDA has verified direct kernel evidence and GPU batch/MD smoke checks.
Only the documented subsets/runtimes were checked. Some work remains on CPU.
The C API six-atom case was 8.27 times slower on GPU including verification and
synchronization. Native small cases were about 8.4 times slower; its separate
216-atom sample had a CPU/GPU time ratio of 1.155 with desktop-load limitations.
These are separate bounded measurements, not comparable backend benchmarks,
general speedup claims, chemical accuracy or long-time stability validation.

C's f839cef validation files were imported exactly as 03965e7 without importing
the operational-only 46d913b parent into candidate history. Its source hash stays
in the receipt as provenance. The final integration checks replay saved outputs
without GPU access and rebuild the feature-enabled CLI/static capabilities;
no GPU acceptance was repeated during this evidence integration.

Review the exact delta with:

    git diff --name-status fce0af9f6a49592a48c13883dfc3aa0d3eb4a28e 907481708ae5f3df4228a9fb5d5b1110c39f5e23

Reproduction entry points (runtime/model setup and limits are in linked docs):

- [C API report](deepmd/CUDA_VALIDATION.md): offline saved-output comparison via
  `python mlmd/deepmd/scripts/compare_cuda.py CPU_RESULT GPU_RESULT`; actual
  evaluation requests/build/runtime details are in [deepmd/README.md](deepmd/README.md).
- [Native CUDA](native/CUDA.md): build/evaluator and same-model benchmark recipe.
- [Native unified CLI](engine/README.md#explicit-native-cuda-inference): opt-in
  build, device selection and `validate_native_cuda_cli.py --run-gpu` replay.

## Published CPU baseline contents

The remaining sections record the scope and ancestry procedure for the already
published CPU baseline, rather than a request to repeat its publication.

The change starts from upstream main
761b2484feee132426183c9b66ff2b126d1f7c4a. The existing omniMD source, MIT license,
public examples, benchmark/test fixtures and documentation are retained.
New or changed content is limited to:

- Root workspace feature selection, README, ROADMAP and ignore rules.
- mlmd/contract: explicit shared Frame, metadata and E/F/virial API.
- mlmd/deepmd: official C API loading/binding, CLI, tests and documentation.
- mlmd/native: bounded se_e2_a inference, safe model conversion, native training,
  focused tests, tiny original synthetic fixtures and reproducible Python scripts.
- mlmd/engine: explicit data import/prediction, minimization and MD CLI, tests,
  one synthetic NVT request example and format documentation.
- mlmd/scripts and mlmd/evidence: reproducible checks and compact numeric summaries.
- omnimd-core and omnimd-sim: optional dependency boundaries, fallible MLIP System
  dynamics bridge, shared minimization policy and required integration support.
- Cargo manifests/locks, algorithm/format documentation and third-party notices.

The two frozen graph profiles are weight-free topology/structural metadata,
about 210 and 230 kB. They are necessary for the bounded importer and retain
separate DeepMD LGPL provenance and license texts. They are not executable
pretrained models. The synthetic fixtures are original test data, not physical
models. Compact evidence summaries are not raw runtime or terminal logs.

Outside mlmd/, the exact changed code paths are:

    omnimd-core/Cargo.toml
    omnimd-core/src/energy/global/mod.rs
    omnimd-core/src/energy/mod.rs
    omnimd-core/src/sys/config/cells.rs
    omnimd-core/src/sys/interactions.rs
    omnimd-core/src/sys/mod.rs
    omnimd-core/src/sys/system.rs
    omnimd-sim/Cargo.toml
    omnimd-sim/src/md/mlip.rs
    omnimd-sim/src/md/mod.rs
    omnimd-sim/src/md/thermostats.rs
    omnimd-sim/src/min/mlip.rs
    omnimd-sim/src/min/mod.rs
    omnimd-sim/src/min/steepest_descent.rs
    omnimd-sim/src/output/mod.rs
    omnimd-sim/src/velocities.rs

The remaining root changes are .gitignore, Cargo.toml, README.md, new ROADMAP.md,
and untracking ls_remote_output.txt. No omnimd-input, legacy research directory,
root example/benchmark data or old TorchPotential implementation is modified.
The exact complete manifest is reproducible with:

    git diff --name-status 3440f605dde61b01e894f6532693612bc57c0aaa shawn_dev

## Exclusions and history boundary

No user model, private research data, HAP input/result, pretrained model binary,
generated training dataset, runtime/toolchain, target directory, Docker image,
credential file or raw operational log is intentionally added. Local generated
fixtures, build products and runtime outputs remain outside the tracked tree.
The inherited ls_remote_output.txt public refs dump is removed from the candidate
index and ignored while its local file is retained. It still exists in inherited
upstream Git history; this delivery does not rewrite or audit all upstream history.

The bounded publication check covers tracked candidate paths, new branch blob
sizes and common credential signatures, without reading external credentials.
It is a scope check, not a general security certification or a repeated source audit.

## Preserve ancestry without restoring old functionality

The observed remote shawn_dev is
3440f605dde61b01e894f6532693612bc57c0aaa. Its only change from upstream main is a
.DS_Store ignore line. The new implementation was developed independently from
main; no old shawn_dev-only implementation is needed.

After committing and testing the candidate, an explicit ours-strategy merge of
that exact old remote commit can preserve its ancestry while leaving every
candidate file unchanged. Verify both merge parents, the old remote as an
ancestor, and exact equality of the pre/post-merge tree hashes. This is
`git merge -s ours`, not the conflict-resolution option `-X ours`.

Once separately authorized to publish, first check current refs again:

    git ls-remote --heads origin main dev shawn_dev
    git merge-base --is-ancestor 3440f605dde61b01e894f6532693612bc57c0aaa shawn_dev
    git push origin shawn_dev:shawn_dev

Only use the ordinary push when the verified remote tip is contained in the
candidate. If the remote has moved, stop and review that new commit first.
Ordinary push rejects unseen concurrent remote changes; no force push is needed.
Do not push any other branch or tag as part of this release.


## Local engine preflight and stepwise guide follow-up

Based on local cache candidate 7be1d97; not published. This tranche changes only
engine dataset IO/CLI/tests, repository integration docs, one stdlib tutorial
script and its compact receipt. `native/**`, `deepmd/**`, shared contracts and
Cargo dependencies are unchanged. No GPU calls or new installs were made.

- `inspect-dataset` validates the complete JSONL without loading a model, hashes
  the original bytes in that pass and reports frame/species/PBC/label coverage.
  Source hashes are declarations; missing E/F coverage is not trainer approval.
- [The pinned feature matrix](FEATURE_MATRIX.md) compares official DeepMD 3.1.3
  with this implementation and marks concurrent C API parameter/batch/atomic
  output and native CUDA kernel work as pending integration.
- [The English stepwise guide](GETTING_STARTED.md) covers setup, conversion,
  C API/native model loading, data, training, minimization, MD/restart, explicit
  CUDA selection and troubleshooting. Root README links the complete workflow.
- [CPU tutorial acceptance](evidence/cpu-tutorial.json): 48 original synthetic
  frames imported with E/F/V preserved; 36 training and 12 explicit held-out IDs;
  readout rank 10; held-out E/atom RMSE 0.129577 -> 6.72e-16 eV and force RMSE
  0.0147253 -> 3.17e-17 eV/A. Ten CSVR steps and resume from step five reproduce
  every corresponding snapshot and final decoded checkpoint exactly.
- Focused offline dataset tests pass (six, including new byte/hash/coverage and
  malformed-input/no-overwrite checks). Existing XYZ reader tests also pass.
  Release CPU engine/evaluator/trainer builds pass; existing legacy deprecation
  warnings remain. No GPU performance or chemical accuracy claim is added.

Generated model weights, data, raw logs and checkpoints remain outside Git.
Publishing this tranche and any pending B/C integrations requires a separate
scope report and publication authorization.


## Local integrated cooperative CUDA and C API batch follow-up

Unpublished. Integration lineage after A's 9eaf284:

- B source b6b7aa7bc239702dafdefb559d758efd4c357eaa -> 887c52d:
  native cooperative fitting/descriptor kernels, persistent host workspace and
  cached type uploads, read-only resource/profile counters, native docs/evidence.
- C source 1aee39ccc64de849eb6fe6aac6d264d786005433 -> 3be7687:
  adapter-local homogeneous multi-frame C calls, explicit frame/atom parameters,
  optional atomic E/virial, standalone model inspection and no-clobber output,
  tests, comparison tooling and English backend guides.
- 9b8d184 records accurate adapter-versus-engine capabilities and parameterized
  acceptance provenance/throughput. This is the exact built/tested runtime.
  Shared Frame/Evaluation/Potential, CPU default and Cargo dependencies are unchanged.

C API integration passes 17 library plus four standalone CLI tests. Native CUDA
packing/cache and native solver library tests pass (three CPU-safe tests); B's
broader original acceptance is retained separately. Final release engine, native
CPU evaluator/trainer, CUDA checker and C API executable build offline. The final
CPU tutorial passes all ten stages, including 48-frame labeled extxyz import,
inspection, 36/12 readout fitting, batch assessment, CSVR NVT and exact restart.
See [its integrated receipt](evidence/cpu-tutorial-integrated.json).

The new C API numerical paths remain **pending real-runtime acceptance**. Earlier
scalar CPU/GPU receipts do not validate batched/parameter/atomic calls. These
features are available through `omnimd-deepmd` and `DeepMd::evaluate`; engine MD
and `batch-eval` retain parameter-free total E/F/V and one frame per call. The
feature matrix and guides make that distinction explicitly.

### Final integrated native GPU measurement

[The final receipt](evidence/native-cuda-integrated-fit.json) fingerprints all
inputs and binaries. The paired baseline is 9eaf284 (native code identical to
7be1d97); both versions use Rust 1.99 GNU release mode on the same Windows host.
One bounded measurement, five warmed calls per case; no statistical/general
speedup claim. CPU geometry/final reductions and per-call transfers remain.

| Same-model case | Previous CUDA ms/eval | Integrated CUDA ms/eval | Previous/integrated |
| --- | ---: | ---: | ---: |
| Original synthetic 216 atoms | 2.56860 | 0.73486 | 3.495 |
| Public converted 192 atoms | 4.52998 | 1.21122 | 3.740 |

These are complete warmed EFV calls, not kernel-only timers. Construction and
first evaluation are separately recorded. Complete MD used the same 192-atom
public-model frame, 20 NVE steps at 0.1 fs, seeded 300 K velocities (20261014),
COM removal, one snapshot per step and an interval checkpoint at step ten:

| Complete job | Wall time ms | Steps/s including startup and IO |
| --- | ---: | ---: |
| Previous CUDA | 661.758 | 30.223 |
| Integrated CPU | 1352.116 | 14.792 |
| Integrated CUDA | 411.725 | 48.576 |

The complete old/new CUDA ratio is 1.607; CPU/new CUDA is 3.284 for this short
workload. Do not substitute warmed inference ratios for MD throughput. Each
process includes loading/compilation, output and checkpoint fsync. Ordinary
desktop graphics were left active; neither runtime construction nor first
inference is a machine-cold driver-cache measurement.

Old/new CUDA snapshots have zero differences. Integrated CPU/GPU maxima:
positions 5.55e-17 A, velocities 1.39e-17 A/fs, energy zero, force 1.44e-15 eV/A,
virial 1.78e-14 eV. Maximum energy drift is 5.47493e-5 eV for all three runs.
Resuming ten steps from the step-ten checkpoint gives identical decoded
snapshots and final checkpoint within the tested CUDA binary/environment.
The moving/rebuild sequence yields cache flags true/false/false/false/false/true/false
and passes independent native CPU parity. No GPU failures or retries occurred;
all verification processes exited and the window was released at 07:50:26 UTC.

This evidence establishes the scoped implementation/IO consistency, not chemical
accuracy, long-time stability, GPU training or full DeepMD compatibility. No
new dependencies, drivers, system settings or research inputs were used. Only
source, documentation and compact receipts are committed; generated weights,
data, trajectories and operational logs remain outside Git. No public push is
part of this integration.

## Local CPU model-deviation workflow

Adds engine-only ensemble loading/streaming, population E/F/virial statistics,
optional candidate flags, exact consumed-dataset provenance and focused tests.
Shared/native/deepmd interfaces and dependencies are unchanged. The
[replay receipt](evidence/model-deviation.json) covers three original models,
12 nonperiodic/triclinic frames, independent Python statistical oracles and a
selected two-step CPU NVE job. C API ensemble runtime validation remains pending;
there is no GPU ensemble route or automatic labeling/training. Code, English
guides, script and compact receipt are local only; generated weights/data/logs
are excluded. Nothing in this tranche has been pushed.

## Local CUDA-assisted frozen-readout integration

Native-only source `dc5449edb165780595abf501a91c595c2c4d4b5b` is integrated unchanged
as `65a7192c83515afe4c2d039644c6890e493baa1f`. It adds explicit feature-device
selection, resident CUDA one-hot readout features, device guards, focused tests,
replay, English docs and a compact [B GPU receipt](native/fixtures/cuda-readout-report.json).
QR, reference evaluation and assessment stay CPU; full-network GPU training
rejects. The tiny measured case was slower on CUDA. A's integration validation
is CPU-only and does not replace or rerun that GPU receipt. Top-level capabilities
and guides distinguish this subset from full-network GPU training. No push.
