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

## Local dataset selection and CPU training integration

Engine-only `select-dataset` adds explicit hash-bound ID selection, complete input
validation before new output creation, bounded buffering, preserved frame/label
values, source-order reindexing and a completion receipt with immediate ancestry.
New files are the selection module/tests, English guide, stdlib workflow replay
and compact [evidence](evidence/dataset-selection.json); shared/native/deepmd
contracts and dependencies are unchanged. Focused tests and actual CLI flow pass:
six candidate records, assessment, three existing-labeled records, two training /
one validation, rank 10. The integrated `dc5449e` readout trainer ran on CPU.
Four additional native device tests and a feature-enabled engine check passed
without a GPU context; B's existing GPU receipt is retained separately. Disk
failures can leave an incomplete new bundle; no atomic-directory guarantee or
automatic labeling is claimed. No generated weights/data/raw logs are committed.

## Local C API default policy and engine parameter propagation

C source `ce27786093dfa3ad5873075d703b8a11fd050b51` is integrated as
`87abf57e69888e0430ea762a71b373e6938ebf2f`, excluding diagnostic `c8e849e`.
It records real CPU TF batch/parameters/atomic and PT explicit/default/Python
comparisons. The TF fixture only received its official declared type-map
metadata; weights/operations remain unchanged, with provenance retained.

Engine additions are confined to engine data/context/CLI/checkpoint code, tests,
English guides and a reproducible CPU replay. Dataset v2 introduces explicit
parameter layout/values without changing shared Frame/Evaluation/Potential.
NPY import, serial batch assessment and fixed-context MD/checkpoint v3 are
implemented; v1 data and parameter-free trajectory/checkpoint behavior remain
compatible. Native and CUDA engine parameter routes, parameterized minimization,
ensembles and native training reject. No new dependency or automatic fallback.

[Actual CPU evidence](evidence/engine-parameters.json): TF engine vs saved C API
E/F/V differences are zero; TF vs Python virial differs by 2.22e-16 eV. PT
explicit/default comparisons differ at most 8.33e-17 eV in virial. Each TF/PT
moving MD frame agrees exactly with a corresponding standalone sequence.
TF four-step NVE and step-2 continuation are bitwise identical. PT restart has
zero coordinate difference, force/virial differences up to 2.78e-17 / 1.11e-16,
and velocity difference up to 2.65e-23 Angstrom/fs. The same first-versus-later
call difference reproduces on an identical geometry in standalone PT C API;
PT restart is numerical parity, not bitwise acceptance. The initial failed exact
criterion and follow-up diagnostic are retained in local runtime artifacts.

Twenty CPU workflow steps and 29 affected engine checks passed; C's integrated
23 library / 5 CLI tests passed. The reused GPU-free, network-disabled build
container was restored to its original stopped state after all work completed.
No model weights, generated datasets, raw logs, HAP files or diagnostics are
included in the commit. No publication is authorized by this local integration.

## Local bounded CUDA full-network training integration

Native source `f4dbf62bb4acb97e8f7e12e62de625e08af07a05` is integrated as
`02b5457`, after the engine parameter commit `c787afc`. Its native-only scope adds
FP64 all-neural-parameter E/F mixed gradients and Adam, strict joint bounds,
CPU-v1/CUDA-v2 checkpoint compatibility checks, original fixtures, tests, English
documentation and compact evidence. Normalization, cutoffs, neighbor selection,
type map and atomic offsets remain fixed. Virial labels are assessment only.
CPU inference/loss seeds, geometry, stable reduction/clipping and assessment
remain in the full training path. There is no new dependency or implicit device
selection. CPU is the default; the sibling trainer needs an explicit CUDA build.

The final integration changes outside native are capability reporting, English
coverage/stepwise guides and a short reproducible acceptance script/receipt.
[Final release-binary evidence](evidence/native-cuda-full-integrated.json) uses
24 original synthetic frames (18 training / 6 validation), 112 parameters and
six steps through the engine's `train-native` command. Every parameter changes;
training objective is 8.83036 to 0.877721 and held-out objective 10.1403 to
0.692830. These are software-acceptance values, not chemical accuracy evidence.
Continuous six steps and three-plus-three restart have exactly equal model,
Adam moments, optimizer step, split and execution identity. Export/reload E/F/V
differences are all zero. Changing the checkpoint backend to CPU rejects without
creating a training output directory. CUDA execution reports the RTX 4090,
FP64 and the same compiled PTX hash as B's acceptance. The GPU window ended at
2026-10-16 11:14:09 UTC; no retries or driver changes were needed.

Seven affected CPU-safe checks passed; three ignored GPU tests were not rerun.
The final feature-enabled engine/evaluator/trainer release build used the locked
offline cache. B's independent gradient, 300-step and 150+150 evidence remains
[separate](native/fixtures/cuda-full-training-report.json). Its 3.46x result is
only a warmed 64-atom full-gradient stage; tiny complete training is slower.
This integration makes no new performance claim. Linux CUDA and representative
physical training accuracy remain unverified.

Only source, English documentation, original tiny fixtures already in B's commit
and compact receipts are in Git. Generated inputs/weights/checkpoints, binaries,
raw logs and HAP material are excluded. This follow-up is local and unpushed.

## Local external-label return workflow

The engine-only tranche after `e764502` adds `prepare-labels` and `import-labels`.
Requests bind exact source/return SHA256 values. Prepared identities bind source,
ID/index, ordered coordinates/species/cell, units/convention and v2 parameter
layout/values/policy using a versioned binary hash. Returned records echo that
context; complete coverage, finite E/F and exact existing-label agreement are
required. Optional virial fills an absence or preserves existing data. Source
frame order, IDs, indices and parameter values remain unchanged. New bundles
contain exact request/return bytes and a last-written completion receipt; no
existing output is replaced. Source/return/output sizes are bounded. No dependency,
shared model contract, native backend or GPU implementation changed.

[Final CPU evidence](evidence/external-labels.json) records 14 CLI steps using
saved original synthetic software fixtures: 12-frame model disagreement, six
selected candidates, label preparation/return, inspection and five-training /
one-validation readout fitting. E/F-labeled frames increased from three to six.
Twelve frame identity hashes matched an independent Python binary-encoding
implementation. V2 parameters/order were preserved in a separate IO-only case.
Five CLI refusal cases produced no output directories. Nineteen affected Rust
checks passed, including four new labeling tests with 24 identity/shape/conflict
cases plus malformed-file, hash, existing-output and byte-bound checks.

The original synthetic held-out E/atom RMSE changed from 0.0432649 to
1.11022e-16 eV and force RMSE from 0.00247869 to 6.59758e-18 eV/angstrom; the
readout design rank is 10. Native-v1 reload E/F/V differences are zero. These
numbers establish software consistency on manufactured labels, not chemical
accuracy or independent physical generalization. The versioned parameter case
only establishes IO/identity preservation; it does not validate an external
parameter-dependent calculator or add native v2 training.

The replay set `CUDA_VISIBLE_DEVICES=-1`, used the default CPU build/trainer and
launched no external scientific calculation. Calculator metadata/artifact hashes
remain unverified declarations; hidden model defaults are not resolved by the
file workflow. Job submission, automatic labeling/training and an unattended
active-learning loop remain outside scope. Generated labels/data/weights/logs
remain outside Git. All changes are local; no public push or HAP access occurred.

## Local optional virial-loss integration

Native source `d12fdb5c36a4fa74a10fb14bbb489a664cae0a2c` is integrated as
`7d3d472` on the clean external-label candidate `69cdf12`. The native-only commit
adds explicit positive virial weight/per-atom scale, all-nine-component row-strain
loss and shared analytic seeds for full CPU/CUDA training. Enabled loss uses
checkpoint v3 with an immutable loss contract. E/F-only CPU-v1 and CUDA-v2 schemas
remain compatible; readout QR remains E/F-only. Architecture/model/data bounds
and native-v1 inference formats are unchanged. No dependency or CUDA kernel
arithmetic was added; the existing mixed-derivative kernel receives the combined
force/strain direction. CPU stages remain explicitly documented.

[Final release-binary acceptance](evidence/native-virial-integrated.json) uses
24 original synthetic frames (18 training / 6 validation). All 112 neural
parameters change in six CUDA steps. Three-plus-three continuation is exactly
equal in model, moments, step, split, configuration and execution/loss identity,
except the intentionally different requested additional-step count. Six-step
CPU/CUDA parameter differences are at most 1.39e-17, moments 2.22e-16, and exported
E/F/V 4.44e-16 / 2.22e-16 / 2.22e-16. Reload E/F/V differences are zero. The prior
E/F-only CUDA-v2 step-3 checkpoint resumed to its saved step-6 state exactly.
Changed loss weight and changed backend both reject without output directories.
Five affected CPU-safe tests and the locked offline release build passed.

The short joint fit's held-out total objective changed from 11.18699 to 0.748744;
virial-per-atom component RMSE changed from 0.102306 to 0.0239598 eV/atom. These
are software fixture outcomes, not scientific accuracy evidence. B's independent
derivative checks, historical CPU-v1 compatibility and 300-step/150+150 checks
remain [separate evidence](native/fixtures/virial-training-report.json) and were
not broadly repeated. The GPU window ended 2026-10-16 11:38:34 UTC without a retry.

B's tiny 300-step joint-loss training process measured 619 ms CPU versus 1816 ms
CUDA, including initialization, input IO, updates, checkpoints, assessment,
reload verification and export. This integration adds no performance claim.
Earlier E/F-only warm-gradient timings do not establish joint E/F/V acceleration.
Linux CUDA, large-model throughput and external-label physical accuracy remain
unverified. Files committed outside native are capability reporting, English
README/feature/usage/release/roadmap updates, the short replay and compact receipt.
Generated data/models/checkpoints/binaries/raw logs remain outside Git. No HAP
material, system configuration changes or public push are included.

## Local optional reverse CUDA integration

Native source `f33d332adcafcc2a360b22e9605d6da43870eaf9` is integrated as
`f4e794af508676e2389d6c76c6d8a275089375a7` on `78cfd034`. Explicit
`config.cuda_gradient="reverse"` enables cached forward directional pairs and
reverse accumulation for the existing E/F/optional-virial neural gradients.
Hyperdual and CPU defaults remain unchanged. Native checkpoint v4 binds loss,
executor, device, driver and compiled PTX; nonzero-step executor switching rejects.
No shared format, dependency, model architecture or existing work bound changes.
The 64 MiB planned payload ceiling excludes context/code/driver allocations.

[Final integrated release acceptance](evidence/native-reverse-integrated.json)
uses 24 original synthetic frames (18 training / 6 validation), six joint-loss
CUDA steps and a 3+3 continuation. All 112 neural parameters changed. Continuous
and resumed models, moments, step, split and loss/execution identity are exactly
equal; fixed model fields are unchanged. CPU/reverse maximum differences are
1.11e-16 for parameters, 2.78e-16 for moments and 4.44e-16 / 3.89e-16 / 4.44e-16
for exported E/F/V. Reload differences are zero. Changed device, loss and executor
requests reject without training output directories. Five CPU-safe checks and
the final locked offline release build passed. The GPU window was released at
2026-10-16 12:07:58 UTC, with no remaining omniMD process or device-failure retry.
Legacy checkpoint regression is retained in B's report and prior A acceptance.

[B's separate native evidence](native/fixtures/cuda-reverse-training-report.json)
contains all-parameter derivative parity, 300 steps and exact 150+150 continuation.
Its three rotated fresh-process medium medians are CPU 2963 ms, hyperdual 1349 ms
and reverse 877 ms; tiny CPU/reverse times are 615/1573 ms. These timings include
input, initialization, training, checkpointing, assessment and export/reload.
A's short integration adds no timing result. CPU primal inference/loss seeds,
geometry, stable reduction, clipping and assessment remain. Neither production
scaling, broader model support nor physical accuracy is established.

Outside the native commit, the tranche contains capability reporting, English
README/guide/matrix/release/roadmap updates, the short acceptance script and a
compact receipt. Generated data, weights, checkpoints, binaries and raw logs
remain outside Git. This is a local unpushed candidate with no HAP input or
system/driver changes. The next proposed optimization is bounded reuse of
immutable frame geometry, explicitly excluding current-loss directional rows.

## Local fixed training-geometry cache integration

Native source `bbfecc639fc045de778344e3ebf3ec97ae07a41c` is integrated as
`a6a841d245107fa48630f9c8d713e97063f7194c` on stable `94427ef`. Explicit
`config.fixed_geometry_cache=true` enables eager fixed-frame geometry reuse for
both CUDA gradient executors; default false and CPU-only rejection remain explicit.
The 64 MiB retained host-payload cap is separate from device-buffer limits and
excludes allocator overhead, original input data and temporary construction.
Exact identity mismatch invalidates the pool and stops training. Current-model
predictions, losses and directional rows are rebuilt each step; only selected
training-frame neighbors, full periodic displacements and normalized R/J persist.
Assessment remains uncached, with per-step GPU input uploads unchanged.

The setting may toggle on resume without a schema or numerical execution-identity
change. Cache arrays/stats are rebuilt, not checkpointed. Dataset, split, objective,
CUDA device/driver/PTX and backend constraints still apply. No dependency or
shared model/dataset format was changed and no supported limit was raised.

[Final integrated acceptance](evidence/native-geometry-cache-integrated.json)
uses the existing 24-frame fixture, 18 training frames, 112 neural parameters and
six reverse E/F/V steps. Continuous and 3+3 cached training agree exactly, as do
on-to-off continuation and off-to-on continuation from the actual prior `94427ef`
step-3 checkpoint. Final state equals its saved step-6 state, excluding only the
runtime flag and additional-step count. CPU/reverse maximum differences remain
1.11e-16 parameters, 2.78e-16 Adam moments and 4.44e-16 / 3.89e-16 / 4.44e-16 E/F/V;
reload differences are zero. Changed loss/device/executor requests still reject
without output. The cache retains 221,414 bytes, with 108 hits for six steps and
54 for the three-step restart. Fifteen affected CPU-safe checks and the final
locked offline release build passed. GPU was released at 2026-10-16 12:31:39 UTC,
with no remaining omniMD process and no device-failure retry.

[Native evidence](native/fixtures/fixed-geometry-training-report.json) separately
covers both executors' gradients/Adam/cache-toggle exactness, historical checkpoint
compatibility and alternating fresh-process benchmarks. Three-run medians for
466-parameter, 64-atom, eight-frame, 100-update synthetic E/F/V training are
865.5/818.4 ms cache off/on for a cluster and 6946.3/956.9 ms for its triclinic
periodic variant. Those include construction, initialization, updates, assessment,
checkpoints and export/reload. This final integration did not repeat them. The
periodic gain reflects avoided repeated CPU image enumeration, not generic GPU
acceleration, physical accuracy or production-model scaling.

Outside the native commit, this tranche updates capability reporting, the English
README/usage/matrix/roadmap/release guides, the short acceptance script and compact
receipt. Generated datasets, weights, checkpoints, binaries and raw logs remain
outside Git. Work is local and unpushed; no HAP input, installation or system/driver
change is included. Linux CUDA, large-model throughput and broader architectures
remain unverified or unsupported as recorded in the feature matrix.

## Local deterministic minibatch integration

Native source `3e7091c143ffa424c935d86adbd9ecb7f1e97d5c` is integrated as
`5be7e49ccfd8f3dbcabbf75dab47c2a262487cdb` on stable `b8ce8fa`. Optional explicit
batch size, seed and sequential/shuffled order apply to CPU and both CUDA full
trainers. Actual-count normalization retains short tails, clips after the batch
sum and advances Adam once. Version 5 checkpoints validate the canonical next-batch
sampler together with loss/config/data and execution identity. Full-batch defaults
and old v1-v4 schemas remain unchanged. Cache toggles are runtime-only; numerical
backend or batch-setting changes reject. No dependency, shared dataset/model
format, physical model or supported resource limit changed.

[Final release CLI acceptance](evidence/native-minibatch-integrated.json) uses the
existing 24-frame synthetic fixture, 18 training / six validation frames, reverse
CUDA and seven updates in batches 7,7,4,7,7,4,7. All seven saved samplers agree with
an independent Python integer reference. The report records 43 processed frames,
two partial batches and 43 cache hits. Restarts at steps 2, 3 and 4 are exact in
model, Adam, sampler and remaining state after excluding only additional-step
count and the runtime cache flag. Both cache-toggle directions pass. CPU maximum
parameter/moment difference is 6.66e-16; exported E/F/V differences are 8.88e-16 /
6.66e-16 / 8.88e-16, with zero reload differences. Changed batch size/seed/order,
missing sampler and corrupt cursor all reject before output. The actual prior
full-batch v4 step-3 checkpoint continues to its saved step-6 state exactly.
Ten focused CPU-safe tests and the locked offline release build passed.

The GPU window ended at 2026-10-16 12:58:46 UTC with no remaining omniMD process
and no failure retries. [B's native receipt](native/fixtures/minibatch-training-report.json)
separately records 29 default CPU tests, 42 feature-enabled CPU tests, a GPU check
covering both executors, 31-update CLI parity and actual historical v1-v4 step-150
to step-300 continuation. Those broader checks and benchmarks were not repeated.
This final integration adds no speed, convergence-equivalence or accuracy claim.

Outside native, changes are engine capability reporting, English README/guide/
feature-matrix/roadmap/release scope, the short replay and a compact receipt.
Generated datasets, models, checkpoints, binaries and raw logs stay outside Git.
The candidate remains local and unpushed, without HAP inputs or system changes.
Dataset streaming is identified as the next infrastructure gap only; it is not
implemented here. The 1024-frame in-memory limit, constant learning rate, native
architecture bounds and remaining CPU stages all persist.

## Local indexed dataset integration

Native source `be573fe0f91d323932c85e47448b0aace5950f32` is integrated as
`dc24b261440be7a17b88427006440c95445c96ba` on stable `8f80ee5`. Explicit top-level
`dataset_access="indexed"` requires minibatching and reads the existing v1 JSONL
format one frame at a time. Offset/length/ID/atom-count/line-hash indexing supports
training, full-partition assessment, geometry preparation and reload checks.
Storage mode is runtime-only; v5 sampling and numeric execution identity remain
unchanged. Default memory and readout paths keep their previous limits/behavior.

Indexed caps are 100000 frames, 16 GiB file, 256 KiB record and 64 MiB retained
index payload. Metadata, sampler, report identities and checkpoints remain O(N);
the payload budget does not include total process memory or construction/optimizer
workspaces. The 64-atom/model/work limits and independent 64 MiB geometry cap are
unchanged. Every accessed line is verified, with whole-file checks before each
checkpoint, after final metrics and before CLI publication. The Windows retained
handle denies writing/replacement; portable checks do not establish an atomic
snapshot against concurrent mutation on unvalidated platforms.

[Final integrated acceptance](evidence/native-indexed-integrated.json) reuses the
24-frame fixture and actual prior memory v5 checkpoints. Seven indexed reverse
updates and seven CPU updates exactly match their prior memory model, Adam,
sampler, all four initial/final partition metrics and exported model bytes.
Indexed-to-memory plus cache-off restart at step 2 and old-memory-to-indexed plus
cache-on restart at step 3 are exact. Reload differences are zero. Indexed without
minibatch, unknown mode and changed raw dataset bytes on resume reject before
output. The run records 19936 index bytes, 229 CUDA-path frame reads (211 CPU),
one decoded frame per read and nine whole-file verifications. These counters do
not represent total live allocations. Eleven focused CPU-safe tests and the
locked offline release build passed.

GPU was released at 2026-10-16 13:32:49 UTC with no remaining omniMD process and
no retry. A subsequent engine-only static capability wording correction received
an offline rebuild, a capability read and an exact one-step CPU indexed resume;
GPU work was not repeated and the trainer binary was unchanged. Both engine
hashes are retained in the receipt.

[B's separate capacity evidence](native/fixtures/indexed-training-report.json)
records 34 default CPU checks, 48 feature CPU checks, one GPU check covering both
executors, actual old v1/v5 compatibility and a 2051-frame input deliberately
whitespace padded to 128.19 MiB. That boundary run measured 12.83 MiB Windows peak
working set and 319722 bytes of retained index, with exact resume. It is neither
representative physical throughput nor a universal memory limit. The complete
100000-frame/16-GiB ceiling remains unbenchmarked. No capacity/memory/throughput
benchmark was repeated by this integration.

Outside native, files are capability reporting, English README/guide/matrix/
roadmap/release updates, the new short replay and compact receipt, plus the prior
minibatch replay's capability assertion updated for the new access limits. No
shared data/model schema or dependency was added. Generated models, datasets,
checkpoints, binaries and raw logs remain outside Git. Work remains local and
unpushed with no HAP input or system changes. The next proposed work is measuring
and removing only confirmed duplicate indexed startup validation, preserving all
integrity/publication checks; no follow-up implementation starts in this tranche.
