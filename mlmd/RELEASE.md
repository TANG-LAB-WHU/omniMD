# Release scope

The accepted CPU baseline fce0af9f6a49592a48c13883dfc3aa0d3eb4a28e was published on
shawn_dev. Following separate user authorization, the CUDA release
[907481708ae5f3df4228a9fb5d5b1110c39f5e23](https://github.com/TANG-LAB-WHU/omniMD/commit/907481708ae5f3df4228a9fb5d5b1110c39f5e23)
was published to the same branch with an ordinary non-force push. Its remote
hash was verified after publication.

Entries below record successive milestones. Earlier statements that Linux native
CUDA had not been validated describe those milestones; the later
[Linux platform acceptance](#local-linux-native-cuda-acceptance) records the exact
WSL2/container scope now tested. It does not expand C API or general Linux claims.

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

## Local indexed startup integration

Native source `cdf66b43e8899466d4c4cb4fb23a9c874af66a57` is integrated as
`958b362cd3fab0d22ac2c543ccd6503a8bbe30df` on stable
`798d2d496dee1471539a43418c31abe66f3e4ab0`. The trainer now reuses the layout
returned by its complete checkpoint/device validation, and the CLI removes its
immediately preceding duplicate validation. Public constructor and validator
retain their checks; trainer entry validates before CUDA setup or save callbacks.
There is no token or cache across calls. Every line identity/hash and every
publication whole-file hash remains. No request/dataset/model/checkpoint schema,
optimizer identity, resource ceiling, dependency or GPU kernel changes.

Reports add cumulative startup timings and an indexed frame-read counter at the
first-update boundary. Initial partition assessments are included in trainer
startup. Existing initialization timing stops before assessments. Cumulative
fields overlap and exclude process launch; they must not be summed indiscriminately.
See [step 20](GETTING_STARTED.md#20-interpret-startup-validation-and-io-counters).

[Final integrated CPU receipt](evidence/native-indexed-startup-integrated.json)
records three successful paths on existing 24-frame synthetic input: fresh seven
updates, historical step-6-to-7 continuation, and default memory. Fresh and resume
checkpoint contents, model bytes, four initial/final partition metrics and sampler
exactly match the saved pre-change outputs. Default memory agrees exactly; reload
E/F/V differences are zero. Frame reads fall 211 to 163 and 151 to 103, respectively;
whole-file verification counts remain nine and three. Startup frame reads are 72
and 48, excluding index construction. Zero steps, CPU geometry cache and malformed
resumed Adam moments reject before output. Nine focused CPU tests and the locked
offline release build passed. CUDA support was compiled but no CUDA context or
GPU test ran in this tranche. Existing GPU evidence remains historical.

[B's separate comparison](native/fixtures/indexed-startup-report.json) uses three
alternating baseline/candidate repeats with warm filesystem cache on the existing
2051-frame, 134414893-byte whitespace-padded synthetic input. Fresh startup median
945.07 -> 687.19 ms and process 2127.12 -> 1860.17 ms; resumed invocation startup
819.30 -> 557.27 ms and process 1518.39 -> 1261.83 ms. Each invocation saves 4102
logical frame reads while retaining four/three whole-file checks. The baseline
contains identical timing instrumentation and the original duplicate call chain.
This parsing/IO case does not establish physical-data, cold-disk, Linux or general
GPU training speed. The large comparison was not repeated by A. B records 35
default CPU tests and 49 feature-enabled CPU tests, with nine GPU tests ignored.

Publication scope for this tranche is six native files (trainer implementation,
CLI timing, indexed regression, replay, documentation and compact evidence), root
README/ROADMAP, the MLMD guide/feature matrix/release note, one short CPU-only
integration script and its compact receipt. No engine behavior/shared interface
changes occur. Generated datasets, models, checkpoints, binaries and raw logs
remain outside Git. This candidate is local and unpushed, contains no HAP material,
and makes no system changes. No further feature starts before the consolidated
capability/gap report.

## Local learning-rate schedule integration

Native source `63ea457d7084d17dbd2db25e4c7ee42ca46740d5` is integrated as
`7fdb1e24709d79e949be9bc30ae8b410d9cf65ac` on stable
`e0decf3f8e64c8bad10f79323fa036511f2088fa`. Optional full-training constant or
exponential staircase schedules use globally completed optimizer updates:
`lr(k)=base*decay_rate^floor(k/decay_steps)`. Explicit schedules select v6 and
bind the base/schedule immutably. Absent/null retains constant behavior and
historical v1-v5 identities. The same host f64 scalar feeds CPU/CUDA Adam;
CUDA kernel/PTX sources, device/resource limits and shared model/data contracts
are unchanged. Readout QR does not accept the schedule.

[Final short CPU engine acceptance](evidence/native-learning-rate-integrated.json)
checks static capability reporting and five successful training paths on the
existing 24-frame synthetic input: scheduled seven-update indexed minibatch,
step-2 and step-3 continuation around the first decay, explicit constant, and
null-schedule resume of an actual historical v5 checkpoint. Rates for updates
1..7 are 0.005, 0.005, 0.0025, 0.0025, 0.00125, 0.00125, 0.000625. Both v6
continuations exactly reproduce every checkpoint field except the requested
additional-step count, exported model bytes and final train/validation metrics.
Constant matches old v5 except intentional schedule/schema identity; all four
metrics match. Null preserves exact legacy state. Reload E/F/V differences are
zero. Rate above one and changed/removed resume schedules reject before output.
Three engine CPU routing tests and the locked offline release build passed.
No GPU ran during final integration and no performance benchmark was repeated.

[B's separate native receipt](native/fixtures/learning-rate-report.json) records
40 default CPU tests, 54 feature-enabled CPU tests with nine GPU tests ignored,
plus actual CLI acceptance for CPU and both CUDA executors. It covers broader
fullbatch/minibatch/indexed/cache and restart permutations; maximum final
parameter/moment differences from CPU are 9.44e-16 for hyperdual and 8.88e-16
for reverse. Both compiled PTX hashes remained unchanged; GPU was released at
2026-10-16 14:10:49 UTC. These are bounded synthetic correctness results, without
DeepMD-default equivalence, convergence, physical accuracy or speed claims.

Publication scope for this tranche is the 20 native files in the source commit,
engine static capability metadata, root README/ROADMAP, MLMD guide/matrix/release
notes, the short engine replay and compact receipt. Native additions cover the
schedule implementation, checkpoint/report/Adam scalar integration, focused
regressions, documentation/provenance and synthetic summary evidence. No new
dependency or shared interface was introduced. Generated inputs, models,
checkpoints, binaries and raw logs stay outside Git. Candidate remains local
and unpushed; no HAP data or system changes.

The remaining main larger-GPU workload gap is a memory-bounded batched reverse
training executor with representative scaling evidence. Current parameter/width/
atom limits and host stages remain. The roadmap distinguishes those known
constraints from unmeasured throughput bottlenecks; no next feature was started.

## Local tiled CUDA training integration

Native source `cce7f8c447b22824593ad3b4df333d63932e44bc` is integrated as
`0937cb97733d4a5bb5987389636e9ccd893fbe7b` on stable
`6bb52955315304b1a35842e6065afc7219388805`. Optional `config.cuda_tiling` batches
independent frames into four reverse-gradient kernels per tile. Frame reduction
order, true optimizer-batch denominators, learning-rate/Adam boundaries and failure
atomicity are retained. Explicit reverse CUDA is required. v7 binds capacities
and `cuda_tiled_reverse_f64_v1`; absent/null preserves historical framewise v1-v6.
The tiled program is distinct; the framewise program identity remains unchanged.

No architecture bound is raised. At most 32 frames/1024 aggregate atoms per tile
are requested; effective capacity is reduced by the 50-million-term work policy,
64 MiB explicit device-buffer and 32 MiB host-tile budgets. Single frames remain
at most 64 atoms; widths at most 32 and parameters at most 2048. Payload exclusions
include context/module/local-memory, CPU workspace and other process overhead.

[Final engine acceptance](evidence/native-tiled-integrated.json) uses existing
24-frame original synthetic input (3/4 atoms), 18/6 split, seven scheduled updates
and minibatches 7,7,4,7,7,4,7. Three-frame/16-atom capacity packs up to three frames/
11 atoms. The 43 gradient frames use 19 tiles, 76 gradient kernels and 19 syncs,
compared with 172 kernels/43 syncs framewise. Numerical checkpoint content and all
four partition metrics match exactly after excluding executor metadata; model
bytes match. Step-2 v7 continuation switches indexed to memory and cache on to off,
reproducing final state except additional-step/cache fields. Reload differences
are zero. CPU tiling, changed capacities and framewise resume reject before output.
Four host-only guards/routing checks and the locked offline release build pass.

The initial harness incorrectly expected 12 atoms per tile for mixed 3/4-atom
frames. Both GPU training commands had completed successfully. The assertion was
corrected to use fixture bounds; completed requests/results were matched and
reused, with only the previously unrun continuation executed. There was no CUDA
failure or GPU retry. The GPU window ended at 2026-10-16 15:00:29 UTC with no
remaining omniMD process.

[B's separate receipt](native/fixtures/tiled-training-report.json) records 40 CPU,
56 feature-enabled host-only and three GPU tests, including raw-gradient/Adam
parity, failure atomicity and a 32-frame/1024-atom aggregate boundary. Three
alternating 128-update whole-process synthetic medians are 1071.17->550.51 ms
for 112 parameters/3-4 atoms and 4483.70->3567.77 ms for 466 parameters/32-64 atoms.
A did not repeat those benchmarks or broad permutations. No physical accuracy,
production scaling, new architecture or speed relative to DeepMD is claimed.

Publication scope is the 21 native source-commit files, engine static capability
metadata, root README/ROADMAP, guide/matrix/release notes, one short final engine
replay and its compact receipt. No dependency/shared model or dataset interface
changes. Generated weights/data/checkpoints/binaries/raw logs stay outside Git;
no HAP inputs or system changes. Candidate remains local and unpushed.

The next meaningful measured cost is current-model CPU prediction/loss seeds:
about 57% of the retained medium tiled gradient sample. GPU tile prediction/seeds
with reusable intermediates is the next proposed implementation target, subject
to independent derivative/restart/resource and whole-process checks. Existing
architecture limits and constrained tile occupancy remain separate gaps. No new
feature starts before reporting this candidate.

## Local GPU prediction/loss integration

Native source `36ee5cf4cc261437b6dcc03b182073dee72c5cae` is integrated as
`230279d2adb6151321720f27af20c45252f1f096` on stable
`f6030da6cef085e37bd2d135878c591218f30b2e`. Explicit `config.cuda_loss="gpu"`
requires reverse CUDA with tiling. It computes E/F/V and detached residual seeds
on GPU and reuses primal intermediates for mixed derivatives. Ten kernels execute
per tile with one synchronization; batch normalization/reduction/Adam semantics
remain. v8 binds loss mode and prior immutable identity. Absent/CPU loss keeps
historical v7 state/PTX. Geometry/scalars, gradient reduction/clipping, assessment
and exports stay on CPU. No dependency, architecture or shared schema expansion.

[Final engine acceptance](evidence/native-device-loss-integrated.json) uses the
existing 24-frame, 18/6-split synthetic fixture for seven scheduled updates.
Default CPU loss exactly reproduces the actual old v7 checkpoint, model bytes,
four metrics and PTX. Opt-in GPU loss has maximum parameter/Adam difference
8.8817842e-16, maximum partition-metric difference 2.2204461e-16. All 43 gradient
frames use GPU predictions in 19 tiles/190 kernels; host prediction time is zero
because work moved into GPU kernel timing. v8 step-2 continuation changes indexed
to memory and cache on to off, exactly matching final state except additional
steps/cache flag, exported bytes and final metrics. Reload E/F/V differences are
zero. Missing tiling, loss-mode changes and v7-to-v8 resume reject before output.
Four host-only configuration/resource/routing tests and the locked offline build
passed. No failure/retry occurred. GPU was released at 2026-10-16 15:37:37 UTC
with no remaining omniMD process.

[B's separate evidence](native/fixtures/device-loss-training-report.json) records
40 CPU, 57 feature host-only and four GPU tests. Independent maximum differences
are E/F/V 4.44e-16, detached seeds 8.88e-16 and mixed gradients 4.44e-15; the
nonconstant-embedding dimer parameter finite-difference error is 4.84e-9. Aggregate
1024-atom tiles, failure atomicity and boundary/restart cases passed. A did not
repeat the broad native checks or benchmarks.

Keep GPU loss opt-in. Three alternating 128-update synthetic whole-process medians
(including initialization, assessment and IO) are 604.53->756.03 ms for 112
parameters, a regression, and 3505.78->3060.90 ms for 466 parameters, a 1.15x ratio.
Tiny initialization rises about 329->523 ms; medium kernel/sync and upload costs
increase as CPU work moves. No physical accuracy, universal acceleration, larger
architecture or speed relative to DeepMD is claimed.

Publication scope is the 22 native source-commit files plus engine static capability
metadata, root README/ROADMAP, guide/workflow/matrix/release and engine/native entry
cross-links, the short final engine replay and its compact receipt. No generated
weights, datasets, checkpoints, binaries or raw logs are tracked. No HAP material,
dependency installation or system change. Candidate remains local and unpushed.

Overall limits remain bounded native tanh se_e2_a, 2048 CUDA parameters, width 32,
64 atoms/frame, requested 32-frame/1024-atom tiles with work/payload planning, CPU
geometry/reduction/assessment, O(N) index/sampler state and publication hashes.
Payload budgets are not total-memory caps. Broader models, large physical training,
Linux CUDA, distributed execution, generic DeepMD training and production chemical
validation remain unsupported or unverified. No next feature was started.

## Local Linux native CUDA acceptance

Evidence commit `1dc936caa3ec643b3a86d93f4d1bd4f8a2c688a9`, directly based on
`06439f1cf91306b4d19c570e20abaeb7e140adf6`, is integrated unchanged. It adds
[platform documentation](LINUX_CUDA_VALIDATION.md), a
[compact receipt](evidence/native-linux-cuda.json), exact executed replay helpers,
requests, a source manifest and an original 24-frame synthetic fixture. This
follow-up updates the entry guides, feature matrix and platform boundaries.
No Rust/CUDA implementation, dependency, model format or checkpoint rule changes.
The candidate remains local; nothing was pushed.

The tested runtime was Ubuntu 22.04.5 x86_64 under WSL2 in the cached official
DeepMD 3.1.0 CUDA 12.9 image, with RTX 4090, driver 617.42 and NVRTC 12.9.86.
Release compilation passed after correcting an initial read-only Cargo cache
mount; no source fix, download or installation was needed. Native FP64 inference
matched CPU E/F/V within 0/4.44e-16/4.44e-16. The 24-frame tiled v8/v7 training
comparison differed by at most 5.83e-16 in parameters/Adam moments. Newly created
Linux v8 step-2 to step-7 continuation matched model bytes and final state/metrics
exactly, excluding only the deliberately changed step-count/cache fields.
No GPU fallback, fault or retry occurred. Containers were stopped and the GPU
released. Integration checked evidence hashes and documentation without rebuilding
or rerunning GPU tests.

Single-process v8/v7 observations were 1.034/0.806 seconds; the small v8 case was
slower. These include startup/JIT, IO and assessment and are not a speed benchmark.
Other Linux configurations, cross-platform checkpoint restore, C API behavior,
large physical training and production accuracy were outside this acceptance.
CPU geometry/reduction/assessment and existing architecture/payload bounds remain.
No generated trained weights, checkpoints, executables, raw logs or HAP material
are included; the only dataset added is the small original replay fixture.

## Local bounded large-frame training integration

Native-only source `c7ec6f187e296e0e4066ed4978d11a03a6dd08cd` is integrated
unchanged on `cfe1680e850815d329110e3c1baa6fda16bcbb75`. Explicit reverse tiling
accepts `max_atoms_per_frame=64..256`, subject to unchanged work/payload limits.
Default/explicit 64 keeps historical configuration/PTX and v7/v8. CPU/untiled
training remains 64. Above 64 selects v9; GPU loss uses backend v2 with strided
coverage of every force/virial output. Whole-dataset and whole-batch preflight
prevent oversize frames from reaching cache/context or tile allocation/launch
at their respective API levels. No shared contract or dependency changed.

[Final engine evidence](evidence/native-large-frame-integrated.json) records
8 original 32/128/256-atom frames (6 train/2 validation), four scheduled updates,
12 gradient-frame visits and 6 tiles, reaching 256 atoms/frame and 512 atoms/tile.
CPU/GPU-loss parameter/Adam difference was 5.5511151e-17; maximum partition-metric
difference was 1.5182300e-14. GPU loss recorded 12 device-predicted frames and 60
gradient kernels. Exact v9 step-2 to step-4 continuation changed indexed-to-memory
and cache-on-to-off while preserving final state (except requested steps/cache),
model bytes and final metrics. Reload E/F/V differences were zero. Default 64,
466-parameter effective capacity 138 capacity and changed restart tile capacity rejected
before output. Four host-only checks and the locked offline release build passed.
No unexpected failure/retry occurred; GPU was released at 2026-10-16 16:35:47 UTC
with no remaining omniMD process. A did not repeat B's broad matrix or benchmark.

B's [separate native evidence](native/fixtures/large-frame-training-report.json)
covers 32/83/128/256 atoms, independent derivatives/finite differences, late-frame
preflight, old v8 state/PTX compatibility and exact 6+6 continuation. Single-process
12-update observations were CPU-loss CUDA 1554 ms versus GPU-loss CUDA 1721 ms,
with 4.91/6.19 MB explicit device-buffer peaks. GPU loss was slower; these are not
benchmark medians or total-memory caps. A configured 256 ceiling does not guarantee
admission: the 466-parameter example is limited to 138 atoms by work accounting.

Publication scope is the 22 native source-commit files plus engine static capability
metadata, entry/usage/matrix/roadmap/release cross-links, the short engine acceptance
script and compact receipt. Generated weights, datasets, checkpoints, binaries
and raw logs remain ignored. No HAP material, installation or system change.
Candidate stays local and unpushed. Prior `06439f1` Linux v7/v8 evidence remains valid
for its recorded candidate only; new v9 Linux execution is unverified. Wider
architectures, production physical data and general throughput remain open.
No further feature started in this integration.

## Local physical-label pipeline evidence: model quality unsuitable

Documentation-only commit `fd19b9f492d32fe597599ebcfe7e1e0e5fea3710`, directly
based on `8167a4a3f3ff8d67f53c22f911f3ff31f053fa5d`, is integrated unchanged.
It adds [the experiment report](PHYSICAL_DATA_VALIDATION.md), a validation index
entry and [compact receipt](evidence/physical-nopbc-training.json). This follow-up
updates the top-level scope, guide/workflow, engine entry and feature matrix.
Runtime code, dependencies and schema behavior are unchanged. No tests, GPU runs,
downloads or tuning were repeated, and no public push was performed.

The official pinned DeepMD `nopbc` example supplied published Gaussian 16
MN15/6-31G** E/F labels with retained source/license attribution. The 531-parameter
native model used a fixed 168 training/43 index-held-out split and 1,000 GPU updates.
Held-out E/atom RMSE improved 0.569513->0.455950 eV/atom, but force RMSE worsened
3.933565->4.164336 eV/angstrom. Training force error also worsened. **Pipeline
feasibility passed; the resulting model is unsuitable for scientific acceptance
or production use.** Numerical parity cannot replace physical-label accuracy.

Three held-out frames gave maximum CPU/GPU E/F/virial differences
0/4.44e-16/8.88e-16; export reload differences were zero. No physical virial labels
were available. This checks inference/export, not optimizer-trajectory equivalence.
The Linux/WSL2 run exercised only 7-10-atom frames with the default 64-atom
ceiling and v8; new v9 remains unverified on Linux. Its 100+900 continuation was a timing-only decision, not a
new uninterrupted/resumed equivalence experiment. Original trajectory identities
are unavailable, and the index-held-out split does not prove independence or
transferability. No performance or chemical-accuracy claim follows.

Downloaded source arrays, experiment helpers, initialized/trained models,
checkpoints and raw logs remain task-local and are not added to Git. This
integration retains the recorded negative baseline without diagnosing its cause
or duplicating the separate fixed-split follow-up. No HAP material or system
change is included; the coherent candidate remains local and unpushed.

## Local controlled physical-label diagnosis

Diagnosis source `bd82b250769f3b6f05339e2af69f0801ec035306`, based on `fd19b9f`,
is integrated as `0338e57`. Its report/index/receipt are retained unchanged;
entry documentation now includes the full comparison and validation boundary.
Runtime remains the tested `8167a4a`; this integration adds no code, tests,
training, feature, dependency or public push.

The only numerical configuration change was `energy_weight: 1 -> 0.1`; both fits
use the same initial model, 168/43 split, 531 neural parameters and 1,000 updates.
Validation E/atom RMSE was initial/original/new 0.569513/0.455950/0.523472 eV/atom;
force RMSE was 3.933565/4.164336/3.707381 eV/angstrom. Training E/atom RMSE was
0.775821/0.593012/0.765865 eV/atom, force RMSE 6.970691/7.170289/6.619143
eV/angstrom, and total-E RMSE 6.626241/5.099892/6.650750 eV. The new total-energy training error worsens
versus initialization. All composition groups remain in the
[report](PHYSICAL_DATA_VALIDATION.md#bounded-training-diagnosis) and
[diagnosis receipt](evidence/physical-nopbc-diagnosis.json).

The result supports a loss-weight tradeoff; both models remain unsuitable for
scientific acceptance or production. The 43 index-held-out frames are now used
for diagnostic validation, not an untouched final test. Normalization was fixed;
its independent effect was not tested. Clipping threshold was fixed, but changing
the loss may alter clipping behavior, whose frequency was not measured.

A two-update comparison of 531 CPU/GPU parameters and Adam state over 16 frame
visits had maximum difference about 1.11e-16. It supports limited numerical
consistency, not agreement throughout 1,000 updates or exclusion of shared bugs.
New-model three-frame E/F/V inference differences were 0/8.88e-16/4.44e-16;
reload differences were zero. No physical virial labels or v9 execution were
included. The Linux case remains default-limit/v8 with 7-10 atoms/frame.

Static checks cover the recorded request differences, retained baseline metrics,
diagnostic evidence hashes and document links. Downloaded data, models, helpers,
checkpoints and raw logs remain task-local. No further diagnosis or compute was
performed here; the coherent candidate remains local and unpushed.

## Engine diagnostic reuse before CUDA performance integration

The comparison baseline remains `201cdef0626bd9308bcff07faef4c5ae3aa9d93a`.
The engine now reuses scalar diagnostics while validating and writing an immutable
trajectory snapshot, and reuses total energy for the two drift calculations per
step. This removes redundant CPU particle traversals while retaining the existing
kinetic sum, temperature arithmetic, finite checks, output schema and flush cadence.
Native code, shared contracts, dependency files and GPU execution are unchanged.

CPU-only release builds and two 16-step NVE/NVT comparisons passed: every parsed
trajectory field matched the baseline after normalizing only the output checkpoint
path; periodic and final checkpoint bytes matched exactly. The
[receipt](evidence/engine-diagnostic-reuse.json) retains executable/input hashes
and the local replay path. This is a narrow CPU-work reduction, not a measured
end-to-end or GPU speedup. Native CUDA performance integration and workload-specific
English guidance await measured evidence. No public push is included.

## Local packed-transfer and resident-state integration

Native sources `cb2b94fafce69ec8bcd1a317673c54c05b91ed5f` and
`f4ca7f3a6ec59e7164c6482d574eb7ca650b2a9a` are integrated in order as `9d389ca`
and `8743e0a277c1dd07e9635cc7c65494f836280ab1`, preserving `76d60c6` and the
comparison baseline `201cdef`. No native source/evidence changes were needed.
Existing explicit GPU-loss full training now uses resident parameter/Adam state
until the original checkpoint boundaries, plus 256-byte packed-input offsets.
Per-update status checks, full snapshot validation, FP64 kernels, checkpoint formats
and all-frame export verification remain. CPU and CPU-loss defaults are unchanged.

The actual rebuilt engine/sibling trainer passed the minimal unchanged public
1,000-update request and historical v8 500+500 continuation: model/checkpoint bytes,
metrics, sampler and execution identity match the references; export E/F/V error is
zero. [Evidence and command](evidence/native-resident-integrated.json) are retained;
native 29-test/CPU-only/v9 acceptance was reused rather than rerun. Capability JSON
and English entry guides describe activation, diagnostics and workload limits.

The separate native fixed-work comparison records whole-process medians versus
`201cdef`: public 1332.67->1229.95 ms (7.7% less time), medium synthetic
2946.89->2875.74 ms (2.4%). Medium improvement over transfer-only is unresolved;
tiny CPU 504.19 ms beats CUDA 703.56 ms. Earlier unfavorable transfer-only evidence
remains. These combined gains are specific to the recorded Windows/RTX4090 workloads
and final-only checkpoint cadence; no independent residency/alignment, cold-cache,
Linux or general speed claim. Scientific model-quality limitations remain unchanged.

No benchmark suite, new architecture, dependency installation, system setting or
public push is included. Generated data, weights and raw outputs remain local.
