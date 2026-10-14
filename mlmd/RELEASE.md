# Release scope

The accepted CPU baseline fce0af9f6a49592a48c13883dfc3aa0d3eb4a28e was published on
shawn_dev. Following separate user authorization, the CUDA release
[907481708ae5f3df4228a9fb5d5b1110c39f5e23](https://github.com/TANG-LAB-WHU/omniMD/commit/907481708ae5f3df4228a9fb5d5b1110c39f5e23)
was published to the same branch with an ordinary non-force push. Its remote
hash was verified after publication.

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
