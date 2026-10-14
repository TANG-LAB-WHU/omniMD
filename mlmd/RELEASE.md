# First local delivery: publication scope

No remote push is part of this preparation. The final commit/tree IDs, current
remote refs and validation result are recorded in the delivery handoff.

## Candidate contents

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
