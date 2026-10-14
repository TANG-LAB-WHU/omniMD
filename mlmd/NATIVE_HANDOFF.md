# Native line handoff (B)

Start from the shared-contract commit on local branch `mlmd/native-se-e2-a`.
The sibling worktree is `D:\\project\\omniMD-native`; the A worktree is
`D:\\project\\omniMD-deepmd`; integration branch `shawn_dev` is in
`D:\\project\\omniMD`.

Own `mlmd/native/**` and add "native" to `mlmd/Cargo.toml` members.
Use package `omnimd-native` and implement `omnimd_ml_contract::Potential`.
Do not change shared contract semantics without coordinating with A.

Implement CPU float64 se_e2_a + energy fitting. Use a clearly versioned JSON
weight format with architecture, ordered type_map, cutoff/smoothing, sel,
normalization, embedding/fitting layers and weights. Reject unknown fields,
unsupported hybrid/se_e3/spin/type embedding, malformed dimensions, nonfinite
weights and unsupported precision. Do not claim arbitrary .pb/.pth support.
Use correct image enumeration for periodic cells; do not assume fractional
rounding is sufficient for triclinic cells or small cells relative to cutoff.
Overflowing neighbor capacity must fail, not silently truncate neighbors.

Provide energy, analytic/AD forces and strain-gradient virial in one eval,
including normalization, smooth cutoff, padding, network residual semantics.
Test a deterministic synthetic fixture, two species/mapping, boundary images,
triclinic PBC, finite-difference forces and full strain tensor virial,
translation/permutation/rotation consistency, and explicit unsupported errors.
Document exactly which se_e2_a options are implemented and fixture provenance.
No full training or GPU claim. No HAP files, old models, logs or simulations.

The parent will dispatch the cloud task. This local task will not spawn native
subagents to access the user's PC. Cloud can reproduce the shared contract from
the handoff/bundle supplied by the parent if it cannot access these local paths.
