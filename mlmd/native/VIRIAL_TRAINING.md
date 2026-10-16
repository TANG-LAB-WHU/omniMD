# Optional virial supervision for full neural-parameter training

The supported tanh `se_e2_a` full trainer now accepts an explicit virial loss on
both CPU and CUDA. This adds supervision of the shared row-strain virial to the
existing E/F loss. The native model format, dataset JSONL v1, inference outputs,
parameter layout and all architecture/resource bounds remain unchanged. The
frozen-readout QR trainer still fits E/F only.

## Configuration and label contract

Add this optional object inside the full request's `config`:

```json
"virial_loss": {"weight": 1.0, "scale_ev_per_atom": 0.1}
```

Use `"device":"cpu"` or, with the CUDA feature, explicit `"device":"cuda:0"`.
Omitted or null `virial_loss` disables the term and preserves historical E/F
behavior. Both object fields must be finite and strictly positive; zero weight
is rejected rather than treated as another spelling for disabled. Unknown
fields are errors. Normalization coefficients that underflow to zero or
become nonfinite for the actual split/frame dimensions are rejected before
CUDA initialization or checkpoint output.

Enabled virial loss requires all nine finite virial components on every
training and validation frame. Missing labels are never substituted with zero.
E and F labels remain required even when their weights are zero. Both E/F
weights may be zero only when virial supervision is enabled. E/F scales remain
positive. No label, unit, sign, pressure or volume conversion is implicit.

The dataset field is the existing `labels.virial`, in eV, with convention
`MinusDEpsilonRow`. Cell rows and positions undergo the same deformation:

```text
r'_b = r_b + sum_a r_a epsilon_ab
W_ab = -dE/d epsilon_ab
```

Nonperiodic frames are supported using the same affine-coordinate convention.
All nine entries contribute, so a symmetric off-diagonal residual is counted
twice in the Frobenius norm. No extra factor of two is applied to its seed.
Targets are not silently symmetrized: an antisymmetric part contributes
constant loss that this rotationally invariant energy model cannot learn.
Do not interpret that residual as evidence that more optimizer steps can fit it.

## Loss, normalization and analytic derivatives

For each split independently, let `M` be its frame count, `N_f` the physical atom
count of frame `f`, and `sV` the configured per-atom scale in eV/atom. Add

```text
L_V = wV/(9 M) sum_f sum_ab [(W_fab - W*_fab)/(N_f sV)]^2
L_total = L_EF + L_V
C_ab = dL/dW_ab = 2 wV (W_ab-W*_ab)/(9 M N^2 sV^2)
```

There is no one-half factor and no division by volume. The existing E/F loss is
unchanged. Reports retain `normalized_ef_objective` and add
`normalized_virial_objective`, `normalized_total_objective`, and
`virial_per_atom_component_rmse_ev` when applicable. The existing
`virial_component_rmse_ev` remains the total-virial component RMSE. Assessment
uses one inference pass, and validation labels never enter parameter updates.

Using detached loss seeds `a=dL/dE`, `v_i=dL/dF_i` and `C=dL/dW`, each full image
displacement `d = r_j-r_i+nH` receives this directional perturbation:

```text
u_b = v_jb - v_ib + sum_a d_a C_ab        # row product d*C, not d*C^T
R_u = J_environment * u
grad_theta L_total = a grad_theta E - grad_theta E_u
```

The same shared host seed construction feeds CPU reverse-over-forward AD and
CUDA four-component differentiation. Geometry, cutoffs and normalization are
fixed parameters, so only their existing first displacement Jacobian is needed.
Normalized padding keeps its constant value and has zero tangent. For periodic
self-images the force contribution cancels but the `d*C` contribution remains.
Cell-image choices are held fixed during neural-parameter differentiation.

No CUDA kernel arithmetic or extra kernel launch is needed: the existing
mixed-derivative kernel consumes the combined direction, and Adam remains on
CUDA. Prediction evaluation, loss seeds, geometry, stable reduction/clipping,
assessment and checkpoint IO remain CPU work. No finite differences are used
in production training. Virial-only optimization cannot identify constant
energy offsets or final fitting biases; those receive zero virial gradient.

## Checkpoint compatibility

| Loss | CPU checkpoint | CUDA checkpoint |
| --- | --- | --- |
| Existing E/F, virial option absent/null | v1, unchanged fields | v2, unchanged fields |
| Virial enabled, including virial-only | v3 | v3 with existing CUDA identity |

The full names are `omnimd-native-full-checkpoint-vN`. A v3 checkpoint records
`loss_contract="energy-force-row-virial-per-atom-nine-v1"` and the immutable
`virial_loss` object. It retains all existing model/anchor, optimizer, dataset
SHA, split and layout fields. The CPU execution default is still omitted;
CUDA retains its backend/ordinal/device/driver/PTX identity. The checkpoint loss
contract and execution identity together specify numerical semantics.

Resuming with a changed loss weight/scale, disabled virial term, incompatible
schema/contract or backend is rejected. The existing same-device/runtime rules
continue to apply. v1/v2 checkpoints cannot enable V on resume; start a new
optimizer explicitly from their model to change the objective. Older binaries
reject v3 rather than silently dropping its virial loss. Exact resume is only
claimed within the tested platform/toolchain/runtime.

The low-level CPU and CUDA `frame_gradient` methods still compute E/F. Their
new `frame_gradient_with_virial` methods take the separate optional `VirialLoss`
(exported from `full_training`); `FullConfig::loss()` is the legacy E/F component.
Normal `train_full` / `train_full_with_device` dispatch both components correctly.

## Reproduction and results

```sh
cargo test --locked --offline --manifest-path mlmd/native/Cargo.toml --no-default-features
cargo test --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda
cargo build --release --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda --bins
# Only in an authorized GPU window; these tests are ignored by default:
cargo test --release --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda \
  --test virial_training -- --ignored --nocapture --test-threads=1
python mlmd/native/tools/replay_full_training.py \
  --eval-exe /path/to/omnimd-native-eval --train-exe /path/to/omnimd-native-train \
  --device cuda:0 --compare-cpu --virial-weight 1 --virial-scale-ev-per-atom 0.1 \
  --output-dir /new/replay/directory
```

Windows GNU/RTX 4090 verification on 2026-10-16 used original synthetic inputs:

- Closed-form dimer energy/all nine virials and its parameter gradient; independent
  nine-component strain differences; explicit nonsymmetric row-product oracle.
- Every one of 112 parameters checked against central differences for V-only and
  E/F/V losses on a cluster, triclinic PBC and a one-atom periodic self-image case:
  maximum absolute error `9.44e-9`. Force is zero in that self-image case while
  virial gradients are nonzero.
- Unequal atom counts (1, 3, 4), frame averaging, nonunit scales, zero residuals,
  rotational and wrapping invariance, antisymmetric target behavior and validation
  isolation passed. Actual CLI rejection checks leave no output directory and
  fail before the deliberately nonexistent CUDA device can be initialized.
- CUDA/CPU gradient maximum difference `2.66e-15`; clipped two-step Adam parameter
  difference `6.94e-18`, moment difference `2.60e-18`. Serialized `2+1` equals
  continuous `3` exactly; v3 CLI `150+150` equals continuous `300` exactly.
- The 24-frame CLI fit (18 training, six validation) updated all 112 neural
  parameters. Validation total objective fell from `11.18699` to `0.00468772`;
  validation virial-per-atom component RMSE fell from `0.102306` to `0.00259499`
  eV/atom. CPU/CUDA final parameters differed by `1.11e-15`; model reload E/F/V
  errors were zero.
- Actual historical E/F-only CPU v1 and CUDA v2 checkpoints from step 150 were
  resumed with this implementation. All resulting checkpoint fields matched
  the historical step-300 state exactly except the requested additional-step
  count. Existing E/F GPU derivative and update regressions also passed.

For the tiny 300-step joint-loss run, CPU/CUDA gradient phases were
575.93/1415.20 ms and the complete training subprocesses were 619.07/1816.23 ms.
The subprocess measurement includes initialization, input IO, updates,
assessment, checkpoints, model reload verification and export; it excludes
teacher-label generation, separate prediction and restart replay processes.
This is one bounded fixture measurement, not a production benchmark or an
acceleration claim. No additional performance comparison is inferred from the
older E/F-only 64-atom timing.

The compact [evidence report](fixtures/virial-training-report.json) records
commands, numerical results, provenance hashes and timing scope. Generated
models, datasets, checkpoints, raw logs and binaries remain ignored. There are
no new dependencies. Linux CUDA execution, external-label physical accuracy,
large-model throughput and general DeepMD training compatibility remain outside
this validation. The implementation uses the existing published descriptor,
chain-rule and Adam mathematics; no algorithmic novelty is claimed.
