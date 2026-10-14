# Validation record: 2026-10-14

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

## Limits and local provenance

The C API guarantees double buffers, not all models' internal precision or
training units. GPU requests fail. Native inference accepts only its documented
CPU se_e2_a subset; full native training/GPU and hybrid se_e3 are unimplemented.
Safe DeepMD conversion and same-generated-model native/C-API parity are separate
work in progress; the public regression .pb lacks official serialization metadata.

Upstream main was `761b2484feee132426183c9b66ff2b126d1f7c4a`;
dev `186817f5a85837df5f1df28e0fd2e8b2de6ecade`
(main...dev 13/1, final diff README only);
old shawn_dev `3440f605dde61b01e894f6532693612bc57c0aaa`
(main...shawn_dev 0/1, only .DS_Store ignore).
New shared baseline is `8121e214e062077b88c39d0ccd861c55efaecc2c`.
No old shawn_dev-specific work or private HAP inputs were imported. No remote
was changed. Toolchains/runtimes were installed only in task-local locations
and the isolated CPU container; no global PATH, driver or security changes.
