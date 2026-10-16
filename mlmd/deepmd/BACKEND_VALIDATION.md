# Backend tranche validation and integration notes

Scope: adapter-local multi-frame inference, explicit frame/atomic parameters,
optional atomic outputs and standalone inspection. Shared `Frame`, `Evaluation`,
`Potential`, engine code and native inference code are unchanged in this tranche.

## Evidence from this revision

- Windows CPU-only build: the DeepMD and engine release binaries compile with
  the cached toolchain using `--offline --locked`; no CUDA SDK or new dependency
  is required. Existing deprecation warnings originate in core/sim code.
- Focused Rust tests: 17 adapter/library tests and 4 standalone CLI tests pass.
  These cover model/string lifetime, legacy loading, a single combined call for
  a two-frame batch, parameter packing, periodic cell order, requested atomic
  output shapes/sums in the test double, signed C-integer overflow, invalid
  inputs before FFI, whole-batch failure, CLI chunking and no-clobber publication.
- An optional API27 charge/spin-dimension query rejects conditioned models when
  that metadata is available. Missing optional metadata preserves API26 loading.
  This is rejection support, not charge/spin inference support.
- CPU/GPU result comparison now counts combined C API calls, which can each
  cover multiple frames. It also compares requested atomic outputs and checks
  compatible batch sizes/metadata. All 17 focused synthetic/saved-output Python
  regressions passed, including under `python -O`. Existing single-frame receipts
  remain usable; these script checks do not invoke a GPU.
- The old executed CPU/GPU result is preserved separately in
  [CUDA_VALIDATION.md](CUDA_VALIDATION.md). Its numerical values and timings do
  not validate the newly added multi-frame/parameter/atomic paths.

Commands used from the repository root (with the cached Windows environment and
an external `CARGO_TARGET_DIR`):

```sh
cargo test --offline --locked --manifest-path mlmd/Cargo.toml -p omnimd-deepmd
cargo build --offline --locked --release --manifest-path mlmd/Cargo.toml -p omnimd-deepmd -p omnimd-mlmd --bins
```

## Remaining runtime validation

The intended action is a small CPU-only, same-model comparison of a homogeneous
multi-frame call against serial calls, including atomic output shapes and the
public fixture's energy/virial sums. The previously reused official CPU runtime
is in Docker. A read-only `docker ps --filter name=omnimd` probe failed because
`//./pipe/dockerDesktopLinuxEngine` did not exist. The Docker daemon is therefore
unavailable in this environment; no model evaluation or Linux build was started
for this tranche. This is an availability blocker, not an approval denial or an
uncertain inference outcome.

A follow-up attempt used the explicitly authorized ordinary startup of the
existing Docker Desktop installation. `Start-Process` succeeded;
`docker desktop start --timeout 45` subsequently reported the application was
already running. The actual engine remained stopped. Its backend log at
`2026-10-16T07:46:15.920487200Z` records:

```text
running services: running services: running OTel manager: OTel manager:
removing stale socket: remove <HOME>\AppData\Local\Docker\run\userAnalyticsOtlpHttp.sock:
The file cannot be accessed by the system.
```

The backend then reported `backend crashed`; a read-only `docker desktop status`
confirmed `stopped`, and the Linux-engine pipe was still absent. One ordinary
`docker desktop restart --timeout 45` then failed to stop the remaining Desktop
frontend/backend processes, returning `context deadline exceeded` and exit 1;
it did not restore the engine. No forced process termination was attempted.
Desktop UI/error
reporter processes remaining alive do not mean its engine is available. No tool
approval was denied, and no inference had an uncertain outcome. Recovery stopped
at this application runtime-file failure: no socket deletion, settings change,
Docker reset/reinstall, WSL repair, credentials change or unrelated-container
operation was attempted. The exact startup outcome is also recorded in
[the check receipt](validation/backend-tranche-checks.json).

No GPU process/container was started. The coordinated GPU window remains with
B; the new features are not claimed hardware-validated. No driver, BIOS,
security setting or unrelated container was modified. No Windows C API runtime
was installed as a workaround.

When the official CPU runtime is available:

1. Build the current adapter inside that runtime without GPU access.
2. Reuse the existing public model and a small homogeneous frame set; use
   identical library, model, units and type map in both requests.
3. Run `inspect` with direct `LoadOptions` JSON. Confirm expected dimensions.
4. Run once with `batch_size:1`, `atomic:true` and no timing; run once with a
   batch size covering the same frames. Retain both outputs under new paths.
5. Compare every E/F/virial component and each atomic array with a precision-
   appropriate tolerance. For the ordinary energy fixture, also compare sums of
   atomic energies and virials with their totals. Do not treat atomic per-atom
   decompositions as universal physical observables.
6. For actual parameterized-model validation, use an official matching fixture
   with nonzero dimensions, distinct parameters per frame/atom and an official
   runtime reference. The current synthetic parameter test proves ABI packing,
   not that a trained parameter-dependent model is numerically correct.

## Integration into A's assembly branch

All changes are confined to `mlmd/deepmd/**`. Existing request JSON remains valid:
`device` still defaults to CPU, `batch_size` to 1, and `atomic` to false. Existing
`evaluations` records retain their shared schema. New response keys are additive.
The CLI's new `inspect OPTIONS.json NEW_INFO.json` takes `LoadOptions` directly;
the normal evaluator continues to take an object containing `options` and
`frames`. Output publication now requires a filesystem supporting hard links.

Parameter-aware or atomic MD integration is intentionally not added to shared
interfaces. A can choose a later shared-contract extension; until then, callers
use `DeepMd::evaluate` or this standalone CLI. Parameter-free MD continues to
use the existing `Potential::eval` method. The English
[step-by-step guide](BACKEND_GUIDE.md) can be linked from the top-level README.

This remains delegated official DeepMD inference. It does not imply parity of
`mlmd/native`, implement training in Rust, or promise every v3.2.0 model family.
