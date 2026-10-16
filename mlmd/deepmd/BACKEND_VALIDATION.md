# Backend validation and integration notes

CPU numerical validation completed on 2026-10-16 with official DeepMD 3.1.3,
TensorFlow 2.19.1 and PyTorch 2.10.0. The CPU container exposes no GPU devices;
all workers and Python references use `CUDA_VISIBLE_DEVICES=-1` and one thread
for OpenMP and DeepMD intra/inter-op settings. Every numerical comparison covers
E, all force/virial components and requested atomic energies/virials.

## Executed numerical checks

| Fixture | Comparison | Max absolute E / F / virial error |
| --- | --- | --- |
| Public model, four periodic six-atom frames | Serial C API vs official Python | 0 / 4.440892098500626e-16 / 1.3322676295501878e-15 |
| Same public frames | Serial vs batch; C API calls 4 to 1 | 0 / 4.440892098500626e-16 / 6.661338147750939e-16 |
| TF parameter model, two six-atom frames | Serial vs batch; calls 2 to 1 | 0 / 0 / 0 |
| Same TF parameter frames | C API vs official Python | 0 / 0 / 2.220446049250313e-16 |
| PT default model, one six-atom frame | Explicit vs stored defaults vs official Python | 0 / 0 / 0 |

Energy and virial units are eV; force units are eV/angstrom. The public case's
atomic energy error against Python is zero and atomic virial error is at most
4.440892098500626e-16. TF parameter and PT comparisons have zero atomic-output
error against Python. Fixture-specific atomic sums also pass. See the compact
[batch/parameter receipt](validation/cpu-batch-parameters.json) and
[default-parameter receipt](validation/model-default-fparam.json) for all
component errors, tolerances, source URLs, model fingerprints and output hashes.

The TF parameter case keeps coordinates fixed and changes the second frame's
frame parameter by +0.1 and atom i's parameter by +0.01*(i+1). Its energy changes
by 0.00211578262527512 eV, exercising distinct parameter packing. The original
official TF fixture has an empty type map and is unsupported as supplied. The
tested derivative changes only `model_attr/tmap` to `O`, using the matching
official testcase's declared map; restoring that value reproduces the original
GraphDef bytes exactly. Weights and computational operations are unchanged.
The official loader catches that fixture's missing optional `aparam_nall`
metadata and uses local-atom parameters; numerical results and C API status pass.

The PT fixture is the unmodified official `fparam_aparam_default.pth`. It reports
frame/atomic dimensions 1/1 and stored-default availability true. Explicit
`fparam:[0.25852028]` and opt-in NULL-fparam calls match, with explicit atomic
parameters retained. Results also match the upstream C API fixture baseline:
the largest error over every checked PT output is 7.875644580934704e-16 (atomic
virial). The upstream PT test tolerance is 1e-7; the TF fixture tolerance is 1e-8.

These are small compatibility fixtures, not validation of a production model's
physical accuracy. Double input/output does not establish internal model precision.
The tests measure correctness and reduced call count; no new speedup is claimed.

## Wrapper checks and builds

- The saved default-policy suite passed 22 library and 5 CLI tests. It covers
  optional metadata, unchanged strict behavior, NULL-fparam handling, explicit
  atomic parameters, validation before FFI, cleanup, chunking and timed policy
  propagation. Eight additional comparator cases passed; the preceding batch
  tranche had 17 comparator cases, including optimized Python execution.
- One subsequent focused library regression passed for the `.pth` runtime
  restriction, proving no compute on multi-frame input, valid single-frame
  output and preserved TensorFlow batching. The other 22 tests were filtered
  for this narrow follow-up.
- Real CPU negative cases reject TF model-default requests and PT multi-frame
  requests with nonzero status and no published result.
- Final Windows and Linux CPU release builds of `omnimd-deepmd` and
  `omnimd-mlmd` passed using cached dependencies and `--offline --locked`.
  No CUDA SDK or new Rust dependency is required. Existing deprecation warnings
  originate in core/sim code.

```sh
cargo test --offline --locked --manifest-path mlmd/Cargo.toml -p omnimd-deepmd
cargo build --offline --locked --release --manifest-path mlmd/Cargo.toml -p omnimd-deepmd -p omnimd-mlmd --bins
```

## Supported boundaries

The optional `DP_DeepPotHasDefaultFParam` query is cached once. Missing metadata
means unknown, distinct from false. Legacy `evaluate` / `Potential::eval` remain
strict. Only `evaluate_with_options` or CLI `frame_parameter_policy:model_default`
can request stored values; atomic parameters remain explicit. No retry or inferred
parameter fallback occurs. DeepMD 3.1.3 TensorFlow backend does not provide these defaults.

Default policy is limited to one frame per call. All PyTorch `.pth` ordinary C
API calls, including explicit-parameter calls, have the same limit because the
[official runtime fixes nframes=1](https://github.com/deepmodeling/deepmd-kit/blob/v3.1.3/source/api_cc/src/DeepPotPT.cc#L337).
Use `batch_size:1` to process several frames serially. Other backend/model
families and higher parameter dimensions remain outside the real fixture evidence.
Spin and available nonzero charge/spin conditioning metadata are rejected.

No new GPU inference or timings were run for these additions. The older executed
single-frame CUDA result remains in [CUDA_VALIDATION.md](CUDA_VALIDATION.md);
it does not validate batch, parameter or atomic CUDA paths. Operational Docker
logs, raw model downloads and local troubleshooting data are not release assets.

## Integration

Changes stay in `mlmd/deepmd/**`; shared contracts, engine source and native
inference are unchanged. Existing JSON keeps CPU, batch size 1, atomic false and
explicit parameter policy as defaults. New response keys are additive. `inspect
OPTIONS.json NEW_INFO.json` reads `LoadOptions` directly; evaluation reads an
object containing `options` and `frames`. Output publication requires hard links.

Parameterized or atomic MD needs a later shared-contract extension. For now use
the adapter-local methods or standalone CLI; parameter-free MD retains
`Potential::eval`. This remains delegated official DeepMD inference and does not
imply native-backend parity or implement Rust training. See the
[backend guide](BACKEND_GUIDE.md) for request examples and error handling.
