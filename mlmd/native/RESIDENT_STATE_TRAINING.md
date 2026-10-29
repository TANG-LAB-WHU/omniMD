# Resident optimizer state for bounded CUDA training

The existing `cuda_loss="gpu"` full-training path now retains neural parameters
and Adam moments on the GPU between the **original checkpoint boundaries**.
It still downloads and checks the device status after every update. A private
consumed wrapper publishes a fully validated model/moment snapshot before every
save callback and the final assessment/export, including a partial final interval.
The public eager `CudaFullTrainer` API still refreshes its host model and moments
on every successful `adam_step`.

This is an implementation change within the supported FP64 tanh-network subset.
The loss, derivatives, CUDA kernels/PTX, optimizer, update count, sampler, learning
rate, checkpoint formats/backend identity and output validation are unchanged.
It adds no dependency, runtime installation, model architecture or training option.

## Ownership and failure boundary

`ResidentFullTrainer` consumes the ordinary trainer and exposes no model or moment
accessor. Its host model is an immutable structural template: types, normalization,
cutoffs, offsets and network shapes. The GPU-loss path never reads its stale neural
values for current predictions. Geometry caching stores only fixed geometry;
trainable values remain in the existing device arrays.

Each update validates the gradient/configuration, runs the same Adam kernel and
checks its synchronized status. That kernel checks all mutable parameters and
moments for finiteness, and second moments for nonnegativity. Immutable shapes and
normalization are validated on entry. The global step advances only after status
success. Any returned update/snapshot error poisons the wrapper; no uncertain
update is retried or partial checkpoint returned.

At every original save boundary, three arrays (parameters, first and second
moments) are downloaded into temporary vectors. Moment checks and full model
validation succeed before any checkpoint fields are replaced. Dataset integrity
verification and the existing save callback follow. Save errors stop execution;
the last successfully saved checkpoint remains available. Host parameters are
current for final metrics, parameter-change reporting and serialization.

The packed-input arena now uses **256-byte offsets relative to its device base**.
The planner includes a conservative 2,048-byte alignment reserve for at most nine
input fields, counts actual host vector capacity and device high-water allocations,
and keeps the 32 MiB host payload, 64 MiB device buffer and 50-million-term limits.
No interior pointer is independently owned/freed. CPU-loss tiled inputs use this
alignment too, but retain their eager optimizer state. Alignment is included in
this combined comparison; its individual speed contribution was not isolated.

## Reporting and retained work

`report.json` adds `cuda_state_residency`: successful updates/status downloads,
validated snapshots, state-download calls/bytes, and snapshot download/model
validation timings. Counts follow the implementation's actual calls; they are not
an external API trace. `timing.checkpoint_io_ms` includes dataset verification and
save callbacks. Snapshot work remains in `adam_ms`. `cuda_teardown_ms` measures the
existing owned runtime cleanup, now performed explicitly before returning.

CLI `output_timing` separates final checkpoint writing and model serialization,
reload, dataset verification and all-frame export comparison. **Both fitted and
reloaded CPU potentials still evaluate every dataset frame**: 422 evaluations for
the public case and 256 for each synthetic case. `before_output_write_ms` stops
before final model/report writes; external whole-process time includes these and
process lifetime overhead. Timing fields describe overlapping/nested scopes and
must not all be added together. Host initial/final assessments are retained.

With final-only checkpoints in the measured public run, state downloads fall from
3,000 calls to 3 (12,744 bytes); its 1,000 status downloads remain. The 128-update
synthetic cases fall from 384 state downloads to 3 (11,184 and 2,688 bytes).
Frequent checkpointing reduces this benefit; no save cadence is changed.

## Fixed-work comparison, 2026-10-17

Independent reference commits are `201cdef0626bd9308bcff07faef4c5ae3aa9d93a`
(baseline) and `cb2b94fafce69ec8bcd1a317673c54c05b91ed5f` (packed transfers only).
The current implementation adds private resident state, stronger input alignment
and phase instrumentation. All three executables and source hashes are recorded
in [the replay receipt](fixtures/resident-state-report.json).

Windows, RTX 4090 (compute 8.9), installed official CUDA/NVRTC 12.8, driver API
13040, all FP64. Three fresh GPU processes per implementation per case ran in
fixed rotating order: baseline/transfer/current, current/transfer/baseline,
transfer/baseline/current. Two native CPU references bracketed these nine runs.
There was exclusive GPU use but no OS isolation, clock change or driver-cache
reset. Trainer startup includes admission/template work, NVRTC/driver initialization
and initial assessments, with existing uncontrolled caches; **cold-cache performance
was not measured**. These are small
single-machine samples, not a general speed guarantee or a DeepMD speed comparison.

Whole-process milliseconds, **median [minimum, maximum]**:

| Case | Baseline | Transfer only | Current | Native CPU |
| --- | ---: | ---: | ---: | ---: |
| Published E/F | 1332.67 [1300.27, 1350.58] | 1281.57 [1260.53, 1386.76] | 1229.95 [1215.48, 1237.98] | 2576.25 [2574.87, 2577.63] |
| Medium synthetic E/F/V | 2946.89 [2908.25, 2952.10] | 2879.23 [2861.56, 2926.93] | 2875.74 [2839.78, 2878.48] | 37114.17 [36819.10, 37409.24] |
| Tiny synthetic E/F/V | 709.35 [695.91, 761.34] | 682.22 [678.83, 690.34] | 703.56 [685.09, 713.87] | 504.19 [488.99, 519.40] |

The combined changes reduce elapsed median time versus the baseline by **7.7% on
public E/F and 2.4% on medium synthetic E/F/V**; observed ranges do not overlap.
Public current/transfer ranges also do not overlap. The medium incremental change
versus transfer-only is only 0.12%, with overlapping ranges: no reliable additional
whole-process benefit is established there. Tiny GPU ranges overlap; current is
3.1% slower in median than transfer-only. **CPU remains faster for the tiny case.**
The CPU/current median ratios are 2.095, 12.906 and 0.717 respectively, specific to
these native implementations/workloads. They do not measure physical accuracy.

Separating startup from gradient + Adam update work (milliseconds; median [range]):

| Case | Baseline startup | Current startup | Baseline updates | Current updates |
| --- | ---: | ---: | ---: | ---: |
| Published E/F | 584.09 [571.39, 587.85] | 580.63 [579.41, 586.96] | 668.33 [652.97, 682.55] | 569.68 [556.46, 572.97] |
| Medium E/F/V | 991.15 [973.44, 991.98] | 981.82 [978.63, 996.49] | 1211.43 [1206.86, 1219.90] | 1128.71 [1115.35, 1145.91] |
| Tiny E/F/V | 507.15 [501.05, 550.14] | 524.97 [511.91, 539.02] | 132.39 [117.99, 137.18] | 101.31 [100.67, 108.46] |

Public Adam median falls from 101.54 to 46.05 ms, while current state download and
model validation together take about 0.05 ms at its single snapshot. Current
public teardown is 17.70 ms [17.52, 18.92], save-callback IO 0.99 ms [0.97, 1.24],
final checkpoint IO 0.68 ms [0.67, 0.71], and all-frame export validation 11.48 ms
[11.12, 11.50]. Medium export validation remains 455.18 ms [442.56, 476.77].
These measurements expose retained costs; the earlier public process outlier is
not retrospectively attributed to a single phase. CPU geometry, reduction,
clipping, assessments, integrity checks, serialization and per-update status
transfers remain. NVRTC/driver startup still dominates small workloads.

The earlier [transfer-only receipt](TRANSFER_BATCHING.md) retains its unfavorable
public result (4.5% median regression). A preliminary resident/8-byte-alignment
single-run pilot is also retained in the new receipt and is not used to select a
best run or infer alignment speedup. Acceptance uses every run of the fixed final
three-implementation comparison above.

## Workloads and numerical acceptance

- Public: 531 neural parameters, 211 frames of 7/8/10 atoms, explicit 168/43
  training/validation split, E/F loss, 1,000 Adam updates, batch 8, 8,000 frame
  visits, final-only checkpoint. LR 0.001, decay every 250 updates by 0.5;
  E/F weights 1/1 and scales 0.1 eV/atom and 1 eV/angstrom. Dataset in memory.
- Medium: original synthetic teacher, 466 parameters, 128 frames of 32/64 atoms,
  120/8 split, E/F/V loss, 128 updates, batch 32 with final batches of 24,
  3,840 visits; indexed dataset, final-only checkpoint.
- Tiny: original synthetic teacher, 112 parameters and 3/4 atoms per frame;
  the same 128-frame split, update count, batching and storage as medium.

The receipt contains exact configs, validation IDs, dataset/model hashes,
execution/PTX identities, counts and all per-process timings. Within each case,
all nine GPU model and checkpoint files are **byte-identical** across the three
implementations. Metrics, sampler, frame/tile/kernel/synchronization counts and
effective atom capacity are unchanged. CPU repeats agree exactly; GPU/CPU model
and moment maximum absolute differences are 1.76e-14, 2.22e-16 and 3.66e-15.
Every export/reload E/F/virial comparison has zero error.

Current peak host tile payload/device buffers (bytes) are 638,504/1,539,432
(public), 1,170,336/3,531,156 (medium), and 1,513,856/758,316 (tiny), all within
existing bounds. These counts exclude CUDA driver reservation and process RSS;
the receipt separately retains process peak working sets.

Historical v8 step-500 checkpoints resume for 500 more updates while switching
storage/cache settings with exact final state, metrics and PTX identity. A
historical v9 run with 128/256-atom frames resumes with byte-identical checkpoint
and model files and identical metrics/PTX. This is tested compatibility, not a
claim about every possible checkpoint. No model/dataset/checkpoint schema changed.

The public data is the already audited [physical-data subset](../PHYSICAL_DATA_VALIDATION.md),
from DeepMD commit `b494a0d5413d5f1895202c82f20477300b716bf2`, `examples/nopbc/data`.
Attribution: Jinzhe Zeng and Tong Zhu, *Dataset for methane combustion* (2020),
[version 3](https://doi.org/10.6084/m9.figshare.12973055.v3), CC BY 4.0;
example repository distribution LGPL-3.0-or-later. Published Gaussian16
MN15/6-31G** E/F labels stored as float32 are promoted exactly to float64; there
are no public virial labels in this case. Raw arrays and generated weights are
not redistributed here. The original fit worsens force RMSE; its 43-frame
validation split is not an untouched test set. **No production accuracy or
scientific validation claim follows from this performance replay.**

## Reproduce and inspect

The existing dependencies/toolchain must already be available. No downloads or
configuration changes are performed. Build each historical revision in an
isolated checkout to preserve the two reference executables; build current with:

```sh
cargo build --release --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda --bin omnimd-native-train
```

With an exclusive GPU window and the same prepared `physical`, `medium`, `tiny`
inputs (`initial.json`, `dataset.jsonl`, `profile.request.json` in each directory):

```sh
python mlmd/native/tools/replay_resident_training.py \
  --baseline-exe /path/to/201cdef/omnimd-native-train.exe \
  --transfer-exe /path/to/cb2b94f/omnimd-native-train.exe \
  --current-exe /path/to/current/omnimd-native-train.exe \
  --prepared-dir /path/to/prepared-inputs --output-dir /path/to/new-results
```

Preparation follows `tools/replay_transfer_batching.py` and the public integration
recipe linked above. Keep generated datasets, weights, executables and raw logs
under ignored local directories. The replay uses Python standard-library helpers,
checks identical work/states and records each process, not just the fastest run.

Targeted acceptance commands (29 tests total; all passed):

```sh
cargo test --release --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda --lib -- --include-ignored --test-threads=1
cargo test --release --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda --test cuda_reverse_training resident_snapshots -- --include-ignored --test-threads=1
cargo test --release --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda --test cuda_reverse_training device_loss -- --include-ignored --test-threads=1
cargo test --release --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda --test large_frame_training -- --include-ignored --test-threads=1
cargo check --locked --offline --manifest-path mlmd/native/Cargo.toml --no-default-features --bins
```

The 22 unit tests include finite-but-overflowing Adam input and poisoned-wrapper
rejection. The new integration test compares eager/resident model and moments
exactly at steps 2, 4 and final partial interval 5, checks a resumed learning-rate
boundary and a failing save callback. Four existing GPU-loss and two large-frame
tests cover derivatives, invalid tiles, capacities, finite differences and resource
bounds. Historical restart receipts are separate from this test count.
