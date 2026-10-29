# Bounded packed inputs for CUDA training

**Historical implementation/evidence:** this document records `cb2b94f`, including
its eight-byte alignment and 64-byte reserve. The subsequent
[resident-state implementation](RESIDENT_STATE_TRAINING.md) uses 256-byte offsets,
a 2,048-byte reserve and deferred state snapshots. The measurements below are
preserved unchanged, including the unfavorable public result.

Tiled training now uploads each tile's inputs in one persistent device allocation,
with one separate status reset. The GPU-loss path changes from ten synchronous
HtoD calls to two per tile; the CPU-loss tiled path changes from five to two.
FP64 kernels, launch order, reductions, sampler, Adam, and supported model/data
formats are unchanged. No additional option or checkpoint version is introduced.
This is an implementation optimization, not a new descriptor or training algorithm.

The input-upload phase is faster in the measured cases. **A reliable complete-process
speedup over the previous GPU implementation is not established for the public-data
case.** The final three-repeat median regressed there, despite faster updates.
The tiny task still completes faster on the native CPU reference. Measurements and
all per-run timings are in [the compact receipt](fixtures/transfer-batching-report.json).

## Storage and failure boundary

`transfer.rs` computes checked eight-byte offsets for initialized f64/i32 slices.
Rust never creates a typed reference into the byte staging vector. Padding is
initialized, and only the active span is uploaded. Each tile repacks current
geometry, types, labels and scales; this is not a cross-tile identity cache.
Offsets are rebound on every call, including a shrink or changed frame count.

The runtime owns only the arena base. Temporary kernel argument pointers refer to
its interior; the owning buffer list never stores those aliases. Replaced legacy
input allocations stay empty. Growing scratch and the old arena are freed before
replacement. The synchronous copy and existing ordered stream preserve lifetimes.
No pinned host allocation, async stream, CUDA graph or allocator dependency is added.

The host planner counts both the typed tile and retained staging capacity. A
precheck rejects requests outside the remaining budget, and the actual allocated
Vec capacity is checked before CUDA work. The device plan adds at most 64 bytes
of alignment reserve. Independent high-water capacities are counted across shape
changes. Existing limits remain 32 MiB host tile payload, 64 MiB device buffers,
50 million estimated scalar terms, and the existing frame/atom/parameter limits.
These payload bounds are not process RSS or CUDA driver reservation bounds.

Any returned admission, driver-allocation, upload or kernel error still poisons the trainer.
No optimizer update commits from a failed tile batch. Default framewise execution
and inference are unchanged. Report fields `input_transfer_calls` and
`input_transfer_bytes` cover tiled gradient inputs plus status resets, excluding
initialization, Adam and DtoH. `upload_ms` includes staging reservation/packing and
driver upload calls. Counts follow actual tile orchestration, not an external API trace.

## Identical-work Windows measurements

Baseline native code: `201cdef0626bd9308bcff07faef4c5ae3aa9d93a`.
Hardware: RTX 4090, compute 8.9; official installed CUDA/NVRTC 12.8, driver API 13040.
All runs use FP64. Each GPU binary ran in three alternating-order fresh processes.
Times include initialization/JIT, data admission, training, assessment, checkpoint
and model export. Driver/JIT cache and GPU clocks were not modified; a fresh process
is not a guaranteed cold driver cache. There was exclusive GPU use, not OS isolation.

| Case | Fixed work | Old whole, ms | New whole, ms | Old/new | CPU whole, ms | CPU/new GPU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Published E/F | 531 parameters; 211 frames; 7/8/10 atoms; 1,000 updates | 1351.75 | 1412.19 | 0.957x | 2453.11 | 1.737x |
| Medium synthetic E/F/V | 466 parameters; 128 frames; 32/64 atoms; 128 updates | 3050.80 | 2959.17 | 1.031x | 40599.84 | 13.720x |
| Tiny synthetic E/F/V | 112 parameters; 128 frames; 3/4 atoms; 128 updates | 724.15 | 705.06 | 1.027x | 510.06 | 0.723x |

GPU columns are medians. CPU is one fresh-process run per final case using the same
final executable, inputs, labels, sampler, update count and loss; CUDA-only selectors
and geometry-cache option are omitted because CPU explicitly rejects them. This is
comparison with the native CPU AD reference, not with an optimized external CPU
implementation or DeePMD-kit. Synthetic cases are computational checks, not material
validation. No precision, tolerance, workload or checkpoint-work reduction was used.

Separate timing totals, in milliseconds (not per-update latency):

| Case | Startup old -> new | Gradient + Adam old -> new | Update ratio | Upload old -> new |
| --- | ---: | ---: | ---: | ---: |
| Published E/F | 596.89 -> 599.14 | 671.37 -> 650.23 | 1.033x | 97.25 -> 28.43 |
| Medium | 1057.78 -> 1025.26 | 1245.57 -> 1193.04 | 1.044x | 196.28 -> 58.95 |
| Tiny | 531.73 -> 528.02 | 116.59 -> 108.07 | 1.079x | 21.69 -> 5.64 |

“Gradient + Adam” is the measured update computation after startup, excluding
assessment/checkpoint/IO; it is not the whole warm process. Medians of separate
phases need not sum to the median process time. Kernel timing and other costs also
vary, so the upload improvement does not translate directly into process speedup.
The preliminary build had public whole-process ratio 1.054x, while the final build
measured 0.957x; both summaries are retained. Do not select only the favorable run.

Public-data uploads drop from 10,000 to 2,000 calls for 1,000 tiles; medium from
17,630 to 3,526; tiny from 2,560 to 512. Tile/kernel/frame counts stay exact.
Observed host payload peaks increase from 489,576 to 637,088 bytes (public),
845,888 to 1,169,920 (medium), and 1,355,008 to 1,513,536 (tiny). Device payload
increases by eight bytes for public and is unchanged for the other two cases.

## Data, numerical checks and reproduction

The public case reuses the previously audited
[physical E/F subset and initialization](../PHYSICAL_DATA_VALIDATION.md):
168 training / 43 validation frames from the pinned official DeepMD `nopbc`
example. Its published methane-combustion E/F labels are retained unchanged,
including exact promotion from source float32. Attribution and CC BY 4.0 /
LGPL-3.0-or-later distribution details are in [notices](THIRD_PARTY_NOTICES.md).
The existing 1,000-update E/F-weight-one request is used, without the later loss-weight
diagnosis. The fit worsens force RMSE; neither this optimization nor its speed
establishes production model accuracy. The held-out frames are validation, not an
untouched final test. No new source arrays or fitted weights are tracked.

Old/new complete checkpoints, model exports, both moments, sampler order, final
metrics and compiled PTX identity are exact on all three cases. A historical
500-update public checkpoint resumes for 500 more updates with the new binary,
indexed storage and cache disabled, matching the uninterrupted result exactly.
Against native CPU training, maximum absolute differences across model parameters
and both moments are 1.76e-14 / 2.22e-16 / 3.67e-15 (public/medium/tiny), below the
fixed absolute 1e-10 replay check. Native-v1 export/reload E/F/virial differences
are zero. No checkpoint compatibility weakening is needed.

Verification used the existing dependency cache; nothing was installed:

```sh
cargo test --release --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda --lib
cargo test --release --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda --test cuda_reverse_training device_loss -- --include-ignored --test-threads=1
cargo test --release --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda --test cuda_reverse_training tiled_gradients -- --ignored --test-threads=1
cargo test --release --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda --test large_frame_training -- --include-ignored --test-threads=1
cargo check --locked --offline --manifest-path mlmd/native/Cargo.toml --no-default-features --bins
```

All 21 library, four device-loss, one tiled grouping/failure and two large-frame
tests pass. They cover grow/shrink/reorder, odd-sized integer offsets, signed zero,
PBC/self images, mixed E/F/V seeds, clipped Adam, later-tile failure, and 32/83/128/256
atom frames. GPU gradient error is at most 4.45e-15 on the existing focused cases;
the analytic finite-difference check is 4.85e-9. Large-frame E/F/V error is at most
7.11e-15 and its eight-parameter finite-difference check is 1.96e-10. CPU-only build passes.
This milestone has not been executed on Linux CUDA.

For the process comparison, prepare `physical`, `medium`, `tiny` directories under
one local prepared-input directory. Each contains `initial.json`, `dataset.jsonl`
and `profile.request.json`; use the recorded input hashes/configs. Public preparation
is defined by the linked physical-data receipt. Synthetic inputs use the existing
`tools/replay_tiled_training.py` recipe and `tools/replay_device_loss.py` configuration
with final-only checkpoint intervals. Keep generated inputs/outputs ignored.
Save the baseline executable before rebuilding. Then, in an authorized GPU window:

```sh
python mlmd/native/tools/replay_transfer_batching.py \
  --before-exe BASELINE_TRAINER --after-exe CURRENT_TRAINER \
  --prepared-dir PREPARED_INPUTS --output-name NEW_REPLAY --cpu
```

The script refuses to reuse its output directory, preserves labels and loss scales,
checks exact state/metrics/PTX and resource/work counts, and emits per-case progress
receipts plus `acceptance.json`. Final local evidence was collected under ignored
`fixtures/generated-deepmd/transfer-batching-01/replay-final`.

Remaining costs include NVRTC/driver startup, host geometry and stable reduction,
CPU assessment, per-update parameter/moment downloads and duplicated host model
processing. No general GPU speedup or speedup over DeePMD-kit is claimed.
