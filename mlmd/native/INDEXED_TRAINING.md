# Bounded indexed datasets for native minibatch training

The native full-neural-parameter trainer can read the existing
`omnimd-dataset-jsonl-v1` file through a bounded offset index. It builds the index with one complete strict input scan, then loads one frame
at a time in the exact
minibatch order. Training, full-partition assessments, reload verification and
optional geometry preparation share that access path. There is no eager batch
of decoded frames, new data representation, persisted sidecar, or new dependency.

## Step 1: select storage explicitly

Add a top-level field to an existing `omnimd-native-full-train-v1` request:

```json
{
  "schema": "omnimd-native-full-train-v1",
  "dataset_access": "indexed",
  "device": "cpu",
  "initial_model": "initial.json",
  "dataset": "dataset.jsonl",
  "output_directory": "trained",
  "config": {
    "split": {"mode": "explicit", "validation_ids": ["set.000/1025"]},
    "steps": 19,
    "minibatch": {"batch_size": 128, "seed": 20261016, "order": "shuffle_each_epoch"},
    "energy_weight": 1.0,
    "force_weight": 1.0,
    "virial_loss": {"weight": 0.7, "scale_ev_per_atom": 0.2},
    "energy_scale_ev_per_atom": 0.1,
    "force_scale_ev_per_angstrom": 0.1,
    "learning_rate": 0.005,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "gradient_clip_norm": 10.0,
    "checkpoint_every": 16
  }
}
```

Run `omnimd-native-train REQUEST.json`; the engine's existing forwarding command
passes the request unchanged. `dataset_access` accepts `memory` (the default)
or `indexed`. Indexed access requires explicit `config.minibatch`; there is no
fallback to memory. Readout-only requests do not accept this option. Relative
paths and new-output-directory rules remain unchanged.

The default reader keeps its 128 MiB file/16 MiB record limits, and ordinary
full training keeps its 1024-frame limit. Indexed mode explicitly supports up
to 100000 frames and 16 GiB of input, with stricter parsing bounds below. The
acceptance case actually crosses both old limits; the full 100000-frame/16-GiB
ceiling has not been benchmarked. Model architecture, parameter, neighbor-slot,
AD/workspace, CUDA work caps and the 64-atoms-per-frame bound are unchanged.

## Step 2: build a bounded index

The scanner keeps byte offset, byte length, SHA256 of the exact line, original
ID and atom count for each frame. Consecutive index, unique nonempty ID, strict
header/completed placement, final newline, frame/type/unit/label validation and
no trailing records remain mandatory. Hashes include whitespace and line endings.
The complete raw-file SHA256 remains the checkpoint identity; source hashes are
provenance, not a replacement for that hash.

The implementation uses the shared `Frame` type. Serde remote decoding applies
indexed parser limits to that type; it does not introduce another coordinate or
unit representation. The indexed subset has these explicit bounds:

| Resource | Indexed limit |
| --- | --- |
| File | regular file, at most 16 GiB |
| Frames | 100000 |
| Input record, including newline | 256 KiB, checked while reading |
| Positions, species, force vectors | at most 64 atoms; shape/labels must agree |
| Frame ID | 256 UTF-8 bytes |
| Header type map | 118 entries, still validated as unique chemical symbols |
| Header source text | 64 KiB |
| Header selected sets | 4096 strings, also bounded by the record input limit |
| Source hash map | 4096 entries, keys at most 256 bytes, 64-hex values |
| Retained index payload budget | 64 MiB |

Index accounting includes vector capacity, entry/hash/offset storage, boxed IDs,
header string/vector capacities and provenance metadata. Provenance tree nodes
receive a conservative 1024-byte allowance per entry for this implementation.
Duplicate-ID validation sorts a temporary vector of borrowed string references;
it never duplicates the ID strings. The report records its scratch size. Vector
reallocation may temporarily retain old and new storage during index construction.
The scanner's buffered reader and whole-file verifier use 64 KiB buffers. Raw
record vector capacity may exceed record length through normal Vec growth; both
largest length and observed peak capacity are reported.

The 64 MiB budget is an **index payload budget, not a process memory limit**.
It excludes allocator overhead, serde's bounded per-record temporary objects,
construction scratch, model/AD/CUDA workspaces, partition lookup tables, sampler
permutations, report identities and checkpoint serialization. Arrays and metadata
have explicit count/input bounds, but the standard allocator's exact overhead is
not promised. The trainer's decoded-frame lifetime is one loop iteration; raw
record bytes and parser temporaries are dropped after each read. Public library
callers can retain returned owned frames themselves and must budget that memory.

Partition/sampler/report/checkpoint memory remains O(frame count), with the
explicit 100000-frame and ID bounds. This is bounded indexed access, not an O(1)
streaming optimizer. Reports separately record retained index payload, maximum
successfully decoded frame payload, raw input/buffer sizes and IO/integrity timing.
`frames_decoded_per_read:1` describes the access operation, not a measurement of
all allocations or arbitrary caller-held frames.

## Step 3: preserve deterministic sampling and restart

The existing split and versioned SplitMix64 sampler operate on original absolute
frame indices. A seek uses the indexed offset and length, verifies the exact line
hash before parsing, and checks the decoded index/ID/atom count. No read-ahead,
worker scheduling, frame sorting or batch aggregation changes gradient reduction
order, actual-batch loss normalization, Adam step/LR semantics, or metrics.
Explicit validation-ID lookup uses a bounded temporary map to avoid quadratic
lookup when the selected validation set grows.

Storage mode is runtime-only and is not part of optimizer identity. There is no
checkpoint schema bump for storage mode: explicit minibatching uses v5 unless
an [explicit learning-rate schedule](LEARNING_RATE.md) selects v6. The index is rebuilt
and all input validated on every invocation. Resume still requires the exact raw
dataset hash, same model/loss/backend/split/batch/seed/order and valid sampler state.
Small supported datasets can switch `memory` to `indexed` and back with identical
model, moments and sampler. Large indexed datasets cannot resume in memory mode
unless they independently satisfy its old limits. Historical full-batch requests
and checkpoints retain the old memory path.

## Step 4: fail closed on changed input

A run retains the opened file handle. On Windows it denies write/delete sharing,
so an incompatible writer, truncation or path replacement fails at the OS boundary.
On other platforms, path replacement leaves the run bound to the original opened
file; a new invocation opens the current path and must match the checkpoint hash.
Linux/macOS execution was not exercised in this tranche. The later
[Linux CUDA check](../LINUX_CUDA_VALIDATION.md) verifies a small indexed-to-memory
training continuation under WSL2; it does not establish platform mutation-race
coverage. macOS execution remains unvalidated.

Length and modification time from the retained handle are checked around reads.
They are only early checks: per-frame SHA256 catches same-size mutations even
when the original modification time is restored. Before each checkpoint callback,
after final metrics, and before CLI model/report publication, a fixed-buffer
whole-file hash must match the original SHA256. This also checks unread frames,
header, completed marker and source hashes. IO/integrity failures invalidate the
reader; reopening and full validation are required. A failed read/gradient/update
or integrity check publishes no new successful state; earlier completed checkpoints
remain available. The native CLI's existing checkpoint temporary-file/sync/rename
behavior is retained.

The portable checks are not an atomic filesystem snapshot against a concurrent
adversarial writer. Each consumed frame is checked against its indexed bytes and
publication requires successful whole-file verification; immutable input is the
intended operating condition. Whole-file verification costs O(file bytes) per
checkpoint and is reported separately. This first implementation favors a clear
integrity boundary over frequent-checkpoint IO throughput.

## Step 5: combine with bounded fixed geometry

CUDA hyperdual and reverse training can enable the existing
`config.fixed_geometry_cache:true`. Its preflight uses indexed atom counts, then
prepares selected training frames one by one. It retains only the existing geometry
entries, never complete labeled records. The unchanged 64 MiB geometry payload cap
is independent of the index budget; a pool that does not fit fails before CUDA
initialization. It is all-or-error: there is no silent eviction, partial cache or
fallback. Disable it explicitly for larger pools. Cache and storage modes can be
toggled together on restart without changing numerical state within a compatible
backend. Geometry hashes/identity checks and current-weight/loss recomputation
remain as documented in [FIXED_GEOMETRY_TRAINING.md](FIXED_GEOMETRY_TRAINING.md).

## Step 6: reproduce the focused evidence

```text
cargo test --locked --offline --manifest-path mlmd/native/Cargo.toml
cargo test --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda
cargo build --release --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda --bins
python mlmd/native/tools/replay_indexed_training.py --eval-exe EVAL_EXE --train-exe TRAIN_EXE --output-dir mlmd/native/fixtures/generated-deepmd/indexed-training-01
```

The Python replay uses only the standard library and original synthetic teacher
labels. Its 2051-frame file is deliberately whitespace padded to 128.19 MiB, so it
is a reader/identity/memory test, not a representative physical dataset or a
throughput comparison. It selects 2050 training frames and one validation frame,
performs 19 updates at batch size 128 (2306 processed frames, including a two-frame
tail), and exactly resumes from step 16 before that tail. The measured native
process peak working set was 12.83 MiB; retained index accounting was 319722 bytes,
maximum decoded frame payload 584 bytes, and raw record buffer capacity 130436
bytes. These measured Windows process numbers are not universal hard limits.

The small replay matches all CPU checkpoint fields and initial/final E/F/V metrics
exactly across storage modes and resumes in both directions. Tests also cover
malformed/truncated/trailing records, duplicate IDs, per-record/shape bounds,
same-length/same-time mutation, poisoning and geometry-budget rejection. Actual
historical CPU v1 full-batch (150 to 300) and v5 minibatch (2 to 31, memory to indexed)
checkpoints reproduce their saved final states exactly.

During an explicitly authorized GPU window only:

```text
cargo test --release --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda --test indexed_training indexed_cuda -- --ignored --nocapture --test-threads=1
```

This test covers both CUDA gradient executors with exact memory/indexed state,
geometry-cache hits and serialized storage/cache-switch restart. See
[compact evidence](fixtures/indexed-training-report.json) for test counts, hashes,
measured IO/memory and GPU release time. Generated weights, padded datasets,
checkpoints and raw logs remain ignored. There is no claim of algorithmic novelty,
physical accuracy, distributed training or production-scale IO performance.


## Startup validation consolidation

The trainer now obtains its parameter layout from the same complete checkpoint
and device validation that authorizes training. Previously it discarded that
layout and ran checkpoint/dataset validation again. The CLI also called the
public validator immediately before the trainer repeated it. Removing these two
redundant calls saves two complete frame passes per CLI invocation.

The public constructor and public `validate_for_device` retain their checks.
The trainer always validates on entry, before CUDA construction or checkpoint
callback invocation; the CLI creates output directories only inside/after that
validated training operation. There is no validation token or cache across calls:
callers may change a public checkpoint, model or dataset between invocations.
Index construction, per-read metadata/hash/identity checks, model/loss/device/
sampler guards, and every publication whole-file hash are unchanged. This means
one validation scan inside the trainer, not one scan for the entire CLI process.

Full-training reports add these diagnostic fields without changing request,
dataset, model, checkpoint or optimizer identities:

- `startup_timing.dataset_open_ms`: initial complete scan and index construction
  (or the existing memory reader).
- `startup_timing.state_preparation_ms`: initial model parsing and constructor
  validation, or checkpoint parsing on resume.
- `startup_timing.before_trainer_ms`: cumulative CLI elapsed time before entering
  the trainer, including the preceding phases.
- `timing.startup_ms`: trainer entry through validation, initialization and full
  initial train/validation assessments, immediately before the first update.
- `startup_timing.before_first_update_ms`: the sum of the preceding cumulative
  CLI time and trainer startup time. It excludes process launch.
- `timing.startup_frame_reads`: optional cumulative indexed frame-read counter
  at that boundary, including earlier caller reads; index construction is separate.

Existing `timing.initialization_ms` still ends before initial assessments.
Cumulative fields overlap; do not add all timing fields. Timing/counter fields
are diagnostics, never restart identity.

A controlled Windows CPU comparison used the existing 2051-frame, 134414893-byte
synthetic input, batch size 128, 19 updates fresh, or three additional updates
from its historical step-16 checkpoint. Both binaries included identical timing
instrumentation. Baseline/candidate order alternated across three repeats per
scenario; these are repeated local-file measurements with warm filesystem cache.

| Scenario | Startup median, old -> new | Complete process median, old -> new | Startup frame reads, old -> new | Whole-file checks |
| --- | --- | --- | --- | --- |
| Fresh 19 updates | 945.07 -> 687.19 ms (27.29% lower) | 2127.12 -> 1860.17 ms (12.55% lower) | 10255 -> 6153 | 4 -> 4 |
| Resume step 16, three updates | 819.30 -> 557.27 ms (31.98% lower) | 1518.39 -> 1261.83 ms (16.90% lower) | 8204 -> 4102 | 3 -> 3 |

Each individual invocation removes exactly 4102 logical frame reads. The resumed
row measures only the resumed invocation, not a combined fresh-plus-resume run.
All checkpoint fields, model parameters, Adam moments, sampler state, initial/
final E/F/virial metrics and native-v1 reload results match exactly within each
scenario. Fresh midpoint and final state also match the saved pre-change replay;
resume matches its final state except the intentionally different requested step
count. Invalid step count, CPU geometry-cache request and malformed checkpoint
moments all fail before creating an output directory.

This whitespace-padded synthetic input emphasizes parsing/hash/IO work. The
numbers do not establish physical accuracy, general training acceleration,
cold-disk behavior, Linux performance or production-scale throughput. Remaining
costs include the initial strict scan, independent constructor validation on fresh
runs, one trainer validation, full initial/final assessments, reload verification,
and O(file bytes) publication hashes. These checks remain enabled. No GPU was used
for this change; GPU-dependent tests remained ignored.

Reproduce the input with `replay_indexed_training.py` above, then compare:

```text
python mlmd/native/tools/replay_indexed_startup.py --baseline-exe BASELINE_EXE --candidate-exe TRAIN_EXE --fixture-dir mlmd/native/fixtures/generated-deepmd/indexed-training-01 --output-dir mlmd/native/fixtures/generated-deepmd/indexed-startup-01
```

The controlled baseline is base commit
`be573fe0f91d323932c85e47448b0aace5950f32` plus the same timing/helper refactor,
with the original duplicate call chain retained. To reconstruct it from this
change, temporarily restore exactly these two call sites, build release bins
with the same features, copy the baseline executable, then restore the candidate
source. Do not remove the timing instrumentation:

1. In the CLI full-training arm, insert
   `state.validate_for_device(dataset, &config, device)?;` immediately before
   the `before_trainer_ms` measurement.
2. At trainer entry, replace
   `let layout = state.validated_layout_for_device(dataset, config, device)?;`
   with `state.validate_for_device(dataset, config, device)?;` followed by
   `let layout = state.validate(dataset, config)?;`.

[Startup evidence](fixtures/indexed-startup-report.json) retains every measurement,
binary/source/input hashes and regression counts. The focused regression asserts
one trainer validation scan plus initial assessment, and invalid-state rejection
before the save callback. The existing same-size/same-time mutation, unread-header
mutation, truncation, append, poisoning and Windows file-sharing tests also pass.
Generated models, datasets, checkpoints, binaries and raw logs stay ignored.
