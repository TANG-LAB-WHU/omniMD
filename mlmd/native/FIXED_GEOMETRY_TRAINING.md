# Fixed training-frame geometry cache

`config.fixed_geometry_cache: true` enables a bounded host cache for explicitly
selected, immutable training frames in CUDA full-neural-parameter training. It
works with both hyperdual and cached reverse gradient executors. The default is
false and is omitted from serialized configuration. CPU-only full training
rejects the option. Model/data shape limits and all existing work caps are unchanged.
This cache is separate from the moving-frame Verlet cache used by CUDA inference.

## Step 1: prepare immutable geometry

After dataset/split validation, prepare only the training indices, before creating
a CUDA context or advancing the optimizer. Each entry preserves the original
exact neighbor ordering, physical atom index, periodic image and full displacement,
normalized four-component environment `R`, and its displacement Jacobian `J`.
Padding preserves its normalized value and zero Jacobian. No coordinate wrapping,
neighbor truncation, cutoff approximation or floating-point precision change occurs.

Entries use fixed-size boxed arrays and retain bit-exact coordinate/cell snapshots,
resolved species types and a geometry specification. Checked arithmetic reserves
all retained cache payload before construction. The fixed 64 MiB cap counts the
cache/entry structures, type-name storage, specification arrays, identity snapshots,
rows, Jacobians, neighbor records and audit digest. It is a host cache payload
limit, not a process-RAM or GPU-memory limit: allocator overhead, original dataset,
current neural working buffers and temporary construction/search buffers are separate.
Over-budget requests fail explicitly; there is no silent partial cache or fallback.
If construction fails, the partial pool is dropped and no optimizer update occurs.

## Step 2: reuse only model-independent quantities

On every gradient call, the trainer checks the requested dataset index and exact
geometry identity. Identity includes coordinates and cell/periodicity (floating
bits, including signed zero), species/order, units, ordered model type map,
`se_e2_a` descriptor identity, neighbor capacities, both cutoffs, and every
normalization value. The immutable native-v1 model constructor already validates
format, precision and units. A changed or absent frame invalidates the pool and
stops the trainer; recreating a trainer/cache is required. This is not an adaptive
cache for moving atoms. SHA256 is recorded for auditing; reuse is guarded by exact
comparisons rather than trusting a hash or frame ID alone.

CPU inference and the CUDA gradient input builder borrow the same prepared rows
and neighbor records. CPU inference shares its existing atomic contraction code
with the uncached path. Neural embeddings, fitting outputs, energies, forces,
virials, loss residuals and cotangents are computed again using the current weights
and current labels. After that, every directional row is rebuilt as

```text
u_delta[b] = v_neighbor[b] - v_center[b] + sum_a delta[a] C[a][b]
R_u = J u_delta
```

Here `v` and `C` are current detached force/virial loss cotangents. The full periodic
displacement is retained, so self-image force directions cancel while virial
directions remain. Activations, parameter gradients and optimizer state are never
cached. Weight updates replace the current potential but retain the separate
immutable geometry pool. Validation/assessment continues through the uncached
reference path; geometry arrays still transfer to the GPU each gradient call.

## Step 3: restart without changing numerical identity

The cache is a runtime performance setting. Its arrays, hit count and construction
time are not checkpointed. The setting is reported/stored in configuration but
excluded from optimizer restart compatibility, allowing cache off-to-on and
on-to-off resume within the same numerical backend. The pool rebuilds from the
validated dataset and selected training indices on restart. Checkpoint schema,
loss contract, CUDA backend/device/driver/PTX checks and dataset provenance rules
remain unchanged. Exact equality excludes only the requested runtime cache flag
and additional-step count; neural weights, moments and all other state agree.

## Step 4: measured acceptance and scope

[Compact evidence](fixtures/fixed-geometry-training-report.json) records:

- 22 default CPU tests and 35 CUDA-feature CPU tests passing, including bitwise
  R/J and current-model E/F/V equality, triclinic/self-image geometry, normalization,
  signed-zero cell changes, all-or-error construction and memory-cap checks.
- Both CUDA gradient executors producing identical cache-on/off gradients and
  Adam state while neural weights, labels and active loss components change.
  Serialized resume in both cache-toggle directions is exact.
- An actual checkpoint from commit `f33d332`, step 150, resumed by the new binary
  to step 300 with cache off and on: both match the historical final state exactly.
- Three complete fresh-process runs per cache setting, alternating order, with
  exact repeated/cross-setting checkpoints. The synthetic workload uses 466 neural
  parameters, 64 atoms/frame, eight frames (six training/two validation), E/F/V loss
  and 100 updates. Model, dataset and executable SHA256 values are recorded.

| Synthetic geometry | Full process off / on | Training gradients off / on |
| --- | ---: | ---: |
| Separated-pair cluster | 865.5 / 818.4 ms | 432.2 / 375.6 ms |
| Same pairs in triclinic periodic cell | 6946.3 / 956.9 ms | 6348.9 / 366.9 ms |

Values are medians of three runs. Complete-process times include input, cache
construction, driver/NVRTC initialization, updates, assessment, checkpoints and
export/reload verification. Both cases retain 1,290,058 bytes for six cached frames;
cache construction takes approximately 0.62 ms and 23.0 ms respectively. The larger
periodic benefit comes from eliminating repeated CPU image enumeration in this
bounded synthetic case. It is not a production-size benchmark, evidence of physical
accuracy, or a general GPU speedup claim. Cluster improvement is modest; CUDA launch,
transfer, neural inference and assessment costs remain.

## Reproduce

```sh
cargo test --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda
# Only during an authorized exclusive GPU window:
cargo test --release --locked --offline --manifest-path mlmd/native/Cargo.toml \
  --features cuda --test cuda_reverse_training \
  fixed_geometry_gradients_adam_switching_restart_and_profile -- --ignored --nocapture --test-threads=1
python mlmd/native/tools/replay_cuda_reverse.py --eval-exe EVAL --train-exe TRAIN \
  --output-dir NEW_DIR --compare-geometry-cache
python mlmd/native/tools/replay_cuda_reverse.py --eval-exe EVAL --train-exe TRAIN \
  --output-dir ANOTHER_NEW_DIR --compare-geometry-cache --periodic
```

The existing 24-frame replay also accepts `--fixed-geometry-cache` with an explicit
CUDA device. Generated datasets, weights and raw results remain under ignored
`fixtures/generated-deepmd/`. Implementation and synthetic tests are original MIT
code; no dependency was added and no third-party cache implementation was copied.
No HAP data or private training was used.

For explicit minibatching, [indexed dataset access](INDEXED_TRAINING.md) can read
larger datasets one frame at a time. It preserves the existing memory-mode limits
and numerical/checkpoint semantics; its own parsing/index bounds and the unchanged
geometry payload cap are documented separately.

The cache also supports explicit [128/256-atom tiled CUDA frames](LARGE_FRAME_TRAINING.md)
within its unchanged 64 MiB payload budget. Trainer admission and total tile
budgets are checked separately before cache construction; parser/cache support
does not enlarge the default 64-atom training mode.
