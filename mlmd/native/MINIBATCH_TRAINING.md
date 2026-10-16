# Deterministic minibatch training

The existing `omnimd-native-full-train-v1` request accepts optional
`config.minibatch`. This changes sampling for the supported full neural-parameter
trainer on CPU or explicit CUDA; it does not extend the model architecture,
physics, dataset limits or readout-only trainer. The existing default remains
full-batch. Absent or null `minibatch` takes the historical path and omits all
sampler fields from serialization. No new dependency is required.

## Step 1: choose a batch and order explicitly

Add this object to the configuration in [FULL_TRAINING.md](FULL_TRAINING.md):

```json
"minibatch": {
  "batch_size": 2,
  "seed": 20261016,
  "order": "shuffle_each_epoch"
}
```

All three fields are required. `batch_size` is an integer in 1..1024. `seed` is
an explicit unsigned 64-bit integer, separate from the train/validation split
seed. `order` is `sequential` or `shuffle_each_epoch`. Sequential order retains
ascending original dataset indices; its seed is still recorded and checked.
Validation indices never appear in a batch. Existing dataset byte SHA256,
source hashes, index/ID records and explicit or seeded partition rules apply.

Each epoch visits every training frame exactly once. Batches never cross epoch
boundaries, and the last short batch is neither dropped nor filled from the
next epoch. With five training frames and batch size two, sizes are `2,2,1`.
A batch size larger than the selected training count is allowed and produces
one short batch per epoch. A sequential batch at least as large as the training
set preserves full-batch arithmetic, but still uses explicit minibatch state.
Shuffling all frames changes reduction order and is not promised bit-identical
to full-batch arithmetic.

## Step 2: interpret loss and optimizer updates

For the actual current batch `B`, let `b=|B|`, `n_f` be the frame's atom count,
and `dE`, `dF`, `dW` be prediction minus label. The loss is:

```text
L_B = (1/b) sum_(f in B) [
        wE (dE_f / (n_f sE))^2
      + wF/(3 n_f) sum_q (dF_fq / sF)^2
      + wV/9 sum_ab (dW_fab / (n_f sV))^2 ]
```

The last term exists only with `virial_loss`; it uses all nine row-strain
virial components without a volume conversion. There is no one-half factor.
The denominator is the actual short-batch count, never the requested size or
full training count. Current cotangents and all neural gradients are rebuilt
for each update. Global gradient clipping happens after accumulating the
current batch and before Adam. Zero-gradient parameters still receive the
usual moment decay. E/F labels remain required; enabling virial loss requires
all nine labels on every frame, including validation.

`steps` means additional Adam updates, not epochs. The learning rate is constant
per completed update; Adam bias correction uses the global update count across
restarts. A short batch receives one update at the same learning rate. There
is no claim that this has the same trajectory, convergence, or epoch weighting
as full-batch training. Initial/final training and validation metrics continue
to use their complete partitions and their full-set denominators. Validation
labels never enter gradients, a schedule, or model selection.

Positive derived normalization coefficients are checked for finite nonzero
values for both full-sized and short batches before optimizer initialization.
Existing model/data/workspace bounds remain in force. Minibatching does not
remove the 1024-frame in-memory dataset limit or CUDA per-frame work caps.

## Step 3: portable sampler state and exact restart

Explicit minibatching uses `omnimd-native-full-checkpoint-v5` on all backends,
with the existing explicit E/F or E/F/V loss contract and backend identity.
The `sampler` object stores algorithm `omnimd-epoch-splitmix64-v1`, seed, order
mode, zero-based epoch, cursor, absolute dataset-index permutation, and derived
RNG state. It describes the **next** batch. For `N` training frames, requested
size `B`, and `Q=ceil(N/B)`, after `t` completed updates:

```text
epoch  = floor(t/Q)
cursor = (t mod Q)*B
pending batch = order[cursor .. min(cursor+B,N)]
```

After the final short batch the state immediately becomes the next epoch with
cursor zero. There is no alternate cursor-at-end representation. For shuffling,
start each epoch with the ascending training indices, use descending
Fisher-Yates, and draw indices with unsigned rejection sampling:

```text
G = 0x9e3779b97f4a7c15
source = seed + G*epoch                 # wrapping u64
rng_state = SplitMix64.next(source)     # output, not the updated source
for i from N-1 down to 1:
    bound = i+1
    threshold = (-bound as wrapping u64) mod bound
    draw v = SplitMix64.next(rng_state) until v >= threshold
    swap order[i], order[v mod bound]
```

The serialized RNG state is the updated internal `rng_state` after these draws.
Sequential mode uses the same epoch-derived initial state without draws. All
integer operations are specified as wrapping u64, without platform floating
point. Golden vectors for seed 20261016, training indices `[0,2,4,5,7]`:

| Epoch | Permutation | Final RNG state |
| --- | --- | --- |
| 0 | 4,2,7,5,0 | 13274578833983480863 |
| 1 | 0,2,5,7,4 | 18024458041679499801 |
| 2 | 4,5,2,7,0 | 1678407417306061623 |

Resume derives the expected epoch directly from the global step, regenerates
that permutation in O(N), and verifies every sampler field exactly. It does
not replay prior epochs. Missing/corrupt state, changed batch size/seed/order,
changed partition/data, and changed numerical backend are errors. The existing
same-platform/toolchain/device/driver/PTX limits on exact numerical restart
remain; portable integer sampling does not imply cross-platform bitwise neural
training. Historical full-batch v1/v2/v3/v4 checkpoints remain on their original
path. Enabling minibatching while resuming one is rejected. Start a new optimizer
explicitly from its exported native model to change training semantics.

Candidate sampler progress is prepared before computing a gradient and committed
only with the successful model/moments/global-step update. Errors abort that
trainer invocation. The CLI writes consistent checkpoints using its existing
new temporary file, sync and rename operation. A failed later update or write
leaves prior completed checkpoints resumable. No partially advanced sampler is
published. `steps` may change on resume; other immutable settings and checkpoint
interval must still match.

## Step 4: combine with the optional fixed geometry cache

For CUDA only, `fixed_geometry_cache:true` prepares the entire selected training
pool once, independent of the current sampled order. Current neural values,
loss seeds, and directional inputs are still recomputed for every sampled frame.
The existing 64 MiB retained host payload cap, exact identity checks and fail-closed
rules apply. The cache can be toggled on resume without changing numerical state;
it rebuilds from the original dataset, not from checkpointed cache arrays.
See [FIXED_GEOMETRY_TRAINING.md](FIXED_GEOMETRY_TRAINING.md).

Reports add current-invocation `minibatch.start` and `end` sampler snapshots,
`frames_processed`, `partial_batches`, and a histogram `batch_size_counts`.
Counts exclude assessments and prior invocations. The ordinary training-frame
identity list remains the complete selected pool, not just the last batch.

## Step 5: reproduce acceptance

Build using the existing offline toolchain:

```text
cargo test --locked --offline --manifest-path mlmd/native/Cargo.toml
cargo test --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda
cargo build --release --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda --bins
```

Only during an explicitly authorized CUDA window:

```text
cargo test --release --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda --test minibatch_training cuda_minibatch -- --ignored --nocapture --test-threads=1
```

The CLI replay generates original synthetic teacher labels and checks exact
serialized resumes before/after a short batch, cache toggling, corrupt checkpoint
rejection, and CPU/CUDA parameter/moment differences. Substitute local executable
paths and a new ignored output directory:

```text
python mlmd/native/tools/replay_minibatch_training.py --eval-exe EVAL_EXE --train-exe TRAIN_EXE --output-dir mlmd/native/fixtures/generated-deepmd/minibatch-cpu-01
python mlmd/native/tools/replay_minibatch_training.py --eval-exe EVAL_EXE --train-exe TRAIN_EXE --output-dir mlmd/native/fixtures/generated-deepmd/minibatch-cuda-01 --device cuda:0 --cuda-gradient reverse --fixed-geometry-cache
```

The bounded periodic throughput replay uses the existing original 466-parameter,
64-atom separated-pair fixture, six training frames, two validation frames, and
100 updates. With batch size four it processes 300 frames in batches `4,2`, with
50 short batches. Three fresh subprocesses per cache mode rotate execution order:

```text
python mlmd/native/tools/replay_cuda_reverse.py --eval-exe EVAL_EXE --train-exe TRAIN_EXE --output-dir mlmd/native/fixtures/generated-deepmd/minibatch-medium-01 --compare-geometry-cache --periodic --minibatch-size 4 --minibatch-seed 20261016
```

[Compact evidence](fixtures/minibatch-training-report.json) records 29 default
CPU tests, 42 CUDA-feature CPU tests, the explicitly run test covering both CUDA
gradient executors, and exact historical v1/v2/v3/v4 full-batch checkpoint replay
from step 150 to 300. The 31-update minibatch CLI replay agrees with CPU parameters
and moments to 3.84e-15 absolute and resumes exactly at steps 2, 3 and 4. The
periodic throughput case has identical cache-off/on model and optimizer state;
complete-process medians were 3120/771 ms on the recorded RTX 4090 environment.
This comparison uses identical minibatch work in both cache modes. Timing uses actual processed frames and separates
reported gradient time from whole-process time. These tiny bounded cases do
not establish production throughput, full-batch speedup, physical accuracy or
convergence equivalence. Generated weights, datasets and raw logs stay ignored.

The sampler reuses the repository's existing SplitMix64 arithmetic; its standard
reference is Sebastiano Vigna's [public-domain SplitMix64 implementation](https://prng.di.unimi.it/splitmix64.c).
The epoch policy, state validation, integration, tests and replay are original
MIT code. No third-party training implementation or new dependency is vendored.
