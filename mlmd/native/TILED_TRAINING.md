# Bounded multi-frame CUDA training

The full supported tanh-network trainer can execute independent frames together
with the existing reverse CUDA algorithm. This is actual kernel batching: one
tile uploads packed inputs once, launches four reverse kernels, synchronizes once
and downloads atomic derivatives once. It does not alter the optimizer batch,
loss formulas, sampler order or learning-rate schedule. CPU remains the default.
This page describes tiled CPU prediction/seeds (v7). Optional
[GPU prediction/seeds](DEVICE_LOSS_TRAINING.md) reuse primal intermediates, launch
ten kernels per tile and select a distinct v8 identity under unchanged bounds.

## Request and compatibility

Build with the existing cuda feature and use an explicit CUDA device. Add this
inside the full-training request's existing config:

~~~json
{
  "cuda_gradient": "reverse",
  "cuda_tiling": {"max_frames": 16, "max_atoms": 512}
}
~~~

This fragment augments the complete request documented in
[FULL_TRAINING.md](FULL_TRAINING.md). It works with either the existing bounded
full batch or explicit [minibatches](MINIBATCH_TRAINING.md), and with memory or
[indexed JSONL](INDEXED_TRAINING.md) access. Indexed access still requires explicit
minibatches. The engine's existing request forwarding requires no shared-contract
or engine change. Absent/null cuda_tiling preserves the framewise executor and
historical v1-v6 serialization. CPU or hyperdual plus tiling is an error.

Tiling selects checkpoint omnimd-native-full-checkpoint-v7 and backend
cuda_tiled_reverse_f64_v1. Tile capacities are immutable restart fields.
Dataset/model/layout, split, loss, sampler, schedule, global-step, device, driver
and compiled-program checks all remain applicable. Changing executors or tile
capacities requires explicitly starting a new optimizer from the exported model.
Geometry caching and dataset storage mode may change on restart. Exact restart
was verified on the same Windows build and installed runtime; cross-platform,
driver or build equality is not promised.

## Computation and packing

Each frame retains its own coordinates, cell, neighbors and detached E/F/V loss
seeds. Packing concatenates independent centers into arrays indexed as follows:

- normalized rows and directional rows: [total_atoms, sum(sel), 4], float64;
- mapped center types: [total_atoms], int32;
- energy loss seeds: [total_atoms], float64, repeated only within each frame;
- explicit host boundaries: start/end atom offsets, predicted energy, objective
  and dataset index.

The device parameter order, slot-type routing, weights, moments and reverse
scratch layout are unchanged. The tiled module specializes the energy seed
argument from one scalar to an atom-indexed pointer; the historical framewise
CUDA source and compiled identity remain unchanged. Neighbor construction is
never performed on concatenated frames.

For each original frame, the unchanged reverse objective is:

~~~text
S = a E - E_u
u_delta[b] = v_neighbor[b] - v_center[b] + sum_a delta[a] C[a][b]
R_u = J u_delta
~~~

Actual periodic displacements, normalized padding and self-image virial
directions retain the semantics in
[CUDA_REVERSE_TRAINING.md](CUDA_REVERSE_TRAINING.md).
Every frame uses the original current minibatch denominator, including its
smaller final partial batch; the tile's frame or atom count is never substituted.

Frames are packed greedily in sampler order. A capacity-triggering frame becomes
the first frame of the next tile. Each final partial tile executes once. Per-frame
CUDA primal energies, including atomic offsets, are checked against CPU inference
with the existing 1e-11 * (1 + abs(E_cpu)) tolerance. Host reduction sums atoms
within each frame, then frames in original order; it never inserts a tile subtotal.
There are no floating-point atomics or mixed-precision conversions.

CPU work remains substantial: one-record loading, neighbor/environment preparation,
current E/F/V inference, detached loss seeds, frame reduction, clipping, schedule
scalars, assessment and serialization. GPU work is mixed parameter derivatives and
Adam. Every tile must succeed before a single Adam update or sampler commit.
Preparation, read, kernel or primal failures discard the local batch result and
poison the trainer. Earlier completed checkpoints remain usable.

## Resource policy

No model architecture bound is raised. The supported family still has at most
2048 neural parameters, widths at most 32, 64 selected neighbor slots, eight types
and eight layers per network, and **64 atoms per frame**. Unsupported models fail
explicitly. Normalization, type mapping, descriptor settings and atomic offsets
remain fixed.

User capacities are max_frames in 1..32 and max_atoms in 1..1024.
The planner reduces the effective atom capacity to fit all of these constraints:

| Policy | Scope |
| --- | --- |
| 50,000,000 estimated scalar terms | Entire tile: atoms * parameters * maximum_center_terms; also retains the single-frame check |
| 64 MiB device buffer payload | Weights, moments, uploaded gradient, metadata, routing, packed inputs, outputs and reverse scratch |
| 32 MiB host tile payload | Retained input capacities, boundaries, downloaded atomic gradients/energies and two reduction accumulators |

No frame is split. A dataset frame that cannot fit alone is rejected in host
preflight. Capacity reduction is reported; it does not truncate data or change
the selected batch. Device buffers retain their largest allocation; growing
scratch buffers are freed before replacement, avoiding simultaneous old/new
payloads. Host input capacities are allocated once per optimizer batch and reused
across its tiles. Indexed records are decoded and released one at a time.

These payload limits are **not total VRAM or process-RSS limits**. They exclude
CUDA context/module/driver and kernel local-memory allocations, allocator overhead,
the existing CPU inference workspace, one decoded/preparing frame, model/optimizer,
geometry-cache/index and report/checkpoint storage. The optional geometry cache
retains its separate 64 MiB payload cap. Windows subprocess PeakWorkingSetSize is
recorded separately as an observed process peak.

Reports add cuda_tiling with requested/effective capacities and planned
payload/work limits. cuda_gradient_profile counts processed frames, tiles,
gradient-kernel launches/synchronizations, maximum actual frames/atoms per launch,
and peak allocated device buffers/host tile payload observed in the gradient
path. Its launch counts exclude Adam. Phase wall times cover CPU prediction/seeds,
geometry, uploads, kernels plus sync, downloads and reduction; complete gradient
time also includes loading/validation/packing. Whole-process timing is external.

## Verification and measured scope

Commands from the repository root, using existing offline dependencies:

~~~text
cargo test --locked --offline --manifest-path mlmd/native/Cargo.toml --no-default-features
cargo test --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda
cargo build --release --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda --bins

# Only during an authorized GPU window:
cargo test --release --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda tiled_ -- --ignored --nocapture --test-threads=1
python mlmd/native/tools/replay_tiled_training.py --eval-exe EVAL_EXE --train-exe TRAIN_EXE --reference-exe PRE_TILE_TRAINER --output-dir NEW_IGNORED_DIRECTORY
~~~

The replay requires an existing pre-tile trainer binary. The recorded reference
is native code from integration commit
6bb52955315304b1a35842e6065afc7219388805; it is not downloaded or rebuilt by
the replay. All data and teacher labels are original synthetic fixtures.

Windows / RTX 4090 / installed CUDA 12.8 NVRTC, driver API 13040 checks passed:

- 40 CPU tests; 56 CUDA-feature tests requiring no GPU.
- Actual tiled GPU checks cover E-only, F-only, V-only and joint losses,
  heterogeneous atom counts, triclinic PBC, self-images, capacity-one execution,
  exact-fit tile boundaries, growth/shrink and changing species. Raw gradients,
  four successive Adam updates and moments exactly matched sequential reverse.
- A 32-frame / 1024-atom tile followed by a partial tail matched the reference.
  A later-frame failure after a completed tile preserved model/moments and
  prevented further optimizer use.
- Seven-step scheduled/minibatch runs exactly resumed at steps 2, 3 and 4 with
  indexed-to-memory and cache-on-to-off changes. Altered capacities, schema,
  backend and compiled-program identity were rejected.
- The actual CLI ran each workload for 128 updates over 120 training / eight
  validation frames, shuffled batches of 32 with 24-frame partial batches:
  3840 processed frames. Three alternating-order repeats matched historical and
  current framewise checkpoints exactly after removing execution/config metadata.
  Both workloads also matched a 64+64-step serialized restart exactly. Exported
  model E/F/V reload differences were all zero. Twelve invalid CLI requests were
  rejected before output creation.

The initial 32-step framewise profile identified transfer/launch/synchronization
overhead, rather than allocation on every frame, as the main removable cost.
Those GPU phases were about 81% of tiny-case gradient time and 42% of medium-case
gradient time. CPU prediction/seeds took about half of the medium gradient time;
host reduction was comparatively small.

Complete-process medians include startup, validation, assessment, checkpoint and
report IO. There is no benchmark pass/fail speed threshold:

| Original synthetic workload | Framewise whole process | Tiled whole process | Ratio | Gradient phase ratio |
| --- | ---: | ---: | ---: | ---: |
| 112 parameters, 3/4 atoms per frame | 1071.17 ms | 550.51 ms | 1.95x | 4.20x |
| 466 parameters, 32/64 atoms per frame | 4483.70 ms | 3567.77 ms | 1.26x | 1.31x |

Both use requested capacities of 16 frames / 512 atoms. The tiny workload actually
launches up to 16 frames / 62 atoms and uses 256 tiles instead of 3840 frame launches
(1024 vs 15360 gradient kernels). The medium work cap reduces capacity to 144 atoms;
actual launches contain up to four frames / 128 atoms, using 1763 tiles / 7052
kernels. Observed device buffer payloads were 598,004 and 3,200,388 bytes; host tile
payloads were 588,912 and 635,488 bytes. Planned device payloads at full effective
capacity were 4,905,404 and 3,598,276 bytes.

The largest process working sets across the three tiled repeats were 136,298,496
and 189,353,984 bytes, versus 136,527,872 and 156,692,480 framewise. Tiling does not
guarantee lower total memory. The medium cache-off resume took 10,745 ms for its
remaining 64 updates, demonstrating substantial CPU geometry cost outside the
tile optimization.

[Compact evidence](fixtures/tiled-training-report.json) records timings, hashes,
execution identities and checks. Datasets, weights, binaries, checkpoints and raw
logs remain ignored. No dependency or global configuration changed. Linux CUDA,
large physical workloads, larger architectures, multi-GPU, CPU-free training and
end-to-end speed relative to DeepMD remain unverified or unsupported. This is an
implementation optimization, with no new descriptor or algorithmic novelty claim.
