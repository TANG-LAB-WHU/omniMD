# GPU prediction and detached loss seeds for tiled training

This optional FP64 path computes current model E/F/virial predictions and detached
loss seeds on the GPU, then reuses primal network intermediates for the existing
mixed reverse derivatives. It preserves the supported tanh-network architecture,
optimizer batch, sampler, E/F/V loss and Adam boundaries. It is an implementation
optimization, with no new descriptor or differentiation algorithm claim.

The default tiled path keeps CPU prediction/seeds. On the measured 128-update
synthetic cases, the new path improved medium-case whole-process time by 1.15x
but made the tiny case slower (605 to 756 ms). Select it explicitly and measure
the actual workload. This is not CPU-free training or general DeepMD compatibility.

## Enable in steps

1. Build with the existing official NVIDIA Driver/NVRTC installation and existing
   dependencies; nothing is installed or globally configured by the trainer:

   ```sh
   cargo build --release --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda --bins
   ```

2. Start with a valid [full-training request](FULL_TRAINING.md), explicit
   `"device":"cuda:0"`, `config.cuda_gradient="reverse"` and
   `config.cuda_tiling={"max_frames":16,"max_atoms":512}`. Absent `cuda_loss`
   or `"cuda_loss":"cpu"` retains the reference tiled executor and v7 identity.
   Verify the dataset, explicit validation split and supported model first.

3. For a new optimizer run, add `"cuda_loss":"gpu"` inside `config` and select
   a new output directory. The complete relevant config fragment is:

   ```json
   {
     "cuda_gradient": "reverse",
     "cuda_tiling": {"max_frames": 16, "max_atoms": 512},
     "cuda_loss": "gpu"
   }
   ```

   Run `omnimd-native-train request.json`, or the engine's existing
   `omnimd-mlmd train-native request.json` forwarding command. All other required
   request fields, model/dataset formats and label rules stay unchanged.
   [Minibatches](MINIBATCH_TRAINING.md), [indexed storage](INDEXED_TRAINING.md),
   [geometry caching](FIXED_GEOMETRY_TRAINING.md) and
   [learning-rate schedules](LEARNING_RATE.md) remain optional as documented.

4. Inspect `report.json`: backend must be `cuda_tiled_device_loss_f64_v1` and
   `device_prediction_frames` must count the gradient-path frames. Check effective
   capacity, gradient and whole-process timings. The host
   `prediction_and_seeds_ms` field is zero in this mode; GPU prediction/seed work
   is included in `kernels_and_sync_ms`, so zero does not mean free computation.

The Rust constructor is `CudaFullTrainer::new_device_loss`. CPU execution,
hyperdual execution, missing tiling, unsupported architecture and insufficient
capacity fail explicitly; there is no implicit fallback.

## Arithmetic, reuse and execution boundary

For each directed center/neighbor image edge, let `delta` be its full periodic
image displacement, `R` its normalized four-component environment row and
`J=dR/d(delta)`. The first-order reverse computes `dE/dR` through **both** the
direct descriptor contraction and the radial embedding input. Then:

```text
q = transpose(J) * dE/dR
F_center += q; F_physical_neighbor -= q
W[a,b] -= delta[a] * q[b]
a = 2 * es^2 * (E - E_label)
v = 2 * fs^2 * (F - F_label)
C = 2 * vs^2 * (W - W_label)
u_delta[b] = v_neighbor[b] - v_center[b] + sum_a delta[a] * C[a,b]
R_u = J * u_delta
grad_theta L = grad_theta(a*E - E_u), with a,v,C held fixed
```

`es`, `fs` and `vs` are the existing per-frame scales from
[FULL_TRAINING.md](FULL_TRAINING.md) and [VIRIAL_TRAINING.md](VIRIAL_TRAINING.md).
They use the **original optimizer batch count**, including a partial final batch;
tile counts never replace it. The objective is the sum of squared scaled
residuals, including all nine virial entries when enabled. E is in eV, F in
eV/angstrom and W in eV; W is negative row-strain energy derivative, without
volume division. Labels retain their existing required/optional semantics.
Disabled virial seeds are explicitly zero; absent virials are not dataset labels.

GPU residuals and cotangents are detached numerical data. They are not
differentiated again. Training uses analytic mixed derivatives; finite differences
are verification only. Normalized padded rows participate in neural values but
have no geometric scatter or tangent. Self-image forces cancel at the same
physical atom while their full-displacement virial contribution remains.

Every tile launches ten kernels in one ordered stream: primal embedding, primal
fitting and input reverse, edge gradients, frame E/F/V gather, detached seeds,
coordinate directions, cached embedding directional propagation, cached fitting
mixed reverse, embedding mixed reverse and parameter reduction. There is one
synchronization per tile. Force gathering scans only the owning frame's edges,
uses stable order and needs no floating-point atomics.

Primal values, activations, descriptor values and contracted environment values
are reused only within that tile. Directional halves and adjoints are rewritten.
All tiles in an optimizer batch use the same weights; only after every tile
succeeds are frame gradients reduced in original order, clipped once and passed
to one Adam update. Any preparation/read/kernel/nonfinite failure discards the
batch result and poisons the trainer without committing an optimizer update.

CPU work remains: dataset reads/validation, neighbors and normalized geometry,
normalization/schedule scalars, stable atom/frame gradient reduction, clipping,
candidate validation, assessment, checkpointing and export. GPU work includes
predictions, seeds, mixed neural-parameter derivatives and Adam. All supported
weights, biases and present timesteps train; descriptor settings, normalization,
type mapping and atomic offsets stay fixed. No per-frame CPU prediction check
runs in the new gradient path; independent parity tests provide that validation.

Optional Rust `set_device_loss_diagnostics(true)` retains predictions, seeds and
normalized directions for only the final tile of the last successful batch,
before its update. It is disabled by default and cleared before the next batch.
Normal execution downloads only atomic parameter derivatives and frame seed/
objective records, rather than downloading E/F/V for CPU seed construction.

## Checkpoint and resource limits

The new mode requires `omnimd-native-full-checkpoint-v8`, taking precedence over
v7 tiling and v6 schedules. Loss mode, tile capacities and the existing immutable
configuration are checked on restart, together with backend, device ordinal/name,
driver API version and actual compiled PTX SHA-256. A nonzero v7 checkpoint cannot
be silently resumed as v8. To switch mode, explicitly initialize a new optimizer
from exported `model.json`. Cache and storage modes may change on restart.
Default/absent CPU loss preserves historical serialization and the old CUDA PTX.
Exact restart is verified in the tested build/runtime, not across devices,
platforms or builds. PTX identity alone does not identify all host code.

The architecture and tile policies are unchanged: at most 2048 neural parameters,
widths at most 32, 64 selected slots, eight types, eight layers per network,
**64 atoms per frame**, and requested tiles of 1..32 frames / 1..1024 atoms.
The planner may lower effective capacity. No frame is split or truncated.

The policies remain 50 million estimated scalar terms, 64 MiB explicit device
buffer payload and 32 MiB host tile payload. New owners, full-image displacements,
Jacobians, labels, scales, predictions, seeds, edge gradients and primal cache
are counted. Optional final-tile diagnostic storage is conservatively reserved.
The work estimate adds first-order forward/reverse and bounded frame-gather work;
it is an acceptance heuristic, not an exact FLOP count or runtime ceiling.
`cuda_tiling.device_loss` reports extra bytes and estimated terms.

Device buffers retain peak capacity; growing buffers are freed before replacement.
These are payload limits, **not total VRAM or RSS limits**: context/module/kernel
local memory, allocator overhead, CPU geometry preparation, one decoded frame,
geometry cache/index, model/optimizer and report/checkpoint storage are outside
them. The separate geometry-cache limit still applies. Process working set is
measured independently, not capped by the tile policy.

## Verification and measured results

From the repository root, with existing offline toolchain/dependencies:

```sh
cargo test --locked --offline --manifest-path mlmd/native/Cargo.toml --no-default-features
cargo test --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda
# Only within an authorized GPU window:
cargo test --release --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda device_loss_ -- --ignored --nocapture --test-threads=1
python mlmd/native/tools/replay_device_loss.py --eval-exe EVAL_EXE --train-exe TRAIN_EXE --reference-exe PRE_DEVICE_LOSS_TRAINER --output-dir NEW_IGNORED_DIRECTORY
```

The reference binary is the verified native code at integration commit
`f6030da6cef085e37bd2d135878c591218f30b2e` (native content equal to `cce7f8c`).
The replay uses existing binaries, original synthetic teacher fixtures and Python
standard library only. It does not download dependencies or weights.

Windows / RTX 4090 / installed CUDA 12.8 NVRTC / driver API 13040 passed 40 CPU
tests, 57 CUDA-feature tests without GPU execution, and four actual GPU tests:

- Heterogeneous nonperiodic, triclinic and periodic self-image fixtures; separate
  E/F/V and joint losses. Maximum absolute errors: GPU E/F/V vs CPU `4.44e-16`,
  seeds vs independently calculated CPU seeds `8.88e-16`, parameter gradients vs
  CPU-prediction tiled reverse `4.44e-15`.
- A nonconstant-embedding analytic dimer checks both force derivative paths and
  all nine virials. Independent CPU-loss central parameter differences
  (`h=1e-5`) differ by at most `4.84e-9`. Equal predictions/labels give exactly
  zero loss, seeds, directions and gradients.
- 32 frames / 1024 atoms plus a tail, shrinking/reusing buffers, three clipped
  Adam updates: gradient/parameter/moment differences at most `4.11e-15`.
  A nonfinite later tile leaves weights/moments unchanged and blocks Adam.
- Scheduled seven-update runs exactly resume at steps 2, 3 and 4 across indexed/
  memory and cache changes. Mode/schema/capacity/PTX alterations are rejected.

Each process workload uses 128 updates, 120 training / eight validation frames,
shuffled batches of 32 (partial 24), explicit exponential schedule and 3840 frame
visits. Three alternating-order repeats, with startup, validation, assessment,
checkpoint/export IO included, produced these medians:

| Synthetic workload | CPU loss whole process | GPU loss whole process | CPU/GPU ratio | Gradient ratio |
| --- | ---: | ---: | ---: | ---: |
| 112 parameters, 3/4 atoms per frame | 604.53 ms | 756.03 ms | 0.80x | 1.26x |
| 466 parameters, 32/64 atoms per frame | 3505.78 ms | 3060.90 ms | 1.15x | 1.40x |

There is no speed threshold. The tiny-case initialization median rises from
329 to 523 ms; its short gradient saving does not offset that cost. In the medium
case, CPU prediction/seeds previously took about 1005 ms. New combined GPU kernel/
sync time rises from about 264 to 616 ms, uploads from 161 to 255 ms and geometry/
packing from 20 to 60 ms. Moving work is not eliminating it. CPU assessment still
takes about 446 ms. Cache-off 64-step medium resume takes 6249 ms, demonstrating
remaining geometry cost; that resume is a correctness check, not a speed comparison.

Actual tiny tiles reach 16 frames / 62 atoms (256 tiles); medium reaches four
frames / 128 atoms (1763 tiles). Effective requested-512 atom capacity is 512 and
142 respectively, reduced by unchanged policies. Each tile now has ten kernels
rather than four. Peak explicit device buffers are 757,996 / 3,530,740 bytes and
host tile payloads 1,355,008 / 845,888 bytes. Maximum observed whole-process working
sets are 137,682,944 / 158,306,304 bytes. Memory reduction is not claimed.

Historical v7 PTX and optimizer states remain exact; new versus historical final
parameters/moments differ by at most `6.94e-16`. Same-backend repeated runs and
64+64-step restarts match exactly; exported model reload E/F/V differences are
zero. Twelve invalid CLI requests fail before creating output directories.
[Compact evidence](fixtures/device-loss-training-report.json) records identities,
hashes, phase/process measurements and checks. Generated datasets, weights,
binaries and raw logs remain ignored. Linux CUDA, larger/physical workloads,
multi-GPU, architecture expansion and speed relative to DeepMD are unverified.
