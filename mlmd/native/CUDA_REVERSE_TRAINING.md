# Cached reverse CUDA parameter gradients

The opt-in `config.cuda_gradient: "reverse"` executor computes all supported
neural E/F/optional-V gradients using cached forward directional pairs and
explicit reverse accumulation. The default remains `"hyperdual"` (omitted in
serialized default configuration); CPU remains the default device. All existing
[full-training bounds](CUDA_FULL_TRAINING.md), including the 50,000,000-term
policy, remain in force. This milestone removes repeated differentiated network
forwards per parameter; it does not expand supported architectures or dataset
sizes. Normalization, geometry settings and separate atomic offsets stay fixed.

## Usage and restart

Build with `--features cuda`, use an explicit `"device":"cuda:0"`, and add
`"cuda_gradient":"reverse"` inside the existing full-training request's `config`.
Optional virial supervision is unchanged from [VIRIAL_TRAINING.md](VIRIAL_TRAINING.md).
No new dependency or external AD library is required. CPU plus reverse is an
error, with no fallback. Engine forwarding requires no engine/shared changes.

Reverse checkpoints use `omnimd-native-full-checkpoint-v4`, an explicit E/F or
E/F/V loss contract, and backend `cuda_reverse_f64_v1`. Restart checks the
configuration, device, driver and compiled PTX identity. A nonzero-step checkpoint
cannot switch gradient executors. Export the native-v1 model and initialize a
new optimizer explicitly to change backends. Historical CPU v1, hyperdual v2 and
virial v3 checkpoint serialization and the hyperdual compiled source are preserved.
Exact restart is promised only under the recorded same-backend identity; floating
arithmetic order differs between backends, which are compared with tolerances.

## Algorithm

The CPU still evaluates the current E/F/V, constructs detached loss cotangents,
PBC neighbors, normalized environment rows and directional rows. With energy
cotangent `a`, force cotangents `v` and ordered virial cotangents `C`, the target is

```text
S = a E - E_u
u_delta[b] = v_neighbor[b] - v_center[b] + sum_a delta[a] C[a][b]
R_u = J u_delta
```

`delta` includes the actual periodic image. Padding retains its normalized value
and zero direction. Periodic self-image force directions cancel; virial directions
remain. Seeds refer to the immutable current weights and are not differentiated.
Four kernels share cached intermediates:

1. Each atom/neighbor slot evaluates its embedding network once, retaining each
   layer input/output pair `(x, x_u)` and pre-timestep activation `(h, h_u)`.
2. Each atom forms `B = G^T R / sum(sel)`, `D_mk = sum_b B_mb B_kb`, evaluates its
   fitting network, then reverses from output adjoints `(a, -1)` through fitting
   and descriptor. Both descriptor product operands accumulate, including `m=k`.
3. Each slot reverses the embedding using descriptor adjoints and its cache.
4. Each atom/embedding parameter sums its matching slots in fixed order. Fitting
   parameters already have a unique writer. CPU atom/frame reduction stays fixed.

Adjoints are partial derivatives with respect to each pair component. For a
`tanh` activation, `h=tanh(z)` and `h_u=(1-h*h) z_u`. Given activation adjoints
`g_h,g_hu`, use `g_z=g_h*(1-h*h)-2*h*h_u*g_hu` and
`g_zu=g_hu*(1-h*h)`. A weight receives `g_z*x + g_zu*x_u`; bias receives `g_z`.
A timestep receives `g_y*h + g_yu*h_u`, including at zero timestep. Residual
branches propagate both adjoints. There is no division by a saturated tanh slope
or zero timestep, finite difference training, floating-point atomic sum or full
Hessian. This is an original MIT implementation of standard reverse-over-forward
chain rules, not a new differentiation algorithm.

Network arithmetic now scales with network operations and neighbor slots rather
than replaying those operations for every parameter. Parameter output/reduction
still costs at least O(atoms * parameters), and per-slot parameter buffers remain.
Host inference/geometry, transfers, launch overhead and the serial per-atom fitting
thread remain bottlenecks. No production scaling or universal speedup is claimed.

## Memory and timing

Checked planning bounds device buffer payload at 64 MiB, including cached pairs,
descriptor adjoints, private slot gradients, atom gradients/energies, parameters,
moments, uploaded batch gradient, metadata and input rows. For maximum layer count
`L` and network input/output width `T`, each cached network uses `(4L+2)T` doubles.
Buffers retain the largest processed frame's capacity. Before growth, all growing
scratch buffers are freed before replacements are allocated, avoiding simultaneous
old/new scratch allocations. Persistent optimizer state remains resident.
`reverse_resources(model, maximum_atoms)` reports this payload bound; it excludes
CUDA context, compiled code and driver-managed local-memory allocations, and is
not a bound on total GPU/process memory. Nonfinite or device failure stops the
trainer; a failed Adam update leaves the last valid host snapshot unchanged.

The report adds host-clock phase totals for CPU prediction/seeds, geometry,
upload/allocation, all four kernels plus one synchronization, download, and stable
reduction/primal checking. There are no profiling synchronizations between kernels.
Complete process measurements also include driver/NVRTC initialization, Adam,
assessment, checkpoints and export/reload verification.

## Recorded acceptance

[Compact evidence](fixtures/cuda-reverse-training-report.json) records Windows GNU
Rust, CUDA 12.8 and RTX 4090 results. Tests compare every parameter for energy-only,
force-only, virial-only and combined objectives against CPU AD and retained CUDA
hyperdual, including triclinic PBC, self images, padding, width 32, axis width 2,
zero/negative timestep, saturated tanh, buffer growth/shrink and repeatability.
Two Adam steps with clipping agree, and serialized 2+3 steps equal continuous 5.
The 24-frame 18/6-split synthetic replay trains all 112 neural parameters for 300
steps, verifies improvement and exact 150+150 restart, and reloads native-v1 output.

A separate 466-parameter, 64-atom, 8-frame 6/2-split E/F/V replay performs 100 updates
in each of three complete fresh processes per backend, rotating execution order.
The measured complete-process medians were CPU 2963 ms, hyperdual 1349 ms, reverse
877 ms (1.54x versus hyperdual for this bounded case). Each backend's three final
checkpoints were identical; final parameters differed from CPU by at most
4.45e-16. The small 112-parameter replay remained faster on CPU: 615 ms versus
1573 ms reverse CUDA. These synthetic implementation checks do not establish
physical accuracy, generalization, or performance on production DeepMD models.

```sh
cargo test --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda
# Only in an authorized exclusive GPU window:
cargo test --release --locked --offline --manifest-path mlmd/native/Cargo.toml \
  --features cuda --test cuda_reverse_training -- --ignored --nocapture --test-threads=1
python mlmd/native/tools/replay_full_training.py --eval-exe EVAL --train-exe TRAIN \
  --output-dir NEW_DIR --device cuda:0 --cuda-gradient reverse --compare-cpu --virial-weight 1
python mlmd/native/tools/replay_cuda_reverse.py --eval-exe EVAL --train-exe TRAIN \
  --output-dir ANOTHER_NEW_DIR
```

Raw generated data, model weights and logs stay under ignored
`fixtures/generated-deepmd/`. Only code, tiny existing synthetic fixtures,
licenses and compact replay summaries are tracked. No HAP data was accessed.

The next opt-in infrastructure step is the [fixed training-frame geometry cache](FIXED_GEOMETRY_TRAINING.md), shared by both CUDA gradient executors.

The optional [tiled reverse executor](TILED_TRAINING.md) packs multiple independent
frames per launch. It retains the original optimizer/loss/sampler semantics and
uses a separate v7 backend identity; omitted tiling preserves this framewise path.
