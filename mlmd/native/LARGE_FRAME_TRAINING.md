# Resource-bounded CUDA training with up to 256 atoms per frame

Tiled reverse CUDA training now accepts an explicit larger-frame ceiling of up
to 256 atoms. This applies to both CPU prediction/loss and optional GPU
prediction/loss. The default remains 64 atoms per frame. The supported FP64
se_e2_a/tanh neural family, loss formulas, parameter set and optimizer are unchanged.
CPU full-gradient and untiled CUDA executors retain their 64-atom bound.

This is a capacity extension with explicit resource admission. A 256-atom frame
was verified with the 112-parameter synthetic model; a 466-parameter model is
rejected for that frame because the unchanged work budget permits only 138 tile
atoms in GPU-loss mode. The configured ceiling is not a guarantee that every
supported model can fit every frame.

## Enable deliberately

1. Build the existing CUDA feature with the installed official Driver/NVRTC
   runtime. Existing dependencies suffice; the implementation installs nothing:

   ```sh
   cargo build --release --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda --bins
   ```

2. Start from a complete [full-training request](FULL_TRAINING.md), use an explicit
   CUDA device such as `"device":"cuda:0"`, then add the following config:

   ```json
   {
     "cuda_gradient": "reverse",
     "cuda_tiling": {
       "max_frames": 4,
       "max_atoms": 512,
       "max_atoms_per_frame": 256
     }
   }
   ```

   `max_atoms` limits the **total** atoms in a tile; `max_atoms_per_frame` limits
   each frame. Its accepted range is 64..256. Omitted or explicit 64 retains
   historical behavior and serialization. The example retains CPU prediction/
   seeds; adding `"cuda_loss":"gpu"` selects the large-frame GPU gather.
   Both choices perform neural-parameter gradients and Adam on CUDA.

3. Run `omnimd-native-train request.json`, or use the engine's existing
   `omnimd-mlmd train-native request.json` forwarding command. No shared interface,
   Frame, units or dataset format changes are required. Both memory and indexed
   dataset access work; indexed access still requires explicit minibatches.

4. Check `report.json`: requested and effective tile capacities, payload/work
   estimates, `cuda_gradient_profile.max_atoms_per_frame`, and measured process
   cost. Frames are never split into independent subframes or truncated to fit.
   A frame that cannot fit alone is rejected. Choose a new optimizer run when
   changing the configured frame ceiling.

## Actual fixed boundary and implementation

Reverse network caches and derivative buffers were already sized by total tile
atoms. Per-thread local arrays are sized by neural width/layers, not frame atoms.
The limiting kernel was GPU prediction gather: a fixed 256-thread block assigned
one output to each thread, but a frame needs `1 + 3*N + 9` outputs. Larger frames
would leave late forces and virials unwritten if only the host guard were removed.

The new `loss_predictions_large` kernel retains 256 threads per frame and visits
outputs with `t = threadIdx.x; t < 1+3*N+9; t += blockDim.x`. A 256-atom frame
therefore covers all 778 outputs. Each force/virial component uses the same edge
summation order, restricted to its own frame. There are no atom-sized local
arrays, floating-point atomics or barriers inside the unequal-iteration loop.
Neighbor IDs and boundaries use signed int32, including the -1 padding marker.
The old kernel source is preserved and is selected for the default 64-atom mode.

All physical atoms in a frame share its cell and neighbor list; periodic image
edges retain full displacements. Frame-level prediction, detached loss seeds,
normalization and all nine row-strain virials keep their prior definitions.
Every tile uses the same weights until all tiles in its optimizer batch finish;
reduction/clipping and one Adam update retain their original boundaries.

## Admission and resource accounting

Architecture limits remain 2048 neural parameters, widths at most 32, 64 selected
neighbor slots, eight types and eight layers per network. Tile policies remain
32 frames / 1024 total atoms, 50 million estimated scalar terms, 64 MiB explicit
device-buffer payload and 32 MiB host tile payload. No budget is increased.

The planner bounds total atoms by `max_frames * max_atoms_per_frame` as well as
requested atoms and payload/work limits. GPU-loss work accounting includes
`max_atoms_per_frame * slots * 3` per atom for the bounded quadratic force gather,
plus the existing forward/reverse and virial-direction terms. It uses the
configured ceiling conservatively, even when actual frames are smaller. This is
an acceptance heuristic, not an exact FLOP count or runtime guarantee.

The high-level trainer checks every dataset frame and effective capacity before
geometry-cache construction or CUDA context creation. The low-level batch API
checks all selected frame counts before allocating tile arrays or launching the
first tile. Its model/context may already exist because construction is separate.
An oversized later frame therefore cannot cause earlier tiles to execute. Any
subsequent geometry/read/kernel/nonfinite failure still poisons the trainer and
prevents a partial optimizer update.

Indexed positions/species/forces now have a strict technical parsing ceiling of
256 entries; all shapes, labels and the configured training limit are checked.
The 256 KiB indexed record limit, 16 GiB file limit and 64 MiB index payload limit
are unchanged. The fixed geometry cache supports these frames under its separate
unchanged 64 MiB payload limit. Neither parser nor cache silently enables larger
frames for the default trainer.

Payload limits are not total GPU-memory or process-RSS limits. CUDA context/module/
kernel local memory, allocator overhead, CPU geometry construction, one decoded
frame, index, model/optimizer and report/checkpoint storage are accounted for
separately or excluded as documented in [TILED_TRAINING.md](TILED_TRAINING.md).
The report records observed buffers, geometry cache, index and process working set
separately; it does not claim a total hard memory cap.

## Checkpoints and compatibility

A configured frame ceiling above 64 selects
`omnimd-native-full-checkpoint-v9`, taking precedence over v7/v8 tiling and v6
schedules. The ceiling, total tile capacities and loss mode are immutable restart
fields. GPU-loss large-frame execution uses `cuda_tiled_device_loss_f64_v2` with
its own compiled PTX identity. CPU-loss tiling retains
`cuda_tiled_reverse_f64_v1` because its arithmetic/kernels are unchanged; v9 and
the immutable config distinguish the expanded admission rules.

Default 64-atom v7/v8 checkpoints keep their prior schema/config/kernel identity.
An old checkpoint cannot be silently continued with a larger ceiling; initialize
a new optimizer from the exported native model. Same-mode restart may change
storage and geometry-cache settings as before. Exact equality was checked in the
same Windows build/runtime; cross-build/platform/driver equality is not promised.
The prior Linux v7/v8 validation is preserved but does not validate this new v9
large-frame kernel.

## Focused verification and observed cost

Commands from the repository root with existing offline dependencies:

```sh
cargo test --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda
# During an authorized GPU window only:
cargo test --release --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda --test large_frame_training large_frame_gpu -- --ignored --nocapture --test-threads=1
python mlmd/native/tools/replay_large_frame_training.py --eval-exe EVAL_EXE --train-exe TRAIN_EXE --reference-exe PRE_LARGE_FRAME_TRAINER --output-dir NEW_IGNORED_DIRECTORY
```

Windows / RTX 4090 / CUDA 12.8 NVRTC / driver API 13040 results:

- 58 CUDA-feature tests requiring no GPU execution passed. One focused actual-GPU
  test covered 256, 128, 83 and 32 atoms, including large-to-small reuse and a
  mixed tile containing 467 atoms. The 83-atom case crosses the first strided
  output tail, with all forces and virials checked.
- The 128/256-atom fixtures are periodic supercells of a perturbed 32-atom O/H
  unit cell, with nonzero forces and interactions across cell boundaries. They
  are real single Frame values, not collections of smaller independent frames.
  Maximum absolute E/F/V difference from CPU inference was `7.11e-15`.
- All 112 parameter gradients matched the CPU-loss CUDA route within `8.88e-16`.
  The per-atom normalized loss/gradient of periodic replicas matched the 32-atom
  CPU AD reference within `5.33e-15` for gradients. Independent central differences
  of CPU-evaluated 256-atom loss for eight parameter coordinates (`h=1e-5`)
  differed by at most `1.96e-10`; these differences are validation, not training.
- An oversized later frame was rejected before any tile, with unchanged weights/
  moments and further Adam use blocked. Nine CLI failures cover default ceiling,
  257-atom configuration/parser, insufficient tile/work budget, CPU device,
  changed restart capacity/schema/PTX; no output directories were created.
- The actual CLI trained on six heterogeneous 32/128/256-atom frames, with two
  validation frames, 12 updates, shuffled batches of four (partial two), explicit
  exponential schedule and 36 frame visits. CPU/GPU-loss final parameters and
  moments differed by at most `4.75e-14`. A serialized 6+6-step v9 restart changed
  indexed-to-memory access and cache-on-to-off, then matched the continuous run
  and final metrics exactly. Native model reload E/F/V differences were zero.
- A two-update default-mode comparison against the preserved pre-change trainer
  matched the full v8 checkpoint and PTX exactly. This is focused compatibility
  evidence, not a rerun of the previous complete CUDA matrix.

One observed complete process per mode, including startup, data validation,
assessment and output IO, produced the following costs. These are not benchmark
medians or a speedup claim:

| 12-update synthetic run | CPU-loss CUDA | GPU-loss CUDA |
| --- | ---: | ---: |
| Whole process | 1554.24 ms | 1721.33 ms |
| Gradient phase | 33.22 ms | 51.05 ms |
| Peak explicit device buffers | 4,905,404 bytes | 6,190,356 bytes |
| Peak host tile payload | 995,232 bytes | 1,755,624 bytes |
| Windows process PeakWorkingSetSize | 138,911,744 bytes | 139,915,264 bytes |

Both modes reached 256 atoms per frame and 512 atoms per tile, with 18 tiles.
The fixed training-geometry cache retained 3,115,722 bytes; the index retained
19,626 bytes. GPU-loss planned capacity was 6,190,636 device bytes / 2,313,280 host
tile bytes, including conservative diagnostic reserves. The cache-off six-step
resume took 2003.58 ms, illustrating remaining CPU geometry cost.

The deterministic GPU gather still scans a frame's edges for every force component
and has quadratic frame cost. CPU neighbors/geometry, stable reduction/clipping,
assessment and serialization remain. Larger frames do not imply faster training,
physical accuracy, general DeepMD compatibility or production scaling. New v9
Linux execution, larger models/frames and representative physical datasets remain
unverified. No architecture, dependency, shared contract or global configuration
changed; generated weights/datasets/raw logs remain ignored.

[Compact evidence](fixtures/large-frame-training-report.json) contains actual
measurements, execution identities, hashes and commands. All new code and fixture
recipes are original MIT contributions; no external training code was copied.
