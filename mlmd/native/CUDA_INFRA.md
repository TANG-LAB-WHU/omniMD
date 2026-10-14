# CUDA neighbor-cache optimization

This change targets the CPU preparation bottleneck in the existing optional
float64 CUDA inference path. It preserves the CPU default, explicit CUDA device
selection, model format, shared `Potential` interface, kernels and EFV tolerance.
It adds no dependencies. GPU training is outside this implementation.

## Measured reason for the change

A single segmented profile of the existing 216-atom case on baseline
`a74804d8cbba86c1a2563dbbca6444059e740903` spent 221.84 of 232.68 ms
(95.3%) preparing five evaluations on the host. Kernel execution plus
synchronization took 9.33 ms. Repeating the exhaustive periodic-image search
was therefore the first target. The baseline executable adds only host timers;
its computation is unchanged. Engine inspection did not establish an additional
duplicate-EFV bottleneck, so no engine or shared-contract changes were made.

`CudaPotential::last_profile()` exposes host wall-clock segments for preparation,
upload/allocation, kernel execution plus synchronization, download and final
reduction, together with cache rebuild/candidate counts. These are not CUDA
event times. Complete evaluation times include per-call work outside the named
segments too. `omnimd-native-cuda-check` reports cold first evaluations and
initialization separately from warmed timings.

## Conservative image cache

The CUDA backend builds all candidate images inside `rcut + 0.4 Angstrom`, with
a conservative floating-point margin. Each candidate retains the atom index and
integer lattice shift. The list is not truncated to model `sel` capacities.

Reuse requires identical cell, species and atom count, and every atom's raw
Cartesian displacement from the build frame strictly below 0.2 Angstrom. Thus
any pair's relative motion is below the skin. Wrapping a coordinate can trigger
a conservative rebuild. Any cell change, including strain, also rebuilds. The
model and cutoff are immutable for the lifetime of the cache.

Every evaluation recomputes image displacements, exact active-cutoff filtering,
neighbor ordering, normalized environment rows, displacement Jacobians,
networks, forces and virial. Only image membership candidates are cached. The
same distance/atom/displacement ordering as the original search is retained;
periodic self-images remain present. Actual active neighbor capacity overflow
and near-coincident atoms still return errors.

The margin accounts for coordinate subtraction and skew-cell image
cancellation. If its conservative arithmetic bound is nonfinite or exceeds
0.001 of the skin, caching is disabled. The cache is also bounded to 1,048,576
stored candidates. Any expanded-search resource error discards the candidate
list and uses the original actual-cutoff neighbor search. This fallback is
limited to host neighbor search; neural computation still uses the explicitly
selected CUDA backend, and GPU failures still poison the potential.

## Recorded result

See [the compact evidence](fixtures/cuda-infra-report.json). The environment was
an RTX 4090 (compute 8.9, 24564 MiB), Windows WDDM driver 595.97, installed CUDA
12.8 NVRTC and Rust 1.99 GNU release mode. The project GPU window was exclusive
to B. Desktop/game graphics remained active (initial observed GPU use 47%,
2942 MiB); no other application was stopped. Timings are one bounded run under
that background load, not a production benchmark.

Both timing cases use one identical frame and five warmed complete evaluations:

| Case | Atoms / neural scalars | Baseline CUDA ms/eval | Cached CUDA ms/eval | Before/after ratio | Cached preparation ms/eval |
| --- | --- | ---: | ---: | ---: | ---: |
| Original synthetic medium | 216 / 28258 | 46.5362 | 2.4812 | 18.76 | 0.2660 |
| Public converted water model | 192 / 826 | 46.3018 | 4.2460 | 10.90 | 1.3326 |

The medium uses 32 neighbor slots, embedding width 32 and descriptor width 128;
the public model uses 138 slots, width 8 and descriptor width 32. The public
weights come from the existing documented DeepMD v3.1.3 test asset. Its new
geometry is an original 64-water molecular lattice in a 12.4 Angstrom cube,
not equilibrated liquid or an accuracy dataset. All weights, frame hashes and
shape details are recorded in the evidence. No public weights are committed.

The same-run native CPU times were 59.24 and 54.67 ms/eval. The observed advantage
is for this warmed CPU/CUDA path; the default CPU neighbor search is unchanged.
The GPU kernels were not accelerated by this change. Initialization still cost
910/727 ms, and the first cached evaluations cost 102.35/78.93 ms respectively,
including their first builds and runtime warm-up. Frequent rebuilds, changing
cells, independent frames and tiny systems can reduce or remove the benefit.

Maximum absolute E/F/virial errors were respectively
`3.55e-15 / 1.82e-18 / 6.71e-19` for the medium and
`0 / 7.77e-16 / 5.33e-15` for public water. The acceptance tolerance remains
`1e-9 absolute + 1e-9 relative` per component.

The unmodified unified engine also completed matching 20-step NVE runs for
public water: 0.1 fs, deterministic 300 K velocities, seed 20261014, COM removed.
Across 21 snapshots the maximum position/velocity differences were
`5.55e-17 Angstrom / 1.39e-17 Angstrom/fs`; E/F/virial differences were
`0 / 1.44e-15 / 1.78e-14`. Both runs had maximum energy drift
`5.47e-5 eV`. This short numerical check does not establish physical accuracy or
long-time stability. Whole process times, including initialization and JSONL
output, were 1.303 s CPU and 1.357 s CUDA; this short run was slightly slower
on CUDA despite the faster warmed evaluations.

The timing pair and full MD replay preceded the final conservative
coordinate-scale guard. After that guard, a final actual CUDA check replayed
MD snapshots 0, 10 and 20 against fresh CPU evaluation: one candidate build,
all parity checks passed, maximum E/F/virial errors
`0 / 1.44e-15 / 7.11e-15`. No kernels or physical arithmetic changed.

Offline default-CPU tests passed 13/13, CUDA-feature CPU tests passed 15/15.
The focused cache test checks cutoff entry, two atoms approaching by nearly
half the skin, exact rebuild threshold, triclinic images, wrapping, cell/type
changes, periodic self-images, capacity errors and discarded-cache fallback
for large coordinates. Existing EFV finite differences, symmetries and training
checks remain passing. There were no CUDA errors or retries. Verification
processes exited and the GPU window was released at
`2026-10-14T07:40:02.7085468Z`.

## Reproduce

Build the baseline check executable from `a74804d` in a separate existing
checkout/build directory and retain it before building the optimized executable.
The original baseline reports complete timings without the optional phase
instrumentation used for the recorded profile. No baseline patch is needed
for the before/after comparison. Build current native and unified engine:

```sh
cargo build --release --locked --offline --manifest-path mlmd/native/Cargo.toml \
  --features cuda --bins
cargo build --release --locked --offline --manifest-path mlmd/Cargo.toml \
  -p omnimd-mlmd --features native-cuda --bin omnimd-mlmd
cargo test --locked --offline --manifest-path mlmd/native/Cargo.toml --no-default-features
cargo test --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda
```

Use `tools/replay_cuda.py` without `--run-gpu` to prepare the original medium
model/frame. Use the already converted public model documented in `FROZEN.md`.
The following command prepares the 192-atom frame and matching MD requests in a
new directory; it neither starts a GPU nor downloads anything:

```sh
python mlmd/native/tools/replay_cuda_infra.py \
  --baseline-check-exe /absolute/baseline/omnimd-native-cuda-check \
  --check-exe /absolute/current/omnimd-native-cuda-check \
  --engine-exe /absolute/current/omnimd-mlmd \
  --public-model /absolute/public/native.json \
  --medium-model /absolute/prepared/medium-model.json \
  --medium-frames /absolute/prepared/medium-frames.jsonl \
  --output-dir mlmd/native/fixtures/generated-deepmd/cuda-infra-new
```

Append `.exe` on Windows. For an actual replay choose another new output
directory and add `--run-gpu` only after the coordinator grants an exclusive GPU
window. The script performs two before/after cases and the short CPU/CUDA MD
comparison, stops on the first failure/timeout and does not retry or reset.
Output includes input/executable hashes and comparisons. Generated weights,
requests, trajectories, checkpoints and raw logs remain ignored. The generator
and MD comparison helper were checked against the recorded fixtures; the entire
new orchestration script was not rerun after GPU release.

The cache, profiling and generator code are independently written under this
project's MIT license. Image-list reuse is an established Verlet-list technique,
not an algorithmic novelty. See `THIRD_PARTY_NOTICES.md` for existing public
model and CUDA runtime provenance.
