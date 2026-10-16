# Optional native CUDA inference

Status: the bounded Windows CUDA subset has passed actual same-model E/F/virial
checks on original tiny, public converted and 216-atom synthetic fixtures. The
216-atom case was modestly faster in one end-to-end measurement; tiny cases were
slower. See the measured results below. The CPU default remains unchanged.

A subsequent host neighbor-cache optimization and same-work before/after
measurements are documented in [CUDA_INFRA.md](CUDA_INFRA.md). The next cooperative
fitting/scratch and persistent-host-workspace tranche is in [CUDA_FIT.md](CUDA_FIT.md).

## Computation boundary

Enable `cuda` explicitly. `cuda::CudaPotential` implements the existing shared
`Potential` trait and returns energy, analytic force and row-strain virial in
float64. This is a CPU/CUDA inference pipeline, not CUDA training:

1. Rust validates native-v1 weights/frame/types and builds or conservatively
   reuses periodic-image candidates, then filters/sorts the exact active CPU
   periodic/triclinic neighbor list. It computes normalized environment rows `R`
   and their displacement Jacobians `J`, including normalized padding.
2. CUDA `embed` evaluates every slot's embedding value `G` and radial derivative
   `dG/dR0`. One thread owns one atom-slot.
3. CUDA `fit` contracts `B=G^T R/sum(sel)` and `D[m,k]=B[m,:].B[k,:]`, evaluates
   the fitting network and backpropagates `dE/dD` and `dE/dB`. One 128-thread
   block cooperates on each central atom using shared memory. Both descriptor factors contribute, including doubled diagonal
   derivatives.
4. CUDA `pullback` computes `dE/dR`, including the radial embedding chain term.
   Rust downloads only atomic energies and these environment gradients, then
   contracts `J` and accumulates `F_i+=g`, `F_j-=g`, `W_ab-=delta_a*g_b` in the
   existing deterministic neighbor order. Periodic self-image forces cancel;
   their virial need not vanish.

All neural/descriptor arithmetic listed in steps 2-4 executes in real CUDA
kernels. Geometry and final force/virial reduction remain CPU work. Weights stay
resident; grow-only device and host workspace buffers are reused. Atom types are
uploaded again only when their ordered vector changes. CUDA compilation occurs once
per potential. Parameters are uploaded arrays, not generated source literals.
The CUDA source specializes dimensions while retaining ordinary bounded loops.
No fast-math flag is used; FMA contraction is disabled for closer CPU comparison.

## Build and use

```sh
# Default build has no CUDA runtime, library or header requirement.
cargo test --locked --offline --manifest-path mlmd/native/Cargo.toml

# Optional feature still needs no CUDA libraries at Rust link/build time.
cargo build --release --locked --offline --manifest-path mlmd/native/Cargo.toml \
  --features cuda --bins
omnimd-native-cuda-eval model.json < frames.jsonl
```

The CUDA CLI explicitly selects device 0. Rust callers can select another device
ordinal with `CudaPotential::new(model, ordinal)`. CUDA failures return errors;
there is no silently mislabeled CPU fallback. Any failed evaluation poisons the
instance and subsequent calls fail without retrying. To use CPU after a failure,
create an ordinary `NativePotential` explicitly. No device reset or global
configuration change is performed. The calling thread's prior CUDA context is
restored after operations; this backend does not promise Send/Sync.

The runtime dynamically loads official NVIDIA Driver API and NVRTC libraries.
Windows uses `SystemRoot/System32/nvcuda.dll` and
`CUDA_PATH/bin/nvrtc64_120_0.dll` (CUDA 12), with absolute-path DLL search rules.
Linux uses `libcuda.so.1` and `CUDA_PATH/lib64/libnvrtc.so`, defaulting to
`/usr/local/cuda/lib64/libnvrtc.so`. An explicit absolute
`OMNIMD_NVRTC_LIBRARY` path can name a different installed official NVRTC library.
No dependency or driver is automatically installed. Rust builds can run on a
machine without CUDA; using the CUDA backend requires compatible installed
NVIDIA driver/NVRTC and a supported GPU. Windows x64 is the development target;
Linux dynamic-loading code has not yet been runtime-verified.

Compile kernels without making driver calls or using a GPU:

```sh
omnimd-native-cuda-check --compile-only model.json
```

This uses installed NVRTC and targets `compute_75`; normal execution queries the
selected device's actual compute capability. An unsupported runtime/compiler or
architecture gives a clear error, including NVRTC compilation diagnostics.

## Bounded subset

The model must pass existing native-v1 validation. CUDA accepts tanh and linear
activations, none/identity/concat2 residuals, explicit branch timesteps and all
native normalization rules. It rejects sigmoid/softplus and unsupported native
architectures. Current limits are 8 types, 256 total neighbor slots, embedding
width 64, descriptor width 128, network layer width 128, 8 layers per network,
two million neural scalar values and 2048 atoms per frame. The existing public
converted model fits these bounds (138 slots, embedding width 8, descriptor
width 32). Neighbor capacity overflow still fails; neighborhoods are not
truncated. There are no compression tables, GPU neighbor search, GPU Adam or
mixed-derivative training kernels.

Embedding and pullback use one thread per slot; fitting uses one cooperative
block per atom. Scalar sums retain explicit order and avoid floating-point
atomic reduction. Embedding local scratch can still spill, and consumer GPUs
may have low FP64 throughput. Small models
can be slower after CPU geometry, transfers and launch overhead; no blanket
speedup or production scaling claim is made.

## Controlled acceptance and timing

CPU-only test suite with `--features cuda` currently passes 15 tests, including
strict packing and all existing CPU inference/training checks. NVRTC compilation
has succeeded for tiny original, public converted and medium synthetic models.
Actual GPU execution was subsequently checked in the controlled run below.

`tools/replay_cuda.py` prepares original tiny and 216-atom medium fixtures and
optionally adds the existing public converted model. Preparation alone never
runs a GPU:

```sh
python mlmd/native/tools/replay_cuda.py \
  --check-exe mlmd/native/target/release/omnimd-native-cuda-check \
  --output-dir mlmd/native/fixtures/generated-deepmd/cuda-prepared
```

Append `.exe` on Windows. For actual execution, use a new output directory and
add `--run-gpu` **only during the coordinator-approved exclusive GPU window**.
Optionally pass both `--public-model` and `--public-frames`. The script stops on
the first failed/timed-out process, performs no retry/reset and preserves the
failure. Previously observed GPU FECS errors must not trigger blind reruns.

The check executable evaluates identical frames/weights on native CPU and CUDA,
with component tolerance `1e-9 absolute + 1e-9 relative` for E/F/virial. Tiny cases
include periodic/nonperiodic geometries, wrapping and a periodic self-image.
After parity warm-up, it measures a small fixed number of complete evaluations:
startup compilation/model upload are reported separately; CPU geometry, host-
device transfers, kernels, synchronization and final reduction are included.
Reports include device identity, input hashes, maximum errors, elapsed totals
and CPU/CUDA time ratio. This is one bounded measurement, not a benchmark suite
or evidence of physical model accuracy. Generated data/models/PTX/raw logs stay
in ignored output directories; only compact acceptance evidence is tracked.

## Recorded GPU result (2026-10-14 UTC)

`fixtures/cuda-validation-report.json` records actual execution of implementation
commit `1f572deab644a6da2caa0f9ad703b3dd818080ee`. Hardware was an NVIDIA
GeForce RTX 4090 (compute 8.9, 24564 MiB), Windows WDDM driver 595.97, existing
CUDA 12.8 NVRTC, and Rust 1.99.0 GNU release build. The parent granted a project
exclusive window after the other worker stopped. Preflight found no other
project compute process; ordinary desktop/game graphics remained active
(observed GPU use 14-46%, about 5.3 GiB). No other application was stopped.
This background load limits the interpretation of the single timing run.

| Case | Atoms / neural scalars | Max absolute E / F / virial error | CPU ms/eval | CUDA ms/eval | CPU/CUDA ratio |
| --- | --- | --- | ---: | ---: | ---: |
| Original tiny | 4,4,4,1 / 112 | 0 / 4.44e-16 / 4.44e-16 | 0.02897 | 0.24293 | 0.119 |
| Public converted | 6 per frame / 826 | 0 / 4.44e-16 / 8.88e-16 | 0.37143 | 3.14343 | 0.118 |
| Original medium | 216 / 28258 | 3.55e-15 / 1.82e-18 / 6.71e-19 | 60.38526 | 52.28750 | 1.155 |

The tiny and public cases each used four parity frames and three timed repeats
(12 evaluations); the medium used one frame and five timed repeats. All
component checks passed `1e-9 absolute + 1e-9 relative`. Initialization was
846/961/1164 ms respectively and is excluded from the warmed evaluation table.
All per-evaluation CPU geometry, transfers, CUDA kernels/synchronization and
force/virial reduction are included. Model shape/count/hash details are in the
report. The medium's modest 1.15x measured ratio is not a general acceleration
claim; the two small cases were about 8.4x slower on this measured path.

The planned run completed once without CUDA errors or retries. All verification
processes exited and a subsequent process check was empty. The GPU window was
explicitly released at `2026-10-14T04:50:18.0658055Z`; no further GPU work was
started. This validates the documented inference subset, not GPU training,
physical model accuracy or production performance.
