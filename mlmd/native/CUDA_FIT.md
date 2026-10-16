# Cooperative CUDA fitting and persistent host workspace

This native-only tranche follows the cache implementation in `b6f36dd`, integrated
in candidate `7be1d976654f1a3c58ded4f69b2c9759cb9b3cbf`. Native source at those two
commits is identical. It improves the existing se_e2_a float64 inference path;
it adds no model family, GPU training, dependency or shared-interface change.

## Why this tranche

After candidate caching, the fresh 216-atom baseline spent 9.896 of 12.178 ms for
five evaluations in kernels plus synchronization. The old fitting kernel launched
one thread per atom: only four 64-thread blocks for 216 atoms. Driver-reported
resources were 255 registers and 7312 bytes of local memory per fitting thread.
Embedding scratch also used the maximum width across both embedding and fitting
networks, making its temporary arrays larger than its own network required.

The change is bounded to the measured kernel bottleneck and adjacent per-call
workspace overhead. The existing CPU default and error/poison behavior remain.

## Implementation and semantics

The fitting kernel now assigns one 128-thread block per central atom. Independent
components of `B=G^T R/sum(sel)`, descriptor products and neurons are computed in
parallel. `B`, forward values, activation slopes and backward adjoints use shared
memory. All threads participate in each barrier; only the uniform atom-bound
check returns before the first barrier. Components larger than the thread count
use strided ownership (256 B components at the supported maximum).

Each individual scalar sum retains the old order: slot contributions to B,
four descriptor dot products, bias-first neuron dot products, backward weighted
terms followed by residual additions, and ordered `(m,k)` contributions to dB.
Both diagonal descriptor contributions remain separate. No floating-point atomic
sum, mixed precision, fast-math or FMA contraction was introduced. The existing
`--fmad=false` setting and `1e-9 absolute + 1e-9 relative` tolerance remain.

Embedding scratch width is now derived only from embedding layers; fitting
width/depth are derived only from fitting layers. All existing dimension limits
still apply. Worst-case fitting shared storage is 14336 bytes at width 128,
depth 8 and embedding output width 64. This boundary was executed, not only
compiled. Wide embedding networks can still use substantial local memory.

Host environment rows, neighbor/Jacobian records, downloaded energies and
gradients retain capacity across calls. Device buffers were already persistent.
Kernel argument tables now use fixed stack arrays. The ordered atom-type vector
is uploaded only when it changes, including a same-count species change.
Environment rows and gradient outputs still transfer on every evaluation;
three launches and one context synchronization remain. This is not asynchronous
batching, a GPU-resident MD loop or CUDA graph execution.

`last_profile()` adds cumulative `type_uploads`, per-call `upload_bytes`, and
host `row_capacity`/`record_capacity` in elements. `kernel_resources()` exposes
read-only driver-reported registers, shared bytes per block and local bytes per
thread. There is no change to the shared `Potential` trait or model JSON format.

The mathematical semantics were checked against official DeepMD 3.1.3
[se_e2_a](https://github.com/deepmodeling/deepmd-kit/blob/v3.1.3/deepmd/dpmodel/descriptor/se_e2_a.py)
and [network](https://github.com/deepmodeling/deepmd-kit/blob/v3.1.3/deepmd/dpmodel/utils/network.py)
sources. CUDA synchronization and resource reporting follow the official
[programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) and
[driver function attributes](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html).
This is independently written MIT implementation code; no upstream code or
NVIDIA headers were copied. See THIRD_PARTY_NOTICES.md.

## Measured result (2026-10-16 UTC)

[Compact evidence](fixtures/cuda-fit-report.json) records model/frame/executable
hashes and resource measurements. Hardware was an RTX 4090, compute 8.9,
24564 MiB, Windows WDDM driver 617.42, existing CUDA 12.8 NVRTC and Rust 1.99 GNU
release mode. The project window was exclusive to B. Ordinary desktop graphics
remained active (initial GPU use 33%, 3809 MiB); nothing was stopped or installed.
The driver differs from the earlier October 14 measurement, so both baselines
were rerun in this window. Five identical warmed evaluations were measured once
per before/after case. Initialization is separate; this is not a cold driver-cache
benchmark or a statistical/general scaling claim.

| Case | Atoms / neural scalars | Baseline CUDA ms/eval | New CUDA ms/eval | Before/after ratio | Baseline/new kernel+sync ms/eval |
| --- | --- | ---: | ---: | ---: | --- |
| Original synthetic medium | 216 / 28258 | 2.4356 | 0.5421 | 4.49 | 1.9793 / 0.2364 |
| Public converted water model | 192 / 826 | 4.7410 | 1.1999 | 3.95 | 2.6313 / 0.2781 |

All complete-call timings include CPU geometry/normalization, host/device copies,
kernels, synchronization, output checks and force/virial assembly. The separate
optimizations were not benchmarked individually, so the combined improvement is
not attributed entirely to one mechanism. Skipping types saves only 864/768 bytes
per warmed medium/public call; it does not remove the main environment transfers.

Medium fitting resources changed from 255 registers / 7312 local bytes per
thread to 42 registers / zero local bytes, with 8192 shared bytes per block.
Public fitting changed from 164 registers / 1536 local bytes to 42 / zero, with
2304 shared bytes. Embedding local storage fell from 4096 to 1024 bytes per thread
for medium and from 1024 to 256 for public water.

Maximum absolute E/F/virial errors against native CPU were unchanged from the
baseline: medium `3.55e-15 / 1.82e-18 / 6.71e-19`, public
`0 / 7.77e-16 / 5.33e-15`. The public model is the previously converted DeepMD
public inference-test model, evaluated on the original 192-atom molecular
lattice from CUDA_INFRA.md. It is not an equilibrated-liquid accuracy test.

The four tiny periodic/nonperiodic/wrapped/self-image fixtures passed. A separate
six-frame sequence varied atom counts `1,4,4,4,8,4`, changed types without changing
count and repeated the same types. First-pass cumulative type uploads were
`1,2,3,3,4,5`; capacities grew and remained reusable through shrink/regrowth.
E/F/virial parity passed throughout. An original synthetic eight-layer boundary
model exercised embedding output width 64, hidden width 128, fitting width 128,
128 descriptor components and 256 B components. Its fitting shared memory was
14336 bytes and E/F/virial errors were `0 / 6.78e-21 / 5.08e-21`.

The tiny case remained slower than CPU (0.091 versus 0.034 ms/eval), as did the
four-atom boundary model (2.271 versus 1.770 ms). GPU is never selected by an
atom-count heuristic. These results do not establish production performance or
an advantage over the official DeepMD runtime.

The unchanged unified engine completed matched public-model NVE runs: 20 steps,
0.1 fs, 300 K initialization, seed 20261016, COM removed. Across 21 snapshots,
maximum position/velocity differences were `8.88e-16 Angstrom / 1.39e-17 Angstrom/fs`,
and E/F/virial differences were `0 / 1.55e-15 / 1.42e-14`. Both runs had maximum
energy drift `4.37e-5 eV`. Whole process time, including instance initialization
and JSONL output, was CPU 1.400 s versus CUDA 0.388 s. This is a short numerical
check, not evidence of long-time stability or physical accuracy.

Default CPU tests passed 13/13 and CUDA-feature CPU tests passed 15/15. GPU checks
reported no errors or retries. All verification processes exited; the exclusive
window was released at `2026-10-16T07:37:21.4622354Z`. Subsequent checks were CPU-only.

## Exact replay steps

Build and retain a baseline `omnimd-native-cuda-check` from `7be1d97` (or native
`b6f36dd`) before building current code. The recorded baseline additionally used
read-only `cuFuncGetAttribute` diagnostics; they do not change computation.
Original baseline binaries suffice for full-call before/after timing.

```sh
cargo build --release --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda --bins
cargo build --release --locked --offline --manifest-path mlmd/Cargo.toml -p omnimd-mlmd --features native-cuda --bin omnimd-mlmd
cargo test --locked --offline --manifest-path mlmd/native/Cargo.toml --no-default-features
cargo test --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda
```

The following commands assume those executables are on PATH; append `.exe` on
Windows. Prepare new ignored directories and use an existing strictly converted
public model from FROZEN.md. Preparation alone executes no GPU:

```sh
python mlmd/native/tools/replay_cuda.py --check-exe /absolute/current/omnimd-native-cuda-check --output-dir mlmd/native/fixtures/generated-deepmd/fit-medium-new
python mlmd/native/tools/prepare_cuda_fit_checks.py --public-model /absolute/public/native.json --output-dir mlmd/native/fixtures/generated-deepmd/fit-edges-new
```

Only during an authorized exclusive GPU window, run each executable once for
each identical workload (five warmed repeats, unchanged parity tolerance):

```sh
/absolute/baseline/omnimd-native-cuda-check mlmd/native/fixtures/generated-deepmd/fit-medium-new/medium-model.json mlmd/native/fixtures/generated-deepmd/fit-medium-new/medium-frames.jsonl 5
omnimd-native-cuda-check mlmd/native/fixtures/generated-deepmd/fit-medium-new/medium-model.json mlmd/native/fixtures/generated-deepmd/fit-medium-new/medium-frames.jsonl 5
/absolute/baseline/omnimd-native-cuda-check /absolute/public/native.json mlmd/native/fixtures/generated-deepmd/fit-edges-new/public-water192.jsonl 5
omnimd-native-cuda-check /absolute/public/native.json mlmd/native/fixtures/generated-deepmd/fit-edges-new/public-water192.jsonl 5
omnimd-native-cuda-check mlmd/native/fixtures/full-tanh-v1.json mlmd/native/fixtures/generated-deepmd/fit-medium-new/tiny-frames.jsonl 1
omnimd-native-cuda-check mlmd/native/fixtures/full-tanh-v1.json mlmd/native/fixtures/generated-deepmd/fit-edges-new/lifecycle.jsonl 1
omnimd-native-cuda-check mlmd/native/fixtures/generated-deepmd/fit-edges-new/boundary-model.json mlmd/native/fixtures/generated-deepmd/fit-edges-new/boundary-frame.jsonl 1
omnimd-mlmd run mlmd/native/fixtures/generated-deepmd/fit-edges-new/md-cpu.request.json mlmd/native/fixtures/generated-deepmd/fit-edges-new/md-cpu.jsonl
omnimd-mlmd run mlmd/native/fixtures/generated-deepmd/fit-edges-new/md-cuda.request.json mlmd/native/fixtures/generated-deepmd/fit-edges-new/md-cuda.jsonl
```

Stop on the first failed or timed-out command; do not reset/retry GPU errors.
The recorded executions used a 60-second subprocess timeout. For each matching
MD step, compare positions, velocities and all E/F/virial components using the
same `1e-9 + 1e-9*abs(reference)` tolerance. The existing
`tools/replay_cuda_infra.py::compare_md` helper performs that comparison and
checks matching model, timestep, initialization and masses in trajectory headers.
It reads the reported seed from the header (20261016 in this replay).

Generated models, trajectories, requests, checkpoints and raw logs remain
ignored. Only the original preparation recipe and compact report are tracked.
Remaining work includes GPU training, GPU environment/neighbor computation,
reduced full-environment transfers and asynchronous multi-frame execution.
No implementation or parity claim is made for those features.
