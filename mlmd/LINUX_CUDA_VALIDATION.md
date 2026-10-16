# Linux native CUDA validation

The exact integrated candidate `06439f1cf91306b4d19c570e20abaeb7e140adf6`
passed bounded Linux x86_64 native CUDA inference, tiled E/F/V training and
checkpoint-resume validation on 2026-10-16. The build used an isolated checkout,
mounted read-only; the integration worktree was untouched. No native portability
change was needed. This evidence commit changes documentation and replay assets
only. The [receipt](evidence/native-linux-cuda.json) includes executable, input,
request and CUDA program hashes.

## Runtime and build

| Component | Observed value |
| --- | --- |
| Official cached image | `deepmodeling/deepmd-kit:3.1.0_cuda129` |
| Image ID | `sha256:23d23b922459eb49b23f9d4977b6e453fd17feb2ee03a280c70fb7f4a6452e8c` |
| Container | `omnimd-native-linux-06439f1-v2`, network disabled |
| Platform | Ubuntu 22.04.5 LTS, x86_64, WSL2 kernel `5.15.167.4-microsoft-standard-WSL2` |
| GPU / driver | RTX 4090, compute 8.9, 24,564 MiB / NVIDIA 617.42 |
| Driver API / NVRTC | `13040` / 12.9.86 (`nvrtcVersion`: 12.9) |
| Toolchain | Rust/Cargo 1.99.0, conda-forge GCC 13.3.0-2, Python 3.12.11 |
| Build | Release, offline, locked, native `cuda` and engine `native-cuda` features |

The image supplied NVRTC and matching builtins. The injected driver library was
`/usr/lib/x86_64-linux-gnu/libcuda.so.1`, not a toolkit stub. The explicit
`OMNIMD_NVRTC_LIBRARY=/opt/deepmd-kit/lib/libnvrtc.so.12` resolved to
`/opt/deepmd-kit/targets/x86_64-linux/lib/libnvrtc.so.12.9.86`.
The existing Rust toolchain and dependency registry were read-only mounts;
Cargo output and locks were task-local. Nothing was downloaded or installed.
No driver, BIOS, security setting or HAP resource was changed.

The first offline build failed before execution because a Cargo registry symlink
target was not mounted. Correcting that mount resolved it; release compilation
finished in 40.80 seconds. This was a setup issue, not a source defect. Both
project validation containers were stopped afterward and the GPU window released.

## Results and timing scope

Compile-only NVRTC emitted 47,649 PTX bytes for `compute_75` without accessing the
GPU. The original four-atom fixture then completed exactly one explicit CUDA
E/F/V evaluation: three kernels followed by context synchronization, with no CPU
fallback. Same-model native CPU/CUDA maximum absolute errors were E `0`, F
`4.440892098500626e-16`, and virial `4.440892098500626e-16`. The component tolerance
was `1e-9 + 1e-9 * abs(CPU)` in the contract's eV/angstrom units.

Training used a frozen original synthetic dataset: 24 frames of 3/4 atoms,
12 periodic frames, seeded 18/6 train/validation split, batch size 7, shuffled
epochs, all E/F/V losses enabled, exponential learning-rate decay, and tiles of
at most 3 frames / 16 requested atoms. The actual maximum was 11 atoms per tile;
two partial optimizer batches were exercised. All three requests went through
the integrated `omnimd-mlmd train-native` command.

| Case | Updates | Gradient frames / tiles | Kernels / syncs | Wall seconds | Kernels + sync ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current v7 reference | 7 | 43 / 19 | 76 / 19 | 0.805996 | 1.619669 |
| v8 GPU loss | 7 | 43 / 19 | 190 / 19 | 1.034141 | 2.499069 |
| v8 resume from step 2 | 5 | 29 / 13 | 130 / 13 | 0.315907 | 1.822802 |

The v7 reference uses CPU predictions/loss seeds and **CUDA reverse derivatives
and Adam**. The v8 path also performs E/F/V predictions and detached loss seeds
on CUDA; it recorded 43 device-predicted training frames and zero host prediction
time. Geometry, normalization scalars, stable reduction, clipping and assessment
remain on CPU. Zero host prediction time means the computation moved into the
CUDA kernel time. These are application launch counters, not hardware-profiler
counts; gradient-profile kernel counts exclude Adam. Phase timers use host wall
time, and `kernels_and_sync_ms` includes launches and synchronization.

Maximum v8/v7 parameter-or-moment difference was `5.828670879282072e-16` (absolute
limit `3e-12`). Maximum final training/validation metric differences were
`6.938893903907228e-17` / `2.7755575615628914e-17`. CPU evaluation of all 24 frames
from exported models gave E/F/V differences
`4.440892098500626e-16 / 4.440892098500626e-16 / 5.551115123125783e-16`.

Resume used the newly created **Linux** v8 step-2 checkpoint. Five further updates
exactly matched uninterrupted step 7: model, Adam moments, sampler, step, execution
identity, exported model bytes and final partition metrics. Only the deliberately
changed step-count and geometry-cache configuration fields were excluded from
whole-checkpoint equality. Resume changed dataset access from indexed to memory
and disabled the geometry cache. Its first resumed update was 3 at rate `0.0025`.
All three CPU export evaluations passed. There were no GPU errors or retries.

These are single process timings, including startup/JIT, I/O, assessment and
checkpoint writes. Load, warmup and steady state were not measured separately.
Cold single-frame CPU/CUDA processes took 0.022680 / 1.609159 seconds. The small
v8 training case was slower than v7. Desktop/background GPU activity was observed
before testing and left alone. No larger-system speedup is claimed.

## Bounded reproduction

The [replay directory](validation/linux-cuda-06439f1) contains the exact executed
helpers, requests and frozen synthetic dataset. Its `.gitattributes` preserves
their bytes. Only informational source paths in the packaged manifest were made
portable; the receipt retains the executed manifest hash.

The dataset follows `native/tools/replay_full_training.py`: perturb teacher
parameters by `0.12 * sin(j * 0.63)` in parameter order, use geometry seed
`20261014`, alternate nonperiodic/triclinic frames and remove the final atom every
third frame, then label E/F/V with the native CPU evaluator. The bundled bytes
retain historical generator/teacher hashes. This is software validation data,
not a physical-accuracy or cross-platform byte-regeneration claim.

Use an isolated exact-candidate source checkout, this replay package, a fresh
output directory and the existing official runtime/toolchain/cache. Set the five
host paths below to their actual locations. The commands describe the mounts and
executed sequence; no package installation is necessary. A failed GPU stage must
end the sequence: retain its markers/logs, stop the container and report the error.

```bash
set -eu
src=/absolute/path/to/isolated-06439f1-checkout
replay=/absolute/path/to/evidence-checkout/mlmd/validation/linux-cuda-06439f1
work=/absolute/path/to/new-empty-validation-output
cargo_cache=/absolute/path/to/existing/cargo-linux
toolchain=/absolute/path/to/existing/1.99.0-x86_64-unknown-linux-gnu
test "$(git -C "$src" rev-parse HEAD)" = 06439f1cf91306b4d19c570e20abaeb7e140adf6
git -C "$src" diff --exit-code HEAD --
mkdir "$work"
cp "$replay"/*.py "$replay"/*.request.json "$replay"/source-manifest.json \
   "$replay"/dataset.jsonl "$work/"

# The observed Windows-mounted initial fixture used CRLF. Preserve its hash
# even when the exact candidate is checked out with LF on a Linux host.
python3 - "$src/mlmd/native/fixtures/full-tanh-v1.json" "$work/initial.json" <<'PY'
import hashlib, pathlib, sys
data = pathlib.Path(sys.argv[1]).read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
if hashlib.sha256(data).hexdigest() != '407278c6eab122e6dfee2511b3c3fb4c033f58c04819a580ffc0810df3ce7b9a':
    raise SystemExit('Unexpected initial fixture')
with pathlib.Path(sys.argv[2]).open('xb') as stream:
    stream.write(data)
PY

name=omnimd-native-linux-replay
image=sha256:23d23b922459eb49b23f9d4977b6e453fd17feb2ee03a280c70fb7f4a6452e8c
docker run --pull never --network none --gpus device=0 --name "$name" -d \
  --mount "type=bind,src=$src,dst=/src,readonly" \
  --mount "type=bind,src=$work,dst=/work" \
  --mount "type=bind,src=$cargo_cache,dst=/shared/cargo-linux,readonly" \
  --mount "type=bind,src=$toolchain,dst=/toolchain,readonly" \
  --entrypoint /bin/sleep "$image" infinity
trap 'docker stop -t 1 "$name" >/dev/null' EXIT
docker exec "$name" mkdir -p /work/cargo
docker exec "$name" ln -s /shared/cargo-linux/registry /work/cargo/registry
docker exec -e CARGO_HOME=/work/cargo -e CARGO_TARGET_DIR=/work/target \
  -e CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/opt/deepmd-kit/bin/x86_64-conda-linux-gnu-cc \
  -e PATH=/toolchain/bin:/opt/deepmd-kit/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  "$name" cargo build --offline --locked --release --manifest-path /src/mlmd/Cargo.toml \
  -p omnimd-native -p omnimd-mlmd \
  --features omnimd-native/cuda,omnimd-mlmd/native-cuda --bins

run_python() {
  docker exec -e OMNIMD_NVRTC_LIBRARY=/opt/deepmd-kit/lib/libnvrtc.so.12 \
    -e LD_LIBRARY_PATH=/opt/deepmd-kit/lib -e CUDA_VISIBLE_DEVICES=0 \
    -e OMP_NUM_THREADS=1 "$name" /opt/deepmd-kit/bin/python "$@"
}
# Run sequentially; never proceed past a failed stage.
run_python /work/probe.py compile
run_python /work/probe.py probe
run_python /work/run_linux_validation.py --execute --timeout 120
```

The inference helper records raw source-fixture hashes, which may differ with
checkout line endings; training preparation above enforces the recorded initial
model identity. Both helpers refuse prior attempts. The training helper stops on
the first failed command or comparison and terminates only its own process group
on timeout. Reports, checkpoints, output models and raw logs remain in the
separate output directory. It verifies input hashes, profile counts, numerical
limits and exact resume. It records executable hashes but does not infer Git
identity; establish clean source and build provenance separately.

## Support boundary

This acceptance covers one Linux x86_64/WSL2 runtime and an original native-v1
`se_e2_a` scalar-energy tanh fixture in FP64. It does not validate ARM64, other
drivers, cross-platform checkpoint restore, a production potential, long MD,
general DeepMD training or C API behavior. The CPU-only build and Windows matrix
were not repeated in this platform check.

Existing inference/training limits remain; see [native CUDA](native/CUDA.md),
[full training](native/README.md), [indexed data](native/INDEXED_TRAINING.md) and
[tiling](native/TILED_TRAINING.md). Inference supports its bounded tanh/linear
native-v1 subset. Full GPU training requires tanh embedding and hidden layers
with a linear fitting output, at most 2,048 neural parameters, width/descriptor
width 32, 64 slots and 64 atoms per frame, plus the existing work/memory caps.
This case does not establish correctness or performance at those limits. CPU
defaults and explicit device selection are unchanged.
