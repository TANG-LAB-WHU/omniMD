# C API CUDA validation - 2026-10-14

**Completed:** official TensorFlow C API inference on RTX 4090, per-evaluation
CUDA kernel evidence, CPU/GPU numerical agreement, paired timing, and GPU batch/MD
integration. This six-atom sample is **8.27 times slower on GPU** with the current
verified path. This is not a general speedup or production-stability claim.

## Numerical agreement and execution evidence

The same public float64 O/H model was evaluated on four six-atom frames:
nonperiodic, orthorhombic, triclinic, and permuted atoms in the triclinic cell.
All 112 output scalars passed atol=1e-8, rtol=1e-8:

| Output | Maximum absolute CPU/GPU difference |
| --- | ---: |
| Total energy | 1.1368683772161603e-13 eV |
| Force component | 1.5543122344752192e-15 eV/angstrom |
| All nine virial components | 2.4424906541753444e-15 eV |

The evaluator observed **24,863 successful direct CUDA kernel launches across
92 evaluations** (four initial, eight warmup, eighty timed). Every evaluation
synchronized the actual observed contexts before publishing EFV. Metadata was
`deepmd-c-api-cuda-kernels-verified`. This establishes GPU computation, not that
every TensorFlow operation runs on GPU.

The unified GPU batch command completed four frames; maximum differences from
standalone GPU output were E=0, F=1.11e-16 and W=2.22e-16. GPU MD completed two
steps and its checkpoint, with verified CUDA metadata and maximum absolute
total-energy drift 3.64e-12 eV. This is an integration smoke test. All four
commands (GPU eval, paired CPU eval, GPU batch, GPU MD) exited zero. No GPU
context/launch/synchronization failure or FECS report was observed.

## Paired host-wall timing

Both devices used the same final binary, model, frames and official runtime in
the same container. OMP, TF intra/inter-op and DeepMD intra/inter-op thread
settings were one. CPU workers hid accelerator visibility. Each request retained
one model through an initial pass, two warmup passes and twenty timed passes.

| Measurement | CPU seconds | GPU seconds |
| --- | ---: | ---: |
| Model/runtime load | 1.565642704 | 2.765030132 |
| First pass, four frames | 0.194486163 | 0.343639422 |
| Two warmup passes | 0.003633622 | 0.032982485 |
| 80 steady evaluations | 0.038843781 | 0.321416035 |
| Mean per frame | 0.0004855473 | 0.0040177004 |

GPU/CPU elapsed ratio is **8.27458**, CPU/GPU speed ratio **0.120852**. This is
one bounded paired sample, not a statistical benchmark. Timing includes host
validation/packing, combined C API EFV, required transfers, CUDA CUPTI callback
checks and context synchronization, and output validation. It excludes process
startup, input parsing and output JSON serialization. Loading and warmup are
separate. CPU remains the default for small systems. These results do not
determine larger-system or native CUDA performance. No large benchmark was run.

## Builds and identities

- Implementation: `bf6f2cfba09a4d01ad551ee036876060828e50fd`; source HEAD at run:
  `46d913b3dbe3726843cca52e7b61d655756da7dd` (documentation follow-up).
- Windows Rust 1.99: 12 CAPI plus 18 engine integration tests pass without CUDA
  initialization. Linux release binaries build offline with the existing official
  CPU image/compiler. No manifest/lock changes. CPU-only DeepMD 3.1.3 model-info,
  four-frame batch and two MD steps also passed before GPU validation.
- GPU runtime: DeepMD 3.1.0, TensorFlow 2.18.0, official image
  `deepmodeling/deepmd-kit:3.1.0_cuda129`, digest
  `sha256:23d23b922459eb49b23f9d4977b6e453fd17feb2ee03a280c70fb7f4a6452e8c`.
- GPU: NVIDIA GeForce RTX 4090, PCI bus `0000:01:00.0`, compute capability 8.9,
  worker logical GPU 0.
- Model SHA-256: `fbe846eed0e503411cac336d76ac81fabfd9d0d195a03c96dd9fe00fecacca1f`.
  Public [official deeppot.pbtxt fixture](https://github.com/deepmodeling/deepmd-kit/blob/v3.1.3/source/tests/infer/deeppot.pbtxt),
  source SHA-256 `89e6aaa544dafc4620ecf49340ac2cd9e74ca0cfc2aae2c7bb060028550283a3`.
  Its constants were previously identified as float64. Double C API buffers do
  not establish other models' internal precision.
- Runtime/library and binary hashes: [compact receipt](validation/cuda-public-six-atom.json).

## Completion, scope and reproduction

Earlier container requests were denied before execution during window
coordination. Direct user approval resolved that block; the same creation was
accepted. No bypass or alternate execution route was used. The dedicated
`omnimd-capi-gpu-20261014` container stopped at
**2026-10-14T06:13:51.533641632Z**, with `Running=false`, `Pid=0`, `OOMKilled=false`.
The exclusive GPU window was released. Container exit 137 is the stopped idle
PID-1 sleep: all validation processes had already exited zero, and docker top
showed only sleep before the bounded container stop.

GPU scope remains official TensorFlow `.pb`, no spin or extra frame/atom
parameters, direct kernels without CUDA graphs/stream capture. Model-info alone
does not attest evaluation. CPU selection remains available; no silent fallback.
Model-load-once and one-combined-EFV-per-frame semantics are retained. No HAP
inputs or old HAP containers were read or changed. No driver, BIOS, clocks or
security settings were changed. No models or runtime binaries are committed.

Raw results, logs and requests are under
`C:\Users\Shawn\Documents\Codex\2026-10-13\task-4`: `gpu.result.json`,
`cpu.same-runtime.result.json`, `cuda-parity-report.json`, `cuda-success-receipt.json`,
`gpu.batch.jsonl`, `gpu.md.jsonl`, and corresponding requests/logs. Compare saved
results without GPU access:

```sh
python mlmd/deepmd/scripts/compare_cuda.py CPU_RESULT GPU_RESULT
```

The comparator checks all E/F/virial scalars, execution evidence, matching timing
counts and finite positive times, including under python -O. It does not infer
model/input identity: the recorded hashes and requests establish that here.
Stop future validation at the first actual context/kernel/FECS failure; retain
evidence instead of changing system settings.
