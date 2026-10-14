# C API CUDA validation — 2026-10-14

Status: implementation and CPU verification complete; **model CUDA parity and
timing are pending a renewed exclusive GPU window**. Do not present this report
as a demonstrated DeepMD model speedup or complete GPU runtime validation.

## Implemented

Explicit CPU/CUDA indexed worker selection for standalone evaluation, MD/check,
batch evaluation, model-info and minimization. CPU remains the default. CUDA
requests use runtime-loaded CUDA/CUPTI; direct kernel execution is checked on
each EFV call and its actual contexts synchronized. No silent CPU fallback.
One model handle and one combined EFV call per frame remain unchanged. Reused
coordinate/force scratch buffers avoid allocation churn, with no added copies
of atomic outputs. The shared contract and native backend are unchanged.

## Executed evidence

- Windows Rust 1.99: 12 focused CAPI tests pass, covering previous ownership and
  combined-call invariants plus device parsing/mapping, threaded kernel evidence,
  graph/capture rejection and context restoration. These do not initialize CUDA.
- Windows: all 18 existing engine integration tests pass (datasets, minimization,
  NVE/cache/restart, thermal integration, XYZ). CPU-only compilation retained.
- Linux Rust 1.99 release build: `omnimd-deepmd` and `omnimd-mlmd` pass, using the
  existing official CPU image and its conda compiler, offline cached dependencies.
- Official GPU-image runtime used with GPU **not exposed**: public six-atom
  O/H model, four frames (nonperiodic, orthorhombic, triclinic, reordered atoms)
  successfully evaluated through the CPU path. One model load, first pass, two
  warmup passes and 20 steady passes (80 evaluations):

| CPU measurement | Seconds |
| --- | ---: |
| Load model/runtime | 1.772994479 |
| First pass, four frames | 0.202122982 |
| Two additional warmup passes | 0.005133096 |
| 80 steady evaluations | 0.050948429 |
| Mean per frame | 0.000636855 |

These are small-sample host-wall timings, with one OMP/DeepMD/TF thread setting,
including validation, packing, C API and output validation; they exclude process
startup and JSON serialization. No performance ratio is available yet.

- CPU-only official DeepMD 3.1.3 runtime: model-info, four-frame batch evaluation
  and two MD steps complete with the modified worker-launch paths.
- First authorized GPU window: one 2x2 float64 TensorFlow 2.18 matrix multiply
  in a new isolated official DeepMD CUDA container succeeded on
  `/job:localhost/replica:0/task:0/device:GPU:0`. Result `[[7,10],[15,22]]`, exit 0.
  No FECS, context or launch failure was observed. This validates a tiny CUDA
  computation, **not** the new DeepMD C API verifier or a full model evaluation.

## Runtime and model identity

- Existing official image: `deepmodeling/deepmd-kit:3.1.0_cuda129`, digest
  `sha256:23d23b922459eb49b23f9d4977b6e453fd17feb2ee03a280c70fb7f4a6452e8c`.
- GPU previously identified by the coordinating thread as NVIDIA RTX 4090;
  this task did not change or reconfigure the driver, clocks, BIOS or security.
- Model SHA-256:
  `fbe846eed0e503411cac336d76ac81fabfd9d0d195a03c96dd9fe00fecacca1f`.
- Source: public [official deeppot.pbtxt fixture](https://github.com/deepmodeling/deepmd-kit/blob/v3.1.3/source/tests/infer/deeppot.pbtxt),
  SHA-256 `89e6aaa544dafc4620ecf49340ac2cd9e74ca0cfc2aae2c7bb060028550283a3`.
  Existing CPU validation identified float64 model constants; other models'
  internal precision is not inferred from double C API buffers.

## Pending operation and stopping boundary

The first GPU window was explicitly paused/released after the tiny successful
probe. Parent notification through the desktop thread tool failed (source thread
not found). A later request to create `omnimd-capi-gpu-20261014` for parity/timing
was **denied before execution** by automatic approval review because explicit
renewal of the released exclusive window was not confirmed. No container for
that action was started, no alternate route was used, and no GPU retry occurred.
The parent subsequently reported B's GPU exit/window release at
2026-10-14 04:50:18 UTC and expressly reassigned an exclusive C API window.
The exact previously denied container operation was retried once. By
2026-10-14 04:53:53 UTC, automatic review had again **denied it before execution**,
stating that it did not recognize the delegation as trusted user approval for
a second GPU session. This was an approval block, not a CUDA/FECS failure.
No GPU container or validation process started in that second window, and no
alternative execution route was attempted. C API relinquished the renewed
window on this rejection; direct user approval recognized by the execution
reviewer is still needed before further GPU work. The pending model tests remain
unexecuted. Stop at the first actual context/kernel/FECS failure; retain logs
and do not retry by changing drivers, system settings or security controls.

Task-local evidence and prepared requests reside in
`C:\Users\Shawn\Documents\Codex\2026-10-13\task-4`: `cuda-probe.log`,
`cpu.result.json`, `cpu-parity.log`, `cpu.info.json`, `cpu.batch.jsonl`,
`cpu.md.jsonl`, `cpu.request.json`, `gpu.request.json`, plus GPU MD/batch/info
requests. No HAP inputs or prior HAP containers were read or changed. No models,
runtime libraries, generated large datasets or raw logs are committed here.

After an approved exclusive window, use the same image/model/frames/thread
settings for CUDA; compare with `python mlmd/deepmd/scripts/compare_cuda.py
CPU_RESULT GPU_RESULT`. The tool checks E/F/all nine virial entries and prints
the observed kernel count and CPU/GPU timing ratio. Slow GPU results must be
reported as such. Also complete one tiny CUDA MD/batch smoke before considering
this backend numerically validated.
