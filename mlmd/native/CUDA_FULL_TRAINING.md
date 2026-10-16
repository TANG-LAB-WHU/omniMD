# Bounded full neural-parameter CUDA training

The `omnimd-native-full-train-v1` request now accepts explicit `"device":"cuda:N"`
with a CUDA-enabled build. All supported embedding/fitting weights, biases and
present timesteps receive E/F and optional [virial-loss](VIRIAL_TRAINING.md)
gradients and Adam updates computed in CUDA
float64. The default remains `"cpu"`. This is distinct from the frozen-readout
feature/CPU-QR mode, and is not an end-to-end GPU training system.

The default hyperdual gradient executor described below is retained as a numerical
reference. An opt-in [cached reverse executor](CUDA_REVERSE_TRAINING.md) removes
per-parameter network forward replay under the same supported bounds.

## Computation and bounds

| Work | Execution |
| --- | --- |
| Current model E/F inference and detached loss cotangents | CPU reference |
| PBC/image neighbors, normalized environment and displacement Jacobians | CPU |
| All neural parameter gradients, including force mixed derivatives | CUDA |
| Atom/frame gradient reduction, global norm/clipping, bias-correction scalars | CPU in stable order |
| Adam first/second moments and parameter updates | CUDA, resident arrays |
| Candidate validation, checkpoint serialization and metrics | CPU |

CUDA full training requires tanh in every embedding/hidden layer, a linear
fitting output, native-v1 validation, and the following joint limits: 2048 neural
parameters, every network width and descriptor input width at most 32,
`sum(sel)<=64`, at most 64 atoms/frame, 1024 frames, eight types and eight layers
per network. Descriptor width is `embedding_width * axis_neuron`, so bounding
just embedding width is insufficient. Existing 128 MiB dataset and 16 MiB
record limits also apply. Unsupported cases fail explicitly; they do not fall
back to CPU. The CPU trainer retains its separately documented bounds.

A further per-gradient policy cap rejects combined estimated work above
50,000,000 scalar contraction terms. For each center type, the estimate sums
`sum_t(sel_t * embedding_matrix_entries(center,t))`, fitting matrix entries,
`4*sum(sel)*embedding_width` and `4*embedding_width*axis_neuron`; the maximum
center cost is multiplied by atom count and total parameter count with checked
arithmetic. It is a conservative acceptance policy, not a proven runtime or
memory ceiling, and supplements rather than replaces the shape bounds. The
64-atom measured fixture estimates 1,792,000 terms, well below the policy cap.
The trainer checks its largest frame before context creation, and the low-level
API checks every actual frame before prediction or GPU work.

All network copies are independent parameters, including repeated embeddings
from one-side source models. Normalization, cutoffs, neighbor capacities,
architecture, type map and separate atomic energy offsets are fixed. Both E and
F labels remain mandatory even for a zero-weight loss component. Virial labels
are assessment-only by default; explicit `config.virial_loss` enables the
[V3 nine-component loss](VIRIAL_TRAINING.md) and requires them on every frame.
There is no automatic normalization/offset fitting, mixed precision, minibatching
or unsupported-architecture emulation.

## Parameter-coordinate mixed derivatives

The loss and scales are unchanged from [FULL_TRAINING.md](FULL_TRAINING.md).
For the default E/F case, detached loss cotangents `a=dL/dE`, `v=dL/dF`, and `F=-grad_r E`:

```text
grad_theta L = a * grad_theta E - grad_theta (v . grad_r E)
R_u = J * (v_neighbor - v_center)
```

One CUDA thread owns a central atom and one selected parameter. Each scalar
carries four components `(value, coordinate_direction, parameter_direction,
mixed_direction)`. Raw environment rows have parameter/mixed components zero;
the selected parameter has parameter component one at every occurrence.
Normalized padding retains its value but has zero tangent. Periodic self-image
force tangents cancel; their optional virial tangents remain. The virial extension
adds `J*(d*C)` using the full lattice-image displacement, as derived in
[VIRIAL_TRAINING.md](VIRIAL_TRAINING.md). Parameters and loss seeds refer to the same current
model snapshot; the loss seeds are not differentiated again.

Addition, multiplication, tanh, residual and trainable timestep operations
propagate all four components. Multiplication includes both cross terms. For
`tanh(x)` the mixed component is
`(1-y*y)*x_mixed - 2*y*(1-y*y)*x_coordinate*x_parameter`, where `y=tanh(x)`.
The kernel returns `a*E_parameter - E_mixed`. No parameter finite differences or
full Hessian are used in training. Atomic gradients return to the CPU for a
fixed-order sum; no floating-point atomic accumulation is used. The CUDA primal
is checked against CPU inference on every frame.

This is a bounded forward-differentiation implementation: work grows with the
number of parameters. It establishes an independently checked all-parameter
GPU path and a reference for later shared-intermediate/reverse-mode kernels.
It is not a claim of optimal complexity or a new differentiation algorithm.
The four-component chain rule follows established hyper-dual differentiation;
see [Neuenhofen's review](https://arxiv.org/abs/1801.03614). Code is original MIT;
no external AD/optimizer implementation was copied.

## Usage and checkpoint semantics

```sh
cargo build --release --locked --offline --manifest-path mlmd/native/Cargo.toml \
  --features cuda --bins
omnimd-native-train full-train.request.json
# The engine forwards the same request to its adjacent trainer executable:
omnimd-mlmd train-native full-train.request.json
```

Start from the full request in [FULL_TRAINING.md](FULL_TRAINING.md), add
`"device":"cuda:0"`, and choose a new output directory. Existing model and dataset
formats, loss settings, validation selection and output names are unchanged.
The Rust entry point is `train_full_with_device`; `train_full` still selects CPU.
Low-level `CudaFullTrainer::adam_step` expects an already reduced/clipped batch
gradient; the normal trainer performs that reduction and clipping exactly once.

For E/F-only runs, CPU checkpoints remain `omnimd-native-full-checkpoint-v1`; their default CPU
execution tag is omitted, preserving the existing serialized schema. CUDA
checkpoints use `omnimd-native-full-checkpoint-v2` and include
`execution.backend="cuda_hyperdual_f64_v1"`, visible device ordinal, reported
name/compute capability, CUDA driver API version, and SHA-256 of the actual
compiled PTX. They also retain the model, original anchor, moments, global step,
config, dataset/layout hashes and split indices. A nonzero-step resume must use
the same backend/device and recorded runtime/program identity. The original
CPU parser cannot read a GPU v2 checkpoint. To switch backend intentionally,
start a new optimizer from `model.json`; do not edit checkpoint provenance.
Enabled virial loss uses v3 on both devices with an immutable loss contract;
see [virial checkpoint compatibility](VIRIAL_TRAINING.md#checkpoint-compatibility).

`driver_version` is the CUDA API version reported by the driver, not its vendor
package build number. These checks detect recorded incompatibilities; they do
not identify a physical board by serial number or authenticate a toolchain.
Exact resume was verified within the same build, platform and installed runtime;
cross-build/platform/driver bitwise equality is not promised. Checkpoints are
not cryptographically signed. Backend identity together with the checkpoint
loss contract specifies host numerical semantics. Historical E/F semantics stay
unchanged; new virial semantics require v3. The PTX hash alone does not cover
host code.

Model/data/split/bounds/backend/global-step checks precede CUDA creation. PTX
and runtime identity are checked after context creation but before model upload
or kernel execution. Geometry and numerical failures remain fallible during
training. A rejected CUDA candidate leaves the last validated host snapshot
and earlier saved checkpoints intact, and poisons the CUDA instance. No automatic
retry or fallback occurs. Output directories are created only when saving the
first completed checkpoint. A later error may leave earlier valid checkpoints.

Reports record the execution split and host wall times for initialization,
complete gradient computation, Adam plus host candidate transfer/validation,
and assessment. Timings exclude checkpoint/file IO and export. Weights and Adam
moments remain resident during a run; candidate state is copied back after each
step for coherent CPU inference and checkpoints.

## Verification and measured limits

```sh
cargo test --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda
# GPU tests are ignored by default. Run only in an authorized GPU window:
cargo test --release --locked --offline --manifest-path mlmd/native/Cargo.toml \
  --features cuda --test cuda_full_training -- --ignored --nocapture --test-threads=1
python mlmd/native/tools/replay_full_training.py \
  --eval-exe /path/to/omnimd-native-eval --train-exe /path/to/omnimd-native-train \
  --device cuda:0 --compare-cpu --output-dir /new/output/directory
```

Windows RTX 4090 checks on 2026-10-16 used original synthetic inputs only:

- All 112 neural parameters, E-only/F-only/combined losses, periodic/nonperiodic
  frames, normalized padding, zero/one parameter values, both residual forms and
  trainable timesteps: maximum gradient difference from the independent CPU
  reverse-over-forward AD was `7.11e-15`. Central differences gave a maximum
  absolute difference of `7.54e-8` at step size `1e-5`.
- Width-32 and axis-2 cases passed; gradient maxima were `7.11e-15` and `8.88e-16`.
  Zero loss gave zero gradient; frame averaging was checked independently.
- With active global clipping, two-step Adam parameters differed from CPU by at
  most `9.02e-17`, moments by `1.69e-21`. Five continuous steps exactly matched a
  serialized two-plus-three restart. A deliberately overflowing Adam candidate
  was rejected, preserving the host model and stopping that instance.
- The actual CLI trained all 56 embedding, 48 fitting-hidden and eight output
  parameters for 300 steps on 18 frames, with six held-out frames. The final
  validation E/F objective was `0.00445656` versus initial `10.1403`.
  CPU/CUDA final parameters differed by `1.67e-15`; saved-model E/F/virial maxima
  were `8.88e-16 / 2.22e-15 / 2.44e-15`. A 150+150 checkpoint replay matched all
  model values and moments exactly. Serialization/reload differences were zero.
- Existing CUDA inference and frozen-readout paths passed affected regressions.

The tiny 3/4-atom, 300-step case was slower on CUDA: gradient phases totaled
1481.18 ms versus CPU 603.68 ms; CUDA initialization was another 208.34 ms.
A separate 64-atom/112-parameter synthetic separated-pair frame gave 10.98 ms
CPU versus 3.17 ms CUDA across five warmed full-gradient calls, a **3.46x ratio**.
Those calls include CPU inference/loss, geometry, transfers, kernel and reduction;
they exclude initialization, Adam and checkpoint IO. This one bounded result is
not a whole-training speedup, scaling benchmark or physical-accuracy claim.

[The committed report](fixtures/cuda-full-training-report.json) contains commands,
hashes, scopes, metrics, timings and runtime identity. Generated datasets,
weights, checkpoints, executables and raw logs remain ignored. No dependency or
runtime installation was performed. Linux CUDA execution, external driver fault
injection and representative physical training validation remain unverified.
