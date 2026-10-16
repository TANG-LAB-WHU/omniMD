# Native Rust se_e2_a: CPU/CUDA inference and bounded training

`omnimd-native` implements the shared `omnimd_ml_contract::Potential` interface
with float64 energy, analytic forces and strain-gradient virial in one call.
CPU is the default and needs no DeepMD, Python, BLAS or CUDA runtime. Optional
CUDA inference dynamically uses an existing official NVIDIA Driver/NVRTC
installation. The crate is already a member of the `mlmd` Cargo workspace.

This is a progressively implemented subset, not all DeepMD functionality:

| Operation | Native CPU | Native CUDA |
| --- | --- | --- |
| Native-v1 se_e2_a scalar-energy inference, E/F/virial | Implemented | Implemented within [CUDA bounds](CUDA.md) |
| tanh/linear, explicit residuals and timesteps | Implemented | Implemented |
| sigmoid/softplus inference | Implemented | Rejected |
| Final-readout E/F fitting | Implemented, bounded | Implemented: GPU features, CPU QR/assessment |
| Full supported tanh-network E/F + optional virial training and Adam restart | Implemented, bounded | [Bounded GPU gradients/Adam; optional GPU loss](DEVICE_LOSS_TRAINING.md) |
| Arbitrary DeepMD models, attention/spin/hybrid/compression | Not implemented | Not implemented |

Conversion supports a pinned DeepMD 3.1.3 interchange subset and two audited
frozen graph profiles. Unsupported formats/architectures fail explicitly.
See [conversion](CONVERSION.md), [frozen export](FROZEN.md),
[training scope](FULL_TRAINING.md) and [remaining training work](TRAINING_NEXT.md).
GPU inference is described in [CUDA_FIT.md](CUDA_FIT.md); full neural-parameter
training, its exact computation boundary and timings are in
[CUDA_FULL_TRAINING.md](CUDA_FULL_TRAINING.md). The optional nine-component
virial loss and compatibility rules are in [VIRIAL_TRAINING.md](VIRIAL_TRAINING.md).

## Step 1: Build the CPU baseline

Run from the repository root with Rust and its platform linker available:

```sh
cargo build --release --locked --manifest-path mlmd/native/Cargo.toml --bins
cargo test --locked --manifest-path mlmd/native/Cargo.toml --no-default-features
```

Add `--offline` when the existing dependency cache is complete. No package or
CUDA driver is automatically installed. Executables normally appear under
`mlmd/target/release`; `CARGO_TARGET_DIR` overrides that location. Commands below
assume that directory is on PATH. Append `.exe` on Windows.

## Step 2: Select or convert a model

For a software smoke test, use `mlmd/native/fixtures/full-tanh-v1.json`. Its
original synthetic weights are not a physical production potential.
For an already supported DeepMD model, choose exactly one applicable converter:

```sh
# Existing h5py/PyYAML is required by the interchange converter.
python mlmd/native/tools/convert_deepmd.py model.dp native.json \
  --units EvAngstrom --geometry tensorflow

# Existing TensorFlow/protobuf/NumPy is required for GraphDef decoding only.
python mlmd/native/tools/export_frozen.py model.pb native.json --units EvAngstrom
```

Use `--geometry dpmodel` only when that is the intended source geometry;
TensorFlow environment cutoff attributes have distinct float32 storage semantics.
Do not choose geometry mode to minimize an observed error. Output must be new.
The exporter does not execute the supplied graph. Read the linked exact profile
limits before conversion; `.pth` or an arbitrary `.pb` cannot be renamed to JSON.

## Step 3: Supply shared-contract frames and evaluate on CPU

Each nonblank input line is one `Frame`, not a labeled dataset record:

```json
{"positions":[[0,0,0],[0.9572,0,0],[-0.239,0.927,0]],"species":["O","H","H"],"cell":null,"units":"EvAngstrom"}
```

Species must exist in the model's ordered `type_map`; no atomic-number or mass
inference occurs. Coordinates/cell use angstrom. Cell rows are lattice vectors;
`null` is nonperiodic, while a supplied cell is periodic in all three directions.

```sh
omnimd-native-eval mlmd/native/fixtures/full-tanh-v1.json \
  < mlmd/native/fixtures/frame.jsonl > predictions.jsonl
```

PowerShell equivalent:

```powershell
Get-Content mlmd/native/fixtures/frame.jsonl | omnimd-native-eval.exe mlmd/native/fixtures/full-tanh-v1.json
```

Each output contains energy (eV), forces (eV/angstrom), virial (eV) and units.
Virial is the negative row-strain energy derivative, not stress. A failed frame
returns a diagnostic and nonzero status without publishing that frame's partial
result; earlier successful lines remain in the output.

## Step 4: Enable CUDA explicitly

```sh
cargo build --release --locked --manifest-path mlmd/native/Cargo.toml --features cuda --bins
omnimd-native-cuda-check --compile-only mlmd/native/fixtures/full-tanh-v1.json
omnimd-native-cuda-check mlmd/native/fixtures/full-tanh-v1.json mlmd/native/fixtures/frame.jsonl 3
omnimd-native-cuda-eval mlmd/native/fixtures/full-tanh-v1.json < mlmd/native/fixtures/frame.jsonl
```

`--compile-only` invokes installed NVRTC without GPU execution. The other two
commands execute on visible device 0; use them only when GPU use is authorized.
On Windows `CUDA_PATH` points to the installed official toolkit, or set the
process-local `OMNIMD_NVRTC_LIBRARY` to its absolute NVRTC library path. See
[CUDA.md](CUDA.md) for exact library names and Linux runtime limitations. A CUDA
failure is an error and poisons the instance; CPU fallback is an explicit new
CPU request. Small systems may be faster on CPU.

Keep one potential alive across frames to reuse model weights, host/device
workspace and conservative neighbor candidates. Each frame still recomputes
its E/F/virial. JSONL streaming is sequential; no batched GPU API is claimed.

## Step 5: Connect to Rust or the shared MD engine

```rust
use omnimd_ml_contract::Potential;
use omnimd_native::NativePotential;

let mut potential = NativePotential::load("native.json")?;
let evaluation = potential.eval(&frame)?;

// With the cuda Cargo feature and explicitly selected visible device 0:
let mut gpu = omnimd_native::cuda::CudaPotential::new(potential.model().clone(), 0)?;
let gpu_evaluation = gpu.eval(&frame)?;
```

The caller handles fallible results; do not assume Send/Sync. `last_profile()`
and `kernel_resources()` provide optional diagnostics without changing the
shared trait. Reuse the same ordered model type map at every boundary.

```sh
cargo build --release --locked --manifest-path mlmd/Cargo.toml \
  -p omnimd-mlmd --features native-cuda --bin omnimd-mlmd
omnimd-mlmd check request.json
omnimd-mlmd run request.json new-trajectory.jsonl
```

Copy `mlmd/engine/examples/native-nvt.json` to a new request. Set
`options.native_model` to the **absolute** path of your supported model,
`options.device` to `"cpu"` or explicit `"cuda:0"`, and review the frame, masses,
velocity initialization, timestep, ensemble and step count. For the software
smoke model, use `full-tanh-v1.json`. Existing outputs are not overwritten.
The engine handles isolated device selection, unit conversion and integration;
this crate supplies one combined E/F/virial evaluation. See
[engine usage and restart](../engine/README.md), [data import](../engine/DATA.md)
and [minimization](../engine/MINIMIZATION.md). No shared-interface changes are
required for the CUDA improvements.

## Step 6: Train the supported subset and reload

Import/prepare the exact `omnimd-dataset-jsonl-v1` schema through the engine data
workflow. Native trainers require every selected frame to have E and F labels;
missing labels are errors. Optional virial labels are assessment-only by default;
full-network requests can enable [virial supervision](VIRIAL_TRAINING.md). Preserve
ordered types/units and choose explicit validation IDs or a recorded seed.

Create a request using [TRAINING.md](TRAINING.md) for frozen-hidden readout
regression or [FULL_TRAINING.md](FULL_TRAINING.md) for all supported tanh-network
parameters. Both require an initialized native model and a new output directory:

```sh
omnimd-native-train train.request.json
# Optional engine forwarding, with the trainer executable beside the engine:
omnimd-mlmd train-native train.request.json
omnimd-native-eval /absolute/trained/model.json < held-out-frames.jsonl
```

Training writes provenance, split IDs and metrics; the full-network mode also
supports documented CPU deterministic checkpoint/resume. Readout requests may
set `"device":"cuda:0"` in a CUDA-enabled build for GPU feature generation with
CPU QR/assessment; omitted device remains CPU. Full-network requests also accept
explicit `"cuda:N"` within the tighter [CUDA training bounds](CUDA_FULL_TRAINING.md),
with GPU mixed derivatives/Adam and CPU loss/geometry/reduction. GPU checkpoints
use v2 for E/F-only or v3 with virial loss, and require the recorded
backend/device/runtime identity on resume.
Check held-out results
before MD. GPU inference can consume a trained native model only if it meets
CUDA's architecture/size bounds. These trainers do not implement end-to-end
GPU training, automatic normalization/offset fitting or generic DeepMD training
parity. Tiny training examples are slower on CUDA; the measured 64-atom gradient
case is faster, with its timing scope documented separately.

## Model format v1

The complete executable example is `fixtures/synthetic-v1.json`; public Rust
schema types are in `src/model.rs`. This is a custom intermediate format, **not
DeepMD's serialization format**, and is not a direct Rust loader for `.pb`, `.pth`, `.dp`, HAP, or hybrid models.
`tools/convert_deepmd.py` now converts a strict DeepMD-kit 3.1.3 `.dp` / safe YAML
subset into this representation. See [CONVERSION.md](CONVERSION.md) for the
explicit geometry mode, units, supported schema, and complete replay command.
`tools/export_frozen.py` separately exports two audited frozen `.pb` graph
profiles, including the public DeepMD test model; see [FROZEN.md](FROZEN.md).
Every native model object rejects unknown fields.

| Field | Required meaning |
| --- | --- |
| `format`, `version` | `"omnimd-native-se-e2-a"`, integer `1` |
| `precision`, `units` | `"float64"`, `"EvAngstrom"` |
| `type_map` | Ordered unique chemical symbols; no inferred atomic-number mapping |
| `descriptor.kind` | `"se_e2_a"` |
| `rcut_smth`, `rcut` | Finite radii with `0 <= rcut_smth < rcut` |
| `sel` | Fixed per-neighbor-species capacities in `type_map` order |
| `axis_neuron` | Positive, no larger than embedding output width |
| `normalization` | One `{mean: [4], std: [4]}` per central type, repeated across slots |
| `embedding` | Network table `[central type][neighbor type]`; duplicate rows for one-side models |
| `fitting` | One network per central type; final output is a scalar |
| `atom_energy_bias` | One finite additive energy offset per central type |

Normalization requires zero angular means, equal positive angular standard
deviations, and a positive radial standard deviation. All embedding networks
start with a scalar input and share one output width. Fitting input width is
`embedding_width * axis_neuron`. No units or precision conversion is implicit.

Each network has a nonempty `layers` array. Each layer has `weights[input][output]`,
`bias[output]`, `activation`, `residual`, and optional `timestep[output]` (or null).
Supported activations are `linear`, `tanh`, `sigmoid`, and `softplus`. Its exact rule is:

```text
y = timestep * activation(x W + bias) + residual(x)
```

Absent timestep means one. Residual is explicitly `none`, `identity` (equal
widths), or `concat2` (output twice input width, concatenation `[x,x]`). It is never
inferred from dimensions. The last fitting layer must be linear, have no residual,
and no timestep. Exporters must not add an energy offset twice.

## Mathematics and derivatives

For every central atom, enumerate all neighbors with displacement
`d = r_j - r_i + n_a*a + n_b*b + n_c*c` and `0 < |d| < rcut`. This includes
nonzero periodic self-images. For `u=(r-rcut_smth)/(rcut-rcut_smth)`, use
`q=1` below smoothing, `q=1-10u^3+15u^4-6u^5` in the switching interval,
and zero outside the cutoff. With `s=q/r`:

```text
R_raw = [s, s*d_x/r, s*d_y/r, s*d_z/r]
R = (R_raw - mean) / std
G = embedding[central_type][neighbor_type](R[0])
B = G^T R / sum(sel)
D[m,k] = sum_a B[m,a] B[k,a],  k < axis_neuron
E = sum_i (fitting[type_i](flatten_m_then_k(D_i)) + atom_energy_bias[type_i])
```

Unused slots are zero **before** normalization. Their normalized radial values
and resulting descriptor contributions are retained, including for isolated
atoms. The denominator is fixed capacity, not the current neighbor count.

Forces use the analytic chain rule through the quintic switch, environment
normalization, embeddings, descriptor contraction, and fitting networks.
Embedding derivatives propagate forward from the scalar input; fitting
sensitivities propagate backward. There is no finite-difference force evaluation.
For `g = dE_i/dd`, contributions are `F_i += g`, `F_j -= g`, and
`W_ab -= d_a*g_b`, without a pairwise half factor. This exactly follows the
shared row-strain convention `r'_b = r_b + sum_a r_a*epsilon_ab` with the cell
deformed identically. Virial is not divided by volume.

Periodic search bounds use reciprocal-cell column norms: any in-cutoff image
must satisfy `|fractional_delta[k] + n[k]| <= rcut*norm(inverse_cell[:,k])`.
All candidate images in the resulting integer box are checked. This handles
triclinic and small cells; it does not assume nearest fractional rounding.
Capacity overflow fails instead of selecting a truncated neighborhood.

## Scope, acceptance, and next integration step

The fixture is synthetic and deterministic, created by
`python mlmd/native/tools/generate_fixture.py`; it is not a trained chemical
potential and must not be used to draw physical conclusions. It exercises two
species in reversed atomic-number order, nonzero radial means, distinct type-pair
embeddings, both residual forms, timesteps, biases, and non-collinear geometry.

Seven integration tests cover coordinate and all nine affine-strain differences,
a nonperiodic cluster, triclinic cells and periodic self-images, an independent
closed-form normalized-padding energy, a skew-cell integer-cube oracle,
translation/rotation/permutation/wrapping, cutoff/smoothing boundaries,
activation derivatives, explicit type ordering, and required error cases.
The initial Windows GNU build used the already available Rust 1.99.0 toolchain
and cached dependencies offline; all seven tests passed. The sample periodic
fixture energy is approximately `-0.9040314614729545` eV. This sample value is
from this implementation, not an independent DeepMD reference.

For later cross-engine acceptance, evaluate the **same converted weights and
same ordered frame** with A's DeepMD backend, then compare shared-contract JSONL
outputs using `tools/compare_evaluations.py`. A fixed-seed, untrained DeepMD
3.1.3 model has now passed actual C API/native comparison on three frames; the committed conversion report records maximum
E/F/virial errors of 4.44e-16 / 5.03e-17 / 3.47e-17. This establishes numerical
compatibility for that fixture, not pretrained-model physical accuracy.
Native training modes are documented below. This crate has no MD integrator;
MD is provided by the shared engine. Compression tables, attention, spin,
type embedding, `se_e3`, hybrid descriptors, exclusions,
frame/atomic parameters, environment protection and vacuum subtraction are not
implemented. Such architectures
must not be relabeled as this format; unknown fields and unsupported enums fail.

The default CPU implementation rebuilds direct neighbor lists and temporary
network buffers each evaluation. CUDA inference reuses conservative candidate
lists and host/device workspaces as documented in CUDA_INFRA.md and CUDA_FIT.md. Neighbor enumeration is quadratic in atom count before
image expansion; no production scaling or speedup claim is made. Explicit
resource limits are 256 types, 65536 total neighbor slots, 128 layers per network,
65536 outputs per layer, and 2,000,000 candidate images per central atom. Image
ranges beyond representable bounds, ill-conditioned cells, distances <= 1e-12
angstrom, nonfinite parameters/intermediates/outputs, and malformed shapes fail.

See `THIRD_PARTY_NOTICES.md` for mathematical references and licenses.

The public frozen-model replay now passes four frames with maximum E/F/virial
errors of 2.27e-13 / 4.22e-15 / 6.66e-15. Exact graph-profile limits and replay
evidence are in [FROZEN.md](FROZEN.md). Frozen-hidden final-readout training is now implemented; see
[TRAINING.md](TRAINING.md) for the native CLI, exact E/F features, reproducible
split and actual teacher-recovery results. Full supported tanh-network E/F
training with exact mixed AD, Adam and deterministic checkpoint/restart is now
implemented in [FULL_TRAINING.md](FULL_TRAINING.md), with an explicit bounded
[CUDA gradient/Adam backend](CUDA_FULL_TRAINING.md). Its scope is float64, small
models and fixed descriptor normalization; no general DeepMD training or
physical-accuracy claim is made. [TRAINING_NEXT.md](TRAINING_NEXT.md) lists
remaining training limitations.

Optional native CUDA inference is developed behind the `cuda` Cargo feature;
CPU remains the default. See [CUDA.md](CUDA.md) for actual GPU/CPU computation
boundaries, build/runtime requirements, strict limits and verification status.

Opt-in CUDA full-training gradient optimization: [cached reverse accumulation](CUDA_REVERSE_TRAINING.md).

Optional fixed-frame CUDA training preparation: [bounded geometry cache](FIXED_GEOMETRY_TRAINING.md).

Optional [deterministic minibatch training](MINIBATCH_TRAINING.md) preserves the
full-batch default and adds exact mid-epoch v5 checkpoint restart for the supported
CPU/CUDA E/F/V trainer.

Optional [bounded indexed dataset access](INDEXED_TRAINING.md) supports native
minibatch random access and exact restart without retaining all decoded frames.

Optional [constant and exponential learning-rate schedules](LEARNING_RATE.md)
use global optimizer-update counts and exact v6 checkpoint restart. Omitting the
schedule preserves legacy constant-rate behavior and checkpoint formats.

Optional [bounded multi-frame CUDA training](TILED_TRAINING.md) batches independent
frames through the reverse kernels while preserving optimizer batches and exact
v7 restart. Architecture and per-frame bounds remain unchanged.

Optional [GPU prediction and detached loss seeds](DEVICE_LOSS_TRAINING.md) reuse
primal intermediates within a tile and use explicit v8 restart identity. CPU
loss remains the default. Three-repeat 128-update synthetic whole-process medians
regressed 604.53->756.03 ms for 112 parameters and improved 3505.78->3060.90 ms
for 466 parameters (1.15x). Geometry/reduction/assessment stay on CPU; existing
architecture and per-frame bounds remain. See the [final engine guide](../GETTING_STARTED.md#23-select-gpu-prediction-and-loss-preparation-explicitly).
