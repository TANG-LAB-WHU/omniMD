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
| Native-v1 se_e2_r (radial descriptor) inference, E/F/virial | Implemented | On the [GPU runtime](../gpu/README.md) (`gpu` feature), whose [trainer](GPU_TRAINING.md) also trains it; the CUDA subset and the bounded trainers refuse it |
| Native-v1 se_e3 (three-body descriptor) inference, E/F/virial | Implemented | On the [GPU runtime](../gpu/README.md) (`gpu` feature), with [tables for its embedding nets](#embedding-tables) if asked, whose [trainer](GPU_TRAINING.md) also trains it; the CUDA subset and the bounded trainers refuse it |
| Native-v1 hybrid (several of these descriptors in one model) inference, E/F/virial | Implemented | On the [GPU runtime](../gpu/README.md) (`gpu` feature), with [tables](#embedding-tables) for every descriptor if asked; its [trainer](GPU_TRAINING.md) trains a hybrid of any of the three descriptors; the CUDA subset and the bounded trainers refuse it |
| tanh/linear, explicit residuals and timesteps | Implemented | Implemented |
| sigmoid/softplus inference | Implemented | Rejected |
| Final-readout E/F fitting | Implemented, bounded | Implemented: GPU features, CPU QR/assessment |
| Full supported tanh-network E/F + optional virial training and Adam restart | Implemented, bounded | [Bounded GPU gradients/Adam; optional GPU loss](DEVICE_LOSS_TRAINING.md) |
| Arbitrary DeepMD models, attention/spin/type embedding, compression | Not implemented | Not implemented (the GPU runtime has embedding tables) |

Conversion supports a pinned DeepMD 3.1.3 interchange subset and two audited
frozen graph profiles. Unsupported formats/architectures fail explicitly.
See [conversion](CONVERSION.md), [frozen export](FROZEN.md),
[training scope](FULL_TRAINING.md) and [remaining training work](TRAINING_NEXT.md).
GPU inference is described in [CUDA_FIT.md](CUDA_FIT.md); full neural-parameter
training, its exact computation boundary and timings are in
[CUDA_FULL_TRAINING.md](CUDA_FULL_TRAINING.md). Tiled trainers use
[bounded packed input transfers](TRANSFER_BATCHING.md) and
[resident GPU optimizer state](RESIDENT_STATE_TRAINING.md); their dated evidence
separates whole-process, startup and update timings, including small-case regressions. The optional nine-component
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
[CUDA.md](CUDA.md) for exact library names and the bounded
[Linux x86_64/WSL2 acceptance](../LINUX_CUDA_VALIDATION.md). A CUDA
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
DeepMD's serialization format**, and is not a direct Rust loader for `.pb`, `.pth`, `.dp` or HAP models.
`tools/convert_deepmd.py` now converts a strict DeepMD-kit 3.1.3 `.dp` / safe YAML
subset into this representation. See [CONVERSION.md](CONVERSION.md) for the
explicit geometry mode, units, supported schema, and complete replay command.
`tools/export_frozen.py` separately exports two audited frozen `.pb` graph
profiles, including the public DeepMD test model; see [FROZEN.md](FROZEN.md).
Every native model object rejects unknown fields.

| Field | Required meaning |
| --- | --- |
| `format`, `version` | `"omnimd-native-se-e2-a"`, `"omnimd-native-se-e2-r"`, `"omnimd-native-se-e3"` or `"omnimd-native-hybrid"`, integer `1` |
| `precision`, `units` | `"float64"`, `"EvAngstrom"` |
| `type_map` | Ordered unique chemical symbols; no inferred atomic-number mapping |
| `descriptor.kind` | `"se_e2_a"` in the first format, `"se_e2_r"` in the second, `"se_e3"` in the third, `"hybrid"` in the fourth; any other pairing is refused |
| `rcut_smth`, `rcut` | Finite radii with `0 <= rcut_smth < rcut`. hybrid: the largest `rcut` and the smallest `rcut_smth` of the descriptors in `list` |
| `sel` | Fixed per-neighbor-species capacities in `type_map` order. hybrid: for every type the largest of the descriptors in `list` |
| `list` | hybrid only: 1 to 16 descriptors, each an se_e2_a, se_e2_r or se_e3 descriptor with all the fields of this table that describe one (`kind`, radii, `sel`, `axis_neuron`, `normalization`, `type_one_side`, `embedding`). The hybrid itself has no `normalization`, no `embedding` and no axis neurons |
| `axis_neuron` | se_e2_a: positive, no larger than embedding output width. se_e2_r and se_e3: zero, or left out |
| `normalization` | One `{mean: [4], std: [4]}` per central type, repeated across slots. se_e2_r uses the first entry of each; the others are a zero mean and a unit width. se_e3 uses the width of the other three; its first entry is a zero mean and a unit width |
| `type_one_side` | Optional, false when left out. True says that the embedding nets depend on the neighbor type only (DeePMD's key of that name): every row of `embedding` must then hold the same nets, and the [GPU trainer](GPU_TRAINING.md) trains one net per neighbor type. False leaves the nets of every pair of types their own, alike or not. Not for se_e3 |
| `embedding` | Network table `[central type][neighbor type]`; with `type_one_side` every row repeats the nets of the neighbor types. se_e3: `[neighbor type][neighbor type]`, the net of a pair of neighbor types, the same in either order |
| `fitting` | One network per central type; final output is a scalar |
| `atom_energy_bias` | One finite additive energy offset per central type |

Normalization requires zero angular means, equal positive angular standard
deviations, and a positive radial standard deviation. All embedding networks
start with a scalar input and share one output width. Fitting input width is
`embedding_width * axis_neuron` for se_e2_a and `embedding_width` for
se_e2_r and se_e3, and for a hybrid the sum over the descriptors of its
list. No units or precision conversion is implicit.

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

An se_e2_r model (DeePMD-kit's `se_e2_r`, also `se_r`) keeps the radial entry
only. With the same `s`, `R[0]` and `G`:

```text
D[m] = (1/5) * sum_over_slots G[slot][m] / sum(sel)
E = sum_i (fitting[type_i](D_i) + atom_energy_bias[type_i])
```

DeePMD-kit averages the embedding rows of each neighbor type, weights the
types by `sel[t] / sum(sel)` and divides by five; that is the same sum. Unused
slots count as above. The embedding table is `[central type][neighbor type]`
here as well: DeePMD-kit 3.1.3 builds this descriptor with one net per
neighbor type, which the converter repeats for every central type and marks
with `type_one_side`.

An se_e3 model (DeePMD-kit's `se_e3`, also `se_at` and `se_a_3be`) takes
the neighbors of an atom in pairs. With `a = R[1..4]`, the three direction
entries of a slot's normalized row:

```text
theta_jk = a_j . a_k
D[m] = sum over neighbor types t <= u, slots j of t, slots k of u:
           theta_jk * embedding[t][u](theta_jk)[m] / (sel[t] * sel[u])
E = sum_i (fitting[type_i](D_i) + atom_energy_bias[type_i])
```

Within one type the sum has every ordered pair of slots, a slot with itself
included. An unused slot has zero direction entries (their mean is zero) and
adds nothing, so only neighbors count, while the denominator stays the
capacity. The nets do not depend on the central type; the central type
enters through the width of its direction entries. The number of pairs grows
with the square of the neighbors: n(n + 1)/2 evaluations of an embedding net
for an atom with n neighbors, where the other two descriptors have n.

A hybrid model (DeePMD-kit's `hybrid`) has several of these descriptors.
Their components, in the order of `list`, are the input of the fitting nets:

```text
D = [D_1, D_2, ..., D_n]        D_k: the descriptor of list[k]
E = sum_i (fitting[type_i](D_i) + atom_energy_bias[type_i])
```

All descriptors of a hybrid read one neighbor list, as in DeePMD-kit: it is
made with the largest of their cutoffs and, for every type, the most slots
any of them has (the `rcut` and `sel` of the hybrid). A descriptor has the
nearest `sel[t]` of the list's neighbors of type `t` in its own slots (its
own `sel`), and a neighbor at or beyond its own cutoff is an empty slot to
it: its switch is zero from there on. Each descriptor then is what it would
be alone, with its own radii, slots, normalization and nets. The one
difference is which neighbors it has when slots are short. Alone, a
descriptor keeps the nearest `sum(sel)` neighbors of any type and of those
at most `sel[t]` of each type; in a hybrid that selection is made once, with
the slots of the list, and the descriptor takes the first of its own slots
from the result. The CPU reference does not select at all: it refuses a
frame in which the list, or a descriptor within its own cutoff, has more
neighbors of a type than slots. The GPU runtime selects as described and
reports it.

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

## Embedding tables

On the GPU runtime the embedding nets can be replaced by piecewise quintics
(DeePMD-kit's model compression): `GpuPotential::compress(min_distance,
step)`, `--compress RMIN[:STEP]` of `omnimd-native-gpu-eval`,
`compress_min_distance` and `table_step` of the engine's `gpu_runtime`. A
frame with an input outside the tables is evaluated with the nets instead,
so a result never depends on the range of the tables; `table_fallbacks`
counts such frames.

**Why not a table of one step.** A trained net turns within a hundredth of
a unit of its input where its data is and not at all over most of its
range. A table with a knot every so many units is then coarse where the net
turns and fine where nothing happens:

- The nets of se_e2_a and se_e2_r take the normalized radial entry of a
  neighbor, some tens of units: in the HAP models up to 18 for a neighbor
  at 0.9 Å and up to 160 at 0.1 Å. Until 2026-11-18 their tables had a
  knot every `step`, 0.01 unless given. On the 83 frames of the HAP sealed
  test (largest force 6.2 eV/Å, float64, `--compress 0.5`) the forces from
  such tables were within 2.1e-7 eV/Å of those of the nets for se_e2_a
  models to 6 Å trained by omniMD, within 3.9e-6 for the pipeline's models
  of iteration 4, trained by DeePMD-kit, within 9.0e-6 for se_e2_r models
  trained by omniMD, and within 1.1e-3 for se_e2_a models to 4 Å trained
  by omniMD, in float32 as in float64. They were within 1.1e-10 for the
  se_e2_r models that DeePMD-kit trained here for 20,000 steps, whose
  weights are no larger than 1.1 where those of the others reach 2.5 to
  6.0. The tables of a model (25 nets of 100 outputs) took 392 MiB in
  float64 and 196 in float32.
- The nets of se_e3 take θ, the product of the normalized directions of two
  neighbors, and that spans thousands of units: ±1,400 for neighbors from
  0.9 Å in the HAP models, whose directions have a width of 0.03. With the
  step of 0.1 that DeePMD-kit takes for this descriptor the worst net of
  the HAP model is off by 2% of its largest value and 27% of its largest
  slope at its worst place, with 0.01 by 1e-5 and 7e-5, and tables of that
  step would take 4 GB in float32
  ([`se_e3_uniform_tables.py`](../bench/se_e3_uniform_tables.py),
  [numbers](../bench/results/hap-se-e3-uniform-tables-2026-11-18.json)).

**Pieces of uneven length.** So the tables of every descriptor have pieces
as long as their net allows
([`adaptive_table.rs`](src/adaptive_table.rs)). The range of a table, a
power of two from where it counts, is halved again and again, and a piece
is kept when

- no unit of the net that may be in the bent part of its activation can
  move by more than 2 in its argument over the piece (bounds by interval
  arithmetic through all layers), so that the net has no feature between
  the points a piece is checked at, and
- the quintic through value, slope and curvature of the net at the ends of
  the piece is within the tolerance of the net at a quarter, the middle and
  three quarters of it (between those places it strays a little further:
  the tests allow half as much again): the value within 1e-13 of the
  largest value of the net at the step 0.01, and (step/0.01)⁶ times that at
  another step, the slope within a hundred times that fraction of the
  largest slope. Float32 has 1.5e-8 at best, below what it rounds at. Where
  the net itself is not computed that exactly (a unit whose large weight
  and bias cancel where it turns), the bound on its rounding is allowed on
  top, and for the slope what the rounding of the values at the ends of a
  piece comes to over its width.

`step` is thus no distance between knots any more. It names a tolerance by
the table of one step it stands for: 1e-13 is where a table with a knot
every 0.01 follows a smooth net, and the error of such a table goes with
the sixth power of its step.

The ends of all pieces are whole multiples of powers of two, and the piece
of an input is found from the input itself: its sign, its exponent and its
first six bits name a cell of a directory, and a cell says how wide its
pieces are and where they start, `piece = base + floor(|x| / width)`; the
fraction that is left is the place in the piece. Nothing is rounded on the
way in float32 or float64, whatever the size of the input (a table that
counts steps from the lower end of a range of thousands rounds a float32
input to a thousandth of a unit). For this a cell that holds pieces of
several lengths has them all cut to its shortest (0.2% more pieces for the
se_e3 nets of a HAP model in a prototype of the builder). A net that turns
faster than the precision of its input has room for is refused, with the
net and the place, and the potential then keeps the nets of all its
descriptors: a hybrid has tables for every descriptor or for none. None of
the 147 models on disk here was refused, in either precision (142 trained
on the HAP data, se_e2_a, se_e2_r, se_e3 and hybrids of them, and the five
of the water benchmark).

**What a table covers.** Every table reaches to the inputs of neighbors at
a fifth of `min_distance`, at the same tolerance all the way (DeePMD-kit's
tables take ten times the step below `min_distance`).

- se_e2_a and se_e2_r: the table of a net counts from the entry of a
  neighbor at the cutoff, which is also that of an empty slot, and has
  pieces above that only. The place of a neighbor in it is s(r) over the
  width of the radial entry of the central type; the environment kernel
  computes it in double precision for a descriptor that runs from its
  tables, where it writes the normalized entry for one that runs its nets.
  A frame with a pair closer than a fifth of `min_distance` goes to the
  nets.
- se_e3: the table of a net covers θ of either sign, as far as two
  neighbors at a fifth of `min_distance` can bring it around a center of
  the type with the narrowest directions. With these tables the descriptor
  keeps no rows of embedding values: the kernels of its components and of
  its forces read the pieces where they would read the rows, and a chunk
  holds as many times more pairs as a pair takes less memory.

**What they come to.** The 83 frames of the HAP sealed test, `--compress
0.5`, four models of each kind: the largest difference of a force
component from the same model with its nets in float64, and the tables of
the first of the four:

| models | force (eV/Å) | float64 tables | float32 tables |
|---|---:|---|---|
| se_e2_a to 6 Å, the pipeline's models of iteration 4 (DeePMD-kit) | 7.8e-11 | 44,096 pieces, 202 MiB | 5,659 pieces, 13 MiB |
| se_e2_a to 6 Å, trained by omniMD | 1.4e-10 | 35,378 pieces, 163 MiB | 4,576 pieces, 11 MiB |
| se_e2_a to 4 Å, trained by omniMD | 1.1e-10 | 29,592 pieces, 136 MiB | 3,827 pieces, 9.0 MiB |
| se_e2_r to 6 Å, trained by DeePMD-kit for 20,000 steps | 1.1e-10 | 26,391 pieces, 121 MiB | 3,320 pieces, 7.8 MiB |
| se_e2_r to 6 Å, trained by omniMD | 4.4e-11 | 37,542 pieces, 172 MiB | 4,808 pieces, 11 MiB |
| se_e2_a to 4 Å with se_e2_r to 6 Å, trained by omniMD | 1.5e-10 | 66,552 pieces, 306 MiB | 8,697 pieces, 21 MiB |
| se_e3 to 4 Å | 1.6e-10 | 26,966 pieces, 50 MiB | 4,092 pieces, 4.1 MiB |
| se_e2_a to 6 Å with se_e3 to 4 Å | 8.6e-11 | 53,518 pieces, 186 MiB | 7,426 pieces, 13 MiB |
| the earlier hybrid, se_e2_a to 5 Å with se_e3 to 3.2 Å | 2.1e-10 | 46,973 pieces, 113 MiB | 6,358 pieces, 7.8 MiB |

The tables of a model are made in 0.1 to 0.9 s in float64 and in 0.02 to
0.11 s in float32. In float32 the tables are 4.4e-6 to 3.1e-5 eV/Å from the
float64 nets of their model, where its float32 nets are 5.2e-6 to 3.3e-5
from them; by kind of model, and beside the tables of one step, in the
[summary](../bench/results/SUMMARY.md), with the timings (an evaluation
costs what it did with the tables of one step).

In float64 the step trades size for closeness. For the pipeline's model 0
on the same frames: 268 MiB and 1.3e-11 eV/Å at 0.005 (the tolerance is
then 1e-14, the best there is), 202 MiB and 7.8e-11 at 0.01, 96 MiB and
2.5e-9 at 0.02, 37 MiB and 2.7e-7 at 0.05. In float32 these are all the
same tables, at the 1.5e-8 that float32 has at best.

A net as smooth as an untrained one was followed to rounding by a table
with a knot every 0.01, and is followed to the tolerance by these: for the
models of the water benchmark the float64 forces are 1e-14 to 2e-13 eV/Å
from those of the nets, where they were within 1e-15.

Records:
[`hap-tables-against-nets-...json`](../bench/results/hap-tables-against-nets-rtx4090-2026-11-18.json),
[`tables-of-every-model-...json`](../bench/results/tables-of-every-model-rtx4090-2026-11-18.json).

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
MD is provided by the shared engine. Compression tables on the CPU (the GPU
runtime has them), attention, spin, type embedding, exclusions,
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

Optional [larger-frame CUDA training](LARGE_FRAME_TRAINING.md) supports an explicit
per-frame ceiling up to 256 under unchanged total tile/workspace/work budgets.
Default CPU/untiled and default tiled training retain 64 atoms; expanded tiled
runs use v9 restart identity and fail if the selected model/frame cannot fit.
