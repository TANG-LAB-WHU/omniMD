# Native CPU se_e2_a inference

`omnimd-native` implements `omnimd_ml_contract::Potential` with float64 Rust
arithmetic. It evaluates energy, analytic forces, and strain-gradient virial in
one call. The only dependencies are the existing shared contract, serde and
serde_json. No DeepMD runtime, BLAS, Python, GPU, or training is used by inference.

## Run

From the repository root, with Rust and its platform linker available:

```sh
cargo test --locked --manifest-path mlmd/native/Cargo.toml
cargo run --locked --release --manifest-path mlmd/native/Cargo.toml \
  --bin omnimd-native-eval -- mlmd/native/fixtures/synthetic-v1.json \
  < mlmd/native/fixtures/frame.jsonl
```

PowerShell input can be piped with `Get-Content mlmd/native/fixtures/frame.jsonl`.
The CLI reads one shared-contract `Frame` per nonblank input line and writes one
`Evaluation` JSON line. A failed frame produces a diagnostic on stderr and a
nonzero exit code, with no partial result for that frame. Earlier successful
frames remain in the output stream. Positions/cell use angstrom; energy/virial
use eV; force uses eV/angstrom. Cell rows are lattice vectors. `null` cell means
nonperiodic; a supplied cell is periodic in all three directions.

```rust
use omnimd_ml_contract::Potential;
use omnimd_native::NativePotential;

let mut potential = NativePotential::load("model.json")?;
let evaluation = potential.eval(&frame)?;
```

Line B owns only `mlmd/native/**`, so this crate currently declares its own
workspace and depends on `../contract`. For workspace integration, remove the
empty `[workspace]` section in this crate and add `native` to the parent
`mlmd/Cargo.toml` members. No shared interface changes are required.

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
No trainer, optimizer, MD integrator, GPU kernel, compression table, attention,
spin, type embedding,
`se_e3`, hybrid descriptor, exclusions, frame/atomic parameters, environment
protection, or vacuum-subtraction feature is implemented. Such architectures
must not be relabeled as this format; unknown fields and unsupported enums fail.

This first implementation rebuilds direct neighbor lists and temporary network
buffers each evaluation. Neighbor enumeration is quadratic in atom count before
image expansion; no production scaling or speedup claim is made. Explicit
resource limits are 256 types, 65536 total neighbor slots, 128 layers per network,
65536 outputs per layer, and 2,000,000 candidate images per central atom. Image
ranges beyond representable bounds, ill-conditioned cells, distances <= 1e-12
angstrom, nonfinite parameters/intermediates/outputs, and malformed shapes fail.

See `THIRD_PARTY_NOTICES.md` for mathematical references and licenses.

The public frozen-model replay now passes four frames with maximum E/F/virial
errors of 2.27e-13 / 4.22e-15 / 6.66e-15. Exact graph-profile limits and replay
evidence are in [FROZEN.md](FROZEN.md). A bounded next training milestone is
specified in [TRAINING_NEXT.md](TRAINING_NEXT.md); training is not implemented.
