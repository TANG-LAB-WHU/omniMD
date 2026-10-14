# Strict frozen-graph export

`tools/export_frozen.py INPUT.pb OUTPUT.json --units EvAngstrom` converts an
existing binary TensorFlow GraphDef into native format v1. The output must be
new. It needs an already installed TensorFlow/protobuf/NumPy environment for
GraphDef and tensor decoding. It never imports the supplied graph into a
TensorFlow session, executes it, or calls DeepMD model deserialization. Native
Rust inference still has no TensorFlow dependency.

## Supported profiles and validation

This first exporter supports **two exact audited graph profiles with replaceable
parameters**, not arbitrary DeepMD graphs. Both use graph format `1.1`, float64,
two species, separate type-pair embeddings, and scalar type-specific fitting.
DeepMD package version 3.1.3 is distinct from graph format version 1.1.

| Profile | `sel` | Embedding widths | Axis | Fitting widths |
| --- | --- | --- | --- | --- |
| `public-deeppot-1.1-v1` | 46, 92 | 1, 2, 4, 8 | 4 | 32, 6, 6, 6, 1 |
| `synthetic-tf-3.1.3-v1` | 32, 32 | 1, 2, 4, 4 | 2 | 8, 5, 5, 1 |

The public profile derives from DeepMD v3.1.3's public inference-test
[`deeppot.pbtxt`](https://github.com/deepmodeling/deepmd-kit/blob/v3.1.3/source/tests/infer/deeppot.pbtxt).
A's existing binary conversion is used unchanged. The modern profile comes from
our original fixed-seed, untrained `tools/make_deepmd_fixture.py` model.

The readable JSONL profiles in `tools/frozen_profiles/` contain all 1202 / 1291
ancestors of `o_energy`, `o_force`, `o_virial`, and all model/descriptor/fitting metadata.
Matching pins every node name, operation, ordered input (including output ports
and control edges), attribute, structural constant, tensor shape and dtype.
It therefore checks descriptor slicing, type routing, contraction transpose
flags, normalization divisors, activations, applied residual timesteps, output
arithmetic, and derivative provenance. Nodes are compared by name after sorting;
protobuf node ordering does not matter. Node names alone never authorize import.
Function libraries are rejected. Unconnected training metadata is not required.

Only explicit parameter payloads may vary: finite float64 matrices, biases,
applied timesteps, normalization statistics, and the two-entry ordered type map.
Normalization must repeat exactly across slots, with zero angular means and
positive isotropic angular scales. No averaging or inferred replacement occurs.
C API type counts and extra-input dimensions are pinned. The cutoff metadata
must equal the effective environment-op cutoff. The two cutoff attributes may vary with `0 <= rcut_smth < rcut`; their decoded
protobuf float values are preserved as effective geometry. Fixed selection
sizes and all connected scalar normalizers must still match the chosen profile.
Equivalent graphs with renamed nodes, different optimization, widths, selection,
activations, or residual topology are intentionally rejected. Non-unit output
scale, alternate energy heads, hybrid/compressed/spin/type-embedded models, and
unknown versions are unsupported. No training JSON is guessed or synthesized.

The graph does not attest physical units: `--units EvAngstrom` is an explicit
caller assertion. Source files are limited to 256 MiB and decoded tensors to
8 million elements. An unsupported or malformed graph yields a nonzero exit
without writing a native model.

## Audited forward mapping

`ProdEnvMatA` consumes positions, types, natoms, cell, mesh, and constant average
and standard deviation arrays. `sel_r` is zero. Normalized rows, including
zero-padded slots, are routed by central and neighbor types. Each radial column
feeds its own embedding network. The public embedding layers use tanh plus
concatenated-input residuals; modern layers use tanh, connected timesteps, and
concat2/concat2/identity residuals.

For each central type the graph sums `R_t^T G_t`, divides by `sum(sel)`, and forms
`A^T A[:,:,:axis]`. The public profile's actual multiplier is exactly `1/138`;
the modern graph divides by 64. Both match the native fixed-capacity convention.
Flattening varies the axis index fastest. Fitting hidden layers use tanh; all
but the first hidden layer use their connected identity residual and timestep.
The final fitting layer is affine.

The public graph's energy offsets are already in the final affine biases;
there is no separate offset addition on its output path. The modern graph adds
`fitting_attr/t_bias_atom_e`, applies unit `t_out_std`, then adds
`model_attr/model_attr/t_out_bias` through a type gather. The native model keeps
final affine biases in place and sums the two separate offset arrays once.
The old official `.pb` serialization roundtrip lost the latter array in this
synthetic case; this exporter retains it. Standard nonnegative species mapping
is used; virtual/negative species are not native inputs.

`make_frozen_profiles.py PUBLIC.pb SYNTHETIC.pb` reproduces the profiles only
from the exact two recorded binary SHA-256 values. It refuses any other source;
it is not an automatic way to approve a new graph. Adding a profile requires a
reviewed mapping and new actual cross-backend evidence.

## Acceptance and replay

Run with existing inputs, the existing local trusted image, A's Linux C API
executable, and a built host native executable:

```sh
python mlmd/native/tools/replay_frozen.py --image EXISTING_IMAGE \
  --pb model.pb --frames frames.jsonl --units EvAngstrom \
  --capi-exe /host/path/omnimd-deepmd \
  --native-exe /host/path/omnimd-native-eval --output-dir /new/output/path
```

The harness uses `--pull never --network none`, mounts inputs read-only, and
writes conversion, both E/F/virial outputs, hashes and report to the new output
directory. It does not download inputs or install dependencies. Reuse the public
model and frames from A's existing validation, or generate the synthetic inputs
with the prior [conversion replay](CONVERSION.md).

`python tools/check_frozen.py synthetic.pb native-from-dp.json` checks both
energy-bias sites, byte-equivalent model content to the complete interchange
conversion, effective geometry, and seven targeted rejection cases. Run it in the
same existing TensorFlow environment. The checks do not rerun the Rust finite
difference suite because no Rust inference code changed in this milestone.

The actual DeepMD 3.1.3 C API replay used 1e-12 absolute plus 1e-12 relative
tolerances on every component and passed:

| Model / frames | Max energy error (eV) | Max force error (eV/A) | Max virial error (eV) |
| --- | ---: | ---: | ---: |
| Public / 4 | 2.2737367544323206e-13 | 4.218847493575595e-15 | 6.661338147750939e-15 |
| Synthetic / 3 | 4.440892098500626e-16 | 5.0306980803327406e-17 | 3.469446951953614e-17 |

Public frames cover nonperiodic, orthogonal, triclinic and atom-reordered cases.
Synthetic frames cover nonperiodic, triclinic and wrapped coordinates. The
synthetic output JSON hash is identical to the prior complete `.dp` conversion.
[The committed report](fixtures/frozen-graph-report.json) records source/image/
executable/output hashes and checks. Generated `.pb` binaries and weights remain
local ignored artifacts. These results establish numerical compatibility for
these profiles and frames, not chemical accuracy, general DeepMD compatibility,
or a speed advantage.
