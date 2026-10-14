# DeepMD 3.1.3 weight conversion

`tools/convert_deepmd.py` imports **DeepMD-kit 3.1.3** interchange `.dp` (HDF5)
or `.yaml`/`.yml` data. It produces a new native v1 JSON file without executing a
model graph, importing DeepMD/TensorFlow, loading pickle, or installing anything.
The Python conversion tool needs existing `h5py` for `.dp` or existing `PyYAML`
for YAML; native Rust inference does not acquire either dependency.

```sh
python mlmd/native/tools/convert_deepmd.py model.dp native.json \
  --units EvAngstrom --geometry tensorflow
```

The output path must not already exist. Both switches are mandatory: DeepMD
serialization does not reliably attest the training units, and its backends do
not encode geometry identically. Use `--geometry dpmodel` to preserve the stored
float64 cutoff/smoothing radii. Use `--geometry tensorflow` to reproduce a
TensorFlow frozen model's effective radii: TF `AttrValue.float` stores those
operator attributes as IEEE float32, even with float64 weights and coordinates.
The converter rounds only these two radii, then stores/evaluates them as float64.
This is not a general downcast of model weights or native arithmetic.

For example, `rcut_smth=2.1` becomes `2.0999999046325684` in a TF environment
operator. Ignoring this caused a measured periodic virial discrepancy of
3.82e-9 eV in the new fixture; the explicit conversion resolves it to roundoff.
Do not choose a mode merely to obtain a smaller comparison error: it must match
the intended source backend. Source/output SHA-256 values and the selected mode
are printed by the converter. Native JSON remains schema v1.

## Exact supported input subset

- Envelope: `software=deepmd-kit`, `version=3.1.3`; no claimed support for other
  producer versions. The `model` tree is authoritative, not `model_def_script`.
- Serialized classes/versions: Model 2, Descriptor 2, Fitting 4,
  NetworkCollection 1, EmbeddingNetwork 2, FittingNetwork 1, Layer 2, ndarray 1.
- A standard energy model with a single `se_e2_a` descriptor; explicit float64
  arrays; scalar, species-specific energy fitting; ordered chemical type maps.
- One-sided or two-sided embeddings. For two-sided collections, DeepMD's flat
  index is `central + ntypes*neighbor`; the importer reorders this into native
  `[central][neighbor]`. Per-layer activation, residual, and timestep are retained.
- Uniform per-central-type normalization across all slots, zero angular means,
  equal positive angular standard deviations; quintic switching and zero
  environment protection. Stored means are retained even when `set_davg_zero`
  appears in training metadata.
- `fitting.bias_atom_e` and `model.out_bias` are added exactly once. `out_std`
  must be all ones because v3.1.3 dpmodel and TensorFlow interpret it differently.
- No nonempty exclusions, spin, type embedding, hybrid/attention/compression,
  frame/atomic/case parameters, mixed/shared fitting, total-energy-zero
  transform, or meaningful `atom_ener` vacuum subtraction. Unknown fields,
  classes, schema versions, malformed shapes, unsupported activations, and
  nonfinite parameter values fail. Known training-only metadata such as
  `rcond`, `preset_out_bias`, layer names, and trainability do not replace stored
  inference arrays.
- HDF5 reads ordinary internal, uncompressed float64 datasets; external/soft
  links, virtual/external datasets, object dtypes, and other layouts fail. YAML
  uses SafeLoader with duplicate-key rejection; executable tags are not accepted.
  Input size is capped at 256 MiB and each HDF5 dataset at 8 million values.

This interchange converter does **not** consume `.pb`, `.pth`, checkpoint
folders, or pickle. A separate [frozen-graph exporter](FROZEN.md) now supports
two audited exact graph profiles, including the public `deeppot.pb` fixture
without `train_attr/training_script`. It verifies the actual output graph and
does not reconstruct a training configuration.

An exploratory official TF `.pb` -> serialized-dictionary roundtrip also lost
nonzero **model-level** `out_bias` in this synthetic case, shifting total energy
by 0.02 eV while forces/virial stayed equal. The accepted path starts from the
complete original `.dp`/YAML arrays and freezes the same complete model for the
C API comparison. Do not assume an external exporter is lossless merely because
its output passes structural validation.

## Replay the actual C API/native comparison

First build the native evaluator with Cargo. Supply an **already installed,
trusted local** DeepMD 3.1.3 + TensorFlow Docker image, A's already built Linux
`omnimd-deepmd` evaluator, and the host native executable:

```sh
python mlmd/native/tools/replay_deepmd.py \
  --image YOUR_EXISTING_DEEPMD_IMAGE \
  --capi-exe /path/on/host/to/linux/omnimd-deepmd \
  --native-exe /path/on/host/to/omnimd-native-eval \
  --output-dir /new/output/directory
```

Windows host executable paths are supported. The C API executable runs in the
existing Linux image, mounted read-only. Containers use `--pull never` and
`--network none`; source is read-only and only the new output directory is
writable. No private input is used. The harness creates an original untrained
model with seeds 20261014 and 20261015, nonzero normalization and both energy
bias locations, two distinct element types and type-pair networks, and residual
timesteps. It writes official `.dp`/YAML, freezes the same weights to `.pb`, checks
both encodings convert identically, and evaluates nonperiodic, triclinic-periodic,
and periodically wrapped versions of the same four-atom frame.

`tools/check_conversion.py` performs the focused encoding, pair-order, bias,
geometry, and eight rejection checks. `tools/compare_evaluations.py` compares all
E/F/virial components at absolute+relative tolerances of 1e-12. No finite-difference
or broad regression suite is rerun by this harness. It writes source models,
converted weights, frames, both evaluation streams, and `report.json` including
hashes. Generated binary models are local artifacts, not committed dependencies.
The committed `fixtures/deepmd-3.1.3-report.json` records the successful run.

Measured maximum absolute differences over three frames:

| Energy (eV) | Force (eV/angstrom) | Virial (eV) |
| ---: | ---: | ---: |
| 4.440892098500626e-16 | 5.0306980803327406e-17 | 3.469446951953614e-17 |

This is **synthetic-model numerical compatibility**, not a trained potential,
chemical-accuracy result, full DeepMD compatibility claim, or speed benchmark.
The separate frozen-graph exporter now retains those complete offsets and
effective radii; see [FROZEN.md](FROZEN.md) for its narrower graph-profile scope.
