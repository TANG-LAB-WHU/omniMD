# Parameterized CPU models in the engine

The engine can now propagate frame/atom conditions through DeepMD NPY import,
shared dataset v2, CPU `batch-eval`, and fixed-parameter `run` / `check`.
The official DeepMD runtime computes E/F/virial through FFI. Shared geometry
`Frame`, `Evaluation` and `Potential` types are unchanged. The native Rust model
subset does not gain parameterized-model support.

This route is CPU-only. It retains one model and makes one combined call per
frame, including initialization and every MD force update. `.pth` and stored
frame defaults work through this single-frame path; no engine multi-frame C
batch, automatic grouping or CUDA parameter acceptance is claimed.

## 1. Inspect the model and declare parameter meaning

Use standalone `omnimd-deepmd inspect` or engine `model-info`. C API metadata
reports frame dimension `df`, atom dimension `da`, and optional
`has_default_fparam`. The availability query does not expose default numbers.
Both model and runtime must support the requested policy. Names and units are
user-supplied provenance; they are not matched to model tensor names or converted.
Use the order and numeric convention from the model's training data.

A layout has one entry per dimension:

```json
{
  "fparam": [{"name": "condition", "unit": "REPLACE_WITH_TRAINING_UNIT"}],
  "aparam": [{"name": "atomic_condition", "unit": "REPLACE_WITH_TRAINING_UNIT"}]
}
```

Replace the illustrative names/units with actual meaning. Each category allows
0..1024 dimensions, with at least one dimension overall. Names must be unique
within a category; names/units are nonempty and at most 128 bytes. Values must
be finite and match exact declared dimensions. The engine also checks dimensions
against the loaded runtime model. A frame is bounded to 1,000,000 parameter
values; existing record and runtime buffer limits still apply.

## 2. Import explicit NPY parameter arrays

Add `parameter_layout` to the existing `import-deepmd` request:

```json
{
  "system": "/absolute/system",
  "sets": ["set.000", "set.001"],
  "units": "EvAngstrom",
  "pbc": "Periodic",
  "parameter_layout": {
    "fparam": [{"name": "condition", "unit": "REPLACE_WITH_TRAINING_UNIT"}],
    "aparam": [{"name": "atomic_condition", "unit": "REPLACE_WITH_TRAINING_UNIT"}]
  }
}
```

```sh
omnimd-mlmd import-deepmd import.json new-data.jsonl
omnimd-mlmd inspect-dataset /absolute/new-data.jsonl new-inspection.json
```

In every selected set, use C-order float64 `fparam.npy` with shape `[frames,df]`
and `aparam.npy` with shape `[frames,atoms*da]`. Atomic rows follow `type.raw` and
coordinate order. A positive dimension requires the corresponding file; a zero
dimension requires it to be absent. Frame counts must match `coord.npy` exactly.
No one-dimensional shorthand, constant broadcasting, 3-D atomic layout, raw
parameter files, mixed atom types or inferred defaults are accepted. Coordinate,
cell and optional label rules remain those of the existing importer.

This opt-in emits `omnimd-dataset-jsonl-v2`; omitting the layout retains v1 and
continues to reject parameter arrays. XYZ/extxyz import remains v1 only. Failed
imports may leave a partial stream without `completed`; it is not a dataset.

## 3. Understand versioned records and default policy

A v2 header is the v1 header with schema `omnimd-dataset-jsonl-v2` and a required
`parameter_layout`. Every frame has a required `parameters` object, for example
with two atoms and `df=da=1`:

```json
"parameters": {
  "frame_parameter_policy": "explicit",
  "fparam": [0.3],
  "aparam": [[0.1], [0.7]]
}
```

These are shape examples, not physical settings. `explicit` is the default if
policy is omitted; missing/empty values never select model defaults. `fparam`
and `aparam` arrays are required. For `da=0`, the outer `aparam` array is empty;
for `da>0`, each atom has exactly `da` entries. Atomic conditions are never
broadcast, defaulted or sorted by species. Reordering coordinates externally
requires applying the identical permutation to species and atomic rows.

To opt into model-stored frame defaults, set policy to `model_default` and use
`fparam:[]`. Declared `df` must remain positive, runtime availability must be
`Some(true)`, and all atomic rows must still be explicit. False or unavailable
metadata rejects. The default request is passed to the official API as NULL
frame parameters, not manufactured zeros. The model file hash binds the stored
defaults, whose numeric values are not exposed by the query. This mode was
verified with the official PyTorch CPU fixture; DeepMD 3.1.3 TF defaults reject.

Inspection and `select-dataset` retain v2 layout, values and policy. Selection
preserves rows and IDs and only reindexes frames. Parameterized native training,
minimization and `model-deviation` currently reject instead of discarding context.
Inspection's E/F label-coverage counts do not establish trainer compatibility.

## 4. Evaluate the dataset on CPU

The existing request shape is sufficient:

```json
{
  "options": {
    "library": "/opt/deepmd-kit/lib/libdeepmd_c.so",
    "model": "/absolute/parameter-model.pb",
    "device": "cpu",
    "type_map": ["O", "H"],
    "units": "EvAngstrom"
  },
  "dataset": "/absolute/new-data.jsonl"
}
```

```sh
omnimd-mlmd batch-eval evaluation.json new-predictions.jsonl
```

For v2 inputs, prediction schema v2 retains parameter values/policy in each
prediction and layout in the dataset header. Model and dataset file fingerprints
are recorded as in ordinary batch assessment. Each result is validated before
emission; an error aborts without a terminal `completed`. Earlier successful
frames may be present in a partial stream. No fallback policy or backend retry
occurs. Ordinary parameter-free datasets retain the v1 prediction shape.

## 5. Run MD with an immutable parameter context

A `dataset_frame` reference to v2 automatically binds that record's layout,
values/policy and atom sequence for the entire new trajectory. It cannot be
combined with a `fixed_parameters` override. Supply normal masses, velocities or
initialization, timestep and NVE/CSVR settings; see [engine usage](README.md).

For a directly supplied `frame` (or a parameter-free v1 frame reference), add:

```json
"fixed_parameters": {
  "schema": "omnimd-fixed-parameters-v1",
  "layout": {
    "fparam": [{"name": "condition", "unit": "REPLACE_WITH_TRAINING_UNIT"}],
    "aparam": [{"name": "atomic_condition", "unit": "REPLACE_WITH_TRAINING_UNIT"}]
  },
  "values": {
    "frame_parameter_policy": "explicit",
    "fparam": [0.3],
    "aparam": [[0.1], [0.7]]
  },
  "atom_species": ["O", "H"]
}
```

Atom indices are immutable in this runner. Ordered species/count checks detect
incompatible geometry, but cannot identify external swaps between equal species;
users must preserve index identity. Every integration evaluation uses the same
fixed context. No thermostat-temperature coupling, coordinate-dependent
condition update, time schedule or parameter dynamics is implemented. A field
named temperature does not automatically follow the kinetic or thermostat
temperature. Forces and virial are evaluated at fixed conditions.

Parameterized trajectories use schema v3 and include the complete fixed context
in their header. Parameter-free trajectory headers remain v2. Checkpoint v3
stores the same context plus the existing model hash, coordinates, velocities,
ensemble and RNG state. Resume restores it; an optional supplied context must
match every field exactly. Changing values, policy, layout order, names, units
or atom sequence rejects. Old v1/v2 checkpoints remain readable for parameter-
free runs and cannot acquire parameters on restart. Generic parameter-free
`Checkpoint::restore` refuses a v3 context; library users must explicitly pass
the expected context to `restore_with_parameters`.

The TF fixture demonstrated bitwise exact same-environment replay. The PT
fixture showed first-call versus repeated-identical-frame roundoff in the standalone
C API, so its MD restart has numerical parity, not bitwise identity. Parameter
context and geometry restoration remain exact. No bitwise guarantee is made
across runtime versions or hardware. A model or condition
change starts a new trajectory. Stored-default MD remains bound to the same
model bytes, including when the numeric defaults cannot be inspected.

## 6. Reproduce the recorded CPU acceptance

Use the already saved official TF/PT fixtures and C's reference results described
in [backend validation](../deepmd/BACKEND_VALIDATION.md). Do not substitute private
research models or rebuild references merely to match engine output:

```sh
python mlmd/scripts/validate_engine_parameters.py \
  --engine /path/to/omnimd-mlmd --capi /path/to/omnimd-deepmd \
  --references /path/to/phase3-defaults --work /new/output/directory
```

The replay imports TF NPY parameters, compares TF and PT explicit/default batch
results with saved C API and official Python E/F/V, permutes same-species atoms
and atomic rows together, and checks TF/PT four-step fixed-parameter NVE through
both dataset and direct requests. Every moving frame is independently evaluated
by the standalone C API. Restart at step 2 must preserve the exact parameter context; TF snapshots and
checkpoint match exactly, while PT uses explicit numerical limits and records
its standalone first-call roundoff diagnostic. Changed parameters reject. Native/CUDA/ensemble
parameter routes and unsupported TF defaults also reject. The compact
[engine receipt](../evidence/engine-parameters.json) records numeric results and
fingerprints. These are compatibility and software checks, not physical accuracy,
long-time dynamics validation or GPU/performance claims.
