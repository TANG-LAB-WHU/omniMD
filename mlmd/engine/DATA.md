# Dataset input, model queries and batch assessment

The CPU CLI imports an explicitly selected **DeepMD standard NPY subset** into
one normalized JSONL format, evaluates it through either backend, and can use a
selected frame to initialize MD. It does not train, choose splits, shuffle,
fit energy offsets or infer physical units.

## Commands

    omnimd-mlmd capabilities
    omnimd-mlmd model-info model.json new-model-info.json
    omnimd-mlmd import-deepmd import.json new-dataset.jsonl
    omnimd-mlmd batch-eval evaluate.json new-predictions.jsonl

The capabilities command prints static JSON without loading a model.
The model-info request takes {"options": ...} with the same native/C API options
as MD, loads it once and writes validated type map, cutoff, backend, precision,
units, model SHA256 and C API library SHA256 if applicable. It does not infer a
C API model's architecture, training units or internal weight precision.

Import request (use real absolute paths; escape Windows backslashes):

    {
      "system": "/absolute/deepmd-system",
      "sets": ["set.000", "set.002"],
      "units": "EvAngstrom",
      "pbc": "Periodic"
    }

The set list is required, ordered and unique. No directory is implicitly chosen
as training, testing or validation. IDs retain the set name and original frame
index, with consecutive zero-based dataset indices.

## Supported input boundary

Require type.raw and type_map.raw. IDs are nonnegative indices into that
ordered map; names must be distinct chemical symbols. Preserve atom order and
map model types by symbol, so dataset and model type orders need not match.

| File within each selected set | C-order float64 shape | Required |
| --- | --- | --- |
| coord.npy | (frames, 3 * atoms) | yes |
| box.npy | (frames, 9) | only for Periodic |
| energy.npy | (frames,) or (frames, 1) | no |
| force.npy | (frames, 3 * atoms) | no |
| virial.npy | (frames, 9) | no |

All present arrays must have the same positive frame count. Atom count is
positive and fixed within the imported system. Labels may differ in availability
between sets. Coordinates and box vectors are angstrom; energy is eV, force
eV/angstrom, virial eV. Boxes and virials flatten by rows, without transposition,
symmetrization or volume normalization. Virial is W_ab = -dE/d epsilon_ab with
row coordinates r' = r (I + epsilon); deform both positions and lattice for
periodic strain derivatives. It is not stress or pressure.

Periodic requires no nopbc marker and a valid right-handed, nonsingular box for
every frame. NonPeriodic requires the nopbc file and forbids box.npy; its
normalized frames contain cell: null. Only all-axis PBC is supported.

NPY versions 1.0, 2.0 and 3.0 and either float64 byte order are accepted.
Float32, integer/object/structured arrays, pickle, Fortran order, mixed types,
virtual atoms, active extra inputs and unknown NPY/raw fields are rejected.
Nonfinite values, malformed shapes, truncated/trailing NPY bytes and unknown
JSON fields fail. No automatic type or unit conversion occurs. Ordinary
unrelated metadata files are not interpreted.

Limits: 256 MiB per source file, 64 KiB NPY header with nesting at most four,
16 MiB per JSONL record. The importer retains one set's decoded arrays plus a
temporary input buffer (up to about 1.5 GiB for five maximum-size arrays).
Dataset readers stream frames but retain IDs to detect duplicates. This first
implementation is not a constant-memory, arbitrarily large dataset engine.
Files must resolve within the selected system, including symlink targets.

The parser is npyz 0.9.1. The checked-in lockfile retains pest 2.7.15 for the
isolated Windows GNU toolchain; newer parser dependencies introduced an
unavailable C/assembly dependency. Build with --locked. No Python is needed.

## Normalized JSONL v1

Three record kinds, one JSON object per newline-terminated line:

    {"kind":"header","header":{"schema":"omnimd-dataset-jsonl-v1","type_map":["H"],"units":"EvAngstrom","virial_convention":"MinusDEpsilonRow","source":"synthetic example","selected_sets":["set.000"]}}
    {"kind":"frame","data":{"index":0,"id":"set.000/0","frame":{"positions":[[0,0,0]],"species":["H"],"cell":null,"units":"EvAngstrom"},"labels":{"energy":0.0,"forces":[[0,0,0]],"virial":null}}}
    {"kind":"completed","frames":1,"source_sha256":{}}

The frame is the shared Frame without a second geometry representation.
Independent optional labels use the shared Evaluation field semantics;
null/omitted means absent, not zero. The header fixes their units and tensor
convention. The importer records every consumed source file's SHA256 in the
completed record. A manually authored dataset may have an empty source map but
must still give honest provenance; batch evaluation fingerprints actual JSONL bytes.

Exactly one header and one final completed record are required. Frame IDs must
be nonempty and unique (maximum 256 bytes), indices consecutive, and completed
count exact and positive. A reader must consume the whole stream before treating
it as successful. Duplicate/extra/trailing records and truncated files fail.

This engine-owned version is also the agreed input for B's bounded native
training work. The shared model contract remains unchanged. A training consumer
must explicitly enforce its label requirements and report selected frame IDs;
importing data does not authorize an automatic split or training run.

## Evaluation and statistics

    {
      "options": {"native_model": "/absolute/native-model.json"},
      "dataset": "/absolute/new-dataset.jsonl"
    }

Each prediction contains the original index/ID, atom count, combined E/F/virial
evaluation and available labels. The header records model/dataset SHA256,
validated model metadata and the input dataset header. One model is loaded per
request; every frame is evaluated once. Failed/truncated runs have no completed
record, even if earlier predictions were written. All outputs use create-new
semantics; existing files are never replaced.

The completed record gives label coverage plus count, MAE, RMSE and maximum
absolute residual for total energy, energy/atom, force components and all nine
virial components. Energy metrics weight labeled frames equally; energy/atom
divides each frame residual by its own atom count. F/V pool scalar components,
not vector norms or an average of per-frame RMSEs. Missing labels contribute no
samples; zero-count statistics are null. No reference-energy subtraction,
automatic offset fit or physical accuracy claim is made.

## Dataset frames for MD

Replace the MD request's frame with:

    "dataset_frame": {"dataset": "/absolute/new-dataset.jsonl", "index": 0}

The complete dataset is validated before initialization. Supply explicit masses
and either velocities or seeded initialization exactly as before. This input
is exclusive with frame and restart. Labels are not used as forces or energy.
The trajectory stores the selected dataset reference; checkpoint geometry is
self-contained.

## Replay and sources

The validate_dataset.py script in mlmd/scripts documents arguments through --help.
It generates actual NumPy input, checks imported geometry and hashes, computes
error statistics independently, tests incomplete input, and compares a
20-step/0.2 fs native/C API trajectory. Its offset labels are synthetic software
checks, not physical reference data.

Input conventions follow the official
[DeepMD 3.1.3 system format](https://docs.deepmodeling.com/projects/deepmd/en/v3.1.3/data/system.html)
and [NumPy NPY specification](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html).
The bounded subset above is intentionally narrower than those general formats.
## XYZ and extended XYZ

    omnimd-mlmd import-xyz xyz-import.json new-dataset.jsonl

    {
      "input": "/absolute/structures.extxyz",
      "format": "Extxyz",
      "units": "EvAngstrom",
      "pbc": "Periodic",
      "type_map": ["O", "H"]
    }

The format, units, PBC policy and allowed chemical-symbol map are explicit.
Multiple frames and varying atom counts are accepted; atom and frame order are
preserved. IDs are xyz/0, xyz/1, etc. The normalized schema is unchanged and the
source bytes are fingerprinted. This importer performs no split or training.

Plain Xyz requires exactly symbol x y z on each atom line, with an arbitrary
comment line and no inferred labels. For Periodic plain XYZ additionally supply
cell as three row vectors in the request; for NonPeriodic omit cell.
Structured metadata in plain mode is rejected instead of being silently used
as a second format.

The Extxyz mode supports the common standard profile:

- Required Properties triplets species:S:1 and pos:R:3, in any order.
- Optional force:R:3 or forces:R:3, with exact column counts.
- Optional energy scalar and virial containing nine numbers.
- Lattice containing three concatenated lattice vectors; required for Periodic
  and forbidden for NonPeriodic. Request cell overrides are forbidden.
- Optional pbc with three T/F (or standard true/false) values must agree with
  the required request policy. Partial PBC is unsupported.
- Optional units must be EvAngstrom. Nonphysical config_type, comment,
  model_sha256 and minimization_status metadata are accepted but not labels.

Metadata keys are unquoted key=value pairs; whitespace around equals is allowed.
Values may be bare or enclosed in double/single quotes or flat brackets/braces.
Nested/JSON arrays, arbitrary per-atom properties, numeric atomic IDs, stress
conversion, six-component tensors, constraints, extra input features and unknown
physical metadata are rejected. This is a bounded standard profile, not every
extension of every XYZ producer. Missing Properties never triggers a fallback.

Lattice's nine numbers are concatenated a,b,c vectors and become contract rows.
A standard extxyz virial's nine numbers use Fortran tensor order and are
converted once into contract rows. An asymmetric tensor fixture tests this
distinction. No symmetrization, sign change or volume division is applied.
Energy/forces/virial units and physical convention are the caller's explicit
EvAngstrom assertion. Absent labels stay null; no value is inferred from a model.

Atom records accept finite decimal/E/D-exponent numbers. Input is UTF-8,
limited to 256 MiB, 1 MiB per line and one million atoms per frame; the existing
16 MiB dataset-record bound also applies. Atom counts/column shapes, symbols,
cells, conflicting units/PBC, duplicate keys and truncated frames fail.
Only trailing blank lines are allowed.

The geometry-only extxyz exported by minimization roundtrips through this importer.
It has no energy/force reference labels; see [MINIMIZATION.md](MINIMIZATION.md).

Conventions were checked against the
[libAtoms specification](https://github.com/libAtoms/extxyz) and the official
[ASE extxyz reader/writer](https://gitlab.com/ase/ase/-/raw/master/ase/io/extxyz.py),
including its separate lattice transpose and Fortran-order tensor handling.
