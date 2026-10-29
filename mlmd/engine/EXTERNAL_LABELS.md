# Return external labels without losing frame identity

This workflow connects reviewed model-deviation candidates to an explicit label
return file and then to existing training. `prepare-labels` and `import-labels`
perform file validation and transformation only. They do not load a potential,
run a calculator, submit jobs, select a device or infer units. A successful import
establishes file/context consistency; it does not establish scientific accuracy,
calculator convergence or authenticity of the returned values.

## 1. Select and inspect the exact frames

Use [model-deviation](MODEL_DEVIATION.md) and [select-dataset](DATASET_SELECTION.md)
to create a reviewed subset. Keep its receipt, then run:

```powershell
& $engine inspect-dataset D:/work/selected/dataset.jsonl D:/work/selected-inspection.json
```

Use the SHA256 of this immediate selected dataset, not the original parent file.
Selection preserves IDs but gives frames new consecutive indices. Keep labels
already present; the import below only fills missing values and accepts exact
matches. It cannot replace conflicting existing labels.

## 2. Prepare a label job bundle

Write `prepare.json`:

```json
{
  "schema": "omnimd-prepare-labels-v1",
  "dataset": "D:/work/selected/dataset.jsonl",
  "expected_source_sha256": "REPLACE_WITH_ACTUAL_64_HEX_SELECTED_DATASET_SHA256"
}
```

```powershell
& $engine prepare-labels D:/work/prepare.json D:/work/new-label-job
```

The source is read once, fully validated and hashed over exactly those consumed
bytes. The new bundle contains `request.json` (exact supplied bytes), `job.jsonl`
and `receipt.json` (written last). The receipt binds the input dataset, request,
job and each frame identity. Existing files/directories are never overwritten.

`job.jsonl` has one header, one frame record per selected frame and one terminal
count record. The header has schema `omnimd-label-job-v1`, identity schema
`omnimd-label-identity-v1`, `source_dataset_sha256`, and `context` containing the
ordered `type_map`, `units`, `virial_convention` and optional `parameter_layout`.
Each frame record contains:

```json
{
  "kind": "frame",
  "identity": {
    "index": 0,
    "id": "original-frame-id",
    "identity_sha256": "COPY_THE_PREPARED_HASH",
    "frame": {
      "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
      "species": ["O", "H"],
      "cell": null,
      "units": "EvAngstrom"
    }
  },
  "existing_labels": {"energy": null, "forces": null, "virial": null}
}
```

This is an illustrative record, not executable data or a valid hash. Copy the
actual generated `identity` objects exactly. Parameterized v2 records also carry
`identity.parameters`, including frame-default policy and atomic rows. The
existing Frame and parameter formats are reused; there is no atom remapping.

## 3. Adapt the external results explicitly

Use your chosen external process separately. The return adapter must establish
that its output belongs to these exact input coordinates, species order, cell
and conditions. Any conversion from another unit or tensor convention belongs
in that adapter and must be recorded in its method description. omniMD does not
perform or infer these conversions. Reordered atoms or relaxed geometries are
new inputs and cannot be attached to the prepared frame by copying its ID.

Create a UTF-8 JSONL file with a newline after every record:

1. Copy the prepared job header, change only `schema` to
   `omnimd-label-return-v1`, and add `provenance`:

   ```json
   {
     "provider": "Name and version of the external calculator or adapter",
     "method": "Exact calculation settings, label extraction and explicit conversions",
     "source_sha256": {
       "calculation-input": "ACTUAL_64_HEX_HASH",
       "calculation-output": "ACTUAL_64_HEX_HASH"
     }
   }
   ```

2. For every frame, copy its complete `identity`, remove `existing_labels`, and
   supply `labels` with finite total `energy` in eV and `forces` of shape `[N,3]`
   in eV/angstrom. Optional `virial` is `[3,3]`, all nine row-major entries in eV,
   using `W_ab = -dE/d epsilon_ab` for `r' = r(I+epsilon)`. It is not stress and
   is not divided by volume. Missing virial is `null` or omitted.
3. Finish with `{"kind":"completed","frames":N}` for the exact source count,
   then EOF. Returned frame order may differ. Every source ID must occur exactly
   once with its matching local index; unknown, duplicate or missing IDs reject.

The full header retains the exact source hash and scientific `context`. Only
`EvAngstrom` and `MinusDEpsilonRow` are accepted. Both E and F must be present
in every returned record, even if already present in the source. Existing labels
must match bit for bit; a missing returned virial preserves an existing virial.
New virial values fill only absent values. There is no tolerance, averaging,
energy-offset fitting or overwrite mode.

Provider/method and artifact hashes are declarations, not verified files or
signatures. An empty artifact-hash map is allowed but explicit provider/method
are required. A `model_default` parameter policy is preserved as a policy marker;
this file workflow cannot resolve hidden model defaults or prove that the
calculator used them. Record their model/context provenance explicitly. Never
interpret a preserved marker as resolved numerical conditions.

## 4. Bind and import the return file

Calculate the SHA256 of the final return-file bytes, then write `import.json`:

```json
{
  "schema": "omnimd-import-labels-v1",
  "dataset": "D:/work/selected/dataset.jsonl",
  "expected_source_sha256": "ACTUAL_64_HEX_SELECTED_DATASET_SHA256",
  "labels": "D:/work/returned.jsonl",
  "expected_labels_sha256": "ACTUAL_64_HEX_RETURN_FILE_SHA256"
}
```

```powershell
& $engine import-labels D:/work/import.json D:/work/new-labeled
& $engine inspect-dataset D:/work/new-labeled/dataset.jsonl D:/work/labeled-inspection.json
```

Dataset and labels paths, and the new output directory, must be absolute. The
parent output directory must already exist. Unknown fields, repeated JSON fields,
invalid shapes/nonfinite numbers, mismatching source/return hashes, changed frame
context, incomplete streams and conflicting existing labels fail before output
creation. Source validation includes its terminal count and trailing bytes.

The completed output bundle contains:

- `dataset.jsonl`: same header, frame IDs, indices, order, coordinates, species,
  cell and v2 parameters, with accepted labels added. Numeric float64 values
  round-trip exactly; JSON formatting and omitted optional fields may change.
- `request.json` and `labels.jsonl`: exact bytes parsed for this import.
- `receipt.json`: input/request/return/output hashes, source-order frame bindings,
  fields added per frame, and explicitly unverified calculator/upstream provenance.

The dataset completion record adds observed `input_dataset`, `label_return` and
`label_import_request` hashes. Inherited declarations are prefixed `upstream:`;
calculator declarations are prefixed `calculator:`. Those namespaces cannot
replace the observed hashes. All input validation precedes directory creation;
files are exclusively created and synced, with the receipt last. An IO/disk
failure can still leave an incomplete directory. Require successful exit, a
parseable completed receipt and matching recorded hashes before reuse. Do not
retry into or automatically delete an incomplete directory. There is no atomic
whole-directory or power-loss guarantee.

## 5. Retrain and assess

Point an existing readout/full-network training request at the new
`dataset.jsonl`, choose explicit validation IDs and a new output directory, and
select `"device":"cpu"` for this workflow:

```powershell
& $engine train-native D:/work/retrain.json
```

The native trainer still accepts dataset v1 only; preserving a v2 dataset does
not add parameterized native training. Parameterized v2 results can use the
existing CPU C API assessment/MD path described in [PARAMETERS.md](PARAMETERS.md).
A random frame split is not evidence of independent validation trajectories.
The commands never automatically retrain, run new dynamics or close an unattended
active-learning loop.

## Identity encoding and limits

The identity excludes labels, since labels are the intended changing fields.
It includes the source dataset SHA256 (which itself binds existing labels),
frame ID/index and all immutable scientific context. Version 1 SHA256 input is:

1. UTF-8 strings `omnimd-label-identity-v1`, lowercase source SHA256; index; ID;
   `EvAngstrom`; `MinusDEpsilonRow`; type-map length and each ordered symbol.
2. Atom count, then each ordered species string and its three position values.
   Cell-present flag, then nine row-major lattice values if present.
3. Layout-present flag; for fparam then aparam, field count and ordered name/unit
   strings. Parameters-present flag; if present, policy string (`explicit` or
   `model_default`), fparam vector, atomic-row count and each atomic vector.

Every integer/count/flag is unsigned 64-bit little-endian; flags are 0 or 1.
Every string is its UTF-8 byte length encoded that way followed by its bytes.
Each vector is its length followed by values. Numbers are exact IEEE-754 binary64
bits in little-endian order. `+0.0` and `-0.0` differ. There is no coordinate
rounding, periodic wrapping, tolerance, unit conversion or JSON canonicalization.
Return import recomputes both the original and echoed context hash and compares
them with the supplied identity hash. Copying a hash without preserving geometry
fails; copying a hash cannot prove the truth of the labels themselves.

Requests are at most 4 MiB; source input and normalized buffered content at most
256 MiB; returned label bytes at most 128 MiB; output JSONL at most 256 MiB;
records at most 16 MiB; source frames at most 10000. Existing dataset ID, geometry,
parameter shape and finite-value checks apply. JSON uses serde's normal nesting
limit. Each declared provenance map has at most 1000 unique keys (1..256 bytes),
provider at most 1024 bytes and method at most 4096 bytes. This bounded version
buffers source frames, return bytes and output; it is not an unlimited streaming
merge or a guarantee of low peak memory.

## Reproduce the focused acceptance

Reuse the original outputs of `validate_model_deviation.py`; no new external
scientific computation is necessary:

```powershell
python mlmd/scripts/validate_external_labels.py --engine $engine `
  --deviation-work D:/work/original-model-deviation --work D:/work/new-label-replay
```

The replay computes CPU disagreement on the original 12-frame fixture, selects
six frames, prepares identities, imports saved synthetic teacher predictions in
reverse order and fits the CPU readout with five training / one validation frame.
It independently recomputes identity hashes in Python, tests v2 parameter/order
preservation and verifies CLI rejections. V2 labels are an IO fixture only, not
validation of a parameter-dependent calculator. [Compact evidence](../evidence/external-labels.json)
records the actual outcome. Generated labels, weights and logs remain outside Git.
