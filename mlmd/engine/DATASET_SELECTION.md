# Extract a reproducible dataset subset

`omnimd-mlmd select-dataset REQUEST.json NEW_DIRECTORY` closes the gap between
reviewing candidate IDs and using an unchanged subset in assessment or training.
It needs no model, runtime or GPU. It consumes the existing shared JSONL format.

## 1. Bind the selection to a specific input

Obtain the dataset SHA256 from `inspect-dataset`, or from the terminal `completed`
record of [model-deviation](MODEL_DEVIATION.md). Review the desired IDs. Write:

```json
{
  "schema": "omnimd-select-dataset-v1",
  "dataset": "D:/work/data.jsonl",
  "expected_source_sha256": "PUT_THE_ACTUAL_64_HEX_DATASET_SHA256_HERE",
  "ids": ["frame-17", "frame-4"]
}
```

The placeholder must be replaced with an actual SHA256. The request accepts
1..10000 unique nonempty IDs, each at most 256 bytes, and at most 4 MiB of JSON.
IDs are exact opaque strings; there is no trimming, normalization or regex.
Their order is a membership list: output always follows source order. Unknown
fields, missing IDs and a mismatching source hash fail. Zero selected frames
are rejected because the shared dataset requires a positive completed count.

## 2. Create a new bundle

```powershell
& $engine select-dataset D:/work/select.request.json D:/work/new-selection
```

Both dataset and output-directory paths must be absolute. The output directory
must not exist and its parent must exist. The command reads and validates the
entire input, including unselected records and the terminal count/trailing bytes,
before creating the output directory. SHA256 covers the exact bytes consumed in
that read, not a separate reopening of the source.

Selected JSONL is buffered with a 256 MiB serialized bound, plus at most one
current input record and the reader's source-wide ID set. This bounded first
version does not spool unlimited datasets to disk. Source ID uniqueness memory
still grows with the number of input frames.

The new directory contains:

- `dataset.jsonl`: the standard shared dataset. IDs, positions, species, cell,
  units, virial convention and optional label values are preserved. Indices are
  contiguous from zero. JSON formatting may change, including explicit nulls;
  numerical float64 values round-trip exactly.
- `request.json`: the exact request bytes used for selection.
- `receipt.json`: written last, with status `completed`, request/input/output
  hashes, requested IDs, source header and the source/output index mapping.

`source_index` always refers to the immediate input dataset. Repeated selection
preserves IDs and records each new mapping rather than pretending a local index
is the root trajectory index. The output dataset declares the observed
`input_dataset` and `selection_request` hashes, with inherited hashes under the
injective `upstream:` prefix. Inherited source files are not verified. Preserve
the receipt chain for ancestry.

## 3. Verify completion before reuse

A successful exit plus a parseable `receipt.json` with matching dataset/request
file hashes marks a complete bundle. Files are exclusively created and synced;
existing outputs are never replaced. A disk/IO failure may leave an incomplete
new directory. Do not use it or retry into it. There is no atomic-directory or
power-loss guarantee. The command performs no deletion or cleanup of outputs.

```powershell
& $engine inspect-dataset D:/work/new-selection/dataset.jsonl D:/work/subset-inspection.json
```

The selected dataset works directly with `batch-eval`, `dataset_frame` in MD or
minimization, and native training. Always use the new index within the selected
dataset, or resolve it from the receipt's preserved ID. Existing labels and their
absence are retained; this command never evaluates a model to manufacture labels.
Training still requires E and F for every selected record. Unlabeled candidates
require an external labeling process before training. The CLI does not launch
that process or an automatic active-learning loop.

## 4. Reproduce the complete CPU workflow

First run `validate_model_deviation.py` as documented in its guide, then:

```powershell
python mlmd/scripts/validate_dataset_selection.py --engine $engine `
  --deviation-work D:/work/model-deviation-run --work D:/work/new-selection-run
```

The original synthetic replay selects six of twelve candidate frames, verifies
scientific content and hashes, inspects and assesses the subset, then selects
its three already-labeled records. It invokes the integrated CPU readout trainer
with two training and one held-out frame. Missing-label training, wrong input
hash and existing output are refused. The [compact receipt](../evidence/dataset-selection.json)
records the mappings and result. This is a workflow acceptance test; it does not
establish chemical accuracy or a statistically useful training set.
