# CPU model deviation

`omnimd-mlmd model-deviation REQUEST.json NEW_DEVIATIONS.jsonl` evaluates a
retained ensemble on the shared dataset stream. Each of 2..16 named models is
loaded once and returns one combined energy/forces/virial evaluation per frame.
CPU is required and runs in a fresh device-isolated worker. No new dependency,
coordinate representation or training format is introduced.

## 1. Prepare models and inspect input

Use absolute model/dataset paths. All members and the dataset must have the same
ordered type map and `EvAngstrom` units. Atom order is the input order. Model
cutoffs and supported architectures may differ; each backend enforces its own
limits. Use independently developed/trained models when interpreting disagreement.
Byte-identical models are allowed for regression, counted in
`distinct_model_byte_hashes`, and cannot establish ensemble diversity.

Run `inspect-dataset` first when preparing a dataset. Labels may be absent;
provided labels must still have valid finite values and shapes. The deviation
calculation never uses reference labels or fits energy offsets.

## 2. Write an explicit request

```json
{
  "schema": "omnimd-model-deviation-v1",
  "models": [
    {"id": "seed-1", "options": {"native_model": "D:/work/model-1.json", "device": "cpu"}},
    {"id": "seed-2", "options": {"native_model": "D:/work/model-2.json", "device": "cpu"}}
  ],
  "dataset": "D:/work/data.jsonl",
  "per_atom": true,
  "force_window": {"min_ev_per_angstrom": 0.05, "max_ev_per_angstrom": 0.15}
}
```

The thresholds above illustrate syntax, not scientific recommendations. Choose
and assess them for the intended models and data. Omit `force_window` to compute
statistics without selecting candidates; `candidate` and `candidate_frames` are
then null. Omit `per_atom` to omit the per-atom vector while retaining aggregates.
IDs must be unique, nonempty and at most 128 bytes. Unknown fields are rejected.

Parameter-free C API members can use the existing `LoadOptions` object as
`options`. All such members must use the same canonical trusted runtime library.
Required frame/atom parameters reject here. The native CPU ensemble is covered
by the actual replay below; this new ensemble command has no real C API runtime
acceptance yet. Existing standalone C API tests do not establish this route.

## 3. Run and read the completed stream

```powershell
& $engine model-deviation D:/work/ensemble.request.json D:/work/deviations.jsonl
```

The output schema is `omnimd-model-deviation-jsonl-v1`: one header, one `deviation`
record per input frame, then `completed`. Each frame retains the source `index`
and `id`, its atom count, statistics and candidate flag. The header records
member IDs, metadata, model hashes and C API library hashes when applicable.
The terminal record contains exact SHA256 of the dataset bytes consumed, declared
upstream source hashes, model/evaluation counts and candidate count. Upstream
source files are not reopened or verified. Keep model/input files stable while
running; model hashes are file fingerprints taken before loading.

A failed frame is not emitted. An invalid/truncated dataset or evaluation error
fails the command; a partial output without `completed` is invalid. Output files
are created exclusively and existing files are never replaced. Input is streamed;
working statistics use O(atoms) memory, plus retained models and the dataset
reader's unique-ID set.

## 4. Interpret quantities and review candidates

All deviations are population standard deviations across M equally weighted
models, with divisor M. Total energy and total virial are divided by that frame's
atom count before statistics. The force deviation per atom is the Euclidean norm
of the three component standard deviations. Its min/max/arithmetic mean are
reported in eV/Angstrom. Virial uses all nine components in eV/atom; its aggregate
is the RMS of those nine deviations. It is not stress. The implemented convention
matches the absolute, reference-free branch of the pinned
[DeepMD 3.1.3 implementation](https://github.com/deepmodeling/deepmd-kit/blob/v3.1.3/deepmd/infer/model_devi.py).
This Rust implementation and JSON output are independent; no relative or
reference-error mode is provided.

A frame is a candidate exactly when
`min_ev_per_angstrom <= max_force_deviation < max_ev_per_angstrom`.
No automatic labeling, export, training or energy-zero alignment occurs.
For reviewed IDs, [select-dataset](DATASET_SELECTION.md) explicitly extracts an
unchanged subset bound to the completed dataset SHA256.
Disagreement is not calibrated uncertainty or proof of physical accuracy.

To examine a reviewed candidate with existing MD/minimization commands, refer to
its original source index via `dataset_frame: {"dataset": "D:/work/data.jsonl",
"index": 17}`. Supply the other fields required by that command, including
explicit masses for MD. Keep the source dataset hash from `completed` with the
selected index/ID. Selection thresholds do not establish simulation suitability.

## 5. Reproduce software acceptance

Build engine and `omnimd-native-eval` in the nested workspace, then:

```powershell
python mlmd/scripts/validate_model_deviation.py --engine $engine `
  --eval-exe $eval --work D:/work/new-model-deviation-run
```

The stdlib script generates three original synthetic model variants and 12
frames (3/4 atoms, nonperiodic/triclinic PBC), evaluates each model independently,
then compares every statistic with Python `statistics.pstdev` and separate vector
aggregation. It checks label independence, identical-model zeros, exact dataset
hash, candidate identity, eight refusal cases and a selected two-step CPU NVE job.
These are software fixtures, not trained physical potentials. Raw models, frames
and logs stay in the new work directory. See the compact
[recorded receipt](../evidence/model-deviation.json).
