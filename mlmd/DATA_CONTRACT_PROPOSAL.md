# Dataset contract proposal, v1

Status: implemented in engine and agreed with B for its bounded training reader; the shared Frame/Potential contract
and native backend remain unchanged. A future training consumer should agree on
this version explicitly before reusing it.

Use a JSONL header, consecutive frame records and one mandatory completed record.
Frames embed the existing Frame unchanged. Optional energy, forces and full 3x3
virial labels are independent; absent labels are never zero or training targets.
Units are explicitly EvAngstrom; virial is W_ab=-dE/d epsilon_ab with row
coordinates r'=r(I+epsilon), in eV, not stress. Header type_map defines allowed
chemical elements, not atomic numbers; actual model mapping remains by symbol.

First importer: a bounded DeepMD standard NPY subset. The user names the system,
ordered set.* directories, units and Periodic/NonPeriodic policy. No automatic
set selection, train/test split, shuffle, training, offset fitting or label
generation. Require type.raw and type_map.raw. Validate nopbc against the stated
policy. Accept C-order float64 coordinates, boxes and optional E/F/virial arrays.
Unsupported arrays/active auxiliary inputs, mixed types and nonfinite/malformed
data fail. Nonperiodic box.npy is rejected rather than silently ignored.
Source file digests and selected sets are recorded for replay.

Batch evaluation loads a model once, emits every prediction in input order, and
requires a complete dataset. Report pooled component MAE/RMSE/max and counts for
F and virial, frame-weighted total-E and E/atom metrics, and label coverage.
No bias subtraction, pressure conversion, vector-norm error or per-system
reweighting. Incomplete output lacks the completed marker. Outputs are new files.

A dataset frame reference (absolute path plus zero-based index) can initialize
the MD CLI with explicit masses and velocity policy. It does not consume labels.
Static capabilities and loaded-model metadata are separate query commands.

The exact implemented schema and CLI are documented in [engine/DATA.md](engine/DATA.md).
