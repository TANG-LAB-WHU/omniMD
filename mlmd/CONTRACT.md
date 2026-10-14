# Shared MLMD interface v1

Crate: `omnimd-ml-contract`. Both backends implement:
`Potential::metadata(&self) -> &ModelMetadata`,
`Potential::eval(&mut self, frame: &Frame) -> Result<Evaluation>`.

Frame has `positions: Vec<[f64;3]>`, `species: Vec<String>`,
`cell: Option<[[f64;3];3]>`, and mandatory `units: Units::EvAngstrom`.
No atomic-number guesses or mass matching. Type IDs are indices in the
model's ordered, unique chemical-element `type_map`.
None means nonperiodic; Some means periodic in ALL three axes.
Cell rows are lattice vectors a,b,c (angstrom), positions are Cartesian.
Cells must be finite, right handed and nonsingular.

Evaluation has `energy: f64` in eV, `forces: Vec<[f64;3]>` in eV/angstrom,
`virial: [[f64;3];3]` in eV, plus `units`.
All three outputs are required from one evaluation. No missing-output zero
fallback. Force = -dE/dr. Virial follows DeepMD row-major tensor order:
W_ab = -dE/d epsilon_ab where r'_b = r_b + sum_a r_a epsilon_ab,
and cell rows deform the same way. It is virial, not stress; no volume division.
Check off-diagonal strain derivatives when implementing the native backend.

Backend validates frame, mapping and metadata before compute, checks runtime
errors after compute, and validates every output shape and value.
Use `Frame::validate`, `ModelMetadata::validate`, `Frame::type_ids`,
`Evaluation::validate`. Errors are `Error(String)` with diagnostic context.
The caller must not change a model's ordered type map to fit a frame.

All storage and native math are float64. The FFI double API only guarantees
double I/O; it cannot upgrade the internal precision of a frozen model.

The legacy omniMD cell stores column vectors: transpose at that future adapter
boundary. Convert eV to legacy internal energy exactly once, and apply the same
energy conversion to forces (length already angstrom) and virial.

No Send/Sync guarantee is imposed. A mutable eval owns any backend cache.
A failed evaluation returns no publishable partial output.
