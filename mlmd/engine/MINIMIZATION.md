# Fixed-cell geometry minimization

Both CPU backends use the same command and fallible combined E/F/virial interface:

    omnimd-mlmd minimize minimize.json new-minimum.jsonl

A request supplies exactly one shared Frame or dataset_frame reference, the
usual model options, and explicit minimization settings:

    {
      "options": {"native_model": "/absolute/native-model.json"},
      "dataset_frame": {"dataset": "/absolute/structures.jsonl", "index": 0},
      "minimization": {
        "force_tolerance_ev_per_angstrom": 0.001,
        "energy_tolerance_ev": 0.000001,
        "initial_step_angstrom2_per_ev": 0.02,
        "max_displacement_angstrom": 0.05,
        "max_backtracks": 32,
        "max_iterations": 500,
        "armijo": 0.0001
      }
    }

For DeepMD replace options with the documented library/model/device/type_map/units
object. No masses, velocities, thermostat or timestep enter geometric relaxation.
The cell and species/order remain fixed. Periodic positions remain unwrapped.
No cell optimization, constraints, classical mixing, pressure control or GPU
execution is implemented.

## Algorithm and convergence

The upstream steepest-descent step policy is shared with this new path: halve
the step on rejection, grow it by 1.1 after an accepted unbacktracked step.
The legacy infallible System force/energy methods are not used. A separate
feature-gated sim::min::MlipMinimizer owns one model, accepted Frame and its
combined evaluation. The legacy minimizer remains available independently.

Trial displacement is alpha * F, with alpha in angstrom squared/eV.
One common alpha is capped so no atom moves more than max_displacement_angstrom.
An Armijo test requires:

    E_trial - E_accepted <= -armijo * alpha * sum(force_component^2)

A trial must also strictly decrease representable energy. Every trial starts
from the last accepted coordinates; a rejected trial never becomes the next
starting point. There are at most max_backtracks + 1 trials per accepted step.
Each trial calls the combined model evaluation once and validates all outputs.
Errors abort immediately, preserving the last accepted frame and evaluation.

For accepted moves, convergence requires BOTH:

- Maximum per-atom Euclidean force norm <= force_tolerance_ev_per_angstrom.
- Absolute last accepted total-energy change <= energy_tolerance_ev.

A tiny energy change alone cannot establish convergence. The energy tolerance
is total eV, not eV/atom. Initial configurations already satisfying the force
criterion report already_converged with energy_change_ev and the energy-change
criterion null: no iteration occurred, so no energy-change claim is made.

All scalar settings must be positive finite, armijo strictly between zero and
one, max_iterations positive and max_backtracks between 1 and 128.
These are user-selected stopping thresholds, not evidence of physical quality
or a global energy minimum. Steepest descent can be slow or fail on difficult
landscapes; use the explicit status and recorded force norm.

## Results and termination

The JSONL stream contains a header, initial/accepted iteration snapshots and a
completed or failed record. Snapshots include the full Frame, combined E/F/virial,
accepted iteration count, attempted evaluation count, maximum force and signed
last energy change. A terminal unsuccessful search can repeat the last accepted
snapshot with the same iteration number.

| Status | Meaning | CLI exit |
| --- | --- | --- |
| converged | Both accepted-step criteria hold | zero |
| already_converged | Initial force criterion holds, energy change untested | zero |
| max_iterations | Accepted-step budget exhausted | nonzero |
| line_search_failed | No sufficient decrease within trial budget | nonzero |
| stalled | No representable coordinate move; no artificial zero delta-E | nonzero |
| evaluation_or_numerical_error | Evaluation/shape/finite arithmetic failed | nonzero |

None of the nonzero-exit outcomes implies a successful minimum.
If force is small but the last accepted energy change is too large, an
unrepresentable next step is reported as stalled, not false convergence.

Outputs also include:

- new-minimum.jsonl.final.extxyz: geometry only, with explicit lattice/PBC/units
  and the minimization status.
- new-minimum.jsonl.final.dataset.jsonl: one complete, **unlabeled** Frame suitable
  for a dataset_frame MD input. Predictions are not silently made training labels.

All three output names must be new. They are reserved before optimization.
A model error after initialization produces a failed record and exports the
last accepted structure; an initial validation/evaluation failure produces no
successful result. Filesystem/write failures can leave incomplete artifacts.
Normal nonconvergence writes completed with converged=false and its reason,
then exits nonzero. Consumers must check converged/status, not just completion.

The exported structure is a configuration, not a minimizer checkpoint.
Minimization restart state is not implemented. MD checkpoints remain separate.