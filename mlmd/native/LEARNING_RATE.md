# Native optimizer-update learning-rate schedules

The supported full-neural-parameter CPU and CUDA trainers accept optional
`config.learning_rate_schedule` in the existing
`omnimd-native-full-train-v1` request. The engine forwards this request unchanged.
This extends neither model architecture nor physics. The readout-only least-squares
trainer has no iterative learning rate and rejects this field.

## Enable an explicit schedule

The existing positive finite `learning_rate` remains the base rate. Add either
of these configuration fragments:

```json
{
  "learning_rate": 0.005,
  "learning_rate_schedule": {"kind": "constant"}
}
```

```json
{
  "learning_rate": 0.005,
  "learning_rate_schedule": {
    "kind": "exponential",
    "decay_steps": 2,
    "decay_rate": 0.5
  }
}
```

Absent or null `learning_rate_schedule` preserves the historical constant
behavior and is omitted when serialized. Explicit `constant` gives the same
numerical rate, but is an opt-in checkpoint identity. Exponential schedules
require both fields: integer `decay_steps` in 1..1000000000 and finite
`decay_rate` in (0,1]. Unknown kinds, extra fields, missing fields, fractional
intervals, zero intervals, and invalid rates fail explicitly. There is no
automatic choice from dataset size, epoch count, validation labels or total steps.

## Formula and update numbering

Let `k` be the number of globally completed optimizer updates immediately
before the next update. Let `lr0` be `config.learning_rate`, `d` be
`decay_steps`, and `q` be `decay_rate`:

```text
constant:     lr(k) = lr0
exponential:  j = floor(k / d)
              lr(k) = lr0 * q^j
```

This is always staircase exponential decay. Updates 1 through `d` use
`lr0`; update `d+1` first uses `lr0*q`. For the example above, the first
seven updates use `0.005, 0.005, 0.0025, 0.0025, 0.00125, 0.00125, 0.000625`.

Integer division determines `j`. The Rust host evaluates
`lr0 * decay_rate.powi(j as i32)` in float64; the existing billion-update
ceiling makes the integer conversion exact and safe. Constant schedules return
`lr0` directly. The rate is derived from the original base and absolute step
on every update, never from a recursively modified rate. The same host function
feeds CPU Adam and the existing CUDA Adam scalar argument. CUDA kernels, PTX
sources and backend identities do not change.

Adam still uses the one-based update number `t=k+1` for bias correction:
`m_hat=m_t/(1-beta1^t)`, `v_hat=v_t/(1-beta2^t)`, and
`theta_t=theta_(t-1)-lr(k)*m_hat/(sqrt(v_hat)+epsilon)`.
Clipping, gradient reduction and loss normalization are unchanged.

A full batch, an ordinary minibatch and a final short minibatch each advance
`k` by one only when the update succeeds. A short batch receives no additional
learning-rate scaling. Epoch transitions, selected-frame count, indexed/memory
access and geometry-cache reuse do not reset or advance the schedule. Validation
labels never enter schedule selection or updates.

These are explicitly defined native formulas. No equivalence to DeepMD defaults,
configuration names, interpolation, stop-rate rules or training trajectories is
claimed. This is standard scheduling, not an algorithmic novelty.

## Validation and finite float64 limits

For a checkpoint at `k0` and `R>0` additional updates, the used rates are
`lr(k0)..lr(k0+R-1)`. Preflight checks the first and last rates of the
nonincreasing schedule, with checked endpoint arithmetic, before CUDA context
creation or optimizer/checkpoint publication. The following unused
`lr(k0+R)` is not required to be representable. Every actual update also checks
its rate.

The computed rate must be positive and finite. Zero from float64 underflow is
an error, with no implicit floor or clamping. Positive subnormal rates remain
accepted, consistent with historical constant-rate acceptance. The power is
evaluated before multiplication by the base; there is no logarithmic rescue of
an underflowed intermediate. A tiny valid rate does not promise a representable
parameter change or useful convergence. There is no warmup, continuous decay,
cosine/plateau scheduler, minimum rate, schedule restart or scheduled loss weights.

Existing limits remain: 1..100000 additional updates per request and at most
1000000000 globally completed updates. Zero-update requests remain invalid.

## Checkpoint and report identity

An explicit schedule, including explicit constant, uses
`omnimd-native-full-checkpoint-v6` on every backend. v6 keeps the applicable
E/F or E/F/V loss contract, backend/program identity, exact dataset/layout hash,
frozen model fields, Adam state and optional minibatch sampler checks. No
separate mutable scheduler state is needed: the immutable configuration and
completed `step` determine the next rate.

Base rate, schedule presence/kind, interval and decay rate must match exactly on
resume. Do not canonicalize absent into constant or rate-one exponential into
constant; they intentionally have distinct identities. `config.steps` still
means additional requested updates and may change. Runtime storage/cache choices
retain their existing compatibility rules. To change a schedule intentionally,
initialize a new optimizer from the exported model; do not rewrite a checkpoint.

Missing/null schedules preserve historical v1-v5 checkpoint selection and
serialization. Exact restart is verified within the same build/platform/runtime;
Rust's power implementation and floating-point math do not imply bitwise
portability across toolchains or architectures. Checkpoints remain unsigned.

Reports add a `learning_rate` object with contract
`omnimd-optimizer-update-learning-rate-v1`, the effective schedule, base rate,
one-based first/last update numbers and first/last **used** rates for that
invocation. On resume these refer to the resumed interval. The report does not
store an unbounded per-step trace, select a best model or affect restart identity.

## Focused verification and replay

```text
cargo test --locked --offline --manifest-path mlmd/native/Cargo.toml
cargo test --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda
cargo build --release --locked --offline --manifest-path mlmd/native/Cargo.toml --features cuda --bins

python mlmd/native/tools/replay_minibatch_training.py --eval-exe EVAL_EXE --train-exe TRAIN_EXE --steps 7 --output-dir FIXTURE_DIR
python mlmd/native/tools/replay_learning_rate.py --train-exe TRAIN_EXE --fixture-dir FIXTURE_DIR --output-dir CPU_OUTPUT

# Execute only when GPU use is authorized:
python mlmd/native/tools/replay_learning_rate.py --train-exe TRAIN_EXE --fixture-dir FIXTURE_DIR --output-dir CUDA_OUTPUT --device cuda:0 --cpu-reference CPU_OUTPUT
python mlmd/native/tools/replay_learning_rate.py --train-exe TRAIN_EXE --fixture-dir FIXTURE_DIR --output-dir REVERSE_OUTPUT --device cuda:0 --cuda-gradient reverse --cpu-reference CPU_OUTPUT
```

Use new output directories and existing official CUDA libraries. All replay
fixtures and labels are original synthetic examples; no new dependency is needed.

The Rust checks include independent explicit Adam updates using hard-coded rates,
decay boundaries, strict configuration parsing, constant/legacy serialization and
numerical parity, index/memory and minibatch resume, sampler/schema corruption,
immutable schedule rejection, billion-step arithmetic, and the last-used versus
next-unused rate underflow boundary.

Actual CLI checks on Windows/RTX 4090 exercised CPU and both CUDA gradient
executors, each with full-batch and E/F/V minibatch training. Seven-update
continuous runs exactly matched serialized resumes at steps 1, 2, 3 and 4 for
minibatches, including partial batches and indexed/memory switches; full-batch
resumes at steps 2 and 3 also matched. CUDA resume toggled fixed geometry reuse.
Constant and rate-one exponential runs matched legacy parameters/moments exactly.
The CPU legacy replay also matched its historical v5 checkpoint. Invalid
configuration and incompatible resume requests failed before output creation.

Maximum absolute differences from CPU in scheduled final parameters and Adam
moments were 9.44e-16 (CUDA hyperdual) and 8.88e-16 (CUDA reverse). These tiny
synthetic runs verify scheduling behavior, not physical accuracy, convergence,
production scaling or speed. [Compact evidence](fixtures/learning-rate-report.json)
records commands, hashes, execution identities, rates and checks. Generated
weights, datasets, checkpoints and raw logs stay ignored.

Tiled reverse CUDA execution takes checkpoint-version precedence (v7), while
retaining these exact schedule semantics; see [TILED_TRAINING.md](TILED_TRAINING.md).
