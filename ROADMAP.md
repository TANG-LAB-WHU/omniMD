# Rust-based MLIP-MD roadmap

## Milestone 1: strict model evaluation

- Shared fallible, unit-explicit, float64 energy/force/virial contract.
- A: official DeepMD C API dynamic binding, single-load handle lifecycle,
  explicit device policy, metadata/type checks, finite/shape/error checks.
- B: native CPU float64 se_e2_a environment matrix, neighbor images/PBC,
  smooth cutoff, normalization, embedding and energy fitting networks,
  versioned weight loader, analytic or automatic energy derivatives.
- Synthetic/public fixtures only; finite-difference force and strain checks.
- Unsupported architectures (including se_e2_a + se_e3 hybrids), precisions,
  units, and features must fail clearly.

## Milestone 2: engine integration and scientific validation

- Connect combined evaluation to the upstream simulation engine with a
  fallible API and an explicit conversion at the internal-unit boundary.
- Reuse each evaluated state across energy/forces/virial consumers.
- Public-model parity against official DeepMD, NVE energy conservation,
  timestep convergence, pressure/virial orientation validation.
- Model format exporters only after a supported architecture is documented.
- Benchmarks with fixed hardware, models, precision and error tolerances.

## Milestone 3: training and acceleration (not implemented)

- Native training, optimizer/checkpoints/data pipeline and force-loss gradients.
- Sampling and active-learning workflows with explicit supported data formats.
- Native GPU kernels and deterministic numerical comparison.
- Broader architectures only with explicit implementation and validation.

## Publication gate

Before any push, report the exact code/document/fixture scope and validation
limits. Keep the existing remote shawn_dev until replacement is explicitly
authorized for publication. Never include private research data/models/logs.
