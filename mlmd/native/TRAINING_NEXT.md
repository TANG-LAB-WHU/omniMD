# Next bounded native training milestone (proposed, not implemented)

Keep the descriptor, embedding networks, fitting hidden layers, and separate
`atom_energy_bias` fixed. Fit only the final affine readout for each species.
This gives a useful small CPU float64 energy-and-force training path with exact
derivatives and no shared `Potential` or model-format change.

For fixed hidden features `h_i`, per-type weights `w_t`, and biases `b_t`:

```text
E = sum_t [w_t . sum_{i:type_i=t} h_i + b_t N_t] + E_fixed
F_q = -sum_t w_t . sum_{i:type_i=t} d(h_i)/d(r_q)
```

Energy and force are linear in the trainable coefficients. Build force design
rows from analytic hidden-feature coordinate derivatives, not detached force
predictions. Bias columns have zero force derivative. Existing fixed offsets
are included exactly once. Selection/cutoff boundaries remain subject to the
native neighbor-capacity and differentiability constraints.

Implement a native-only feature/Jacobian path and a small training CLI reading
shared-contract frames with explicit energy/force labels. Use a documented
weighted least-squares objective with separate mean-square energy and force
normalization, optional explicit ridge regularization, and a rank-aware QR
solve. Do not invert normal equations. Fixed-composition data cannot identify
all species bias offsets; freeze biases by default or report deficient rank.
Keep any needed label/training structs inside `mlmd/native` until the parent
approves a shared dataset contract. No dependency installation is assumed.

Minimal acceptance:

1. Arbitrary readout coefficients reproduce ordinary native E/F through the
   feature matrices, including fixed offsets.
2. A coordinate finite-difference check verifies the sign and indexing of
   force features away from neighbor-topology changes.
3. Fit a small manufactured dataset from known readouts, reduce the stated
   E/F objective, and reproduce independent-frame predictions at a stated
   tolerance. Compare predictions, not non-identifiable coefficient vectors.
4. Save native v1 weights, reload and evaluate through the unchanged Potential;
   report actual training/validation errors, with no chemical-accuracy claim.

A later full fitting-MLP trainer with frozen descriptors requires mixed
parameter-coordinate derivatives. For half squared loss with seeds
`a = lambda_E (E-E*)` and `v = lambda_F (F-F*)`:

```text
grad_theta L = a grad_theta E - grad_theta [v . grad_r E]
```

Treat the residual seeds as fixed during this seeded derivative. Forward
coordinate-directional differentiation followed by reverse parameter
differentiation computes the mixed term without a full Hessian. It needs
activation second derivatives and derivative propagation through residuals and
timesteps. Ordinary energy backpropagation alone is insufficient for force
training. Gate that extension on one mixed-gradient finite-difference check and
actual force-loss reduction; end-to-end descriptor training is a later scope.
