# Training follow-up after the first implementation

The frozen-hidden final-readout milestone is implemented and exercised through
the native CLI. See [TRAINING.md](TRAINING.md) for its actual input, objective,
limits and results. It does not train descriptor or hidden-network parameters.

Full fitting-network training would first need parameter layouts for every
matrix, bias and timestep; reusable forward/reverse network tapes; and activation
first/second derivatives. Frozen descriptor values and coordinate derivatives
can be cached. For half squared loss with residual seeds
`a=lambda_E(E-E*)` and `v=lambda_F(F-F*)`, after applying the documented loss
normalization:

```text
grad_theta L = a grad_theta E - grad_theta [v . grad_r E]
```

Keep the residual seeds fixed during this seeded derivative. Reverse parameter
differentiation through a forward coordinate-directional derivative computes
the mixed term without materializing a full Hessian. Residual paths and timestep
products must contribute to both derivative levels. Ordinary energy
backpropagation alone cannot optimize the force loss correctly.

Training embedding networks as well requires extending that derivative path
through environment normalization, embedding activations, the descriptor
contraction, and all neighbor contributions. Keep type mapping, selection,
normalization statistics and cutoff policy fixed initially. The existing
quintic switch has the needed smooth second derivative at its boundaries;
neighbor-capacity overflow remains an error. Virial-loss training would add
strain-directional seeds with the same cell deformation convention.

Replace repeated basis evaluations with cached/shared computation before making
large-model performance claims. A later optimizer needs native minibatching,
seeded ordering, checkpointed optimizer state, explicit learning-rate settings,
and validation-only reporting. The current dataset and inference model formats
can stay intact; any shared training-state contract needs parent coordination.

Acceptance for the next extension: one small analytic mixed parameter-coordinate
gradient versus finite differences, real energy-and-force objective reduction,
and exported-model inference agreement. Finite differences are a test oracle,
not a production parameter-gradient implementation. Full-network and embedding
training remain future work; no new dependency or optimizer install is assumed.
