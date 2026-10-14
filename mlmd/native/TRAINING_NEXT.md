# Training status after the full supported-network milestone

Both final-readout regression ([TRAINING.md](TRAINING.md)) and full supported
tanh-network E/F Adam training ([FULL_TRAINING.md](FULL_TRAINING.md)) now work.
The latter includes exact parameter-coordinate mixed derivatives, all embedding
and fitting matrices/biases/timesteps, stable full-batch ordering, explicit
validation selection and normalization, and serialized checkpoint/restart.

Remaining work is outside this delivery candidate: large-model memory and
performance work, minibatches, randomized initializers, learning-rate schedules,
normalization-statistics estimation, weight tying for one-side embedding models,
virial-loss optimization, other activations, other DeepMD descriptors, and
physically representative data validation. There is no GPU implementation or
full DeepMD training-system compatibility claim. Existing converted native-v1
models can be used as initial weights if they satisfy the documented trainer
subset and resource bounds. Source framework aliases are not preserved as tied
parameters in the native format.

The synthetic teacher and split validation demonstrate implementation behavior
only. They do not establish chemical accuracy, extrapolation, stable production
MD, algorithmic novelty or speed advantages. No further feature expansion is
part of this milestone.
