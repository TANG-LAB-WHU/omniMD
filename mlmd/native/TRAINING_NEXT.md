# Training status after the full supported-network milestone

Both final-readout regression ([TRAINING.md](TRAINING.md)) and full supported
tanh-network E/F Adam training ([FULL_TRAINING.md](FULL_TRAINING.md)) now work.
The latter also supports optional [virial supervision](VIRIAL_TRAINING.md) on
CPU/CUDA with the shared row-strain convention.
The latter includes exact parameter-coordinate mixed derivatives, all embedding
and fitting matrices/biases/timesteps, stable full-batch ordering, explicit
validation selection and normalization, and serialized checkpoint/restart.

Remaining work is outside this delivery candidate: large-model memory and
performance work, minibatches, randomized initializers, learning-rate schedules,
normalization-statistics estimation, weight tying for one-side embedding models,
other activations, other DeepMD descriptors, and
physically representative data validation. The [GPU-assisted readout path](TRAINING.md#explicit-cpu-or-cuda-features) now
generates analytic final-layer columns on CUDA and solves/assesses them on CPU.
The [bounded full CUDA trainer](CUDA_FULL_TRAINING.md) now computes all neural
parameter/mixed derivatives and Adam updates on GPU, while loss cotangents,
geometry, stable reduction/clipping and assessment remain CPU work. Its per-
parameter forward computation still needs shared-intermediate or reverse-mode
optimization for larger networks. CPU v1 and backend-bound GPU v2 checkpoint
resume and v3 virial-loss checkpoints are independently verified.
There is no full DeepMD training-system compatibility claim. Existing converted native-v1
models can be used as initial weights if they satisfy the documented trainer
subset and resource bounds. Source framework aliases are not preserved as tied
parameters in the native format.

The synthetic teacher and split validation demonstrate implementation behavior
only. They do not establish chemical accuracy, extrapolation, stable production
MD, algorithmic novelty or speed advantages. No further feature expansion is
part of this milestone.
