# Current supported training scope and remaining work

Final-readout regression ([TRAINING.md](TRAINING.md)) and full supported
tanh-network E/F Adam training ([FULL_TRAINING.md](FULL_TRAINING.md)) work on CPU,
with bounded CUDA execution. Full training supports optional
[virial supervision](VIRIAL_TRAINING.md),
[deterministic minibatching](MINIBATCH_TRAINING.md),
[bounded indexed data](INDEXED_TRAINING.md), and explicit
[constant/exponential update-step schedules](LEARNING_RATE.md). Validation
selection, exact mixed parameter-coordinate derivatives and serialized optimizer/
sampler restart remain explicit.

CUDA offers the existing hyperdual reference and
[cached reverse gradients](CUDA_REVERSE_TRAINING.md), plus optional
[fixed geometry reuse](FIXED_GEOMETRY_TRAINING.md) and
[bounded multi-frame kernel execution](TILED_TRAINING.md). All neural weights, biases and
present timesteps train; normalization, atomic offsets, architecture and type
mapping remain fixed. An explicit [GPU prediction/seed path](DEVICE_LOSS_TRAINING.md)
reuses primal intermediates within a tile. CPU loss remains the default. Geometry,
stable reduction/clipping, schedule scalars and assessment remain on CPU.

Remaining work includes larger-model memory/performance, randomized initialization,
normalization-statistics and offset estimation, tied one-side embeddings, other
training activations/descriptors, broader schedules/optimizers, multi-GPU or
distributed execution, and physically representative training validation. The
current staircase schedule does not implement warmup, continuous decay, a rate
floor, adaptive scheduling or scheduled loss weights.

This is not full DeepMD training-system compatibility. Converted native-v1 models
can initialize a new optimizer only when they satisfy the documented subset and
resource bounds. Source framework aliases are not preserved as tied parameters.
Historical missing-schedule v1-v5 checkpoints remain supported; explicit schedule
v6 retains applicable backend, loss and sampler checks. Tiled reverse execution
uses v7 with immutable capacities and its own backend/program identity. Optional
GPU prediction/seeds takes precedence with a separate v8 execution identity.

Original synthetic teacher tests establish implementation behavior. They do not
establish chemical accuracy, extrapolation, stable production MD, algorithmic
novelty or general speed advantages.
