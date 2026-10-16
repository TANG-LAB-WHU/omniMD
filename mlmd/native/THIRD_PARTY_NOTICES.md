# Sources and licenses

The new Rust implementation and original synthetic fixtures are MIT-licensed,
under the repository root LICENSE. The original inference fixture weights are
generated from integer arithmetic in `tools/generate_fixture.py`; the full
training fixture uses original deterministic trigonometric weights described
below. No proprietary weights or HAP data are included. Weight-free frozen graph profiles are
separately attributed below.

The implementation follows published DeepPot-SE mathematics and the documented
behavior of DeePMD-kit. It is independently written Rust; no DeePMD-kit source,
header, binary, or runtime is vendored. The graph-profile data described below
is derived from upstream model topology. DeePMD-kit is LGPL-3.0-or-later.
Documentation/source behavior was checked at version v3.1.0:

- Zhang et al., *End-to-end Symmetry Preserving Inter-atomic Potential Energy
  Model for Finite and Extended Systems*, NeurIPS 2018:
  https://arxiv.org/abs/1805.09003
- Descriptor theory:
  https://docs.deepmodeling.org/projects/deepmd/en/latest/model/train-se-e2-a.html
- Descriptor contraction and fixed-capacity normalization:
  https://github.com/deepmodeling/deepmd-kit/blob/v3.1.0/deepmd/dpmodel/descriptor/se_e2_a.py
- Environment rows, switching function, and normalized padding:
  https://github.com/deepmodeling/deepmd-kit/blob/v3.1.0/deepmd/dpmodel/utils/env_mat.py
- Isotropic normalization statistics:
  https://github.com/deepmodeling/deepmd-kit/blob/v3.1.0/deepmd/dpmodel/utils/env_mat_stat.py
- Explicit network layer semantics:
  https://github.com/deepmodeling/deepmd-kit/blob/v3.1.0/deepmd/dpmodel/utils/network.py
- TF fitting residual distinction:
  https://github.com/deepmodeling/deepmd-kit/blob/v3.1.0/deepmd/tf/fit/ener.py
- Periodic ghost-neighbor behavior:
  https://github.com/deepmodeling/deepmd-kit/blob/v3.1.0/deepmd/dpmodel/utils/nlist.py
- DeePMD-kit license:
  https://github.com/deepmodeling/deepmd-kit/blob/v3.1.0/LICENSE

The Rust dependencies serde, serde_json, and sha2 use MIT OR Apache-2.0 licenses.
Cargo.lock records their resolved versions and transitive dependencies; no new
runtime dependency beyond those already selected by the shared MLMD project was
introduced. This work claims neither a new descriptor algorithm nor a measured
speed advantage over DeePMD-kit.

## DeepMD 3.1.3 conversion tools

The independently written importer follows the documented serialization schema;
it does not vendor or execute DeePMD source. The optional synthetic-fixture
recipe calls an already installed DeepMD/TensorFlow runtime and uses original
fixed-seed weights. No public pretrained weights are redistributed. DeepMD 3.1.3
source references (LGPL-3.0-or-later):

- https://github.com/deepmodeling/deepmd-kit/blob/v3.1.3/deepmd/dpmodel/utils/serialization.py
- https://github.com/deepmodeling/deepmd-kit/blob/v3.1.3/deepmd/dpmodel/utils/network.py
- https://github.com/deepmodeling/deepmd-kit/blob/v3.1.3/deepmd/dpmodel/descriptor/se_e2_a.py
- https://github.com/deepmodeling/deepmd-kit/blob/v3.1.3/deepmd/dpmodel/atomic_model/base_atomic_model.py
- https://github.com/deepmodeling/deepmd-kit/blob/v3.1.3/deepmd/dpmodel/fitting/general_fitting.py
- https://github.com/deepmodeling/deepmd-kit/blob/v3.1.3/deepmd/tf/utils/serialization.py

TensorFlow 2.19.1 uses Apache-2.0. Its protobuf schema establishes the float32
storage of operation attributes; no TensorFlow source is copied:
https://github.com/tensorflow/tensorflow/blob/v2.19.1/tensorflow/core/framework/attr_value.proto

Optional tooling uses existing h5py (BSD-3-Clause), PyYAML (MIT), and NumPy
(BSD-3-Clause), under their upstream distribution licenses. No package, image,
or dependency was installed or downloaded for this conversion milestone.

## Frozen graph profile data

The independently written exporter and replay tools use TensorFlow only as a
protobuf/tensor decoder. The two `tools/frozen_profiles/*.jsonl` files are
weight-free, canonicalized structural data derived from DeepMD-generated
graphs; they retain upstream graph topology and structural constants. Treat
these profile files as LGPL-3.0-or-later, separately from the MIT Rust code.
Copyright and source attribution: DeepModeling Team / DeePMD-kit contributors.
The unchanged bundled DeepMD license and its incorporated GPL v3 text are
included beside the profiles as `LICENSE.deepmd` and `LICENSE.GPL-3`.

- Public source: https://github.com/deepmodeling/deepmd-kit/blob/v3.1.3/source/tests/infer/deeppot.pbtxt
- Public source SHA-256: `89e6aaa544dafc4620ecf49340ac2cd9e74ca0cfc2aae2c7bb060028550283a3`
- Descriptor graph mapping: https://github.com/deepmodeling/deepmd-kit/blob/v3.1.3/deepmd/tf/descriptor/se_a.py
- Network/residual mapping: https://github.com/deepmodeling/deepmd-kit/blob/v3.1.3/deepmd/tf/utils/network.py
- Atomic offsets: https://github.com/deepmodeling/deepmd-kit/blob/v3.1.3/deepmd/tf/fit/ener.py
- Model output offsets: https://github.com/deepmodeling/deepmd-kit/blob/v3.1.3/deepmd/tf/model/ener.py
- Frozen graph version: https://github.com/deepmodeling/deepmd-kit/blob/v3.1.3/source/config/MODEL_VER

Changes to graph data: discard unconnected training metadata, replace explicitly
mapped learned parameters with shape-checked placeholders, replace two effective
cutoff attributes with placeholders, decode structural tensors, and sort nodes.
The synthetic profile comes from the original fixed-seed recipe already cited.
`tools/make_frozen_profiles.py` plus the two recorded source hashes reproduces
these modified structural files. No public trained weights or `.pb` binaries
are committed. The corresponding source is the linked versioned public graph,
the synthetic recipe, and the profile-building script in this directory.

## Native readout training

Readout feature extraction and the small column-pivoted Householder QR solver
are independently written Rust. The training replay generates original synthetic
teacher labels using the native evaluator; no external training labels or
models are added. sha2 0.10.9 is reused from the existing shared engine dependency
and its approved isolated offline cache for dataset/model provenance. Its
transitive versions are recorded in Cargo.lock. The existing serde_json
`float_roundtrip` feature was enabled to preserve frozen float64 parameters
through parse/export. No new package download or global configuration change
was performed. This is final-readout regression, not a new training algorithm.

## Full supported-network training

The scalar reverse tape, differentiable directional tangents, full-batch Adam,
checkpoint code and tiny tanh fixture are independently written MIT code/data.
No external AD or optimizer source is copied or vendored. The added fixture
uses original deterministic trigonometric weights; the replay perturbs them
and generates original teacher labels. No pretrained weights or external
training dataset is included. Existing dependencies and licenses are unchanged.

Mathematical references (not redistributed):

- Kingma and Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015:
  https://arxiv.org/abs/1412.6980
- Baydin et al., *Automatic Differentiation in Machine Learning: a Survey*,
  JMLR 18(153), 2018: https://arxiv.org/abs/1502.05767

Reverse/forward AD composition and Adam are established methods, not algorithmic
innovations claimed by this implementation.

## Optional CUDA inference

The CUDA kernel source, Rust dynamic bindings and replay generator are original
MIT code. No NVIDIA headers, binaries, libraries or samples are copied into the
repository. Official installed NVIDIA Driver/NVRTC components remain subject to
their NVIDIA license terms; they are dynamically loaded only for explicit CUDA
use. No dependency or driver installation is automated. References:

- NVIDIA NVRTC documentation: https://docs.nvidia.com/cuda/nvrtc/index.html
- NVIDIA CUDA Driver API: https://docs.nvidia.com/cuda/cuda-driver-api/index.html
- NVIDIA CUDA Toolkit EULA: https://docs.nvidia.com/cuda/eula/index.html

The original synthetic timing model/coordinates are generated by
`tools/replay_cuda.py`; public converted weights remain separately sourced and
ignored as documented in FROZEN.md. CUDA is an implementation backend for the
same published descriptor/network mathematics, not an algorithmic novelty.

## CUDA host neighbor cache

The conservative periodic-image candidate cache, host profiling and
`tools/replay_cuda_infra.py` molecular-lattice recipe are independently written
MIT code. No external neighbor-list implementation was copied. Verlet-style
candidate reuse is an established method, not a new descriptor or algorithm.
The 192-atom geometry is original synthetic data; its optional public model
weights retain the separate DeepMD source provenance above and remain ignored.
No additional dependency, toolkit component or driver was installed.

## Cooperative CUDA fitting and persistent host workspace

The block-per-atom fitting kernel, separate scratch bounds, workspace reuse,
resource diagnostics and boundary/lifecycle fixture recipe are independently
written MIT code. No external kernel or NVIDIA header is copied. The existing
DeepMD 3.1.3 descriptor/network sources above were consulted for the unchanged
contraction, residual and timestep semantics. NVIDIA's programming guide and
Driver API function-attribute documentation were consulted for shared-memory
barriers and resource reporting. No additional dependencies or runtime
components were installed; existing third-party runtime license terms remain.

## CUDA-assisted final-readout training

The private resident one-hot coefficient selection, explicit training-device
routing and synthetic replay are independently written MIT code. They reuse
this project's existing analytic CUDA inference and CPU pivoted QR; no external
training or optimizer code was copied. The teacher fixture and all generated
labels are original synthetic data. Existing DeepMD mathematical references,
NVIDIA runtime terms and dependency licenses above remain applicable. No new
dependency, source download, binary redistribution or algorithmic novelty is
introduced by this backend change.

## Full neural-parameter CUDA mixed derivatives and Adam

The four-component arithmetic, full CUDA gradient/Adam kernels, execution
routing, GPU checkpoint metadata and synthetic checks are independently written
MIT code. No external AD, hyper-dual class, optimizer, CUDA sample/header or
trained model is copied. The mathematical hyper-dual chain rule is established;
reference: Martin Neuenhofen, *Review of theory and implementation of hyper-dual
numbers for first and second order automatic differentiation* (2018),
https://arxiv.org/abs/1801.03614. Adam and the original se_e2_a references above
remain applicable. These are implementation choices, not algorithmic novelty.
Existing NVIDIA runtime terms and approved dependency licenses are unchanged;
no dependency installation, download or global configuration change occurred.

## Optional virial-loss supervision

The shared loss-seed construction, row-strain directional extension, v3 loss
contract and analytic/finite-difference verification are independently written
MIT code. They use the existing DeepPot-SE row virial convention and standard
chain-rule differentiation; no external training or autodiff source was copied.
The dimer, triclinic/self-image cases and teacher labels are original synthetic
fixtures. Existing source references and dependency licenses remain applicable.
No dependency, model, driver or toolkit download was introduced.

The cached CUDA reverse-over-forward kernels, scratch planner and synthetic
replay are original MIT code implementing standard chain rules; no third-party
AD or optimizer source was copied and no additional dependency was introduced.

The fixed training-frame geometry cache, exact identity checks and cache-on/off
replays are original MIT code, sharing the existing native environment and neighbor
formulas. No external cache implementation or new dependency was incorporated.

## Deterministic minibatch sampling

The epoch sampler, v5 checkpoint validation and synthetic tests/replay are original
MIT code. They reuse this repository's existing SplitMix64 integer arithmetic.
Reference: Sebastiano Vigna (2015), https://prng.di.unimi.it/splitmix64.c, released
with a public-domain dedication and unrestricted permission to use/modify/distribute.
No external sampler or training library is vendored and no dependency was added.

## Indexed dataset access

The bounded JSONL index, storage adapter, integrity/memory tests and synthetic
replay are original MIT code using existing Rust standard-library, serde and
sha2 facilities. No external dataset engine or source was copied and no dependency
was added. The 2051-frame acceptance input repeats original synthetic geometries
and adds JSON whitespace for an explicit file-size boundary check.
