# Sources and licenses

The new Rust implementation and original synthetic fixtures are MIT-licensed,
under the repository root LICENSE. The fixture weights are generated locally
from integer arithmetic in `tools/generate_fixture.py`; no trained or proprietary
weights, HAP data, or external model fixture is included.

The implementation follows published DeepPot-SE mathematics and the documented
behavior of DeePMD-kit. It is independently written Rust; no DeePMD-kit source,
header, model, binary, or runtime is vendored. DeePMD-kit is LGPL-3.0-or-later.
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

The runtime dependencies serde and serde_json use MIT OR Apache-2.0 licenses.
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
