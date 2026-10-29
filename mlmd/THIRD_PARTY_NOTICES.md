# Third-party notices

omniMD is MIT licensed; the original root LICENSE is retained.
The new Rust code is distributed under the same MIT license.

DeePMD-kit is LGPL-3.0-or-later:
https://github.com/deepmodeling/deepmd-kit/tree/v3.1.3
The FFI declarations follow the official C API v26 header and implementation:
https://github.com/deepmodeling/deepmd-kit/blob/v3.1.3/source/api_c/include/c_api.h
https://github.com/deepmodeling/deepmd-kit/blob/v3.1.3/source/api_c/src/c_api.cc
No DeepMD runtime, pretrained model, source implementation or C header is vendored.
Audited frozen-graph topology profiles derived from public test graphs have
separate LGPL provenance and license files in native/tools/frozen_profiles; see
[profile notices](native/tools/frozen_profiles/README.md).
Runtime distributors remain responsible for the applicable LGPL obligations.

The project uses serde, serde_json, sha2, rand and rand_chacha under their upstream
MIT/Apache-2.0 licensing and libloading under ISC; see Cargo.lock and the upstream package notices.

Native implementation provenance and its generated-fixture notices are documented
in [native/THIRD_PARTY_NOTICES.md](native/THIRD_PARTY_NOTICES.md).

The dataset importer uses npyz 0.9.1 and py_literal 0.4.0 (MIT), with their
locked parser dependencies; upstream crate licenses remain applicable.

The model-deviation statistics follow the absolute population convention described
by DeepMD 3.1.3 `deepmd/infer/model_devi.py` (LGPL-3.0-or-later). The Rust moments,
streaming workflow and stdlib replay are independently written; no upstream
implementation is copied or vendored for this feature.
