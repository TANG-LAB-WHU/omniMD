# Third-party notices

omniMD is MIT licensed; the original root LICENSE is retained.
The new Rust code is distributed under the same MIT license.

DeePMD-kit is LGPL-3.0-or-later:
https://github.com/deepmodeling/deepmd-kit/tree/v3.1.3
The FFI declarations follow the official C API v26 header and implementation:
https://github.com/deepmodeling/deepmd-kit/blob/v3.1.3/source/api_c/include/c_api.h
https://github.com/deepmodeling/deepmd-kit/blob/v3.1.3/source/api_c/src/c_api.cc
No DeepMD runtime, model, source implementation or header is vendored.
Runtime distributors remain responsible for the applicable LGPL obligations.

The project uses serde, serde_json and libloading under their upstream
MIT/Apache-2.0 licensing; see Cargo.lock and the upstream package notices.
