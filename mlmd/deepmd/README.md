# Official DeepMD C API backend

This crate dynamically loads a trusted official DeepMD C library. Each request
loads one model handle and keeps it for all frames or MD steps. Each homogeneous group uses
one combined `DP_DeepPotCompute2` call for energy/force/virial and optional atomic
outputs. Scratch buffers are reused; parameters are explicit per frame.
The official runtime still controls its own host/device transfers and internal
atomic outputs. This is inference through FFI, not native Rust inference.

Start with the English [step-by-step backend guide](BACKEND_GUIDE.md) for runtime
setup, model inspection, training/freeze/test connection, batched/parameterized
inference and error handling. See [current validation status](BACKEND_VALIDATION.md)
for the distinction between tested ABI behavior and actual runtime results.

CPU is the default and needs no CUDA SDK, libraries or driver to build/test.
CUDA is opt-in, dynamically loaded and currently restricted to official
TensorFlow frozen `.pb` models and direct CUDA kernel execution. CPU retains the
existing runtime-supported model scope. No automatic fallback is performed.
For an explicit CPU fallback, run a new request with `"device":"cpu"` and a
trusted CPU-compatible DeepMD runtime. A CUDA error is never published as GPU
success, and a failed GPU verifier refuses further evaluations on that handle.

## Devices and isolation

All CLI paths (`omnimd-deepmd`, MD/check, batch-eval, model-info, minimization)
start a fresh worker and configure visibility before loading any runtime.
Accepted device forms are `"cpu"` (also `"Cpu"`), `"cuda"` (index zero),
`"cuda:N"`, or legacy `{"Cuda":N}` / `{"cuda":N}`. Omission defaults to CPU.

A CUDA index selects an entry in the parent's `CUDA_VISIBLE_DEVICES` list when
that variable is set. UUID and MIG tokens are preserved. Without an inherited
mask, it selects a CUDA ordinal using the inherited CUDA device-order policy;
this is not necessarily the ordinal displayed by nvidia-smi. The worker exposes
only that entry and passes **rank zero** to DeepMD, avoiding its modulo device
mapping. Invalid/out-of-range/disabled inherited masks fail. A CUDA runtime
count other than exactly one fails. HIP/ROCR visibility is disabled in every
worker; the parent's environment is unchanged. Containers/schedulers may impose
an additional device visibility boundary.

The unsafe library API requires the same fresh-process initialization discipline:
configure a child with `Device::configure_worker` before any accelerator runtime
loads. Loading native libraries is a trust boundary, not a sandbox.

## CUDA execution evidence

The official TensorFlow C API enables soft placement and provides no portable
actual-device query. We therefore require explicit matching NVIDIA CUDA runtime
and CUPTI library paths. The guard checks device availability, then observes
successful direct driver kernel launches during **each** combined C API call (one frame or a homogeneous batch).
It synchronizes the actual observed CUDA contexts before returning the outputs.
Zero launches, driver/launch/synchronization errors, failed tracing setup, CUDA
graphs or stream capture fail explicitly. The fresh worker must be dedicated to
one model and must not run unrelated concurrent GPU work. Concurrent profilers
may prevent CUPTI subscription; this is an explicit runtime error.

This evidence proves GPU kernels executed during the model evaluation. It does
not assert that every operation runs on GPU: shape, bookkeeping and unsupported
TensorFlow operations can remain on CPU. Graph-only runtimes are unsupported by
this verifier, even if their graphs could contain GPU kernels.

Before evaluation metadata says `deepmd-c-api-cuda-pending-evaluation`; after a
validated CUDA evaluation it says `deepmd-c-api-cuda-kernels-verified`. Thus
model-info alone cannot attest GPU execution. The standalone evaluator includes
the total observed `cuda_kernel_launches`; CPU output omits that field. Batch
output's terminal record includes final metadata, while its initial header can
still be pending. Failed evaluations never return publishable partial EFV.

## Model and precision scope

- Interface: official DeepMD 3.x C API v26 symbols; CPU baseline tested on 3.1.3.
  Symbols alone are not complete ABI validation. Use matching trusted builds.
- Explicit ordered chemical-element type map and declared `EvAngstrom` units.
  The runtime cannot infer training units; this is an operator assertion.
- Finite coordinates/outputs, exact output shapes, nonsingular right-handed
  row-vector cells, full periodicity or no periodicity. No partial PBC.
- No spin or missing type map; these fail at load. Frame/atom parameter
  dimensions are inspected once; nonzero dimensions require explicit values via
  `DeepMd::evaluate` or the standalone CLI. Plain `Potential::eval` returns an
  error for such models. No inferred defaults or broadcasting.
- Double input/output buffers; metadata remains `float64-io`. This cannot upgrade
  a model's hidden float32 constants to float64. Internal model precision remains
  model dependent. The public validation fixture has float64 constants.
- C API strings use `DP_DeleteChar`; the model is destroyed before its library.

## Run

```sh
cargo build --locked --manifest-path mlmd/Cargo.toml -p omnimd-deepmd -p omnimd-mlmd --bins
omnimd-deepmd request.json new-result.json
```

Output paths must not exist. Library/model paths must be absolute. Example for
the official Linux CUDA image (replace paths and the type map for your model):

```json
{
  "options": {
    "library": "/opt/deepmd-kit/lib/libdeepmd_c.so",
    "model": "/absolute/path/to/public-model.pb",
    "device": "cuda:0",
    "cuda": {
      "runtime": "/opt/deepmd-kit/lib/libcudart.so.12",
      "cupti": "/opt/deepmd-kit/lib/libcupti.so.12"
    },
    "type_map": ["O", "H"],
    "units": "EvAngstrom"
  },
  "frames": [{
    "positions": [[0,0,0], [0.9572,0,0], [-0.239,0.927,0]],
    "species": ["O", "H", "H"],
    "cell": null,
    "units": "EvAngstrom"
  }],
  "timing": {"warmup": 2, "iterations": 20}
}
```

For CPU, set device to cpu and omit cuda. Windows needs matching official DLLs
and available dependent libraries; CUDA compilation on Windows does not imply
that a compatible Windows DeepMD GPU runtime was numerically validated.

`timing` is optional and only supported by the standalone evaluator. It records
model loading, the initial pass, additional warmup passes, and steady repeated
passes separately. One model is retained throughout. Steady timing includes host
validation, packing, the combined C call, required transfers, CUDA tracing/context
synchronization and output validation. It excludes process startup, JSON input
parsing and result serialization. Small models can be slower on GPU. Compare the
same model/frames/runtime and threading settings; do not infer speedups from GPU
visibility or memory usage. Warmup must be 1..100, iterations 1..1000.

## Verification and sources

Mock tests cover ownership, one load/combined EFV, multi-frame parameter packing,
optional atomic outputs, error/shape/finite checks,
device JSON compatibility, inherited masks, cross-thread kernel evidence,
graph/capture rejection and context restoration. They do not prove physical
accuracy. See [CUDA validation](CUDA_VALIDATION.md) for actual executed tests and
remaining runtime limitations.

Upstream behavior was checked against [DeepPotTF initialization](https://github.com/deepmodeling/deepmd-kit/blob/v3.1.3/source/api_cc/src/DeepPotTF.cc)
and the [official C API](https://github.com/deepmodeling/deepmd-kit/blob/v3.1.3/source/api_c/include/c_api.h).
CUPTI layouts follow the installed NVIDIA CUDA 12.8 `cupti_callbacks.h`; no SDK
headers or runtime libraries are vendored. No driver, BIOS or security changes
are needed or performed by this crate.
