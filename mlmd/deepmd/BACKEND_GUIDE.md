# DeepMD backend: step-by-step integration

This guide describes the `omnimd-deepmd` adapter and its standalone CLI. The
numerical model is executed by the official DeepMD runtime through its C API.
Training, automatic differentiation and descriptors remain in that runtime.
The separate `mlmd/native` implementation has its own supported model subset;
loading a model through this adapter does not convert it to native Rust.

## 1. Choose a compatible model and runtime

Start with a nonspin energy model frozen as TensorFlow `.pb`. This is the
numerically exercised route: DeepMD 3.1.3 CPU and DeepMD 3.1.0/TensorFlow 2.18.0
CUDA, using a public fixture. The adapter uses the `DP_DeepPotCompute2` ABI subset
present in C API v26. Official DeepMD v3.2.0 has newer APIs; this does not mean
that every new model family works through this subset.

| Artifact or capability | Status in this adapter |
| --- | --- |
| TensorFlow energy `.pb`, CPU | Established runtime-tested route |
| TensorFlow energy `.pb`, CUDA | Explicit opt-in; single-frame public-model validation recorded in [CUDA_VALIDATION.md](CUDA_VALIDATION.md) |
| Nonzero frame/atom parameters | Explicit per-frame inputs through this CLI or `DeepMd::evaluate`; focused FFI tests, real parameterized-model validation pending |
| Multi-frame C API and atomic E/virial | Implemented; focused FFI tests; this revision's real-runtime validation status is in [BACKEND_VALIDATION.md](BACKEND_VALIDATION.md) |
| PyTorch `.pth`; newer exportable `.pt2`; Paddle `.json`/`.pdmodel` | CPU load can be delegated to an installed compatible C runtime; unvalidated here, not a compatibility promise |
| `.savedmodel` / `.savedmodeltf` directories | Rejected by the current file-only loader |
| Python-only formats, checkpoints, `.pte` | Not an advertised C API route; freeze to an explicitly supported runtime format first |
| Spin, charge/spin conditioning, tensor/DOS/Hessian outputs | Not implemented by the EFV adapter; spin and available nonzero charge/spin metadata are rejected |
| Training, fine-tuning, compression, model-deviation ensembles | Official external tools, not Rust adapter operations |

The [official C++ format dispatcher](https://github.com/deepmodeling/deepmd-kit/blob/v3.2.0/source/api_cc/src/common.cc)
and [backend documentation](https://docs.deepmodeling.com/projects/deepmd/en/v3.2.0/backend.html)
are different compatibility boundaries. For example, Python lists `.pte`, while
the v3.2.0 C++ dispatcher does not. Runtime backend plugins and their dependencies
must exist; a filename suffix cannot install or enable them.

Record the model hash, runtime version, training units, ordered type map,
parameter meaning and internal precision. `float64-io` means double host input
and output arrays; it does not change a model's trained precision. omniMD accepts
`EvAngstrom`: energy in eV, position in angstrom, force in eV/angstrom and total
virial in eV. Parameter units and normalization must match training data.

## 2. Install or select the official runtime

Reuse a known compatible installation when possible. The Rust adapter build
needs neither DeepMD nor a CUDA SDK; execution needs a trusted shared C library
and its dependencies.

For a new Linux runtime, select a pinned version from the
[official releases](https://github.com/deepmodeling/deepmd-kit/releases) and follow
[the official C library installation instructions](https://docs.deepmodeling.com/projects/deepmd/en/v3.2.0/install/install-from-c-library.html).
The published archive is named `libdeepmd_c.tar.gz`; the current prebuilt package
has its own GLIBC/CUDA requirements and excludes some runtime dependencies.
Use a CPU-compatible distribution/build for CPU-only hosts. Do not infer that a
Python wheel contains the C API library. If using PyTorch, match the required
libtorch version and make its library directory discoverable.

For an existing Linux installation such as `/opt/deepmd-kit`:

```sh
export DEEPMD_ROOT=/opt/deepmd-kit
export PATH="$DEEPMD_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$DEEPMD_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
test -f "$DEEPMD_ROOT/lib/libdeepmd_c.so"
dp --version
```

For Windows, use a compatible official C API DLL build and put its dependencies
on `PATH` before launching the CLI. Build success alone does not verify a
Windows DeepMD runtime. The bundled Python wheel's `deepmd.dll` is not by itself
evidence that the required C symbols are available.

## 3. Build the standalone evaluator

From the omniMD repository root:

```sh
cargo build --release --locked --manifest-path mlmd/Cargo.toml -p omnimd-deepmd --bin omnimd-deepmd
```

The default output is `mlmd/target/release/omnimd-deepmd` (`.exe` on Windows).
`CARGO_TARGET_DIR` overrides this location. Add the output directory to `PATH`
or substitute the full executable path in subsequent commands. CPU remains the
default when the `device` field is omitted.

## 4. Inspect a model once before inference

Create `options.cpu.json`, substituting real absolute paths and the model's
exact type order. This file contains `LoadOptions` directly, without an outer
`options` key:

```json
{
  "library": "/opt/deepmd-kit/lib/libdeepmd_c.so",
  "model": "/absolute/path/to/model.pb",
  "device": "cpu",
  "type_map": ["O", "H"],
  "units": "EvAngstrom"
}
```

Run:

```sh
omnimd-deepmd inspect options.cpu.json new-model-info.json
```

The isolated worker loads one model, queries its metadata, and exits without
inference. The response contains `metadata` (cutoff, type map, units and host
precision) and `model_details.frame_parameter_dim` / `atom_parameter_dim`.
A mismatch in the requested type order fails explicitly; do not fix it by
reordering positions without also preserving species and parameter rows.
Inspection is not an accuracy check or evidence of CUDA kernels executing.

## 5. Evaluate frames with one persistent model

Create `request.cpu.json`. This three-atom geometry illustrates the input schema;
use a model and geometry appropriate for your system.

```json
{
  "options": {
    "library": "/opt/deepmd-kit/lib/libdeepmd_c.so",
    "model": "/absolute/path/to/model.pb",
    "device": "cpu",
    "type_map": ["O", "H"],
    "units": "EvAngstrom"
  },
  "frames": [
    {
      "positions": [[0,0,0], [0.9572,0,0], [-0.239,0.927,0]],
      "species": ["O", "H", "H"],
      "cell": null,
      "units": "EvAngstrom"
    },
    {
      "positions": [[0,0,0], [0.96,0,0], [-0.239,0.93,0]],
      "species": ["O", "H", "H"],
      "cell": null,
      "units": "EvAngstrom"
    }
  ],
  "batch_size": 2,
  "atomic": true,
  "timing": {"warmup": 2, "iterations": 5}
}
```

```sh
omnimd-deepmd request.cpu.json new-result.json
```

`evaluations[i]` contains energy, forces and the full 3-by-3 total virial for
input frame `i`. `atomic_evaluations[i]` is present only with `atomic:true` and
contains `energies[N]` and `virials[N][3][3]` in input atom order. These are the
runtime's atomic decomposition, not an independent second evaluation.
Atomic energies and virials have eV units; no volume division is applied.

All output paths must be new, with a writable parent directory on a filesystem
supporting hard links (for example NTFS or ext4). Complete JSON is published via
a sibling temporary file and an atomic no-clobber hard link. Unsupported output
filesystems fail explicitly. JSON, geometry, parameter or runtime errors return
a nonzero status; no successful numerical result is published for a failed
request. The worker retains one model for first, warmup and measured passes.

## 6. Set batching and parameter inputs deliberately

`batch_size` defaults to 1, preserving the earlier serial behavior and allowing
heterogeneous frame collections. A value greater than 1 groups consecutive
frames into a single `DP_DeepPotCompute2` call. Within each group, atom count,
ordered species and periodicity must match. Cells and coordinates may differ.
The last group may be smaller. Use `batch_size:1` or explicitly partition inputs
when these conditions are not satisfied. A larger batch is not guaranteed faster.

Cells are either `null` (nonperiodic) or three right-handed, nonsingular lattice
row vectors (fully periodic). Mixed or partial periodic boundary conditions
are not supported. The implementation validates C integer products and limits
requested scratch-array values to 256 MiB per call; reduce the batch size when
that bound is exceeded. This is not a bound on total runtime/model memory.

If inspection reports `frame_parameter_dim=2` and `atom_parameter_dim=1`, add
one parameter object per frame to the preceding two-frame request:

```json
"parameters": [
  {"fparam": [300.0, 1.0], "aparam": [[0.1], [0.2], [0.3]]},
  {"fparam": [600.0, 2.0], "aparam": [[0.4], [0.5], [0.6]]}
]
```

These numbers are shape examples, not universal temperature or charge settings.
The first dimension of `aparam` follows the frame's original atom order, not
the type-map order. Values must be finite. A zero dimension requires an empty
array (or omit that key). If both dimensions are zero, omit `parameters`.
Nonzero dimensions always require explicit values, even if a newer runtime
could provide model defaults. The adapter performs no broadcasting, zero
filling, unit conversion or semantic guessing.

## 7. Connect official training to inference

Run training in the official DeepMD environment, outside the Rust evaluator.
Prepare a valid DeepMD dataset and training configuration for the intended
energy model, including its ordered type map, units and parameter definitions.
Use separate training and validation systems. The upstream examples are useful
for checking the toolchain, not establishing production model accuracy.
See the [official training workflow](https://docs.deepmodeling.com/projects/deepmd/en/v3.2.0/train/training.html).

For the TensorFlow graph route, from your training directory:

```sh
dp --tf train input.json
dp --tf freeze -o model.pb
dp test -m model.pb -s /absolute/path/to/validation-system -n 4
```

For a compatible PyTorch TorchScript installation the upstream commands are
`dp --pt train input.json` and `dp --pt freeze -o model.pth`; this adapter's
PyTorch runtime path remains unvalidated. Multi-task checkpoints require
selecting the intended branch during freezing. Keep the training checkpoint
for resuming training, and supply the frozen inference artifact to omniMD.
[Freezing](https://docs.deepmodeling.com/projects/deepmd/en/v3.2.0/freeze/freeze.html)
and [official model testing](https://docs.deepmodeling.com/projects/deepmd/en/v3.2.0/test/test.html)
are separate steps.

After the official model test passes, update `options.cpu.json`, run inspection,
and evaluate a small set of the same validation geometries. Compare E, every
force component and all nine virial components against the official runtime.
Use tolerances appropriate for the model precision. Only then connect the model
to the main omniMD workflow. This adapter does not schedule training, download
models, select loss functions or convert arbitrary models into native weights.

## 8. Connect to Rust and the shared MD interface

In a fresh child process, configure device visibility with
`Device::configure_worker` before loading any accelerator/runtime. Then load the
trusted runtime once with `unsafe { DeepMd::load(&options) }` and retain the
handle. The unsafe boundary is native library trust, not input validation.

For ordinary parameter-free models, existing `Potential::eval(&Frame)` remains
compatible with the engine's MD/evaluation interface. For parameterized or
atomic/batched work, use the adapter-local method:

```rust,ignore
let details = model.details();
// Supply one EvalParameters object per frame when dimensions are nonzero.
let outputs = model.evaluate(&frames, Some(&parameters), true)?;
```

`outputs[i].evaluation` is the shared `Evaluation`; `outputs[i].atomic` is the
optional decomposition. The shared `Frame` and `Potential` contracts were not
extended in this tranche. Consequently parameterized models require the
standalone CLI or a caller of this adapter method; existing engine MD commands
cannot supply these extra parameters. Calling plain `eval` on such a model
returns an actionable error. A handle is deliberately neither `Send` nor `Sync`.

## 9. Opt into CUDA and interpret timing

Follow [README.md](README.md#devices-and-isolation) for GPU index/mask behavior.
Change `device` to `cuda:0` and provide matching absolute `cuda.runtime` and
`cuda.cupti` paths. CUDA requests currently require `.pb`, direct CUDA kernels
and a working CUPTI subscriber. The adapter checks actual launch callbacks and
synchronizes their contexts for each combined C API call, including a batch.
It does not claim every model operation ran on GPU. There is no automatic CPU
fallback; an explicit new CPU request is the fallback action.

Timing separates model loading, the first pass, extra warmup passes and steady
passes. It includes host packing/validation, the combined runtime call,
transfers and CUDA tracing/synchronization, and excludes process launch and JSON
serialization. `steady_evaluations` counts frames, while `steady_capi_calls`
counts batch calls. `seconds_per_evaluation` is time per frame.
Use equal frames, model, runtime, thread configuration, atomic flag and batch
size for CPU/GPU comparison. The recorded six-atom case was slower on GPU; no
acceleration factor is promised. New batch/parameter/atomic paths have not yet
been revalidated on GPU in this tranche.

## 10. Resolve common errors

| Error or symptom | Next step |
| --- | --- |
| Library cannot load / required symbol missing | Check absolute path, matching C API build and dependent library search paths; a Python-only install is insufficient |
| Model path must be an existing file | Supply the frozen file; SavedModel directories are currently unsupported |
| Requested type order differs | Use the model's exact ordered chemical-element map |
| Model requires explicit parameters | Inspect dimensions and provide one correctly shaped parameter object per frame |
| `fparam` or `aparam` shape/finite error | Correct lengths, atom order and finite values before running again |
| Batch requires identical inputs | Reduce `batch_size` to 1 or partition frames by atom count, species order and periodicity |
| Missing/nonfinite total or atomic output | Treat the request as failed; validate the model/runtime with official tools |
| CUDA error, no kernels, graphs/capture or tracing unavailable | Preserve the error; choose an explicit CPU request or resolve the runtime issue; do not label CPU execution a GPU success |
| Output exists or cannot be published | Choose a new writable output path; existing results are never overwritten |

The underlying buffers and shared-contract conventions follow
[the official C wrapper](https://github.com/deepmodeling/deepmd-kit/blob/v3.2.0/source/api_c/src/c_api.cc)
and [C API declarations](https://github.com/deepmodeling/deepmd-kit/blob/v3.2.0/source/api_c/include/c_api.h).
No additional SDK libraries are vendored by this tranche.
