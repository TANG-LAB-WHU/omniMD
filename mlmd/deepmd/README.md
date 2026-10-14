# Official DeepMD C API backend

This Rust crate loads an explicitly supplied trusted DeepMD C library at runtime.
It loads one model per `DeepMd` instance, then evaluates all frames on that handle.
No Python, LibTorch or DeepMD install is needed to compile or run the mock tests.
A compatible official runtime and frozen model are needed for real inference.

The implementation follows DeePMD-kit v3.1.3 C API v26. It resolves the required
symbols and rejects missing ones. A successful symbol lookup is not full ABI
validation: only use a trusted matching official build. C API strings are
released by DP_DeleteChar; the model is released before unloading its library.
No header or runtime is vendored.

## Supported scope

- A single frame per C call; every call returns total E/F/virial together.
- Angstrom, eV, eV/angstrom, eV virial. The options must explicitly declare units.
  The C API cannot infer a model's training units; this is an operator assertion.
- Strict ordered model type map check, symbol-based frame mapping, finite input
  and output checks, right-handed row-vector cells, all-periodic or nonperiodic.
- No frame/atomic parameters, spin model or missing type map. These fail at load.
- CPU device only. CUDA requests fail with a diagnostic. The official API
  can silently pick/fall back to devices and offers no portable placement query;
  a negative rank is NOT a CPU guarantee.
- The CLI starts a fresh worker with CUDA/HIP/ROCR visibility disabled, scoped
  to that child only. The library API requires the same initialization order.
- Double C API buffers and output. This does not convert a float32 frozen model
  into a float64 model. Use a model exported with float64 precision if that is
  required; metadata reports `float64-io`, not a claim about its hidden weights.
- This is inference through FFI, not pure Rust inference or native GPU support.

## Run

Build from the nested workspace:

```sh
cargo build --manifest-path mlmd/Cargo.toml -p omnimd-deepmd
cargo run --manifest-path mlmd/Cargo.toml -p omnimd-deepmd -- request.json result.json
```

The output path must not exist. Runtime messages go to the console; only complete
validated results are written to the output JSON. Paths must be absolute.
Request example (replace paths and map with your public model):

```json
{
  "options": {
    "library": "/opt/deepmd/lib/libdeepmd_c.so",
    "model": "/absolute/path/to/public-model.pb",
    "device": "Cpu",
    "type_map": ["O", "H"],
    "units": "EvAngstrom"
  },
  "frames": [{
    "positions": [[0,0,0], [0.9572,0,0], [-0.239,0.927,0]],
    "species": ["O", "H", "H"],
    "cell": null,
    "units": "EvAngstrom"
  }]
}
```

Windows uses a matching official DLL and Windows JSON paths with escaped
backslashes. Dependent runtime libraries must be available to the process.
No driver changes, global PATH changes or production model downloads are made.

## Validation limits

Mock C functions verify ownership, data order, error propagation, combined
evaluation and failure paths. They are not a DeepMD numerical parity test.
A public float64 model plus a compatible runtime is required to verify actual
numerics, device placement and frozen-model precision end to end.
See the parent workspace validation report for what was actually executed.
