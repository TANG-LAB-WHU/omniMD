# DeepMD feature coverage: reference version 3.1.3

Matrix revision 1, checked 2026-10-16. Runtime baseline: local candidate
`7be1d976654f1a3c58ded4f69b2c9759cb9b3cbf`; this tranche adds engine dataset
inspection and an executable CPU tutorial. Pending work in other branches is
excluded. DeepMD 3.1.3 is a pinned comparison target, not a claim about the newest
release. Recorded C API CPU tests use 3.1.3; recorded C API CUDA tests use 3.1.0.
The native model format version, DeepMD package version, C API version and frozen
graph version are different identifiers.

**omniMD does not implement all DeepMD features.** Native means arithmetic in
this Rust/CUDA implementation. Delegated means the official external runtime
performs inference through FFI. Partial means a documented bounded subset.
Missing means no implementation in this MLIP runner. External tools listed below
are not functionality implemented or automatically invoked by omniMD.

## Models and inference

Official comparison sources: [model catalogue](https://docs.deepmodeling.com/projects/deepmd/en/v3.1.3/model/index.html),
[backend formats and differences](https://docs.deepmodeling.com/projects/deepmd/en/v3.1.3/backend.html),
and [C/C++ interfaces](https://docs.deepmodeling.com/projects/deepmd/en/v3.1.3/inference/cxx.html).
The official project's own backend feature sets differ; a descriptor name alone
is insufficient evidence of compatibility.

| Feature | Native Rust / optional native CUDA | Official backend through this adapter | Evidence or remaining gate |
| --- | --- | --- | --- |
| Scalar total energy, forces, virial | Implemented for supported native-v1 models; CPU f64, optional CUDA subset | Delegated combined `DP_DeepPotCompute2`, one model load | [Contract](CONTRACT.md), [CPU validation](VALIDATION.md) |
| `se_e2_a` energy model | Partial: explicit normalization, cutoff, selection, residual and fitting restrictions | Delegated for accepted runtime/model pair | [Native scope](native/README.md), [conversion](native/CONVERSION.md) |
| `se_e2_r`, `se_e3`, hybrid, attention, DPA-2/3 | Missing | No adapter-side architecture implementation; an ordinary parameter-free DeepPot may be delegated if the runtime supports it, but these architectures have not been accepted by a fixture matrix here | Native implementation and per-architecture parity remain open |
| Type embedding, masked descriptors, `se_e3_tebd` | Missing | Unverified runtime-dependent scope; no mixed-type/virtual-atom input route | New typed input or model fixture coverage required |
| Spin, dipole/polarizability, DOS, other tensor/property outputs, Hessian | Missing | Missing: this wrapper uses DeepPot E/F/V, not DeepSpin/DeepTensor or Hessian APIs | Shared typed outputs/input extensions required |
| Frame/atomic parameters (`fparam`, `aparam`) | Missing | Rejected at load in this baseline | C worker owns upcoming work; shared schema change must be coordinated |
| DPLR, range correction, linear/pair-potential combinations | Missing | No electrostatic/modifier/coupling orchestration; no acceptance claim | Dedicated model/physics interfaces required |
| Nonperiodic and triclinic all-axis PBC | Implemented, analytic image derivatives | Delegated after common cell/type validation | Partial PBC remains rejected |
| Atomic energy / atomic virial output | Internal values may exist; no shared public output | Not requested/exposed | Total E/F/V only |
| Precision | Native f64 arithmetic and parameters | Double IO only; internal precision is model-dependent | Neither backend can infer training units |
| Native JSON loading | Implemented with strict schema checks | Not applicable | No auto architecture or unit inference |
| DeepMD 3.1.3 `.dp` / safe YAML conversion | Partial, bounded serialized `se_e2_a` energy model | Not a C API conversion path | [Exact import rules](native/CONVERSION.md) |
| TensorFlow `.pb` conversion | Two exact audited graph profiles, finite replaceable parameters | Direct official runtime loading bypasses native conversion | [Frozen graph rules](native/FROZEN.md); arbitrary graphs reject natively |
| PyTorch / JAX / Paddle model formats | No native importer; no pickle loading | CPU library-dependent and unverified here; CUDA currently accepts TF `.pb` only | Do not infer support from file extension or symbols |
| Model compression / tabulation | Missing | Runtime-dependent, unverified compressed models | [Official compression](https://docs.deepmodeling.com/projects/deepmd/en/v3.1.3/freeze/compress.html) is external |
| CUDA inference | Partial: NN/descriptor/pullback kernels; CPU geometry and final force/virial reduction | Partial: official TF kernels observed through CUPTI, synchronized per evaluation | [Native](native/CUDA.md), [C API](deepmd/CUDA_VALIDATION.md); no automatic fallback |
| CUDA cache / profiling | Conservative CPU candidate cache, retained weights/buffers and segmented timing | Runtime manages its internal memory/work | [Final candidate evidence](evidence/native-cuda-infra-final.json); whole short MD can still be slower than CPU |
| CUDA graphs, multi-GPU, MPI/domain decomposition | Missing | Graph/stream capture rejected; no distributed adapter | Single selected device per fresh worker |

## Data, training and workflows

Official comparison sources: [data formats](https://docs.deepmodeling.com/projects/deepmd/en/v3.1.3/data/index.html),
[training catalogue](https://docs.deepmodeling.com/projects/deepmd/en/v3.1.3/train/index.html)
and [command-line interface](https://docs.deepmodeling.com/projects/deepmd/en/v3.1.3/cli.html).
The native trainer is a separate implementation with its own request schema;
an official `input.json` is not a native training request.

| Feature | Current omniMD implementation | What remains |
| --- | --- | --- |
| DeepMD standard NPY data | Partial: explicitly ordered sets, C-order float64 arrays, fixed real type map, optional E/F/V | Mixed types, spin/parameters, arbitrary dtypes/layouts and other formats |
| XYZ / extxyz | Plain XYZ and strict labeled extxyz profile, explicit units/PBC | General ASE metadata and stress conversion are not inferred |
| Dataset preflight | `inspect-dataset`: complete structural check, exact read-byte SHA256, counts, PBC and label coverage | Source hashes are declarations; model suitability and trainer limits are separate |
| Batch E/F/V assessment | One model retained, streamed frames, actual labeled counts, no fitted energy offset | No model-ensemble deviation/active-learning workflow |
| Readout training | CPU native final linear coefficients, optional bias, exact E/F features and rank-aware QR | Frozen hidden layers; bounded dense solve |
| Full supported-network training | CPU tanh/linear network, all neural weights/biases/timesteps, exact mixed E/F derivatives, full-batch Adam, checkpoint/restart | Fixed normalization/cutoff/selection/atomic offsets; small-model limits; duplicated one-side weights not tied |
| Split/provenance | Seeded or explicit held-out IDs, dataset/source hashes, validation isolation | No trajectory-independent split guarantee from random frames |
| Generic DeepMD training options | Missing | GPU/minibatch/distributed/multi-task training, learning-rate schedules, virial loss, pretrained fine-tuning and broader architectures |
| Training monitoring | JSON reports and checkpoints | No TensorBoard integration or live monitoring service |
| Freeze/export | Native training writes native-v1 JSON, reload checked | No native export into generic official training checkpoints or `.pb` |
| Model information | `capabilities` static and GPU-free; `model-info` loads selected model | Metadata does not prove graph architecture, training units or GPU execution |
| Geometry minimization | Fixed-cell Armijo steepest descent, explicit force/energy criteria, geometry-only exports | Cell relaxation, other minimizers and production convergence validation |
| MD | Actual omniMD System, NVE Verlet and CSVR NVT, combined evaluation cache, exact same-environment restart | NPT, constraints, domain decomposition, classical mixing and full legacy pipeline bridge |
| Third-party ecosystem | No new LAMMPS/ASE/i-PI/GROMACS interface provided by this runner | Official external integrations remain separate projects |

## Concurrent branches, not accepted in this snapshot

- The C API branch has implemented adapter-level multi-frame E/F/V in one call,
  explicit frame/atom parameters and optional per-atom outputs. Layout and strict
  error checks are in progress, CPU-only. Integration must verify single-frame
  compatibility and typed batch/parameter/output contracts before engine exposure.
  Current engine `batch-eval` still streams single-frame calls.
- The native branch is changing fitting/descriptor kernels to one cooperative
  block per atom, preserving scalar summation order and reusing host buffers.
  CPU/GPU parity and complete timing evidence must precede an accepted claim.
- No code from either concurrent branch is included in this tranche; `native/**`,
  `deepmd/**` and shared Frame/Evaluation schemas are unchanged here.

## Implementation order and acceptance gates

1. Engine preflight and reproducible executable workflow: delivered in this
   tranche, without a shared contract change. See [stepwise guide](GETTING_STARTED.md).
2. Native CUDA kernel/host pipeline and C API parameter/architecture extensions:
   separate owners; add rows as implemented only after integration and same-model
   E/F/V evidence. Keep CPU defaults and explicit device/runtime selection.
3. New typed inputs/outputs (parameters, spin, tensor properties) need versioned
   contracts, bounded validation, unsupported-input rejection and fixtures before
   engine exposure. A wrapper around an official API is labeled delegation.
4. Broader native descriptors/training and distributed simulation require actual
   implementations and independent derivative, parity and restart tests. These
   are substantial future milestones, not enabled flags.

No percentage of complete DeepMD parity is claimed. Update the pinned baseline,
source version and evidence links together when the implemented scope changes.
