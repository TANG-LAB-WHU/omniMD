# DeepMD feature coverage: reference version 3.1.3

Matrix revision 2, checked 2026-10-16. Tested integrated runtime:
`9b8d1847f892e32751c73d3262e0245d0efb4cdd`. This includes engine inspection/tutorial,
native source `b6b7aa7` and C API source `1aee39c`. Implementation and real-runtime
validation are separate: the new C API numerical follow-up is still pending. DeepMD 3.1.3 is a pinned comparison target, not a claim about the newest
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
| Frame/atomic parameters (`fparam`, `aparam`) | Missing | Implemented through adapter `evaluate` and standalone CLI, explicit dimensions/values, no broadcasting | FFI packing tests pass; real parameterized-model validation pending. Engine MD/batch cannot supply them |
| DPLR, range correction, linear/pair-potential combinations | Missing | No electrostatic/modifier/coupling orchestration; no acceptance claim | Dedicated model/physics interfaces required |
| Nonperiodic and triclinic all-axis PBC | Implemented, analytic image derivatives | Delegated after common cell/type validation | Partial PBC remains rejected |
| Atomic energy / atomic virial output | Internal values may exist; no shared public output | Optional adapter/standalone CLI outputs from the same combined call | Mock shape/finite/order tests pass; real-runtime follow-up pending; engine still exposes total E/F/V |
| Precision | Native f64 arithmetic and parameters | Double IO only; internal precision is model-dependent | Neither backend can infer training units |
| Native JSON loading | Implemented with strict schema checks | Not applicable | No auto architecture or unit inference |
| DeepMD 3.1.3 `.dp` / safe YAML conversion | Partial, bounded serialized `se_e2_a` energy model | Not a C API conversion path | [Exact import rules](native/CONVERSION.md) |
| TensorFlow `.pb` conversion | Two exact audited graph profiles, finite replaceable parameters | Direct official runtime loading bypasses native conversion | [Frozen graph rules](native/FROZEN.md); arbitrary graphs reject natively |
| PyTorch / JAX / Paddle model formats | No native importer; no pickle loading | CPU library-dependent and unverified here; CUDA currently accepts TF `.pb` only | Do not infer support from file extension or symbols |
| Model compression / tabulation | Missing | Runtime-dependent, unverified compressed models | [Official compression](https://docs.deepmodeling.com/projects/deepmd/en/v3.1.3/freeze/compress.html) is external |
| CUDA inference | Partial: NN/descriptor/pullback kernels; CPU geometry and final force/virial reduction | Partial: official TF kernels observed through CUPTI, synchronized per evaluation | [Native](native/CUDA.md), [C API](deepmd/CUDA_VALIDATION.md); no automatic fallback |
| CUDA cache / profiling | Conservative CPU candidate cache, cooperative per-atom fitting, retained host/device buffers and type uploads, segmented timing | Runtime manages its internal memory/work | [Final integrated evidence](evidence/native-cuda-integrated-fit.json): warmed and whole-MD improvements reported separately, limited to measured cases |
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
| Batch E/F/V assessment | Engine retains one model and streams single-frame calls. C API standalone CLI additionally packs homogeneous multi-frame calls | New C API call layout has mock tests; real-runtime follow-up pending. No ensemble-deviation/active-learning workflow |
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

## Integrated scope and validation boundaries

- C API source `1aee39c` is integrated as `3be7687`: adapter-local batched E/F/V,
  explicit frame/atom parameters, optional atomic outputs and standalone
  inspection. All 17 library and four CLI tests pass on the integrated tree.
  The new real-runtime batch/atomic/parameter numerical follow-up is pending;
  the older single-frame CPU/GPU receipts do not validate those new paths.
- Existing shared Frame/Evaluation/Potential schemas are unchanged. Engine MD and
  `batch-eval` still call parameter-free single-frame E/F/V. Use the separate
  [C API CLI guide](deepmd/BACKEND_GUIDE.md) for its new options. Native inference
  has not acquired C API-only parameter/atomic/model capabilities.
- Native source `b6b7aa7` is integrated as `887c52d`. Final integrated GPU checks
  passed warmed same-model parity, old/new/CPU complete 20-step NVE, exact restart
  and a seven-frame candidate-cache motion/rebuild sequence. See the
  [integrated receipt](evidence/native-cuda-integrated-fit.json).
- The complete [CPU tutorial](evidence/cpu-tutorial-integrated.json) was rerun with
  the final feature-enabled engine selecting CPU explicitly: import, inspection,
  readout training, batch assessment, CSVR NVT and exact checkpoint continuation.

## Implementation order and acceptance gates

1. Engine preflight and reproducible executable workflow: delivered in this
   tranche, without a shared contract change. See [stepwise guide](GETTING_STARTED.md).
2. Native CUDA cooperative fitting is integrated and verified. Complete the
   separate C API real-runtime batch/atomic/parameter acceptance before claiming
   numerical support for those new paths. Keep CPU defaults and explicit devices.
3. New shared typed inputs/outputs (parameters, spin, tensor properties) need versioned
   contracts, bounded validation, unsupported-input rejection and fixtures before
   engine exposure. A wrapper around an official API is labeled delegation.
4. Broader native descriptors/training and distributed simulation require actual
   implementations and independent derivative, parity and restart tests. These
   are substantial future milestones, not enabled flags.

No percentage of complete DeepMD parity is claimed. Update the pinned baseline,
source version and evidence links together when the implemented scope changes.
