# DeepMD feature coverage: reference version 3.1.3

Matrix revision 12, checked 2026-10-16. Previous integrated CUDA runtime:
`9b8d1847f892e32751c73d3262e0245d0efb4cdd`. This includes engine inspection/tutorial,
native source `b6b7aa7` and C API source `1aee39c`. Implementation and real-runtime
validation are separate. C API CPU follow-up is now verified in `ce27786`
(integrated as `87abf57`); the new engine parameter route has its own receipt. DeepMD 3.1.3 is a pinned comparison target, not a claim about the newest
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
| Frame/atomic parameters (`fparam`, `aparam`) | Missing | Explicit adapter/CLI inputs; engine dataset v2, NPY import, CPU batch and fixed-parameter MD | [Engine contract and evidence](engine/PARAMETERS.md); native/CUDA engine parameter routes and parameterized ensembles/minimization reject |
| Model-stored frame defaults | Missing | Explicit opt-in, confirmed runtime availability, one frame per call; atomic values remain explicit | PT CPU fixture verified; TF defaults rejected; model hash binds unexposed defaults |
| DPLR, range correction, linear/pair-potential combinations | Missing | No electrostatic/modifier/coupling orchestration; no acceptance claim | Dedicated model/physics interfaces required |
| Nonperiodic and triclinic all-axis PBC | Implemented, analytic image derivatives | Delegated after common cell/type validation | Partial PBC remains rejected |
| Atomic energy / atomic virial output | Internal values may exist; no shared public output | Optional adapter/standalone CLI outputs from the same combined call | TF/PT CPU comparisons passed; engine still exposes total E/F/V; new GPU paths untested |
| Precision | Native f64 arithmetic and parameters | Double IO only; internal precision is model-dependent | Neither backend can infer training units |
| Native JSON loading | Implemented with strict schema checks | Not applicable | No auto architecture or unit inference |
| DeepMD 3.1.3 `.dp` / safe YAML conversion | Partial, bounded serialized `se_e2_a` energy model | Not a C API conversion path | [Exact import rules](native/CONVERSION.md) |
| TensorFlow `.pb` conversion | Two exact audited graph profiles, finite replaceable parameters | Direct official runtime loading bypasses native conversion | [Frozen graph rules](native/FROZEN.md); arbitrary graphs reject natively |
| PyTorch / JAX / Paddle model formats | No native importer; no pickle loading | PT `.pth` CPU single-frame parameter/default fixture verified; other formats unverified; CUDA currently accepts TF `.pb` only | Do not infer support from file extension or symbols |
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
| DeepMD standard NPY data | Partial: explicitly ordered sets, C-order float64 arrays, fixed real type map, optional E/F/V | Optional v2 explicit fparam/aparam float64 arrays added; mixed types, spin, arbitrary dtypes/layouts and other formats remain unsupported |
| XYZ / extxyz | Plain XYZ and strict labeled extxyz profile, explicit units/PBC | General ASE metadata and stress conversion are not inferred |
| Dataset preflight | `inspect-dataset`: complete structural check, exact read-byte SHA256, counts, PBC and label coverage | Source hashes are declarations; model suitability and trainer limits are separate |
| Batch E/F/V assessment | Engine retains one model and streams single-frame calls. C API standalone CLI additionally packs homogeneous multi-frame calls | TF homogeneous C API batch/atomic CPU comparisons passed; engine retains one call per frame, including parameterized TF/PT |
| Dataset subset selection | Hash-bound explicit IDs; complete input validation; source-order frames, preserved IDs/labels, new contiguous indices and bundle receipt | [Scope and replay](engine/DATASET_SELECTION.md); 10000 IDs / 256 MiB subset bounds; no external labeling or automatic active learning |
| Model disagreement | CPU `model-deviation`: 2..16 retained models, absolute population E/atom, vector F and nine-component V/atom deviations, explicit candidate interval | [Scope and replay](engine/MODEL_DEVIATION.md); native CPU verified; C API ensemble route unverified; no automatic active-learning loop |
| External label return | Hash-bound job preparation and E/F/optional-V import; exact geometry/atom/parameter identity; fill missing labels, preserve source order/v2 | [Contract and replay](engine/EXTERNAL_LABELS.md); calculator/provenance declarations unverified; no job submission or automatic training |
| Readout training | CPU default; optional native CUDA E/F/V feature columns, CPU rank-aware QR and assessment, optional bias | Frozen hidden layers; bounded dense solve; [microcase slower on CUDA](native/fixtures/cuda-readout-report.json); separate full-network route below |
| Full supported-network training | CPU default; optional bounded CUDA all-neural E/F and optional V gradients and Adam, tanh/linear, checkpoint/restart | CPU inference/loss seeds/geometry/reduction/assessment retained; fixed normalization/cutoff/selection/atomic offsets; independent one-side copies; [CUDA bounds](native/CUDA_FULL_TRAINING.md) |
| CUDA gradient executor | Hyperdual default; explicit cached reverse accumulation, same shape/work bounds, 64 MiB planned device-buffer payload | [Activation and evidence](native/CUDA_REVERSE_TRAINING.md); full-batch reverse checkpoint v4 (minibatch v5), no executor switch on resume; payload excludes driver/context memory |
| Fixed training geometry | Explicit CUDA full-training cache, both gradient executors; default false; 64 MiB retained host payload; exact identity checks | [Contract](native/FIXED_GEOMETRY_TRAINING.md); fixed training frames only; cache may toggle on resume; excludes temporary/allocator memory, assessment and current-loss directions |
| Virial supervision | Optional full-network CPU/CUDA loss on all nine row-strain virial entries per atom; explicit positive weight/scale; loss-bound full-batch CPU/hyperdual v3 or reverse v4; minibatch v5 | [Contract](native/VIRIAL_TRAINING.md); E/F labels still required; readout QR remains E/F-only; no stress/volume conversion |
| Split/provenance | Seeded or explicit held-out IDs, dataset/source hashes, validation isolation | No trajectory-independent split guarantee from random frames |
| Deterministic minibatches | Explicit batch size/seed/order on CPU and both CUDA executors; actual-count tail normalization; v5 next-batch sampler and exact restart | [Contract](native/MINIBATCH_TRAINING.md); full-batch default; memory 1024 frames, explicit indexed access has separate bounds; no convergence equivalence claim |
| Indexed training input | Explicit indexed v1 JSONL with minibatches; framewise read, offset/ID/line SHA index, whole-file checks before publication; runtime storage-mode switching | [Bounds and evidence](native/INDEXED_TRAINING.md); max 100000 frames/16 GiB input, 256 KiB record, 64 MiB index payload; metadata/sampler O(N), 64 atoms unchanged; maximum ceiling not benchmarked |
| Generic DeepMD training options | Missing | Sharded/end-to-end GPU/distributed/multi-task training, learning-rate schedules, generic pretrained adaptation and broader architectures |
| Training monitoring | JSON reports and checkpoints | No TensorBoard integration or live monitoring service |
| Freeze/export | Native training writes native-v1 JSON, reload checked | No native export into generic official training checkpoints or `.pb` |
| Model information | `capabilities` static and GPU-free; `model-info` loads selected model | Metadata does not prove graph architecture, training units or GPU execution |
| Geometry minimization | Fixed-cell Armijo steepest descent, explicit force/energy criteria, geometry-only exports | Cell relaxation, other minimizers and production convergence validation |
| MD | Actual omniMD System, NVE Verlet and CSVR NVT, combined evaluation cache, exact same-environment restart | Fixed parameters added for CPU C API; PT restart is numerical, not bitwise, due to runtime first-call roundoff. NPT/constraints/domain decomposition/classical mixing remain missing |
| Third-party ecosystem | No new LAMMPS/ASE/i-PI/GROMACS interface provided by this runner | Official external integrations remain separate projects |

## Integrated scope and validation boundaries

- C API source `1aee39c` is integrated as `3be7687`: adapter-local batched E/F/V,
  explicit frame/atom parameters, optional atomic outputs and standalone
  inspection. All 17 library and four CLI tests pass on the integrated tree.
  Follow-up `ce27786` is integrated as `87abf57`: real CPU TF batch/parameters,
  PT explicit/defaults and atomic comparisons passed; 23 library and 5 CLI
  checks pass. No new C API GPU-path validation is claimed.
- Existing shared Frame/Evaluation/Potential schemas are unchanged. Engine MD and
  `batch-eval` retain single-frame E/F/V calls. Engine dataset v2 and fixed MD now
  carry CPU C API parameters; parameter-free records remain compatible. Use the separate
  [C API CLI guide](deepmd/BACKEND_GUIDE.md) and [engine guide](engine/PARAMETERS.md). Native inference
  has not acquired C API-only parameter/atomic/model capabilities.
- Native source `b6b7aa7` is integrated as `887c52d`. Final integrated GPU checks
  passed warmed same-model parity, old/new/CPU complete 20-step NVE, exact restart
  and a seven-frame candidate-cache motion/rebuild sequence. See the
  [integrated receipt](evidence/native-cuda-integrated-fit.json).
- The complete [CPU tutorial](evidence/cpu-tutorial-integrated.json) was rerun with
  the final feature-enabled engine selecting CPU explicitly: import, inspection,
  readout training, batch assessment, CSVR NVT and exact checkpoint continuation.

- Native readout source `dc5449e` is integrated as `65a7192`. CUDA computes feature
  columns only; QR/reference/assessment remain CPU. B recorded GPU parity, while
  A runs CPU-safe integration checks without a GPU context.

- Engine CPU model-deviation is committed as `304e818`; native-only CUDA readout
  integration is `65a7192`, with top-level scope update `4f41f15`. The subsequent
  hash-bound subset workflow is covered by [its receipt](evidence/dataset-selection.json),
  including CPU `train-native` on preserved existing labels. These new checks
  do not rerun GPU acceptance. The subsequent [engine parameter receipt](evidence/engine-parameters.json)
  records TF/PT CPU batch, explicit/default fixed MD and restart. TF replay is
  bitwise exact; PT has measured runtime first-call roundoff, with zero coordinate
  difference and maximum restart velocity difference 2.65e-23 Angstrom/fs.

- Native CUDA full-training source `f4dbf62` is integrated as `02b5457`. B records
  independent CPU-gradient parity, all 112 parameters trained, 300-step learning
  and exact 150+150 continuation. The 64-atom 3.46x ratio is a warmed gradient
  stage measurement; tiny whole training is slower. [B receipt](native/fixtures/cuda-full-training-report.json).
  Linux CUDA and physical training accuracy remain unverified. The final integrated
  [engine CLI receipt](evidence/native-cuda-full-integrated.json) records six steps,
  exact 3+3 model/moment continuation, all 112 parameters changed, zero reload
  E/F/V difference and rejected backend switching. Seven CPU-safe affected
  checks and the final release build passed.

- Optional virial-loss source `d12fdb5` is integrated as `7d3d472`, following the
  external-label workflow `69cdf12`. [Final short CLI acceptance](evidence/native-virial-integrated.json)
  records exact v3 3+3 continuation, six-step CPU/CUDA parameter/model parity,
  unchanged E/F CUDA-v2 continuation and rejected objective/backend changes.
  B's independent gradient and 300-step evidence remains separate. Tiny complete
  joint training was slower on CUDA (1816 ms vs CPU 619 ms); no acceleration claim.

- Native reverse-gradient source `f33d332` is integrated as `f4e794a` on `78cfd034`.
  It is explicitly selected with `config.cuda_gradient="reverse"`; hyperdual
  remains the default. [Final CLI acceptance](evidence/native-reverse-integrated.json)
  records six E/F/V steps, exact v4 3+3 model/moment continuation, CPU E/F/V
  agreement within 4.45e-16, zero reload difference, and rejected loss/device/
  executor changes. Five CPU-safe checks and the offline release build passed.
  [B's separate receipt](native/fixtures/cuda-reverse-training-report.json) records
  all-parameter derivatives, 300-step learning and rotated fresh-process timings:
  medium CPU/hyperdual/reverse 2963/1349/877 ms; tiny CPU/reverse 615/1573 ms.
  CPU primal/loss seeds, geometry, stable reduction and assessment remain. No
  architecture/dataset bound was raised; Linux CUDA remains unverified.

- Native fixed-geometry source `bbfecc6` is integrated as `a6a841d` on `94427ef`.
  [Final cache CLI acceptance](evidence/native-geometry-cache-integrated.json)
  records six reverse E/F/V steps, exact same-cache and both cache-toggle
  continuation, old uncached checkpoint compatibility, CPU parity and zero reload
  difference. Fifteen CPU-safe checks and the offline release build passed.
  [Native evidence](native/fixtures/fixed-geometry-training-report.json) separately
  covers both CUDA executors and bounded three-run timing medians. Cache arrays
  remain transient; fixed training frames only, unchanged numerical limits.

- Native minibatch source `3e7091c` is integrated as `5be7e49` on `b8ce8fa`.
  [Final engine acceptance](evidence/native-minibatch-integrated.json) records
  seven reverse E/F/V updates in batches 7,7,4,7,7,4,7; exact sampler agreement
  with an independent integer reference; mid-epoch/tail/cache-toggle restart;
  five refusal cases; CPU parity and exact old-v4 full-batch continuation.
  Ten CPU-safe checks and the locked offline release build passed. The
  [native receipt](native/fixtures/minibatch-training-report.json) separately
  covers both CUDA executors and actual v1-v4 150-to-300 continuation. No new
  performance result, streaming, larger dataset or physical accuracy is claimed.

- Native indexed-access source `be573fe` is integrated as `dc24b26` on `8f80ee5`.
  [Final engine acceptance](evidence/native-indexed-integrated.json) records
  seven indexed reverse updates and CPU comparison against actual prior memory
  checkpoints: exact model/Adam/sampler and initial/final train/validation metrics,
  identical exported model bytes, both storage/cache-switch directions and three
  refusal cases. Eleven CPU-safe checks and the offline release build passed.
  The [native capacity receipt](native/fixtures/indexed-training-report.json)
  separately covers both CUDA executors and the deliberately padded 2051-frame
  boundary input. That is not representative physical throughput; 100000 frames /
  16 GiB is the enforced ceiling, not a fully benchmarked operating point.

## Implementation order and acceptance gates

1. Engine preflight and reproducible executable workflow: delivered in this
   tranche, without a shared contract change. See [stepwise guide](GETTING_STARTED.md).
2. Native CUDA cooperative fitting is integrated and verified. Complete the
   separate GPU acceptance for any newly exposed C API parameter/batch path.
   CPU TF/PT acceptance is recorded. Keep CPU defaults and explicit devices.
3. Engine parameters use a versioned side context without changing shared Frame/EFV.
   New typed inputs/outputs (spin, tensor properties) still need versioned
   contracts, bounded validation, unsupported-input rejection and fixtures before
   engine exposure. A wrapper around an official API is labeled delegation.
4. Broader native descriptors/training and distributed simulation require actual
   implementations and independent derivative, parity and restart tests. These
   are substantial future milestones, not enabled flags.

No percentage of complete DeepMD parity is claimed. Update the pinned baseline,
source version and evidence links together when the implemented scope changes.
