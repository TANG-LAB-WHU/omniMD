//! Data CLI commands use the same model loader and device-isolated worker as dynamics.
use super::BackendOptions;
use omnimd_ml_contract::Potential;
use omnimd_mlmd::dataset::{self, DatasetReader, ImportRequest, Metrics};
use serde::Deserialize;
use serde_json::json;
use std::{
    error::Error,
    ffi::OsString,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EvaluationRequest {
    options: BackendOptions,
    dataset: PathBuf,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelRequest {
    options: BackendOptions,
}

fn capabilities() -> serde_json::Value {
    json!({
        "schema":"omnimd-capabilities-v1",
        "device":["Cpu",{"Cuda":"explicit index; CAPI requires CUDA/CUPTI, native requires native-cuda feature and Driver/NVRTC"}],
        "evaluation":{"outputs":["energy","forces","virial"],"units":"EvAngstrom","one_model_load_per_request":true},
        "backends":{
            "native":{"architecture":"bounded se_e2_a format v1","precision":"float64","runtime_dependencies":"none for CPU","default_device":"Cpu","cuda_compiled":cfg!(feature="native-cuda"),"cuda_scope":"tanh/linear inference only; CPU geometry and force/virial reduction; installed NVIDIA Driver/NVRTC required; no automatic selection"},
            "deepmd_c_api":{"precision":"float64-io","internal_weight_precision":"model dependent","runtime_dependencies":"trusted official DeepMD library","default_device":"Cpu","cuda_scope":"official TensorFlow .pb inference with matching CUDA/CUPTI; verified direct kernels and context synchronization; some operations may remain on CPU","spin":false,"frame_atom_parameters":{"adapter_and_standalone_cli":true,"engine_commands":["CPU batch-eval","CPU fixed-parameter run/check"],"real_model_validation":"CPU TF explicit and PT explicit/model_default fixtures; see engine-parameters receipt"},"atomic_outputs":{"adapter_and_standalone_cli":true,"engine_commands":false,"real_runtime_validation":"CPU TF/PT fixture comparisons passed; no new GPU validation"},"multi_frame_c_api":{"adapter_and_standalone_cli":true,"engine_batch_eval":"one call per frame","real_runtime_validation":"CPU TF batching passed; .pth/model_default single-frame only; no new GPU validation"}}
        },
        "cuda_validation":{"kind":"recorded fixture numerical agreement; not a probe of this host","native_evidence":["mlmd/native/fixtures/cuda-validation-report.json","mlmd/evidence/native-cuda-cli.json"],"deepmd_c_api_evidence":"mlmd/deepmd/validation/cuda-public-six-atom.json","performance":"bounded measurements only; small models were slower on both GPU paths; no general speedup claim"},
        "parameters":{"engine_device":"cpu","backend":"official DeepMD C API","md":"fixed values/policy and immutable atom order; checkpoint v3","defaults":"explicit opt-in with confirmed model availability; never atomic defaults","unsupported":["native parameter models","CUDA engine parameter route","parameterized model-deviation/minimization/native training"]},
        "datasets":{"schema":dataset::SCHEMA,"parameter_schema":"omnimd-dataset-jsonl-v2","parameter_import":"explicit optional parameter_layout plus float64 fparam.npy/aparam.npy; no broadcast/default inference","import":["DeepMD standard NPY C-order float64","XYZ","bounded extxyz profile"],"explicit_sets":true,"partial_labels":true,"training":false,"automatic_split":false,"inspection":"inspect-dataset; full structural validation, streamed-byte SHA256 and E/F/V coverage; no model or GPU load"},
        "dataset_selection":{"entry":"select-dataset","expected_source_sha256":"required","output":"new bundle: dataset.jsonl, request.json, receipt.json","order":"source order; IDs preserved, contiguous new indices","bounds":"10000 IDs; 4 MiB request; 256 MiB selected JSONL","labels":"preserved including absence; none generated"},
        "external_labels":{"commands":["prepare-labels","import-labels"],"identity":"source SHA256, frame ID/index, exact geometry/atom/parameter context","merge":"fill absent E/F/optional virial; exact existing labels only","output":"new bundle and receipt; source order and dataset v2 parameters preserved","calculator_launched":false},
        "model_deviation":{"entry":"model-deviation","device":"cpu","members":"2..16","statistics":"absolute population E/atom, force vector and virial/atom deviation","selection":"explicit max-force interval; marks candidates only","runtime_scope":"native CPU tested; parameter-free C API route requires compatible external runtime"},
        "training":{"entry":"train-native","scope":["frozen-hidden final-readout E+F fit","full supported tanh network E+F Adam"],"backend":"native float64; CPU default","cuda":{"frozen_readout_features":true,"requires_sibling_trainer_cuda_feature":true,"solver":"CPU QR","assessment":"CPU","full_network":true,"full_network_scope":"bounded FP64 all-neural E/F gradients and Adam; CPU geometry/reduction/assessment; optional tiled GPU E/F/V prediction and loss seeds","gradient_executors":{"hyperdual":"default","reverse":"explicit opt-in; framewise by default"},"tiling":{"config":"cuda_tiling with max_frames/max_atoms and optional max_atoms_per_frame; explicit reverse CUDA only","default":"absent/null retains framewise and historical v1-v6","max_frames":32,"max_atoms_per_tile":1024,"max_atoms_per_frame":256,"default_max_atoms_per_frame":64,"frame_admission":"explicit 64..256 ceiling; default/CPU/untiled remain 64; actual capacity also limited by unchanged work/payload budgets","device_buffer_payload_limit_bytes":67108864,"host_tile_payload_limit_bytes":33554432,"planning":"effective atom capacity reduced by 50M work and payload limits; no frame split; payload is not total memory","gradient_kernels_per_tile":4,"gpu_loss_kernels_per_tile":10,"optimizer":"batch normalization/order and one Adam per optimizer batch preserved","checkpoint":"default CPU loss v7, GPU loss v8; max_atoms_per_frame>64 selects v9 and GPU-loss backend v2; immutable frame/tile capacities/mode; storage/cache switches allowed"},"loss_preparation":{"config":"cuda_loss=gpu requires reverse CUDA and explicit tiling","default":"absent or cpu preserves historical state/PTX","gpu_scope":"E/F/V predictions, detached seeds and reused forward intermediates; ten kernels per tile","cpu_remaining":"data/geometry/scalars, stable gradient reduction/clipping, assessment/export","checkpoint":"v8 by default; explicit frame ceiling>64 selects v9 and cuda_tiled_device_loss_f64_v2; loss mode and all prior immutable identity bound","performance":"128-update synthetic median3 tiny112param605->756ms slower; medium466param3506->3061ms 1.15x; no universal speedup"},"large_frames":{"config":"cuda_tiling.max_atoms_per_frame=64..256; omitted/64 preserves historical state/PTX","requires":"explicit CUDA reverse tiling; CPU/untiled limit64 unchanged","admission":"all dataset sizes before context/cache; all batch sizes before tile allocation/launch; no splitting/truncation","limits":"50M estimated work,64MiB device,32MiB host tile payload unchanged;466-parameter example effective138 rejects256","gpu_loss":"strided gather covers all1+3N+9 outputs; backend cuda_tiled_device_loss_f64_v2","checkpoint":"v9 when configured frame ceiling>64; capacity immutable","platform":"new v9 Windows acceptance only; prior Linux06439f1 v7/v8 does not cover v9","performance":"one12-update synthetic process CPU-lossCUDA1554ms/GPU-loss1721ms; no speed claim"},"fixed_geometry_cache":{"config":"fixed_geometry_cache=true; default false","executors":["hyperdual","reverse"],"scope":"immutable selected training frames; current model/loss directions rebuilt each step","budget":"64 MiB retained host payload; excludes allocator overhead and temporary buffers","identity":"exact geometry mismatch invalidates pool and stops","restart":"runtime flag may toggle; numerical checkpoint identity unchanged","cpu_only":false},"checkpoint":"CPU E/F v1; CUDA hyperdual E/F v2; virial CPU/hyperdual v3; reverse full-batch v4; explicit minibatch v5 binds sampler/loss/executor/device/driver/PTX; explicit schedule v6; explicit tiling v7; GPU loss v8; expanded frame ceiling v9 supersedes these versions while retaining their checks","performance":"bounded 466-parameter/64-atom/100-step full-process medians CPU2963ms/hyperdual1349ms/reverse877ms; 112-parameter tiny reverse slower than CPU; no general speedup claim"},"virial_loss":{"full_network":true,"readout":false,"default":"disabled","config":"optional positive weight/scale_ev_per_atom","normalization":"all nine row-strain virial components per atom; no volume factor","checkpoint":"enabled: CPU/hyperdual v3, reverse v4; immutable loss contract; schedule v6, tiling v7, GPU loss v8 or expanded frame ceiling v9; absent options retain historical versions"},"dataset_access":{"default":"memory","indexed":{"config":"top-level dataset_access=indexed; explicit minibatch required","frame_limit":100000,"file_limit_bytes":17179869184u64,"record_limit_bytes":262144,"index_payload_limit_bytes":67108864,"frames_decoded_per_read":1,"metadata_memory":"O(frame count); index payload is not process memory","integrity":"exact per-line hash and whole-file verification before checkpoints/publication","restart":"runtime access mode may switch when both input limits hold"}},"minibatch":{"config":"optional explicit batch_size, seed and order; absent/null means full batch","backends":["cpu","cuda hyperdual","cuda reverse"],"checkpoint":"v5 with validated next-batch sampler; explicit schedule v6; tiling v7; GPU loss v8; expanded frame ceiling v9","normalization":"actual batch count; tail neither dropped nor filled","steps":"additional optimizer updates","default_memory_frame_limit":1024,"indexed_frame_limit":100000,"indexed_requires_explicit_access":true},"learning_rate":{"config":"optional learning_rate_schedule; base is learning_rate","default":"absent/null preserves constant behavior and historical v1-v5","kinds":["constant","exponential"],"formula":"base * decay_rate^floor(k/decay_steps); k completed optimizer updates","checkpoint":"explicit schedule v6, tiling v7, GPU loss v8 or expanded frame ceiling v9; base and schedule immutable on resume","scope":"full neural training only; host f64 scalar shared by CPU and CUDA Adam","limits":"decay_steps 1..1000000000; decay_rate finite (0,1]; used rates must remain positive finite"},"full_network":true,"full_network_scope":"native-v1 tanh hidden / linear output; fixed normalization, cutoff, type-map, selection and atomic offsets; reference-size bounds","request_schemas":["omnimd-native-readout-train-v1","omnimd-native-full-train-v1"]},
        "minimization":{"method":"bounded steepest descent","fixed_cell":true,"force_and_energy_criteria":true},
        "dynamics":{"ensembles":["nve","nvt_csvr"],"fixed_cell":true,"units":"EvAngstromFsU","dataset_frame_input":true},
        "unsupported":["CAPI CUDA graphs/stream capture","mixed-type/virtual atoms","partial PBC","NPT/barostat","generic DeepMD training","end-to-end GPU/unbounded streaming/distributed training"],
        "virial_convention":"W_ab=-dE/d epsilon_ab; r'=r(I+epsilon), all nine entries, eV; not stress"
    })
}
fn model_info(input: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    let request: ModelRequest = serde_json::from_str(&fs::read_to_string(input)?)?;
    request.options.validate()?;
    let hash = dataset::sha256(request.options.model_path())?;
    let model = request.options.load()?;
    let library_hash = match &request.options {
        BackendOptions::DeepMd(options) => Some(dataset::sha256(&options.library)?),
        BackendOptions::Native(_) => None,
    };
    let mut writer = dataset::new_output(output)?;
    serde_json::to_writer_pretty(
        &mut writer,
        &json!({
            "schema":"omnimd-model-info-v1", "model_sha256":hash,
            "library_sha256":library_hash, "metadata":model.metadata(),
            "model_details":match &model {super::Backend::DeepMd(m)=>Some(m.details()),_=>None},
            "device":request.options.device(), "outputs":["energy","forces","virial"],
            "placement_attestation":"CAPI CUDA checks observed kernels during evaluation; native CUDA executes explicit kernels with CPU geometry/reduction. Model-info alone does not evaluate or establish numerical parity.",
            "virial_convention":"MinusDEpsilonRow", "geometry":"nonperiodic or all-axis periodic row lattice",
            "architecture_attestation":"native format validated; C API architecture is not inferred from metadata"
        }),
    )?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}
fn evaluate(input: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    let request: EvaluationRequest = serde_json::from_str(&fs::read_to_string(input)?)?;
    request.options.validate()?;
    let dataset_hash = dataset::sha256(&request.dataset)?;
    let model_hash = dataset::sha256(request.options.model_path())?;
    let mut reader = DatasetReader::open(&request.dataset)?;
    super::parameter_backend::validate_request(
        &request.options,
        reader.header.parameter_layout.as_ref(),
    )?;
    let mut model = request.options.load()?;
    if let (super::Backend::DeepMd(model), Some(layout)) = (&model, &reader.header.parameter_layout)
    {
        super::parameter_backend::validate_layout(model, layout)?;
    }
    let mut writer = dataset::new_output(output)?;
    dataset::write_line(
        &mut writer,
        &json!({
            "kind":"header","schema":if reader.header.parameter_layout.is_some() {"omnimd-predictions-jsonl-v2"} else {"omnimd-predictions-jsonl-v1"},
            "dataset_sha256":dataset_hash,"model_sha256":model_hash,
            "metadata":model.metadata(),"dataset_header":reader.header,
            "metrics_policy":"frame-weighted E and E/atom; pooled scalar F and nine virial components; no fitted offsets"
        }),
    )?;
    let mut metrics = Metrics::default();
    let mut frames = 0u64;
    while let Some(data) = reader.next_frame()? {
        let prediction = match (
            &mut model,
            &reader.header.parameter_layout,
            &data.parameters,
        ) {
            (super::Backend::DeepMd(model), Some(layout), Some(values)) => {
                super::parameter_backend::evaluate(model, &data.frame, layout, values)?
            }
            (model, None, None) => model.eval(&data.frame)?,
            _ => return Err("unsupported parameterized batch configuration".into()),
        };
        metrics.add(&prediction, &data.labels, data.frame.positions.len())?;
        let mut record = json!({
            "kind":"prediction","index":data.index,"id":data.id,"natoms":data.frame.positions.len(),
            "evaluation":prediction,"labels":data.labels
        });
        if let Some(parameters) = data.parameters {
            record["parameters"] = serde_json::to_value(parameters)?;
        }
        dataset::write_line(&mut writer, &record)?;
        frames += 1;
    }
    dataset::write_line(
        &mut writer,
        &json!({"kind":"completed","frames":frames,"metrics":metrics.report(),"metadata":model.metadata()}),
    )?;
    writer.flush()?;
    Ok(())
}
pub fn dispatch(args: &[OsString]) -> Result<bool, Box<dyn Error>> {
    if args.len() == 2 && args[0] == "train-native" {
        let executable = std::env::current_exe()?.with_file_name(if cfg!(windows) {
            "omnimd-native-train.exe"
        } else {
            "omnimd-native-train"
        });
        if !executable.is_file() {
            return Err(
                "build omnimd-native binaries beside omnimd-mlmd before train-native".into(),
            );
        }
        let status = Command::new(executable).arg(&args[1]).status()?;
        if !status.success() {
            return Err(format!("native training failed: {status}").into());
        }
        return Ok(true);
    }
    if args.len() == 1 && args[0] == "capabilities" {
        println!("{}", serde_json::to_string_pretty(&capabilities())?);
        return Ok(true);
    }
    if args.len() == 3 && (args[0] == "prepare-labels" || args[0] == "import-labels") {
        let request = Path::new(&args[1]);
        let output = Path::new(&args[2]);
        let report = if args[0] == "prepare-labels" {
            omnimd_mlmd::labeling::prepare_labels(request, output)?
        } else {
            omnimd_mlmd::labeling::import_labels(request, output)?
        };
        println!("Completed label bundle for {} frames; verify receipt.json and recorded hashes before reuse.",report.frames);
        return Ok(true);
    }
    if args.len() == 3 && args[0] == "select-dataset" {
        let report =
            omnimd_mlmd::selection::select_dataset(Path::new(&args[1]), Path::new(&args[2]))?;
        println!("Selected {} of {} frames; verify receipt.json and dataset/request hashes in the new bundle.", report.selected_frames, report.input_frames);
        return Ok(true);
    }
    if args.len() == 3 && args[0] == "inspect-dataset" {
        let output = Path::new(&args[2]);
        if output.exists() {
            return Err("output already exists".into());
        }
        let report = dataset::inspect_dataset(Path::new(&args[1]))?;
        let mut writer = dataset::new_output(output)?;
        serde_json::to_writer_pretty(&mut writer, &report)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        return Ok(true);
    }
    if args.len() == 3 && args[0] == "import-xyz" {
        let request: omnimd_mlmd::xyz::XyzRequest =
            serde_json::from_str(&fs::read_to_string(&args[1])?)?;
        let frames = omnimd_mlmd::xyz::import_xyz(&request, Path::new(&args[2]))?;
        println!("Imported {frames} XYZ frames with explicit format/units/PBC.");
        return Ok(true);
    }
    if args.len() == 3 && args[0] == "import-deepmd" {
        let request: ImportRequest = serde_json::from_str(&fs::read_to_string(&args[1])?)?;
        let frames = dataset::import_deepmd(&request, Path::new(&args[2]))?;
        println!("Imported {frames} frames; units/PBC/sets are explicit; no training or split performed.");
        return Ok(true);
    }
    if args.len() == 4 && args[0] == "--data-worker" {
        match args[1].to_str() {
            Some("minimize") => {
                super::minimize_command::worker(Path::new(&args[2]), Path::new(&args[3]))?
            }
            Some("batch-eval") => evaluate(Path::new(&args[2]), Path::new(&args[3]))?,
            Some("model-info") => model_info(Path::new(&args[2]), Path::new(&args[3]))?,
            _ => return Err("unknown data worker command".into()),
        }
        return Ok(true);
    }
    if args.len() == 3
        && (args[0] == "batch-eval" || args[0] == "model-info" || args[0] == "minimize")
    {
        if Path::new(&args[2]).exists() {
            return Err("output already exists".into());
        }
        // All three request schemas share options; workers still validate their full schema.
        let request: serde_json::Value = serde_json::from_str(&fs::read_to_string(&args[1])?)?;
        let options: BackendOptions = serde_json::from_value(
            request
                .get("options")
                .ok_or("request requires options")?
                .clone(),
        )?;
        options.validate()?;
        let mut command = Command::new(std::env::current_exe()?);
        command.arg("--data-worker").args(args);
        options.device().configure_worker(&mut command)?;
        let status = command.status()?;
        if !status.success() {
            return Err(format!(
                "model worker failed: {status}; inspect output status (failed predictions lack completed; nonconverged minimization reports its reason)"
            )
            .into());
        }
        return Ok(true);
    }
    Ok(false)
}
