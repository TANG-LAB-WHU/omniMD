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
        "model_deviation":{"entry":"model-deviation","device":"cpu","members":"2..16","statistics":"absolute population E/atom, force vector and virial/atom deviation","selection":"explicit max-force interval; marks candidates only","runtime_scope":"native CPU tested; parameter-free C API route requires compatible external runtime"},
        "training":{"entry":"train-native","scope":["frozen-hidden final-readout E+F fit","full supported tanh network E+F Adam"],"backend":"native float64; CPU default","cuda":{"frozen_readout_features":true,"requires_sibling_trainer_cuda_feature":true,"solver":"CPU QR","assessment":"CPU","full_network":false,"performance":"recorded 3/4-atom feature microcase slower than CPU"},"full_network":true,"full_network_scope":"native-v1 tanh hidden / linear output; fixed normalization, cutoff, type-map, selection and atomic offsets; reference-size bounds","request_schemas":["omnimd-native-readout-train-v1","omnimd-native-full-train-v1"]},
        "minimization":{"method":"bounded steepest descent","fixed_cell":true,"force_and_energy_criteria":true},
        "dynamics":{"ensembles":["nve","nvt_csvr"],"fixed_cell":true,"units":"EvAngstromFsU","dataset_frame_input":true},
        "unsupported":["full-network CUDA training","CAPI CUDA graphs/stream capture","mixed-type/virtual atoms","partial PBC","NPT/barostat","generic DeepMD training","full-network GPU/minibatch/virial-loss training"],
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
