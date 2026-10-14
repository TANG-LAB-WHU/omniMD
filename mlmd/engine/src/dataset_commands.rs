//! Data CLI commands use the same model loader and fresh CPU worker as dynamics.
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
        "device":["Cpu"],
        "evaluation":{"outputs":["energy","forces","virial"],"units":"EvAngstrom","one_model_load_per_request":true},
        "backends":{
            "native":{"architecture":"bounded se_e2_a format v1","precision":"float64","runtime_dependencies":"none"},
            "deepmd_c_api":{"precision":"float64-io","internal_weight_precision":"model dependent","runtime_dependencies":"trusted official DeepMD library","extra_fparam_aparam_spin":false}
        },
        "datasets":{"schema":dataset::SCHEMA,"import":["DeepMD standard NPY C-order float64","XYZ","bounded extxyz profile"],"explicit_sets":true,"partial_labels":true,"training":false,"automatic_split":false},
        "training":{"entry":"train-native","scope":["frozen-hidden final-readout E+F fit","full supported tanh network E+F Adam"],"backend":"native CPU float64","full_network":true,"full_network_scope":"native-v1 tanh hidden / linear output; fixed normalization, cutoff, type-map, selection and atomic offsets; reference-size bounds","request_schemas":["omnimd-native-readout-train-v1","omnimd-native-full-train-v1"]},
        "minimization":{"method":"bounded steepest descent","fixed_cell":true,"force_and_energy_criteria":true},
        "dynamics":{"ensembles":["nve","nvt_csvr"],"fixed_cell":true,"units":"EvAngstromFsU","dataset_frame_input":true},
        "unsupported":["GPU","mixed-type/virtual atoms","partial PBC","NPT/barostat","generic DeepMD training","GPU/minibatch/virial-loss training"],
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
            "device":"Cpu", "outputs":["energy","forces","virial"],
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
    let mut model = request.options.load()?;
    let mut writer = dataset::new_output(output)?;
    dataset::write_line(
        &mut writer,
        &json!({
            "kind":"header","schema":"omnimd-predictions-jsonl-v1",
            "dataset_sha256":dataset_hash,"model_sha256":model_hash,
            "metadata":model.metadata(),"dataset_header":reader.header,
            "metrics_policy":"frame-weighted E and E/atom; pooled scalar F and nine virial components; no fitted offsets"
        }),
    )?;
    let mut metrics = Metrics::default();
    let mut frames = 0u64;
    while let Some(data) = reader.next_frame()? {
        let prediction = model.eval(&data.frame)?;
        metrics.add(&prediction, &data.labels, data.frame.positions.len())?;
        dataset::write_line(
            &mut writer,
            &json!({
                "kind":"prediction","index":data.index,"id":data.id,"natoms":data.frame.positions.len(),
                "evaluation":prediction,"labels":data.labels
            }),
        )?;
        frames += 1;
    }
    dataset::write_line(
        &mut writer,
        &json!({"kind":"completed","frames":frames,"metrics":metrics.report()}),
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
        let status = Command::new(std::env::current_exe()?)
            .arg("--data-worker")
            .args(args)
            .env("CUDA_VISIBLE_DEVICES", "-1")
            .env("HIP_VISIBLE_DEVICES", "-1")
            .env("ROCR_VISIBLE_DEVICES", "-1")
            .status()?;
        if !status.success() {
            return Err(format!(
                "CPU worker failed: {status}; inspect output status (failed predictions lack completed; nonconverged minimization reports its reason)"
            )
            .into());
        }
        return Ok(true);
    }
    Ok(false)
}
