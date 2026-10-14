//! Train only final affine readouts; consume A's dataset JSONL v1 unchanged.
use omnimd_ml_contract::Potential;
use omnimd_native::{
    training::{train, TrainConfig},
    training_data::Dataset,
    NativePotential,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: String,
    initial_model: PathBuf,
    dataset: PathBuf,
    output_directory: PathBuf,
    config: TrainConfig,
}
fn path(base: &Path, value: &Path) -> PathBuf {
    if value.is_absolute() {
        value.to_path_buf()
    } else {
        base.join(value)
    }
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 {
        return Err("usage: omnimd-native-train REQUEST.json".into());
    }
    let request_path = fs::canonicalize(&args[1])?;
    let request_bytes = fs::read(&request_path)?;
    let request: Request = serde_json::from_slice(&request_bytes)?;
    if request.schema != "omnimd-native-readout-train-v1" {
        return Err("unsupported training request schema".into());
    }
    let base = request_path.parent().unwrap();
    let initial_path = path(base, &request.initial_model);
    let dataset_path = path(base, &request.dataset);
    let output = path(base, &request.output_directory);
    if output.exists() {
        return Err("training output directory must not exist".into());
    }
    let initial_bytes = fs::read(&initial_path)?;
    let initial = NativePotential::from_json(std::str::from_utf8(&initial_bytes)?)?;
    let dataset = Dataset::load(dataset_path)?;
    let outcome = train(initial.model(), &dataset, &request.config)?;
    let model_json = serde_json::to_string_pretty(&outcome.model)? + "\n";
    let mut reloaded = NativePotential::from_json(&model_json)?;
    let mut fitted = NativePotential::new(outcome.model)?;
    let mut difference = [0.0_f64; 3];
    for frame in &dataset.frames {
        let a = fitted.eval(&frame.frame)?;
        let b = reloaded.eval(&frame.frame)?;
        difference[0] = difference[0].max((a.energy - b.energy).abs());
        for (x, y) in a.forces.iter().flatten().zip(b.forces.iter().flatten()) {
            difference[1] = difference[1].max((x - y).abs());
        }
        for (x, y) in a.virial.iter().flatten().zip(b.virial.iter().flatten()) {
            difference[2] = difference[2].max((x - y).abs());
        }
    }
    if difference.iter().any(|v| *v > 1e-12) {
        return Err("native-v1 serialization/reload mismatch exceeds 1e-12".into());
    }
    let mut report = serde_json::to_value(outcome.report)?;
    let object = report.as_object_mut().unwrap();
    object.insert("initial_model_sha256".into(), sha(&initial_bytes).into());
    object.insert(
        "output_model_sha256".into(),
        sha(model_json.as_bytes()).into(),
    );
    object.insert("request_sha256".into(), sha(&request_bytes).into());
    object.insert(
        "native_v1_reload_max_abs_energy_force_virial".into(),
        serde_json::to_value(difference)?,
    );
    let report_json = serde_json::to_string_pretty(&report)? + "\n";
    fs::create_dir(&output)?;
    fs::write(output.join("model.json"), model_json)?;
    fs::write(output.join("report.json"), report_json)?;
    println!(
        "{}",
        serde_json::json!({"model":output.join("model.json"),"report":output.join("report.json"),
        "scope":report["scope"],"data_rank":report["data_rank"],"solve_rank":report["solve_rank"],
        "final_training":report["final_training"],"final_validation":report["final_validation"]})
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("readout training failed: {e}");
        std::process::exit(1);
    }
}
