//! Native readout or full supported-network training; A dataset JSONL v1.
use omnimd_ml_contract::Potential;
use omnimd_native::{
    full_training::{train_full_with_device, Checkpoint, FullConfig},
    indexed_dataset::IndexedDataset,
    training::{train_with_device, TrainConfig, TrainingDevice},
    training_data::{Dataset, DatasetAccess},
    NativePotential,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Deserialize, Default)]
#[serde(rename_all = "snake_case")]
enum DatasetMode {
    #[default]
    Memory,
    Indexed,
}

#[derive(Deserialize)]
#[serde(tag = "schema", deny_unknown_fields)]
enum Request {
    #[serde(rename = "omnimd-native-readout-train-v1")]
    Readout {
        initial_model: PathBuf,
        dataset: PathBuf,
        output_directory: PathBuf,
        config: TrainConfig,
        #[serde(default)]
        device: TrainingDevice,
    },
    #[serde(rename = "omnimd-native-full-train-v1")]
    Full {
        #[serde(default)]
        dataset_access: DatasetMode,
        initial_model: Option<PathBuf>,
        checkpoint: Option<PathBuf>,
        dataset: PathBuf,
        output_directory: PathBuf,
        config: FullConfig,
        #[serde(default)]
        device: TrainingDevice,
    },
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
// New files in a new output directory: no overwrite of an existing checkpoint.
fn save_checkpoint(
    output: &Path,
    name: &str,
    state: &Checkpoint,
) -> omnimd_ml_contract::Result<()> {
    let operation = || -> Result<(), Box<dyn std::error::Error>> {
        let bytes = serde_json::to_vec_pretty(state)?;
        let temp = output.join(format!("{name}.tmp"));
        let target = output.join(name);
        if target.exists() {
            return Err("checkpoint target already exists".into());
        }
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.write_all(&bytes)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        drop(file);
        fs::rename(temp, target)?;
        Ok(())
    };
    operation().map_err(|e| omnimd_ml_contract::Error(format!("save checkpoint: {e}")))
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let startup = std::time::Instant::now();
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 {
        return Err("usage: omnimd-native-train REQUEST.json".into());
    }
    let request_path = fs::canonicalize(&args[1])?;
    let request_bytes = fs::read(&request_path)?;
    let request: Request = serde_json::from_slice(&request_bytes)?;
    // Reject unsupported devices before dataset/checkpoint IO or output creation.
    match &request {
        Request::Readout { device, .. } => device.validate_readout()?,
        Request::Full { device, .. } => device.validate_full()?,
    }
    let base = request_path.parent().unwrap();
    let (dataset_path, output) = match &request {
        Request::Readout {
            dataset,
            output_directory,
            ..
        }
        | Request::Full {
            dataset,
            output_directory,
            ..
        } => (path(base, dataset), path(base, output_directory)),
    };
    if output.exists() {
        return Err("training output directory must not exist".into());
    }
    let use_index = matches!(
        &request,
        Request::Full {
            dataset_access: DatasetMode::Indexed,
            ..
        }
    );
    if matches!(&request,Request::Full{dataset_access:DatasetMode::Indexed,config,..} if config.minibatch.is_none())
    {
        return Err("indexed dataset access requires explicit minibatch configuration".into());
    }
    let dataset_open = std::time::Instant::now();
    let memory = if use_index {
        None
    } else {
        Some(Dataset::load(&dataset_path)?)
    };
    let indexed = if use_index {
        Some(IndexedDataset::open(&dataset_path)?)
    } else {
        None
    };
    let dataset: &dyn DatasetAccess = match (&memory, &indexed) {
        (Some(d), _) => d,
        (_, Some(d)) => d,
        _ => unreachable!(),
    };
    let dataset_open_ms = dataset_open.elapsed().as_secs_f64() * 1000.;
    let mut startup_timing = None;
    let mut final_checkpoint_io_ms = 0.;
    let (model, mut report, input_key, input_sha) = match request {
        Request::Readout {
            initial_model,
            config,
            device,
            ..
        } => {
            let bytes = fs::read(path(base, &initial_model))?;
            let initial = NativePotential::from_json(std::str::from_utf8(&bytes)?)?;
            let outcome =
                train_with_device(initial.model(), memory.as_ref().unwrap(), &config, device)?;
            (
                outcome.model,
                serde_json::to_value(outcome.report)?,
                "initial_model_sha256",
                sha(&bytes),
            )
        }
        Request::Full {
            initial_model,
            checkpoint,
            config,
            device,
            ..
        } => {
            let prepare = std::time::Instant::now();
            let (state, key, hash) = match (initial_model, checkpoint) {
                (Some(initial), None) => {
                    let bytes = fs::read(path(base, &initial))?;
                    let model = NativePotential::from_json(std::str::from_utf8(&bytes)?)?;
                    (
                        Checkpoint::start(model.model(), dataset, &config)?,
                        "initial_model_sha256",
                        sha(&bytes),
                    )
                }
                (None, Some(checkpoint)) => {
                    let file = path(base, &checkpoint);
                    if fs::metadata(&file)?.len() > 128 * 1024 * 1024 {
                        return Err("checkpoint exceeds 128 MiB".into());
                    }
                    let bytes = fs::read(file)?;
                    (
                        serde_json::from_slice::<Checkpoint>(&bytes)?,
                        "resume_checkpoint_sha256",
                        sha(&bytes),
                    )
                }
                _ => {
                    return Err(
                        "full training requires exactly one of initial_model or checkpoint".into(),
                    )
                }
            };
            let state_preparation_ms = prepare.elapsed().as_secs_f64() * 1000.;
            let before_trainer_ms = startup.elapsed().as_secs_f64() * 1000.;
            let outcome = train_full_with_device(state, dataset, &config, device, |s| {
                if !output.exists() {
                    fs::create_dir(&output).map_err(|e| {
                        omnimd_ml_contract::Error(format!("create checkpoint directory: {e}"))
                    })?;
                }
                save_checkpoint(&output, &format!("checkpoint-step-{}.json", s.step), s)
            })?;
            startup_timing = Some(serde_json::json!({
                "dataset_open_ms":dataset_open_ms,
                "state_preparation_ms":state_preparation_ms,
                "before_trainer_ms":before_trainer_ms,
                "before_first_update_ms":before_trainer_ms+outcome.report.timing.startup_ms
            }));
            let checkpoint_start = std::time::Instant::now();
            save_checkpoint(&output, "checkpoint.json", &outcome.checkpoint)?;
            final_checkpoint_io_ms = checkpoint_start.elapsed().as_secs_f64() * 1000.;
            (
                outcome.checkpoint.model,
                serde_json::to_value(outcome.report)?,
                key,
                hash,
            )
        }
    };
    let export_start = std::time::Instant::now();
    let model_json = serde_json::to_string_pretty(&model)? + "\n";
    let mut reloaded = NativePotential::from_json(&model_json)?;
    let mut fitted = NativePotential::new(model)?;
    let mut difference = [0.0_f64; 3];
    for i in 0..dataset.len() {
        let frame = dataset.frame(i)?;
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
    dataset.verify()?;
    let export_validation_ms = export_start.elapsed().as_secs_f64() * 1000.;
    let object = report.as_object_mut().unwrap();
    object.insert("output_timing".into(), serde_json::json!({"final_checkpoint_io_ms":final_checkpoint_io_ms, "export_validation_ms":export_validation_ms,"reload_checked_frames":dataset.len(),"reload_evaluations":2*dataset.len(),"before_output_write_ms":startup.elapsed().as_secs_f64()*1000.}));
    if let Some(timing) = startup_timing {
        object.insert("startup_timing".into(), timing);
    }
    if let Some(access) = dataset.access_report() {
        object.insert("dataset_access".into(), serde_json::to_value(access)?);
    }
    object.insert(input_key.into(), input_sha.into());
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
    if !output.exists() {
        fs::create_dir(&output)?;
    }
    fs::write(output.join("model.json"), model_json)?;
    fs::write(output.join("report.json"), report_json)?;
    println!(
        "{}",
        serde_json::json!({"model":output.join("model.json"),"report":output.join("report.json"),
        "scope":report["scope"],"data_rank":report["data_rank"],"solve_rank":report["solve_rank"],
        "step_end":report["step_end"],"final_training":report["final_training"],"final_validation":report["final_validation"]})
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("native training failed: {e}");
        std::process::exit(1);
    }
}
