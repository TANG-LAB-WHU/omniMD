//! JSON batch evaluator. It starts a fresh CPU worker and loads one model.
use omnimd_deepmd::{DeepMd, Device, LoadOptions};
use omnimd_ml_contract::{Evaluation, Frame, ModelMetadata, Potential};
use serde::{Deserialize, Serialize};
use std::{error::Error, fs, io::Write, path::Path, process::Command};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    options: LoadOptions,
    frames: Vec<Frame>,
}

#[derive(Serialize)]
struct Response {
    metadata: ModelMetadata,
    evaluations: Vec<Evaluation>,
}

fn worker(input: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    let request: Request = serde_json::from_str(&fs::read_to_string(input)?)?;
    if request.frames.is_empty() {
        return Err("at least one frame is required".into());
    }
    for frame in &request.frames {
        frame.validate()?;
    }
    // Safe under the documented trust boundary: the parent launches this fresh
    // worker with all accelerator visibility disabled before loading any runtime.
    let mut model = unsafe { DeepMd::load(&request.options) }?;
    let evaluations = request
        .frames
        .iter()
        .map(|f| model.eval(f))
        .collect::<Result<_, _>>()?;
    let response = Response {
        metadata: model.metadata().clone(),
        evaluations,
    };
    let encoded = serde_json::to_vec_pretty(&response)?;
    // No partial numerical results are serialized on an evaluation error.
    // create_new prevents accidental overwrite of a user's prior result.
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    file.write_all(&encoded)?;
    file.write_all(b"\n")?;
    Ok(())
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() == 3 && args[0] == "--worker" {
        return worker(Path::new(&args[1]), Path::new(&args[2]));
    }
    if args.len() != 2 {
        return Err(
            "usage: omnimd-deepmd REQUEST.json NEW_OUTPUT.json (CPU worker; output must not exist)"
                .into(),
        );
    }
    let request: Request = serde_json::from_str(&fs::read_to_string(&args[0])?)?;
    if !matches!(request.options.device, Device::Cpu) {
        return Err(
            "only explicit Cpu is currently supported; CUDA requests are not silently downgraded"
                .into(),
        );
    }
    if Path::new(&args[1]).exists() {
        return Err("output already exists".into());
    }
    let status = Command::new(std::env::current_exe()?)
        .arg("--worker")
        .args(&args)
        .env("CUDA_VISIBLE_DEVICES", "-1")
        .env("HIP_VISIBLE_DEVICES", "-1")
        .env("ROCR_VISIBLE_DEVICES", "-1")
        .status()?;
    if !status.success() {
        return Err(format!("DeepMD worker failed: {status}").into());
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("omnimd-deepmd: {error}");
        std::process::exit(1);
    }
}
