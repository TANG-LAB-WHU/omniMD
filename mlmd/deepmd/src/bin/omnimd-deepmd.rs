//! JSON batch evaluator. It starts a fresh device-isolated worker and loads one model.
use omnimd_deepmd::{DeepMd, LoadOptions};
use omnimd_ml_contract::{Evaluation, Frame, ModelMetadata, Potential};
use serde::{Deserialize, Serialize};
use std::{error::Error, fs, io::Write, path::Path, process::Command, time::Instant};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    options: LoadOptions,
    frames: Vec<Frame>,
    timing: Option<TimingRequest>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TimingRequest {
    warmup: usize,
    iterations: usize,
}

#[derive(Serialize)]
struct Response {
    metadata: ModelMetadata,
    evaluations: Vec<Evaluation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cuda_kernel_launches: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timing: Option<serde_json::Value>,
}

fn worker(input: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    let request: Request = serde_json::from_str(&fs::read_to_string(input)?)?;
    if request.frames.is_empty() {
        return Err("at least one frame is required".into());
    }
    for frame in &request.frames {
        frame.validate()?;
    }
    if let Some(timing) = &request.timing {
        if timing.warmup == 0
            || timing.iterations == 0
            || timing.warmup > 100
            || timing.iterations > 1000
        {
            return Err("timing requires warmup=1..100 and iterations=1..1000".into());
        }
    }
    // Safe under the documented trust boundary: the parent launches this fresh
    // worker with explicit device visibility before loading any runtime.
    let start = Instant::now();
    let mut model = unsafe { DeepMd::load(&request.options) }?;
    let load_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let evaluations = request
        .frames
        .iter()
        .map(|f| model.eval(f))
        .collect::<Result<_, _>>()?;
    let first_pass_seconds = start.elapsed().as_secs_f64();
    let timing = if let Some(timing) = request.timing {
        let start = Instant::now();
        for _ in 0..timing.warmup {
            for frame in &request.frames {
                model.eval(frame)?;
            }
        }
        let warmup_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        for _ in 0..timing.iterations {
            for frame in &request.frames {
                model.eval(frame)?;
            }
        }
        let steady_seconds = start.elapsed().as_secs_f64();
        Some(
            serde_json::json!({"load_seconds":load_seconds,"first_pass_seconds":first_pass_seconds,
            "warmup_seconds":warmup_seconds,"warmup_passes":timing.warmup,"steady_seconds":steady_seconds,
            "steady_passes":timing.iterations,"steady_evaluations":timing.iterations * request.frames.len(),
            "seconds_per_evaluation":steady_seconds / (timing.iterations * request.frames.len()) as f64,
            "scope":"host validation, packing, combined CAPI EFV, transfers, GPU attestation/synchronization and output validation; excludes process startup and JSON serialization"}),
        )
    } else {
        None
    };
    let response = Response {
        metadata: model.metadata().clone(),
        evaluations,
        cuda_kernel_launches: model.cuda_kernel_launches(),
        timing,
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
            "usage: omnimd-deepmd REQUEST.json NEW_OUTPUT.json (isolated worker; output must not exist)"
                .into(),
        );
    }
    let request: Request = serde_json::from_str(&fs::read_to_string(&args[0])?)?;
    if Path::new(&args[1]).exists() {
        return Err("output already exists".into());
    }
    let mut command = Command::new(std::env::current_exe()?);
    command.arg("--worker").args(&args);
    request.options.device.configure_worker(&mut command)?;
    let status = command.status()?;
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
