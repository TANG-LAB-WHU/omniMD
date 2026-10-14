use omnimd_deepmd::{DeepMd, Device, LoadOptions};
use omnimd_ml_contract::{Frame, Potential};
use omnimd_mlmd::{system_from_frame, Checkpoint, MlipDynamics};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs,
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeOptions {
    native_model: PathBuf,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum BackendOptions {
    DeepMd(LoadOptions),
    Native(NativeOptions),
}

enum Backend {
    DeepMd(DeepMd),
    Native(omnimd_native::NativePotential),
}

impl BackendOptions {
    fn model_path(&self) -> &Path {
        match self {
            Self::DeepMd(options) => &options.model,
            Self::Native(options) => &options.native_model,
        }
    }

    fn validate(&self) -> Result<(), Box<dyn Error>> {
        if !self.model_path().is_absolute() || !self.model_path().is_file() {
            return Err("model path must be an existing absolute file path".into());
        }
        if let Self::DeepMd(options) = self {
            if !matches!(options.device, Device::Cpu) {
                return Err("only Cpu is supported".into());
            }
        }
        Ok(())
    }

    fn load(&self) -> Result<Backend, Box<dyn Error>> {
        self.validate()?;
        match self {
            // The CLI accepts only a trusted official DeepMD C API library.
            Self::DeepMd(options) => Ok(Backend::DeepMd(unsafe { DeepMd::load(options) }?)),
            Self::Native(options) => Ok(Backend::Native(omnimd_native::NativePotential::load(
                &options.native_model,
            )?)),
        }
    }
}

impl Potential for Backend {
    fn metadata(&self) -> &omnimd_ml_contract::ModelMetadata {
        match self {
            Self::DeepMd(model) => model.metadata(),
            Self::Native(model) => model.metadata(),
        }
    }
    fn eval(
        &mut self,
        frame: &Frame,
    ) -> omnimd_ml_contract::Result<omnimd_ml_contract::Evaluation> {
        match self {
            Self::DeepMd(model) => model.eval(frame),
            Self::Native(model) => model.eval(frame),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    options: BackendOptions,
    frame: Option<Frame>,
    masses: Option<Vec<f64>>,
    velocities: Option<Vec<[f64; 3]>>,
    restart: Option<PathBuf>,
    timestep_fs: f64,
    steps: u64,
    sample_every: u64,
}

fn model_hash(path: &Path) -> Result<String, Box<dyn Error>> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn checkpoint_path(output: &Path) -> PathBuf {
    let mut name = output.as_os_str().to_os_string();
    name.push(".checkpoint.json");
    PathBuf::from(name)
}

fn write_snapshot<P: Potential>(
    writer: &mut impl Write,
    md: &MlipDynamics<P>,
) -> Result<(), Box<dyn Error>> {
    let velocities: Vec<[f64; 3]> = md
        .system()
        .particles()
        .velocity
        .iter()
        .map(|v| [v[0], v[1], v[2]])
        .collect();
    serde_json::to_writer(
        &mut *writer,
        &json!({
            "kind":"frame", "step":md.system().step,
            "time_fs": md.system().step as f64 * md.timestep_fs(),
            "frame":md.frame(), "velocities": velocities,
            "evaluation":md.evaluation(), "kinetic_energy_ev": md.kinetic_energy_ev(),
            "total_energy_ev":md.total_energy_ev()
        }),
    )?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn worker(input: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    let request: Request = serde_json::from_str(&fs::read_to_string(input)?)?;
    if request.sample_every == 0 {
        return Err("sample_every must be positive".into());
    }
    let hash = model_hash(request.options.model_path())?;
    let system = match (
        &request.restart,
        &request.frame,
        &request.masses,
        &request.velocities,
    ) {
        (Some(path), None, None, None) => {
            let checkpoint: Checkpoint = serde_json::from_str(&fs::read_to_string(path)?)?;
            checkpoint.restore(&hash, request.timestep_fs)?
        }
        (None, Some(frame), Some(masses), Some(velocities)) => {
            system_from_frame(frame, masses, velocities)?
        }
        _ => return Err("provide either frame+masses+velocities or restart, exclusively".into()),
    };
    let start_step = system.step;
    start_step
        .checked_add(request.steps)
        .ok_or("step counter overflow")?;
    let model = request.options.load()?;
    let mut md = MlipDynamics::new(system, model, request.timestep_fs)?;
    let restart_output = checkpoint_path(output);
    if restart_output.exists() {
        return Err("checkpoint output already exists".into());
    }
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(
        &mut writer,
        &json!({
            "kind":"header","schema":"omnimd-mlmd-jsonl-v1","ensemble":"NVE",
            "metadata":md.metadata(),"timestep_fs":request.timestep_fs,
            "requested_steps":request.steps,"start_step":start_step,
            "masses_u":md.system().particles().mass, "model_sha256":hash
        }),
    )?;
    writer.write_all(b"\n")?;
    write_snapshot(&mut writer, &md)?;
    let initial = md.total_energy_ev();
    let mut max_drift = 0.0_f64;
    for step in 1..=request.steps {
        md.step()?;
        max_drift = max_drift.max((md.total_energy_ev() - initial).abs());
        if step % request.sample_every == 0 || step == request.steps {
            write_snapshot(&mut writer, &md)?;
            writer.flush()?;
        }
    }
    let checkpoint = Checkpoint::capture(&md, hash);
    let mut checkpoint_file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&restart_output)?;
    serde_json::to_writer_pretty(&mut checkpoint_file, &checkpoint)?;
    checkpoint_file.write_all(b"\n")?;
    checkpoint_file.sync_all()?;
    serde_json::to_writer(
        &mut writer,
        &json!({
            "kind":"completed", "steps":request.steps,"final_step":md.system().step,
            "initial_total_energy_ev":initial,"final_total_energy_ev":md.total_energy_ev(),
            "max_absolute_energy_drift_ev":max_drift, "checkpoint":restart_output
        }),
    )?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}
fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() == 3 && args[0] == "--worker" {
        return worker(Path::new(&args[1]), Path::new(&args[2]));
    }
    if args.len() != 2 {
        return Err("usage: omnimd-mlmd REQUEST.json NEW_TRAJECTORY.jsonl".into());
    }
    let request: Request = serde_json::from_str(&fs::read_to_string(&args[0])?)?;
    request.options.validate()?;
    if Path::new(&args[1]).exists() || checkpoint_path(Path::new(&args[1])).exists() {
        return Err("trajectory or checkpoint output already exists".into());
    }
    let status = Command::new(std::env::current_exe()?)
        .arg("--worker")
        .args(&args)
        .env("CUDA_VISIBLE_DEVICES", "-1")
        .env("HIP_VISIBLE_DEVICES", "-1")
        .env("ROCR_VISIBLE_DEVICES", "-1")
        .status()?;
    if !status.success() {
        return Err(
            format!("MD worker failed: {status}; any trajectory lacks a completed record").into(),
        );
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("omnimd-mlmd: {e}");
        std::process::exit(1);
    }
}
