use omnimd_deepmd::{DeepMd, Device, LoadOptions};
use omnimd_ml_contract::{Frame, Potential};
use omnimd_mlmd::{
    initialize_velocities, prepare_velocities, system_from_frame, Checkpoint, Ensemble,
    EnsembleState, MlipDynamics, VelocityInitialization,
};
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

#[derive(Clone, Copy, Default, Deserialize, serde::Serialize)]
enum DynamicsUnits {
    #[default]
    EvAngstromFsU,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    options: BackendOptions,
    #[serde(default)]
    dynamics_units: DynamicsUnits,
    frame: Option<Frame>,
    masses: Option<Vec<f64>>,
    velocities: Option<Vec<[f64; 3]>>,
    initialize_velocities: Option<VelocityInitialization>,
    remove_center_of_mass: Option<bool>,
    ensemble: Option<Ensemble>,
    restart: Option<PathBuf>,
    timestep_fs: f64,
    steps: u64,
    sample_every: u64,
    checkpoint_every: Option<u64>,
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

fn checkpoint_path(output: &Path, step: Option<u64>) -> PathBuf {
    let mut name = output.as_os_str().to_os_string();
    if let Some(step) = step {
        name.push(format!(".step-{step}"));
    }
    name.push(".checkpoint.json");
    PathBuf::from(name)
}

fn prepare(
    request: &Request,
) -> Result<(MlipDynamics<Backend>, EnsembleState, String), Box<dyn Error>> {
    request.options.validate()?;
    if request.sample_every == 0 || request.checkpoint_every == Some(0) {
        return Err("sample_every and checkpoint_every must be positive".into());
    }
    let hash = model_hash(request.options.model_path())?;
    let (system, state) = if let Some(path) = &request.restart {
        if request.frame.is_some()
            || request.masses.is_some()
            || request.velocities.is_some()
            || request.initialize_velocities.is_some()
        {
            return Err(
                "restart cannot be combined with a fresh frame/masses/velocity input".into(),
            );
        }
        let checkpoint: Checkpoint = serde_json::from_str(&fs::read_to_string(path)?)?;
        let system = checkpoint.restore(&hash, request.timestep_fs)?;
        let state = checkpoint
            .ensemble_state
            .unwrap_or(EnsembleState::new(Ensemble::Nve, request.timestep_fs)?);
        if request
            .ensemble
            .as_ref()
            .is_some_and(|e| *e != state.ensemble)
            || request
                .remove_center_of_mass
                .is_some_and(|v| v != checkpoint.remove_center_of_mass)
        {
            return Err("restart ensemble and COM policy must match the checkpoint".into());
        }
        (system, state)
    } else {
        let frame = request.frame.as_ref().ok_or("fresh run requires frame")?;
        let masses = request
            .masses
            .as_ref()
            .ok_or("fresh run requires masses in u")?;
        let remove_com = request.remove_center_of_mass.unwrap_or(false);
        let system = match (&request.velocities, &request.initialize_velocities) {
            (Some(velocities), None) => {
                let mut system = system_from_frame(frame, masses, velocities)?;
                prepare_velocities(&mut system, remove_com)?;
                system
            }
            (None, Some(settings)) => {
                let mut system =
                    system_from_frame(frame, masses, &vec![[0.; 3]; frame.positions.len()])?;
                initialize_velocities(&mut system, settings, remove_com)?;
                system
            }
            _ => {
                return Err(
                    "provide exactly one of velocities (angstrom/fs) or initialize_velocities"
                        .into(),
                )
            }
        };
        let state = EnsembleState::new(
            request.ensemble.clone().unwrap_or_default(),
            request.timestep_fs,
        )?;
        (system, state)
    };
    state.validate(request.timestep_fs)?;
    state.validate_system(&system)?;
    let final_step = system
        .step
        .checked_add(request.steps)
        .ok_or("step counter overflow")?;
    if !(final_step as f64 * request.timestep_fs).is_finite() {
        return Err("simulated time would overflow".into());
    }
    let model = request.options.load()?;
    let md = MlipDynamics::new(system, model, request.timestep_fs)?;
    Ok((md, state, hash))
}

fn write_snapshot<P: Potential>(
    writer: &mut impl Write,
    md: &MlipDynamics<P>,
    state: &EnsembleState,
) -> Result<(), Box<dyn Error>> {
    let diagnostics = [
        md.system().step as f64 * md.timestep_fs(),
        md.kinetic_energy_ev(),
        md.total_energy_ev(),
        md.temperature_kelvin(),
        state.heat_ev,
        md.total_energy_ev() - state.heat_ev,
    ];
    if diagnostics.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite trajectory diagnostic".into());
    }
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
            "total_energy_ev":md.total_energy_ev(), "temperature_kelvin":md.temperature_kelvin(),
            "thermostat_heat_ev":state.heat_ev,
            "effective_energy_ev":md.total_energy_ev()-state.heat_ev
        }),
    )?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn write_checkpoint<P: Potential>(
    path: &Path,
    md: &MlipDynamics<P>,
    state: &EnsembleState,
    hash: &str,
) -> Result<(), Box<dyn Error>> {
    let mut checkpoint = Checkpoint::capture(md, hash.to_owned());
    checkpoint.ensemble_state = Some(state.clone());
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    serde_json::to_writer_pretty(&mut file, &checkpoint)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

fn worker(input: &Path, output: Option<&Path>) -> Result<(), Box<dyn Error>> {
    let request: Request = serde_json::from_str(&fs::read_to_string(input)?)?;
    let (mut md, mut state, hash) = prepare(&request)?;
    let Some(output) = output else {
        println!(
            "Valid CPU model/run: {} atoms, {} DOF, {:.6} K, E = {:.12} eV, ensemble {:?}",
            md.system().size(),
            md.system().degrees_of_freedom(),
            md.temperature_kelvin(),
            md.evaluation().energy,
            state.ensemble
        );
        return Ok(());
    };
    let start_step = md.system().step;
    let restart_output = checkpoint_path(output, None);
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
            "kind":"header", "schema":"omnimd-mlmd-jsonl-v2", "ensemble":state.ensemble,
            "dynamics_units":request.dynamics_units,
            "metadata":md.metadata(), "timestep_fs":request.timestep_fs,
            "requested_steps":request.steps, "start_step":start_step,
            "masses_u":md.system().particles().mass, "model_sha256":hash,
            "degrees_of_freedom":md.system().degrees_of_freedom(),
            "initialize_velocities":request.initialize_velocities,
            "remove_center_of_mass":matches!(md.system().simulated_degrees_of_freedom, omnimd_core::DegreesOfFreedom::Frozen(3)),
            "checkpoint_every":request.checkpoint_every
        }),
    )?;
    writer.write_all(b"\n")?;
    write_snapshot(&mut writer, &md, &state)?;
    let initial = md.total_energy_ev();
    let initial_effective = initial - state.heat_ev;
    let mut max_drift = 0.0_f64;
    let mut max_effective_drift = 0.0_f64;
    for step in 1..=request.steps {
        state.advance(&mut md)?;
        max_drift = max_drift.max((md.total_energy_ev() - initial).abs());
        max_effective_drift = max_effective_drift
            .max((md.total_energy_ev() - state.heat_ev - initial_effective).abs());
        if step % request.sample_every == 0 || step == request.steps {
            write_snapshot(&mut writer, &md, &state)?;
            writer.flush()?;
        }
        if request.checkpoint_every.is_some_and(|n| step % n == 0) && step != request.steps {
            write_checkpoint(
                &checkpoint_path(output, Some(md.system().step)),
                &md,
                &state,
                &hash,
            )?;
        }
    }
    write_checkpoint(&restart_output, &md, &state, &hash)?;
    let nve_drift = matches!(state.ensemble, Ensemble::Nve).then_some(max_drift);
    serde_json::to_writer(
        &mut writer,
        &json!({
            "kind":"completed", "steps":request.steps, "final_step":md.system().step,
            "initial_total_energy_ev":initial, "final_total_energy_ev":md.total_energy_ev(),
            "max_absolute_energy_drift_ev":nve_drift,
            "max_absolute_effective_energy_drift_ev":max_effective_drift,
            "thermostat_heat_ev":state.heat_ev, "checkpoint":restart_output
        }),
    )?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() == 3 && args[0] == "--worker" {
        return worker(Path::new(&args[1]), Some(Path::new(&args[2])));
    }
    if args.len() == 2 && args[0] == "--check-worker" {
        return worker(Path::new(&args[1]), None);
    }
    if args.len() == 1 && (args[0] == "--help" || args[0] == "-h") {
        println!(
            "omnimd-mlmd: CPU DeepMD C API / native se_e2_a dynamics\n\
Usage:\n  omnimd-mlmd run REQUEST.json NEW_TRAJECTORY.jsonl\n\
  omnimd-mlmd check REQUEST.json\n\
  omnimd-mlmd REQUEST.json NEW_TRAJECTORY.jsonl\n\
Units: angstrom, fs, u, K, eV. Ensembles: nve, nvt_csvr.\n\
Use explicit velocities or seeded initialize_velocities; restart preserves RNG state.\n\
Configuration: mlmd/engine/README.md"
        );
        return Ok(());
    }
    if args.len() == 1 && args[0] == "--version" {
        println!("omnimd-mlmd {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    if args.first().is_some_and(|arg| arg == "run") {
        args.remove(0);
    }
    let check = args.len() == 2 && args[0] == "check";
    if args.len() != 2 {
        return Err("use omnimd-mlmd --help for usage".into());
    }
    let input = if check { &args[1] } else { &args[0] };
    let request: Request = serde_json::from_str(&fs::read_to_string(input)?)?;
    request.options.validate()?;
    if !check
        && (Path::new(&args[1]).exists() || checkpoint_path(Path::new(&args[1]), None).exists())
    {
        return Err("trajectory or checkpoint output already exists".into());
    }
    let mut child = Command::new(std::env::current_exe()?);
    if check {
        child.arg("--check-worker").arg(input);
    } else {
        child.arg("--worker").args(&args);
    }
    let status = child
        .env("CUDA_VISIBLE_DEVICES", "-1")
        .env("HIP_VISIBLE_DEVICES", "-1")
        .env("ROCR_VISIBLE_DEVICES", "-1")
        .status()?;
    if !status.success() {
        return Err(format!(
            "MD worker failed: {status}; a failed trajectory has no completed record"
        )
        .into());
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("omnimd-mlmd: {e}");
        std::process::exit(1);
    }
}
