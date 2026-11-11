//! Several independent runs of one model advanced together: every step asks
//! the model for all systems in one batch, which a backend that shares work
//! between frames (the omnimd-gpu runtime) answers in one device pass. Each
//! replica writes the trajectory and the checkpoints it would write alone.
use super::{
    Backend, BackendOptions, DeviationInput, DynamicsUnits, Recorder, Schedule, SystemInput,
};
use omnimd_ml_contract::Frame;
use omnimd_mlmd::{
    advance_replicas, dataset::FrameReference, Ensemble, MlipReplicas, VelocityInitialization,
};
use serde::Deserialize;
use serde_json::json;
use std::{
    collections::HashSet,
    error::Error,
    ffi::OsString,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;

const SCHEMA: &str = "omnimd-replicas-v1";
/// Most replicas of one run.
pub const MAX_REPLICAS: usize = 1024;

/// One system of the run: what a `run` request says of its system, and the
/// name its files get.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Replica {
    name: String,
    frame: Option<Frame>,
    dataset_frame: Option<FrameReference>,
    masses: Option<Vec<f64>>,
    velocities: Option<Vec<[f64; 3]>>,
    initialize_velocities: Option<VelocityInitialization>,
    remove_center_of_mass: Option<bool>,
    ensemble: Option<Ensemble>,
    restart: Option<PathBuf>,
}

impl Replica {
    fn system(&self) -> SystemInput<'_> {
        SystemInput {
            frame: self.frame.as_ref(),
            dataset_frame: self.dataset_frame.as_ref(),
            masses: self.masses.as_ref(),
            velocities: self.velocities.as_ref(),
            initialize_velocities: self.initialize_velocities.as_ref(),
            remove_center_of_mass: self.remove_center_of_mass,
            ensemble: self.ensemble.as_ref(),
            restart: self.restart.as_ref(),
            fixed_parameters: None,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: String,
    options: BackendOptions,
    #[serde(default)]
    dynamics_units: DynamicsUnits,
    replicas: Vec<Replica>,
    timestep_fs: f64,
    steps: u64,
    sample_every: u64,
    checkpoint_every: Option<u64>,
    /// The other members of the model's ensemble, compared with it on the
    /// sampled frames of every replica.
    deviation: Option<DeviationInput>,
}

impl Request {
    fn read(path: &Path) -> Result<Self> {
        let request: Self = serde_json::from_str(&fs::read_to_string(path)?)?;
        if request.schema != SCHEMA {
            return Err("unsupported replicas request schema".into());
        }
        request.options.validate()?;
        if request.sample_every == 0 || request.checkpoint_every == Some(0) {
            return Err("sample_every and checkpoint_every must be positive".into());
        }
        if let Some(deviation) = &request.deviation {
            deviation.validate(&request.options)?;
        }
        if !(1..=MAX_REPLICAS).contains(&request.replicas.len()) {
            return Err(format!("a replica run takes 1..={MAX_REPLICAS} replicas").into());
        }
        // Names become file names: plain, and distinct whatever the case.
        let mut names = HashSet::new();
        for replica in &request.replicas {
            let name = replica.name.as_str();
            let plain = name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'));
            if name.is_empty()
                || name.len() > 64
                || !plain
                || name.starts_with('.')
                || !names.insert(name.to_ascii_lowercase())
            {
                return Err(format!(
                    "replica name {name:?}: names are 1..=64 characters of [A-Za-z0-9_.-], not starting with a dot, and distinct without regard to case"
                )
                .into());
            }
        }
        Ok(request)
    }
}

fn worker(input: &Path, directory: &Path) -> Result<()> {
    let request = Request::read(input)?;
    if directory.exists() {
        return Err("output directory already exists".into());
    }
    let hash = super::model_hash(request.options.model_path())?;
    let mut systems = Vec::with_capacity(request.replicas.len());
    let mut states = Vec::with_capacity(request.replicas.len());
    for replica in &request.replicas {
        let named = |e: Box<dyn Error>| format!("replica {}: {e}", replica.name);
        let (system, state, parameters) =
            super::prepare_system(&replica.system(), &hash, request.timestep_fs, request.steps)
                .map_err(named)?;
        if parameters.is_some() {
            return Err(format!(
                "replica {}: frame and atom parameters are not carried by a replica run",
                replica.name
            )
            .into());
        }
        systems.push(system);
        states.push(state);
    }
    super::parameter_backend::validate_request(&request.options, None)?;
    let (model, deviation) = match &request.deviation {
        Some(deviation) => {
            let (committee, header) = deviation.load(&request.options)?;
            (Backend::Members(committee), Some(header))
        }
        None => (request.options.load()?, None),
    };
    let mut md = MlipReplicas::new(systems, model, request.timestep_fs)?;
    // What the members say of the frames all replicas are at, in one
    // evaluation; nothing for a single model.
    let disagreement = |md: &mut MlipReplicas<Backend>| -> Result<_> {
        let frames: Vec<Frame> = (0..md.len()).map(|index| md.view(index).frame()).collect();
        let frames: Vec<&Frame> = frames.iter().collect();
        super::deviations(md.potential_mut(), &frames)
    };
    let schedule = Schedule {
        dynamics_units: request.dynamics_units,
        timestep_fs: request.timestep_fs,
        steps: request.steps,
        sample_every: request.sample_every,
        checkpoint_every: request.checkpoint_every,
    };
    fs::create_dir(directory)?;
    let count = request.replicas.len();
    let mut recorders = Vec::with_capacity(count);
    for (index, replica) in request.replicas.iter().enumerate() {
        recorders.push(Recorder::create(
            &directory.join(format!("{}.jsonl", replica.name)),
            &schedule,
            &replica.system(),
            md.view(index),
            &states[index],
            &hash,
            None,
            match &deviation {
                Some(header) => json!({"deviation": header,
                    "replica": {"name": replica.name, "index": index, "of": count}}),
                None => json!({"replica": {"name": replica.name, "index": index, "of": count}}),
            },
            request.deviation.as_ref().map(|d| d.force_window),
        )?);
    }
    let first = disagreement(&mut md)?;
    for (index, recorder) in recorders.iter_mut().enumerate() {
        let of_replica = first.as_ref().map(|all| &all[index]);
        recorder.sample(md.view(index), &states[index], of_replica)?;
    }
    let started = std::time::Instant::now();
    for step in 1..=request.steps {
        advance_replicas(&mut states, &mut md)
            .map_err(|e| format!("step {step} of the replicas: {e}"))?;
        let now = match schedule.samples(step) {
            true => disagreement(&mut md)?,
            false => None,
        };
        for (index, recorder) in recorders.iter_mut().enumerate() {
            let of_replica = now.as_ref().map(|all| &all[index]);
            recorder.after_step(step, &schedule, md.view(index), &states[index], of_replica)?;
        }
    }
    let stepping_seconds = started.elapsed().as_secs_f64();
    super::report_rate(request.steps, count, stepping_seconds);
    let mut finished = Vec::with_capacity(count);
    for (index, (recorder, replica)) in recorders.into_iter().zip(&request.replicas).enumerate() {
        let trajectory = recorder.output.clone();
        let checkpoint = recorder.finish(&schedule, md.view(index), &states[index])?;
        finished.push(json!({"name": replica.name, "trajectory": trajectory,
            "checkpoint": checkpoint, "final_step": md.view(index).system().step,
            "ensemble": states[index].ensemble}));
    }
    // Written last: a directory with this file holds complete trajectories.
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join("replicas.json"))?;
    serde_json::to_writer_pretty(
        &mut file,
        &json!({"schema": "omnimd-replicas-run-v1", "model_sha256": hash,
            "metadata": md.metadata(), "timestep_fs": request.timestep_fs,
            "steps": request.steps, "stepping_seconds": stepping_seconds,
            "replicas": finished}),
    )?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

pub fn dispatch(args: &[OsString]) -> Result<bool> {
    if args.len() == 3 && args[0] == "--replicas-worker" {
        worker(Path::new(&args[1]), Path::new(&args[2]))?;
        return Ok(true);
    }
    if args.len() == 3 && args[0] == "run-replicas" {
        if Path::new(&args[2]).exists() {
            return Err("output directory already exists".into());
        }
        let request = Request::read(Path::new(&args[1]))?;
        let mut command = Command::new(std::env::current_exe()?);
        command.arg("--replicas-worker").args(&args[1..]);
        request.options.device().configure_worker(&mut command)?;
        let status = command.status()?;
        if !status.success() {
            return Err(format!(
                "replica worker failed: {status}; trajectories without replicas.json beside them are incomplete"
            )
            .into());
        }
        return Ok(true);
    }
    Ok(false)
}
