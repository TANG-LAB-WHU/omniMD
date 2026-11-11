mod dataset_commands;
mod deviation_command;
mod minimize_command;
mod parameter_backend;
mod replicas_command;
use omnimd_deepmd::{DeepMd, Device, LoadOptions};
use omnimd_ml_contract::{Committee, Frame, Potential, Separate};
use omnimd_mlmd::{
    deviation::{self, ForceWindow, ModelDeviation, MAX_MODELS},
    initialize_velocities, prepare_velocities, system_from_frame, Checkpoint, Ensemble,
    EnsembleState, MlipDynamics, StateView, VelocityInitialization,
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

#[derive(Clone, Copy, Default, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "lowercase")]
enum NetworkPrecision {
    #[default]
    Float64,
    Float32,
}

/// Selects the omnimd-gpu se_e2_a runtime for a native model on `cuda:N`.
#[derive(Clone, Copy, Default, Deserialize, PartialEq, Debug)]
#[serde(deny_unknown_fields)]
struct GpuRuntimeOptions {
    /// Network precision; geometry, forces and virial are always float64.
    #[serde(default)]
    precision: NetworkPrecision,
    /// Tabulate the embedding nets for neighbor distances down to this many
    /// angstrom; closer pairs fall back to the exact networks.
    compress_min_distance: Option<f64>,
    /// Table step in the normalized radial input (default 0.01).
    table_step: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeOptions {
    native_model: PathBuf,
    #[serde(default)]
    device: Device,
    /// Absent: the CPU path, or the bounded `native-cuda` path on `cuda:N`.
    gpu_runtime: Option<GpuRuntimeOptions>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum BackendOptions {
    DeepMd(LoadOptions),
    Native(NativeOptions),
}

enum Backend {
    DeepMd(DeepMd),
    Conditioned(parameter_backend::ConditionedPotential),
    Native(omnimd_native::NativePotential),
    #[cfg(feature = "native-cuda")]
    NativeCuda(omnimd_native::cuda::CudaPotential),
    #[cfg(feature = "native-gpu")]
    NativeGpu(Box<dyn Potential>),
    /// A model with the other members of its ensemble: as a potential, the
    /// first of them.
    Members(Box<dyn Committee>),
}

/// Frames of a dataset that a backend which shares work between frames
/// evaluates in one pass: at most this many, and no more once they hold this
/// many atoms for each model evaluated with them. Past a few thousand atoms
/// a pass gains nothing more per frame.
const BATCH_FRAMES: usize = 64;
const BATCH_ATOMS: usize = 8192;

/// The omnimd-gpu options of a request.
#[cfg(feature = "native-gpu")]
fn gpu_options(runtime: GpuRuntimeOptions) -> omnimd_native::gpu_model::GpuOptions {
    omnimd_native::gpu_model::GpuOptions {
        float32: runtime.precision == NetworkPrecision::Float32,
        compress: runtime
            .compress_min_distance
            .map(|r| (r, runtime.table_step.unwrap_or(0.01))),
    }
}

/// The ordinal of `cuda:index` in this worker. Same fresh-worker contract as
/// the other CUDA paths: the selected device is the only visible one,
/// ordinal zero.
#[cfg(feature = "native-gpu")]
fn gpu_ordinal(index: u32) -> Result<usize, Box<dyn Error>> {
    Ok(native_cuda_ordinal(
        index,
        std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref(),
        std::env::var("OMNIMD_CUDA_INDEX").ok().as_deref(),
    )? as usize)
}

/// One evaluation per frame of a batch, in one pass where the backend can.
/// An error is traced to its frame by evaluating the frames one at a time.
fn eval_frames(
    model: &mut impl Potential,
    batch: &[omnimd_mlmd::dataset::DatasetFrame],
) -> Result<Vec<omnimd_ml_contract::Evaluation>, Box<dyn Error>> {
    let frames: Vec<&Frame> = batch.iter().map(|data| &data.frame).collect();
    match model.eval_batch(&frames) {
        Ok(evaluations) if evaluations.len() == batch.len() => Ok(evaluations),
        Ok(evaluations) => Err(format!(
            "the backend returned {} evaluations for {} frames",
            evaluations.len(),
            batch.len()
        )
        .into()),
        Err(error) => {
            for data in batch {
                model
                    .eval(&data.frame)
                    .map_err(|e| format!("dataset frame {} ({}): {e}", data.index, data.id))?;
            }
            Err(error.into())
        }
    }
}

impl Backend {
    /// Whether a batch of frames costs this backend less than the frames
    /// one by one.
    fn shares_frames(&self) -> bool {
        match self {
            #[cfg(feature = "native-gpu")]
            Self::NativeGpu(_) => true,
            _ => false,
        }
    }
}

/// What the members of one ensemble must share to be evaluated in one
/// worker: the CPU for all, or the same `cuda:N` with the omnimd-gpu runtime
/// for all; and, for C API members, one runtime library.
fn validate_members(members: &[&BackendOptions]) -> Result<(), Box<dyn Error>> {
    let mut library = None;
    for member in members {
        let device = member.device();
        if device != members[0].device()
            || (device != Device::Cpu && member.gpu_runtime().is_none())
        {
            return Err(
                "the members of an ensemble need the CPU for every member, or the same cuda:N with gpu_runtime for every member"
                    .into(),
            );
        }
        member.validate()?;
        if let BackendOptions::DeepMd(options) = member {
            let path = options.library.canonicalize()?;
            if library.as_ref().is_some_and(|old| *old != path) {
                return Err("all C API ensemble members must use the same runtime library".into());
            }
            library = Some(path);
        }
    }
    Ok(())
}

/// The members of an ensemble, loaded.
struct Members {
    committee: Box<dyn Committee>,
    /// The metadata of each member.
    metadata: Vec<omnimd_ml_contract::ModelMetadata>,
    /// How the members are evaluated, for the record.
    evaluation: &'static str,
    /// Whether a batch of frames costs less than its frames one by one.
    batched: bool,
}

/// The members as one committee on the omnimd-gpu runtime, when every
/// member selects that runtime with the same options and the models share
/// one architecture. `None` otherwise.
#[cfg(feature = "native-gpu")]
fn gpu_committee(
    members: &[&BackendOptions],
) -> Result<Option<Box<dyn Committee>>, Box<dyn Error>> {
    let (Some(runtime), Device::Cuda(index)) = (members[0].gpu_runtime(), members[0].device())
    else {
        return Ok(None);
    };
    if members
        .iter()
        .any(|member| member.gpu_runtime() != Some(runtime))
    {
        return Ok(None);
    }
    let loaded = members
        .iter()
        .map(|member| omnimd_native::NativePotential::load(member.model_path()))
        .collect::<Result<Vec<_>, _>>()?;
    let models: Vec<&omnimd_native::Model> = loaded.iter().map(|p| p.model()).collect();
    if let Some(reason) = omnimd_native::gpu_model::committee_mismatch(&models) {
        eprintln!("omnimd-mlmd: the members are evaluated one by one: {reason}");
        return Ok(None);
    }
    Ok(Some(omnimd_native::gpu_model::open_committee(
        &models,
        gpu_ordinal(index)?,
        gpu_options(runtime),
    )?))
}

#[cfg(not(feature = "native-gpu"))]
fn gpu_committee(_: &[&BackendOptions]) -> Result<Option<Box<dyn Committee>>, Box<dyn Error>> {
    Ok(None)
}

/// Loads the members of an ensemble: as one committee on the device where
/// they can share its passes, else one after the other. `name` names member
/// `index` in error messages.
fn load_members(
    members: &[&BackendOptions],
    name: &dyn Fn(usize) -> String,
) -> Result<Members, Box<dyn Error>> {
    if let Some(committee) = gpu_committee(members)? {
        // Alike in all that their metadata says.
        return Ok(Members {
            metadata: vec![committee.metadata().clone(); members.len()],
            committee,
            evaluation: "all members together, a batch of frames in one device pass",
            batched: true,
        });
    }
    let mut models = Vec::with_capacity(members.len());
    for (index, member) in members.iter().enumerate() {
        let model = member.load()?;
        if let Backend::DeepMd(deepmd) = &model {
            let details = deepmd.details();
            if details.frame_parameter_dim != 0 || details.atom_parameter_dim != 0 {
                return Err(format!(
                    "{} requires parameters; use the standalone parameter-aware C API instead",
                    name(index)
                )
                .into());
            }
        }
        models.push(model);
    }
    let batched = models.iter().all(Backend::shares_frames);
    Ok(Members {
        metadata: models
            .iter()
            .map(|model| model.metadata().clone())
            .collect(),
        committee: Box::new(Separate::new(models)?),
        evaluation: match batched {
            true => "member by member, a batch of frames in one device pass each",
            false => "member by member, frame by frame",
        },
        batched,
    })
}

/// The other members of the ensemble of a run's model, to compare with it on
/// the sampled frames.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeviationInput {
    /// The models besides that of `options`, which is the first member.
    models: Vec<BackendOptions>,
    /// Marks sampled frames as candidates, as `model-deviation` does.
    force_window: Option<ForceWindow>,
}

impl DeviationInput {
    /// The model of the run and the others, in the order of the members.
    fn members<'a>(&'a self, driving: &'a BackendOptions) -> Vec<&'a BackendOptions> {
        std::iter::once(driving).chain(&self.models).collect()
    }

    fn validate(&self, driving: &BackendOptions) -> Result<(), Box<dyn Error>> {
        if self.models.is_empty() || self.models.len() >= MAX_MODELS {
            return Err(format!(
                "deviation takes 1..={} models besides the one of the run",
                MAX_MODELS - 1
            )
            .into());
        }
        if let Some(window) = self.force_window {
            window.validate()?;
        }
        validate_members(&self.members(driving))
    }

    /// The members, loaded, and what the header of a trajectory says of them.
    fn load(
        &self,
        driving: &BackendOptions,
    ) -> Result<(Box<dyn Committee>, serde_json::Value), Box<dyn Error>> {
        let members = self.members(driving);
        let name = |index: usize| match index {
            0 => "the model of the run".to_string(),
            other => format!("deviation model {other}"),
        };
        let loaded = load_members(&members, &name)?;
        let models = members
            .iter()
            .map(|member| {
                Ok(json!({"model": member.model_path(),
                    "model_sha256": model_hash(member.model_path())?}))
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        let header = json!({"models": models, "evaluation": loaded.evaluation,
            "force_window": self.force_window,
            "statistics": "as model-deviation: absolute population std over the models on each sampled frame"});
        Ok((loaded.committee, header))
    }
}

/// What the members of an ensemble say of the sampled frames of a run.
struct DeviationLog {
    window: Option<ForceWindow>,
    frames: u64,
    /// Frames below, inside and at or above the force window.
    accurate: u64,
    candidates: u64,
    failed: u64,
    maximum: f64,
}

impl DeviationLog {
    fn new(window: Option<ForceWindow>) -> Self {
        Self {
            window,
            frames: 0,
            accurate: 0,
            candidates: 0,
            failed: 0,
            maximum: 0.0,
        }
    }

    /// Counts a frame and returns whether it is a candidate (no window, no
    /// answer).
    fn note(&mut self, statistics: &ModelDeviation) -> Option<bool> {
        let largest = statistics.force_ev_per_angstrom.max;
        self.frames += 1;
        self.maximum = self.maximum.max(largest);
        let window = self.window?;
        self.accurate += u64::from(largest < window.min_ev_per_angstrom);
        self.failed += u64::from(largest >= window.max_ev_per_angstrom);
        let candidate = window.contains(largest);
        self.candidates += u64::from(candidate);
        Some(candidate)
    }
}

/// The deviation among the members on each of `frames`, when the model is an
/// ensemble; `None` for a single model.
fn deviations(
    model: &mut Backend,
    frames: &[&Frame],
) -> Result<Option<Vec<ModelDeviation>>, Box<dyn Error>> {
    match model {
        Backend::Members(committee) => Ok(Some(deviation::evaluate_frames(
            committee.as_mut(),
            frames,
            false,
        )?)),
        _ => Ok(None),
    }
}

impl BackendOptions {
    /// The omnimd-gpu runtime options, when this request selects it.
    fn gpu_runtime(&self) -> Option<GpuRuntimeOptions> {
        match self {
            Self::Native(options) => options.gpu_runtime,
            Self::DeepMd(_) => None,
        }
    }

    fn device(&self) -> Device {
        match self {
            Self::DeepMd(options) => options.device,
            Self::Native(options) => options.device,
        }
    }

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
        if let Some(runtime) = self.gpu_runtime() {
            if self.device() == Device::Cpu {
                return Err("gpu_runtime requires device cuda:N; omit gpu_runtime for CPU".into());
            }
            if runtime
                .compress_min_distance
                .is_some_and(|r| !(r.is_finite() && r > 0.0))
                || runtime
                    .table_step
                    .is_some_and(|s| !(s.is_finite() && s > 0.0))
            {
                return Err(
                    "gpu_runtime compress_min_distance and table_step must be positive".into(),
                );
            }
            if runtime.table_step.is_some() && runtime.compress_min_distance.is_none() {
                return Err("gpu_runtime table_step requires compress_min_distance".into());
            }
            #[cfg(not(feature = "native-gpu"))]
            return Err("gpu_runtime requires a build with --features native-gpu".into());
            #[cfg(feature = "native-gpu")]
            return Ok(());
        }
        #[cfg(not(feature = "native-cuda"))]
        if matches!(
            self,
            Self::Native(NativeOptions {
                device: Device::Cuda(_),
                ..
            })
        ) {
            return Err("native CUDA inference requires a build with --features native-cuda (or select the omnimd-gpu runtime with gpu_runtime and --features native-gpu); select device=cpu explicitly for CPU".into());
        }
        Ok(())
    }

    fn load(&self) -> Result<Backend, Box<dyn Error>> {
        self.validate()?;
        match self {
            // The CLI accepts only a trusted official DeepMD C API library.
            Self::DeepMd(options) => Ok(Backend::DeepMd(unsafe { DeepMd::load(options) }?)),
            Self::Native(options) => {
                let model = omnimd_native::NativePotential::load(&options.native_model)?;
                #[cfg(feature = "native-gpu")]
                if let (Some(runtime), Device::Cuda(index)) = (options.gpu_runtime, options.device)
                {
                    return Ok(Backend::NativeGpu(omnimd_native::gpu_model::open(
                        model.model(),
                        gpu_ordinal(index)?,
                        gpu_options(runtime),
                    )?));
                }
                match options.device {
                    Device::Cpu => Ok(Backend::Native(model)),
                    Device::Cuda(index) => {
                        #[cfg(feature = "native-cuda")]
                        {
                            // The shared parent launcher exposes one selected device.
                            // Its visible CUDA ordinal is zero, not the original index.
                            let ordinal = native_cuda_ordinal(
                                index,
                                std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref(),
                                std::env::var("OMNIMD_CUDA_INDEX").ok().as_deref(),
                            )?;
                            Ok(Backend::NativeCuda(
                                omnimd_native::cuda::CudaPotential::new(
                                    model.model().clone(),
                                    ordinal,
                                )?,
                            ))
                        }
                        #[cfg(not(feature = "native-cuda"))]
                        {
                            let _ = (model, index);
                            Err("native CUDA inference was not compiled".into())
                        }
                    }
                }
            }
        }
    }
}

// Validate only the launch contract; this helper never loads a CUDA library.
// Native Driver API indices use the same isolated visibility mapping as C API.
#[cfg(any(feature = "native-cuda", feature = "native-gpu", test))]
fn native_cuda_ordinal(
    index: u32,
    visible: Option<&str>,
    selected: Option<&str>,
) -> Result<i32, Box<dyn Error>> {
    let single = visible.is_some_and(|value| {
        !value.contains(',')
            && (value.parse::<u32>().is_ok()
                || (value.starts_with("GPU-") && value.len() > 4)
                || (value.starts_with("MIG-") && value.len() > 4))
    });
    if !single || selected != Some(index.to_string().as_str()) {
        return Err("native CUDA requires the matching fresh device-isolated worker; do not call internal worker modes directly".into());
    }
    Ok(0)
}

impl Potential for Backend {
    fn metadata(&self) -> &omnimd_ml_contract::ModelMetadata {
        match self {
            Self::DeepMd(model) => model.metadata(),
            Self::Conditioned(model) => model.metadata(),
            Self::Native(model) => model.metadata(),
            #[cfg(feature = "native-cuda")]
            Self::NativeCuda(model) => model.metadata(),
            #[cfg(feature = "native-gpu")]
            Self::NativeGpu(model) => model.metadata(),
            Self::Members(model) => model.metadata(),
        }
    }
    fn eval(
        &mut self,
        frame: &Frame,
    ) -> omnimd_ml_contract::Result<omnimd_ml_contract::Evaluation> {
        match self {
            Self::DeepMd(model) => model.eval(frame),
            Self::Conditioned(model) => model.eval(frame),
            Self::Native(model) => model.eval(frame),
            #[cfg(feature = "native-cuda")]
            Self::NativeCuda(model) => model.eval(frame),
            #[cfg(feature = "native-gpu")]
            Self::NativeGpu(model) => model.eval(frame),
            Self::Members(model) => model.eval(frame),
        }
    }
    fn eval_batch(
        &mut self,
        frames: &[&Frame],
    ) -> omnimd_ml_contract::Result<Vec<omnimd_ml_contract::Evaluation>> {
        match self {
            #[cfg(feature = "native-gpu")]
            Self::NativeGpu(model) => model.eval_batch(frames),
            Self::Members(model) => model.eval_batch(frames),
            _ => frames.iter().map(|frame| self.eval(frame)).collect(),
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
    dataset_frame: Option<omnimd_mlmd::dataset::FrameReference>,
    masses: Option<Vec<f64>>,
    velocities: Option<Vec<[f64; 3]>>,
    initialize_velocities: Option<VelocityInitialization>,
    remove_center_of_mass: Option<bool>,
    ensemble: Option<Ensemble>,
    restart: Option<PathBuf>,
    fixed_parameters: Option<omnimd_mlmd::parameters::FixedParameters>,
    timestep_fs: f64,
    steps: u64,
    sample_every: u64,
    checkpoint_every: Option<u64>,
    deviation: Option<DeviationInput>,
}

/// The part of a request that describes one system and how it is driven.
struct SystemInput<'a> {
    frame: Option<&'a Frame>,
    dataset_frame: Option<&'a omnimd_mlmd::dataset::FrameReference>,
    masses: Option<&'a Vec<f64>>,
    velocities: Option<&'a Vec<[f64; 3]>>,
    initialize_velocities: Option<&'a VelocityInitialization>,
    remove_center_of_mass: Option<bool>,
    ensemble: Option<&'a Ensemble>,
    restart: Option<&'a PathBuf>,
    fixed_parameters: Option<&'a omnimd_mlmd::parameters::FixedParameters>,
}

impl Request {
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
            fixed_parameters: self.fixed_parameters.as_ref(),
        }
    }
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

/// The system a run starts from, fresh or from a checkpoint of a model with
/// this hash, with the state of its ensemble and its fixed parameters.
fn prepare_system(
    input: &SystemInput<'_>,
    hash: &str,
    timestep_fs: f64,
    steps: u64,
) -> Result<
    (
        omnimd_core::System,
        EnsembleState,
        Option<omnimd_mlmd::parameters::FixedParameters>,
    ),
    Box<dyn Error>,
> {
    let (system, state, fixed_parameters) = if let Some(path) = input.restart {
        if input.frame.is_some()
            || input.dataset_frame.is_some()
            || input.masses.is_some()
            || input.velocities.is_some()
            || input.initialize_velocities.is_some()
        {
            return Err(
                "restart cannot be combined with a fresh frame/masses/velocity input".into(),
            );
        }
        let checkpoint: Checkpoint = serde_json::from_str(&fs::read_to_string(path)?)?;
        if input
            .fixed_parameters
            .is_some_and(|p| Some(p) != checkpoint.fixed_parameters.as_ref())
        {
            return Err("restart fixed parameters must exactly match the checkpoint, including layout/policy/atom order".into());
        }
        let fixed_parameters = checkpoint.fixed_parameters.clone();
        let system =
            checkpoint.restore_with_parameters(hash, timestep_fs, fixed_parameters.as_ref())?;
        let state = checkpoint
            .ensemble_state
            .unwrap_or(EnsembleState::new(Ensemble::Nve, timestep_fs)?);
        if input.ensemble.is_some_and(|e| *e != state.ensemble)
            || input
                .remove_center_of_mass
                .is_some_and(|v| v != checkpoint.remove_center_of_mass)
        {
            return Err("restart ensemble and COM policy must match the checkpoint".into());
        }
        (system, state, fixed_parameters)
    } else {
        let resolved = input
            .dataset_frame
            .map(|r| r.resolve_with_parameters())
            .transpose()?;
        let frame = match (input.frame, &resolved) {
            (Some(frame), None) => frame,
            (None, Some((frame, _))) => frame,
            _ => return Err("fresh run requires exactly one frame or dataset_frame".into()),
        };
        let inherited = resolved.as_ref().and_then(|(_, p)| p.as_ref());
        if inherited.is_some() && input.fixed_parameters.is_some() {
            return Err("dataset v2 parameters cannot be overridden by fixed_parameters".into());
        }
        let fixed_parameters = inherited.or(input.fixed_parameters).cloned();
        if let Some(p) = &fixed_parameters {
            p.validate(frame)?;
        }
        let masses = input.masses.ok_or("fresh run requires masses in u")?;
        let remove_com = input.remove_center_of_mass.unwrap_or(false);
        let system = match (input.velocities, input.initialize_velocities) {
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
        let state = EnsembleState::new(input.ensemble.cloned().unwrap_or_default(), timestep_fs)?;
        (system, state, fixed_parameters)
    };
    state.validate(timestep_fs)?;
    state.validate_system(&system)?;
    let final_step = system
        .step
        .checked_add(steps)
        .ok_or("step counter overflow")?;
    if !(final_step as f64 * timestep_fs).is_finite() {
        return Err("simulated time would overflow".into());
    }
    Ok((system, state, fixed_parameters))
}

/// A run made ready: its dynamics and ensemble state, the hash of its model,
/// its fixed parameters, and what the header says of a model deviation.
type Prepared = (
    MlipDynamics<Backend>,
    EnsembleState,
    String,
    Option<omnimd_mlmd::parameters::FixedParameters>,
    Option<serde_json::Value>,
);

fn prepare(request: &Request) -> Result<Prepared, Box<dyn Error>> {
    request.options.validate()?;
    if request.sample_every == 0 || request.checkpoint_every == Some(0) {
        return Err("sample_every and checkpoint_every must be positive".into());
    }
    if let Some(deviation) = &request.deviation {
        deviation.validate(&request.options)?;
    }
    let hash = model_hash(request.options.model_path())?;
    let (system, state, fixed_parameters) =
        prepare_system(&request.system(), &hash, request.timestep_fs, request.steps)?;
    parameter_backend::validate_request(
        &request.options,
        fixed_parameters.as_ref().map(|p| &p.layout),
    )?;
    let (model, deviation) = match &request.deviation {
        Some(_) if fixed_parameters.is_some() => {
            return Err("a run with deviation does not carry frame or atom parameters".into())
        }
        Some(deviation) => {
            let (committee, header) = deviation.load(&request.options)?;
            (Backend::Members(committee), Some(header))
        }
        None => (request.options.load()?, None),
    };
    let model = match (model, fixed_parameters.clone()) {
        (Backend::DeepMd(model), Some(fixed)) => {
            parameter_backend::validate_layout(&model, &fixed.layout)?;
            Backend::Conditioned(parameter_backend::ConditionedPotential { model, fixed })
        }
        (model, None) => model,
        (_, Some(_)) => return Err("fixed parameters require CPU C API".into()),
    };
    let md = MlipDynamics::new(system, model, request.timestep_fs)?;
    Ok((md, state, hash, fixed_parameters, deviation))
}

/// Writes the frame record of the current state; `extra` are further
/// entries of the record.
fn write_snapshot(
    writer: &mut impl Write,
    md: StateView<'_>,
    state: &EnsembleState,
    extra: Option<serde_json::Value>,
) -> Result<(), Box<dyn Error>> {
    // Reuse diagnostics while this immutable snapshot is validated and written.
    // Keep the existing kinetic sum and temperature calculation order.
    let time_fs = md.system().step as f64 * md.timestep_fs();
    let kinetic_energy_ev = md.kinetic_energy_ev();
    let total_energy_ev = md.evaluation().energy + kinetic_energy_ev;
    let temperature_kelvin = md.temperature_kelvin();
    let effective_energy_ev = total_energy_ev - state.heat_ev - state.barostat_work_ev;
    let diagnostics = [
        time_fs,
        kinetic_energy_ev,
        total_energy_ev,
        temperature_kelvin,
        state.heat_ev,
        state.barostat_work_ev,
        effective_energy_ev,
    ];
    if diagnostics.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite trajectory diagnostic".into());
    }
    // Periodic frames also carry the instantaneous pressure and the volume.
    let pressure = match md.frame().cell {
        Some(_) => Some(md.pressure_tensor()?),
        None => None,
    };
    let velocities: Vec<[f64; 3]> = md
        .system()
        .particles()
        .velocity
        .iter()
        .map(|v| [v[0], v[1], v[2]])
        .collect();
    let mut record = json!({
        "kind":"frame", "step":md.system().step,
        "time_fs": time_fs,
        "frame":md.frame(), "velocities": velocities,
        "evaluation":md.evaluation(), "kinetic_energy_ev": kinetic_energy_ev,
        "total_energy_ev":total_energy_ev, "temperature_kelvin":temperature_kelvin,
        "thermostat_heat_ev":state.heat_ev,
        "effective_energy_ev":effective_energy_ev
    });
    if let Some((tensor, volume)) = pressure {
        let bar = omnimd_mlmd::barostat::BAR_PER_EV_PER_ANGSTROM3;
        record["volume_angstrom3"] = json!(volume);
        record["pressure_bar"] = json!((tensor[0][0] + tensor[1][1] + tensor[2][2]) / 3. * bar);
        record["pressure_tensor_bar"] = json!(tensor.map(|row| row.map(|p| p * bar)));
    }
    if state.ensemble.is_variable_cell() {
        record["barostat_work_ev"] = json!(state.barostat_work_ev);
    }
    if let Some(serde_json::Value::Object(entries)) = extra {
        for (key, value) in entries {
            record[key] = value;
        }
    }
    serde_json::to_writer(&mut *writer, &record)?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn write_checkpoint(
    path: &Path,
    md: StateView<'_>,
    state: &EnsembleState,
    hash: &str,
    fixed_parameters: Option<&omnimd_mlmd::parameters::FixedParameters>,
) -> Result<(), Box<dyn Error>> {
    let mut checkpoint = Checkpoint::of(md, hash.to_owned());
    checkpoint.ensemble_state = Some(state.clone());
    if let Some(fixed) = fixed_parameters {
        fixed.validate(&md.frame())?;
        checkpoint.schema_version = 3;
        checkpoint.fixed_parameters = Some(fixed.clone());
    }
    if state.ensemble.is_variable_cell() {
        checkpoint.schema_version = 4;
    }
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    serde_json::to_writer_pretty(&mut file, &checkpoint)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

/// What all trajectories of a run share.
struct Schedule {
    dynamics_units: DynamicsUnits,
    timestep_fs: f64,
    steps: u64,
    sample_every: u64,
    checkpoint_every: Option<u64>,
}

impl Schedule {
    /// Whether the frame after step `step` of the run is written.
    fn samples(&self, step: u64) -> bool {
        step % self.sample_every == 0 || step == self.steps
    }
}

/// One trajectory in the writing: its file, the model and parameters its
/// checkpoints are bound to, and the energies its completed record reports.
struct Recorder {
    output: PathBuf,
    writer: BufWriter<fs::File>,
    hash: String,
    fixed_parameters: Option<omnimd_mlmd::parameters::FixedParameters>,
    initial: f64,
    initial_effective: f64,
    max_drift: f64,
    max_effective_drift: f64,
    /// The members' disagreement on the sampled frames, in a run with
    /// model deviation.
    deviation: Option<DeviationLog>,
}

impl Recorder {
    /// Creates the trajectory at `output` with its header; the frame of the
    /// starting state follows with [`Self::sample`]. `input` is the request
    /// the system came from; `extra` are further header entries. `window`
    /// is given in a run with model deviation: its force window, if any.
    #[allow(clippy::too_many_arguments)]
    fn create(
        output: &Path,
        schedule: &Schedule,
        input: &SystemInput<'_>,
        md: StateView<'_>,
        state: &EnsembleState,
        hash: &str,
        fixed_parameters: Option<omnimd_mlmd::parameters::FixedParameters>,
        extra: serde_json::Value,
        window: Option<Option<ForceWindow>>,
    ) -> Result<Self, Box<dyn Error>> {
        if checkpoint_path(output, None).exists() {
            return Err("checkpoint output already exists".into());
        }
        let file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)?;
        let mut writer = BufWriter::new(file);
        let mut header = json!({
            "kind":"header", "schema":if state.ensemble.is_variable_cell() {"omnimd-mlmd-jsonl-v4"} else if fixed_parameters.is_some() {"omnimd-mlmd-jsonl-v3"} else {"omnimd-mlmd-jsonl-v2"}, "ensemble":state.ensemble,
            "dynamics_units":schedule.dynamics_units,
            "metadata":md.metadata(), "timestep_fs":schedule.timestep_fs,
            "dataset_frame":input.dataset_frame,
            "requested_steps":schedule.steps, "start_step":md.system().step,
            "masses_u":md.system().particles().mass, "model_sha256":hash,
            "degrees_of_freedom":md.system().degrees_of_freedom(),
            "initialize_velocities":input.initialize_velocities,
            "remove_center_of_mass":matches!(md.system().simulated_degrees_of_freedom, omnimd_core::DegreesOfFreedom::Frozen(3)),
            "checkpoint_every":schedule.checkpoint_every
        });
        if let Some(fixed) = &fixed_parameters {
            header["fixed_parameters"] = serde_json::to_value(fixed)?;
            header["parameter_policy"]=json!("fixed for the complete trajectory; atomic rows follow immutable input indices; model defaults bound to model SHA256");
        }
        if let Some(entries) = extra.as_object() {
            for (key, value) in entries {
                header[key] = value.clone();
            }
        }
        serde_json::to_writer(&mut writer, &header)?;
        writer.write_all(b"\n")?;
        let initial = md.total_energy_ev();
        Ok(Self {
            output: output.to_owned(),
            writer,
            hash: hash.to_owned(),
            fixed_parameters,
            initial,
            initial_effective: initial - state.heat_ev - state.barostat_work_ev,
            max_drift: 0.0,
            max_effective_drift: 0.0,
            deviation: window.map(DeviationLog::new),
        })
    }

    /// Writes the frame of the current state. In a run with model deviation
    /// `deviation` is what the members say of it: the frame carries it, and
    /// whether that makes it a candidate.
    fn sample(
        &mut self,
        md: StateView<'_>,
        state: &EnsembleState,
        deviation: Option<&ModelDeviation>,
    ) -> Result<(), Box<dyn Error>> {
        let extra = match (&mut self.deviation, deviation) {
            (Some(log), Some(statistics)) => {
                let candidate = log.note(statistics);
                Some(json!({"model_deviation": statistics, "candidate": candidate}))
            }
            (None, None) => None,
            _ => return Err("a run with model deviation has it for every sampled frame".into()),
        };
        write_snapshot(&mut self.writer, md, state, extra)
    }

    /// After step `step` of the run: the energy drift so far, the frame if
    /// this is a step to sample (with its model deviation, see
    /// [`Self::sample`]), and a checkpoint if one is due.
    fn after_step(
        &mut self,
        step: u64,
        schedule: &Schedule,
        md: StateView<'_>,
        state: &EnsembleState,
        deviation: Option<&ModelDeviation>,
    ) -> Result<(), Box<dyn Error>> {
        let total_energy_ev = md.total_energy_ev();
        self.max_drift = self.max_drift.max((total_energy_ev - self.initial).abs());
        self.max_effective_drift = self.max_effective_drift.max(
            (total_energy_ev - state.heat_ev - state.barostat_work_ev - self.initial_effective)
                .abs(),
        );
        if schedule.samples(step) {
            self.sample(md, state, deviation)?;
            self.writer.flush()?;
        }
        if schedule.checkpoint_every.is_some_and(|n| step % n == 0) && step != schedule.steps {
            write_checkpoint(
                &checkpoint_path(&self.output, Some(md.system().step)),
                md,
                state,
                &self.hash,
                self.fixed_parameters.as_ref(),
            )?;
        }
        Ok(())
    }

    /// The checkpoint to continue from and the completed record. Returns
    /// the path of the checkpoint.
    fn finish(
        mut self,
        schedule: &Schedule,
        md: StateView<'_>,
        state: &EnsembleState,
    ) -> Result<PathBuf, Box<dyn Error>> {
        let restart_output = checkpoint_path(&self.output, None);
        write_checkpoint(
            &restart_output,
            md,
            state,
            &self.hash,
            self.fixed_parameters.as_ref(),
        )?;
        let nve_drift = matches!(state.ensemble, Ensemble::Nve).then_some(self.max_drift);
        let mut completed = json!({
            "kind":"completed", "steps":schedule.steps, "final_step":md.system().step,
            "initial_total_energy_ev":self.initial, "final_total_energy_ev":md.total_energy_ev(),
            "max_absolute_energy_drift_ev":nve_drift,
            "max_absolute_effective_energy_drift_ev":self.max_effective_drift,
            "thermostat_heat_ev":state.heat_ev, "barostat_work_ev":state.barostat_work_ev,
            "barostat_moves":state.barostat_moves, "barostat_refused":state.barostat_refused,
            "checkpoint":restart_output
        });
        if let Some(log) = &self.deviation {
            // The counts of `model-deviation`, over the sampled frames.
            let counted = |count: u64| log.window.map(|_| count);
            completed["deviation"] = json!({"frames": log.frames,
                "accurate_frames": counted(log.accurate),
                "candidate_frames": counted(log.candidates),
                "failed_frames": counted(log.failed),
                "max_force_deviation_ev_per_angstrom": log.maximum});
        }
        serde_json::to_writer(&mut self.writer, &completed)?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        Ok(restart_output)
    }
}

fn worker(input: &Path, output: Option<&Path>) -> Result<(), Box<dyn Error>> {
    let request: Request = serde_json::from_str(&fs::read_to_string(input)?)?;
    let (mut md, mut state, hash, fixed_parameters, deviation) = prepare(&request)?;
    let Some(output) = output else {
        println!(
            "Valid model/run: {} atoms, {} DOF, {:.6} K, E = {:.12} eV, ensemble {:?}",
            md.system().size(),
            md.system().degrees_of_freedom(),
            md.temperature_kelvin(),
            md.evaluation().energy,
            state.ensemble
        );
        return Ok(());
    };
    let schedule = Schedule {
        dynamics_units: request.dynamics_units,
        timestep_fs: request.timestep_fs,
        steps: request.steps,
        sample_every: request.sample_every,
        checkpoint_every: request.checkpoint_every,
    };
    // What the other members of the model's ensemble say of the frame the
    // run is at, when it has them.
    let disagreement = |md: &mut MlipDynamics<Backend>| -> Result<_, Box<dyn Error>> {
        let frame = md.frame();
        Ok(deviations(md.potential_mut(), &[&frame])?.map(|mut all| all.remove(0)))
    };
    let mut recorder = Recorder::create(
        output,
        &schedule,
        &request.system(),
        md.view(),
        &state,
        &hash,
        fixed_parameters,
        match deviation {
            Some(header) => json!({"deviation": header}),
            None => json!({}),
        },
        request.deviation.as_ref().map(|d| d.force_window),
    )?;
    let first = disagreement(&mut md)?;
    recorder.sample(md.view(), &state, first.as_ref())?;
    let started = std::time::Instant::now();
    for step in 1..=request.steps {
        state.advance(&mut md)?;
        let now = match schedule.samples(step) {
            true => disagreement(&mut md)?,
            false => None,
        };
        recorder.after_step(step, &schedule, md.view(), &state, now.as_ref())?;
    }
    report_rate(request.steps, 1, started.elapsed().as_secs_f64());
    recorder.finish(&schedule, md.view(), &state)?;
    Ok(())
}

/// Says on standard error how fast the steps of a run went, trajectory
/// output included and everything before the first step left out.
fn report_rate(steps: u64, systems: usize, seconds: f64) {
    if steps > 0 {
        eprintln!(
            "omnimd-mlmd: {steps} steps of {systems} system(s) in {seconds:.3} s, {:.4} ms per step",
            seconds * 1e3 / steps as f64
        );
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args: Vec<_> = std::env::args_os().skip(1).collect();
    if deviation_command::dispatch(&args)?
        || dataset_commands::dispatch(&args)?
        || replicas_command::dispatch(&args)?
    {
        return Ok(());
    }
    if args.len() == 3 && args[0] == "--worker" {
        return worker(Path::new(&args[1]), Some(Path::new(&args[2])));
    }
    if args.len() == 2 && args[0] == "--check-worker" {
        return worker(Path::new(&args[1]), None);
    }
    if args.len() == 1 && (args[0] == "--help" || args[0] == "-h") {
        println!(
            "omnimd-mlmd: CPU/CUDA DeepMD C API / native se_e2_a dynamics (native CUDA requires native-cuda feature)\n\
Usage:\n  omnimd-mlmd prepare-labels REQUEST.json NEW_DIRECTORY | import-labels REQUEST.json NEW_DIRECTORY\n  omnimd-mlmd model-deviation REQUEST.json NEW_DEVIATIONS.jsonl | select-dataset REQUEST.json NEW_DIRECTORY\n  omnimd-mlmd train-native TRAIN.json\n  omnimd-mlmd minimize REQUEST.json NEW_MINIMIZATION.jsonl\n  omnimd-mlmd import-xyz IMPORT.json NEW_DATASET.jsonl\n  omnimd-mlmd capabilities\n  omnimd-mlmd model-info OPTIONS.json NEW_INFO.json\n  omnimd-mlmd inspect-dataset DATASET.jsonl NEW_REPORT.json\n  omnimd-mlmd import-deepmd IMPORT.json NEW_DATASET.jsonl | import-trajectory IMPORT.json NEW_DATASET.jsonl\n  omnimd-mlmd merge-datasets MERGE.json NEW_DATASET.jsonl | collect-candidates REQUEST.json NEW_DIRECTORY\n  omnimd-mlmd batch-eval EVAL.json NEW_PREDICTIONS.jsonl\n  omnimd-mlmd run REQUEST.json NEW_TRAJECTORY.jsonl | run-replicas REPLICAS.json NEW_DIRECTORY\n\
  omnimd-mlmd check REQUEST.json\n\
  omnimd-mlmd REQUEST.json NEW_TRAJECTORY.jsonl\n\
Units: angstrom, fs, u, K, eV, bar. Ensembles: nve, nvt_csvr, npt_csvr_crescale.\n\
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
    request.options.device().configure_worker(&mut child)?;
    let status = child.status()?;
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

#[cfg(test)]
mod backend_tests {
    use super::*;
    fn native_options(device: Option<&str>) -> BackendOptions {
        let mut value = json!({"native_model": Path::new(env!("CARGO_MANIFEST_DIR")).join("../native/fixtures/full-tanh-v1.json")});
        if let Some(device) = device {
            value["device"] = json!(device);
        }
        serde_json::from_value(value).unwrap()
    }
    #[test]
    fn native_cpu_is_default_and_does_not_load_cuda() {
        for device in [None, Some("cpu")] {
            let options = native_options(device);
            assert_eq!(options.device(), Device::Cpu);
            let loaded = options.load().unwrap();
            assert!(matches!(loaded, Backend::Native(_)));
        }
        let cuda = native_options(Some("cuda:2"));
        assert_eq!(cuda.device(), Device::Cuda(2));
        #[cfg(not(feature = "native-cuda"))]
        assert!(cuda
            .validate()
            .unwrap_err()
            .to_string()
            .contains("--features native-cuda"));
        #[cfg(feature = "native-cuda")]
        cuda.validate().unwrap(); // Validation only: never construct a GPU potential.
    }
    #[test]
    fn gpu_runtime_options_are_validated_before_any_driver_load() {
        let model =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../native/fixtures/full-tanh-v1.json");
        let parse = |device: Option<&str>, runtime: serde_json::Value| {
            let mut value = json!({"native_model": model, "gpu_runtime": runtime});
            if let Some(device) = device {
                value["device"] = json!(device);
            }
            serde_json::from_value::<BackendOptions>(value)
        };
        // The runtime needs a CUDA device.
        let cpu = parse(None, json!({})).unwrap();
        assert!(cpu
            .validate()
            .unwrap_err()
            .to_string()
            .contains("requires device cuda"));
        let gpu = parse(
            Some("cuda:0"),
            json!({"precision": "float32", "compress_min_distance": 0.5}),
        )
        .unwrap();
        let runtime = gpu.gpu_runtime().unwrap();
        assert_eq!(runtime.precision, NetworkPrecision::Float32);
        assert_eq!(runtime.compress_min_distance, Some(0.5));
        #[cfg(not(feature = "native-gpu"))]
        assert!(gpu
            .validate()
            .unwrap_err()
            .to_string()
            .contains("--features native-gpu"));
        #[cfg(feature = "native-gpu")]
        gpu.validate().unwrap(); // Validation only: never construct a GPU potential.
        assert_eq!(
            parse(Some("cuda:1"), json!({})).unwrap().gpu_runtime(),
            Some(GpuRuntimeOptions::default())
        );
        for bad in [
            json!({"compress_min_distance": 0.0}),
            json!({"table_step": 0.01}),
            json!({"compress_min_distance": 0.5, "table_step": -1.0}),
        ] {
            assert!(parse(Some("cuda:0"), bad).unwrap().validate().is_err());
        }
        // Unknown precisions and fields do not parse.
        for bad in [json!({"precision": "float16"}), json!({"tables": true})] {
            assert!(parse(Some("cuda:0"), bad).is_err());
        }
        // Without gpu_runtime, cuda:N still means the bounded native-cuda path.
        assert_eq!(native_options(Some("cuda:0")).gpu_runtime(), None);
    }
    #[test]
    fn parameter_requests_reject_native_and_cuda_before_loading_backends() {
        use omnimd_mlmd::parameters::{Field, Layout};
        let layout = Layout {
            fparam: vec![Field {
                name: "condition".into(),
                unit: "fixture".into(),
            }],
            aparam: vec![],
        };
        assert!(
            parameter_backend::validate_request(&native_options(Some("cpu")), Some(&layout))
                .is_err()
        );
        let capi:BackendOptions=serde_json::from_value(json!({"library":"/not-loaded/runtime","model":"/not-loaded/model.pb","device":"cuda:0","type_map":["O"],"units":"EvAngstrom"})).unwrap();
        assert!(parameter_backend::validate_request(&capi, Some(&layout)).is_err());
    }
    #[test]
    fn isolated_native_cuda_ordinal_is_zero_and_mismatches_fail_before_driver_load() {
        assert_eq!(native_cuda_ordinal(2, Some("5"), Some("2")).unwrap(), 0);
        assert_eq!(
            native_cuda_ordinal(1, Some("GPU-abc"), Some("1")).unwrap(),
            0
        );
        for mask in [None, Some(""), Some("-1"), Some("0,1"), Some("GPU-")] {
            assert!(native_cuda_ordinal(2, mask, Some("2")).is_err());
        }
        assert!(native_cuda_ordinal(2, Some("5"), Some("0")).is_err());
        assert!(native_cuda_ordinal(2, Some("5"), None).is_err());
    }
}
