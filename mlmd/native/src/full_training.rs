//! Bounded deterministic full-batch or explicitly sampled minibatch Adam for the supported tanh networks.
use crate::{
    full_gradient::{frame_gradient_with_virial, NetworkLocation, ParameterLayout},
    full_loss::{metrics, validate_loss, validate_record},
    training::{partition, FrameIdentity, Split, TrainConfig, TrainingDevice},
    training_data::{DatasetAccess, DatasetHeader},
    Model,
};
use omnimd_ml_contract::{Error, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashSet},
    time::Instant,
};

pub use crate::full_loss::{FullMetrics, VirialLoss};
pub use crate::learning_rate::{LearningRateReport, LearningRateSchedule};
pub use crate::minibatch::{BatchOrder, MiniBatchConfig, MiniBatchReport, SamplerState};
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CudaGradient {
    #[default]
    Hyperdual,
    Reverse,
}
impl CudaGradient {
    fn is_default(&self) -> bool {
        *self == Self::Hyperdual
    }
}
/// Execution chunks only; never changes minibatch normalization or Adam steps.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CudaTiling {
    pub max_frames: usize,
    pub max_atoms: usize,
}
impl CudaTiling {
    pub fn validate(&self) -> Result<()> {
        if !(1..=32).contains(&self.max_frames) || !(1..=1024).contains(&self.max_atoms) {
            return Err(Error(
                "CUDA tiling requires max_frames in 1..32 and max_atoms in 1..1024".into(),
            ));
        }
        Ok(())
    }
}
const VIRIAL_LOSS_CONTRACT: &str = "energy-force-row-virial-per-atom-nine-v1";

/// CPU v1 checkpoints remain byte-schema compatible (this default is omitted).
/// CUDA v2 records the compiled program and reported device/driver identity.
/// Backend identity and checkpoint loss_contract together cover host numerical
/// semantics. New virial semantics require v3; historical E/F semantics and
/// v1/v2 serialization stay unchanged. Opt-in reverse execution uses v4;
/// explicit minibatch sampling uses v5 on every backend; an explicit learning-rate
/// schedule uses v6. Tiled reverse execution takes precedence with v7; absent
/// schedules/tiling preserve older schemas.
/// PTX hashing alone does not cover host loss semantics.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "backend", rename_all = "snake_case", deny_unknown_fields)]
pub enum FullExecution {
    #[default]
    CpuReferenceV1,
    CudaHyperdualF64V1 {
        visible_device: i32,
        device_info: String,
        driver_version: i32,
        program_sha256: String,
    },
    CudaReverseF64V1 {
        visible_device: i32,
        device_info: String,
        driver_version: i32,
        program_sha256: String,
    },
    CudaTiledReverseF64V1 {
        visible_device: i32,
        device_info: String,
        driver_version: i32,
        program_sha256: String,
    },
}
impl FullExecution {
    fn is_cpu(&self) -> bool {
        matches!(self, Self::CpuReferenceV1)
    }
}
#[derive(Default, Serialize)]
pub struct FullTiming {
    /// Within this trainer call, through initial assessments, before the first update.
    pub startup_ms: f64,
    /// Indexed reader counter at that boundary; includes earlier caller reads.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub startup_frame_reads: Option<u64>,
    pub initialization_ms: f64,
    pub gradient_ms: f64,
    pub adam_ms: f64,
    pub assessment_ms: f64,
}

fn cache_disabled(value: &bool) -> bool {
    !*value
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FullConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minibatch: Option<MiniBatchConfig>,
    #[serde(default, skip_serializing_if = "cache_disabled")]
    pub fixed_geometry_cache: bool,
    #[serde(default, skip_serializing_if = "CudaGradient::is_default")]
    pub cuda_gradient: CudaGradient,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_tiling: Option<CudaTiling>,
    pub split: Split,
    /// Additional updates in this invocation, including after restart.
    pub steps: u64,
    pub energy_weight: f64,
    pub force_weight: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub virial_loss: Option<VirialLoss>,
    pub energy_scale_ev_per_atom: f64,
    pub force_scale_ev_per_angstrom: f64,
    /// Initial rate; absent/null schedule preserves historical constant behavior.
    pub learning_rate: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub learning_rate_schedule: Option<LearningRateSchedule>,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub gradient_clip_norm: Option<f64>,
    pub checkpoint_every: u64,
}
impl FullConfig {
    /// Rate for the next update after completed_updates globally committed updates.
    pub fn learning_rate_at(&self, completed_updates: u64) -> Result<f64> {
        self.learning_rate_schedule
            .as_ref()
            .unwrap_or(&LearningRateSchedule::Constant {})
            .rate(self.learning_rate, completed_updates)
    }
    fn validate_learning_rate_range(&self, start: u64) -> Result<()> {
        let end = start
            .checked_add(self.steps)
            .filter(|end| self.steps > 0 && *end <= crate::learning_rate::MAX_UPDATES)
            .ok_or_else(|| Error("full trainer global step exceeds limit".into()))?;
        // All supported schedules are nonincreasing. Check the entire requested
        // interval via its endpoints before any CUDA context/update/publication.
        self.learning_rate_at(start)?;
        self.learning_rate_at(end - 1)?;
        Ok(())
    }
    /// Legacy E/F component; pass virial_loss separately to the extended gradient API.
    pub fn loss(&self) -> TrainConfig {
        TrainConfig {
            split: self.split.clone(),
            fit_bias: false,
            energy_weight: self.energy_weight,
            force_weight: self.force_weight,
            energy_scale_ev_per_atom: self.energy_scale_ev_per_atom,
            force_scale_ev_per_angstrom: self.force_scale_ev_per_angstrom,
            ridge_inverse_ev2: 0.0,
            rank_rtol: 1e-12,
        }
    }
    pub(crate) fn validate(&self) -> Result<()> {
        validate_loss(&self.loss(), self.virial_loss.as_ref())?;
        if let Some(c) = &self.minibatch {
            c.validate()?;
        }
        if let Some(c) = &self.cuda_tiling {
            c.validate()?;
            if self.cuda_gradient != CudaGradient::Reverse {
                return Err(Error(
                    "cuda_tiling requires cuda_gradient reverse; no fallback".into(),
                ));
            }
        }
        if self.steps == 0
            || self.steps > 100_000
            || self.checkpoint_every == 0
            || !(self.learning_rate > 0.0 && self.learning_rate.is_finite())
            || !(0.0 <= self.beta1 && self.beta1 < 1.0 && 0.0 <= self.beta2 && self.beta2 < 1.0)
            || !(self.epsilon > 0.0 && self.epsilon.is_finite())
            || self
                .gradient_clip_norm
                .is_some_and(|x| !x.is_finite() || x <= 0.0)
        {
            return Err(Error(
                "invalid full trainer Adam configuration (steps must be 1..100000)".into(),
            ));
        }
        self.learning_rate_at(0)?;
        Ok(())
    }
    fn restart_key(&self) -> serde_json::Value {
        let mut key = serde_json::to_value(self).unwrap();
        key.as_object_mut().unwrap().remove("steps");
        // Geometry reuse does not change numerical state; safe to toggle at restart.
        key.as_object_mut().unwrap().remove("fixed_geometry_cache");
        key
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampler: Option<SamplerState>,
    pub schema: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loss_contract: Option<String>,
    #[serde(default, skip_serializing_if = "FullExecution::is_cpu")]
    pub execution: FullExecution,
    pub initial_model: Model,
    pub model: Model,
    pub config: FullConfig,
    pub step: u64,
    pub first_moment: Vec<f64>,
    pub second_moment: Vec<f64>,
    pub dataset_sha256: String,
    pub parameter_layout_sha256: String,
    pub training_indices: Vec<usize>,
    pub validation_indices: Vec<usize>,
}
#[derive(Serialize)]
pub struct GroupChange {
    pub parameters: usize,
    pub changed: usize,
    pub delta_l2: f64,
}
#[derive(Serialize)]
pub struct FullReport {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_access: Option<crate::indexed_dataset::AccessReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minibatch: Option<MiniBatchReport>,
    pub scope: String,
    pub algorithm: String,
    pub derivative_method: String,
    pub loss_contract: String,
    pub execution: FullExecution,
    pub execution_scope: String,
    pub timing: FullTiming,
    pub learning_rate: LearningRateReport,
    #[cfg(feature = "cuda")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cuda_gradient_profile: Option<crate::cuda::FullGradientProfile>,
    #[cfg(feature = "cuda")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fixed_geometry_cache: Option<crate::cuda::FixedGeometryStats>,
    #[cfg(feature = "cuda")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cuda_tiling: Option<crate::cuda::TiledResources>,
    pub config: FullConfig,
    pub step_start: u64,
    pub step_end: u64,
    pub parameters: usize,
    pub parameter_layout_sha256: String,
    pub dataset_header: DatasetHeader,
    pub dataset_sha256: String,
    pub source_sha256: BTreeMap<String, String>,
    pub training_frames: Vec<FrameIdentity>,
    pub validation_frames: Vec<FrameIdentity>,
    pub parameter_changes_from_initial: BTreeMap<String, GroupChange>,
    /// Start of this invocation; checkpoint anchor is separately preserved.
    pub initial_training: FullMetrics,
    pub final_training: FullMetrics,
    pub initial_validation: FullMetrics,
    pub final_validation: FullMetrics,
}
pub struct FullOutcome {
    pub checkpoint: Checkpoint,
    pub report: FullReport,
}

fn checkpoint_schema(execution: &FullExecution, config: &FullConfig) -> &'static str {
    if config.cuda_tiling.is_some() {
        "omnimd-native-full-checkpoint-v7"
    } else if config.learning_rate_schedule.is_some() {
        "omnimd-native-full-checkpoint-v6"
    } else if config.minibatch.is_some() {
        "omnimd-native-full-checkpoint-v5"
    } else if config.cuda_gradient == CudaGradient::Reverse {
        "omnimd-native-full-checkpoint-v4"
    } else if config.virial_loss.is_some() {
        "omnimd-native-full-checkpoint-v3"
    } else if execution.is_cpu() {
        "omnimd-native-full-checkpoint-v1"
    } else {
        "omnimd-native-full-checkpoint-v2"
    }
}
fn loss_contract(config: &FullConfig) -> Option<&'static str> {
    if config.virial_loss.is_some() {
        Some(VIRIAL_LOSS_CONTRACT)
    } else if config.cuda_gradient == CudaGradient::Reverse
        || config.minibatch.is_some()
        || config.learning_rate_schedule.is_some()
    {
        Some("energy-force-per-atom-v1")
    } else {
        None
    }
}
fn validate_virial_scales(
    dataset: &dyn DatasetAccess,
    partitions: &[&[usize]],
    virial: Option<&VirialLoss>,
) -> Result<()> {
    if let Some(v) = virial {
        for indices in partitions {
            for &i in *indices {
                v.scale(indices.len(), dataset.atoms(i)?)?;
            }
        }
    }
    Ok(())
}
fn validate_minibatch_scales(
    dataset: &dyn DatasetAccess,
    training: &[usize],
    config: &FullConfig,
) -> Result<()> {
    if let Some(c) = &config.minibatch {
        let loss = config.loss();
        let full = c.batch_size.min(training.len());
        let remainder = training.len() % c.batch_size;
        for count in [full, if remainder == 0 { full } else { remainder }] {
            for &i in training {
                let n = dataset.atoms(i)?;
                let (es, fs) = crate::training::scales(&loss, count, n);
                for (weight, scale) in [(config.energy_weight, es), (config.force_weight, fs)] {
                    if weight > 0.
                        && (!scale.is_finite()
                            || scale <= 0.
                            || !(2. * scale * scale).is_finite()
                            || 2. * scale * scale <= 0.)
                    {
                        return Err(Error(
                            "minibatch E/F normalization underflow/overflow".into(),
                        ));
                    }
                }
                if let Some(v) = &config.virial_loss {
                    v.scale(count, n)?;
                }
            }
        }
    }
    Ok(())
}
fn validate_dataset(
    model: &Model,
    dataset: &dyn DatasetAccess,
    virial: Option<&VirialLoss>,
) -> Result<()> {
    let metadata = model.validate()?;
    if dataset.header().type_map != model.type_map {
        return Err(Error(
            "dataset and model ordered type maps must match".into(),
        ));
    }
    if dataset.len()
        > if dataset.indexed() {
            crate::indexed_dataset::MAX_FRAMES
        } else {
            1024
        }
    {
        return Err(Error("full trainer dataset frame count exceeds access mode limit (memory 1024, indexed 100000)".into()));
    }
    let mut ids = HashSet::new();
    for i in 0..dataset.len() {
        let record = dataset.frame(i)?;
        record.frame.type_ids(&metadata)?;
        validate_record(&record, virial)?;
        if record.frame.positions.len() > 64 {
            return Err(Error(
                "full trainer supports at most 64 atoms per frame".into(),
            ));
        }
        if record.index != i as u64
            || record.id.is_empty()
            || (!dataset.indexed() && !ids.insert(record.id.clone()))
        {
            return Err(Error("invalid dataset frame identities".into()));
        }
    }
    Ok(())
}
impl Checkpoint {
    pub fn start(model: &Model, dataset: &dyn DatasetAccess, config: &FullConfig) -> Result<Self> {
        config.validate()?;
        config.validate_learning_rate_range(0)?;
        if dataset.indexed() && config.minibatch.is_none() {
            return Err(Error(
                "indexed dataset requires explicit minibatch configuration".into(),
            ));
        }
        validate_dataset(model, dataset, config.virial_loss.as_ref())?;
        let layout = ParameterLayout::new(model)?;
        let (training_indices, validation_indices) = partition(dataset, &config.split)?;
        validate_virial_scales(
            dataset,
            &[&training_indices, &validation_indices],
            config.virial_loss.as_ref(),
        )?;
        validate_minibatch_scales(dataset, &training_indices, config)?;
        let sampler = config
            .minibatch
            .as_ref()
            .map(|c| SamplerState::for_step(c, &training_indices, 0))
            .transpose()?;
        let p = layout.parameters.len();
        Ok(Self {
            sampler,
            schema: checkpoint_schema(&FullExecution::CpuReferenceV1, config).into(),
            loss_contract: loss_contract(config).map(str::to_owned),
            execution: FullExecution::CpuReferenceV1,
            initial_model: model.clone(),
            model: model.clone(),
            config: config.clone(),
            step: 0,
            first_moment: vec![0.0; p],
            second_moment: vec![0.0; p],
            dataset_sha256: dataset.dataset_sha256().to_owned(),
            parameter_layout_sha256: layout.sha256(),
            training_indices,
            validation_indices,
        })
    }
    /// Validates all host inputs and resume mode before creating a CUDA context.
    pub fn validate_for_device(
        &self,
        dataset: &dyn DatasetAccess,
        config: &FullConfig,
        device: TrainingDevice,
    ) -> Result<()> {
        self.validated_layout_for_device(dataset, config, device)
            .map(|_| ())
    }
    // Validation and its layout result are one operation. No cross-call cache:
    // public callers may edit checkpoint/model/data between invocations.
    fn validated_layout_for_device(
        &self,
        dataset: &dyn DatasetAccess,
        config: &FullConfig,
        device: TrainingDevice,
    ) -> Result<ParameterLayout> {
        device.validate_full()?;
        if config.cuda_gradient == CudaGradient::Reverse && device == TrainingDevice::Cpu {
            return Err(Error(
                "cuda_gradient reverse requires an explicit CUDA device; no CPU fallback".into(),
            ));
        }
        let layout = self.validate(dataset, config)?;
        if config.fixed_geometry_cache && device == TrainingDevice::Cpu {
            return Err(Error(
                "fixed_geometry_cache requires explicit CUDA full training; no fallback".into(),
            ));
        }
        #[cfg(feature = "cuda")]
        if config.fixed_geometry_cache {
            crate::fixed_geometry::estimate(&self.model, dataset, &self.training_indices)?;
        }
        config.validate_learning_rate_range(self.step)?;
        if self.step > 0 {
            let compatible = match (&self.execution, device) {
                (FullExecution::CpuReferenceV1, TrainingDevice::Cpu) => true,
                (
                    FullExecution::CudaHyperdualF64V1 { visible_device, .. },
                    TrainingDevice::Cuda {
                        visible_device: requested,
                    },
                ) => {
                    config.cuda_gradient == CudaGradient::Hyperdual && *visible_device == requested
                }
                (
                    FullExecution::CudaReverseF64V1 { visible_device, .. },
                    TrainingDevice::Cuda {
                        visible_device: requested,
                    },
                ) => {
                    config.cuda_gradient == CudaGradient::Reverse
                        && config.cuda_tiling.is_none()
                        && *visible_device == requested
                }
                (
                    FullExecution::CudaTiledReverseF64V1 { visible_device, .. },
                    TrainingDevice::Cuda {
                        visible_device: requested,
                    },
                ) => {
                    config.cuda_gradient == CudaGradient::Reverse
                        && config.cuda_tiling.is_some()
                        && *visible_device == requested
                }
                _ => false,
            };
            if !compatible {
                return Err(Error("checkpoint training backend/device changed; resume rejected (initialize explicitly from model for a new optimizer)".into()));
            }
        }
        #[cfg(feature = "cuda")]
        if matches!(device, TrainingDevice::Cuda { .. }) {
            let atoms = (0..dataset.len())
                .try_fold(0usize, |n, i| Ok::<_, Error>(n.max(dataset.atoms(i)?)))?;
            crate::cuda::full_work_estimate(&self.model, atoms)?;
            if config.cuda_gradient == CudaGradient::Reverse {
                crate::cuda::reverse_resources(&self.model, atoms)?;
            }
            if let Some(c) = &config.cuda_tiling {
                let r = crate::cuda::tiled_resources(&self.model, c)?;
                if atoms > r.effective_max_atoms {
                    return Err(Error(format!(
                        "CUDA tile cannot fit largest frame ({atoms} atoms; effective capacity {})",
                        r.effective_max_atoms
                    )));
                }
            }
        }
        Ok(layout)
    }
    fn validate(
        &self,
        dataset: &dyn DatasetAccess,
        config: &FullConfig,
    ) -> Result<ParameterLayout> {
        config.validate()?;
        self.config.validate()?;
        if dataset.indexed() && config.minibatch.is_none() {
            return Err(Error(
                "indexed dataset requires explicit minibatch configuration".into(),
            ));
        }
        validate_dataset(&self.model, dataset, config.virial_loss.as_ref())?;
        let layout = ParameterLayout::new(&self.model)?;
        let initial_layout = ParameterLayout::new(&self.initial_model)?;
        let schema = checkpoint_schema(&self.execution, &self.config);
        let contract = loss_contract(&self.config);
        if self.schema != schema
            || self.loss_contract.as_deref() != contract
            || self.config.restart_key() != config.restart_key()
            || self.dataset_sha256 != dataset.dataset_sha256()
            || self.parameter_layout_sha256 != layout.sha256()
            || initial_layout.sha256() != layout.sha256()
            || self.step > 1_000_000_000
        {
            return Err(Error(
                "checkpoint schema, dataset hash, immutable configuration, layout or step mismatch"
                    .into(),
            ));
        }
        // The initial model anchors every frozen descriptor/normalization/format field.
        let mut restored = self.model.clone();
        layout.apply(&mut restored, &initial_layout.values(&self.initial_model))?;
        if serde_json::to_value(restored).unwrap()
            != serde_json::to_value(&self.initial_model).unwrap()
        {
            return Err(Error("checkpoint changed frozen model fields".into()));
        }
        let (training, validation) = partition(dataset, &config.split)?;
        validate_virial_scales(
            dataset,
            &[&training, &validation],
            config.virial_loss.as_ref(),
        )?;
        validate_minibatch_scales(dataset, &training, config)?;
        match (&config.minibatch, &self.sampler) {
            (Some(c), Some(s)) => s.validate(c, &training, self.step)?,
            (None, None) => {}
            _ => {
                return Err(Error(
                    "checkpoint minibatch configuration/sampler presence mismatch".into(),
                ))
            }
        }
        let p = layout.parameters.len();
        if training != self.training_indices
            || validation != self.validation_indices
            || self.first_moment.len() != p
            || self.second_moment.len() != p
            || self.first_moment.iter().any(|x| !x.is_finite())
            || self
                .second_moment
                .iter()
                .any(|x| !x.is_finite() || *x < 0.0)
            || self.step == 0
                && (self.first_moment.iter().any(|x| *x != 0.0)
                    || self.second_moment.iter().any(|x| *x != 0.0))
        {
            return Err(Error("checkpoint split or Adam state mismatch".into()));
        }
        Ok(layout)
    }
}

/// Full-batch default or explicit deterministic minibatch updates. Callback receives complete last-iterate
/// model and matching Adam moments/step; its errors stop training and propagate.
pub fn train_full(
    state: Checkpoint,
    dataset: &dyn DatasetAccess,
    config: &FullConfig,
    save: impl FnMut(&Checkpoint) -> Result<()>,
) -> Result<FullOutcome> {
    train_full_with_device(state, dataset, config, TrainingDevice::Cpu, save)
}

pub fn train_full_with_device(
    mut state: Checkpoint,
    dataset: &dyn DatasetAccess,
    config: &FullConfig,
    device: TrainingDevice,
    mut save: impl FnMut(&Checkpoint) -> Result<()>,
) -> Result<FullOutcome> {
    let initialization = Instant::now();
    let layout = state.validated_layout_for_device(dataset, config, device)?;
    #[cfg(feature = "cuda")]
    let mut cuda = if let TrainingDevice::Cuda { visible_device } = device {
        let cache = if config.fixed_geometry_cache {
            Some(crate::fixed_geometry::FixedGeometryCache::new(
                &state.model,
                dataset,
                &state.training_indices,
            )?)
        } else {
            None
        };
        let mut gpu = crate::cuda::CudaFullTrainer::restore(&state, visible_device)?;
        gpu.attach_fixed_geometry_cache(cache)?;
        state.execution = gpu.execution().clone();
        state.schema = checkpoint_schema(&state.execution, config).into();
        Some(gpu)
    } else {
        None
    };
    #[cfg(feature = "cuda")]
    let mut cuda_gradient_profile = cuda
        .as_ref()
        .map(|_| crate::cuda::FullGradientProfile::default());
    let mut timing = FullTiming {
        initialization_ms: initialization.elapsed().as_secs_f64() * 1000.0,
        ..FullTiming::default()
    };
    let start = state.step;
    let end = start
        .checked_add(config.steps)
        .filter(|s| *s <= 1_000_000_000)
        .ok_or_else(|| Error("full trainer global step exceeds limit".into()))?;
    let loss = config.loss();
    let assessment = Instant::now();
    let initial_training = metrics(
        &state.model,
        dataset,
        &state.training_indices,
        &loss,
        config.virial_loss.as_ref(),
    )?;
    let initial_validation = metrics(
        &state.model,
        dataset,
        &state.validation_indices,
        &loss,
        config.virial_loss.as_ref(),
    )?;
    timing.assessment_ms += assessment.elapsed().as_secs_f64() * 1000.0;
    state.config = config.clone();
    let mut values = layout.values(&state.model);
    let mut minibatch = state.sampler.as_ref().map(|s| MiniBatchReport {
        normalization: concat!(
            "arithmetic mean of actual batch frames; unchanged per-frame atom/component scales; ",
            "no cross-epoch fill or drop"
        )
        .into(),
        start: s.clone(),
        end: s.clone(),
        frames_processed: 0,
        partial_batches: 0,
        batch_size_counts: BTreeMap::new(),
    });
    timing.startup_ms = initialization.elapsed().as_secs_f64() * 1000.;
    timing.startup_frame_reads = dataset.access_report().map(|r| r.frame_reads);
    while state.step < end {
        let phase = Instant::now();
        let mut gradient = vec![0.0; values.len()];
        let batch = match (&state.sampler, &config.minibatch) {
            (Some(s), Some(c)) => s.batch(c)?,
            _ => state.training_indices.as_slice(),
        };
        let batch_len = batch.len();
        // Prepare sampler progress before touching optimizer state, but publish it only on success.
        let mut next_sampler = state.sampler.clone();
        if let (Some(s), Some(c)) = (&mut next_sampler, &config.minibatch) {
            s.advance(c, &state.training_indices)?;
        }
        #[cfg(feature = "cuda")]
        let tiled = if config.cuda_tiling.is_some() {
            let gpu = cuda
                .as_mut()
                .ok_or_else(|| Error("CUDA tiling requires CUDA".into()))?;
            gradient = gpu
                .batch_gradient_with_virial(dataset, batch, &loss, config.virial_loss.as_ref())?
                .1;
            if let Some(profile) = &mut cuda_gradient_profile {
                profile.add(gpu.last_profile());
            }
            true
        } else {
            false
        };
        #[cfg(not(feature = "cuda"))]
        let tiled = false;
        if !tiled {
            for &i in batch {
                let record = dataset.frame(i)?;
                #[cfg(feature = "cuda")]
                let (_, frame) = match &mut cuda {
                    Some(gpu) => gpu.frame_gradient_with_virial(
                        &record,
                        &loss,
                        config.virial_loss.as_ref(),
                        batch_len,
                    )?,
                    None => frame_gradient_with_virial(
                        &state.model,
                        &record,
                        &loss,
                        config.virial_loss.as_ref(),
                        batch_len,
                    )?,
                };
                #[cfg(not(feature = "cuda"))]
                let (_, frame) = frame_gradient_with_virial(
                    &state.model,
                    &record,
                    &loss,
                    config.virial_loss.as_ref(),
                    batch_len,
                )?;
                #[cfg(feature = "cuda")]
                if let (Some(gpu), Some(profile)) = (&cuda, &mut cuda_gradient_profile) {
                    profile.add(gpu.last_profile());
                }
                for (g, f) in gradient.iter_mut().zip(frame) {
                    *g += f;
                }
            }
        }
        // hypot avoids overflow while computing a norm for optional clipping.
        let norm = gradient.iter().fold(0.0_f64, |a, g| a.hypot(*g));
        if !norm.is_finite() {
            return Err(Error("nonfinite batch gradient".into()));
        }
        if let Some(max) = config.gradient_clip_norm {
            if norm > max {
                for g in &mut gradient {
                    *g *= max / norm;
                }
            }
        }
        timing.gradient_ms += phase.elapsed().as_secs_f64() * 1000.0;
        let phase = Instant::now();
        let learning_rate = config.learning_rate_at(state.step)?;
        let step = state.step + 1;
        #[cfg(feature = "cuda")]
        let cuda_updated = if let Some(gpu) = &mut cuda {
            gpu.adam_step(&gradient, config)?;
            values = layout.values(gpu.model());
            let (first, second) = gpu.moments();
            state.first_moment = first.to_vec();
            state.second_moment = second.to_vec();
            true
        } else {
            false
        };
        #[cfg(not(feature = "cuda"))]
        let cuda_updated = false;
        if !cuda_updated {
            let c1 = 1.0 - config.beta1.powf(step as f64);
            let c2 = 1.0 - config.beta2.powf(step as f64);
            for i in 0..values.len() {
                state.first_moment[i] =
                    config.beta1 * state.first_moment[i] + (1.0 - config.beta1) * gradient[i];
                state.second_moment[i] = config.beta2 * state.second_moment[i]
                    + (1.0 - config.beta2) * gradient[i] * gradient[i];
                values[i] -= learning_rate * (state.first_moment[i] / c1)
                    / ((state.second_moment[i] / c2).sqrt() + config.epsilon);
            }
        }
        if state
            .first_moment
            .iter()
            .chain(&state.second_moment)
            .any(|x| !x.is_finite())
        {
            return Err(Error(
                "nonfinite Adam moments; last completed checkpoint remains usable".into(),
            ));
        }
        layout.apply(&mut state.model, &values)?;
        state.step = step;
        state.sampler = next_sampler;
        if let Some(report) = &mut minibatch {
            report.frames_processed += batch_len as u64;
            report.partial_batches +=
                u64::from(batch_len < config.minibatch.as_ref().unwrap().batch_size);
            *report.batch_size_counts.entry(batch_len).or_default() += 1;
            report.end = state.sampler.as_ref().unwrap().clone();
        }
        timing.adam_ms += phase.elapsed().as_secs_f64() * 1000.0;
        if step % config.checkpoint_every == 0 || step == end {
            dataset.verify()?;
            save(&state)?;
        }
    }
    let mut changes: BTreeMap<String, GroupChange> = BTreeMap::new();
    for p in &layout.parameters {
        let group = match p.location {
            NetworkLocation::Embedding { .. } => "embedding",
            NetworkLocation::Fitting { species }
                if p.layer + 1 == state.model.fitting[species].layers.len() =>
            {
                "fitting_output"
            }
            _ => "fitting_hidden",
        };
        let change = changes.entry(group.into()).or_insert(GroupChange {
            parameters: 0,
            changed: 0,
            delta_l2: 0.0,
        });
        let delta = p.value(&state.model) - p.value(&state.initial_model);
        change.parameters += 1;
        change.changed += usize::from(delta != 0.0);
        change.delta_l2 = change.delta_l2.hypot(delta);
    }
    let identity = |i: &usize| FrameIdentity {
        index: dataset.index(*i).expect("validated dataset index"),
        id: dataset.id(*i).expect("validated dataset ID").to_owned(),
    };
    let phase = Instant::now();
    let final_training = metrics(
        &state.model,
        dataset,
        &state.training_indices,
        &loss,
        config.virial_loss.as_ref(),
    )?;
    let final_validation = metrics(
        &state.model,
        dataset,
        &state.validation_indices,
        &loss,
        config.virial_loss.as_ref(),
    )?;
    timing.assessment_ms += phase.elapsed().as_secs_f64() * 1000.0;
    let derivative_method = if config.cuda_gradient == CudaGradient::Reverse {
        "CUDA cached forward duals with explicit reverse accumulation; exact mixed derivatives, no per-parameter forward replay"
    } else if state.execution.is_cpu() {
        "reverse AD over differentiable coordinate-directional tangents; exact mixed parameter-coordinate derivatives"
    } else {
        "CUDA forward parameter/coordinate hyperduals; exact mixed derivatives, no finite differences"
    };
    let execution_scope = if state.execution.is_cpu() {
        "CPU float64 reference"
    } else {
        "CUDA all-neural-parameter mixed derivatives and Adam; CPU inference/loss cotangents, geometry, stable atom/frame reduction, clipping and assessment"
    };
    #[cfg(feature = "cuda")]
    let fixed_geometry_cache = cuda.as_ref().and_then(|gpu| gpu.fixed_geometry_stats());
    #[cfg(feature = "cuda")]
    let cuda_tiling = cuda
        .as_ref()
        .and_then(|gpu| gpu.tiling_resources().cloned());
    let algorithm = match (minibatch.is_some(), config.learning_rate_schedule.as_ref()) {
        (true, Some(LearningRateSchedule::Exponential { .. })) =>
            "deterministic minibatch Adam; global-update exponential staircase learning rate; last iterate export",
        (false, Some(LearningRateSchedule::Exponential { .. })) =>
            "deterministic full-batch Adam; global-update exponential staircase learning rate; last iterate export",
        (true, _) => "deterministic minibatch Adam; constant learning rate per completed update; last iterate export",
        (false, _) => "deterministic full-batch Adam; constant learning rate; last iterate export",
    };
    let learning_rate = LearningRateReport {
        contract: "omnimd-optimizer-update-learning-rate-v1",
        schedule: config
            .learning_rate_schedule
            .clone()
            .unwrap_or(LearningRateSchedule::Constant {}),
        base_rate: config.learning_rate,
        first_update: start + 1,
        last_update: end,
        first_rate: config.learning_rate_at(start)?,
        last_rate: config.learning_rate_at(end - 1)?,
    };
    dataset.verify()?;
    let report=FullReport {
        dataset_access:dataset.access_report(),
        learning_rate,
        minibatch,
        scope:"all embedding/fitting weights, biases and timesteps; fixed normalization, geometry settings and atom_energy_bias".into(),
        algorithm:algorithm.into(),
        loss_contract:state.loss_contract.clone().unwrap_or_else(|| "energy-force-per-atom-v1".into()),
        #[cfg(feature="cuda")] cuda_gradient_profile,
        #[cfg(feature="cuda")] fixed_geometry_cache,
        #[cfg(feature="cuda")] cuda_tiling,
        derivative_method:derivative_method.into(), execution:state.execution.clone(), execution_scope:execution_scope.into(), timing,
        config:config.clone(),step_start:start,step_end:end,parameters:values.len(),parameter_layout_sha256:layout.sha256(),
        dataset_header:dataset.header().clone(),dataset_sha256:dataset.dataset_sha256().to_owned(),source_sha256:dataset.source_sha256().clone(),
        training_frames:state.training_indices.iter().map(identity).collect(),validation_frames:state.validation_indices.iter().map(identity).collect(),
        parameter_changes_from_initial:changes,initial_training,initial_validation,
        final_training, final_validation,
    };
    Ok(FullOutcome {
        checkpoint: state,
        report,
    })
}
