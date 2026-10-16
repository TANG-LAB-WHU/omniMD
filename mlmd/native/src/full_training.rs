//! Bounded deterministic full-batch Adam for the supported tanh networks.
use crate::{
    full_gradient::{frame_gradient, NetworkLocation, ParameterLayout},
    training::{metrics, partition, FrameIdentity, Metrics, Split, TrainConfig, TrainingDevice},
    training_data::{validate_labels, Dataset, DatasetHeader},
    Model,
};
use omnimd_ml_contract::{Error, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashSet},
    time::Instant,
};

/// CPU v1 checkpoints remain byte-schema compatible (this default is omitted).
/// CUDA v2 records the compiled program and reported device/driver identity.
/// The backend version also covers host loss/reduction/update semantics; bump it
/// when those numerical semantics change. PTX hashing alone does not cover them.
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
}
impl FullExecution {
    fn is_cpu(&self) -> bool {
        matches!(self, Self::CpuReferenceV1)
    }
}
#[derive(Default, Serialize)]
pub struct FullTiming {
    pub initialization_ms: f64,
    pub gradient_ms: f64,
    pub adam_ms: f64,
    pub assessment_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FullConfig {
    pub split: Split,
    /// Additional updates in this invocation, including after restart.
    pub steps: u64,
    pub energy_weight: f64,
    pub force_weight: f64,
    pub energy_scale_ev_per_atom: f64,
    pub force_scale_ev_per_angstrom: f64,
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub gradient_clip_norm: Option<f64>,
    pub checkpoint_every: u64,
}
impl FullConfig {
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
        self.loss().validate()?;
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
        Ok(())
    }
    fn restart_key(&self) -> serde_json::Value {
        let mut key = serde_json::to_value(self).unwrap();
        key.as_object_mut().unwrap().remove("steps");
        key
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    pub schema: String,
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
    pub scope: String,
    pub algorithm: String,
    pub derivative_method: String,
    pub execution: FullExecution,
    pub execution_scope: String,
    pub timing: FullTiming,
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
    pub initial_training: Metrics,
    pub final_training: Metrics,
    pub initial_validation: Metrics,
    pub final_validation: Metrics,
}
pub struct FullOutcome {
    pub checkpoint: Checkpoint,
    pub report: FullReport,
}

fn validate_dataset(model: &Model, dataset: &Dataset) -> Result<()> {
    let metadata = model.validate()?;
    if dataset.header.type_map != model.type_map {
        return Err(Error(
            "dataset and model ordered type maps must match".into(),
        ));
    }
    if dataset.frames.len() > 1024 {
        return Err(Error("full trainer supports at most 1024 frames".into()));
    }
    let mut ids = HashSet::new();
    for (i, record) in dataset.frames.iter().enumerate() {
        record.frame.type_ids(&metadata)?;
        validate_labels(record)?;
        if record.frame.positions.len() > 64 {
            return Err(Error(
                "full trainer supports at most 64 atoms per frame".into(),
            ));
        }
        if record.index != i as u64 || record.id.is_empty() || !ids.insert(&record.id) {
            return Err(Error("invalid dataset frame identities".into()));
        }
    }
    Ok(())
}
impl Checkpoint {
    pub fn start(model: &Model, dataset: &Dataset, config: &FullConfig) -> Result<Self> {
        config.validate()?;
        validate_dataset(model, dataset)?;
        let layout = ParameterLayout::new(model)?;
        let (training_indices, validation_indices) = partition(dataset, &config.split)?;
        let p = layout.parameters.len();
        Ok(Self {
            schema: "omnimd-native-full-checkpoint-v1".into(),
            execution: FullExecution::CpuReferenceV1,
            initial_model: model.clone(),
            model: model.clone(),
            config: config.clone(),
            step: 0,
            first_moment: vec![0.0; p],
            second_moment: vec![0.0; p],
            dataset_sha256: dataset.dataset_sha256.clone(),
            parameter_layout_sha256: layout.sha256(),
            training_indices,
            validation_indices,
        })
    }
    /// Validates all host inputs and resume mode before creating a CUDA context.
    pub fn validate_for_device(
        &self,
        dataset: &Dataset,
        config: &FullConfig,
        device: TrainingDevice,
    ) -> Result<()> {
        device.validate_full()?;
        self.validate(dataset, config)?;
        if self
            .step
            .checked_add(config.steps)
            .is_none_or(|end| end > 1_000_000_000)
        {
            return Err(Error("full trainer global step exceeds limit".into()));
        }
        if self.step > 0 {
            let compatible = match (&self.execution, device) {
                (FullExecution::CpuReferenceV1, TrainingDevice::Cpu) => true,
                (
                    FullExecution::CudaHyperdualF64V1 { visible_device, .. },
                    TrainingDevice::Cuda {
                        visible_device: requested,
                    },
                ) => *visible_device == requested,
                _ => false,
            };
            if !compatible {
                return Err(Error("checkpoint training backend/device changed; resume rejected (initialize explicitly from model for a new optimizer)".into()));
            }
        }
        #[cfg(feature = "cuda")]
        if matches!(device, TrainingDevice::Cuda { .. }) {
            let atoms = dataset
                .frames
                .iter()
                .map(|f| f.frame.positions.len())
                .max()
                .unwrap_or(0);
            crate::cuda::full_work_estimate(&self.model, atoms)?;
        }
        Ok(())
    }
    fn validate(&self, dataset: &Dataset, config: &FullConfig) -> Result<ParameterLayout> {
        config.validate()?;
        self.config.validate()?;
        validate_dataset(&self.model, dataset)?;
        let layout = ParameterLayout::new(&self.model)?;
        let initial_layout = ParameterLayout::new(&self.initial_model)?;
        let schema = if self.execution.is_cpu() {
            "omnimd-native-full-checkpoint-v1"
        } else {
            "omnimd-native-full-checkpoint-v2"
        };
        if self.schema != schema
            || self.config.restart_key() != config.restart_key()
            || self.dataset_sha256 != dataset.dataset_sha256
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

/// Full-batch updates in stable dataset order. Callback receives complete last-iterate
/// model and matching Adam moments/step; its errors stop training and propagate.
pub fn train_full(
    state: Checkpoint,
    dataset: &Dataset,
    config: &FullConfig,
    save: impl FnMut(&Checkpoint) -> Result<()>,
) -> Result<FullOutcome> {
    train_full_with_device(state, dataset, config, TrainingDevice::Cpu, save)
}

pub fn train_full_with_device(
    mut state: Checkpoint,
    dataset: &Dataset,
    config: &FullConfig,
    device: TrainingDevice,
    mut save: impl FnMut(&Checkpoint) -> Result<()>,
) -> Result<FullOutcome> {
    let initialization = Instant::now();
    state.validate_for_device(dataset, config, device)?;
    let layout = state.validate(dataset, config)?;
    #[cfg(feature = "cuda")]
    let mut cuda = if let TrainingDevice::Cuda { visible_device } = device {
        let gpu = crate::cuda::CudaFullTrainer::restore(&state, visible_device)?;
        state.execution = gpu.execution().clone();
        state.schema = "omnimd-native-full-checkpoint-v2".into();
        Some(gpu)
    } else {
        None
    };
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
    let initial_training = metrics(&state.model, dataset, &state.training_indices, &loss)?;
    let initial_validation = metrics(&state.model, dataset, &state.validation_indices, &loss)?;
    timing.assessment_ms += assessment.elapsed().as_secs_f64() * 1000.0;
    state.config = config.clone();
    let mut values = layout.values(&state.model);
    while state.step < end {
        let phase = Instant::now();
        let mut gradient = vec![0.0; values.len()];
        for &i in &state.training_indices {
            #[cfg(feature = "cuda")]
            let (_, frame) = match &mut cuda {
                Some(gpu) => {
                    gpu.frame_gradient(&dataset.frames[i], &loss, state.training_indices.len())?
                }
                None => frame_gradient(
                    &state.model,
                    &dataset.frames[i],
                    &loss,
                    state.training_indices.len(),
                )?,
            };
            #[cfg(not(feature = "cuda"))]
            let (_, frame) = frame_gradient(
                &state.model,
                &dataset.frames[i],
                &loss,
                state.training_indices.len(),
            )?;
            for (g, f) in gradient.iter_mut().zip(frame) {
                *g += f;
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
                values[i] -= config.learning_rate * (state.first_moment[i] / c1)
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
        timing.adam_ms += phase.elapsed().as_secs_f64() * 1000.0;
        if step % config.checkpoint_every == 0 || step == end {
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
        index: dataset.frames[*i].index,
        id: dataset.frames[*i].id.clone(),
    };
    let phase = Instant::now();
    let final_training = metrics(&state.model, dataset, &state.training_indices, &loss)?;
    let final_validation = metrics(&state.model, dataset, &state.validation_indices, &loss)?;
    timing.assessment_ms += phase.elapsed().as_secs_f64() * 1000.0;
    let derivative_method = if state.execution.is_cpu() {
        "reverse AD over differentiable coordinate-directional tangents; exact mixed parameter-coordinate derivatives"
    } else {
        "CUDA forward parameter/coordinate hyperduals; exact mixed derivatives, no finite differences"
    };
    let execution_scope = if state.execution.is_cpu() {
        "CPU float64 reference"
    } else {
        "CUDA all-neural-parameter mixed derivatives and Adam; CPU inference/loss cotangents, geometry, stable atom/frame reduction, clipping and assessment"
    };
    let report=FullReport {
        scope:"all embedding/fitting weights, biases and timesteps; fixed normalization, geometry settings and atom_energy_bias".into(),
        algorithm:"deterministic full-batch Adam; constant learning rate; last iterate export".into(),
        derivative_method:derivative_method.into(), execution:state.execution.clone(), execution_scope:execution_scope.into(), timing,
        config:config.clone(),step_start:start,step_end:end,parameters:values.len(),parameter_layout_sha256:layout.sha256(),
        dataset_header:dataset.header.clone(),dataset_sha256:dataset.dataset_sha256.clone(),source_sha256:dataset.source_sha256.clone(),
        training_frames:state.training_indices.iter().map(identity).collect(),validation_frames:state.validation_indices.iter().map(identity).collect(),
        parameter_changes_from_initial:changes,initial_training,initial_validation,
        final_training, final_validation,
    };
    Ok(FullOutcome {
        checkpoint: state,
        report,
    })
}
