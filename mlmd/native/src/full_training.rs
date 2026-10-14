//! Bounded deterministic full-batch Adam for the supported tanh networks.
use crate::{
    full_gradient::{frame_gradient, NetworkLocation, ParameterLayout},
    training::{metrics, partition, FrameIdentity, Metrics, Split, TrainConfig},
    training_data::{validate_labels, Dataset, DatasetHeader},
    Model,
};
use omnimd_ml_contract::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

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
    fn validate(&self) -> Result<()> {
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
    fn validate(&self, dataset: &Dataset, config: &FullConfig) -> Result<ParameterLayout> {
        config.validate()?;
        self.config.validate()?;
        validate_dataset(&self.model, dataset)?;
        let layout = ParameterLayout::new(&self.model)?;
        let initial_layout = ParameterLayout::new(&self.initial_model)?;
        if self.schema != "omnimd-native-full-checkpoint-v1"
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
    mut state: Checkpoint,
    dataset: &Dataset,
    config: &FullConfig,
    mut save: impl FnMut(&Checkpoint) -> Result<()>,
) -> Result<FullOutcome> {
    let layout = state.validate(dataset, config)?;
    let start = state.step;
    let end = start
        .checked_add(config.steps)
        .filter(|s| *s <= 1_000_000_000)
        .ok_or_else(|| Error("full trainer global step exceeds limit".into()))?;
    let loss = config.loss();
    let initial_training = metrics(&state.model, dataset, &state.training_indices, &loss)?;
    let initial_validation = metrics(&state.model, dataset, &state.validation_indices, &loss)?;
    state.config = config.clone();
    let mut values = layout.values(&state.model);
    while state.step < end {
        let mut gradient = vec![0.0; values.len()];
        for &i in &state.training_indices {
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
        let step = state.step + 1;
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
    let report=FullReport {
        scope:"all embedding/fitting weights, biases and timesteps; fixed normalization, geometry settings and atom_energy_bias".into(),
        algorithm:"deterministic full-batch Adam; constant learning rate; last iterate export".into(),
        derivative_method:"reverse AD over differentiable coordinate-directional tangents; exact mixed parameter-coordinate derivatives".into(),
        config:config.clone(),step_start:start,step_end:end,parameters:values.len(),parameter_layout_sha256:layout.sha256(),
        dataset_header:dataset.header.clone(),dataset_sha256:dataset.dataset_sha256.clone(),source_sha256:dataset.source_sha256.clone(),
        training_frames:state.training_indices.iter().map(identity).collect(),validation_frames:state.validation_indices.iter().map(identity).collect(),
        parameter_changes_from_initial:changes,initial_training,initial_validation,
        final_training:metrics(&state.model,dataset,&state.training_indices,&loss)?,
        final_validation:metrics(&state.model,dataset,&state.validation_indices,&loss)?,
    };
    Ok(FullOutcome {
        checkpoint: state,
        report,
    })
}
