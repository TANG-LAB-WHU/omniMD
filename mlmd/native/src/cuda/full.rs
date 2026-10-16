//! Bounded exact all-neural-parameter CUDA gradients and elementwise Adam.
//! CPU geometry, detached loss seeds, deterministic reduction and assessment.
use super::{
    reverse::{ReversePlan, ReverseResources},
    PackedModel, Runtime,
};
use crate::{
    full_gradient::ParameterLayout,
    full_loss::{validate_loss, validate_record, LossSeeds, VirialLoss},
    full_training::{Checkpoint, CudaGradient, FullConfig, FullExecution},
    neighbor::Geometry,
    training::TrainConfig,
    training_data::DatasetFrame,
    Model, NativePotential,
};
use omnimd_ml_contract::{Error, Potential, Result};

pub(super) fn packed(model: &Model) -> Result<(PackedModel, ParameterLayout, String)> {
    let layout = ParameterLayout::new(model)?;
    let packed = PackedModel::new(model)?;
    if packed.weights.len() > 2048 || packed.slots > 64 {
        return Err(Error(
            "CUDA full training supports at most 2048 neural parameters and 64 neighbor slots"
                .into(),
        ));
    }
    let width = model
        .descriptor
        .embedding
        .iter()
        .flatten()
        .chain(&model.fitting)
        .flat_map(|net| &net.layers)
        .map(|l| l.weights.len().max(l.bias.len()))
        .max()
        .unwrap();
    if width > 32 || packed.width * model.descriptor.axis_neuron > 32 {
        return Err(Error(
            "CUDA full training supports every neural/descriptor width <=32".into(),
        ));
    }
    if packed.weights != layout.values(model) {
        return Err(Error(
            "CUDA packed parameter order differs from the full-training layout".into(),
        ));
    }
    let source = format!(
        "#define NT {}\n#define NSLOTS {}\n#define EW {}\n#define AXIS {}\n#define TW {}\n{}",
        model.type_map.len(),
        packed.slots,
        packed.width,
        model.descriptor.axis_neuron,
        width,
        include_str!("full_kernels.cu")
    );
    Ok((packed, layout, source))
}
/// Validates model-only resource bounds without touching CUDA or NVRTC.
pub fn validate_full_model(model: &Model) -> Result<()> {
    packed(model).map(|_| ())
}

const WORK_LIMIT: u64 = 50_000_000;
#[derive(Debug, serde::Serialize)]
pub struct FullWorkEstimate {
    pub atoms: usize,
    pub parameters: usize,
    pub maximum_center_terms: u64,
    pub estimated_scalar_terms: u64,
    pub policy_limit: u64,
}
fn center_terms(model: &Model) -> u64 {
    let weights = |net: &crate::Network| {
        net.layers
            .iter()
            .flat_map(|l| &l.weights)
            .map(|row| row.len() as u64)
            .sum::<u64>()
    };
    let d = &model.descriptor;
    let width = d.embedding[0][0].output_size() as u64;
    let contraction =
        d.sel.iter().sum::<usize>() as u64 * width * 4 + width * d.axis_neuron as u64 * 4;
    model
        .fitting
        .iter()
        .enumerate()
        .map(|(center, fit)| {
            contraction
                + weights(fit)
                + d.embedding[center]
                    .iter()
                    .zip(&d.sel)
                    .map(|(net, n)| *n as u64 * weights(net))
                    .sum::<u64>()
        })
        .max()
        .unwrap()
}
fn check_work(atoms: usize, parameters: usize, terms: u64) -> Result<FullWorkEstimate> {
    let estimate = (atoms as u64)
        .checked_mul(parameters as u64)
        .and_then(|n| n.checked_mul(terms));
    if atoms == 0 || atoms > 64 || estimate.is_none_or(|n| n > WORK_LIMIT) {
        return Err(Error(format!("CUDA full-gradient work budget rejected: atoms={atoms} (1..64), parameters={parameters}, max_center_terms={terms}, estimated_terms={estimate:?}, limit={WORK_LIMIT}; no truncation or fallback")));
    }
    Ok(FullWorkEstimate {
        atoms,
        parameters,
        maximum_center_terms: terms,
        estimated_scalar_terms: estimate.unwrap(),
        policy_limit: WORK_LIMIT,
    })
}
/// Conservative acceptance cap, not a predicted runtime or memory bound.
pub fn full_work_estimate(model: &Model, atoms: usize) -> Result<FullWorkEstimate> {
    let (_, layout, _) = packed(model)?;
    check_work(atoms, layout.parameters.len(), center_terms(model))
}

pub fn reverse_resources(model: &Model, atoms: usize) -> Result<ReverseResources> {
    full_work_estimate(model, atoms)?;
    let (p, _, s) = packed(model)?;
    ReversePlan::new(model, &p, &s)?.resources(atoms)
}
#[derive(Clone, Copy, Default, Debug, serde::Serialize)]
pub struct FullGradientProfile {
    pub frames: usize,
    pub prediction_and_seeds_ms: f64,
    pub geometry_ms: f64,
    pub upload_ms: f64,
    pub kernels_and_sync_ms: f64,
    pub download_ms: f64,
    pub reduction_ms: f64,
}
impl FullGradientProfile {
    pub fn add(&mut self, p: Self) {
        self.frames += p.frames;
        self.prediction_and_seeds_ms += p.prediction_and_seeds_ms;
        self.geometry_ms += p.geometry_ms;
        self.upload_ms += p.upload_ms;
        self.kernels_and_sync_ms += p.kernels_and_sync_ms;
        self.download_ms += p.download_ms;
        self.reduction_ms += p.reduction_ms;
    }
}
pub struct CudaFullTrainer {
    cpu: NativePotential,
    layout: ParameterLayout,
    runtime: Runtime,
    reverse: Option<ReversePlan>,
    profile: FullGradientProfile,
    step: u64,
    maximum_center_terms: u64,
    first: Vec<f64>,
    second: Vec<f64>,
    execution: FullExecution,
    poisoned: bool,
}
impl CudaFullTrainer {
    pub fn new(model: Model, device: i32) -> Result<Self> {
        Self::new_with_gradient(model, device, CudaGradient::Hyperdual)
    }
    pub fn new_with_gradient(model: Model, device: i32, gradient: CudaGradient) -> Result<Self> {
        Self::build(model, None, device, gradient)
    }
    pub(crate) fn restore(state: &Checkpoint, device: i32) -> Result<Self> {
        Self::build(
            state.model.clone(),
            Some(state),
            device,
            state.config.cuda_gradient,
        )
    }
    fn build(
        model: Model,
        state: Option<&Checkpoint>,
        device: i32,
        gradient: CudaGradient,
    ) -> Result<Self> {
        if device < 0 {
            return Err(Error(
                "CUDA visible device index must be nonnegative".into(),
            ));
        }
        let (packed, layout, source) = packed(&model)?;
        let p = packed.weights.len();
        let (step, first, second) = state.map_or((0, vec![0.0; p], vec![0.0; p]), |s| {
            (s.step, s.first_moment.clone(), s.second_moment.clone())
        });
        if first.len() != p
            || second.len() != p
            || first.iter().any(|x| !x.is_finite())
            || second.iter().any(|x| !x.is_finite() || *x < 0.0)
        {
            return Err(Error("invalid CUDA Adam state".into()));
        }
        let reverse = if gradient == CudaGradient::Reverse {
            Some(ReversePlan::new(&model, &packed, &source)?)
        } else {
            None
        };
        let mut runtime = match &reverse {
            Some(p) => Runtime::with_functions(
                &p.source,
                device,
                &[
                    "reverse_embed",
                    "full_adam",
                    "reverse_fit",
                    "reverse_embedding_backward",
                    "reverse_reduce",
                ],
                16,
            )?,
            None => Runtime::with_functions(&source, device, &["full_gradient", "full_adam"], 12)?,
        };
        let execution = if gradient == CudaGradient::Reverse {
            FullExecution::CudaReverseF64V1 {
                visible_device: device,
                device_info: runtime.device_info.clone(),
                driver_version: runtime.driver_version,
                program_sha256: runtime.program_sha256.clone(),
            }
        } else {
            FullExecution::CudaHyperdualF64V1 {
                visible_device: device,
                device_info: runtime.device_info.clone(),
                driver_version: runtime.driver_version,
                program_sha256: runtime.program_sha256.clone(),
            }
        };
        if let Some(s) = state {
            if s.step > 0 && s.execution != execution {
                return Err(Error("CUDA checkpoint device, driver or compiled-program identity changed; exact resume rejected".into()));
            }
        }
        runtime.upload(0, &packed.metadata)?;
        runtime.upload(1, &packed.weights)?;
        runtime.upload(4, &packed.slot_types)?;
        runtime.upload(9, &first)?;
        runtime.upload(10, &second)?;
        if let Some(p) = &reverse {
            runtime.upload(15, &p.parameter_network)?;
        }
        let maximum_center_terms = center_terms(&model);
        Ok(Self {
            reverse,
            profile: FullGradientProfile::default(),
            maximum_center_terms,
            cpu: NativePotential::new(model)?,
            layout,
            runtime,
            step,
            first,
            second,
            execution,
            poisoned: false,
        })
    }
    pub fn last_profile(&self) -> FullGradientProfile {
        self.profile
    }
    pub fn reverse_resources(&self, atoms: usize) -> Result<Option<ReverseResources>> {
        self.reverse
            .as_ref()
            .map(|p| p.resources(atoms))
            .transpose()
    }
    pub fn model(&self) -> &Model {
        self.cpu.model()
    }
    pub fn execution(&self) -> &FullExecution {
        &self.execution
    }
    pub fn moments(&self) -> (&[f64], &[f64]) {
        (&self.first, &self.second)
    }
    pub fn kernel_resources(&self) -> &[super::KernelResources] {
        &self.runtime.kernel_resources
    }
    pub fn frame_gradient(
        &mut self,
        record: &DatasetFrame,
        loss: &TrainConfig,
        normalizing_frames: usize,
    ) -> Result<(f64, Vec<f64>)> {
        self.frame_gradient_with_virial(record, loss, None, normalizing_frames)
    }
    pub fn frame_gradient_with_virial(
        &mut self,
        record: &DatasetFrame,
        loss: &TrainConfig,
        virial: Option<&VirialLoss>,
        normalizing_frames: usize,
    ) -> Result<(f64, Vec<f64>)> {
        if self.poisoned {
            return Err(Error(
                "CUDA full trainer stopped after earlier failure; no automatic retry".into(),
            ));
        }
        let result = self.gradient(record, loss, virial, normalizing_frames);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
    fn gradient(
        &mut self,
        record: &DatasetFrame,
        loss: &TrainConfig,
        virial: Option<&VirialLoss>,
        normalizing_frames: usize,
    ) -> Result<(f64, Vec<f64>)> {
        validate_loss(loss, virial)?;
        record.frame.validate()?;
        validate_record(record, virial)?;
        let n = record.frame.positions.len();
        if n > 64 || normalizing_frames == 0 {
            return Err(Error(
                "CUDA full gradient requires 1..64 atoms and positive frame normalization".into(),
            ));
        }
        check_work(n, self.layout.parameters.len(), self.maximum_center_terms)?;
        let resources = self.reverse.as_ref().map(|p| p.resources(n)).transpose()?;
        self.profile = FullGradientProfile {
            frames: 1,
            ..Default::default()
        };
        let phase = std::time::Instant::now();
        let predicted = self.cpu.eval(&record.frame)?;
        let seeds = LossSeeds::new(&predicted, record, loss, virial, normalizing_frames)?;
        self.profile.prediction_and_seeds_ms = phase.elapsed().as_secs_f64() * 1000.;
        let phase = std::time::Instant::now();
        let model = self.cpu.model();
        let d = &model.descriptor;
        let types = record.frame.type_ids(self.cpu.metadata())?;
        let geometry = Geometry::new(&record.frame, d.rcut)?;
        let mut rows = Vec::new();
        let mut tangent = Vec::new();
        for center in 0..n {
            let kind = types[center] as usize;
            let groups = geometry.neighbors(&record.frame, &types, center, &d.sel)?;
            for (species, group) in groups.iter().enumerate() {
                for slot in 0..d.sel[species] {
                    let neighbor = group.get(slot);
                    let (r, jac) =
                        crate::environment(neighbor, &d.normalization[kind], d.rcut_smth, d.rcut);
                    rows.extend(r);
                    for a in 0..4 {
                        tangent.push(neighbor.map_or(0.0, |q| seeds.tangent(jac[a], q, center)));
                    }
                }
            }
        }
        self.profile.geometry_ms = phase.elapsed().as_secs_f64() * 1000.;
        let p = self.layout.parameters.len();
        let (atomic_gradient, energy) = match &resources {
            Some(r) => self
                .runtime
                .reverse_frame(&rows, &tangent, &types, p, seeds.energy, r)?,
            None => self
                .runtime
                .full_frame(&rows, &tangent, &types, p, seeds.energy)?,
        };
        self.profile.upload_ms = self.runtime.last_timing_ms[0];
        self.profile.kernels_and_sync_ms = self.runtime.last_timing_ms[1];
        self.profile.download_ms = self.runtime.last_timing_ms[2];
        let phase = std::time::Instant::now();
        let primal = energy
            .iter()
            .enumerate()
            .map(|(i, e)| e + model.atom_energy_bias[types[i] as usize])
            .sum::<f64>();
        if !primal.is_finite()
            || (primal - predicted.energy).abs() > 1e-11 * (1.0 + predicted.energy.abs())
        {
            return Err(Error(
                "CUDA full-gradient primal disagrees with CPU inference".into(),
            ));
        }
        let mut gradient = vec![0.0; p];
        for atom in atomic_gradient.chunks_exact(p) {
            for (g, a) in gradient.iter_mut().zip(atom) {
                *g += a;
            }
        }
        if gradient.iter().any(|x| !x.is_finite()) {
            return Err(Error("nonfinite reduced CUDA gradient".into()));
        }
        self.profile.reduction_ms = phase.elapsed().as_secs_f64() * 1000.;
        Ok((seeds.objective, gradient))
    }
    /// Receives the already batch-reduced, clipped gradient. Commits host state
    /// only after every GPU candidate value/moment has passed validation.
    pub fn adam_step(&mut self, gradient: &[f64], config: &FullConfig) -> Result<()> {
        if self.poisoned {
            return Err(Error(
                "CUDA full trainer stopped after earlier failure; no automatic retry".into(),
            ));
        }
        let result = self.update(gradient, config);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
    fn update(&mut self, gradient: &[f64], config: &FullConfig) -> Result<()> {
        config.validate()?;
        if gradient.len() != self.layout.parameters.len()
            || gradient.iter().any(|x| !x.is_finite())
            || self.step >= 1_000_000_000
        {
            return Err(Error("invalid CUDA Adam gradient or global step".into()));
        }
        let step = self.step + 1;
        let (values, first, second) = self.runtime.full_adam(
            gradient,
            config.beta1,
            config.beta2,
            1.0 - config.beta1.powf(step as f64),
            1.0 - config.beta2.powf(step as f64),
            config.learning_rate,
            config.epsilon,
        )?;
        if first.iter().any(|x| !x.is_finite()) || second.iter().any(|x| !x.is_finite() || *x < 0.0)
        {
            return Err(Error("invalid downloaded CUDA Adam moments".into()));
        }
        let mut model = self.cpu.model().clone();
        self.layout.apply(&mut model, &values)?;
        self.cpu = NativePotential::new(model)?;
        self.first = first;
        self.second = second;
        self.step = step;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn combined_work_budget_is_checked_without_a_driver() {
        assert_eq!(
            check_work(1, 1, WORK_LIMIT).unwrap().estimated_scalar_terms,
            WORK_LIMIT
        );
        assert!(check_work(1, 1, WORK_LIMIT + 1).is_err());
        assert!(check_work(64, 2048, u64::MAX).is_err());
        let mut model: Model =
            serde_json::from_str(include_str!("../../fixtures/full-tanh-v1.json")).unwrap();
        let estimate = full_work_estimate(&model, 64).unwrap();
        assert_eq!(estimate.maximum_center_terms, 250);
        assert_eq!(estimate.estimated_scalar_terms, 1_792_000);
        // Species with a deeper fitting network must determine maximum-center cost.
        let extra = model.fitting[1].layers[1].clone();
        model.fitting[1].layers.insert(1, extra);
        assert_eq!(center_terms(&model), 259);
    }
    #[test]
    fn packed_full_layout_bounds_and_source_without_driver() {
        let mut model: Model =
            serde_json::from_str(include_str!("../../fixtures/full-tanh-v1.json")).unwrap();
        let (p, l, s) = packed(&model).unwrap();
        assert_eq!(p.weights, l.values(&model));
        assert_eq!(p.weights.len(), 112);
        assert!(s.contains("#define TW 3"));
        model.descriptor.sel = vec![33, 32];
        assert!(validate_full_model(&model)
            .unwrap_err()
            .0
            .contains("64 neighbor slots"));
    }
}
