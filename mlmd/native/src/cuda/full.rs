//! Bounded exact all-neural-parameter CUDA gradients and elementwise Adam.
//! CPU geometry, detached loss seeds, deterministic reduction and assessment.
use super::{
    reverse::{ReversePlan, ReverseResources},
    PackedModel, Runtime,
};
use crate::{
    fixed_geometry::{FixedGeometryCache, FixedGeometryStats},
    full_gradient::ParameterLayout,
    full_loss::{validate_loss, validate_record, LossSeeds, VirialLoss},
    full_training::{Checkpoint, CudaGradient, CudaLoss, CudaTiling, FullConfig, FullExecution},
    neighbor::Geometry,
    training::TrainConfig,
    training_data::{Dataset, DatasetFrame},
    Model, NativePotential,
};
use omnimd_ml_contract::{Error, Potential, Result};
#[path = "tiled.rs"]
mod tiled;
pub use tiled::TiledResources;
#[path = "device_loss.rs"]
pub(crate) mod device_loss;
pub use device_loss::{DeviceLossDiagnostics, DeviceLossResources};
pub fn device_loss_resources(model: &Model, config: &CudaTiling) -> Result<TiledResources> {
    let (p, _, s) = packed(model)?;
    let plan = ReversePlan::new(model, &p, &s)?;
    device_loss::plan_resources(model, &plan, config)
}
pub fn tiled_resources(model: &Model, config: &CudaTiling) -> Result<TiledResources> {
    let (p, _, s) = packed(model)?;
    let plan = ReversePlan::new(model, &p, &s)?;
    tiled::plan_resources(model, &plan, config)
}

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
    check_work_bound(atoms, parameters, terms, 64)
}
fn check_work_bound(
    atoms: usize,
    parameters: usize,
    terms: u64,
    frame_bound: usize,
) -> Result<FullWorkEstimate> {
    let estimate = (atoms as u64)
        .checked_mul(parameters as u64)
        .and_then(|n| n.checked_mul(terms));
    if atoms == 0 || atoms > frame_bound || estimate.is_none_or(|n| n > WORK_LIMIT) {
        return Err(Error(format!("CUDA full-gradient work budget rejected: atoms={atoms} (1..{frame_bound}), parameters={parameters}, max_center_terms={terms}, estimated_terms={estimate:?}, limit={WORK_LIMIT}; no truncation or fallback")));
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
    pub max_atoms_per_frame: usize,
    pub device_prediction_frames: usize,
    pub tiles: usize,
    /// Tiled gradient inputs plus status resets; excludes Adam and initialization.
    pub input_transfer_calls: usize,
    pub input_transfer_bytes: usize,
    pub kernel_launches: usize,
    pub synchronizations: usize,
    pub max_frames_per_launch: usize,
    pub max_atoms_per_launch: usize,
    pub peak_device_buffer_bytes: usize,
    pub peak_host_tile_payload_bytes: usize,
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
        self.max_atoms_per_frame = self.max_atoms_per_frame.max(p.max_atoms_per_frame);
        self.device_prediction_frames += p.device_prediction_frames;
        self.tiles += p.tiles;
        self.input_transfer_calls += p.input_transfer_calls;
        self.input_transfer_bytes += p.input_transfer_bytes;
        self.kernel_launches += p.kernel_launches;
        self.synchronizations += p.synchronizations;
        self.max_frames_per_launch = self.max_frames_per_launch.max(p.max_frames_per_launch);
        self.max_atoms_per_launch = self.max_atoms_per_launch.max(p.max_atoms_per_launch);
        self.peak_device_buffer_bytes = self
            .peak_device_buffer_bytes
            .max(p.peak_device_buffer_bytes);
        self.peak_host_tile_payload_bytes = self
            .peak_host_tile_payload_bytes
            .max(p.peak_host_tile_payload_bytes);
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
    tiled: Option<TiledResources>,
    device_loss: bool,
    capture_device_loss: bool,
    device_loss_diagnostics: Option<DeviceLossDiagnostics>,
    profile: FullGradientProfile,
    step: u64,
    maximum_center_terms: u64,
    first: Vec<f64>,
    second: Vec<f64>,
    execution: FullExecution,
    poisoned: bool,
    fixed_geometry: Option<FixedGeometryCache>,
}
impl CudaFullTrainer {
    pub fn new(model: Model, device: i32) -> Result<Self> {
        Self::new_with_gradient(model, device, CudaGradient::Hyperdual)
    }
    pub fn new_with_gradient(model: Model, device: i32, gradient: CudaGradient) -> Result<Self> {
        Self::build(model, None, device, gradient, None, CudaLoss::Cpu)
    }
    pub fn new_tiled(model: Model, device: i32, config: &CudaTiling) -> Result<Self> {
        Self::build(
            model,
            None,
            device,
            CudaGradient::Reverse,
            Some(config),
            CudaLoss::Cpu,
        )
    }
    pub fn new_device_loss(model: Model, device: i32, config: &CudaTiling) -> Result<Self> {
        Self::build(
            model,
            None,
            device,
            CudaGradient::Reverse,
            Some(config),
            CudaLoss::Gpu,
        )
    }
    pub fn tiling_resources(&self) -> Option<&TiledResources> {
        self.tiled.as_ref()
    }
    pub(crate) fn restore(state: &Checkpoint, device: i32) -> Result<Self> {
        Self::build(
            state.model.clone(),
            Some(state),
            device,
            state.config.cuda_gradient,
            state.config.cuda_tiling.as_ref(),
            state.config.cuda_loss,
        )
    }
    fn build(
        model: Model,
        state: Option<&Checkpoint>,
        device: i32,
        gradient: CudaGradient,
        tiling: Option<&CudaTiling>,
        loss: CudaLoss,
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
        let tiled = tiling
            .map(|c| {
                let plan = reverse
                    .as_ref()
                    .ok_or_else(|| Error("tiling requires reverse CUDA".into()))?;
                if loss == CudaLoss::Gpu {
                    device_loss::plan_resources(&model, plan, c)
                } else {
                    tiled::plan_resources(&model, plan, c)
                }
            })
            .transpose()?;
        let large_frame = tiling.is_some_and(|t| t.max_atoms_per_frame > 64);
        let tiled_source = if tiled.is_some() {
            let source = reverse.as_ref().unwrap().tiled_source()?;
            Some(if loss == CudaLoss::Gpu {
                format!(
                    "{source}\n{}{}",
                    include_str!("device_loss_kernels.cu"),
                    if large_frame {
                        include_str!("large_frame_kernels.cu")
                    } else {
                        ""
                    }
                )
            } else {
                source
            })
        } else {
            None
        };
        let mut runtime = if loss == CudaLoss::Gpu {
            Runtime::with_functions(
                tiled_source
                    .as_ref()
                    .ok_or_else(|| Error("device loss requires tiling".into()))?,
                device,
                &[
                    "loss_primal_embed",
                    "full_adam",
                    "loss_primal_fit",
                    "loss_edge_gradient",
                    if large_frame {
                        "loss_predictions_large"
                    } else {
                        "loss_predictions"
                    },
                    "loss_seeds",
                    "loss_directions",
                    "loss_dual_embed",
                    "loss_dual_fit",
                    "reverse_embedding_backward",
                    "reverse_reduce",
                ],
                31,
            )?
        } else {
            match &reverse {
                Some(p) => Runtime::with_functions(
                    tiled_source.as_ref().unwrap_or(&p.source),
                    device,
                    &[
                        "reverse_embed",
                        "full_adam",
                        "reverse_fit",
                        "reverse_embedding_backward",
                        "reverse_reduce",
                    ],
                    if tiled.is_some() { 17 } else { 16 },
                )?,
                None => {
                    Runtime::with_functions(&source, device, &["full_gradient", "full_adam"], 12)?
                }
            }
        };
        let execution = if loss == CudaLoss::Gpu && large_frame {
            FullExecution::CudaTiledDeviceLossF64V2 {
                visible_device: device,
                device_info: runtime.device_info.clone(),
                driver_version: runtime.driver_version,
                program_sha256: runtime.program_sha256.clone(),
            }
        } else if loss == CudaLoss::Gpu {
            FullExecution::CudaTiledDeviceLossF64V1 {
                visible_device: device,
                device_info: runtime.device_info.clone(),
                driver_version: runtime.driver_version,
                program_sha256: runtime.program_sha256.clone(),
            }
        } else if tiled.is_some() {
            FullExecution::CudaTiledReverseF64V1 {
                visible_device: device,
                device_info: runtime.device_info.clone(),
                driver_version: runtime.driver_version,
                program_sha256: runtime.program_sha256.clone(),
            }
        } else if gradient == CudaGradient::Reverse {
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
        if tiling.is_some() {
            runtime.enable_packed_inputs();
        }
        runtime.upload(0, &packed.metadata)?;
        runtime.upload(1, &packed.weights)?;
        runtime.upload(4, &packed.slot_types)?;
        runtime.upload(9, &first)?;
        runtime.upload(10, &second)?;
        if let Some(p) = &reverse {
            runtime.upload(15, &p.parameter_network)?;
        }
        if loss == CudaLoss::Gpu {
            runtime.upload(30, &model.atom_energy_bias)?;
        }
        let maximum_center_terms = center_terms(&model);
        Ok(Self {
            reverse,
            tiled,
            device_loss: loss == CudaLoss::Gpu,
            capture_device_loss: false,
            device_loss_diagnostics: None,
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
            fixed_geometry: None,
        })
    }
    /// Eager all-or-error cache preparation for a fixed, explicitly selected frame set.
    pub fn prepare_fixed_geometry_cache(
        &mut self,
        dataset: &Dataset,
        indices: &[usize],
    ) -> Result<()> {
        if self.fixed_geometry.is_some() {
            return Err(Error(
                "fixed geometry cache already attached; recreate trainer".into(),
            ));
        }
        if self.tiled.is_some() {
            self.preflight_tiled_batch(dataset, indices)?;
        } else {
            for &i in indices {
                check_work(
                    crate::training_data::DatasetAccess::atoms(dataset, i)?,
                    self.layout.parameters.len(),
                    self.maximum_center_terms,
                )?;
            }
        }
        let cache = FixedGeometryCache::new(self.cpu.model(), dataset, indices)?;
        self.attach_fixed_geometry_cache(Some(cache))
    }
    pub(crate) fn attach_fixed_geometry_cache(
        &mut self,
        cache: Option<FixedGeometryCache>,
    ) -> Result<()> {
        if self.fixed_geometry.is_some() {
            return Err(Error("fixed geometry cache already attached".into()));
        }
        self.fixed_geometry = cache;
        Ok(())
    }
    pub fn fixed_geometry_stats(&self) -> Option<FixedGeometryStats> {
        self.fixed_geometry.as_ref().map(FixedGeometryCache::stats)
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
        if self.tiled.is_some() {
            return Err(Error(
                "tiled CUDA executor requires batch_gradient_with_virial".into(),
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
            max_atoms_per_frame: n,
            tiles: 1,
            kernel_launches: if self.reverse.is_some() { 4 } else { 1 },
            synchronizations: 1,
            max_frames_per_launch: 1,
            max_atoms_per_launch: n,
            ..Default::default()
        };
        let phase = std::time::Instant::now();
        let prepared = match &mut self.fixed_geometry {
            Some(cache) => Some(cache.get(&self.cpu, record)?),
            None => None,
        };
        let lookup_ms = phase.elapsed().as_secs_f64() * 1000.;
        let phase = std::time::Instant::now();
        let predicted = match prepared {
            Some(frame) => self.cpu.eval_prepared(frame)?,
            None => self.cpu.eval(&record.frame)?,
        };
        let seeds = LossSeeds::new(&predicted, record, loss, virial, normalizing_frames)?;
        self.profile.prediction_and_seeds_ms = phase.elapsed().as_secs_f64() * 1000.;
        let phase = std::time::Instant::now();
        let model = self.cpu.model();
        let d = &model.descriptor;
        let mut uncached_rows = Vec::new();
        let uncached_types;
        let mut tangent = Vec::with_capacity(n * d.sel.iter().sum::<usize>() * 4);
        let (rows, types) = if let Some(frame) = prepared {
            for (i, slot) in frame.slots.iter().enumerate() {
                let center = i / frame.slots_per_atom;
                for a in 0..4 {
                    tangent.push(
                        slot.neighbor
                            .as_ref()
                            .map_or(0.0, |q| seeds.tangent(slot.jac[a], q, center)),
                    );
                }
            }
            (frame.rows.as_ref(), frame.types.as_ref())
        } else {
            uncached_types = record.frame.type_ids(self.cpu.metadata())?;
            let geometry = Geometry::new(&record.frame, d.rcut)?;
            for center in 0..n {
                let kind = uncached_types[center] as usize;
                let groups = geometry.neighbors(&record.frame, &uncached_types, center, &d.sel)?;
                for (species, group) in groups.iter().enumerate() {
                    for slot in 0..d.sel[species] {
                        let neighbor = group.get(slot);
                        let (r, jac) = crate::environment(
                            neighbor,
                            &d.normalization[kind],
                            d.rcut_smth,
                            d.rcut,
                        );
                        uncached_rows.extend(r);
                        for a in 0..4 {
                            tangent
                                .push(neighbor.map_or(0.0, |q| seeds.tangent(jac[a], q, center)));
                        }
                    }
                }
            }
            (uncached_rows.as_slice(), uncached_types.as_slice())
        };
        self.profile.geometry_ms = lookup_ms + phase.elapsed().as_secs_f64() * 1000.;
        let p = self.layout.parameters.len();
        let (atomic_gradient, energy) = match &resources {
            Some(r) => self
                .runtime
                .reverse_frame(&rows, &tangent, &types, p, seeds.energy, r)?,
            None => self
                .runtime
                .full_frame(&rows, &tangent, &types, p, seeds.energy)?,
        };
        self.profile.peak_device_buffer_bytes = self.runtime.resident_buffer_bytes();
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
            config.learning_rate_at(self.step)?,
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
