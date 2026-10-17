//! Private device-loss execution with coherent snapshots at existing save boundaries.
//! Never exposes the stale structural CPU template as a current trained model.
use super::*;
use crate::training_data::DatasetAccess;

#[derive(Clone, Copy, Default, Debug, serde::Serialize)]
pub struct ResidentStateProfile {
    pub updates: usize,
    pub status_downloads: usize,
    pub snapshots: usize,
    pub state_download_calls: usize,
    pub state_download_bytes: usize,
    pub snapshot_download_ms: f64,
    pub snapshot_model_validation_ms: f64,
}
pub(crate) struct ResidentFullTrainer {
    inner: CudaFullTrainer,
    profile: ResidentStateProfile,
}
impl CudaFullTrainer {
    pub(crate) fn into_resident(self) -> Result<ResidentFullTrainer> {
        if !self.device_loss || self.tiled.is_none() || self.poisoned {
            return Err(Error(
                "resident optimizer state requires a valid GPU-loss tiled trainer".into(),
            ));
        }
        Ok(ResidentFullTrainer {
            inner: self,
            profile: ResidentStateProfile::default(),
        })
    }
}
impl ResidentFullTrainer {
    pub(crate) fn batch_gradient_with_virial(
        &mut self,
        dataset: &dyn DatasetAccess,
        indices: &[usize],
        loss: &TrainConfig,
        virial: Option<&VirialLoss>,
    ) -> Result<(f64, Vec<f64>)> {
        // This path uses host types/geometry/normalization/offsets only, never host neural values.
        self.inner
            .batch_gradient_with_virial(dataset, indices, loss, virial)
    }
    pub(crate) fn last_profile(&self) -> FullGradientProfile {
        self.inner.last_profile()
    }
    pub(crate) fn fixed_geometry_stats(&self) -> Option<FixedGeometryStats> {
        self.inner.fixed_geometry_stats()
    }
    pub(crate) fn tiling_resources(&self) -> Option<&TiledResources> {
        self.inner.tiling_resources()
    }
    pub(crate) fn state_profile(&self) -> ResidentStateProfile {
        self.profile
    }
    pub(crate) fn adam_step(&mut self, gradient: &[f64], config: &FullConfig) -> Result<()> {
        if self.inner.poisoned {
            return Err(Error(
                "resident CUDA trainer stopped after earlier failure".into(),
            ));
        }
        let result = self.update(gradient, config);
        if result.is_err() {
            self.inner.poisoned = true;
        }
        result
    }
    fn update(&mut self, gradient: &[f64], config: &FullConfig) -> Result<()> {
        config.validate()?;
        if gradient.len() != self.inner.layout.parameters.len()
            || gradient.iter().any(|v| !v.is_finite())
            || self.inner.step >= 1_000_000_000
        {
            return Err(Error(
                "invalid resident CUDA Adam gradient or global step".into(),
            ));
        }
        let step = self.inner.step + 1;
        // Existing kernel checks every mutable model value and both moments for
        // finiteness, plus second >= 0. Trainable timestep constraints are finite-only.
        // Architecture, normalization, offsets and all shapes cannot change here.
        self.inner.runtime.full_adam(
            gradient,
            config.beta1,
            config.beta2,
            1.0 - config.beta1.powf(step as f64),
            1.0 - config.beta2.powf(step as f64),
            config.learning_rate_at(self.inner.step)?,
            config.epsilon,
            false,
        )?;
        self.inner.step = step; // only after successful synchronized device status download
        self.profile.updates += 1;
        self.profile.status_downloads += 1;
        Ok(())
    }
    pub(crate) fn snapshot(&mut self, expected_step: u64) -> Result<(Model, Vec<f64>, Vec<f64>)> {
        if self.inner.poisoned {
            return Err(Error(
                "resident CUDA trainer stopped after earlier failure".into(),
            ));
        }
        let result = self.read_snapshot(expected_step);
        if result.is_err() {
            self.inner.poisoned = true;
        }
        result
    }
    fn read_snapshot(&mut self, expected_step: u64) -> Result<(Model, Vec<f64>, Vec<f64>)> {
        if expected_step != self.inner.step {
            return Err(Error("resident CUDA snapshot/global step mismatch".into()));
        }
        let p = self.inner.layout.parameters.len();
        let phase = std::time::Instant::now();
        let (values, first, second) = self.inner.runtime.adam_snapshot(p)?;
        let download_ms = phase.elapsed().as_secs_f64() * 1000.;
        let phase = std::time::Instant::now();
        if first.iter().any(|v| !v.is_finite()) || second.iter().any(|v| !v.is_finite() || *v < 0.)
        {
            return Err(Error("invalid resident CUDA snapshot moments".into()));
        }
        let mut model = self.inner.cpu.model().clone();
        self.inner.layout.apply(&mut model, &values)?; // full model validation before publication
        self.profile.snapshot_download_ms += download_ms;
        self.profile.snapshot_model_validation_ms += phase.elapsed().as_secs_f64() * 1000.;
        self.profile.snapshots += 1;
        self.profile.state_download_calls += 3;
        self.profile.state_download_bytes += p * 3 * 8;
        Ok((model, first, second))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires an authorized exclusive GPU window"]
    fn resident_nonfinite_device_update_poison_preserves_last_snapshot() {
        let model: Model =
            serde_json::from_str(include_str!("../../fixtures/full-tanh-v1.json")).unwrap();
        let config: FullConfig = serde_json::from_value(serde_json::json!({
            "split":{"mode":"explicit","validation_ids":["unused"]},"steps":2,
            "cuda_gradient":"reverse","cuda_loss":"gpu","cuda_tiling":{"max_frames":4,"max_atoms":64},
            "energy_weight":1.,"force_weight":1.,"energy_scale_ev_per_atom":1.,"force_scale_ev_per_angstrom":1.,
            "learning_rate":0.001,"beta1":0.9,"beta2":0.999,"epsilon":1e-8,"checkpoint_every":1
        })).unwrap();
        let mut gpu =
            CudaFullTrainer::new_device_loss(model, 0, config.cuda_tiling.as_ref().unwrap())
                .unwrap()
                .into_resident()
                .unwrap();
        let n = gpu.inner.layout.parameters.len();
        gpu.adam_step(&vec![0.1; n], &config).unwrap();
        let completed = gpu.snapshot(1).unwrap();
        let before = serde_json::to_vec(&completed.0).unwrap();
        assert!(gpu
            .adam_step(&vec![1e308; n], &config)
            .unwrap_err()
            .0
            .contains("nonfinite CUDA Adam"));
        assert!(gpu.snapshot(1).is_err());
        assert!(gpu.adam_step(&vec![0.1; n], &config).is_err());
        assert_eq!(gpu.inner.step, 1);
        assert_eq!(gpu.state_profile().snapshots, 1);
        assert_eq!(before, serde_json::to_vec(&completed.0).unwrap());
    }
}
