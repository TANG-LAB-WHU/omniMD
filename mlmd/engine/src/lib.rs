//! Input conversion for the fallible MLIP dynamics implemented in omnimd-sim.
pub mod dataset;
pub mod deviation;
mod thermal;
pub mod xyz;
use omnimd_core::{Matrix3, Molecule, Particle, System, UnitCell, Vector3D};
use omnimd_ml_contract::{Error, Frame, Result};
pub use omnimd_sim::md::MlipDynamics;
pub use thermal::{
    initialize_velocities, prepare_velocities, set_degrees_of_freedom, Ensemble, EnsembleState,
    VelocityInitialization,
};

/// Convert shared-model coordinates and explicit masses/velocities to omniMD.
/// Masses are in u and velocities in angstrom/fs. Cell rows are transposed once.
pub fn system_from_frame(frame: &Frame, masses: &[f64], velocities: &[[f64; 3]]) -> Result<System> {
    frame.validate()?;
    if masses.len() != frame.positions.len() || velocities.len() != masses.len() {
        return Err(Error("mass/velocity/position counts must match".into()));
    }
    let cell = match frame.cell {
        None => UnitCell::infinite(),
        Some(rows) => UnitCell::from_matrix(Matrix3::new(std::array::from_fn(|i| {
            std::array::from_fn(|j| rows[j][i])
        })))
        .ok_or_else(|| Error("cell is not representable by omniMD".into()))?,
    };
    let mut system = System::with_cell(cell);
    for i in 0..masses.len() {
        if !masses[i].is_finite()
            || masses[i] <= 0.0
            || velocities[i].iter().any(|v| !v.is_finite())
        {
            return Err(Error(format!("atom {i}: invalid mass or velocity")));
        }
        let mut particle =
            Particle::with_position(&frame.species[i], Vector3D::from(frame.positions[i]));
        particle.mass = masses[i];
        particle.velocity = Vector3D::from(velocities[i]);
        system.add_molecule(Molecule::new(particle));
    }
    Ok(system)
}

use omnimd_ml_contract::Potential;
use serde::{Deserialize, Serialize};

/// Portable restart state for the fixed-cell NVE/CSVR paths.
/// Units are angstrom, fs, u and eV. The model file digest prevents silent swaps.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    pub schema_version: u32,
    pub frame: Frame,
    pub masses: Vec<f64>,
    pub velocities: Vec<[f64; 3]>,
    pub step: u64,
    pub timestep_fs: f64,
    pub model_sha256: String,
    #[serde(default)]
    pub remove_center_of_mass: bool,
    #[serde(default)]
    pub ensemble_state: Option<EnsembleState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fixed_parameters: Option<parameters::FixedParameters>,
}

impl Checkpoint {
    /// Capture a successfully committed integration state.
    pub fn capture<P: Potential>(md: &MlipDynamics<P>, model_sha256: String) -> Self {
        let particles = md.system().particles();
        Self {
            schema_version: 2,
            frame: md.frame(),
            masses: particles.mass.to_vec(),
            velocities: particles
                .velocity
                .iter()
                .map(|v| [v[0], v[1], v[2]])
                .collect(),
            step: md.system().step,
            timestep_fs: md.timestep_fs(),
            model_sha256,
            remove_center_of_mass: matches!(
                md.system().simulated_degrees_of_freedom,
                omnimd_core::DegreesOfFreedom::Frozen(3)
            ),
            ensemble_state: None,
            fixed_parameters: None,
        }
    }

    /// Validate the schema/model identity and restore the real omniMD System.
    /// Initial E/F/virial are recomputed by MlipDynamics::new, not trusted from disk.
    pub fn restore(&self, model_sha256: &str, timestep_fs: f64) -> Result<System> {
        self.restore_with_parameters(model_sha256, timestep_fs, None)
    }
    pub fn restore_with_parameters(
        &self,
        model_sha256: &str,
        timestep_fs: f64,
        expected: Option<&parameters::FixedParameters>,
    ) -> Result<System> {
        if !matches!(self.schema_version, 1 | 2 | 3) {
            return Err(Error("unsupported checkpoint version".into()));
        }
        if (self.schema_version == 3) != self.fixed_parameters.is_some() {
            return Err(Error(
                "only checkpoint v3 may contain fixed parameters, and v3 requires them".into(),
            ));
        }
        if self.fixed_parameters.as_ref() != expected {
            return Err(Error("checkpoint fixed parameters differ; parameter-free restore cannot drop conditioning".into()));
        }
        if let Some(fixed) = &self.fixed_parameters {
            fixed.validate(&self.frame)?;
        }
        if self.model_sha256.len() != 64
            || !self.model_sha256.bytes().all(|b| b.is_ascii_hexdigit())
            || !self.model_sha256.eq_ignore_ascii_case(model_sha256)
        {
            return Err(Error(
                "checkpoint model SHA-256 does not match model file".into(),
            ));
        }
        if !timestep_fs.is_finite() || timestep_fs <= 0. || self.timestep_fs != timestep_fs {
            return Err(Error(
                "checkpoint timestep must match the positive requested timestep".into(),
            ));
        }
        if self.schema_version == 1 && (self.ensemble_state.is_some() || self.remove_center_of_mass)
        {
            return Err(Error(
                "version 1 checkpoint cannot contain thermal state".into(),
            ));
        }
        if let Some(state) = &self.ensemble_state {
            state.validate(timestep_fs)?;
        }
        let mut system = system_from_frame(&self.frame, &self.masses, &self.velocities)?;
        set_degrees_of_freedom(&mut system, self.remove_center_of_mass)?;
        system.step = self.step;
        Ok(system)
    }
}

pub mod selection;

pub mod parameters;
