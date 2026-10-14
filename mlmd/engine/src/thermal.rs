//! Thermal initialization and stateful reuse of omniMD's CSVR thermostat.
use crate::MlipDynamics;
use omnimd_core::{DegreesOfFreedom, System};
use omnimd_ml_contract::{Error, Potential, Result};
use omnimd_sim::{
    md::{CSVRThermostat, Control, RemoveTranslation},
    BoltzmannVelocities, InitVelocities,
};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::{cell::RefCell, rc::Rc};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VelocityInitialization {
    pub temperature_kelvin: f64,
    pub seed: u64,
    pub rescale_to_temperature: bool,
}

pub fn set_degrees_of_freedom(system: &mut System, remove_com: bool) -> Result<()> {
    if remove_com && system.size() < 2 {
        return Err(Error(
            "removing COM translation requires at least two atoms".into(),
        ));
    }
    system.simulated_degrees_of_freedom = if remove_com {
        DegreesOfFreedom::Frozen(3)
    } else {
        DegreesOfFreedom::Particles
    };
    Ok(())
}

/// For explicit input velocities, subtract the mass-weighted COM once if requested.
pub fn prepare_velocities(system: &mut System, remove_com: bool) -> Result<()> {
    set_degrees_of_freedom(system, remove_com)?;
    if remove_com {
        RemoveTranslation.control(system);
    }
    validate_kinetic(system)
}

/// Seed and reuse upstream Gaussian sampling without removing periodic-cell rotation.
pub fn initialize_velocities(
    system: &mut System,
    settings: &VelocityInitialization,
    remove_com: bool,
) -> Result<()> {
    if !settings.temperature_kelvin.is_finite() || settings.temperature_kelvin <= 0. {
        return Err(Error(
            "initial temperature_kelvin must be positive and finite".into(),
        ));
    }
    set_degrees_of_freedom(system, remove_com)?;
    let mut init = BoltzmannVelocities::new(settings.temperature_kelvin);
    init.seed(settings.seed);
    init.init_unconstrained(system, remove_com, settings.rescale_to_temperature);
    validate_kinetic(system)
}

fn validate_kinetic(system: &System) -> Result<()> {
    if !system.kinetic_energy().is_finite()
        || system
            .particles()
            .velocity
            .iter()
            .any(|v| !(0..3).all(|a| v[a].is_finite()))
    {
        return Err(Error(
            "velocity initialization produced nonfinite kinetic state".into(),
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Ensemble {
    #[default]
    Nve,
    NvtCsvr {
        temperature_kelvin: f64,
        tau_fs: f64,
        seed: u64,
    },
}

impl Ensemble {
    pub fn validate(&self, dt: f64) -> Result<()> {
        if !dt.is_finite() || dt <= 0. {
            return Err(Error("timestep_fs must be positive and finite".into()));
        }
        if let Self::NvtCsvr {
            temperature_kelvin,
            tau_fs,
            ..
        } = self
        {
            if !temperature_kelvin.is_finite()
                || *temperature_kelvin <= 0.
                || !tau_fs.is_finite()
                || *tau_fs < dt
                || !(tau_fs / dt).is_finite()
            {
                return Err(Error("NVT requires positive finite temperature_kelvin and tau_fs >= timestep_fs (finite ratio)".into()));
            }
        }
        Ok(())
    }
}

/// Algorithm and full RNG state are persisted; a seed alone is not a restart state.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnsembleState {
    pub ensemble: Ensemble,
    rng: Option<ChaCha8Rng>,
    /// Accumulated energy delivered by the thermostat (eV).
    pub heat_ev: f64,
}

#[derive(Clone)]
struct SharedRng(Rc<RefCell<ChaCha8Rng>>);
impl RngCore for SharedRng {
    fn next_u32(&mut self) -> u32 {
        self.0.borrow_mut().next_u32()
    }
    fn next_u64(&mut self) -> u64 {
        self.0.borrow_mut().next_u64()
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.borrow_mut().fill_bytes(dest);
    }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> std::result::Result<(), rand::Error> {
        self.0.borrow_mut().try_fill_bytes(dest)
    }
}

impl EnsembleState {
    pub fn new(ensemble: Ensemble, dt: f64) -> Result<Self> {
        ensemble.validate(dt)?;
        let rng = match ensemble {
            Ensemble::Nve => None,
            Ensemble::NvtCsvr { seed, .. } => Some(ChaCha8Rng::seed_from_u64(seed)),
        };
        Ok(Self {
            ensemble,
            rng,
            heat_ev: 0.,
        })
    }

    pub fn validate(&self, dt: f64) -> Result<()> {
        self.ensemble.validate(dt)?;
        if !self.heat_ev.is_finite()
            || matches!(self.ensemble, Ensemble::Nve) != self.rng.is_none()
            || (matches!(self.ensemble, Ensemble::Nve) && self.heat_ev != 0.)
        {
            return Err(Error("invalid checkpoint thermostat/RNG/heat state".into()));
        }
        Ok(())
    }

    pub fn validate_system(&self, system: &System) -> Result<()> {
        if matches!(self.ensemble, Ensemble::NvtCsvr { .. }) && system.kinetic_energy() <= 0. {
            return Err(Error(
                "NVT requires nonzero initial kinetic energy; use initialize_velocities".into(),
            ));
        }
        Ok(())
    }

    pub fn advance<P: Potential>(&mut self, md: &mut MlipDynamics<P>) -> Result<()> {
        self.validate(md.timestep_fs())?;
        self.validate_system(md.system())?;
        match self.ensemble {
            Ensemble::Nve => {
                md.step()?;
            }
            Ensemble::NvtCsvr {
                temperature_kelvin,
                tau_fs,
                ..
            } => {
                // This clone advances only if the entire dynamics/thermostat step succeeds.
                let shared = SharedRng(Rc::new(RefCell::new(
                    self.rng
                        .as_ref()
                        .ok_or_else(|| Error("missing CSVR RNG state".into()))?
                        .clone(),
                )));
                let mut thermostat = CSVRThermostat::from_rng(
                    temperature_kelvin,
                    tau_fs / md.timestep_fs(),
                    Box::new(shared.clone()),
                );
                let heat = md.step_with_thermostat(&mut thermostat)?;
                self.rng = Some(shared.0.borrow().clone());
                self.heat_ev += heat;
                if !self.heat_ev.is_finite() {
                    return Err(Error("accumulated thermostat heat overflow".into()));
                }
            }
        }
        Ok(())
    }
}
