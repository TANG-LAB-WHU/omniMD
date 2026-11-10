//! Thermal initialization, stateful reuse of omniMD's CSVR thermostat, and
//! constant-pressure dynamics with the stochastic cell rescaling barostat.
use crate::barostat::{
    CellAxes, CellCoupling, CellRescaling, BAR_PER_EV_PER_ANGSTROM3, MAX_STRAIN,
};
use crate::MlipDynamics;
use omnimd_core::{DegreesOfFreedom, System};
use omnimd_ml_contract::{Error, Potential, Result};
use omnimd_sim::{
    md::{Barostat, CSVRThermostat, Control, RemoveTranslation},
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
    /// CSVR thermostat plus stochastic cell rescaling ([`crate::barostat`]).
    NptCsvrCrescale {
        temperature_kelvin: f64,
        tau_fs: f64,
        seed: u64,
        /// Target pressure.
        pressure_bar: f64,
        /// Relaxation time of the volume for a system whose compressibility
        /// equals `compressibility_per_bar`.
        barostat_tau_fs: f64,
        /// Assumed isothermal compressibility; it only sets the time scale.
        compressibility_per_bar: f64,
        #[serde(default)]
        coupling: CellCoupling,
        /// The Cartesian axes the cell may change along (default all three).
        #[serde(default, skip_serializing_if = "CellAxes::is_all")]
        axes: CellAxes,
        /// Steps between cell moves; each move costs one model evaluation.
        #[serde(default = "default_barostat_every")]
        barostat_every: u64,
    },
}

fn default_barostat_every() -> u64 {
    10
}

impl Ensemble {
    /// True when the cell changes during the run.
    pub fn is_variable_cell(&self) -> bool {
        matches!(self, Self::NptCsvrCrescale { .. })
    }

    pub fn validate(&self, dt: f64) -> Result<()> {
        if !dt.is_finite() || dt <= 0. {
            return Err(Error("timestep_fs must be positive and finite".into()));
        }
        if let Self::NptCsvrCrescale {
            pressure_bar,
            barostat_tau_fs,
            compressibility_per_bar,
            coupling,
            axes,
            barostat_every,
            ..
        } = self
        {
            coupling.groups(*axes)?;
            let interval = *barostat_every as f64 * dt;
            if !pressure_bar.is_finite()
                || !(compressibility_per_bar.is_finite() && *compressibility_per_bar > 0.)
                || *barostat_every == 0
                || !(barostat_tau_fs.is_finite() && *barostat_tau_fs >= interval)
            {
                return Err(Error("NPT requires finite pressure_bar, positive finite compressibility_per_bar, positive barostat_every and barostat_tau_fs >= barostat_every * timestep_fs".into()));
            }
        }
        if let Self::NvtCsvr {
            temperature_kelvin,
            tau_fs,
            ..
        }
        | Self::NptCsvrCrescale {
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
                return Err(Error("NVT and NPT require positive finite temperature_kelvin and tau_fs >= timestep_fs (finite ratio)".into()));
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
    /// Accumulated kinetic plus potential energy delivered by the cell moves
    /// of the barostat (eV).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub barostat_work_ev: f64,
    /// Cell moves tried, and how many of them the acceptance test refused.
    #[serde(default, skip_serializing_if = "is_none")]
    pub barostat_moves: u64,
    #[serde(default, skip_serializing_if = "is_none")]
    pub barostat_refused: u64,
}

fn is_zero(value: &f64) -> bool {
    *value == 0.
}

fn is_none(count: &u64) -> bool {
    *count == 0
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
            Ensemble::NvtCsvr { seed, .. } | Ensemble::NptCsvrCrescale { seed, .. } => {
                Some(ChaCha8Rng::seed_from_u64(seed))
            }
        };
        Ok(Self {
            ensemble,
            rng,
            heat_ev: 0.,
            barostat_work_ev: 0.,
            barostat_moves: 0,
            barostat_refused: 0,
        })
    }

    pub fn validate(&self, dt: f64) -> Result<()> {
        self.ensemble.validate(dt)?;
        if !self.heat_ev.is_finite()
            || !self.barostat_work_ev.is_finite()
            || matches!(self.ensemble, Ensemble::Nve) != self.rng.is_none()
            || (matches!(self.ensemble, Ensemble::Nve) && self.heat_ev != 0.)
            || (!self.ensemble.is_variable_cell()
                && (self.barostat_work_ev != 0. || self.barostat_moves != 0))
            || self.barostat_refused > self.barostat_moves
        {
            return Err(Error(
                "invalid checkpoint thermostat/barostat/RNG/heat state".into(),
            ));
        }
        Ok(())
    }

    pub fn validate_system(&self, system: &System) -> Result<()> {
        if !matches!(self.ensemble, Ensemble::Nve) && system.kinetic_energy() <= 0. {
            return Err(Error(
                "NVT and NPT require nonzero initial kinetic energy; use initialize_velocities"
                    .into(),
            ));
        }
        if self.ensemble.is_variable_cell() && system.cell.is_infinite() {
            return Err(Error("NPT requires a periodic cell".into()));
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
            Ensemble::NptCsvrCrescale {
                temperature_kelvin,
                tau_fs,
                pressure_bar,
                barostat_tau_fs,
                compressibility_per_bar,
                coupling,
                axes,
                barostat_every,
                ..
            } => {
                // As above: the thermostat and the barostat draw from one
                // stream, which advances only if the whole step succeeds.
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
                let interval = barostat_every as f64 * md.timestep_fs();
                let mut rescaling = CellRescaling::new(
                    shared.clone(),
                    md.thermal_energy_ev(temperature_kelvin),
                    pressure_bar / BAR_PER_EV_PER_ANGSTROM3,
                    compressibility_per_bar * BAR_PER_EV_PER_ANGSTROM3 * interval
                        / (3. * barostat_tau_fs),
                    coupling,
                    axes,
                    MAX_STRAIN,
                )?;
                // Moves fall on absolute step numbers, so a restart keeps the schedule.
                let due = md.system().step.wrapping_add(1) % barostat_every == 0;
                let barostat: Option<&mut dyn Barostat> =
                    if due { Some(&mut rescaling) } else { None };
                let exchange = md.step_with_controls(Some(&mut thermostat), barostat)?;
                self.rng = Some(shared.0.borrow().clone());
                self.heat_ev += exchange.heat;
                self.barostat_work_ev += exchange.work;
                if let Some(accepted) = exchange.cell_move {
                    self.barostat_moves += 1;
                    self.barostat_refused += u64::from(!accepted);
                }
                if !(self.heat_ev.is_finite() && self.barostat_work_ev.is_finite()) {
                    return Err(Error(
                        "accumulated thermostat heat or barostat work overflow".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}
