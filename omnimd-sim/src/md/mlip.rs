// Copyright (C) omniMD contributors — MIT license.
//! Fallible velocity-Verlet molecular dynamics using one combined model evaluation.
use omnimd_core::{units, System, Vector3D};
use omnimd_ml_contract::{Error, Evaluation, Frame, ModelMetadata, Potential, Result, Units};

/// Fixed-cell NVE dynamics driven exclusively by a model implementing Potential.
/// Owns the real omniMD System and caches E/F/virial for its current positions.
/// Force-field mixing, thermostats and constraints are not enabled here.
pub struct MlipDynamics<P: Potential> {
    system: System,
    potential: P,
    evaluation: Evaluation,
    timestep: f64,
    ev_to_internal: f64,
}

fn frame(system: &System) -> Frame {
    let particles = system.particles();
    let matrix = system.cell.matrix();
    Frame {
        positions: particles.position.iter().map(|r| [r[0], r[1], r[2]]).collect(),
        species: particles.name.to_vec(),
        cell: if system.cell.is_infinite() {
            None
        } else {
            Some(std::array::from_fn(|i| std::array::from_fn(|j| matrix[j][i])))
        },
        units: Units::EvAngstrom,
    }
}

fn validate_dynamics(system: &System) -> Result<()> {
    if system.has_interactions() {
        return Err(Error("MLIP dynamics requires no installed classical/global potentials; mixing is not implemented".into()));
    }
    for (i, p) in system.particles().iter().enumerate() {
        if !p.mass.is_finite()
            || *p.mass <= 0.0
            || (0..3).any(|a| !p.velocity[a].is_finite())
            || !p.velocity.norm2().is_finite()
        {
            return Err(Error(format!(
                "atom {i}: mass must be positive/finite and velocity finite"
            )));
        }
    }
    frame(system).validate()
}

impl<P: Potential> MlipDynamics<P> {
    /// Start NVE dynamics with timestep in femtoseconds and initial forces evaluated.
    pub fn new(system: System, mut potential: P, timestep: f64) -> Result<Self> {
        if !timestep.is_finite() || timestep <= 0.0 {
            return Err(Error("timestep must be positive and finite (fs)".into()));
        }
        validate_dynamics(&system)?;
        potential.metadata().validate()?;
        let input = frame(&system);
        let _ = input.type_ids(potential.metadata())?;
        let evaluation = potential.eval(&input)?;
        evaluation.validate(system.size())?;
        let ev_to_internal = units::from(1.0, "eV").map_err(|e| Error(e.to_string()))?;
        let kinetic: f64 = system
            .particles()
            .iter()
            .map(|p| 0.5 * p.mass * p.velocity.norm2() / ev_to_internal)
            .sum();
        if !(kinetic + evaluation.energy).is_finite() {
            return Err(Error("initial total energy is nonfinite".into()));
        }
        Ok(Self {
            system,
            potential,
            evaluation,
            timestep,
            ev_to_internal,
        })
    }

    /// Advance one full velocity-Verlet step.
    /// Numerical failure leaves the published System and cached evaluation unchanged.
    pub fn step(&mut self) -> Result<&Evaluation> {
        let mut next = self.system.clone();
        next.step =
            next.step.checked_add(1).ok_or_else(|| Error("step counter overflow".into()))?;
        {
            let particles = next.particles_mut();
            for i in 0..particles.len() {
                let acceleration = Vector3D::from(self.evaluation.forces[i])
                    * (self.ev_to_internal / particles.mass[i]);
                particles.velocity[i] += 0.5 * self.timestep * acceleration;
                particles.position[i] += self.timestep * particles.velocity[i];
            }
        }
        validate_dynamics(&next)?;
        let evaluated = self.potential.eval(&frame(&next))?;
        evaluated.validate(next.size())?;
        {
            let particles = next.particles_mut();
            for i in 0..particles.len() {
                let acceleration =
                    Vector3D::from(evaluated.forces[i]) * (self.ev_to_internal / particles.mass[i]);
                particles.velocity[i] += 0.5 * self.timestep * acceleration;
            }
        }
        validate_dynamics(&next)?;
        let kinetic: f64 = next
            .particles()
            .iter()
            .map(|p| 0.5 * p.mass * p.velocity.norm2() / self.ev_to_internal)
            .sum();
        if !(kinetic + evaluated.energy).is_finite() {
            return Err(Error("updated total energy is nonfinite".into()));
        }
        self.system = next;
        self.evaluation = evaluated;
        Ok(&self.evaluation)
    }

    /// Current system (read-only so cached model outputs cannot become stale).
    pub fn system(&self) -> &System {
        &self.system
    }
    /// Current Cartesian configuration in the shared model contract.
    pub fn frame(&self) -> Frame {
        frame(&self.system)
    }
    /// Current combined model output in eV/angstrom units; no reevaluation.
    pub fn evaluation(&self) -> &Evaluation {
        &self.evaluation
    }
    /// Metadata of the selected backend.
    pub fn metadata(&self) -> &ModelMetadata {
        self.potential.metadata()
    }
    /// Kinetic energy converted from omniMD's internal units to eV.
    pub fn kinetic_energy_ev(&self) -> f64 {
        self.system
            .particles()
            .iter()
            .map(|p| 0.5 * p.mass * p.velocity.norm2() / self.ev_to_internal)
            .sum()
    }
    /// Total energy in eV from the cached potential evaluation and current velocities.
    pub fn total_energy_ev(&self) -> f64 {
        self.evaluation.energy + self.kinetic_energy_ev()
    }
    /// Time increment per step, in femtoseconds.
    pub fn timestep_fs(&self) -> f64 {
        self.timestep
    }
}
