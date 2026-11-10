// Copyright (C) omniMD contributors — MIT license.
//! Fallible velocity-Verlet molecular dynamics using one combined model evaluation.
use super::Thermostat;
use omnimd_core::{
    consts::K_BOLTZMANN, units, DegreesOfFreedom, Matrix3, System, UnitCell, Vector3D,
};
use omnimd_ml_contract::{Error, Evaluation, Frame, ModelMetadata, Potential, Result, Units};

/// Dynamics driven exclusively by a model implementing Potential: NVE, with
/// optional upstream velocity thermostats and an optional [`Barostat`].
/// Owns the real omniMD System and caches E/F/virial for its current positions.
/// Force-field mixing and constraints are not enabled.
pub struct MlipDynamics<P: Potential> {
    system: System,
    potential: P,
    evaluation: Evaluation,
    timestep: f64,
    ev_to_internal: f64,
}

/// A homogeneous cell move tried at the end of a step, after the thermostat.
pub trait Barostat {
    /// The deformation gradient `D` to try for a state with this instantaneous
    /// pressure tensor (eV/angstrom^3) and cell volume (angstrom^3).
    /// Lattice vectors and positions are mapped by `D` (`x -> D x`) and
    /// velocities by its inverse transpose, which keeps the momenta conjugate
    /// to the fractional coordinates unchanged.
    fn deformation(&mut self, pressure: &[[f64; 3]; 3], volume: f64) -> Result<[[f64; 3]; 3]>;

    /// Whether the move just tried stands, given the pressure tensor and
    /// volume after it and the change of kinetic plus potential energy (eV).
    /// A refused move leaves the state as it was before the move.
    fn accept(
        &mut self,
        pressure: &[[f64; 3]; 3],
        volume: f64,
        energy_change: f64,
    ) -> Result<bool>;
}

/// Energy the controls of one step delivered to the system (eV).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Exchange {
    /// Kinetic energy added by the thermostat.
    pub heat: f64,
    /// Kinetic plus potential energy added by the cell move of the barostat
    /// (zero when the move was refused).
    pub work: f64,
    /// Whether a cell move was tried, and whether it stood.
    pub cell_move: Option<bool>,
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
    match system.simulated_degrees_of_freedom {
        DegreesOfFreedom::Particles | DegreesOfFreedom::Frozen(0) => {}
        DegreesOfFreedom::Frozen(3) if system.size() > 1 => {}
        _ => {
            return Err(Error(
                "MLIP supports only particle DOF, optionally minus COM translation".into(),
            ))
        }
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

/// `sum_i m_i v_i v_i^T` in eV.
fn kinetic_tensor(system: &System, ev_to_internal: f64) -> [[f64; 3]; 3] {
    let mut tensor = [[0.0; 3]; 3];
    for p in system.particles().iter() {
        for a in 0..3 {
            for b in 0..3 {
                tensor[a][b] += p.mass * p.velocity[a] * p.velocity[b] / ev_to_internal;
            }
        }
    }
    tensor
}

/// Kinetic energy in eV, summed in the order used throughout this module.
fn kinetic_energy(system: &System, ev_to_internal: f64) -> f64 {
    system
        .particles()
        .iter()
        .map(|p| 0.5 * p.mass * p.velocity.norm2() / ev_to_internal)
        .sum()
}

/// `(sum_i m_i v_i v_i^T + virial) / volume` in eV/angstrom^3, with the virial
/// convention of the model contract (`sum r f^T`, positive when the forces
/// push the cell outward).
fn pressure_tensor(
    system: &System,
    evaluation: &Evaluation,
    ev_to_internal: f64,
) -> Result<([[f64; 3]; 3], f64)> {
    if system.cell.is_infinite() {
        return Err(Error("pressure requires a periodic cell".into()));
    }
    let volume = system.cell.matrix().determinant();
    let kinetic = kinetic_tensor(system, ev_to_internal);
    let pressure = std::array::from_fn(|a| {
        std::array::from_fn(|b| (kinetic[a][b] + evaluation.virial[a][b]) / volume)
    });
    if !(volume.is_finite() && volume > 0.0) || pressure.iter().flatten().any(|p| !p.is_finite())
    {
        return Err(Error("pressure tensor is nonfinite".into()));
    }
    Ok((pressure, volume))
}

/// Map lattice vectors and positions by `d` and velocities by its inverse transpose.
fn deform(system: &mut System, d: &[[f64; 3]; 3]) -> Result<()> {
    let d = Matrix3::new(*d);
    let determinant = d.determinant();
    if d.iter().flatten().any(|x| !x.is_finite()) || !(determinant.is_finite() && determinant > 0.0)
    {
        return Err(Error(
            "cell deformation must be finite with a positive determinant".into(),
        ));
    }
    if system.cell.is_infinite() {
        return Err(Error("cell deformation requires a periodic cell".into()));
    }
    let momentum_map = d.inverse().transposed();
    system.cell = UnitCell::from_matrix(d * system.cell.matrix())
        .ok_or_else(|| Error("deformed cell is not representable".into()))?;
    let particles = system.particles_mut();
    for i in 0..particles.len() {
        particles.position[i] = d * particles.position[i];
        particles.velocity[i] = momentum_map * particles.velocity[i];
    }
    Ok(())
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
        if !(kinetic_energy(&system, ev_to_internal) + evaluation.energy).is_finite() {
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
        let _ = self.step_impl(None, None)?;
        Ok(&self.evaluation)
    }

    /// Apply an upstream velocity-only thermostat after one velocity-Verlet step.
    /// Returns heat delivered to the system in eV. Published state is transactional;
    /// the caller must handle rollback of any thermostat random state on error.
    pub fn step_with_thermostat(&mut self, thermostat: &mut dyn Thermostat) -> Result<f64> {
        Ok(self.step_impl(Some(thermostat), None)?.heat)
    }

    /// One velocity-Verlet step, then the thermostat, then the cell move of
    /// the barostat, tried with a new model evaluation at the deformed
    /// geometry (one extra evaluation whenever a barostat is given) and kept
    /// if the barostat accepts it. Returns the energy both controls delivered. Published state is transactional as for
    /// [`step_with_thermostat`](Self::step_with_thermostat).
    pub fn step_with_controls(
        &mut self,
        thermostat: Option<&mut dyn Thermostat>,
        barostat: Option<&mut dyn Barostat>,
    ) -> Result<Exchange> {
        self.step_impl(thermostat, barostat)
    }

    fn step_impl(
        &mut self,
        thermostat: Option<&mut dyn Thermostat>,
        barostat: Option<&mut dyn Barostat>,
    ) -> Result<Exchange> {
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
        let mut evaluated = self.potential.eval(&frame(&next))?;
        evaluated.validate(next.size())?;
        {
            let particles = next.particles_mut();
            for i in 0..particles.len() {
                let acceleration =
                    Vector3D::from(evaluated.forces[i]) * (self.ev_to_internal / particles.mass[i]);
                particles.velocity[i] += 0.5 * self.timestep * acceleration;
            }
        }
        let mut exchange = Exchange::default();
        if let Some(thermostat) = thermostat {
            let before = next.kinetic_energy();
            if !before.is_finite() || before <= 0.0 {
                return Err(Error(
                    "CSVR requires positive finite kinetic energy; initialize velocities".into(),
                ));
            }
            let evaluated_frame = frame(&next);
            let masses = next.particles().mass.to_vec();
            let dof = next.simulated_degrees_of_freedom.clone();
            thermostat.apply(&mut next);
            let after_frame = frame(&next);
            if after_frame.positions != evaluated_frame.positions
                || after_frame.cell != evaluated_frame.cell
                || after_frame.species != evaluated_frame.species
                || next.particles().mass != masses.as_slice()
                || next.simulated_degrees_of_freedom != dof
            {
                return Err(Error("thermostat changed more than velocities".into()));
            }
            exchange.heat = (next.kinetic_energy() - before) / self.ev_to_internal;
            if !exchange.heat.is_finite() {
                return Err(Error("thermostat produced nonfinite heat".into()));
            }
        }
        if let Some(barostat) = barostat {
            validate_dynamics(&next)?;
            let (pressure, volume) = pressure_tensor(&next, &evaluated, self.ev_to_internal)?;
            let before = kinetic_energy(&next, self.ev_to_internal) + evaluated.energy;
            let mut moved = next.clone();
            deform(&mut moved, &barostat.deformation(&pressure, volume)?)?;
            validate_dynamics(&moved)?;
            let moved_evaluation = self.potential.eval(&frame(&moved))?;
            moved_evaluation.validate(moved.size())?;
            let work =
                kinetic_energy(&moved, self.ev_to_internal) + moved_evaluation.energy - before;
            if !work.is_finite() {
                return Err(Error("barostat produced nonfinite work".into()));
            }
            let (pressure, volume) =
                pressure_tensor(&moved, &moved_evaluation, self.ev_to_internal)?;
            let accepted = barostat.accept(&pressure, volume, work)?;
            exchange.cell_move = Some(accepted);
            if accepted {
                next = moved;
                evaluated = moved_evaluation;
                exchange.work = work;
            }
        }
        validate_dynamics(&next)?;
        if !(kinetic_energy(&next, self.ev_to_internal) + evaluated.energy).is_finite() {
            return Err(Error("updated total energy is nonfinite".into()));
        }
        self.system = next;
        self.evaluation = evaluated;
        Ok(exchange)
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
    /// Kinetic energy converted from the internal units of omniMD to eV.
    pub fn kinetic_energy_ev(&self) -> f64 {
        kinetic_energy(&self.system, self.ev_to_internal)
    }
    /// Total energy in eV from the cached potential evaluation and current velocities.
    pub fn total_energy_ev(&self) -> f64 {
        self.evaluation.energy + self.kinetic_energy_ev()
    }
    /// Instantaneous kinetic temperature with the configured particle/COM DOF.
    pub fn temperature_kelvin(&self) -> f64 {
        2.0 * self.system.kinetic_energy() / (self.system.degrees_of_freedom() as f64 * K_BOLTZMANN)
    }
    /// Instantaneous pressure tensor in eV/angstrom^3 and the cell volume in
    /// angstrom^3: `(sum_i m_i v_i v_i^T + virial) / volume`. An error for an
    /// infinite cell.
    pub fn pressure_tensor(&self) -> Result<([[f64; 3]; 3], f64)> {
        pressure_tensor(&self.system, &self.evaluation, self.ev_to_internal)
    }
    /// The Boltzmann constant times a temperature in kelvin, in eV, with the
    /// constants this integrator uses for the kinetic temperature.
    pub fn thermal_energy_ev(&self, temperature_kelvin: f64) -> f64 {
        K_BOLTZMANN * temperature_kelvin / self.ev_to_internal
    }
    /// Time increment per step, in femtoseconds.
    pub fn timestep_fs(&self) -> f64 {
        self.timestep
    }
}
