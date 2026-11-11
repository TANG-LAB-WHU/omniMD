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
    replica: Replica,
    potential: P,
    timestep: f64,
    ev_to_internal: f64,
}

/// Several independent systems under one model, advanced together: each step
/// asks the model for all of them at once ([`Potential::eval_batch`]), and
/// once more for those whose barostat moves the cell. A backend that
/// evaluates a batch in one device pass then advances many small systems for
/// little more than the cost of one. Every system follows the trajectory it
/// follows alone in an [`MlipDynamics`], to the rounding of the model.
pub struct MlipReplicas<P: Potential> {
    replicas: Vec<Replica>,
    potential: P,
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

/// The controls of one system for one step of [`MlipReplicas`].
#[derive(Default)]
pub struct Controls<'a> {
    pub thermostat: Option<&'a mut dyn Thermostat>,
    pub barostat: Option<&'a mut dyn Barostat>,
}

impl Controls<'_> {
    // The controls for the length of a call: written out so that the borrow
    // of the trait objects can be shortened, which `Option::as_deref_mut`
    // does not do.
    fn thermostat(&mut self) -> Option<&mut dyn Thermostat> {
        match &mut self.thermostat {
            Some(thermostat) => Some(&mut **thermostat),
            None => None,
        }
    }

    fn barostat(&mut self) -> Option<&mut dyn Barostat> {
        match &mut self.barostat {
            Some(barostat) => Some(&mut **barostat),
            None => None,
        }
    }
}

/// The lattice vectors of a periodic system as rows, as the model contract
/// writes a cell.
fn cell_rows(system: &System) -> Option<[[f64; 3]; 3]> {
    if system.cell.is_infinite() {
        return None;
    }
    let matrix = system.cell.matrix();
    Some(std::array::from_fn(|i| std::array::from_fn(|j| matrix[j][i])))
}

fn frame(system: &System) -> Frame {
    let particles = system.particles();
    Frame {
        positions: particles.position.iter().map(|r| [r[0], r[1], r[2]]).collect(),
        species: particles.name.to_vec(),
        cell: cell_rows(system),
        units: Units::EvAngstrom,
    }
}

/// What dynamics asks of a system besides a valid geometry: no classical
/// potentials, supported degrees of freedom, usable masses and velocities.
fn validate_particles(system: &System) -> Result<()> {
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
    Ok(())
}

fn validate_dynamics(system: &System) -> Result<()> {
    validate_particles(system)?;
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

/// What a step changes in a system, kept to take the step back.
struct Saved {
    positions: Vec<Vector3D>,
    velocities: Vec<Vector3D>,
    cell: UnitCell,
    step: u64,
}

impl Saved {
    fn of(system: &System) -> Self {
        let particles = system.particles();
        Self {
            positions: particles.position.to_vec(),
            velocities: particles.velocity.to_vec(),
            cell: system.cell,
            step: system.step,
        }
    }

    fn capture(&mut self, system: &System) {
        let particles = system.particles();
        self.positions.clear();
        self.positions.extend_from_slice(particles.position);
        self.velocities.clear();
        self.velocities.extend_from_slice(particles.velocity);
        self.cell = system.cell;
        self.step = system.step;
    }

    fn restore(&self, system: &mut System) {
        system.cell = self.cell;
        system.step = self.step;
        let particles = system.particles_mut();
        particles.position.copy_from_slice(&self.positions);
        particles.velocity.copy_from_slice(&self.velocities);
    }
}

/// One system under MLIP dynamics, without the model. A step works on the
/// system in place and is taken back if any part of it fails: copying the
/// whole system and its frames, names and all, cost a fifth of a step of a
/// hundred atoms on a fast model.
struct Replica {
    system: System,
    evaluation: Evaluation,
    /// What the model is asked: the species of the system, with the
    /// positions and the cell of the moment written before each evaluation.
    input: Frame,
    /// The state at the start of the step in the making, and before its
    /// cell move.
    start: Saved,
    unmoved: Saved,
    /// Whether a step is in the making: `start` is then the state to go
    /// back to.
    stepping: bool,
    /// Scratch: the masses before a thermostat.
    masses: Vec<f64>,
}

impl Replica {
    /// A valid system, not yet evaluated: `input` holds the frame to ask the
    /// model about.
    fn new(system: System) -> Result<Self> {
        validate_dynamics(&system)?;
        Ok(Self {
            input: frame(&system),
            start: Saved::of(&system),
            unmoved: Saved::of(&system),
            stepping: false,
            masses: Vec::with_capacity(system.size()),
            evaluation: Evaluation {
                energy: 0.0,
                forces: Vec::new(),
                virial: [[0.0; 3]; 3],
                units: Units::EvAngstrom,
            },
            system,
        })
    }

    /// Takes the model's word on the initial positions.
    fn start(&mut self, evaluation: Evaluation, ev_to_internal: f64) -> Result<()> {
        evaluation.validate(self.system.size())?;
        if !(kinetic_energy(&self.system, ev_to_internal) + evaluation.energy).is_finite() {
            return Err(Error("initial total energy is nonfinite".into()));
        }
        self.evaluation = evaluation;
        Ok(())
    }

    /// Checks the system as dynamics needs it and writes its positions and
    /// cell to `input`, validated as the model contract asks.
    fn write_input(&mut self) -> Result<()> {
        validate_particles(&self.system)?;
        let particles = self.system.particles();
        self.input.positions.clear();
        self.input
            .positions
            .extend(particles.position.iter().map(|r| [r[0], r[1], r[2]]));
        self.input.cell = cell_rows(&self.system);
        self.input.validate()
    }

    /// The first half of a step: the kick by the cached forces and the
    /// drift. Leaves the frame to evaluate in `input`.
    fn begin(&mut self, timestep: f64, ev_to_internal: f64) -> Result<()> {
        self.start.capture(&self.system);
        self.stepping = true;
        self.system.step = self
            .system
            .step
            .checked_add(1)
            .ok_or_else(|| Error("step counter overflow".into()))?;
        {
            let particles = self.system.particles_mut();
            for i in 0..particles.len() {
                let acceleration = Vector3D::from(self.evaluation.forces[i])
                    * (ev_to_internal / particles.mass[i]);
                particles.velocity[i] += 0.5 * timestep * acceleration;
                particles.position[i] += timestep * particles.velocity[i];
            }
        }
        self.write_input()
    }

    /// The second half: the kick by the forces of `evaluated`, then the
    /// thermostat. Returns the heat it delivered (eV).
    fn complete(
        &mut self,
        evaluated: &Evaluation,
        timestep: f64,
        ev_to_internal: f64,
        thermostat: Option<&mut dyn Thermostat>,
    ) -> Result<f64> {
        evaluated.validate(self.system.size())?;
        {
            let particles = self.system.particles_mut();
            for i in 0..particles.len() {
                let acceleration =
                    Vector3D::from(evaluated.forces[i]) * (ev_to_internal / particles.mass[i]);
                particles.velocity[i] += 0.5 * timestep * acceleration;
            }
        }
        let Some(thermostat) = thermostat else {
            return Ok(0.0);
        };
        let before = self.system.kinetic_energy();
        if !before.is_finite() || before <= 0.0 {
            return Err(Error(
                "CSVR requires positive finite kinetic energy; initialize velocities".into(),
            ));
        }
        // `input` holds the positions and the cell the model was asked
        // about: a thermostat must leave them, and all but the velocities,
        // as they are.
        self.masses.clear();
        self.masses.extend_from_slice(self.system.particles().mass);
        let dof = self.system.simulated_degrees_of_freedom.clone();
        thermostat.apply(&mut self.system);
        let particles = self.system.particles();
        let unchanged = particles.position.len() == self.input.positions.len()
            && particles
                .position
                .iter()
                .zip(&self.input.positions)
                .all(|(r, asked)| [r[0], r[1], r[2]] == *asked)
            && cell_rows(&self.system) == self.input.cell
            && particles.name == self.input.species.as_slice()
            && particles.mass == self.masses.as_slice()
            && self.system.simulated_degrees_of_freedom == dof;
        if !unchanged {
            return Err(Error("thermostat changed more than velocities".into()));
        }
        let heat = (self.system.kinetic_energy() - before) / ev_to_internal;
        if !heat.is_finite() {
            return Err(Error("thermostat produced nonfinite heat".into()));
        }
        Ok(heat)
    }

    /// The cell move of a barostat, made in place: leaves the frame to
    /// evaluate in `input` and returns the kinetic plus potential energy
    /// before the move.
    fn propose(
        &mut self,
        evaluated: &Evaluation,
        ev_to_internal: f64,
        barostat: &mut dyn Barostat,
    ) -> Result<f64> {
        self.write_input()?;
        let (pressure, volume) = pressure_tensor(&self.system, evaluated, ev_to_internal)?;
        let before = kinetic_energy(&self.system, ev_to_internal) + evaluated.energy;
        self.unmoved.capture(&self.system);
        deform(&mut self.system, &barostat.deformation(&pressure, volume)?)?;
        self.write_input()?;
        Ok(before)
    }

    /// Asks the barostat whether the move stands, with the model's word on
    /// the moved system. A move that stands replaces `evaluated`; a refused
    /// one is taken back. Returns whether it stood and the work it did (eV).
    fn settle(
        &mut self,
        evaluated: &mut Evaluation,
        moved: Evaluation,
        before: f64,
        ev_to_internal: f64,
        barostat: &mut dyn Barostat,
    ) -> Result<(bool, f64)> {
        moved.validate(self.system.size())?;
        let work = kinetic_energy(&self.system, ev_to_internal) + moved.energy - before;
        if !work.is_finite() {
            return Err(Error("barostat produced nonfinite work".into()));
        }
        let (pressure, volume) = pressure_tensor(&self.system, &moved, ev_to_internal)?;
        if barostat.accept(&pressure, volume, work)? {
            *evaluated = moved;
            Ok((true, work))
        } else {
            self.unmoved.restore(&mut self.system);
            Ok((false, 0.0))
        }
    }

    /// The last checks of a step, on the state it would leave.
    fn check(&mut self, evaluated: &Evaluation, ev_to_internal: f64) -> Result<()> {
        self.write_input()?;
        if !(kinetic_energy(&self.system, ev_to_internal) + evaluated.energy).is_finite() {
            return Err(Error("updated total energy is nonfinite".into()));
        }
        Ok(())
    }

    /// The step stands.
    fn commit(&mut self, evaluated: Evaluation) {
        self.evaluation = evaluated;
        self.stepping = false;
    }

    /// Takes a step in the making back, if there is one.
    fn undo(&mut self) {
        if self.stepping {
            self.start.restore(&mut self.system);
            self.stepping = false;
        }
    }

    fn view<'a>(
        &'a self,
        metadata: &'a ModelMetadata,
        timestep: f64,
        ev_to_internal: f64,
    ) -> StateView<'a> {
        StateView {
            replica: self,
            metadata,
            timestep,
            ev_to_internal,
        }
    }
}

/// One system under MLIP dynamics, to read: its state, the model's cached
/// evaluation of its positions, and what follows from the two.
#[derive(Clone, Copy)]
pub struct StateView<'a> {
    replica: &'a Replica,
    metadata: &'a ModelMetadata,
    timestep: f64,
    ev_to_internal: f64,
}

impl<'a> StateView<'a> {
    /// Current system (read-only so cached model outputs cannot become stale).
    pub fn system(&self) -> &'a System {
        &self.replica.system
    }
    /// Current Cartesian configuration in the shared model contract.
    pub fn frame(&self) -> Frame {
        frame(&self.replica.system)
    }
    /// Current combined model output in eV/angstrom units; no reevaluation.
    pub fn evaluation(&self) -> &'a Evaluation {
        &self.replica.evaluation
    }
    /// Metadata of the selected backend.
    pub fn metadata(&self) -> &'a ModelMetadata {
        self.metadata
    }
    /// Kinetic energy converted from the internal units of omniMD to eV.
    pub fn kinetic_energy_ev(&self) -> f64 {
        kinetic_energy(&self.replica.system, self.ev_to_internal)
    }
    /// Total energy in eV from the cached potential evaluation and current velocities.
    pub fn total_energy_ev(&self) -> f64 {
        self.replica.evaluation.energy + self.kinetic_energy_ev()
    }
    /// Instantaneous kinetic temperature with the configured particle/COM DOF.
    pub fn temperature_kelvin(&self) -> f64 {
        let system = &self.replica.system;
        2.0 * system.kinetic_energy() / (system.degrees_of_freedom() as f64 * K_BOLTZMANN)
    }
    /// Instantaneous pressure tensor in eV/angstrom^3 and the cell volume in
    /// angstrom^3: `(sum_i m_i v_i v_i^T + virial) / volume`. An error for an
    /// infinite cell.
    pub fn pressure_tensor(&self) -> Result<([[f64; 3]; 3], f64)> {
        pressure_tensor(
            &self.replica.system,
            &self.replica.evaluation,
            self.ev_to_internal,
        )
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

fn checked_timestep(timestep: f64) -> Result<f64> {
    if !timestep.is_finite() || timestep <= 0.0 {
        return Err(Error("timestep must be positive and finite (fs)".into()));
    }
    units::from(1.0, "eV").map_err(|e| Error(e.to_string()))
}

impl<P: Potential> MlipDynamics<P> {
    /// Start NVE dynamics with timestep in femtoseconds and initial forces evaluated.
    pub fn new(system: System, mut potential: P, timestep: f64) -> Result<Self> {
        let ev_to_internal = checked_timestep(timestep)?;
        let mut replica = Replica::new(system)?;
        potential.metadata().validate()?;
        let _ = replica.input.type_ids(potential.metadata())?;
        let evaluation = potential.eval(&replica.input)?;
        replica.start(evaluation, ev_to_internal)?;
        Ok(Self {
            replica,
            potential,
            timestep,
            ev_to_internal,
        })
    }

    /// Advance one full velocity-Verlet step.
    /// Numerical failure leaves the published System and cached evaluation unchanged.
    pub fn step(&mut self) -> Result<&Evaluation> {
        let _ = self.step_impl(None, None)?;
        Ok(&self.replica.evaluation)
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
        let result = self.advance(thermostat, barostat);
        if result.is_err() {
            self.replica.undo();
        }
        result
    }

    fn advance(
        &mut self,
        thermostat: Option<&mut dyn Thermostat>,
        barostat: Option<&mut dyn Barostat>,
    ) -> Result<Exchange> {
        let (timestep, ev_to_internal) = (self.timestep, self.ev_to_internal);
        let replica = &mut self.replica;
        replica.begin(timestep, ev_to_internal)?;
        let mut evaluated = self.potential.eval(&replica.input)?;
        let mut exchange = Exchange {
            heat: replica.complete(&evaluated, timestep, ev_to_internal, thermostat)?,
            ..Exchange::default()
        };
        if let Some(barostat) = barostat {
            let before = replica.propose(&evaluated, ev_to_internal, barostat)?;
            let moved = self.potential.eval(&replica.input)?;
            let (accepted, work) =
                replica.settle(&mut evaluated, moved, before, ev_to_internal, barostat)?;
            exchange.cell_move = Some(accepted);
            exchange.work = work;
        }
        replica.check(&evaluated, ev_to_internal)?;
        replica.commit(evaluated);
        Ok(exchange)
    }

    /// The state to read, as the replicas of [`MlipReplicas`] give theirs.
    pub fn view(&self) -> StateView<'_> {
        self.replica.view(
            self.potential.metadata(),
            self.timestep,
            self.ev_to_internal,
        )
    }

    /// Current system (read-only so cached model outputs cannot become stale).
    pub fn system(&self) -> &System {
        &self.replica.system
    }
    /// Current Cartesian configuration in the shared model contract.
    pub fn frame(&self) -> Frame {
        self.view().frame()
    }
    /// Current combined model output in eV/angstrom units; no reevaluation.
    pub fn evaluation(&self) -> &Evaluation {
        &self.replica.evaluation
    }
    /// Metadata of the selected backend.
    pub fn metadata(&self) -> &ModelMetadata {
        self.potential.metadata()
    }
    /// Kinetic energy converted from the internal units of omniMD to eV.
    pub fn kinetic_energy_ev(&self) -> f64 {
        self.view().kinetic_energy_ev()
    }
    /// Total energy in eV from the cached potential evaluation and current velocities.
    pub fn total_energy_ev(&self) -> f64 {
        self.view().total_energy_ev()
    }
    /// Instantaneous kinetic temperature with the configured particle/COM DOF.
    pub fn temperature_kelvin(&self) -> f64 {
        self.view().temperature_kelvin()
    }
    /// Instantaneous pressure tensor in eV/angstrom^3 and the cell volume in
    /// angstrom^3: `(sum_i m_i v_i v_i^T + virial) / volume`. An error for an
    /// infinite cell.
    pub fn pressure_tensor(&self) -> Result<([[f64; 3]; 3], f64)> {
        self.view().pressure_tensor()
    }
    /// The Boltzmann constant times a temperature in kelvin, in eV, with the
    /// constants this integrator uses for the kinetic temperature.
    pub fn thermal_energy_ev(&self, temperature_kelvin: f64) -> f64 {
        self.view().thermal_energy_ev(temperature_kelvin)
    }
    /// Time increment per step, in femtoseconds.
    pub fn timestep_fs(&self) -> f64 {
        self.timestep
    }
}

/// `count` evaluations for as many frames, or an error.
fn evaluate_all<P: Potential>(potential: &mut P, frames: &[&Frame]) -> Result<Vec<Evaluation>> {
    let evaluations = potential.eval_batch(frames)?;
    if evaluations.len() != frames.len() {
        return Err(Error(format!(
            "the model returned {} evaluations for {} frames",
            evaluations.len(),
            frames.len()
        )));
    }
    Ok(evaluations)
}

fn of_replica(index: usize) -> impl Fn(Error) -> Error {
    move |e| Error(format!("replica {index}: {e}"))
}

impl<P: Potential> MlipReplicas<P> {
    /// Starts dynamics of every system with timestep in femtoseconds; the
    /// initial forces of all are evaluated together. The systems need not be
    /// alike.
    pub fn new(systems: Vec<System>, mut potential: P, timestep: f64) -> Result<Self> {
        let ev_to_internal = checked_timestep(timestep)?;
        if systems.is_empty() {
            return Err(Error("replica dynamics needs at least one system".into()));
        }
        potential.metadata().validate()?;
        let mut replicas = Vec::with_capacity(systems.len());
        for (index, system) in systems.into_iter().enumerate() {
            let replica = Replica::new(system).map_err(of_replica(index))?;
            let _ = replica
                .input
                .type_ids(potential.metadata())
                .map_err(of_replica(index))?;
            replicas.push(replica);
        }
        let inputs: Vec<&Frame> = replicas.iter().map(|replica| &replica.input).collect();
        let evaluations = evaluate_all(&mut potential, &inputs)?;
        for (index, (replica, evaluation)) in replicas.iter_mut().zip(evaluations).enumerate() {
            replica
                .start(evaluation, ev_to_internal)
                .map_err(of_replica(index))?;
        }
        Ok(Self {
            replicas,
            potential,
            timestep,
            ev_to_internal,
        })
    }

    /// Number of systems.
    pub fn len(&self) -> usize {
        self.replicas.len()
    }

    /// Never: there is at least one system.
    pub fn is_empty(&self) -> bool {
        self.replicas.is_empty()
    }

    /// The state of system `index`.
    pub fn view(&self, index: usize) -> StateView<'_> {
        self.replicas[index].view(
            self.potential.metadata(),
            self.timestep,
            self.ev_to_internal,
        )
    }

    /// Metadata of the selected backend.
    pub fn metadata(&self) -> &ModelMetadata {
        self.potential.metadata()
    }

    /// Time increment per step, in femtoseconds.
    pub fn timestep_fs(&self) -> f64 {
        self.timestep
    }

    /// One step of every system, each with its own controls (one entry per
    /// system; `Controls::default()` is plain velocity Verlet). The model is
    /// asked once for all systems, and once more for those with a barostat.
    /// Returns what the controls delivered to each system.
    ///
    /// The step is transactional over all systems: if any part of it fails
    /// for one of them, none has moved. As for
    /// [`MlipDynamics::step_with_thermostat`], the caller takes back the
    /// random state of its controls on error.
    pub fn step(&mut self, controls: &mut [Controls<'_>]) -> Result<Vec<Exchange>> {
        if controls.len() != self.replicas.len() {
            return Err(Error(format!(
                "{} systems need as many controls, got {}",
                self.replicas.len(),
                controls.len()
            )));
        }
        let result = self.advance(controls);
        if result.is_err() {
            for replica in &mut self.replicas {
                replica.undo();
            }
        }
        result
    }

    fn advance(&mut self, controls: &mut [Controls<'_>]) -> Result<Vec<Exchange>> {
        let (timestep, ev_to_internal) = (self.timestep, self.ev_to_internal);
        for (index, replica) in self.replicas.iter_mut().enumerate() {
            replica
                .begin(timestep, ev_to_internal)
                .map_err(of_replica(index))?;
        }
        let inputs: Vec<&Frame> = self.replicas.iter().map(|r| &r.input).collect();
        let mut evaluated = evaluate_all(&mut self.potential, &inputs)?;
        let mut exchanges = vec![Exchange::default(); self.replicas.len()];
        let all = self.replicas.iter_mut().zip(controls.iter_mut()).enumerate();
        for (index, (replica, control)) in all {
            let thermostat = control.thermostat();
            exchanges[index].heat = replica
                .complete(&evaluated[index], timestep, ev_to_internal, thermostat)
                .map_err(of_replica(index))?;
        }
        // The cell moves that are due, evaluated together.
        let mut moving = Vec::new();
        let all = self.replicas.iter_mut().zip(controls.iter_mut()).enumerate();
        for (index, (replica, control)) in all {
            if let Some(barostat) = control.barostat() {
                let before = replica
                    .propose(&evaluated[index], ev_to_internal, barostat)
                    .map_err(of_replica(index))?;
                moving.push((index, before));
            }
        }
        if !moving.is_empty() {
            let inputs: Vec<&Frame> = moving
                .iter()
                .map(|&(index, _)| &self.replicas[index].input)
                .collect();
            let moved = evaluate_all(&mut self.potential, &inputs)?;
            for (&(index, before), moved) in moving.iter().zip(moved) {
                let Some(barostat) = controls[index].barostat() else {
                    return Err(Error("a barostat went missing during its move".into()));
                };
                let (accepted, work) = self.replicas[index]
                    .settle(
                        &mut evaluated[index],
                        moved,
                        before,
                        ev_to_internal,
                        barostat,
                    )
                    .map_err(of_replica(index))?;
                exchanges[index].cell_move = Some(accepted);
                exchanges[index].work = work;
            }
        }
        // Every system is checked before any of them keeps its step.
        for (index, (replica, evaluated)) in self.replicas.iter_mut().zip(&evaluated).enumerate()
        {
            replica
                .check(evaluated, ev_to_internal)
                .map_err(of_replica(index))?;
        }
        for (replica, evaluated) in self.replicas.iter_mut().zip(evaluated) {
            replica.commit(evaluated);
        }
        Ok(exchanges)
    }
}
