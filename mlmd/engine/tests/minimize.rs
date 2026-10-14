use omnimd_ml_contract::{Error, Evaluation, Frame, ModelMetadata, Potential, Result, Units};
use omnimd_sim::min::{
    MlipMinimizationSettings as Settings, MlipMinimizationStatus as Status, MlipMinimizer,
};
struct Harmonic {
    metadata: ModelMetadata,
    calls: usize,
    fail: bool,
}
impl Harmonic {
    fn new(fail: bool) -> Self {
        Self {
            metadata: ModelMetadata {
                type_map: vec!["H".into()],
                cutoff: 4.,
                backend: "synthetic".into(),
                precision: "float64".into(),
                units: Units::EvAngstrom,
            },
            calls: 0,
            fail,
        }
    }
}
impl Potential for Harmonic {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }
    fn eval(&mut self, frame: &Frame) -> Result<Evaluation> {
        self.calls += 1;
        if self.fail && self.calls == 2 {
            return Err(Error("injected evaluation error".into()));
        }
        Ok(Evaluation {
            energy: frame.positions.iter().flatten().map(|x| 0.5 * x * x).sum(),
            forces: frame.positions.iter().map(|r| r.map(|x| -x)).collect(),
            virial: [[0.; 3]; 3],
            units: Units::EvAngstrom,
        })
    }
}
fn frame() -> Frame {
    Frame {
        positions: vec![[1., 0.3, -0.4]],
        species: vec!["H".into()],
        cell: Some([[8., 0., 0.], [1., 8., 0.], [0., 1., 8.]]),
        units: Units::EvAngstrom,
    }
}
fn settings() -> Settings {
    Settings {
        force_tolerance_ev_per_angstrom: 1e-6,
        energy_tolerance_ev: 1e-9,
        initial_step_angstrom2_per_ev: 5.,
        max_displacement_angstrom: 0.2,
        max_backtracks: 32,
        max_iterations: 200,
        armijo: 1e-4,
    }
}
#[test]
fn descent_converges_with_displacement_cap_and_accepted_output_forces() {
    let original = frame();
    let mut min = MlipMinimizer::new(Harmonic::new(false), original.clone(), settings()).unwrap();
    while min.status() == Status::Running {
        let before = min.frame().positions[0];
        let energy = min.evaluation().energy;
        min.step().unwrap();
        if min.energy_change_ev().is_some() {
            let after = min.frame().positions[0];
            assert!(
                (after[0] - before[0])
                    .hypot(after[1] - before[1])
                    .hypot(after[2] - before[2])
                    <= 0.2 + 1e-14
            );
            assert!(min.evaluation().energy <= energy);
        }
        assert_eq!(min.frame().cell, original.cell);
    }
    assert_eq!(min.status(), Status::Converged);
    assert!(min.energy_change_ev().unwrap().abs() <= 1e-9);
    assert!(min.max_force_ev_per_angstrom() <= 1e-6);
}
#[test]
fn backtracking_and_model_error_preserve_accepted_state() {
    let mut setup = settings();
    setup.max_displacement_angstrom = 100.;
    let mut min = MlipMinimizer::new(Harmonic::new(false), frame(), setup.clone()).unwrap();
    min.step().unwrap();
    assert!(min.evaluations() > 2);
    assert!(min.evaluation().energy < 0.625);
    let mut failed = MlipMinimizer::new(Harmonic::new(true), frame(), setup).unwrap();
    let positions = failed.frame().positions.clone();
    let energy = failed.evaluation().energy;
    assert!(failed.step().is_err());
    assert_eq!(failed.frame().positions, positions);
    assert_eq!(failed.evaluation().energy, energy);
    assert_eq!(failed.iterations(), 0);
}
#[test]
fn initial_stationarity_and_nonconvergence_are_explicit() {
    let mut zero = frame();
    zero.positions = vec![[0.; 3]];
    let min = MlipMinimizer::new(Harmonic::new(false), zero, settings()).unwrap();
    assert_eq!(min.status(), Status::AlreadyConverged);
    assert!(min.energy_change_ev().is_none());
    let mut limited = settings();
    limited.max_iterations = 1;
    let mut min = MlipMinimizer::new(Harmonic::new(false), frame(), limited).unwrap();
    assert_eq!(min.step().unwrap(), Status::MaxIterations);
    // Exactly reach zero force in one large move, but do not invent delta_E=0
    // to pass the energy criterion after that move.
    let mut exact = settings();
    exact.initial_step_angstrom2_per_ev = 1.;
    exact.max_displacement_angstrom = 10.;
    let mut min = MlipMinimizer::new(Harmonic::new(false), frame(), exact).unwrap();
    assert_eq!(min.step().unwrap(), Status::Running);
    assert_eq!(min.step().unwrap(), Status::Stalled);
    assert!(!min.status().converged());
    assert!(min.energy_change_ev().unwrap().abs() > 1e-9);
}

#[test]
fn legacy_harmonic_relaxation_survives_shared_step_policy_refactor() {
    use omnimd_core::{
        Harmonic as PairHarmonic, Molecule, PairInteraction, Particle, System, UnitCell,
    };
    use omnimd_sim::{
        min::{Minimization, SteepestDescent, Tolerance},
        Propagator,
    };
    let mut system = System::with_cell(UnitCell::cubic(20.));
    system.add_molecule(Molecule::new(Particle::with_position(
        "Cl",
        [0., 0., 0.].into(),
    )));
    system.add_molecule(Molecule::new(Particle::with_position(
        "Cl",
        [0., 0., 2.].into(),
    )));
    system.set_pair_potential(
        ("Cl", "Cl"),
        PairInteraction::new(Box::new(PairHarmonic { x0: 2.3, k: 0.1 }), 10.),
    );
    let mut min = Minimization::new(
        Box::new(SteepestDescent::new()),
        Tolerance {
            energy: 1e-10,
            force2: 1e-10,
        },
    );
    for _ in 0..100 {
        min.propagate(&mut system);
    }
    assert!(min.converged());
    assert!((system.distance(0, 1) - 2.3).abs() < 1e-3);
}
