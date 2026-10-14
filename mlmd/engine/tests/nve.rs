use omnimd_core::{units, Harmonic, PairInteraction};
use omnimd_ml_contract::{Error, Evaluation, Frame, ModelMetadata, Potential, Result, Units};
use omnimd_mlmd::{system_from_frame, MlipDynamics};
use std::{cell::Cell, rc::Rc};

struct PairModel {
    meta: ModelMetadata,
    calls: Rc<Cell<usize>>,
    fail_after: usize,
}
impl Potential for PairModel {
    fn metadata(&self) -> &ModelMetadata {
        &self.meta
    }
    fn eval(&mut self, f: &Frame) -> Result<Evaluation> {
        self.calls.set(self.calls.get() + 1);
        if self.calls.get() > self.fail_after {
            return Err(Error("injected model failure".into()));
        }
        let d: [f64; 3] = std::array::from_fn(|a| f.positions[1][a] - f.positions[0][a]);
        let r = d.iter().map(|x| x * x).sum::<f64>().sqrt();
        let force = d.map(|x| (r - 1.) * x / r);
        Ok(Evaluation {
            energy: 0.5 * (r - 1.).powi(2),
            forces: vec![force, force.map(|x| -x)],
            virial: std::array::from_fn(|a| std::array::from_fn(|b| -d[a] * force[b])),
            units: Units::EvAngstrom,
        })
    }
}
fn model(fail_after: usize) -> (PairModel, Rc<Cell<usize>>) {
    let calls = Rc::new(Cell::new(0));
    (
        PairModel {
            meta: ModelMetadata {
                type_map: vec!["H".into()],
                cutoff: 3.,
                backend: "synthetic-harmonic-pair".into(),
                precision: "float64".into(),
                units: Units::EvAngstrom,
            },
            calls: calls.clone(),
            fail_after,
        },
        calls,
    )
}
fn frame() -> Frame {
    Frame {
        positions: vec![[0., 0., 0.], [1.2, 0., 0.]],
        species: vec!["H".into(); 2],
        cell: None,
        units: Units::EvAngstrom,
    }
}

#[test]
fn nve_initial_force_unit_conversion_energy_and_single_evaluation() {
    let (m, calls) = model(usize::MAX);
    let mut md = MlipDynamics::new(
        system_from_frame(&frame(), &[1., 1.], &[[0.; 3]; 2]).unwrap(),
        m,
        0.1,
    )
    .unwrap();
    let initial = md.total_energy_ev();
    md.step().unwrap();
    let conversion = units::from(1., "eV").unwrap();
    assert!((md.frame().positions[0][0] - 0.5 * 0.1_f64.powi(2) * 0.2 * conversion).abs() < 1e-14);
    let mut drift = 0.0_f64;
    for _ in 1..400 {
        md.step().unwrap();
        drift = drift.max((md.total_energy_ev() - initial).abs());
    }
    assert!(drift < 2e-6, "NVE drift {drift}");
    let particles = md.system().particles();
    for a in 0..3 {
        assert!((particles.velocity[0][a] + particles.velocity[1][a]).abs() < 1e-12);
    }
    assert_eq!(calls.get(), 401);
    assert_eq!(md.system().step, 400);
}

#[test]
fn failing_step_is_transactional_and_cell_transposes_once() {
    let mut f = frame();
    f.cell = Some([[5., 0.3, 0.], [0.2, 6., 0.4], [0., 0., 7.]]);
    let (m, _) = model(1);
    let mut md = MlipDynamics::new(
        system_from_frame(&f, &[1., 1.], &[[0.; 3]; 2]).unwrap(),
        m,
        0.5,
    )
    .unwrap();
    assert_eq!(md.frame().cell, f.cell);
    assert!(md.step().is_err());
    assert_eq!(md.system().step, 0);
    assert_eq!(md.frame().positions, f.positions);
}

#[test]
fn rejects_invalid_masses_timestep_and_classical_mixing() {
    assert!(system_from_frame(&frame(), &[0., 1.], &[[0.; 3]; 2]).is_err());
    let (m, _) = model(usize::MAX);
    assert!(MlipDynamics::new(
        system_from_frame(&frame(), &[1., 1.], &[[0.; 3]; 2]).unwrap(),
        m,
        0.
    )
    .is_err());
    let mut s = system_from_frame(&frame(), &[1., 1.], &[[0.; 3]; 2]).unwrap();
    s.set_pair_potential(
        ("H", "H"),
        PairInteraction::new(Box::new(Harmonic { k: 1., x0: 1. }), 3.),
    );
    let (m, _) = model(usize::MAX);
    assert!(MlipDynamics::new(s, m, 0.5).is_err());
}

#[test]
fn checkpoint_resume_preserves_trajectory_and_rejects_model_swaps() {
    use omnimd_mlmd::Checkpoint;
    let (m, _) = model(usize::MAX);
    let mut md = MlipDynamics::new(
        system_from_frame(&frame(), &[1., 1.], &[[0.; 3]; 2]).unwrap(),
        m,
        0.1,
    )
    .unwrap();
    for _ in 0..20 {
        md.step().unwrap();
    }
    let digest = "a".repeat(64);
    let checkpoint = Checkpoint::capture(&md, digest.clone());
    let encoded = serde_json::to_string(&checkpoint).unwrap();
    let decoded: Checkpoint = serde_json::from_str(&encoded).unwrap();
    assert!(decoded.restore(&"b".repeat(64), 0.1).is_err());
    assert!(decoded.restore(&digest, 0.2).is_err());
    let (m, _) = model(usize::MAX);
    let mut resumed = MlipDynamics::new(decoded.restore(&digest, 0.1).unwrap(), m, 0.1).unwrap();
    for _ in 0..20 {
        md.step().unwrap();
        resumed.step().unwrap();
    }
    assert_eq!(resumed.system().step, 40);
    assert_eq!(resumed.frame().positions, md.frame().positions);
    assert_eq!(
        resumed.system().particles().velocity,
        md.system().particles().velocity
    );
    assert_eq!(resumed.evaluation().energy, md.evaluation().energy);
}
