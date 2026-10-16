use omnimd_ml_contract::{Evaluation, Frame, ModelMetadata, Potential, Result, Units};
use omnimd_mlmd::deviation::{evaluate_frame, ForceWindow};

#[derive(Clone)]
struct Fixed {
    metadata: ModelMetadata,
    value: Evaluation,
    calls: usize,
}
impl Potential for Fixed {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }
    fn eval(&mut self, _: &Frame) -> Result<Evaluation> {
        self.calls += 1;
        Ok(self.value.clone())
    }
}
fn fixture() -> (Frame, Vec<Fixed>) {
    let frame = Frame {
        positions: vec![[0., 0., 0.], [1., 0., 0.]],
        species: vec!["O".into(), "H".into()],
        cell: None,
        units: Units::EvAngstrom,
    };
    let a = Fixed {
        metadata: ModelMetadata {
            type_map: vec!["O".into(), "H".into()],
            cutoff: 4.,
            backend: "oracle".into(),
            precision: "f64".into(),
            units: Units::EvAngstrom,
        },
        value: Evaluation {
            energy: 2.,
            forces: vec![[1., 2., 3.], [0.; 3]],
            virial: [[0.; 3]; 3],
            units: Units::EvAngstrom,
        },
        calls: 0,
    };
    let mut b = a.clone();
    b.value.energy = 6.;
    b.value.forces = vec![[3., 6., 3.], [0., 0., 6.]];
    b.value.virial = [[4., 8., 0.], [0., 0., 12.], [0.; 3]];
    (frame, vec![a, b])
}
fn near(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-14, "{a} != {b}");
}
#[test]
fn population_oracle_vector_forces_and_all_nine_virial_components() {
    let (frame, mut models) = fixture();
    let s = evaluate_frame(&mut models, &frame, true).unwrap();
    near(s.energy_per_atom_std_ev, 1.);
    near(s.mean_energy_per_atom_ev, 2.);
    near(s.force_ev_per_angstrom.min, 5_f64.sqrt());
    near(s.force_ev_per_angstrom.max, 3.);
    near(s.force_ev_per_angstrom.mean, (5_f64.sqrt() + 3.) / 2.);
    assert_eq!(
        s.force_ev_per_angstrom.per_atom.unwrap(),
        vec![5_f64.sqrt(), 3.]
    );
    assert_eq!(
        s.virial_per_atom_ev.components,
        [[1., 2., 0.], [0., 0., 3.], [0.; 3]]
    );
    near(s.virial_per_atom_ev.rms, 14_f64.sqrt() / 3.);
    near(s.virial_per_atom_ev.min, 0.);
    near(s.virial_per_atom_ev.max, 3.);
    assert!(models.iter().all(|p| p.calls == 1));
    assert!(evaluate_frame(&mut models, &frame, false)
        .unwrap()
        .force_ev_per_angstrom
        .per_atom
        .is_none());
    assert!(models.iter().all(|p| p.calls == 2));
    models[1].value = models[0].value.clone();
    let zero = evaluate_frame(&mut models, &frame, true).unwrap();
    assert_eq!(zero.energy_per_atom_std_ev, 0.);
    assert_eq!(zero.force_ev_per_angstrom.max, 0.);
    assert_eq!(zero.virial_per_atom_ev.max, 0.);
}
#[test]
fn metadata_output_and_overflow_failures_do_not_return_partial_statistics() {
    let (frame, mut models) = fixture();
    assert!(evaluate_frame(&mut models[..1], &frame, false).is_err());
    models[1].metadata.type_map.reverse();
    assert!(evaluate_frame(&mut models, &frame, false).is_err());
    assert!(models.iter().all(|p| p.calls == 0));
    models[1].metadata.type_map.reverse();
    models[0].value.forces.pop();
    assert!(evaluate_frame(&mut models, &frame, false).is_err());
    assert_eq!(models[1].calls, 0);
    let (_, mut models) = fixture();
    models[0].value.energy = f64::NAN;
    assert!(evaluate_frame(&mut models, &frame, false).is_err());
    assert_eq!(models[1].calls, 0);
    let (_, mut models) = fixture();
    models[0].value.forces[0][0] = 1e308;
    models[1].value.forces[0][0] = -1e308;
    assert!(evaluate_frame(&mut models, &frame, false)
        .unwrap_err()
        .to_string()
        .contains("overflow"));
}
#[test]
fn force_window_has_explicit_half_open_bounds() {
    let window = ForceWindow {
        min_ev_per_angstrom: 0.2,
        max_ev_per_angstrom: 0.4,
    };
    window.validate().unwrap();
    assert!(!window.contains(0.199));
    assert!(window.contains(0.2));
    assert!(window.contains(0.399));
    assert!(!window.contains(0.4));
    for (min, max) in [
        (0., 0.),
        (-1., 1.),
        (1., 0.),
        (0., f64::INFINITY),
        (f64::NAN, 1.),
    ] {
        assert!(ForceWindow {
            min_ev_per_angstrom: min,
            max_ev_per_angstrom: max
        }
        .validate()
        .is_err());
    }
}
