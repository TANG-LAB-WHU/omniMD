use omnimd_ml_contract::{
    Committee, Evaluation, Frame, ModelMetadata, Potential, Result, Separate, Units,
};
use omnimd_mlmd::deviation::{among, evaluate_frame, evaluate_frames, ForceWindow};

#[derive(Clone)]
struct Fixed {
    metadata: ModelMetadata,
    value: Evaluation,
    slope: f64,
    calls: usize,
}
impl Potential for Fixed {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }
    fn eval(&mut self, frame: &Frame) -> Result<Evaluation> {
        self.calls += 1;
        // The answer follows the frame: its first coordinate stretches the
        // value by a factor of the model's own.
        let stretch = 1. + self.slope * frame.positions[0][0];
        let mut value = self.value.clone();
        value.energy *= stretch;
        for component in value.forces.iter_mut().flatten() {
            *component *= stretch;
        }
        for component in value.virial.iter_mut().flatten() {
            *component *= stretch;
        }
        Ok(value)
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
        slope: 0.,
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
#[test]
fn a_committee_gives_each_frame_the_statistics_of_its_members() {
    let (frame, mut models) = fixture();
    let mut third = models[1].clone();
    third.value.energy = -1.;
    third.value.forces[1] = [2., -1., 0.5];
    models.push(third);
    for (member, model) in models.iter_mut().enumerate() {
        model.slope = 0.3 * (member as f64 - 0.8);
    }
    let frames: Vec<Frame> = [0., 1.5, -2.]
        .iter()
        .map(|&x| {
            let mut frame = frame.clone();
            frame.positions[0][0] = x;
            frame
        })
        .collect();
    let batch: Vec<&Frame> = frames.iter().collect();
    let text = |s| serde_json::to_string(&s).unwrap();
    // Frame by frame, member by member.
    let expected: Vec<String> = frames
        .iter()
        .map(|frame| text(evaluate_frame(&mut models, frame, true).unwrap()))
        .collect();
    assert_ne!(expected[0], expected[1]);
    let mut committee = Separate::new(models).unwrap();
    let got = evaluate_frames(&mut committee, &batch, true).unwrap();
    assert_eq!(got.into_iter().map(text).collect::<Vec<_>>(), expected);
    assert!(committee.0.iter().all(|model| model.calls == 6));
    assert!(evaluate_frames(&mut committee, &[], true)
        .unwrap()
        .is_empty());
    let brief = evaluate_frames(&mut committee, &batch[..1], false).unwrap();
    assert!(brief[0].force_ev_per_angstrom.per_atom.is_none());
    // Through a trait object, as the engine holds it.
    let boxed: &mut dyn Committee = &mut committee;
    assert_eq!(
        text(
            evaluate_frames(boxed, &batch[1..2], true)
                .unwrap()
                .remove(0)
        ),
        expected[1]
    );

    // No frame is evaluated when one of them does not fit the models.
    let calls = committee.0[0].calls;
    let mut foreign = frames[0].clone();
    foreign.species[1] = "C".into();
    assert!(evaluate_frames(&mut committee, &[&frames[0], &foreign], true).is_err());
    assert_eq!(committee.0[0].calls, calls);
    // A member's bad output is an error that names it.
    committee.0[2].value.forces.pop();
    let error = evaluate_frames(&mut committee, &batch, true).unwrap_err();
    assert!(error.to_string().contains("ensemble member 2"), "{error}");
    // Two to sixteen members.
    let (_, models) = fixture();
    let mut alone = Separate::new(models[..1].to_vec()).unwrap();
    assert!(evaluate_frames(&mut alone, &batch, true).is_err());
    assert_eq!(alone.0[0].calls, 0);
    let mut crowd = Separate::new(vec![models[0].clone(); 17]).unwrap();
    assert!(evaluate_frames(&mut crowd, &batch, true).is_err());
    let mut full = Separate::new(vec![models[0].clone(); 16]).unwrap();
    assert_eq!(evaluate_frames(&mut full, &batch, true).unwrap().len(), 3);
}
#[test]
fn deviation_among_given_evaluations_needs_two_to_sixteen_of_one_shape() {
    let (frame, mut models) = fixture();
    let evaluations: Vec<Evaluation> = models.iter().map(|m| m.value.clone()).collect();
    let text = |s| serde_json::to_string(&s).unwrap();
    assert_eq!(
        text(among(&evaluations, 2, true).unwrap()),
        text(evaluate_frame(&mut models, &frame, true).unwrap())
    );
    assert!(among(&evaluations[..1], 2, true).is_err());
    assert!(among(&vec![evaluations[0].clone(); 17], 2, true).is_err());
    // The atoms of the frame, not of the first evaluation, set the shape.
    let error = among(&evaluations, 3, true).unwrap_err();
    assert!(error.to_string().contains("ensemble member 0"), "{error}");
    let mut short = evaluations.clone();
    short[1].forces.pop();
    let error = among(&short, 2, true).unwrap_err();
    assert!(error.to_string().contains("ensemble member 1"), "{error}");
}
