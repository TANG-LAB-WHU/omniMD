use omnimd_ml_contract::{Frame, Potential};
use omnimd_native::{
    full_gradient::{frame_gradient, NetworkLocation, ParameterField, ParameterLayout},
    full_training::{train_full, Checkpoint, FullConfig},
    training::{Split, TrainConfig},
    training_data::{Dataset, DatasetFrame, DatasetHeader, Labels, VirialConvention},
    Model, NativePotential,
};
use std::collections::BTreeMap;
fn model() -> Model {
    serde_json::from_str(include_str!("../fixtures/full-tanh-v1.json")).unwrap()
}
fn config(steps: u64) -> FullConfig {
    FullConfig {
        cuda_gradient: Default::default(),
        split: Split::Seeded {
            seed: 20261015,
            validation_fraction: 0.25,
        },
        steps,
        energy_weight: 1.0,
        force_weight: 1.0,
        virial_loss: None,
        energy_scale_ev_per_atom: 0.1,
        force_scale_ev_per_angstrom: 0.1,
        learning_rate: 0.005,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        gradient_clip_norm: Some(10.0),
        checkpoint_every: 2,
    }
}
fn data(count: usize) -> (Model, Dataset) {
    let initial = model();
    let layout = ParameterLayout::new(&initial).unwrap();
    let mut teacher = initial.clone();
    let values: Vec<_> = layout
        .values(&initial)
        .iter()
        .enumerate()
        .map(|(i, x)| x + 0.12 * ((i + 1) as f64 * 0.63).sin())
        .collect();
    layout.apply(&mut teacher, &values).unwrap();
    let mut potential = NativePotential::new(teacher).unwrap();
    let mut frames = Vec::new();
    for i in 0..count {
        let mut f: Frame = serde_json::from_str(include_str!("../fixtures/frame.jsonl")).unwrap();
        if i % 2 == 0 {
            f.cell = None;
        } else {
            f.cell = Some([[3.8, 0.0, 0.0], [0.3, 3.9, 0.0], [0.1, 0.2, 4.0]]);
        }
        for (atom, r) in f.positions.iter_mut().enumerate() {
            for (axis, x) in r.iter_mut().enumerate() {
                *x += 0.19 * ((i + 1) as f64 * (atom * 3 + axis + 1) as f64 * 0.618).sin();
            }
        }
        let e = potential.eval(&f).unwrap();
        frames.push(DatasetFrame {
            index: i as u64,
            id: format!("set.000/{i}"),
            frame: f,
            labels: Labels {
                energy: Some(e.energy),
                forces: Some(e.forces),
                virial: Some(e.virial),
            },
        });
    }
    let header = DatasetHeader {
        schema: "omnimd-dataset-jsonl-v1".into(),
        type_map: initial.type_map.clone(),
        units: initial.units,
        virial_convention: VirialConvention::MinusDEpsilonRow,
        source: "original synthetic full-network teacher".into(),
        selected_sets: vec!["set.000".into()],
    };
    (
        initial,
        Dataset {
            header,
            frames,
            source_sha256: BTreeMap::new(),
            dataset_sha256: "in-memory-test".into(),
        },
    )
}
fn objective(model: &Model, record: &DatasetFrame, c: &TrainConfig) -> f64 {
    let p = NativePotential::new(model.clone())
        .unwrap()
        .eval(&record.frame)
        .unwrap();
    let n = record.frame.positions.len() as f64;
    c.energy_weight
        * ((p.energy - record.labels.energy.unwrap()) / n / c.energy_scale_ev_per_atom).powi(2)
        + c.force_weight / (3.0 * n)
            * p.forces
                .iter()
                .flatten()
                .zip(record.labels.forces.as_ref().unwrap().iter().flatten())
                .map(|(a, b)| ((a - b) / c.force_scale_ev_per_angstrom).powi(2))
                .sum::<f64>()
}
#[test]
fn every_parameter_mixed_gradient_matches_independent_native_loss_difference() {
    let (mut model, dataset) = data(2);
    let layout = ParameterLayout::new(&model).unwrap();
    let mut values = layout.values(&model);
    values[0] = 0.0;
    values[1] = 1.0;
    layout.apply(&mut model, &values).unwrap();
    let mut worst = 0.0_f64;
    for record in &dataset.frames {
        for (ew, fw) in [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
            let mut c = config(1).loss();
            c.energy_weight = ew;
            c.force_weight = fw;
            let (loss, gradient) = frame_gradient(&model, record, &c, 1).unwrap();
            assert!((loss - objective(&model, record, &c)).abs() < 1e-12);
            for i in 0..values.len() {
                let h = 2e-6 * values[i].abs().max(1.0);
                let (mut plus, mut minus) = (model.clone(), model.clone());
                let mut v = values.clone();
                v[i] += h;
                layout.apply(&mut plus, &v).unwrap();
                v[i] -= 2.0 * h;
                layout.apply(&mut minus, &v).unwrap();
                let fd = (objective(&plus, record, &c) - objective(&minus, record, &c)) / (2.0 * h);
                let error = (gradient[i] - fd).abs();
                worst = worst.max(error);
                assert!(
                    error < 2e-8 + 2e-5 * fd.abs(),
                    "param {i} {ew}/{fw}: AD {} FD {fd}",
                    gradient[i]
                );
                if ew == 0.0 {
                    if let NetworkLocation::Fitting { species } = layout.parameters[i].location {
                        if layout.parameters[i].layer + 1 == model.fitting[species].layers.len()
                            && matches!(layout.parameters[i].field, ParameterField::Bias { .. })
                        {
                            assert_eq!(gradient[i], 0.0);
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "all {} parameters x 3 objectives x 2 geometries: max absolute FD error {worst:e}",
        values.len()
    );
}
#[test]
fn adam_two_step_oracle_and_serialized_resume() {
    let (model, dataset) = data(4);
    let layout = ParameterLayout::new(&model).unwrap();
    let c = config(1);
    let initial = Checkpoint::start(&model, &dataset, &c).unwrap();
    let mut state = initial.clone();
    for step in 1..=2 {
        let mut g = vec![0.0; layout.parameters.len()];
        for &i in &state.training_indices {
            let (_, dg) = frame_gradient(
                &state.model,
                &dataset.frames[i],
                &c.loss(),
                state.training_indices.len(),
            )
            .unwrap();
            for (a, b) in g.iter_mut().zip(dg) {
                *a += b;
            }
        }
        let norm = g.iter().fold(0.0_f64, |a, b| a.hypot(*b));
        let clip = (10.0 / norm).min(1.0);
        for x in &mut g {
            *x *= clip;
        }
        let old = layout.values(&state.model);
        let before = state.clone();
        state = train_full(state, &dataset, &c, |_| Ok(()))
            .unwrap()
            .checkpoint;
        let actual = layout.values(&state.model);
        for i in 0..g.len() {
            let m = 0.9 * before.first_moment[i] + 0.1 * g[i];
            let v = 0.999 * before.second_moment[i] + 0.001 * g[i] * g[i];
            let expected = old[i]
                - 0.005 * (m / (1.0 - 0.9_f64.powi(step)))
                    / ((v / (1.0 - 0.999_f64.powi(step))).sqrt() + 1e-8);
            assert!((state.first_moment[i] - m).abs() < 1e-13);
            assert!((state.second_moment[i] - v).abs() < 1e-13);
            assert!((actual[i] - expected).abs() < 1e-13);
        }
        assert_eq!(state.step, step as u64);
    }
    let continuous = train_full(initial, &dataset, &config(5), |_| Ok(())).unwrap();
    let serialized = serde_json::to_string(&state).unwrap();
    let restored: Checkpoint = serde_json::from_str(&serialized).unwrap();
    let split = train_full(restored.clone(), &dataset, &config(3), |_| Ok(())).unwrap();
    assert_eq!(
        serde_json::to_value(&continuous.checkpoint.model).unwrap(),
        serde_json::to_value(&split.checkpoint.model).unwrap()
    );
    assert_eq!(
        continuous.checkpoint.first_moment,
        split.checkpoint.first_moment
    );
    assert_eq!(
        continuous.checkpoint.second_moment,
        split.checkpoint.second_moment
    );
    assert_eq!(
        continuous.report.final_training.normalized_ef_objective,
        split.report.final_training.normalized_ef_objective
    );
    let mut bad = restored.clone();
    bad.model.descriptor.normalization[0].std[0] += 0.1;
    assert!(train_full(bad, &dataset, &config(1), |_| Ok(())).is_err());
    let mut bad = restored.clone();
    bad.dataset_sha256 = "different".into();
    assert!(train_full(bad, &dataset, &config(1), |_| Ok(())).is_err());
    let mut altered = config(1);
    altered.learning_rate *= 2.0;
    assert!(train_full(restored, &dataset, &altered, |_| Ok(())).is_err());
}

#[test]
fn validation_labels_never_enter_updates_and_unsupported_activation_fails() {
    let (model, mut dataset) = data(4);
    let c = config(2);
    let state = Checkpoint::start(&model, &dataset, &c).unwrap();
    let trained = train_full(state.clone(), &dataset, &c, |_| Ok(())).unwrap();
    for &i in &state.validation_indices {
        *dataset.frames[i].labels.energy.as_mut().unwrap() += 9.0;
        for f in dataset.frames[i]
            .labels
            .forces
            .as_mut()
            .unwrap()
            .iter_mut()
            .flatten()
        {
            *f += 4.0;
        }
    }
    let altered = train_full(
        Checkpoint::start(&model, &dataset, &c).unwrap(),
        &dataset,
        &c,
        |_| Ok(()),
    )
    .unwrap();
    assert_eq!(
        serde_json::to_value(trained.checkpoint.model).unwrap(),
        serde_json::to_value(altered.checkpoint.model).unwrap()
    );
    assert_eq!(
        trained.checkpoint.first_moment,
        altered.checkpoint.first_moment
    );
    assert!(
        altered.report.final_validation.normalized_ef_objective
            > trained.report.final_validation.normalized_ef_objective
    );
    let mut unsupported = model;
    unsupported.descriptor.embedding[0][0].layers[0].activation =
        omnimd_native::Activation::Sigmoid;
    assert!(Checkpoint::start(&unsupported, &dataset, &c).is_err());
}
