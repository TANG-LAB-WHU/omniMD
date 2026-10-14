use omnimd_ml_contract::{Frame, Potential};
use omnimd_native::{
    readout::ReadoutBasis,
    training::{train, Split, TrainConfig},
    training_data::{Dataset, DatasetFrame, DatasetHeader, Labels, VirialConvention},
    Model, NativePotential,
};
use std::collections::BTreeMap;

fn model() -> Model {
    serde_json::from_str(include_str!("../fixtures/synthetic-v1.json")).unwrap()
}
fn frame() -> Frame {
    serde_json::from_str(include_str!("../fixtures/frame.jsonl")).unwrap()
}

#[test]
fn analytic_basis_matches_arbitrary_readout_and_coordinate_difference() {
    let mut model = model();
    model.atom_energy_bias = [-9e4, 3e4].to_vec(); // large offsets must not contaminate features
    let mut basis = ReadoutBasis::new(&model, true).unwrap();
    let f = frame();
    let features = basis.features(&f).unwrap();
    let coefficients: Vec<_> = (0..basis.parameters.len())
        .map(|i| 0.3 * ((i + 1) as f64).sin())
        .collect();
    let changed = basis.model_with_coefficients(&coefficients).unwrap();
    let expected = NativePotential::new(changed).unwrap().eval(&f).unwrap();
    let actual = features.evaluate(&coefficients).unwrap();
    assert!((actual.energy - expected.energy).abs() < 1e-9);
    for (a, b) in actual
        .forces
        .iter()
        .flatten()
        .zip(expected.forces.iter().flatten())
    {
        assert!((a - b).abs() < 1e-12);
    }
    for (a, b) in actual
        .virial
        .iter()
        .flatten()
        .zip(expected.virial.iter().flatten())
    {
        assert!((a - b).abs() < 1e-12);
    }
    let h = 1e-5;
    let (mut plus, mut minus) = (f.clone(), f.clone());
    plus.positions[1][1] += h;
    minus.positions[1][1] -= h;
    let fp = basis.features(&plus).unwrap();
    let fm = basis.features(&minus).unwrap();
    for k in 0..coefficients.len() {
        let finite_difference = -(fp.columns[k].energy - fm.columns[k].energy) / (2.0 * h);
        assert!((features.columns[k].forces[1][1] - finite_difference).abs() < 1e-8);
    }
}

fn manufactured() -> (Model, Dataset) {
    let mut initial = model();
    let mut teacher = initial.clone();
    for (kind, net) in teacher.fitting.iter_mut().enumerate() {
        for (i, w) in net
            .layers
            .last_mut()
            .unwrap()
            .weights
            .iter_mut()
            .enumerate()
        {
            w[0] = 0.8 * ((kind * 7 + i + 1) as f64 * 0.7).sin();
        }
    }
    for net in &mut initial.fitting {
        for w in &mut net.layers.last_mut().unwrap().weights {
            w[0] = 0.0;
        }
    }
    let mut potential = NativePotential::new(teacher).unwrap();
    let mut frames = Vec::new();
    for index in 0..32 {
        let mut f = frame();
        if index % 2 == 0 {
            f.cell = None;
        }
        for (atom, r) in f.positions.iter_mut().enumerate() {
            for (axis, x) in r.iter_mut().enumerate() {
                *x += 0.22 * ((index + 1) as f64 * (atom * 3 + axis + 1) as f64 * 0.618).sin();
            }
        }
        let label = potential.eval(&f).unwrap();
        frames.push(DatasetFrame {
            index: index as u64,
            id: format!("set.000/{index}"),
            frame: f,
            labels: Labels {
                energy: Some(label.energy),
                forces: Some(label.forces),
                virial: Some(label.virial),
            },
        });
    }
    let header = DatasetHeader {
        schema: "omnimd-dataset-jsonl-v1".into(),
        type_map: initial.type_map.clone(),
        units: initial.units,
        virial_convention: VirialConvention::MinusDEpsilonRow,
        source: "manufactured in-memory test".into(),
        selected_sets: vec!["set.000".into()],
    };
    (
        initial,
        Dataset {
            header,
            frames,
            source_sha256: BTreeMap::new(),
            dataset_sha256: "in-memory unit fixture".into(),
        },
    )
}

#[test]
fn fits_manufactured_energy_forces_and_never_fits_validation_labels() {
    let (initial, mut dataset) = manufactured();
    let config = TrainConfig {
        split: Split::Seeded {
            seed: 20261014,
            validation_fraction: 0.25,
        },
        fit_bias: false,
        energy_weight: 1.0,
        force_weight: 1.0,
        energy_scale_ev_per_atom: 1.0,
        force_scale_ev_per_angstrom: 0.01,
        ridge_inverse_ev2: 0.0,
        rank_rtol: 1e-12,
    };
    let trained = train(&initial, &dataset, &config).unwrap();
    assert_eq!(trained.report.training_frames.len(), 24);
    assert_eq!(trained.report.validation_frames.len(), 8);
    assert!(
        trained.report.final_training.normalized_ef_objective
            < trained.report.initial_training.normalized_ef_objective * 1e-10
    );
    assert!(trained.report.final_validation.energy_per_atom_rmse_ev < 1e-8);
    assert!(
        trained
            .report
            .final_validation
            .force_component_rmse_ev_per_angstrom
            < 1e-8
    );
    assert_eq!(
        serde_json::to_value(&initial.descriptor).unwrap(),
        serde_json::to_value(&trained.model.descriptor).unwrap()
    );
    assert_eq!(initial.atom_energy_bias, trained.model.atom_energy_bias);
    for (a, b) in initial.fitting.iter().zip(&trained.model.fitting) {
        assert_eq!(
            serde_json::to_value(&a.layers[..a.layers.len() - 1]).unwrap(),
            serde_json::to_value(&b.layers[..b.layers.len() - 1]).unwrap()
        );
        assert_eq!(a.layers.last().unwrap().bias, b.layers.last().unwrap().bias);
    }
    for held_out in &trained.report.validation_frames {
        let labels = &mut dataset.frames[held_out.index as usize].labels;
        *labels.energy.as_mut().unwrap() += 7.0;
        for f in labels.forces.as_mut().unwrap().iter_mut().flatten() {
            *f += 2.0;
        }
    }
    let changed = train(&initial, &dataset, &config).unwrap();
    assert_eq!(
        serde_json::to_string(&trained.model).unwrap(),
        serde_json::to_string(&changed.model).unwrap()
    );
    assert!(changed.report.final_validation.normalized_ef_objective > 1.0);
}
