//! The se_e2_r descriptor on the CPU: against DeePMD-kit itself, against
//! finite differences and closed forms, and what the format refuses.
use omnimd_ml_contract::{Evaluation, Frame, Potential, Units};
use omnimd_native::{
    full_gradient::ParameterLayout, readout::ReadoutBasis, Activation, Descriptor, DescriptorKind,
    Layer, Model, NativePotential, Network, Normalization, Residual,
};

#[derive(serde::Deserialize)]
struct Case {
    model: String,
    frame: Frame,
    energy: f64,
    forces: Vec<[f64; 3]>,
    virial: [[f64; 3]; 3],
}

#[derive(serde::Deserialize)]
struct Reference {
    deepmd_version: String,
    cases: Vec<Case>,
}

fn reference() -> Reference {
    serde_json::from_str(include_str!("../fixtures/se-r-deepmd-reference.json")).unwrap()
}

/// The fixture model whose `sel` holds every neighbor of the fixture frames.
fn model() -> Model {
    serde_json::from_str(include_str!("../fixtures/se-r-model.native.json")).unwrap()
}

fn tight() -> Model {
    serde_json::from_str(include_str!("../fixtures/se-r-tight-model.native.json")).unwrap()
}

fn eval(model: &Model, frame: &Frame) -> Evaluation {
    NativePotential::new(model.clone())
        .unwrap()
        .eval(frame)
        .unwrap()
}

fn close(a: f64, b: f64, tolerance: f64) {
    assert!(
        (a - b).abs() <= tolerance * (1.0 + a.abs().max(b.abs())),
        "{a:.16e} != {b:.16e} (tol {tolerance:e})"
    );
}

#[test]
fn energy_forces_and_virial_match_deepmd_kit() {
    let reference = reference();
    assert_eq!(reference.deepmd_version, "3.1.3");
    let model = model();
    assert_eq!(model.descriptor_kind(), DescriptorKind::SeE2R);
    let mut potential = NativePotential::new(model).unwrap();
    assert_eq!(potential.metadata().backend, "native-se-e2-r-v1");
    let mut compared = 0;
    for case in &reference.cases {
        if case.model != "se-r-model.native.json" {
            continue;
        }
        let ours = potential.eval(&case.frame).unwrap();
        assert!(
            (ours.energy - case.energy).abs() < 1e-11,
            "energy {} against {}",
            ours.energy,
            case.energy
        );
        for (atom, (a, b)) in ours.forces.iter().zip(&case.forces).enumerate() {
            for axis in 0..3 {
                assert!(
                    (a[axis] - b[axis]).abs() < 1e-13,
                    "force of atom {atom}: {} against {}",
                    a[axis],
                    b[axis]
                );
            }
        }
        for (a, b) in ours
            .virial
            .iter()
            .flatten()
            .zip(case.virial.iter().flatten())
        {
            assert!((a - b).abs() < 1e-12, "virial {a} against {b}");
        }
        compared += 1;
    }
    assert_eq!(
        compared, 2,
        "the periodic frame and the same atoms without a cell"
    );
    // The forces are not small: the comparison means something.
    let largest = reference.cases[0]
        .forces
        .iter()
        .flatten()
        .fold(0.0f64, |m, f| m.max(f.abs()));
    assert!(largest > 1e-3, "{largest}");
    // More neighbors than `sel`: the CPU reference refuses, as for se_e2_a.
    let case = reference
        .cases
        .iter()
        .find(|case| case.model == "se-r-tight-model.native.json")
        .unwrap();
    assert!(NativePotential::new(tight())
        .unwrap()
        .eval(&case.frame)
        .is_err());
}

fn deform(f: &Frame, a: usize, b: usize, amount: f64) -> Frame {
    let mut f = f.clone();
    for p in &mut f.positions {
        p[b] += amount * p[a];
    }
    if let Some(cell) = &mut f.cell {
        for row in cell {
            row[b] += amount * row[a];
        }
    }
    f
}

#[test]
fn forces_and_virial_are_the_derivatives_of_the_energy() {
    let model = model();
    for case in &reference().cases[..2] {
        let f = &case.frame;
        let result = eval(&model, f);
        let h = 2e-5;
        for i in (0..f.positions.len()).step_by(4) {
            for a in 0..3 {
                let (mut plus, mut minus) = (f.clone(), f.clone());
                plus.positions[i][a] += h;
                minus.positions[i][a] -= h;
                close(
                    result.forces[i][a],
                    -(eval(&model, &plus).energy - eval(&model, &minus).energy) / (2.0 * h),
                    2e-8,
                );
            }
        }
        for a in 0..3 {
            for b in 0..3 {
                let (plus, minus) = (deform(f, a, b, h), deform(f, a, b, -h));
                close(
                    result.virial[a][b],
                    -(eval(&model, &plus).energy - eval(&model, &minus).energy) / (2.0 * h),
                    3e-8,
                );
            }
        }
        // No net force, and without a cell no net torque either.
        for a in 0..3 {
            let total: f64 = result.forces.iter().map(|force| force[a]).sum();
            assert!(total.abs() < 1e-12, "{total}");
        }
        if f.cell.is_none() {
            let torque: f64 = f
                .positions
                .iter()
                .zip(&result.forces)
                .map(|(r, force)| r[0] * force[1] - r[1] * force[0])
                .sum();
            assert!(torque.abs() < 1e-11, "{torque}");
        }
    }
}

#[test]
fn translation_permutation_rotation_and_wrapping_leave_the_energy() {
    let model = model();
    let reference = reference();
    let f = &reference.cases[0].frame;
    let e = eval(&model, f);
    let mut translated = f.clone();
    for p in &mut translated.positions {
        for a in 0..3 {
            p[a] += [2.4, -3.1, 1.8][a];
        }
    }
    let t = eval(&model, &translated);
    close(t.energy, e.energy, 1e-13);
    for (a, b) in t.forces.iter().flatten().zip(e.forces.iter().flatten()) {
        close(*a, *b, 1e-11);
    }
    // The atoms in reverse order.
    let n = f.positions.len();
    let mut permuted = f.clone();
    permuted.positions.reverse();
    permuted.species.reverse();
    let p = eval(&model, &permuted);
    close(p.energy, e.energy, 1e-13);
    for i in 0..n {
        for a in 0..3 {
            close(p.forces[i][a], e.forces[n - 1 - i][a], 1e-11);
        }
    }
    let cell = f.cell.unwrap();
    let mut wrapped = f.clone();
    for a in 0..3 {
        wrapped.positions[1][a] += 2.0 * cell[0][a] - 3.0 * cell[1][a] + cell[2][a];
    }
    let w = eval(&model, &wrapped);
    close(w.energy, e.energy, 1e-13);
    for (a, b) in w.forces.iter().flatten().zip(e.forces.iter().flatten()) {
        close(*a, *b, 1e-11);
    }
    // A proper rotation: the energy stays, forces and virial turn with it.
    let q = [[0.36, -0.48, 0.8], [0.8, 0.6, 0.0], [-0.48, 0.64, 0.6]];
    let rotate =
        |v: [f64; 3]| -> [f64; 3] { std::array::from_fn(|a| (0..3).map(|b| q[a][b] * v[b]).sum()) };
    let mut rotated = f.clone();
    rotated.positions = f.positions.iter().map(|v| rotate(*v)).collect();
    rotated.cell = Some(cell.map(rotate));
    let r = eval(&model, &rotated);
    close(r.energy, e.energy, 1e-13);
    for (turned, original) in r.forces.iter().zip(&e.forces) {
        let expected = rotate(*original);
        for a in 0..3 {
            close(turned[a], expected[a], 1e-11);
        }
    }
    for a in 0..3 {
        for b in 0..3 {
            let expected: f64 = (0..3)
                .flat_map(|c| (0..3).map(move |d| (c, d)))
                .map(|(c, d)| q[a][c] * e.virial[c][d] * q[b][d])
                .sum();
            close(r.virial[a][b], expected, 1e-11);
        }
    }
}

fn linear(weights: Vec<Vec<f64>>, bias: Vec<f64>) -> Network {
    Network {
        layers: vec![Layer {
            weights,
            bias,
            activation: Activation::Linear,
            residual: Residual::None,
            timestep: None,
        }],
    }
}

/// One type, four slots, the embedding and the fitting net both the
/// identity: the energy of an atom is its descriptor, a fifth of the mean of
/// the normalized radial entries of its slots.
fn identity_model() -> Model {
    Model {
        format: "omnimd-native-se-e2-r".into(),
        version: 1,
        precision: "float64".into(),
        units: Units::EvAngstrom,
        type_map: vec!["O".into()],
        descriptor: Descriptor {
            kind: "se_e2_r".into(),
            rcut: 3.0,
            rcut_smth: 2.9,
            sel: vec![4],
            axis_neuron: 0,
            normalization: vec![Normalization {
                mean: [0.2, 0.0, 0.0, 0.0],
                std: [0.5, 1.0, 1.0, 1.0],
            }],
            type_one_side: false,
            embedding: vec![vec![linear(vec![vec![1.0]], vec![0.0])]],
        },
        fitting: vec![linear(vec![vec![1.0]], vec![0.0])],
        atom_energy_bias: vec![0.0],
    }
}

#[test]
fn the_descriptor_is_a_fifth_of_the_mean_embedding_row() {
    let model = identity_model();
    let mut potential = NativePotential::new(model).unwrap();
    let mut f = Frame {
        positions: vec![[0.0; 3]],
        species: vec!["O".into()],
        cell: None,
        units: Units::EvAngstrom,
    };
    // Alone: four empty slots, each the normalized zero (0 - 0.2) / 0.5.
    close(potential.eval(&f).unwrap().energy, 0.2 * -0.4, 1e-15);
    // A neighbor at 1 Å, inside the flat part of the switch: its entry is
    // (1 - 0.2) / 0.5 = 1.6, the other three slots stay empty.
    f.positions.push([1.0, 0.0, 0.0]);
    f.species.push("O".into());
    let pair = potential.eval(&f).unwrap();
    close(pair.energy, 2.0 * 0.2 * (1.6 - 3.0 * 0.4) / 4.0, 1e-15);
    // E = 2 · 0.05 · (1/r - 0.2) / 0.5 + const: dE/dr = -0.2 at r = 1.
    close(pair.forces[1][0], 0.2, 1e-14);
    close(pair.forces[0][0], -0.2, 1e-14);
    for atom in 0..2 {
        for a in 1..3 {
            close(pair.forces[atom][a], 0.0, 1e-15);
        }
    }
    close(pair.virial[0][0], 0.2, 1e-14);
    assert!(pair
        .virial
        .iter()
        .flatten()
        .skip(1)
        .all(|w| w.abs() < 1e-15));
    // Three neighbors at 1, 2 and 2.5 Å along the axes fill three slots.
    let mut star = f.clone();
    star.positions.extend([[0.0, 2.0, 0.0], [0.0, 0.0, 2.5]]);
    star.species.extend(["O".into(), "O".into()]);
    let entry = |r: f64| (1.0 / r - 0.2) / 0.5;
    let pair_distance = |a: [f64; 3], b: [f64; 3]| -> f64 {
        (0..3).map(|c| (a[c] - b[c]).powi(2)).sum::<f64>().sqrt()
    };
    let mut expected = 0.0;
    for (i, a) in star.positions.iter().enumerate() {
        let mut sum = 0.0;
        let mut filled = 0;
        for (j, b) in star.positions.iter().enumerate() {
            let r = pair_distance(*a, *b);
            if i != j && r < 2.9 {
                sum += entry(r);
                filled += 1;
            }
        }
        assert!(filled > 0, "atom {i} has neighbors in the flat part");
        expected += 0.2 * (sum + (4 - filled) as f64 * -0.4) / 4.0;
    }
    // (The third and the fourth atom are 3.2 Å apart, beyond the cutoff:
    // each has two neighbors, the first two atoms three. No pair lies where
    // the switch bends.)
    assert!(star
        .positions
        .iter()
        .flat_map(|a| star.positions.iter().map(move |b| pair_distance(*a, *b)))
        .all(|r| r == 0.0 || !(2.9..3.0).contains(&r)));
    close(potential.eval(&star).unwrap().energy, expected, 1e-14);
}

#[test]
fn a_format_names_its_descriptor_and_other_models_are_refused() {
    assert!(NativePotential::new(identity_model()).is_ok());
    let refused = |name: &str, change: &dyn Fn(&mut Model), why: &str| {
        let mut model = identity_model();
        change(&mut model);
        let error = NativePotential::new(model).expect_err(name).to_string();
        assert!(error.contains(why), "{name}: {error}");
    };
    refused(
        "the format of the other descriptor",
        &|m| m.format = "omnimd-native-se-e2-a".into(),
        "unsupported descriptor \"se_e2_r\" for native format omnimd-native-se-e2-a",
    );
    refused(
        "the other descriptor in this format",
        &|m| m.descriptor.kind = "se_e2_a".into(),
        "unsupported descriptor \"se_e2_a\" for native format omnimd-native-se-e2-r",
    );
    refused(
        "a descriptor the format does not have",
        &|m| m.descriptor.kind = "se_e3".into(),
        "unsupported descriptor \"se_e3\"",
    );
    refused(
        "another format",
        &|m| m.format = "omnimd-native-se-e3".into(),
        "unsupported native format/version",
    );
    refused(
        "another version",
        &|m| m.version = 2,
        "unsupported native format/version",
    );
    refused(
        "axis neurons",
        &|m| m.descriptor.axis_neuron = 1,
        "se_e2_r has no axis neurons",
    );
    refused(
        "a width beside the radial one",
        &|m| m.descriptor.normalization[0].std = [0.5, 0.9, 0.9, 0.9],
        "se_e2_r has a radial entry only",
    );
    refused(
        "a mean beside the radial one",
        &|m| m.descriptor.normalization[0].mean[2] = 0.1,
        "angular mean must be zero",
    );
    refused(
        "a width of zero",
        &|m| m.descriptor.normalization[0].std[0] = 0.0,
        "positive std",
    );
    // The fitting net takes the embedding width, not width times an axis.
    refused(
        "a fitting net of another input",
        &|m| m.fitting[0] = linear(vec![vec![1.0], vec![1.0]], vec![0.0]),
        "fitting[0] layer 0: expected weights[1][1]",
    );
    // `axis_neuron` may be left out of an se_e2_r file, not of an se_e2_a one.
    let mut value = serde_json::to_value(identity_model()).unwrap();
    assert!(value["descriptor"]
        .as_object_mut()
        .unwrap()
        .remove("axis_neuron")
        .is_some());
    assert!(NativePotential::from_json(&value.to_string()).is_ok());
    let mut other: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/synthetic-v1.json")).unwrap();
    assert!(NativePotential::from_json(&other.to_string()).is_ok());
    other["descriptor"]
        .as_object_mut()
        .unwrap()
        .remove("axis_neuron");
    let error = NativePotential::from_json(&other.to_string())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("axis_neuron must be in 1..=embedding width"),
        "{error}"
    );
    // An se_e2_a model says what it is, as before.
    let se_a: Model = serde_json::from_str(include_str!("../fixtures/synthetic-v1.json")).unwrap();
    assert_eq!(se_a.descriptor_kind(), DescriptorKind::SeE2A);
    assert_eq!(
        NativePotential::new(se_a).unwrap().metadata().backend,
        "native-se-e2-a-v1"
    );
    assert_eq!(model().descriptor.features(), 12);
}

#[test]
fn the_trainers_of_the_other_descriptor_refuse_it() {
    let model = model();
    let error = ParameterLayout::new(&model).err().unwrap().to_string();
    assert_eq!(
        error,
        "the full-network trainer supports the se_e2_a descriptor only, not se_e2_r"
    );
    let error = ReadoutBasis::new(&model, false).err().unwrap().to_string();
    assert_eq!(
        error,
        "the readout trainer supports the se_e2_a descriptor only, not se_e2_r"
    );
}

/// `type_one_side` says that every central type has the same embedding nets:
/// a model that says so and has rows that differ is refused.
#[test]
fn the_one_side_mark_needs_rows_that_are_alike() {
    // The fixture was converted from a DeePMD-kit model with one embedding
    // net per neighbor type.
    let marked = model();
    assert!(marked.descriptor.type_one_side);
    assert_eq!(
        marked.descriptor.embedding[1],
        marked.descriptor.embedding[0]
    );
    assert!(NativePotential::new(marked.clone()).is_ok());
    let mut apart = marked.clone();
    apart.descriptor.embedding[1][0].layers[0].bias[0] += 1e-9;
    let error = NativePotential::new(apart.clone()).unwrap_err().to_string();
    assert!(
        error.contains("type_one_side: the embedding nets of a neighbor type must be the same"),
        "{error}"
    );
    // Without the mark the same nets are a model like any other.
    apart.descriptor.type_one_side = false;
    assert!(NativePotential::new(apart).is_ok());
    // The mark is written only where it is set, and read back.
    let text = serde_json::to_string(&marked).unwrap();
    assert!(text.contains("\"type_one_side\":true"));
    let back: Model = serde_json::from_str(&text).unwrap();
    assert!(back.descriptor.type_one_side);
    let plain = serde_json::to_string(&identity_model()).unwrap();
    assert!(!plain.contains("type_one_side"));
    // An se_e2_a model with nets of its own for every pair cannot carry it.
    let mut pairs: Model =
        serde_json::from_str(include_str!("../fixtures/synthetic-v1.json")).unwrap();
    assert!(!pairs.descriptor.type_one_side);
    pairs.descriptor.type_one_side = true;
    assert!(NativePotential::new(pairs).is_err());
}
