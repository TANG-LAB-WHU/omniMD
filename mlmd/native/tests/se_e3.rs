//! The se_e3 descriptor on the CPU: against DeePMD-kit itself, against
//! finite differences and its definition written out, and what the format
//! refuses.
use omnimd_ml_contract::{Evaluation, Frame, Potential, Units};
use omnimd_native::{
    full_gradient::ParameterLayout, readout::ReadoutBasis, three_body_weight, Activation,
    Descriptor, DescriptorKind, Layer, Model, NativePotential, Network, Normalization, Residual,
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
    serde_json::from_str(include_str!("../fixtures/se-e3-deepmd-reference.json")).unwrap()
}

/// The fixture model whose `sel` holds every neighbor of the fixture frames.
fn model() -> Model {
    serde_json::from_str(include_str!("../fixtures/se-e3-model.native.json")).unwrap()
}

fn tight() -> Model {
    serde_json::from_str(include_str!("../fixtures/se-e3-tight-model.native.json")).unwrap()
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
    assert_eq!(model.descriptor_kind(), Some(DescriptorKind::SeE3));
    assert_eq!(model.type_map.len(), 3);
    // DeePMD-kit's model has a radial mean and width that its descriptor
    // never reads; the native model has neither.
    for entry in &model.descriptor.normalization {
        assert_eq!((entry.mean, entry.std[0]), ([0.0; 4], 1.0));
        assert!(entry.std[1] != 1.0);
    }
    let mut potential = NativePotential::new(model).unwrap();
    assert_eq!(potential.metadata().backend, "native-se-e3-v1");
    let mut compared = 0;
    let mut worst = [0.0f64; 3];
    for case in &reference.cases {
        if case.model != "se-e3-model.native.json" {
            continue;
        }
        let ours = potential.eval(&case.frame).unwrap();
        worst[0] = worst[0].max((ours.energy - case.energy).abs());
        assert!(
            (ours.energy - case.energy).abs() < 1e-11,
            "energy {} against {}",
            ours.energy,
            case.energy
        );
        for (atom, (a, b)) in ours.forces.iter().zip(&case.forces).enumerate() {
            for axis in 0..3 {
                worst[1] = worst[1].max((a[axis] - b[axis]).abs());
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
            worst[2] = worst[2].max((a - b).abs());
            assert!((a - b).abs() < 1e-12, "virial {a} against {b}");
        }
        compared += 1;
    }
    eprintln!(
        "se_e3 on the CPU against DeePMD-kit: energy within {:.1e} eV, forces {:.1e} eV/A, virial {:.1e} eV",
        worst[0], worst[1], worst[2]
    );
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
    assert!(largest > 1e-2, "{largest}");
    // More neighbors than `sel`: the CPU reference refuses, as for se_e2_a.
    let case = reference
        .cases
        .iter()
        .find(|case| case.model == "se-e3-tight-model.native.json")
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

/// The slope of the net of a pair of neighbor types in [`plain_model`].
const SLOPE: [[f64; 2]; 2] = [[1.0, 0.7], [0.7, -0.4]];
const SEL: [usize; 2] = [3, 4];
/// Width of the direction entries per central type.
const WIDTH: [f64; 2] = [0.5, 0.8];

/// Two types; the net of the neighbor types (t, u) is θ -> SLOPE[t][u]·θ + 1
/// and the fitting nets are the identity, so the energy of an atom is its
/// descriptor: the sum over the pairs of θ·(SLOPE·θ + 1), weighted.
fn plain_model() -> Model {
    let net = |t: usize, u: usize| linear(vec![vec![SLOPE[t][u]]], vec![1.0]);
    Model {
        format: "omnimd-native-se-e3".into(),
        version: 1,
        precision: "float64".into(),
        units: Units::EvAngstrom,
        type_map: vec!["O".into(), "H".into()],
        descriptor: Descriptor {
            kind: "se_e3".into(),
            rcut: 3.0,
            rcut_smth: 2.9,
            sel: SEL.to_vec(),
            axis_neuron: 0,
            normalization: WIDTH
                .iter()
                .map(|&width| Normalization {
                    mean: [0.0; 4],
                    std: [1.0, width, width, width],
                })
                .collect(),
            type_one_side: false,
            embedding: (0..2)
                .map(|t| (0..2).map(|u| net(t, u)).collect())
                .collect(),
            list: Vec::new(),
        },
        fitting: vec![
            linear(vec![vec![1.0]], vec![0.0]),
            linear(vec![vec![1.0]], vec![0.0]),
        ],
        atom_energy_bias: vec![0.0, 0.0],
    }
}

/// The energy of `frame` under [`plain_model`] from DeePMD-kit's definition
/// written out: for every atom and every pair of neighbor types t <= u, the
/// sum over ALL slots j of type t and k of type u (both orders within one
/// type, a slot with itself included) of θ·N(θ) / (sel[t]·sel[u]). All
/// distances must lie in the flat part of the switch, where s(r) = 1 / r.
fn written_out(frame: &Frame) -> f64 {
    let kind = |i: usize| usize::from(frame.species[i] == "H");
    let mut energy = 0.0;
    for (i, center) in frame.positions.iter().enumerate() {
        // The direction entries of the neighbors of each type.
        let mut rows: [Vec<[f64; 3]>; 2] = [Vec::new(), Vec::new()];
        for (j, other) in frame.positions.iter().enumerate() {
            let d: [f64; 3] = std::array::from_fn(|a| other[a] - center[a]);
            let r2: f64 = d.iter().map(|x| x * x).sum();
            if i == j || r2.sqrt() >= 3.0 {
                continue;
            }
            assert!(r2.sqrt() < 2.9, "a pair where the switch bends");
            // s(r)·d/r = d / r², over the width of the central type.
            rows[kind(j)].push(d.map(|x| x / r2 / WIDTH[kind(i)]));
        }
        for t in 0..2 {
            assert!(
                rows[t].len() <= SEL[t],
                "more neighbors than slots of type {t}"
            );
            for u in t..2 {
                for a in &rows[t] {
                    for b in &rows[u] {
                        let theta: f64 = (0..3).map(|c| a[c] * b[c]).sum();
                        energy += theta * (SLOPE[t][u] * theta + 1.0) / (SEL[t] * SEL[u]) as f64;
                    }
                }
            }
        }
    }
    energy
}

#[test]
fn the_descriptor_is_the_sum_over_pairs_of_neighbors_as_deepmd_kit_defines_it() {
    let mut potential = NativePotential::new(plain_model()).unwrap();
    let mut f = Frame {
        positions: vec![[0.0; 3]],
        species: vec!["O".into()],
        cell: None,
        units: Units::EvAngstrom,
    };
    // Alone: the empty slots have no direction and add nothing.
    assert_eq!(potential.eval(&f).unwrap().energy, 0.0);
    // One neighbor at 1 Å: each atom has the pair of that neighbor with
    // itself, θ = (1 / 0.5)² = 4, once over 3·3 slots.
    f.positions.push([1.0, 0.0, 0.0]);
    f.species.push("O".into());
    let pair = potential.eval(&f).unwrap();
    close(pair.energy, 2.0 * 4.0 * (4.0 + 1.0) / 9.0, 1e-15);
    close(pair.energy, written_out(&f), 1e-15);
    // E = 2·(θ² + θ) / 9 with θ = 4 / r²: dE/dr = 2·(2θ + 1)·(-2θ / r) / 9.
    let slope = 2.0 * (2.0 * 4.0 + 1.0) * (-2.0 * 4.0) / 9.0;
    close(pair.forces[1][0], -slope, 1e-13);
    close(pair.forces[0][0], slope, 1e-13);
    close(pair.virial[0][0], -slope, 1e-13);
    assert!(pair
        .virial
        .iter()
        .flatten()
        .skip(1)
        .all(|w| w.abs() < 1e-14));
    // Both types around every atom, at angles that are not right ones: the
    // pairs of one type in both orders, the pairs of two types once.
    let cluster = Frame {
        positions: vec![
            [0.0, 0.0, 0.0],
            [1.1, 0.2, -0.1],
            [-0.3, 1.2, 0.4],
            [0.5, -0.6, 1.3],
            [-0.9, -0.7, -0.5],
            [0.8, 1.0, 0.9],
        ],
        species: ["O", "H", "O", "H", "H", "O"].map(String::from).to_vec(),
        cell: None,
        units: Units::EvAngstrom,
    };
    let expected = written_out(&cluster);
    assert!(expected.abs() > 1.0, "{expected}");
    close(potential.eval(&cluster).unwrap().energy, expected, 1e-13);
    // What a pair counts, as the sum above has it.
    assert_eq!(three_body_weight(true, true, 3, 3), 1.0 / 9.0);
    assert_eq!(three_body_weight(false, true, 3, 3), 2.0 / 9.0);
    assert_eq!(three_body_weight(false, false, 3, 4), 1.0 / 12.0);
}

#[test]
fn a_format_names_its_descriptor_and_other_models_are_refused() {
    assert!(NativePotential::new(plain_model()).is_ok());
    let refused = |name: &str, change: &dyn Fn(&mut Model), why: &str| {
        let mut model = plain_model();
        change(&mut model);
        let error = NativePotential::new(model).expect_err(name).to_string();
        assert!(error.contains(why), "{name}: {error}");
    };
    refused(
        "the format of another descriptor",
        &|m| m.format = "omnimd-native-se-e2-r".into(),
        "unsupported descriptor \"se_e3\" for native format omnimd-native-se-e2-r",
    );
    refused(
        "another descriptor in this format",
        &|m| m.descriptor.kind = "se_e2_a".into(),
        "unsupported descriptor \"se_e2_a\" for native format omnimd-native-se-e3",
    );
    refused(
        "a name DeePMD-kit also has for it",
        &|m| m.descriptor.kind = "se_at".into(),
        "unsupported descriptor \"se_at\"",
    );
    refused(
        "another version",
        &|m| m.version = 2,
        "unsupported native format/version",
    );
    refused(
        "axis neurons",
        &|m| m.descriptor.axis_neuron = 1,
        "se_e3 has no axis neurons",
    );
    refused(
        "a radial mean",
        &|m| m.descriptor.normalization[1].mean[0] = 0.1,
        "se_e3 has direction entries only",
    );
    refused(
        "a radial width",
        &|m| m.descriptor.normalization[0].std[0] = 0.3,
        "se_e3 has direction entries only",
    );
    refused(
        "widths that differ between the directions",
        &|m| m.descriptor.normalization[0].std[3] = 0.6,
        "angular std isotropic",
    );
    refused(
        "a net that depends on the order of its pair",
        &|m| m.descriptor.embedding[1][0] = linear(vec![vec![0.69]], vec![1.0]),
        "must be the same in either order",
    );
    refused(
        "nets marked as shared by the central types",
        &|m| {
            m.descriptor.embedding[1] = m.descriptor.embedding[0].clone();
            m.descriptor.embedding[0][1] = m.descriptor.embedding[0][0].clone();
            m.descriptor.embedding[1][0] = m.descriptor.embedding[0][0].clone();
            m.descriptor.embedding[1][1] = m.descriptor.embedding[0][0].clone();
            m.descriptor.type_one_side = true;
        },
        "type_one_side does not apply",
    );
    // The fitting net takes the embedding width.
    refused(
        "a fitting net of another input",
        &|m| m.fitting[0] = linear(vec![vec![1.0], vec![1.0]], vec![0.0]),
        "fitting[0] layer 0: expected weights[1][1]",
    );
    // `axis_neuron` may be left out of an se_e3 file.
    let mut value = serde_json::to_value(plain_model()).unwrap();
    assert!(value["descriptor"]
        .as_object_mut()
        .unwrap()
        .remove("axis_neuron")
        .is_some());
    assert!(NativePotential::from_json(&value.to_string()).is_ok());
    let fixture = model();
    assert_eq!(fixture.descriptor.features(), 12);
    // The converter wrote the net of every pair of neighbor types in both
    // orders, and the nets of different pairs differ.
    let nets = &fixture.descriptor.embedding;
    for t in 0..3 {
        for u in 0..t {
            assert_eq!(nets[t][u], nets[u][t]);
        }
    }
    assert_ne!(nets[0][1], nets[0][2]);
    assert_ne!(nets[0][0], nets[1][1]);
    assert_eq!(
        NativePotential::new(fixture).unwrap().metadata().backend,
        "native-se-e3-v1"
    );
}

#[test]
fn the_trainers_of_the_other_descriptors_refuse_it() {
    let model = model();
    let error = ParameterLayout::new(&model).err().unwrap().to_string();
    assert_eq!(
        error,
        "the full-network trainer supports the se_e2_a descriptor only, not se_e3"
    );
    let error = ReadoutBasis::new(&model, false).err().unwrap().to_string();
    assert_eq!(
        error,
        "the readout trainer supports the se_e2_a descriptor only, not se_e3"
    );
}
