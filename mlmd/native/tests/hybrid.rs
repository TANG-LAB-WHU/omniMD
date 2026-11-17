//! The hybrid descriptor on the CPU: against DeePMD-kit itself, against
//! finite differences, against its descriptors one by one, and what the
//! format refuses.
use omnimd_ml_contract::{Evaluation, Frame, Potential};
use omnimd_native::{
    full_gradient::ParameterLayout, readout::ReadoutBasis, Activation, Descriptor, DescriptorKind,
    Layer, Model, NativePotential, Network, Residual, HYBRID_FORMAT, HYBRID_KIND, MAX_HYBRID_PARTS,
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
    serde_json::from_str(include_str!("../fixtures/hybrid-deepmd-reference.json")).unwrap()
}

/// The fixture model: an se_e2_a, an se_e2_r and an se_e3 descriptor, each
/// with room for every neighbor within its own cutoff.
fn model() -> Model {
    serde_json::from_str(include_str!("../fixtures/hybrid-model.native.json")).unwrap()
}

fn tight() -> Model {
    serde_json::from_str(include_str!("../fixtures/hybrid-tight-model.native.json")).unwrap()
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

/// Deterministic values in (-1, 1).
fn values(count: usize, seed: u64) -> Vec<f64> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        })
        .collect()
}

/// A fitting net of one linear layer: the energy of an atom is the sum of
/// `weights` times its descriptor.
fn linear(weights: &[f64]) -> Network {
    Network {
        layers: vec![Layer {
            weights: weights.iter().map(|&w| vec![w]).collect(),
            bias: vec![0.0],
            activation: Activation::Linear,
            residual: Residual::None,
            timestep: None,
        }],
    }
}

/// The hybrid of `parts` with the neighbor list they need together.
fn hybrid_of(parts: Vec<Descriptor>) -> Descriptor {
    let ntypes = parts[0].sel.len();
    Descriptor {
        kind: HYBRID_KIND.into(),
        rcut: parts.iter().map(|part| part.rcut).fold(0.0, f64::max),
        rcut_smth: parts
            .iter()
            .map(|part| part.rcut_smth)
            .fold(f64::INFINITY, f64::min),
        sel: (0..ntypes)
            .map(|t| parts.iter().map(|part| part.sel[t]).max().unwrap())
            .collect(),
        axis_neuron: 0,
        normalization: Vec::new(),
        type_one_side: false,
        embedding: Vec::new(),
        list: parts,
    }
}

/// The fixture model with another descriptor and fitting nets of one linear
/// layer: `weights[type]` for all components of the descriptor.
fn with_linear_fitting(descriptor: Descriptor, weights: &[Vec<f64>]) -> Model {
    let mut model = model();
    model.format = match descriptor.descriptor_kind() {
        Some(kind) => kind.format().into(),
        None => HYBRID_FORMAT.into(),
    };
    model.descriptor = descriptor;
    model.fitting = weights.iter().map(|w| linear(w)).collect();
    model
}

#[test]
fn energy_forces_and_virial_match_deepmd_kit() {
    let reference = reference();
    assert_eq!(reference.deepmd_version, "3.1.3");
    let model = model();
    // A hybrid has no descriptor of its own: three of them, in this order.
    assert_eq!(model.descriptor_kind(), None);
    assert!(model.descriptor.is_hybrid());
    let kinds: Vec<_> = model
        .descriptor
        .parts()
        .iter()
        .map(|part| part.descriptor_kind().unwrap())
        .collect();
    assert_eq!(
        kinds,
        [
            DescriptorKind::SeE2A,
            DescriptorKind::SeE2R,
            DescriptorKind::SeE3
        ]
    );
    // Each reads the list in its own way: no two have the same cutoff, and
    // none has the slots of the list for every type.
    let d = &model.descriptor;
    assert_eq!((d.rcut, d.rcut_smth), (4.4, 0.4));
    for (a, part) in d.list.iter().enumerate() {
        assert_ne!(part.sel, d.sel);
        for other in &d.list[a + 1..] {
            assert_ne!(part.rcut, other.rcut);
            assert_ne!(part.width(), other.width());
        }
    }
    assert_eq!(d.features(), 8 * 3 + 10 + 6);
    let mut potential = NativePotential::new(model).unwrap();
    assert_eq!(potential.metadata().backend, "native-hybrid-v1");
    assert_eq!(potential.metadata().cutoff, 4.4);
    let mut compared = 0;
    let mut worst = [0.0f64; 3];
    for case in &reference.cases {
        if case.model != "hybrid-model.native.json" {
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
        "hybrid on the CPU against DeePMD-kit: energy within {:.1e} eV, forces {:.1e} eV/A, virial {:.1e} eV",
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
    // More neighbors than slots, in the list and in every descriptor: the
    // CPU reference refuses, as for a model of one descriptor.
    let case = reference
        .cases
        .iter()
        .find(|case| case.model == "hybrid-tight-model.native.json")
        .unwrap();
    let error = NativePotential::new(tight())
        .unwrap()
        .eval(&case.frame)
        .unwrap_err()
        .to_string();
    assert!(error.contains("neighbor capacity overflow"), "{error}");
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

/// Sum of the evaluations of `models` for `frame`.
fn summed(models: &[Model], frame: &Frame) -> Evaluation {
    let mut total = eval(&models[0], frame);
    for model in &models[1..] {
        let more = eval(model, frame);
        total.energy += more.energy;
        for (a, b) in total.forces.iter_mut().zip(&more.forces) {
            for c in 0..3 {
                a[c] += b[c];
            }
        }
        for (a, b) in total.virial.iter_mut().zip(&more.virial) {
            for c in 0..3 {
                a[c] += b[c];
            }
        }
    }
    total
}

fn alike(a: &Evaluation, b: &Evaluation, tolerance: f64) {
    close(a.energy, b.energy, tolerance);
    for (x, y) in a.forces.iter().flatten().zip(b.forces.iter().flatten()) {
        assert!((x - y).abs() < tolerance, "force {x} against {y}");
    }
    for (x, y) in a.virial.iter().flatten().zip(b.virial.iter().flatten()) {
        assert!((x - y).abs() < 10.0 * tolerance, "virial {x} against {y}");
    }
}

/// With fitting nets of one linear layer the energy of a hybrid is the sum
/// of the energies of models that have one of its descriptors each, with
/// that descriptor's share of the weights: each of them alone searches its
/// neighbors with its own cutoff and slots, the hybrid reads them from the
/// list of all. The bias of an atom is counted once for each model.
fn against_its_descriptors(parts: Vec<Descriptor>, seed: u64) {
    let reference = reference();
    let features: Vec<usize> = parts.iter().map(Descriptor::features).collect();
    let total: usize = features.iter().sum();
    let weights: Vec<Vec<f64>> = (0..3).map(|t| values(total, seed + t as u64)).collect();
    let mut hybrid = with_linear_fitting(hybrid_of(parts.clone()), &weights);
    let mut first = 0;
    let mut alone = Vec::new();
    for (part, &count) in parts.iter().zip(&features) {
        let share: Vec<Vec<f64>> = weights
            .iter()
            .map(|w| w[first..first + count].to_vec())
            .collect();
        alone.push(with_linear_fitting(part.clone(), &share));
        first += count;
    }
    // The models of one descriptor count the bias of an atom once each.
    for bias in &mut hybrid.atom_energy_bias {
        *bias *= parts.len() as f64;
    }
    for case in &reference.cases[..2] {
        let together = eval(&hybrid, &case.frame);
        let apart = summed(&alone, &case.frame);
        assert!(together.energy.abs() > 1.0);
        alike(&together, &apart, 1e-12);
    }
}

#[test]
fn a_hybrid_with_linear_fitting_nets_is_the_sum_of_its_descriptors() {
    let parts = model().descriptor.list;
    // The three of the fixture, in its order and in another.
    against_its_descriptors(parts.clone(), 11);
    against_its_descriptors(
        vec![parts[2].clone(), parts[0].clone(), parts[1].clone()],
        12,
    );
    // Two descriptors of one kind that differ in their cutoffs and slots:
    // the second se_e2_a reaches as far as the list and has slots to spare.
    let mut wider = parts[0].clone();
    wider.rcut = 4.4;
    wider.rcut_smth = 1.3;
    wider.sel = vec![18, 19, 14];
    against_its_descriptors(vec![parts[0].clone(), wider.clone()], 13);
    // The same descriptor twice, and all kinds twice.
    against_its_descriptors(vec![parts[1].clone(), parts[1].clone()], 14);
    let mut closer = parts[2].clone();
    closer.rcut = 2.6;
    closer.rcut_smth = 1.9;
    let mut radial = parts[1].clone();
    radial.rcut = 3.1;
    radial.sel = vec![8, 10, 6];
    against_its_descriptors(
        vec![
            closer,
            parts[1].clone(),
            wider,
            parts[2].clone(),
            radial,
            parts[0].clone(),
        ],
        15,
    );
}

#[test]
fn one_descriptor_in_a_list_is_that_descriptor() {
    let reference = reference();
    for (index, part) in model().descriptor.list.into_iter().enumerate() {
        let weights: Vec<Vec<f64>> = (0..3).map(|t| values(part.features(), 20 + t)).collect();
        let plain = with_linear_fitting(part.clone(), &weights);
        let listed = with_linear_fitting(hybrid_of(vec![part]), &weights);
        assert_eq!(listed.format, HYBRID_FORMAT);
        for case in &reference.cases[..2] {
            let (a, b) = (eval(&plain, &case.frame), eval(&listed, &case.frame));
            // The same arithmetic in the same order.
            assert_eq!(a.energy, b.energy, "descriptor {index}");
            assert_eq!(a.forces, b.forces, "descriptor {index}");
            assert_eq!(a.virial, b.virial, "descriptor {index}");
        }
    }
}

#[test]
fn the_order_of_the_list_is_the_order_of_the_components() {
    let reference = reference();
    let frame = &reference.cases[0].frame;
    let model = model();
    let e = eval(&model, frame);
    // The first two descriptors the other way round, the fitting nets as
    // they were: other components meet the weights.
    let mut swapped = model.clone();
    swapped.descriptor.list.swap(0, 1);
    let other = eval(&swapped, frame);
    assert!((other.energy - e.energy).abs() > 1e-3, "{}", other.energy);
    // With the rows of the first layer moved along (the weights of a layer
    // are stored by input), it is the same model.
    let (a, b) = (
        model.descriptor.list[0].features(),
        model.descriptor.list[1].features(),
    );
    for net in &mut swapped.fitting {
        let rows = &mut net.layers[0].weights;
        let mut moved = rows[a..a + b].to_vec();
        moved.extend_from_slice(&rows[..a]);
        moved.extend_from_slice(&rows[a + b..]);
        *rows = moved;
    }
    let same = eval(&swapped, frame);
    close(same.energy, e.energy, 1e-13);
    for (x, y) in same.forces.iter().flatten().zip(e.forces.iter().flatten()) {
        assert!((x - y).abs() < 1e-13, "{x} against {y}");
    }
}

#[test]
fn a_descriptor_reads_the_nearest_of_the_list_within_its_own_cutoff() {
    let reference = reference();
    let frame = &reference.cases[0].frame;
    let parts = model().descriptor.list;
    let weights: Vec<Vec<f64>> = (0..3)
        .map(|t| values(parts[0].features() + parts[1].features(), 30 + t))
        .collect();
    // The se_e2_a descriptor of the fixture has neighbors of the last type
    // in its slots that lie beyond its cutoff (the list reaches further):
    // they are empty slots to it, so its cutoff matters and the cutoff of
    // the list does not.
    let pair = |cutoff: f64| {
        let mut second = parts[1].clone();
        second.rcut = cutoff;
        // Room for every neighbor within the larger cutoff.
        second.sel = vec![30, 40, 20];
        with_linear_fitting(hybrid_of(vec![parts[0].clone(), second]), &weights)
    };
    let only_first = |model: &Model| {
        // The weights of the second descriptor set to zero.
        let mut model = model.clone();
        let count = parts[0].features();
        for net in &mut model.fitting {
            for row in &mut net.layers[0].weights[count..] {
                row[0] = 0.0;
            }
        }
        eval(&model, frame).energy
    };
    let (near, far) = (pair(4.4), pair(5.0));
    assert_eq!(far.descriptor.rcut, 5.0);
    close(only_first(&near), only_first(&far), 1e-13);
    // A descriptor with fewer slots than neighbors within its cutoff is
    // refused, whatever the list holds.
    let mut short = parts[0].clone();
    short.sel[1] = 3;
    let model = with_linear_fitting(hybrid_of(vec![short, parts[1].clone()]), &weights);
    assert_eq!(model.descriptor.sel[1], parts[1].sel[1]);
    assert!(model.descriptor.sel[1] > 10);
    let error = NativePotential::new(model)
        .unwrap()
        .eval(frame)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("type H: more than sel=3 within 3.4 of a descriptor of the hybrid"),
        "{error}"
    );
}

#[test]
fn a_hybrid_names_its_descriptors_and_other_models_are_refused() {
    assert!(NativePotential::new(model()).is_ok());
    let refused = |name: &str, change: &dyn Fn(&mut Model), why: &str| {
        let mut model = model();
        change(&mut model);
        let error = NativePotential::new(model).expect_err(name).to_string();
        assert!(error.contains(why), "{name}: {error}");
    };
    refused(
        "the format of one descriptor",
        &|m| m.format = "omnimd-native-se-e2-a".into(),
        "unsupported descriptor \"hybrid\" for native format omnimd-native-se-e2-a",
    );
    refused(
        "one descriptor in the format of a hybrid",
        &|m| m.descriptor = m.descriptor.list[0].clone(),
        "unsupported descriptor \"se_e2_a\" for native format omnimd-native-hybrid",
    );
    refused(
        "another version",
        &|m| m.version = 2,
        "unsupported native format/version",
    );
    refused(
        "no descriptors",
        &|m| m.descriptor.list.clear(),
        "a hybrid descriptor has 1..=16 descriptors in its list",
    );
    refused(
        "too many descriptors",
        &|m| {
            let part = m.descriptor.list[1].clone();
            m.descriptor.list = vec![part; MAX_HYBRID_PARTS + 1];
        },
        "a hybrid descriptor has 1..=16 descriptors in its list",
    );
    refused(
        "a hybrid in a hybrid",
        &|m| {
            let inner = m.descriptor.clone();
            m.descriptor.list.push(inner);
        },
        "descriptor.list[3]: unsupported descriptor \"hybrid\": a hybrid is made of se_e2_a, se_e2_r and se_e3 descriptors",
    );
    refused(
        "a descriptor with a list of its own",
        &|m| {
            let inner = m.descriptor.list[2].clone();
            m.descriptor.list[1].list.push(inner);
        },
        "descriptor.list[1]: only a hybrid descriptor has a list of descriptors",
    );
    refused(
        "a cutoff that is not the largest of the list",
        &|m| m.descriptor.rcut = 4.5,
        "a hybrid descriptor has the largest rcut (4.4)",
    );
    refused(
        "a smoothing radius that is not the smallest of the list",
        &|m| m.descriptor.rcut_smth = 0.6,
        "the smallest rcut_smth (0.4)",
    );
    refused(
        "slots that are not the most of the list",
        &|m| m.descriptor.sel[0] += 1,
        &format!(
            "of every type the largest sel ({:?})",
            model().descriptor.sel
        ),
    );
    refused(
        "axis neurons of its own",
        &|m| m.descriptor.axis_neuron = 3,
        "a hybrid descriptor has no axis neurons, normalization or embedding nets of its own",
    );
    refused(
        "nets of its own",
        &|m| m.descriptor.embedding = m.descriptor.list[0].embedding.clone(),
        "a hybrid descriptor has no axis neurons, normalization or embedding nets of its own",
    );
    refused(
        "a normalization of its own",
        &|m| m.descriptor.normalization = m.descriptor.list[0].normalization.clone(),
        "a hybrid descriptor has no axis neurons, normalization or embedding nets of its own",
    );
    // What a descriptor refuses alone it refuses in a list, and says where.
    refused(
        "an se_e3 descriptor with a radial mean",
        &|m| m.descriptor.list[2].normalization[1].mean[0] = 0.1,
        "descriptor.list[2]: normalization[1]: se_e3 has direction entries only",
    );
    refused(
        "an se_e2_r descriptor with axis neurons",
        &|m| m.descriptor.list[1].axis_neuron = 2,
        "descriptor.list[1]: se_e2_r has no axis neurons",
    );
    refused(
        "an se_e2_a descriptor without a net",
        &|m| {
            m.descriptor.list[0].embedding[2].pop();
        },
        "descriptor.list[0]: embedding[2] must have one network per neighbor type",
    );
    refused(
        "a descriptor with slots for two types",
        &|m| {
            m.descriptor.list[1].sel.pop();
        },
        "descriptor.list[1]: sel must have one entry per ordered type",
    );
    // The fitting nets take the components of all descriptors.
    refused(
        "a descriptor less",
        &|m| {
            m.descriptor.list.pop();
            let d = hybrid_of(std::mem::take(&mut m.descriptor.list));
            m.descriptor = d;
        },
        "fitting[0] layer 0: expected weights[34][8]",
    );
    // A model of one descriptor has no list.
    let mut plain =
        with_linear_fitting(model().descriptor.list[0].clone(), &vec![vec![0.5; 24]; 3]);
    assert!(NativePotential::new(plain.clone()).is_ok());
    plain
        .descriptor
        .list
        .push(model().descriptor.list[1].clone());
    let error = NativePotential::new(plain).unwrap_err().to_string();
    assert!(
        error.contains("only a hybrid descriptor has a list of descriptors"),
        "{error}"
    );
    // A hybrid is written without the entries it does not have, and read
    // back as it was.
    let value = serde_json::to_value(model()).unwrap();
    let outer = value["descriptor"].as_object().unwrap();
    for absent in ["normalization", "embedding", "type_one_side"] {
        assert!(!outer.contains_key(absent), "{absent}");
    }
    assert_eq!(outer["list"].as_array().unwrap().len(), 3);
    assert!(!value["descriptor"]["list"][0]
        .as_object()
        .unwrap()
        .contains_key("list"));
    let back: Model = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(serde_json::to_value(back).unwrap(), value);
    // The converter wrote one embedding net per neighbor type for the
    // se_e2_r descriptor (DeePMD-kit's `type_one_side`) and says so.
    let fixture = model();
    assert!(!fixture.descriptor.list[0].type_one_side);
    assert!(fixture.descriptor.list[1].type_one_side);
}

#[test]
fn the_trainers_of_one_descriptor_refuse_it() {
    let model = model();
    let error = ParameterLayout::new(&model).err().unwrap().to_string();
    assert_eq!(
        error,
        "the full-network trainer supports the se_e2_a descriptor only, not hybrid"
    );
    let error = ReadoutBasis::new(&model, false).err().unwrap().to_string();
    assert_eq!(
        error,
        "the readout trainer supports the se_e2_a descriptor only, not hybrid"
    );
}
