use omnimd_ml_contract::{Evaluation, Frame, Potential, Units};
use omnimd_native::{Activation, Layer, Model, NativePotential, Network, Normalization, Residual};

fn model() -> Model {
    serde_json::from_str(include_str!("../fixtures/synthetic-v1.json")).unwrap()
}
fn frame() -> Frame {
    serde_json::from_str(include_str!("../fixtures/frame.jsonl")).unwrap()
}
fn eval(f: &Frame) -> Evaluation {
    NativePotential::new(model()).unwrap().eval(f).unwrap()
}
fn close(a: f64, b: f64, tolerance: f64) {
    assert!(
        (a - b).abs() <= tolerance * (1.0 + a.abs().max(b.abs())),
        "{a:.16e} != {b:.16e} (tol {tolerance:e})"
    );
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
fn derivatives(f: &Frame) {
    let result = eval(f);
    let h = 2e-5;
    for i in 0..f.positions.len() {
        for a in 0..3 {
            let mut plus = f.clone();
            let mut minus = f.clone();
            plus.positions[i][a] += h;
            minus.positions[i][a] -= h;
            close(
                result.forces[i][a],
                -(eval(&plus).energy - eval(&minus).energy) / (2.0 * h),
                2e-8,
            );
        }
    }
    for a in 0..3 {
        for b in 0..3 {
            close(
                result.virial[a][b],
                -(eval(&deform(f, a, b, h)).energy - eval(&deform(f, a, b, -h)).energy) / (2.0 * h),
                3e-8,
            );
        }
    }
}

#[test]
fn analytic_forces_and_all_nine_strain_derivatives() {
    derivatives(&frame());
    let mut cluster = frame();
    cluster.cell = None;
    derivatives(&cluster);
    let small = Frame {
        positions: vec![[0.1, 0.2, 0.3]],
        species: vec!["O".into()],
        cell: Some([[2.0, 0.0, 0.0], [0.4, 2.2, 0.0], [0.3, 0.5, 2.4]]),
        units: Units::EvAngstrom,
    };
    derivatives(&small);
    let e = eval(&small);
    for force in e.forces[0] {
        close(force, 0.0, 1e-15);
    }
    assert!(e.virial.iter().flatten().any(|x| x.abs() > 1e-6));
}

#[test]
fn translation_permutation_rotation_and_periodic_wrapping() {
    let f = frame();
    let e = eval(&f);
    let mut translated = f.clone();
    for p in &mut translated.positions {
        for a in 0..3 {
            p[a] += [2.4, -3.1, 1.8][a];
        }
    }
    let t = eval(&translated);
    close(t.energy, e.energy, 1e-13);
    for (a, b) in t.forces.iter().flatten().zip(e.forces.iter().flatten()) {
        close(*a, *b, 1e-12);
    }
    let order = [3, 1, 0, 2];
    let mut permuted = f.clone();
    permuted.positions = order.iter().map(|i| f.positions[*i]).collect();
    permuted.species = order.iter().map(|i| f.species[*i].clone()).collect();
    let p = eval(&permuted);
    close(p.energy, e.energy, 1e-13);
    for i in 0..4 {
        for a in 0..3 {
            close(p.forces[i][a], e.forces[order[i]][a], 1e-12);
        }
    }
    let cell = f.cell.unwrap();
    let mut wrapped = f.clone();
    for a in 0..3 {
        wrapped.positions[1][a] += 2.0 * cell[0][a] - 3.0 * cell[1][a] + cell[2][a];
    }
    let w = eval(&wrapped);
    close(w.energy, e.energy, 1e-13);
    for (a, b) in w.forces.iter().flatten().zip(e.forces.iter().flatten()) {
        close(*a, *b, 1e-12);
    }
    // Proper rotation, neither an axis permutation nor a planar-only test.
    let q = [[0.36, -0.48, 0.8], [0.8, 0.6, 0.0], [-0.48, 0.64, 0.6]];
    let rotate =
        |v: [f64; 3]| -> [f64; 3] { std::array::from_fn(|a| (0..3).map(|b| q[a][b] * v[b]).sum()) };
    let mut rotated = f.clone();
    rotated.positions = f.positions.iter().map(|v| rotate(*v)).collect();
    rotated.cell = Some(cell.map(rotate));
    let r = eval(&rotated);
    close(r.energy, e.energy, 1e-13);
    for i in 0..4 {
        let expected = rotate(e.forces[i]);
        for a in 0..3 {
            close(r.forces[i][a], expected[a], 1e-12);
        }
    }
    for a in 0..3 {
        for b in 0..3 {
            let mut expected = 0.0;
            for c in 0..3 {
                for d in 0..3 {
                    expected += q[a][c] * e.virial[c][d] * q[b][d];
                }
            }
            close(r.virial[a][b], expected, 1e-12);
        }
    }
    for a in 0..3 {
        close(e.forces.iter().map(|v| v[a]).sum(), 0.0, 1e-14);
    }
}

#[test]
fn cutoff_and_smoothing_boundaries() {
    let dimer = |r: f64| Frame {
        positions: vec![[0.0; 3], [r, 0.0, 0.0]],
        species: vec!["O".into(), "H".into()],
        cell: None,
        units: Units::EvAngstrom,
    };
    for boundary in [2.1, 3.0] {
        let left = eval(&dimer(boundary - 1e-7));
        let right = eval(&dimer(boundary + 1e-7));
        close(left.energy, right.energy, 1e-8);
        close(left.forces[0][0], right.forces[0][0], 1e-8);
        derivatives(&dimer(boundary));
    }
    let outside = eval(&dimer(3.1));
    for f in outside.forces.iter().flatten() {
        assert_eq!(*f, 0.0);
    }
}

#[test]
fn normalized_padding_and_descriptor_scale_have_closed_form_energy() {
    let mut m = model();
    m.type_map = vec!["O".into()];
    m.atom_energy_bias = vec![0.0];
    let linear = |weight| Network {
        layers: vec![Layer {
            weights: vec![vec![weight]],
            bias: vec![0.0],
            activation: Activation::Linear,
            residual: Residual::None,
            timestep: None,
        }],
    };
    m.descriptor.sel = vec![4];
    m.descriptor.axis_neuron = 1;
    m.descriptor.normalization = vec![Normalization {
        mean: [0.2, 0.0, 0.0, 0.0],
        std: [1.0; 4],
    }];
    m.descriptor.embedding = vec![vec![linear(1.0)]];
    m.fitting = vec![linear(1.0)];
    let mut p = NativePotential::new(m).unwrap();
    let mut f = Frame {
        positions: vec![[0.0; 3]],
        species: vec!["O".into()],
        cell: None,
        units: Units::EvAngstrom,
    };
    // R0=-0.2, G=R0, B0=0.04, D=B0^2=0.0016.
    close(p.eval(&f).unwrap().energy, 0.0016, 1e-15);
    f.positions.push([1.0, 0.0, 0.0]);
    f.species.push("O".into());
    // Each center: R=[0.8,+/-1,0,0]; three padded rows.
    // B=[(0.64+3*0.04)/4, +/-0.8/4,0,0]; E=2*(0.19^2+0.2^2).
    close(p.eval(&f).unwrap().energy, 0.1522, 1e-14);
}

#[test]
fn rejects_unsupported_and_malformed_models_and_frames() {
    let mut m = model();
    m.version = 2;
    assert!(NativePotential::new(m).is_err());
    let mut m = model();
    m.precision = "float32".into();
    assert!(NativePotential::new(m).is_err());
    let mut m = model();
    m.descriptor.kind = "hybrid".into();
    assert!(NativePotential::new(m).is_err());
    let mut m = model();
    m.descriptor.normalization[0].std[2] = 2.0;
    assert!(NativePotential::new(m).is_err());
    let mut m = model();
    m.fitting[0].layers[0].weights.pop();
    assert!(NativePotential::new(m).is_err());
    let mut m = model();
    m.fitting[0].layers[0].weights[0][0] = f64::NAN;
    assert!(NativePotential::new(m).is_err());
    let mut value = serde_json::to_value(model()).unwrap();
    value["descriptor"]["env_protection"] = 0.1.into();
    assert!(NativePotential::from_json(&value.to_string()).is_err());
    let mut value = serde_json::to_value(model()).unwrap();
    value["fitting"][0]["layers"][0]["activation"] = "gelu".into();
    assert!(NativePotential::from_json(&value.to_string()).is_err());
    let mut p = NativePotential::new(model()).unwrap();
    let mut f = frame();
    f.species[0] = "C".into();
    assert!(p.eval(&f).is_err());
    let mut f = frame();
    f.positions[1] = f.positions[0];
    assert!(p.eval(&f).is_err());
    let mut f = frame();
    f.positions[0][0] = f64::INFINITY;
    assert!(p.eval(&f).is_err());
    let mut f = frame();
    f.cell = Some([[0.0; 3]; 3]);
    assert!(p.eval(&f).is_err());
    let mut m = model();
    m.descriptor.sel = vec![1, 1];
    assert!(NativePotential::new(m)
        .unwrap()
        .eval(&frame())
        .unwrap_err()
        .0
        .contains("capacity overflow"));
}

#[test]
fn highly_skewed_cell_enumerates_images_beyond_nearest_fractional_cell() {
    let mut m = model();
    m.type_map = vec!["O".into()];
    m.atom_energy_bias = vec![0.0];
    m.descriptor.sel = vec![256];
    m.descriptor.rcut = 1.7;
    m.descriptor.rcut_smth = 1.2;
    m.descriptor.axis_neuron = 1;
    m.descriptor.normalization = vec![Normalization {
        mean: [0.0; 4],
        std: [1.0; 4],
    }];
    let constant = Network {
        layers: vec![Layer {
            weights: vec![vec![0.0]],
            bias: vec![1.0],
            activation: Activation::Linear,
            residual: Residual::None,
            timestep: None,
        }],
    };
    m.descriptor.embedding = vec![vec![constant]];
    m.fitting = vec![Network {
        layers: vec![Layer {
            weights: vec![vec![1.0]],
            bias: vec![0.0],
            activation: Activation::Linear,
            residual: Residual::None,
            timestep: None,
        }],
    }];
    let cell = [[2.0, 0.0, 0.0], [1.8, 0.35, 0.0], [0.1, 0.2, 2.1]];
    let f = Frame {
        positions: vec![[0.0; 3]],
        species: vec!["O".into()],
        cell: Some(cell),
        units: Units::EvAngstrom,
    };
    let e = NativePotential::new(m).unwrap().eval(&f).unwrap();
    // Independent overcomplete integer cube, with an embedding G=1 so the
    // descriptor has a simple explicit scalar value. Includes shifts |n|>1.
    let mut radial = 0.0;
    let mut far_image = false;
    for a in -10i32..=10 {
        for b in -10i32..=10 {
            for c in -10i32..=10 {
                if a == 0 && b == 0 && c == 0 {
                    continue;
                }
                let d: [f64; 3] = std::array::from_fn(|k| {
                    a as f64 * cell[0][k] + b as f64 * cell[1][k] + c as f64 * cell[2][k]
                });
                let r = d.iter().map(|x| x * x).sum::<f64>().sqrt();
                if r >= 1.7 {
                    continue;
                }
                far_image |= a.abs().max(b.abs()).max(c.abs()) > 1;
                let q = if r <= 1.2 {
                    1.0
                } else {
                    let t = (r - 1.2) / 0.5;
                    1.0 - 10.0 * t.powi(3) + 15.0 * t.powi(4) - 6.0 * t.powi(5)
                };
                radial += q / r;
            }
        }
    }
    assert!(far_image);
    close(e.energy, (radial / 256.0).powi(2), 1e-14);
}

#[test]
fn type_order_is_explicit_and_activation_gradients_are_consistent() {
    let f = frame();
    let expected = eval(&f);
    let mut m = model();
    m.type_map.swap(0, 1);
    m.descriptor.sel.swap(0, 1);
    m.descriptor.normalization.swap(0, 1);
    m.descriptor.embedding.swap(0, 1);
    for row in &mut m.descriptor.embedding {
        row.swap(0, 1);
    }
    m.fitting.swap(0, 1);
    m.atom_energy_bias.swap(0, 1);
    let reordered = NativePotential::new(m).unwrap().eval(&f).unwrap();
    close(reordered.energy, expected.energy, 1e-14);
    for (a, b) in reordered
        .forces
        .iter()
        .flatten()
        .zip(expected.forces.iter().flatten())
    {
        close(*a, *b, 1e-13);
    }
    for activation in [Activation::Sigmoid, Activation::Softplus] {
        let mut m = model();
        for row in &mut m.descriptor.embedding {
            for net in row {
                for layer in &mut net.layers {
                    layer.activation = activation;
                }
            }
        }
        for net in &mut m.fitting {
            net.layers[0].activation = activation;
        }
        let mut p = NativePotential::new(m).unwrap();
        let e = p.eval(&f).unwrap();
        let mut plus = f.clone();
        let mut minus = f.clone();
        plus.positions[1][0] += 1e-5;
        minus.positions[1][0] -= 1e-5;
        close(
            e.forces[1][0],
            -(p.eval(&plus).unwrap().energy - p.eval(&minus).unwrap().energy) / 2e-5,
            2e-8,
        );
    }
    let mut m = model();
    m.fitting[0].layers[0].weights[0][0] = 1e308;
    m.fitting[0].layers[0].bias[0] = 1e308;
    // Large finite weights are legal, but saturated infinities in intermediate
    // computations must not be silently turned into finite tanh values.
    m.descriptor.normalization[0].std[0] = 1e-20;
    assert!(NativePotential::new(m).unwrap().eval(&f).is_err());
}
