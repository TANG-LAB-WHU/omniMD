use omnimd_ml_contract::{Evaluation, Frame, Potential};
use omnimd_native::{
    full_gradient::{frame_gradient, frame_gradient_with_virial, ParameterLayout},
    full_training::{train_full, Checkpoint, FullConfig, VirialLoss},
    training::{Split, TrainingDevice},
    training_data::{Dataset, DatasetFrame, DatasetHeader, Labels, VirialConvention},
    Activation, Layer, Model, NativePotential, Network, Normalization, Residual,
};
use std::collections::BTreeMap;
fn model() -> Model {
    serde_json::from_str(include_str!("../fixtures/full-tanh-v1.json")).unwrap()
}
fn config(steps: u64) -> FullConfig {
    FullConfig {
        minibatch: None,
        fixed_geometry_cache: false,
        cuda_gradient: Default::default(),
        split: Split::Explicit {
            validation_ids: vec!["case/2".into()],
        },
        steps,
        energy_weight: 0.7,
        force_weight: 0.8,
        virial_loss: Some(VirialLoss {
            weight: 1.3,
            scale_ev_per_atom: 0.4,
        }),
        energy_scale_ev_per_atom: 0.3,
        force_scale_ev_per_angstrom: 0.2,
        learning_rate: 0.003,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        gradient_clip_norm: Some(0.1),
        checkpoint_every: 2,
    }
}
fn evaluate(m: &Model, f: &Frame) -> Evaluation {
    NativePotential::new(m.clone()).unwrap().eval(f).unwrap()
}
fn dataset(m: &Model) -> Dataset {
    let mut teacher = m.clone();
    let layout = ParameterLayout::new(m).unwrap();
    let values: Vec<_> = layout
        .values(m)
        .iter()
        .enumerate()
        .map(|(i, x)| x + 0.12 * ((i + 1) as f64 * 0.63).sin())
        .collect();
    layout.apply(&mut teacher, &values).unwrap();
    let mut f: Frame = serde_json::from_str(include_str!("../fixtures/frame.jsonl")).unwrap();
    f.cell = None;
    let mut periodic = f.clone();
    periodic.cell = Some([[3.8, 0.0, 0.0], [0.3, 3.9, 0.0], [0.1, 0.2, 4.0]]);
    periodic.positions.pop();
    periodic.species.pop();
    let self_image = Frame {
        positions: vec![[0.23, 0.17, 0.11]],
        species: vec!["O".into()],
        cell: Some([[2.5, 0.0, 0.0], [0.3, 2.6, 0.0], [0.2, 0.1, 2.7]]),
        units: m.units,
    };
    let frames = [f, periodic, self_image]
        .into_iter()
        .enumerate()
        .map(|(i, f)| {
            let e = evaluate(&teacher, &f);
            DatasetFrame {
                index: i as u64,
                id: format!("case/{i}"),
                frame: f,
                labels: Labels {
                    energy: Some(e.energy),
                    forces: Some(e.forces),
                    virial: Some(e.virial),
                },
            }
        })
        .collect();
    Dataset {
        header: DatasetHeader {
            schema: "omnimd-dataset-jsonl-v1".into(),
            type_map: m.type_map.clone(),
            units: m.units,
            virial_convention: VirialConvention::MinusDEpsilonRow,
            source: "original virial test".into(),
            selected_sets: vec![],
        },
        frames,
        source_sha256: BTreeMap::new(),
        dataset_sha256: "in-memory-virial-test".into(),
    }
}
fn objective(m: &Model, r: &DatasetFrame, c: &FullConfig, frames: usize) -> f64 {
    let e = evaluate(m, &r.frame);
    let n = r.frame.positions.len() as f64;
    let ef = c.energy_weight
        * ((e.energy - r.labels.energy.unwrap()) / n / c.energy_scale_ev_per_atom).powi(2)
        + c.force_weight / (3.0 * n)
            * e.forces
                .iter()
                .flatten()
                .zip(r.labels.forces.as_ref().unwrap().iter().flatten())
                .map(|(p, t)| ((p - t) / c.force_scale_ev_per_angstrom).powi(2))
                .sum::<f64>();
    let v = c.virial_loss.as_ref().map_or(0.0, |v| {
        v.weight / 9.0
            * e.virial
                .iter()
                .flatten()
                .zip(r.labels.virial.as_ref().unwrap().iter().flatten())
                .map(|(p, t)| ((p - t) / n / v.scale_ev_per_atom).powi(2))
                .sum::<f64>()
    });
    (ef + v) / frames as f64
}
fn gradient(m: &Model, r: &DatasetFrame, c: &FullConfig, frames: usize) -> (f64, Vec<f64>) {
    frame_gradient_with_virial(m, r, &c.loss(), c.virial_loss.as_ref(), frames).unwrap()
}
fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs())),
        "{a} vs {b}"
    );
}
fn vector_close(a: &[f64], b: &[f64], tol: f64) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut max = 0.0_f64;
    for (&x, &y) in a.iter().zip(b) {
        close(x, y, tol);
        max = max.max((x - y).abs());
    }
    max
}
#[test]
fn all_parameter_virial_and_joint_gradients_match_independent_finite_differences() {
    let m = model();
    let d = dataset(&m);
    let layout = ParameterLayout::new(&m).unwrap();
    let values = layout.values(&m);
    let mut maximum = 0.0_f64;
    for r in &d.frames {
        for only_virial in [true, false] {
            let mut c = config(1);
            if only_virial {
                c.energy_weight = 0.0;
                c.force_weight = 0.0;
            }
            let (loss, g) = gradient(&m, r, &c, 3);
            close(loss, objective(&m, r, &c, 3), 1e-13);
            for p in 0..values.len() {
                let h = 1e-5;
                let mut plus = m.clone();
                let mut minus = m.clone();
                let mut v = values.clone();
                v[p] += h;
                layout.apply(&mut plus, &v).unwrap();
                v[p] -= 2.0 * h;
                layout.apply(&mut minus, &v).unwrap();
                let fd = (objective(&plus, r, &c, 3) - objective(&minus, r, &c, 3)) / (2.0 * h);
                maximum = maximum.max((fd - g[p]).abs());
                close(g[p], fd, 3e-7);
            }
        }
    }
    let self_image = &d.frames[2];
    let e = evaluate(&m, &self_image.frame);
    assert!(e.forces.iter().flatten().all(|x| *x == 0.0));
    let mut c = config(1);
    c.energy_weight = 0.0;
    c.force_weight = 0.0;
    assert!(gradient(&m, self_image, &c, 1)
        .1
        .iter()
        .any(|g| g.abs() > 1e-8));
    println!(
        "{}",
        serde_json::json!({"virial_cpu_fd_max_abs":maximum,"parameters":values.len(),"geometries":3,"losses":2})
    );
}
#[test]
fn analytic_dimer_nine_virials_and_parameter_gradient() {
    let mut m = model();
    m.type_map = vec!["O".into()];
    m.atom_energy_bias = vec![0.0];
    m.descriptor.sel = vec![1];
    m.descriptor.normalization = vec![Normalization {
        mean: [0.0; 4],
        std: [1.0; 4],
    }];
    let g = 0.5_f64;
    let alpha = 0.7;
    m.descriptor.embedding = vec![vec![Network {
        layers: vec![Layer {
            weights: vec![vec![0.0]],
            bias: vec![g.atanh()],
            activation: Activation::Tanh,
            residual: Residual::None,
            timestep: None,
        }],
    }]];
    m.fitting = vec![Network {
        layers: vec![Layer {
            weights: vec![vec![alpha]],
            bias: vec![-0.2],
            activation: Activation::Linear,
            residual: Residual::None,
            timestep: None,
        }],
    }];
    let delta = [1.1, 0.4, -0.3];
    let r2 = delta.iter().map(|x| x * x).sum::<f64>();
    let f = Frame {
        positions: vec![[0.0; 3], delta],
        species: vec!["O".into(); 2],
        cell: None,
        units: m.units,
    };
    let e = evaluate(&m, &f);
    close(e.energy, 4.0 * alpha * g * g / r2 - 0.4, 1e-14);
    let mut label = e.virial;
    let mut expected = 0.0;
    let c = config(1);
    let v = c.virial_loss.as_ref().unwrap();
    for a in 0..3 {
        for b in 0..3 {
            let derivative = 8.0 * g * g * delta[a] * delta[b] / (r2 * r2);
            close(e.virial[a][b], alpha * derivative, 1e-14);
            let residual = 0.01 * (1 + a * 3 + b) as f64;
            label[a][b] -= residual;
            expected +=
                2.0 * v.weight * residual / (9.0 * 4.0 * v.scale_ev_per_atom.powi(2)) * derivative;
        }
    }
    let record = DatasetFrame {
        index: 0,
        id: "analytic".into(),
        frame: f,
        labels: Labels {
            energy: Some(e.energy),
            forces: Some(e.forces),
            virial: Some(label),
        },
    };
    let (_, gr) = gradient(&m, &record, &c, 1);
    close(gr[2], expected, 1e-13);
    assert_eq!(gr[3], 0.0);
    // Independently differentiate every ordered affine strain component of E.
    for a in 0..3 {
        for b in 0..3 {
            let h = 1e-6;
            let mut plus = record.frame.clone();
            let mut minus = record.frame.clone();
            for (p, q) in plus.positions.iter_mut().zip(&mut minus.positions) {
                p[b] += h * q[a];
                q[b] -= h * q[a];
            }
            let fd = -(evaluate(&m, &plus).energy - evaluate(&m, &minus).energy) / (2.0 * h);
            close(fd, e.virial[a][b], 2e-9);
        }
    }
}
#[test]
fn normalization_rotation_wrapping_and_zero_residual() {
    let m = model();
    let d = dataset(&m);
    let mut c = config(1);
    c.energy_weight = 0.0;
    c.force_weight = 0.0;
    let v = c.virial_loss.as_ref().unwrap();
    for r in &d.frames {
        let e = evaluate(&m, &r.frame);
        let mut exact = r.clone();
        exact.labels = Labels {
            energy: Some(e.energy),
            forces: Some(e.forces),
            virial: Some(e.virial),
        };
        assert_eq!(gradient(&m, &exact, &c, 1), (0.0, vec![0.0; 112]));
        let mut shifted = exact.clone();
        let mut ss = 0.0;
        for a in 0..3 {
            for b in 0..3 {
                let dv = 0.02 * (1 + a * 3 + b) as f64;
                ss += dv * dv;
                shifted.labels.virial.as_mut().unwrap()[a][b] -= dv;
            }
        }
        let (l1, g1) = gradient(&m, &shifted, &c, 1);
        let (l2, g2) = gradient(&m, &shifted, &c, 2);
        let n = r.frame.positions.len() as f64;
        close(
            l1,
            v.weight * ss / (9.0 * n * n * v.scale_ev_per_atom.powi(2)),
            1e-14,
        );
        close(l1, 2.0 * l2, 1e-14);
        vector_close(&g1, &g2.iter().map(|g| 2.0 * g).collect::<Vec<_>>(), 1e-13);
    }
    // A purely antisymmetric target contributes constant Frobenius loss.
    let mut anti = d.frames[0].clone();
    let e = evaluate(&m, &anti.frame);
    anti.labels = Labels {
        energy: Some(e.energy),
        forces: Some(e.forces),
        virial: Some(e.virial),
    };
    anti.labels.virial.as_mut().unwrap()[0][1] += 0.2;
    anti.labels.virial.as_mut().unwrap()[1][0] -= 0.2;
    let ag = gradient(&m, &anti, &c, 1);
    assert!(ag.0 > 0.0);
    assert!(ag.1.iter().all(|x| x.abs() < 1e-12));
    c.energy_weight = 0.7;
    c.force_weight = 0.8;
    let r = &d.frames[1];
    let base = gradient(&m, r, &c, 1);
    let rotate = |x: [f64; 3]| [-x[1], x[0], x[2]];
    let mut rotated = r.clone();
    rotated
        .frame
        .positions
        .iter_mut()
        .for_each(|x| *x = rotate(*x));
    rotated
        .frame
        .cell
        .as_mut()
        .unwrap()
        .iter_mut()
        .for_each(|x| *x = rotate(*x));
    rotated
        .labels
        .forces
        .as_mut()
        .unwrap()
        .iter_mut()
        .for_each(|x| *x = rotate(*x));
    let q = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let old = r.labels.virial.unwrap();
    rotated.labels.virial = Some(std::array::from_fn(|a| {
        std::array::from_fn(|b| {
            (0..3)
                .flat_map(|i| (0..3).map(move |j| q[i][a] * old[i][j] * q[j][b]))
                .sum()
        })
    }));
    let rg = gradient(&m, &rotated, &c, 1);
    close(base.0, rg.0, 1e-12);
    vector_close(&base.1, &rg.1, 1e-11);
    let mut wrapped = r.clone();
    for b in 0..3 {
        wrapped.frame.positions[0][b] += wrapped.frame.cell.unwrap()[1][b];
    }
    let wg = gradient(&m, &wrapped, &c, 1);
    close(base.0, wg.0, 1e-12);
    vector_close(&base.1, &wg.1, 1e-11);
}
#[test]
fn loss_configuration_checkpoint_compatibility_and_validation_isolation() {
    let m = model();
    let mut d = dataset(&m);
    let c = config(2);
    let old_json = serde_json::to_value({
        let mut ef = c.clone();
        ef.virial_loss = None;
        ef
    })
    .unwrap();
    assert!(old_json.get("virial_loss").is_none());
    let mut null_json = old_json.clone();
    null_json["virial_loss"] = serde_json::Value::Null;
    let ef: FullConfig = serde_json::from_value(null_json).unwrap();
    assert_eq!(serde_json::to_value(&ef).unwrap(), old_json);
    let old = Checkpoint::start(&m, &d, &ef).unwrap();
    let old_value = serde_json::to_value(&old).unwrap();
    assert!(old_value.get("loss_contract").is_none());
    assert!(old_value.get("execution").is_none());
    assert_eq!(old.schema, "omnimd-native-full-checkpoint-v1");
    assert_eq!(
        frame_gradient(&m, &d.frames[0], &ef.loss(), 2).unwrap(),
        gradient(&m, &d.frames[0], &ef, 2)
    );
    let state = Checkpoint::start(&m, &d, &c).unwrap();
    assert_eq!(state.schema, "omnimd-native-full-checkpoint-v3");
    let first = train_full(state.clone(), &d, &c, |_| Ok(())).unwrap();
    for (indices, reported) in [
        (&state.training_indices, &first.report.initial_training),
        (&state.validation_indices, &first.report.initial_validation),
    ] {
        let expected = indices
            .iter()
            .map(|&i| objective(&m, &d.frames[i], &c, indices.len()))
            .sum::<f64>();
        close(reported.normalized_total_objective, expected, 1e-13);
        close(
            reported.normalized_total_objective,
            reported.normalized_ef_objective + reported.normalized_virial_objective.unwrap(),
            1e-14,
        );
    }
    let restored: Checkpoint =
        serde_json::from_slice(&serde_json::to_vec(&first.checkpoint).unwrap()).unwrap();
    let mut c1 = c.clone();
    c1.steps = 1;
    let resume = train_full(restored.clone(), &d, &c1, |_| Ok(())).unwrap();
    let mut c3 = c.clone();
    c3.steps = 3;
    let continuous = train_full(state.clone(), &d, &c3, |_| Ok(())).unwrap();
    assert_eq!(
        serde_json::to_value(&resume.checkpoint).unwrap(),
        serde_json::to_value({
            let mut s = continuous.checkpoint;
            s.config.steps = 1;
            s
        })
        .unwrap()
    );
    for mut bad in [c.clone(), c.clone(), c.clone()].into_iter().enumerate() {
        match bad.0 {
            0 => bad.1.virial_loss = None,
            1 => bad.1.virial_loss.as_mut().unwrap().weight *= 2.0,
            _ => bad.1.virial_loss.as_mut().unwrap().scale_ev_per_atom *= 2.0,
        };
        assert!(restored
            .validate_for_device(&d, &bad.1, TrainingDevice::Cpu)
            .is_err());
    }
    let mut bad = restored.clone();
    bad.schema = "omnimd-native-full-checkpoint-v1".into();
    assert!(bad
        .validate_for_device(&d, &c, TrainingDevice::Cpu)
        .is_err());
    let mut bad = restored.clone();
    bad.loss_contract = Some("six-component-stress".into());
    assert!(bad
        .validate_for_device(&d, &c, TrainingDevice::Cpu)
        .is_err());
    for (weight, scale) in [
        (0.0, 1.0),
        (-1.0, 1.0),
        (1.0, 0.0),
        (f64::MIN_POSITIVE, 1e300),
        (1e300, 1e-300),
    ] {
        let mut bad = c.clone();
        bad.virial_loss = Some(VirialLoss {
            weight,
            scale_ev_per_atom: scale,
        });
        assert!(Checkpoint::start(&m, &d, &bad).is_err());
    }
    let mut only = c.clone();
    only.energy_weight = 0.0;
    only.force_weight = 0.0;
    Checkpoint::start(&m, &d, &only).unwrap();
    only.virial_loss = None;
    assert!(Checkpoint::start(&m, &d, &only).is_err());
    d.frames[2].labels.virial = None;
    assert!(Checkpoint::start(&m, &d, &c).is_err());
    Checkpoint::start(&m, &d, &ef).unwrap();
    let mut changed = dataset(&m);
    for x in changed.frames[2]
        .labels
        .virial
        .as_mut()
        .unwrap()
        .iter_mut()
        .flatten()
    {
        *x += 10.0;
    }
    let altered = train_full(
        Checkpoint::start(&m, &changed, &c).unwrap(),
        &changed,
        &c,
        |_| Ok(()),
    )
    .unwrap();
    assert_eq!(
        serde_json::to_value(first.checkpoint.model).unwrap(),
        serde_json::to_value(altered.checkpoint.model).unwrap()
    );
    assert_eq!(
        first.checkpoint.first_moment,
        altered.checkpoint.first_moment
    );
    assert!(
        altered.report.final_validation.normalized_total_objective
            > first.report.final_validation.normalized_total_objective
    );
}

#[test]
fn invalid_virial_cli_fails_before_device_creation_and_outputs() {
    let m = model();
    let mut d = dataset(&m);
    let c = config(1);
    let path = std::env::temp_dir().join(format!(
        "omnimd-virial-cli-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&path).unwrap();
    let dataset_path = path.join("data.jsonl");
    let request_path = path.join("request.json");
    let model_path = path.join("model.json");
    let output = path.join("must-not-exist");
    std::fs::write(&model_path, serde_json::to_vec(&m).unwrap()).unwrap();
    for missing in [true, false] {
        d.frames[2].labels.virial = if missing { None } else { Some([[0.0; 3]; 3]) };
        let mut records = vec![serde_json::json!({"kind":"header","header":d.header})];
        records.extend(
            d.frames
                .iter()
                .map(|r| serde_json::json!({"kind":"frame","data":r})),
        );
        records.push(serde_json::json!({"kind":"completed","frames":3,"source_sha256":{}}));
        std::fs::write(
            &dataset_path,
            records
                .iter()
                .map(|r| serde_json::to_string(r).unwrap() + "\n")
                .collect::<String>(),
        )
        .unwrap();
        let mut cfg = c.clone();
        if !missing {
            cfg.virial_loss.as_mut().unwrap().scale_ev_per_atom = 1e300;
        }
        let device = if cfg!(feature = "cuda") {
            "cuda:2147483647"
        } else {
            "cpu"
        };
        std::fs::write(
            &request_path,
            serde_json::to_vec(&serde_json::json!({
            "schema":"omnimd-native-full-train-v1","device":device,"initial_model":model_path,
            "dataset":dataset_path,"output_directory":output,"config":cfg}))
            .unwrap(),
        )
        .unwrap();
        let result = std::process::Command::new(env!("CARGO_BIN_EXE_omnimd-native-train"))
            .arg(&request_path)
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(result.stdout.is_empty());
        assert!(!output.exists());
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(
            stderr.contains(if missing {
                "requires all nine"
            } else {
                "normalization underflow/overflow"
            }),
            "{stderr}"
        );
    }
    for file in [dataset_path, request_path, model_path] {
        std::fs::remove_file(file).unwrap();
    }
    std::fs::remove_dir(path).unwrap();
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an authorized exclusive GPU window"]
fn cuda_virial_mixed_gradients_updates_and_restart() {
    use omnimd_native::{cuda::CudaFullTrainer, full_training::train_full_with_device};
    let m = model();
    let d = dataset(&m);
    let mut gpu = CudaFullTrainer::new(m.clone(), 0).unwrap();
    let mut maximum = 0.0_f64;
    for r in &d.frames {
        for only in [true, false] {
            let mut c = config(1);
            if only {
                c.energy_weight = 0.0;
                c.force_weight = 0.0;
            }
            let a = gradient(&m, r, &c, 3);
            let b = gpu
                .frame_gradient_with_virial(r, &c.loss(), c.virial_loss.as_ref(), 3)
                .unwrap();
            assert_eq!(a.0, b.0);
            maximum = maximum.max(vector_close(&a.1, &b.1, 2e-10));
        }
    }
    let c = config(2);
    let state = Checkpoint::start(&m, &d, &c).unwrap();
    let device = TrainingDevice::Cuda { visible_device: 0 };
    let cpu = train_full(state.clone(), &d, &c, |_| Ok(())).unwrap();
    let first = train_full_with_device(state.clone(), &d, &c, device, |_| Ok(())).unwrap();
    let layout = ParameterLayout::new(&m).unwrap();
    let update = vector_close(
        &layout.values(&cpu.checkpoint.model),
        &layout.values(&first.checkpoint.model),
        1e-11,
    );
    let moment = vector_close(
        &cpu.checkpoint.first_moment,
        &first.checkpoint.first_moment,
        1e-11,
    )
    .max(vector_close(
        &cpu.checkpoint.second_moment,
        &first.checkpoint.second_moment,
        1e-11,
    ));
    let restored: Checkpoint =
        serde_json::from_slice(&serde_json::to_vec(&first.checkpoint).unwrap()).unwrap();
    let mut c1 = c.clone();
    c1.steps = 1;
    let resumed = train_full_with_device(restored, &d, &c1, device, |_| Ok(())).unwrap();
    let mut c3 = c.clone();
    c3.steps = 3;
    let continuous = train_full_with_device(state, &d, &c3, device, |_| Ok(())).unwrap();
    let mut expected = continuous.checkpoint.clone();
    expected.config.steps = 1;
    assert_eq!(
        serde_json::to_value(resumed.checkpoint).unwrap(),
        serde_json::to_value(expected).unwrap()
    );
    println!(
        "{}",
        serde_json::json!({"virial_cuda_cpu_gradient_max_abs":maximum,"two_step_parameter_max_abs":update,
        "two_step_moment_max_abs":moment,"resume_2_plus_1_exact":true,"schema":first.checkpoint.schema,"execution":first.report.execution})
    );
}
