use omnimd_ml_contract::{Frame, Potential};
use omnimd_native::{
    full_gradient::{frame_gradient_with_virial, ParameterLayout},
    full_training::{
        train_full, BatchOrder, Checkpoint, FullConfig, MiniBatchConfig, SamplerState, VirialLoss,
    },
    training::{Split, TrainingDevice},
    training_data::{Dataset, DatasetFrame, DatasetHeader, Labels, VirialConvention},
    Model, NativePotential,
};
use std::collections::BTreeMap;
fn model() -> Model {
    serde_json::from_str(include_str!("../fixtures/full-tanh-v1.json")).unwrap()
}
fn config(steps: u64) -> FullConfig {
    FullConfig {
        minibatch: Some(MiniBatchConfig {
            batch_size: 2,
            seed: 20261016,
            order: BatchOrder::ShuffleEachEpoch,
        }),
        fixed_geometry_cache: false,
        cuda_gradient: Default::default(),
        cuda_tiling: None,
        cuda_loss: Default::default(),
        split: Split::Explicit {
            validation_ids: vec!["set.000/2".into()],
        },
        steps,
        energy_weight: 1.0,
        force_weight: 1.0,
        virial_loss: Some(VirialLoss {
            weight: 0.7,
            scale_ev_per_atom: 0.2,
        }),
        energy_scale_ev_per_atom: 0.1,
        force_scale_ev_per_angstrom: 0.1,
        learning_rate: 0.005,
        learning_rate_schedule: None,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        gradient_clip_norm: Some(10.0),
        checkpoint_every: 1,
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
        if i % 3 == 0 {
            f.positions.pop();
            f.species.pop();
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

fn roundtrip(s: &Checkpoint) -> Checkpoint {
    serde_json::from_slice(&serde_json::to_vec(s).unwrap()).unwrap()
}
fn comparable(s: &Checkpoint) -> serde_json::Value {
    let mut v = serde_json::to_value(s).unwrap();
    v["config"].as_object_mut().unwrap().remove("steps");
    v["config"]
        .as_object_mut()
        .unwrap()
        .remove("fixed_geometry_cache");
    v
}
fn close(a: f64, b: f64) {
    assert!(
        (a - b).abs() < 2e-13 * (1. + a.abs().max(b.abs())),
        "{a} vs {b}"
    );
}
#[test]
fn actual_partial_batch_efv_gradient_adam_oracle() {
    let (model, d) = data(6);
    let c = config(1);
    let layout = ParameterLayout::new(&model).unwrap();
    let mut state = Checkpoint::start(&model, &d, &c).unwrap();
    for step in 1..=7 {
        let before = state.clone();
        let batch = before
            .sampler
            .as_ref()
            .unwrap()
            .batch(c.minibatch.as_ref().unwrap())
            .unwrap();
        if step <= 3 {
            let expected: &[usize] = match step {
                1 => &[3, 1],
                2 => &[5, 4],
                _ => &[0],
            };
            assert_eq!(batch, expected);
        }
        let mut g = vec![0.; layout.parameters.len()];
        for &i in batch {
            let (_, dg) = frame_gradient_with_virial(
                &before.model,
                &d.frames[i],
                &c.loss(),
                c.virial_loss.as_ref(),
                batch.len(),
            )
            .unwrap();
            for (a, b) in g.iter_mut().zip(dg) {
                *a += b;
            }
        }
        let norm = g.iter().fold(0.0_f64, |a, b| a.hypot(*b));
        if norm > 10. {
            for x in &mut g {
                *x *= 10. / norm;
            }
        }
        let old = layout.values(&before.model);
        let out = train_full(roundtrip(&before), &d, &c, |_| Ok(())).unwrap();
        let report = out.report.minibatch.as_ref().unwrap();
        assert_eq!(report.frames_processed, batch.len() as u64);
        assert_eq!(report.partial_batches, u64::from(batch.len() < 2));
        state = out.checkpoint;
        let values = layout.values(&state.model);
        for i in 0..g.len() {
            let m = 0.9 * before.first_moment[i] + (1. - 0.9) * g[i];
            let v = 0.999 * before.second_moment[i] + (1. - 0.999) * g[i] * g[i];
            let x = old[i]
                - 0.005 * (m / (1. - 0.9_f64.powf(step as f64)))
                    / ((v / (1. - 0.999_f64.powf(step as f64))).sqrt() + 1e-8);
            close(state.first_moment[i], m);
            close(state.second_moment[i], v);
            close(values[i], x);
        }
        state
            .sampler
            .as_ref()
            .unwrap()
            .validate(c.minibatch.as_ref().unwrap(), &state.training_indices, step)
            .unwrap();
    }
}
#[test]
fn exact_resume_mid_epoch_partial_boundary_and_save_failure() {
    let (model, d) = data(6);
    let c = config(7);
    let initial = Checkpoint::start(&model, &d, &c).unwrap();
    let mut snapshots = Vec::new();
    let all = train_full(initial.clone(), &d, &c, |s| {
        snapshots.push(roundtrip(s));
        Ok(())
    })
    .unwrap();
    assert_eq!(all.report.minibatch.as_ref().unwrap().frames_processed, 12);
    assert_eq!(all.report.minibatch.as_ref().unwrap().partial_batches, 2);
    assert_eq!(
        all.report.minibatch.as_ref().unwrap().batch_size_counts,
        BTreeMap::from([(1, 2), (2, 5)])
    );
    for split in [1, 2, 3, 4, 6] {
        let mut remaining = c.clone();
        remaining.steps = 7 - split;
        let resumed = train_full(
            snapshots[split as usize - 1].clone(),
            &d,
            &remaining,
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(comparable(&all.checkpoint), comparable(&resumed.checkpoint));
    }
    let mut persisted = None;
    let failed = train_full(initial, &d, &c, |s| {
        if s.step == 2 {
            return Err(omnimd_ml_contract::Error(
                "injected checkpoint save failure".into(),
            ));
        }
        persisted = Some(roundtrip(s));
        Ok(())
    });
    assert!(failed.is_err());
    let saved = persisted.unwrap();
    assert_eq!(saved.step, 1);
    let mut remaining = c.clone();
    remaining.steps = 6;
    assert_eq!(
        comparable(&all.checkpoint),
        comparable(
            &train_full(saved, &d, &remaining, |_| Ok(()))
                .unwrap()
                .checkpoint
        )
    );
    // Validation labels can change metrics, never the training trajectory.
    let (_, mut changed) = data(6);
    changed.frames[2].labels.energy = Some(120.);
    let other = train_full(
        Checkpoint::start(&model, &changed, &c).unwrap(),
        &changed,
        &c,
        |_| Ok(()),
    )
    .unwrap();
    assert_eq!(comparable(&all.checkpoint), comparable(&other.checkpoint));
}
#[test]
fn legacy_full_batch_and_sequential_large_batches_keep_numeric_state() {
    let (m, d) = data(6);
    let mut c = config(4);
    c.minibatch = None;
    let legacy = train_full(Checkpoint::start(&m, &d, &c).unwrap(), &d, &c, |_| Ok(()))
        .unwrap()
        .checkpoint;
    assert!(serde_json::to_value(&legacy)
        .unwrap()
        .get("sampler")
        .is_none());
    assert!(serde_json::to_value(&c).unwrap().get("minibatch").is_none());
    for batch_size in [5, 8] {
        c.minibatch = Some(MiniBatchConfig {
            batch_size,
            seed: 0,
            order: BatchOrder::Sequential,
        });
        let actual = train_full(Checkpoint::start(&m, &d, &c).unwrap(), &d, &c, |_| Ok(()))
            .unwrap()
            .checkpoint;
        assert_eq!(actual.schema, "omnimd-native-full-checkpoint-v5");
        assert_eq!(
            serde_json::to_value(&actual.model).unwrap(),
            serde_json::to_value(&legacy.model).unwrap()
        );
        assert_eq!(actual.first_moment, legacy.first_moment);
        assert_eq!(actual.second_moment, legacy.second_moment);
        assert!(legacy
            .validate_for_device(&d, &c, TrainingDevice::Cpu)
            .is_err());
    }
}
#[test]
fn sampler_corruption_and_configuration_changes_fail_closed() {
    let (m, d) = data(6);
    let c = config(1);
    let state = Checkpoint::start(&m, &d, &c).unwrap();
    let check =
        |s: &Checkpoint| assert!(s.validate_for_device(&d, &c, TrainingDevice::Cpu).is_err());
    for field in [
        "algorithm",
        "seed",
        "order_mode",
        "epoch",
        "cursor",
        "order",
        "rng_state",
    ] {
        let mut json = serde_json::to_value(&state).unwrap();
        json["sampler"][field] = match field {
            "algorithm" => serde_json::json!("unknown"),
            "order_mode" => serde_json::json!("sequential"),
            "order" => serde_json::json!([0, 1, 2, 3, 5]),
            _ => serde_json::json!(1),
        };
        check(&serde_json::from_value(json).unwrap());
    }
    let mut bad = state.clone();
    bad.sampler = None;
    check(&bad);
    bad = state.clone();
    bad.step = 1;
    check(&bad);
    bad = state.clone();
    bad.schema = "omnimd-native-full-checkpoint-v3".into();
    check(&bad);
    for field in ["batch_size", "seed", "order"] {
        let mut json = serde_json::to_value(&c).unwrap();
        json["minibatch"][field] = if field == "order" {
            serde_json::json!("sequential")
        } else {
            serde_json::json!(3)
        };
        let other: FullConfig = serde_json::from_value(json).unwrap();
        assert!(state
            .validate_for_device(&d, &other, TrainingDevice::Cpu)
            .is_err());
    }
    let mut ef = c.clone();
    ef.virial_loss = None;
    let ef_state = train_full(Checkpoint::start(&m, &d, &ef).unwrap(), &d, &ef, |_| Ok(()))
        .unwrap()
        .checkpoint;
    assert_eq!(ef_state.schema, "omnimd-native-full-checkpoint-v5");
    assert_eq!(
        ef_state.loss_contract.as_deref(),
        Some("energy-force-per-atom-v1")
    );
    ef_state
        .validate_for_device(&d, &ef, TrainingDevice::Cpu)
        .unwrap();
    let big = SamplerState::for_step(
        c.minibatch.as_ref().unwrap(),
        &[0, 1, 2, 3, 4],
        1_000_000_000,
    )
    .unwrap();
    big.validate(
        c.minibatch.as_ref().unwrap(),
        &[0, 1, 2, 3, 4],
        1_000_000_000,
    )
    .unwrap();
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an explicitly authorized CUDA device window"]
fn cuda_minibatch_cpu_parity_cache_toggles_and_exact_resume() {
    use omnimd_native::full_training::{train_full_with_device, CudaGradient};
    let (model, d) = data(6);
    let base = config(7);
    let layout = ParameterLayout::new(&model).unwrap();
    let cpu = train_full(
        Checkpoint::start(&model, &d, &base).unwrap(),
        &d,
        &base,
        |_| Ok(()),
    )
    .unwrap();
    let mut evidence = Vec::new();
    for mode in [CudaGradient::Hyperdual, CudaGradient::Reverse] {
        let mut c = base.clone();
        c.cuda_gradient = mode;
        let mut snapshots = Vec::new();
        let off = train_full_with_device(
            Checkpoint::start(&model, &d, &c).unwrap(),
            &d,
            &c,
            TrainingDevice::Cuda { visible_device: 0 },
            |s| {
                snapshots.push(roundtrip(s));
                Ok(())
            },
        )
        .unwrap();
        c.fixed_geometry_cache = true;
        let on = train_full_with_device(
            Checkpoint::start(&model, &d, &c).unwrap(),
            &d,
            &c,
            TrainingDevice::Cuda { visible_device: 0 },
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(comparable(&off.checkpoint), comparable(&on.checkpoint));
        for split in [2, 3, 4] {
            let mut remaining = c.clone();
            remaining.steps = 7 - split;
            let resumed = train_full_with_device(
                snapshots[split as usize - 1].clone(),
                &d,
                &remaining,
                TrainingDevice::Cuda { visible_device: 0 },
                |_| Ok(()),
            )
            .unwrap();
            assert_eq!(comparable(&off.checkpoint), comparable(&resumed.checkpoint));
        }
        let mut worst = 0.0_f64;
        for (x, y) in layout
            .values(&off.checkpoint.model)
            .iter()
            .chain(&off.checkpoint.first_moment)
            .chain(&off.checkpoint.second_moment)
            .zip(
                layout
                    .values(&cpu.checkpoint.model)
                    .iter()
                    .chain(&cpu.checkpoint.first_moment)
                    .chain(&cpu.checkpoint.second_moment),
            )
        {
            worst = worst.max((x - y).abs());
            assert!((x - y).abs() < 2e-10, "{x} vs {y}");
        }
        assert_eq!(on.report.minibatch.as_ref().unwrap().frames_processed, 12);
        evidence.push(serde_json::json!({"executor":mode,"max_parameter_moment_difference_to_cpu":worst,"cache":on.report.fixed_geometry_cache,"off_timing":off.report.timing,"on_timing":on.report.timing,"exact_cache_toggle_resume_steps":[2,3,4],"execution":off.report.execution}));
    }
    eprintln!(
        "{}",
        serde_json::to_string(&serde_json::json!({"cpu_cuda_minibatch":evidence})).unwrap()
    );
}

#[test]
fn final_partial_batch_normalization_is_preflighted() {
    let (model, d) = data(6);
    let mut c = config(1);
    c.energy_weight = 0.;
    c.force_weight = 0.;
    c.virial_loss = Some(VirialLoss {
        weight: 1.,
        scale_ev_per_atom: 1e-155,
    });
    // Every training frame has >=3 atoms. For the three-atom frame the coefficient
    // is 2/(9*N*n*n*s*s): finite at N=2, infinite at N=1.
    let s2 = (1.0_f64 / 2. / 9.).sqrt() / 3. / 1e-155;
    let s1 = (1.0_f64 / 9.).sqrt() / 3. / 1e-155;
    assert!((2. * s2 * s2).is_finite());
    assert!(!(2. * s1 * s1).is_finite());
    assert!(Checkpoint::start(&model, &d, &c).is_err());
    c.minibatch = None;
    assert!(Checkpoint::start(&model, &d, &c).is_ok());
}
