#![cfg(feature = "cuda")]
use omnimd_ml_contract::{Evaluation, Frame, Potential};
use omnimd_native::{
    full_gradient::{frame_gradient_with_virial, ParameterLayout},
    full_training::{train_full, Checkpoint, FullConfig, VirialLoss},
    training::{Split, TrainingDevice},
    training_data::{Dataset, DatasetFrame, DatasetHeader, Labels, VirialConvention},
    Activation, Layer, Model, NativePotential, Residual,
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
        learning_rate_schedule: None,
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

fn boundary_model() -> (Model, DatasetFrame) {
    let mut m = model();
    m.type_map.truncate(1);
    m.atom_energy_bias.truncate(1);
    m.descriptor.sel = vec![1];
    m.descriptor.axis_neuron = 1;
    m.descriptor.normalization.truncate(1);
    m.descriptor.embedding.truncate(1);
    m.descriptor.embedding[0].truncate(1);
    m.fitting.truncate(1);
    let mut layer = m.descriptor.embedding[0][0].layers[0].clone();
    layer.weights = vec![(0..32).map(|i| 0.1 * ((i + 1) as f64).sin()).collect()];
    layer.bias = vec![0.02; 32];
    layer.timestep = Some(vec![0.7; 32]);
    layer.residual = Residual::None;
    m.descriptor.embedding[0][0].layers = vec![layer];
    let mut output = m.fitting[0].layers.last().unwrap().clone();
    output.weights = vec![vec![0.13]; 32];
    m.fitting[0].layers = vec![output];
    let f = Frame {
        positions: vec![[0.0; 3], [1.1, 0.2, 0.1]],
        species: vec!["O".into(); 2],
        cell: None,
        units: m.units,
    };
    let pred = NativePotential::new(m.clone()).unwrap().eval(&f).unwrap();
    let mut forces = pred.forces;
    forces[0][0] += 0.03;
    forces[1][0] -= 0.03;
    (
        m,
        DatasetFrame {
            index: 0,
            id: "boundary/0".into(),
            frame: f,
            labels: Labels {
                energy: Some(pred.energy + 0.2),
                forces: Some(forces),
                virial: None,
            },
        },
    )
}

use omnimd_native::{
    cuda::{CudaFullTrainer, FullGradientProfile},
    full_training::{train_full_with_device, CudaGradient, FullExecution},
};
#[test]
fn reverse_configuration_and_backend_guards_without_driver() {
    let m = model();
    let d = dataset(&m);
    let mut c = config(2);
    assert!(serde_json::to_value(&c)
        .unwrap()
        .get("cuda_gradient")
        .is_none());
    c.cuda_gradient = CudaGradient::Reverse;
    let mut s = Checkpoint::start(&m, &d, &c).unwrap();
    assert_eq!(s.schema, "omnimd-native-full-checkpoint-v4");
    assert!(s
        .validate_for_device(&d, &c, TrainingDevice::Cpu)
        .unwrap_err()
        .0
        .contains("explicit CUDA"));
    s.validate_for_device(&d, &c, TrainingDevice::Cuda { visible_device: 0 })
        .unwrap();
    s.step = 1;
    s.execution = FullExecution::CudaHyperdualF64V1 {
        visible_device: 0,
        device_info: "test".into(),
        driver_version: 0,
        program_sha256: "test".into(),
    };
    assert!(s
        .validate_for_device(&d, &c, TrainingDevice::Cuda { visible_device: 0 })
        .unwrap_err()
        .0
        .contains("backend/device"));
    s.execution = FullExecution::CudaReverseF64V1 {
        visible_device: 0,
        device_info: "test".into(),
        driver_version: 0,
        program_sha256: "test".into(),
    };
    s.validate_for_device(&d, &c, TrainingDevice::Cuda { visible_device: 0 })
        .unwrap();
    let mut ef = c.clone();
    ef.virial_loss = None;
    let e = Checkpoint::start(&m, &d, &ef).unwrap();
    assert_eq!(e.loss_contract.as_deref(), Some("energy-force-per-atom-v1"));
    assert!(s
        .validate_for_device(&d, &ef, TrainingDevice::Cuda { visible_device: 0 })
        .is_err());
}
fn compare(m: &Model, records: &[DatasetFrame]) -> (f64, f64) {
    let mut reverse =
        CudaFullTrainer::new_with_gradient(m.clone(), 0, CudaGradient::Reverse).unwrap();
    let mut reference = CudaFullTrainer::new(m.clone(), 0).unwrap();
    let mut cpu_max = 0.0_f64;
    let mut gpu_max = 0.0_f64;
    for r in records {
        for mode in 0..4 {
            let mut c = config(1);
            if mode < 3 {
                c.energy_weight = if mode == 0 { 1.0 } else { 0.0 };
                c.force_weight = if mode == 1 { 1.0 } else { 0.0 };
                if mode < 2 {
                    c.virial_loss = None;
                }
            }
            let (loss, g) = reverse
                .frame_gradient_with_virial(r, &c.loss(), c.virial_loss.as_ref(), 3)
                .unwrap();
            let (loss2, g2) = reference
                .frame_gradient_with_virial(r, &c.loss(), c.virial_loss.as_ref(), 3)
                .unwrap();
            let (loss3, g3) = gradient(m, r, &c, 3);
            assert_eq!(loss, loss2);
            assert_eq!(loss, loss3);
            gpu_max = gpu_max.max(vector_close(&g, &g2, 2e-10));
            cpu_max = cpu_max.max(vector_close(&g, &g3, 2e-10));
            assert_eq!(
                g,
                reverse
                    .frame_gradient_with_virial(r, &c.loss(), c.virial_loss.as_ref(), 3)
                    .unwrap()
                    .1
            );
        }
    }
    (cpu_max, gpu_max)
}
#[test]
#[ignore = "requires an authorized exclusive GPU window"]
fn reverse_gradients_adam_resume_and_resources() {
    let m = model();
    let d = dataset(&m);
    let (mut cpu_max, mut gpu_max) = compare(&m, &d.frames);
    // Zero timestep is differentiable; saturated tanh never divides by its slope.
    let mut edge = m.clone();
    for row in &mut edge.descriptor.embedding {
        for net in row {
            net.layers[0].timestep.as_mut().unwrap()[0] = 0.0;
            net.layers[0].bias[1] = 30.0;
            net.layers[0].timestep.as_mut().unwrap()[1] = -0.4;
        }
    }
    let x = compare(&edge, &d.frames);
    cpu_max = cpu_max.max(x.0);
    gpu_max = gpu_max.max(x.1);
    let mut axis = m.clone();
    axis.descriptor.axis_neuron = 2;
    for net in &mut axis.fitting {
        let rows = net.layers[0].weights.clone();
        net.layers[0].weights.extend(rows);
    }
    let x = compare(&axis, &d.frames);
    cpu_max = cpu_max.max(x.0);
    gpu_max = gpu_max.max(x.1);
    let (b, mut r) = boundary_model();
    r.labels.virial = Some(evaluate(&b, &r.frame).virial);
    let x = compare(&b, &[r]);
    cpu_max = cpu_max.max(x.0);
    gpu_max = gpu_max.max(x.1);
    let mut c = config(2);
    c.gradient_clip_norm = Some(1e-4);
    let initial = Checkpoint::start(&m, &d, &c).unwrap();
    let cpu = train_full(initial.clone(), &d, &c, |_| Ok(())).unwrap();
    let dev = TrainingDevice::Cuda { visible_device: 0 };
    let hyper = train_full_with_device(initial, &d, &c, dev, |_| Ok(())).unwrap();
    c.cuda_gradient = CudaGradient::Reverse;
    let initial = Checkpoint::start(&m, &d, &c).unwrap();
    let first = train_full_with_device(initial.clone(), &d, &c, dev, |_| Ok(())).unwrap();
    let layout = ParameterLayout::new(&m).unwrap();
    let updates = vector_close(
        &layout.values(&cpu.checkpoint.model),
        &layout.values(&first.checkpoint.model),
        1e-11,
    )
    .max(vector_close(
        &layout.values(&hyper.checkpoint.model),
        &layout.values(&first.checkpoint.model),
        1e-11,
    ));
    let moments = vector_close(
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
    let mut c3 = c.clone();
    c3.steps = 3;
    let resumed = train_full_with_device(restored, &d, &c3, dev, |_| Ok(())).unwrap();
    let mut c5 = c.clone();
    c5.steps = 5;
    let continuous = train_full_with_device(initial, &d, &c5, dev, |_| Ok(())).unwrap();
    let mut expected = continuous.checkpoint;
    expected.config.steps = 3;
    assert_eq!(
        serde_json::to_value(&resumed.checkpoint).unwrap(),
        serde_json::to_value(expected).unwrap()
    );
    let mut failing =
        CudaFullTrainer::new_with_gradient(m.clone(), 0, CudaGradient::Reverse).unwrap();
    let before = serde_json::to_vec(failing.model()).unwrap();
    assert!(failing
        .adam_step(&vec![1e308; 112], &c)
        .unwrap_err()
        .0
        .contains("nonfinite CUDA Adam"));
    assert_eq!(before, serde_json::to_vec(failing.model()).unwrap());
    assert!(failing
        .frame_gradient(&d.frames[0], &c.loss(), 1)
        .unwrap_err()
        .0
        .contains("earlier failure"));
    let mut altered = first.checkpoint.clone();
    if let FullExecution::CudaReverseF64V1 { program_sha256, .. } = &mut altered.execution {
        *program_sha256 = "different".into();
    }
    assert!(train_full_with_device(altered, &d, &c, dev, |_| Ok(()))
        .err()
        .unwrap()
        .0
        .contains("compiled-program"));
    println!(
        "{}",
        serde_json::json!({"reverse_cpu_gradient_max_abs":cpu_max,"reverse_hyperdual_gradient_max_abs":gpu_max,"two_step_parameters_max_abs":updates,"two_step_moments_max_abs":moments,"resume_2_plus_3_exact":true,"nonfinite_update_preserves_host_snapshot_and_poisons":true,"changed_program_rejected":true,"execution":first.report.execution,"hyperdual_execution":hyper.report.execution})
    );
}
fn medium() -> Model {
    let mut m = model();
    m.descriptor.axis_neuron = 2;
    let layer = |ni: usize, no: usize, residual: Residual, linear: bool| Layer {
        weights: (0..ni)
            .map(|i| {
                (0..no)
                    .map(|j| 0.12 * ((i * no + j + 1) as f64 * 0.7).sin())
                    .collect()
            })
            .collect(),
        bias: vec![0.03; no],
        activation: if linear {
            Activation::Linear
        } else {
            Activation::Tanh
        },
        residual,
        timestep: if linear { None } else { Some(vec![0.7; no]) },
    };
    for row in &mut m.descriptor.embedding {
        for net in row {
            net.layers = vec![
                layer(1, 4, Residual::None, false),
                layer(4, 4, Residual::Identity, false),
            ];
        }
    }
    for net in &mut m.fitting {
        let mut first = layer(8, 8, Residual::None, false);
        first.timestep = None;
        net.layers = vec![
            first,
            layer(8, 8, Residual::Identity, false),
            layer(8, 1, Residual::None, true),
        ];
    }
    m
}
fn pair_record(m: &Model, n: usize) -> DatasetFrame {
    let positions = (0..n)
        .map(|i| {
            [
                (i / 2) as f64 * 8.0 + (i % 2) as f64 * 1.1,
                (i % 2) as f64 * 0.2,
                0.1,
            ]
        })
        .collect();
    let f = Frame {
        positions,
        species: (0..n).map(|i| m.type_map[i % 2].clone()).collect(),
        cell: None,
        units: m.units,
    };
    let mut e = evaluate(m, &f);
    e.energy += 0.2 * n as f64;
    e.forces[0][0] += 0.03;
    e.virial[0][1] += 0.1;
    DatasetFrame {
        index: 0,
        id: "pairs".into(),
        frame: f,
        labels: Labels {
            energy: Some(e.energy),
            forces: Some(e.forces),
            virial: Some(e.virial),
        },
    }
}
#[test]
#[ignore = "requires an authorized exclusive GPU window"]
fn reverse_profile_and_growing_scratch() {
    let mut reports = Vec::new();
    for (name, m, n) in [("tiny", model(), 4), ("medium", medium(), 64)] {
        let r = pair_record(&m, n);
        let c = config(1);
        let mut grads = Vec::new();
        let mut timings = Vec::new();
        let mut resource = serde_json::Value::Null;
        let begin = std::time::Instant::now();
        let (_, cpu) = gradient(&m, &r, &c, 1);
        let cpu_ms = begin.elapsed().as_secs_f64() * 1000.;
        for choice in [CudaGradient::Hyperdual, CudaGradient::Reverse] {
            let mut gpu = CudaFullTrainer::new_with_gradient(m.clone(), 0, choice).unwrap();
            gpu.frame_gradient_with_virial(&r, &c.loss(), c.virial_loss.as_ref(), 1)
                .unwrap();
            let mut profile = FullGradientProfile::default();
            let begin = std::time::Instant::now();
            let mut g = Vec::new();
            for _ in 0..5 {
                g = gpu
                    .frame_gradient_with_virial(&r, &c.loss(), c.virial_loss.as_ref(), 1)
                    .unwrap()
                    .1;
                profile.add(gpu.last_profile());
            }
            let total_ms = begin.elapsed().as_secs_f64() * 1000.;
            vector_close(&cpu, &g, 2e-10);
            grads.push(g.clone());
            if choice == CudaGradient::Reverse {
                resource = serde_json::to_value(gpu.reverse_resources(n).unwrap()).unwrap();
                for size in [2, 16, 64, 4, 64] {
                    let v = pair_record(&m, size);
                    let a = gpu
                        .frame_gradient_with_virial(&v, &c.loss(), c.virial_loss.as_ref(), 1)
                        .unwrap()
                        .1;
                    let b = gradient(&m, &v, &c, 1).1;
                    vector_close(&a, &b, 2e-10);
                }
            }
            timings.push(serde_json::json!({"executor":choice,"five_frames_wall_ms":total_ms,"profile":profile,"kernels":gpu.kernel_resources()}));
        }
        let difference = vector_close(&grads[0], &grads[1], 2e-10);
        reports.push(serde_json::json!({"case":name,"atoms":n,"parameters":cpu.len(),"cpu_single_frame_ms":cpu_ms,"cross_gpu_max_abs":difference,"resources":resource,"timings":timings}));
    }
    println!(
        "{}",
        serde_json::json!({"profile_cases":reports,"growth_shrink_verified":true})
    );
}

#[test]
fn fixed_geometry_config_keeps_legacy_serialization_and_allows_runtime_toggle() {
    let m = model();
    let d = dataset(&m);
    let mut c = config(2);
    c.cuda_gradient = CudaGradient::Reverse;
    let state = Checkpoint::start(&m, &d, &c).unwrap();
    assert!(serde_json::to_value(&c)
        .unwrap()
        .get("fixed_geometry_cache")
        .is_none());
    c.fixed_geometry_cache = true;
    state
        .validate_for_device(&d, &c, TrainingDevice::Cuda { visible_device: 0 })
        .unwrap();
    assert!(state
        .validate_for_device(&d, &c, TrainingDevice::Cpu)
        .is_err());
    let mut many = dataset(&m);
    let mut frame = pair_record(&m, 64);
    many.frames = (0..400)
        .map(|i| {
            frame.index = i;
            frame.id = format!("large/{i}");
            frame.clone()
        })
        .collect();
    c.split = Split::Explicit {
        validation_ids: vec!["large/399".into()],
    };
    let large = Checkpoint::start(&m, &many, &c).unwrap();
    assert!(large
        .validate_for_device(
            &many,
            &c,
            TrainingDevice::Cuda {
                visible_device: 2147483647
            }
        )
        .unwrap_err()
        .0
        .contains("cache payload"));
}
#[test]
#[ignore = "requires an authorized exclusive GPU window"]
fn fixed_geometry_gradients_adam_switching_restart_and_profile() {
    let m = model();
    let d = dataset(&m);
    let dev = TrainingDevice::Cuda { visible_device: 0 };
    let mut evidence = Vec::new();
    for executor in [CudaGradient::Hyperdual, CudaGradient::Reverse] {
        let mut off = CudaFullTrainer::new_with_gradient(m.clone(), 0, executor).unwrap();
        let mut on = CudaFullTrainer::new_with_gradient(m.clone(), 0, executor).unwrap();
        on.prepare_fixed_geometry_cache(&d, &[0, 1, 2]).unwrap();
        let mut p0 = FullGradientProfile::default();
        let mut p1 = FullGradientProfile::default();
        for step in 0..3 {
            for mode in 0..4 {
                let mut c = config(1);
                c.cuda_gradient = executor;
                if mode < 3 {
                    c.energy_weight = if mode == 0 { 1.0 } else { 0.0 };
                    c.force_weight = if mode == 1 { 1.0 } else { 0.0 };
                    if mode < 2 {
                        c.virial_loss = None;
                    }
                }
                for r in &d.frames {
                    let mut changed = r.clone();
                    changed
                        .labels
                        .energy
                        .as_mut()
                        .map(|x| *x += step as f64 * 0.01);
                    if step == 1 {
                        changed.labels.forces.as_mut().unwrap()[0][1] += 0.02;
                        changed.labels.virial.as_mut().unwrap()[0][2] -= 0.03;
                    }
                    let a = off
                        .frame_gradient_with_virial(&changed, &c.loss(), c.virial_loss.as_ref(), 3)
                        .unwrap();
                    let b = on
                        .frame_gradient_with_virial(&changed, &c.loss(), c.virial_loss.as_ref(), 3)
                        .unwrap();
                    assert_eq!(a, b);
                    p0.add(off.last_profile());
                    p1.add(on.last_profile());
                    if mode == 3 {
                        off.adam_step(&a.1, &c).unwrap();
                        on.adam_step(&b.1, &c).unwrap();
                        assert_eq!(
                            serde_json::to_vec(off.model()).unwrap(),
                            serde_json::to_vec(on.model()).unwrap()
                        );
                        assert_eq!(off.moments(), on.moments());
                    }
                }
            }
        }
        let mut c = config(2);
        c.cuda_gradient = executor;
        let initial = Checkpoint::start(&m, &d, &c).unwrap();
        let a = train_full_with_device(initial.clone(), &d, &c, dev, |_| Ok(())).unwrap();
        c.fixed_geometry_cache = true;
        let b = train_full_with_device(initial.clone(), &d, &c, dev, |_| Ok(())).unwrap();
        let strip = |s: &Checkpoint| {
            let mut v = serde_json::to_value(s).unwrap();
            v["config"]
                .as_object_mut()
                .unwrap()
                .remove("fixed_geometry_cache");
            v["config"].as_object_mut().unwrap().remove("steps");
            v
        };
        assert_eq!(strip(&a.checkpoint), strip(&b.checkpoint));
        assert_eq!(b.report.fixed_geometry_cache.as_ref().unwrap().hits, 4);
        let restored: Checkpoint =
            serde_json::from_slice(&serde_json::to_vec(&a.checkpoint).unwrap()).unwrap();
        let resumed = train_full_with_device(restored, &d, &c, dev, |_| Ok(())).unwrap();
        c.steps = 4;
        let continuous = train_full_with_device(initial.clone(), &d, &c, dev, |_| Ok(())).unwrap();
        assert_eq!(strip(&resumed.checkpoint), strip(&continuous.checkpoint));
        c.steps = 2;
        c.fixed_geometry_cache = false;
        let restored: Checkpoint =
            serde_json::from_slice(&serde_json::to_vec(&b.checkpoint).unwrap()).unwrap();
        let resumed = train_full_with_device(restored, &d, &c, dev, |_| Ok(())).unwrap();
        assert_eq!(strip(&resumed.checkpoint), strip(&continuous.checkpoint));
        let stats = on.fixed_geometry_stats().unwrap();
        let mut changed = d.frames[0].clone();
        changed.frame.positions[0][0] += 0.001;
        assert!(on
            .frame_gradient(&changed, &c.loss(), 1)
            .unwrap_err()
            .0
            .contains("identity changed"));
        assert!(on
            .frame_gradient(&d.frames[0], &c.loss(), 1)
            .unwrap_err()
            .0
            .contains("earlier failure"));
        evidence.push(serde_json::json!({"executor":executor,"all_gradients_and_adam_bitwise_equal":true,"off_to_on_and_on_to_off_restart_exact":true,"identity_mismatch_stops_trainer":true,"cache":stats,"uncached_profile":p0,"cached_profile":p1}));
    }
    println!(
        "{}",
        serde_json::json!({"fixed_geometry_acceptance":evidence})
    );
}
