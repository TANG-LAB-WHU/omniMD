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
        cuda_tiling: None,
        cuda_loss: Default::default(),
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

#[test]
fn tiled_configuration_schema_and_resource_guards_without_driver() {
    use omnimd_native::full_training::CudaTiling;
    let m = model();
    let d = dataset(&m);
    let mut c = config(1);
    assert!(serde_json::to_value(&c)
        .unwrap()
        .get("cuda_tiling")
        .is_none());
    c.cuda_tiling = Some(CudaTiling {
        max_atoms_per_frame: 64,
        max_frames: 2,
        max_atoms: 7,
    });
    assert!(Checkpoint::start(&m, &d, &c).is_err());
    c.cuda_gradient = CudaGradient::Reverse;
    let s = Checkpoint::start(&m, &d, &c).unwrap();
    assert_eq!(s.schema, "omnimd-native-full-checkpoint-v7");
    assert!(s.validate_for_device(&d, &c, TrainingDevice::Cpu).is_err());
    let dev = TrainingDevice::Cuda { visible_device: 0 };
    s.validate_for_device(&d, &c, dev).unwrap();
    let mut invalid = c.clone();
    invalid.cuda_tiling.as_mut().unwrap().max_atoms = 3;
    let q = Checkpoint::start(&m, &d, &invalid).unwrap();
    assert!(q
        .validate_for_device(&d, &invalid, dev)
        .unwrap_err()
        .0
        .contains("largest frame"));
    assert!(s.validate_for_device(&d, &invalid, dev).is_err());
    for value in [
        serde_json::json!({"max_frames":0,"max_atoms":7}),
        serde_json::json!({"max_frames":33,"max_atoms":7}),
        serde_json::json!({"max_frames":2,"max_atoms":1025}),
    ] {
        let mut v = serde_json::to_value(&c).unwrap();
        v["cuda_tiling"] = value;
        let cfg: FullConfig = serde_json::from_value(v).unwrap();
        assert!(Checkpoint::start(&m, &d, &cfg).is_err());
    }
    assert!(serde_json::from_value::<CudaTiling>(
        serde_json::json!({"max_frames":2,"max_atoms":7,"unknown":1})
    )
    .is_err());
    let r = omnimd_native::cuda::tiled_resources(
        &medium(),
        &CudaTiling {
            max_atoms_per_frame: 64,
            max_frames: 32,
            max_atoms: 1024,
        },
    )
    .unwrap();
    assert!(r.effective_max_atoms >= 64 && r.effective_max_atoms < 1024);
    assert!(r.estimated_scalar_terms_at_capacity <= 50_000_000);
    assert!((r.effective_max_atoms as u64 + 1) * 466 * r.maximum_center_terms > 50_000_000);
}

#[test]
#[ignore = "requires an authorized exclusive GPU window"]
fn tiled_gradients_grouping_reuse_adam_and_failure_atomicity() {
    use omnimd_native::full_training::CudaTiling;
    let m = model();
    let d = dataset(&m);
    let layout = ParameterLayout::new(&m).unwrap();
    let mut maximum = 0.0_f64;
    let mut bitwise = true;
    let mut observations = Vec::new();
    for caps in [
        CudaTiling {
            max_atoms_per_frame: 64,
            max_frames: 1,
            max_atoms: 64,
        },
        CudaTiling {
            max_atoms_per_frame: 64,
            max_frames: 2,
            max_atoms: 7,
        },
        CudaTiling {
            max_atoms_per_frame: 64,
            max_frames: 32,
            max_atoms: 1024,
        },
    ] {
        let mut reference =
            CudaFullTrainer::new_with_gradient(m.clone(), 0, CudaGradient::Reverse).unwrap();
        let mut tiled = CudaFullTrainer::new_tiled(m.clone(), 0, &caps).unwrap();
        let mut c = config(1);
        c.cuda_gradient = CudaGradient::Reverse;
        c.cuda_tiling = Some(caps.clone());
        // Grow, shrink, reorder and vary species; include self-images and a partial tail.
        for (turn, indices) in [&[0, 1, 2][..], &[2][..], &[1, 0][..], &[2, 1, 0][..]]
            .into_iter()
            .enumerate()
        {
            let mut ref_gradient = vec![0.; layout.parameters.len()];
            let mut objective = 0.;
            // E-only, F-only, V-only and mixed E/F/V.
            let mut loss = c.loss();
            if turn < 3 {
                loss.energy_weight = if turn == 0 { 1. } else { 0. };
                loss.force_weight = if turn == 1 { 1. } else { 0. };
            }
            let virial = if turn < 2 {
                None
            } else {
                c.virial_loss.as_ref()
            };
            for &i in indices {
                let (value, g) = reference
                    .frame_gradient_with_virial(&d.frames[i], &loss, virial, indices.len())
                    .unwrap();
                objective += value;
                for (a, b) in ref_gradient.iter_mut().zip(g) {
                    *a += b;
                }
            }
            let (value, g) = tiled
                .batch_gradient_with_virial(&d, indices, &loss, virial)
                .unwrap();
            assert_eq!(value, objective);
            bitwise &= g == ref_gradient;
            maximum = maximum.max(vector_close(&g, &ref_gradient, 2e-12));
            let profile = tiled.last_profile();
            assert_eq!(profile.frames, indices.len());
            assert_eq!(profile.input_transfer_calls, 2 * profile.tiles);
            assert!(profile.input_transfer_bytes > 0);
            assert_eq!(profile.kernel_launches, profile.tiles * 4);
            if turn == 0 && caps.max_frames == 2 {
                assert_eq!(profile.tiles, 2);
                assert_eq!(profile.max_frames_per_launch, 2);
                assert_eq!(profile.max_atoms_per_launch, 7);
            }
            assert!(profile.peak_host_tile_payload_bytes <= 32 * 1024 * 1024);
            assert!(profile.peak_device_buffer_bytes <= 64 * 1024 * 1024);
            observations.push(serde_json::to_value(profile).unwrap());
            let mut reference_config = c.clone();
            reference_config.cuda_tiling = None;
            reference
                .adam_step(&ref_gradient, &reference_config)
                .unwrap();
            tiled.adam_step(&g, &c).unwrap();
            maximum = maximum.max(vector_close(
                &layout.values(reference.model()),
                &layout.values(tiled.model()),
                2e-12,
            ));
            maximum = maximum.max(vector_close(
                reference.moments().0,
                tiled.moments().0,
                2e-12,
            ));
            maximum = maximum.max(vector_close(
                reference.moments().1,
                tiled.moments().1,
                2e-12,
            ));
        }
    }
    let mut broken = dataset(&m);
    broken.frames[1].frame.positions[1] = broken.frames[1].frame.positions[0];
    let mut gpu = CudaFullTrainer::new_tiled(
        m.clone(),
        0,
        &CudaTiling {
            max_atoms_per_frame: 64,
            max_frames: 1,
            max_atoms: 64,
        },
    )
    .unwrap();
    let before = serde_json::to_vec(gpu.model()).unwrap();
    let first = gpu.moments().0.to_vec();
    let second = gpu.moments().1.to_vec();
    let c = config(1);
    assert!(gpu
        .batch_gradient_with_virial(&broken, &[0, 1], &c.loss(), c.virial_loss.as_ref())
        .is_err());
    assert_eq!(gpu.last_profile().tiles, 1);
    assert_eq!(before, serde_json::to_vec(gpu.model()).unwrap());
    assert_eq!(first, gpu.moments().0);
    assert_eq!(second, gpu.moments().1);
    assert!(gpu
        .adam_step(&vec![0.; 112], &c)
        .unwrap_err()
        .0
        .contains("earlier failure"));
    assert!(gpu
        .batch_gradient_with_virial(&d, &[0], &c.loss(), None)
        .unwrap_err()
        .0
        .contains("earlier failure"));
    println!(
        "{}",
        serde_json::json!({"tiled_max_absolute_gradient_parameter_moment_difference":maximum,"raw_gradients_bitwise_equal":bitwise,"late_tile_failure_committed_state_unchanged":true,"profiles":observations})
    );
}

#[test]
#[ignore = "requires an authorized exclusive GPU window"]
fn tiled_aggregate_1024_atom_capacity_and_partial_tail() {
    use omnimd_native::full_training::CudaTiling;
    let m = model();
    let mut d = dataset(&m);
    let record = pair_record(&m, 32);
    d.frames = (0..33)
        .map(|i| {
            let mut r = record.clone();
            r.index = i;
            r.id = format!("capacity/{i}");
            r
        })
        .collect();
    let c = config(1);
    let mut reference =
        CudaFullTrainer::new_with_gradient(m.clone(), 0, CudaGradient::Reverse).unwrap();
    let mut tiled = CudaFullTrainer::new_tiled(
        m,
        0,
        &CudaTiling {
            max_atoms_per_frame: 64,
            max_frames: 32,
            max_atoms: 1024,
        },
    )
    .unwrap();
    let mut expected = vec![0.; 112];
    let mut objective = 0.;
    for r in &d.frames {
        let (v, g) = reference
            .frame_gradient_with_virial(r, &c.loss(), c.virial_loss.as_ref(), 33)
            .unwrap();
        objective += v;
        for (a, b) in expected.iter_mut().zip(g) {
            *a += b;
        }
    }
    let indices = (0..33).collect::<Vec<_>>();
    let (value, g) = tiled
        .batch_gradient_with_virial(&d, &indices, &c.loss(), c.virial_loss.as_ref())
        .unwrap();
    assert_eq!(value, objective);
    assert_eq!(g, expected);
    let p = tiled.last_profile();
    assert_eq!(p.tiles, 2);
    assert_eq!(p.max_frames_per_launch, 32);
    assert_eq!(p.max_atoms_per_launch, 1024);
    assert!(p.peak_device_buffer_bytes <= 64 * 1024 * 1024);
    assert!(p.peak_host_tile_payload_bytes <= 32 * 1024 * 1024);
    println!(
        "{}",
        serde_json::json!({"aggregate_1024_atom_capacity_bitwise_exact":true,"profile":p})
    );
}

#[test]
fn device_loss_configuration_and_budget_guards_without_driver() {
    use omnimd_native::full_training::{CudaLoss, CudaTiling};
    let m = model();
    let d = dataset(&m);
    let mut c = config(1);
    assert!(serde_json::to_value(&c).unwrap().get("cuda_loss").is_none());
    c.cuda_loss = CudaLoss::Gpu;
    assert!(Checkpoint::start(&m, &d, &c).is_err());
    c.cuda_tiling = Some(CudaTiling {
        max_atoms_per_frame: 64,
        max_frames: 16,
        max_atoms: 512,
    });
    assert!(Checkpoint::start(&m, &d, &c).is_err());
    c.cuda_gradient = CudaGradient::Reverse;
    let s = Checkpoint::start(&m, &d, &c).unwrap();
    assert_eq!(s.schema, "omnimd-native-full-checkpoint-v8");
    assert!(s.validate_for_device(&d, &c, TrainingDevice::Cpu).is_err());
    s.validate_for_device(&d, &c, TrainingDevice::Cuda { visible_device: 0 })
        .unwrap();
    let mut changed = c.clone();
    changed.cuda_loss = CudaLoss::Cpu;
    assert!(s
        .validate_for_device(&d, &changed, TrainingDevice::Cuda { visible_device: 0 })
        .is_err());
    let caps = c.cuda_tiling.as_ref().unwrap();
    for m in [m, medium()] {
        let old = omnimd_native::cuda::tiled_resources(&m, caps).unwrap();
        let new = omnimd_native::cuda::device_loss_resources(&m, caps).unwrap();
        assert!(new.effective_max_atoms <= old.effective_max_atoms);
        assert!(new.device_loss.as_ref().unwrap().extra_device_bytes > 0);
        assert!(new.device_at_capacity.peak_device_buffer_bytes <= 64 * 1024 * 1024);
        assert!(new.host_tile_payload_at_capacity_bytes <= 32 * 1024 * 1024);
        assert!(new.estimated_scalar_terms_at_capacity <= 50_000_000);
    }
}
#[test]
#[ignore = "requires an authorized exclusive GPU window"]
fn device_loss_predictions_seeds_and_cached_gradients() {
    use omnimd_native::full_training::{CudaLoss, CudaTiling};
    let caps = CudaTiling {
        max_atoms_per_frame: 64,
        max_frames: 32,
        max_atoms: 1024,
    };
    let mut max_prediction = 0.0_f64;
    let mut max_seed = 0.0_f64;
    let mut max_gradient = 0.0_f64;
    for m in [model(), medium()] {
        let d = dataset(&m);
        let mut gpu = CudaFullTrainer::new_device_loss(m.clone(), 0, &caps).unwrap();
        gpu.set_device_loss_diagnostics(true);
        let mut reference = CudaFullTrainer::new_tiled(m.clone(), 0, &caps).unwrap();
        for mode in 0..4 {
            let mut c = config(1);
            c.cuda_gradient = CudaGradient::Reverse;
            c.cuda_tiling = Some(caps.clone());
            c.cuda_loss = CudaLoss::Gpu;
            if mode < 3 {
                c.energy_weight = if mode == 0 { 1.0 } else { 0.0 };
                c.force_weight = if mode == 1 { 1.0 } else { 0.0 };
                if mode < 2 {
                    c.virial_loss = None;
                }
            }
            // Same physical coordinate locations across different frames remain independent.
            let indices = [0, 1, 2];
            let (value, g) = gpu
                .batch_gradient_with_virial(&d, &indices, &c.loss(), c.virial_loss.as_ref())
                .unwrap();
            let (expected, rg) = reference
                .batch_gradient_with_virial(&d, &indices, &c.loss(), c.virial_loss.as_ref())
                .unwrap();
            close(value, expected, 2e-11);
            max_gradient = max_gradient.max(vector_close(&g, &rg, 2e-11));
            let diag = gpu.device_loss_diagnostics().unwrap();
            let mut start = 0;
            let mut total_objective = 0.;
            for (i, r) in d.frames.iter().enumerate() {
                let pred = evaluate(&m, &r.frame);
                let n = r.frame.positions.len();
                let nf = n as f64;
                let (es, fs) = (
                    (c.energy_weight / 3.).sqrt() / nf / c.energy_scale_ev_per_atom,
                    (c.force_weight / 3.).sqrt() / (3. * nf).sqrt() / c.force_scale_ev_per_angstrom,
                );
                let vs = c.virial_loss.as_ref().map_or(0., |v| {
                    (v.weight / 3. / 9.).sqrt() / nf / v.scale_ev_per_atom
                });
                assert_eq!(
                    &diag.frame_bounds[i * 2..i * 2 + 2],
                    &[start as i32, (start + n) as i32]
                );
                let ep = &diag.frame_predictions[i * 10..i * 10 + 10];
                let seed = &diag.frame_seeds[i * 11..i * 11 + 11];
                max_prediction = max_prediction.max(vector_close(
                    ep,
                    &[
                        vec![pred.energy],
                        pred.virial.into_iter().flatten().collect(),
                    ]
                    .concat(),
                    2e-12,
                ));
                let de = pred.energy - r.labels.energy.unwrap();
                let a = 2. * es * es * de;
                max_seed = max_seed.max((a - seed[0]).abs());
                close(a, seed[0], 2e-11);
                let mut objective = (de * es).powi(2);
                for atom in 0..n {
                    for b in 0..3 {
                        let index = (start + atom) * 3 + b;
                        let x = pred.forces[atom][b];
                        let df = x - r.labels.forces.as_ref().unwrap()[atom][b];
                        close(x, diag.forces[index], 2e-12);
                        max_prediction = max_prediction.max((x - diag.forces[index]).abs());
                        let v = 2. * fs * fs * df;
                        close(v, diag.force_seeds[index], 2e-11);
                        max_seed = max_seed.max((v - diag.force_seeds[index]).abs());
                        objective += (df * fs).powi(2);
                    }
                }
                for k in 0..9 {
                    let dv = pred.virial[k / 3][k % 3] - r.labels.virial.unwrap()[k / 3][k % 3];
                    let v = if vs > 0. { 2. * vs * vs * dv } else { 0. };
                    close(v, seed[1 + k], 2e-11);
                    max_seed = max_seed.max((v - seed[1 + k]).abs());
                    if vs > 0. {
                        objective += (dv * vs).powi(2);
                    }
                }
                close(objective, seed[10], 2e-11);
                total_objective += objective;
                start += n;
            }
            close(total_objective, value, 2e-11);
            assert_eq!(gpu.last_profile().device_prediction_frames, 3);
            assert_eq!(
                gpu.last_profile().input_transfer_calls,
                2 * gpu.last_profile().tiles
            );
            assert_eq!(gpu.last_profile().prediction_and_seeds_ms, 0.);
        }
        // Reuse buffers with fewer atoms/species and disabled virial; no stale seeds.
        let c = config(1);
        let a = gpu
            .batch_gradient_with_virial(&d, &[2], &c.loss(), None)
            .unwrap();
        let b = reference
            .batch_gradient_with_virial(&d, &[2], &c.loss(), None)
            .unwrap();
        close(a.0, b.0, 2e-11);
        max_gradient = max_gradient.max(vector_close(&a.1, &b.1, 2e-11));
        assert!(gpu.device_loss_diagnostics().unwrap().frame_seeds[1..10]
            .iter()
            .all(|v| *v == 0.));
    }
    println!(
        "{}",
        serde_json::json!({"device_loss_prediction_max_abs":max_prediction,"device_loss_seed_max_abs":max_seed,"device_loss_gradient_max_abs":max_gradient})
    );
}

#[test]
#[ignore = "requires an authorized exclusive GPU window"]
fn device_loss_analytic_embedding_path_zero_loss_and_failure() {
    use omnimd_native::full_training::{CudaLoss, CudaTiling};
    use omnimd_native::{Network, Normalization};
    let mut m = model();
    m.type_map = vec!["O".into()];
    m.atom_energy_bias = vec![0.2];
    m.descriptor.sel = vec![1];
    m.descriptor.normalization = vec![Normalization {
        mean: [0.; 4],
        std: [1.; 4],
    }];
    let weight = 0.4;
    let bias = 0.1;
    let alpha = 0.7;
    m.descriptor.embedding = vec![vec![Network {
        layers: vec![Layer {
            weights: vec![vec![weight]],
            bias: vec![bias],
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
    let radius = delta.iter().map(|x| x * x).sum::<f64>().sqrt();
    let g = (weight / radius + bias).tanh();
    let e = 4. * alpha * g * g / (radius * radius);
    let derivative =
        -8. * alpha * (g * weight * (1. - g * g) / radius.powi(4) + g * g / radius.powi(3));
    let force0 = delta.map(|x| derivative * x / radius);
    let v = std::array::from_fn::<_, 3, _>(|a| {
        std::array::from_fn::<_, 3, _>(|b| -delta[a] * force0[b])
    });
    let frame = Frame {
        positions: vec![[0.; 3], delta],
        species: vec!["O".into(); 2],
        cell: None,
        units: m.units,
    };
    let mut d = dataset(&model());
    d.header.type_map = m.type_map.clone();
    d.frames = vec![DatasetFrame {
        index: 0,
        id: "analytic/0".into(),
        frame,
        labels: Labels {
            energy: Some(e + 0.2),
            forces: Some(vec![force0.map(|x| x + 0.03), force0.map(|x| -x - 0.03)]),
            virial: Some(v.map(|r| r.map(|x| x + 0.02))),
        },
    }];
    let caps = CudaTiling {
        max_atoms_per_frame: 64,
        max_frames: 4,
        max_atoms: 64,
    };
    let mut c = config(1);
    c.cuda_gradient = CudaGradient::Reverse;
    c.cuda_tiling = Some(caps.clone());
    c.cuda_loss = CudaLoss::Gpu;
    let mut gpu = CudaFullTrainer::new_device_loss(m.clone(), 0, &caps).unwrap();
    gpu.set_device_loss_diagnostics(true);
    let (_, gradient) = gpu
        .batch_gradient_with_virial(&d, &[0], &c.loss(), c.virial_loss.as_ref())
        .unwrap();
    let diag = gpu.device_loss_diagnostics().unwrap();
    close(diag.frame_predictions[0], e, 1e-13);
    vector_close(
        &diag.forces,
        &[force0.to_vec(), force0.map(|x| -x).to_vec()].concat(),
        1e-13,
    );
    vector_close(
        &diag.frame_predictions[1..10],
        &v.into_iter().flatten().collect::<Vec<_>>(),
        1e-13,
    );
    let layout = ParameterLayout::new(&m).unwrap();
    let values = layout.values(&m);
    let mut fd_max = 0.0_f64;
    for k in 0..values.len() {
        let h = 1e-5;
        let mut plus = m.clone();
        let mut minus = m.clone();
        let mut x = values.clone();
        x[k] += h;
        layout.apply(&mut plus, &x).unwrap();
        x[k] -= 2. * h;
        layout.apply(&mut minus, &x).unwrap();
        let fd = (gradient_fn(&plus, &d.frames[0], &c) - gradient_fn(&minus, &d.frames[0], &c))
            / (2. * h);
        close(fd, gradient[k], 2e-7);
        fd_max = fd_max.max((fd - gradient[k]).abs());
    }
    // Use device predictions as labels to require exactly zero residual/seed/direction.
    d.frames[0].labels.energy = Some(diag.frame_predictions[0]);
    d.frames[0].labels.forces = Some(
        diag.forces
            .chunks_exact(3)
            .map(|r| [r[0], r[1], r[2]])
            .collect(),
    );
    d.frames[0].labels.virial = Some(std::array::from_fn(|a| {
        std::array::from_fn(|b| diag.frame_predictions[1 + 3 * a + b])
    }));
    let (z, gz) = gpu
        .batch_gradient_with_virial(&d, &[0], &c.loss(), c.virial_loss.as_ref())
        .unwrap();
    assert_eq!(z, 0.);
    assert!(gz.iter().all(|x| *x == 0.));
    assert!(gpu
        .device_loss_diagnostics()
        .unwrap()
        .directions
        .iter()
        .all(|x| *x == 0.));
    // Overflow on a later tile must leave the pre-update model/moments untouched.
    let mut broken = dataset(&model());
    broken.frames[1].labels.energy = Some(1e308);
    let mut failing = CudaFullTrainer::new_device_loss(
        model(),
        0,
        &CudaTiling {
            max_atoms_per_frame: 64,
            max_frames: 1,
            max_atoms: 64,
        },
    )
    .unwrap();
    let before = serde_json::to_vec(failing.model()).unwrap();
    let moment = failing.moments().0.to_vec();
    assert!(failing
        .batch_gradient_with_virial(&broken, &[0, 1], &c.loss(), c.virial_loss.as_ref())
        .unwrap_err()
        .0
        .contains("nonfinite CUDA"));
    assert_eq!(failing.last_profile().tiles, 1);
    assert_eq!(before, serde_json::to_vec(failing.model()).unwrap());
    assert_eq!(moment, failing.moments().0);
    assert!(failing
        .adam_step(&vec![0.; 112], &c)
        .unwrap_err()
        .0
        .contains("earlier failure"));
    println!(
        "{}",
        serde_json::json!({"nonconstant_embedding_analytic_EFV_passed":true,"finite_difference_gradient_max_abs":fd_max,"exact_zero_loss_directions_gradients":true,"later_tile_nonfinite_update_not_committed":true})
    );
}
fn gradient_fn(m: &Model, r: &DatasetFrame, c: &FullConfig) -> f64 {
    gradient(m, r, c, 1).0
}

#[test]
#[ignore = "requires an authorized exclusive GPU window"]
fn device_loss_large_tile_reuse_clipped_updates_and_boundaries() {
    use omnimd_native::full_training::{CudaLoss, CudaTiling};
    let m = model();
    let layout = ParameterLayout::new(&m).unwrap();
    let mut d = dataset(&m);
    d.frames = (0..33)
        .map(|i| {
            let mut r = pair_record(&m, 32);
            r.index = i;
            r.id = format!("capacity/{i}");
            r
        })
        .collect();
    let caps = CudaTiling {
        max_atoms_per_frame: 64,
        max_frames: 32,
        max_atoms: 1024,
    };
    let mut gpu = CudaFullTrainer::new_device_loss(m.clone(), 0, &caps).unwrap();
    let mut reference = CudaFullTrainer::new_tiled(m, 0, &caps).unwrap();
    let mut c = config(1);
    c.cuda_gradient = CudaGradient::Reverse;
    c.cuda_tiling = Some(caps);
    c.cuda_loss = CudaLoss::Gpu;
    c.gradient_clip_norm = Some(0.01);
    let mut maximum = 0.0_f64;
    for indices in [(0..33).collect::<Vec<_>>(), vec![2], vec![3, 0, 2]] {
        let (a, mut g) = gpu
            .batch_gradient_with_virial(&d, &indices, &c.loss(), c.virial_loss.as_ref())
            .unwrap();
        let (b, mut h) = reference
            .batch_gradient_with_virial(&d, &indices, &c.loss(), c.virial_loss.as_ref())
            .unwrap();
        close(a, b, 2e-11);
        maximum = maximum.max(vector_close(&g, &h, 2e-11));
        if indices.len() == 33 {
            assert_eq!(gpu.last_profile().max_atoms_per_launch, 1024);
            assert_eq!(gpu.last_profile().tiles, 2);
        }
        for vector in [&mut g, &mut h] {
            let norm = vector.iter().fold(0.0_f64, |a, v| a.hypot(*v));
            assert!(norm > 0.01);
            for x in vector {
                *x *= 0.01 / norm;
            }
        }
        gpu.adam_step(&g, &c).unwrap();
        let mut old = c.clone();
        old.cuda_loss = CudaLoss::Cpu;
        reference.adam_step(&h, &old).unwrap();
        maximum = maximum.max(vector_close(
            &layout.values(gpu.model()),
            &layout.values(reference.model()),
            2e-11,
        ));
        maximum = maximum.max(vector_close(gpu.moments().0, reference.moments().0, 2e-11));
        maximum = maximum.max(vector_close(gpu.moments().1, reference.moments().1, 2e-11));
    }
    println!(
        "{}",
        serde_json::json!({"device_loss_capacity_1024_grow_shrink_and_clipped_adam_max_abs":maximum})
    );
}
