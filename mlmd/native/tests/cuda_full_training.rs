#![cfg(feature = "cuda")]
use omnimd_ml_contract::{Frame, Potential};
use omnimd_native::{
    cuda::{validate_full_model, CudaFullTrainer},
    full_gradient::{frame_gradient, ParameterLayout},
    full_training::{train_full, train_full_with_device, Checkpoint, FullConfig, FullExecution},
    training::{Split, TrainConfig, TrainingDevice},
    training_data::{Dataset, DatasetFrame, DatasetHeader, Labels, VirialConvention},
    Model, NativePotential, Residual,
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
fn cpu_checkpoint_schema_and_backend_guards_need_no_driver() {
    let (m, d) = data(4);
    let c = config(1);
    let initial = Checkpoint::start(&m, &d, &c).unwrap();
    let json = serde_json::to_value(&initial).unwrap();
    assert!(json.get("execution").is_none());
    let restored: Checkpoint = serde_json::from_value(json).unwrap();
    assert_eq!(restored.execution, FullExecution::CpuReferenceV1);
    let cpu = train_full(restored, &d, &c, |_| Ok(())).unwrap().checkpoint;
    assert!(cpu
        .validate_for_device(&d, &c, TrainingDevice::Cuda { visible_device: 0 })
        .unwrap_err()
        .0
        .contains("backend/device changed"));
    let mut exhausted = cpu.clone();
    exhausted.step = 1_000_000_000;
    assert!(exhausted
        .validate_for_device(&d, &c, TrainingDevice::Cuda { visible_device: 0 })
        .unwrap_err()
        .0
        .contains("global step"));
    let mut gpu = cpu.clone();
    gpu.schema = "omnimd-native-full-checkpoint-v2".into();
    gpu.execution = FullExecution::CudaHyperdualF64V1 {
        visible_device: 0,
        device_info: "test".into(),
        driver_version: 0,
        program_sha256: "test".into(),
    };
    assert!(gpu
        .validate_for_device(&d, &c, TrainingDevice::Cpu)
        .unwrap_err()
        .0
        .contains("backend/device changed"));
    assert!(gpu
        .validate_for_device(&d, &c, TrainingDevice::Cuda { visible_device: 1 })
        .unwrap_err()
        .0
        .contains("backend/device changed"));
    let mut malformed = gpu;
    malformed.schema = "omnimd-native-full-checkpoint-v1".into();
    assert!(malformed
        .validate_for_device(&d, &c, TrainingDevice::Cuda { visible_device: 0 })
        .unwrap_err()
        .0
        .contains("schema"));
    // Each separate bound is legal, but the Cartesian product is rejected on CPU.
    let (mut expensive, record) = boundary_model();
    expensive.descriptor.sel = vec![64];
    let mut extra = expensive.descriptor.embedding[0][0].layers[0].clone();
    extra.weights = vec![vec![0.01; 32]; 32];
    extra.residual = Residual::Identity;
    expensive.descriptor.embedding[0][0].layers.push(extra);
    validate_full_model(&expensive).unwrap();
    assert!(omnimd_native::cuda::full_work_estimate(&expensive, 64)
        .unwrap_err()
        .0
        .contains("work budget"));
    let mut large = record.clone();
    large.frame.positions = vec![[0.0; 3]; 64];
    large.frame.species = vec!["O".into(); 64];
    large.labels.forces = Some(vec![[0.0; 3]; 64]);
    let mut frames = vec![large.clone(), large];
    frames[1].index = 1;
    frames[1].id = "boundary/1".into();
    let bounded = Dataset {
        header: DatasetHeader {
            schema: "omnimd-dataset-jsonl-v1".into(),
            type_map: vec!["O".into()],
            units: expensive.units,
            virial_convention: VirialConvention::MinusDEpsilonRow,
            source: "CPU guard".into(),
            selected_sets: vec![],
        },
        frames,
        source_sha256: BTreeMap::new(),
        dataset_sha256: "guard".into(),
    };
    let mut cfg = c.clone();
    cfg.split = Split::Explicit {
        validation_ids: vec!["boundary/1".into()],
    };
    let fresh = Checkpoint::start(&expensive, &bounded, &cfg).unwrap();
    assert!(fresh
        .validate_for_device(&bounded, &cfg, TrainingDevice::Cuda { visible_device: 0 })
        .unwrap_err()
        .0
        .contains("work budget"));
    let mut oversized = m.clone();
    oversized.descriptor.sel = vec![33, 32];
    assert!(validate_full_model(&oversized)
        .unwrap_err()
        .0
        .contains("64 neighbor slots"));
}
fn max_diff(a: &[f64], b: &[f64], tolerance: f64) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut max = 0.0_f64;
    for (a, b) in a.iter().zip(b) {
        let e = (a - b).abs();
        max = max.max(e);
        assert!(e <= tolerance * (1.0 + a.abs()), "{a} vs {b}: {e}");
    }
    max
}
// Joint boundary: embedding width=descriptor input width=32, still <2048 parameters.
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
#[test]
#[ignore = "requires an authorized exclusive GPU window"]
fn all_parameter_mixed_gradients_adam_and_exact_resume() {
    let (mut m, d) = data(4);
    let layout = ParameterLayout::new(&m).unwrap();
    let mut values = layout.values(&m);
    values[0] = 0.0;
    values[1] = 1.0;
    layout.apply(&mut m, &values).unwrap();
    let mut gpu = CudaFullTrainer::new(m.clone(), 0).unwrap();
    let mut gradient_max = 0.0_f64;
    let mut fd_max = 0.0_f64;
    for (we, wf) in [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
        let mut loss = config(1).loss();
        loss.energy_weight = we;
        loss.force_weight = wf;
        for record in &d.frames[..2] {
            let (objective_gpu, g) = gpu.frame_gradient(record, &loss, 3).unwrap();
            let (objective_cpu, reference) = frame_gradient(&m, record, &loss, 3).unwrap();
            assert_eq!(objective_gpu, objective_cpu);
            gradient_max = gradient_max.max(max_diff(&reference, &g, 2e-10));
            if we == 1.0 && wf == 1.0 {
                for p in 0..values.len() {
                    let h = 1e-5;
                    let mut plus = m.clone();
                    let mut minus = m.clone();
                    let mut v = values.clone();
                    v[p] += h;
                    layout.apply(&mut plus, &v).unwrap();
                    v[p] -= 2.0 * h;
                    layout.apply(&mut minus, &v).unwrap();
                    let fd = (objective(&plus, record, &loss) - objective(&minus, record, &loss))
                        / (2.0 * h * 3.0);
                    fd_max = fd_max.max((fd - g[p]).abs());
                    assert!(
                        (fd - g[p]).abs() < 3e-6 * (1.0 + fd.abs()),
                        "parameter {p}: GPU {} FD {fd}",
                        g[p]
                    );
                }
            }
        }
    }
    let loss = config(1).loss();
    let (_, g1) = gpu.frame_gradient(&d.frames[0], &loss, 1).unwrap();
    let (_, mut g2) = gpu.frame_gradient(&d.frames[0], &loss, 2).unwrap();
    for x in &mut g2 {
        *x *= 2.0;
    }
    max_diff(&g1, &g2, 1e-11);
    let mut zero = d.frames[0].clone();
    let exact = NativePotential::new(m.clone())
        .unwrap()
        .eval(&zero.frame)
        .unwrap();
    zero.labels.energy = Some(exact.energy);
    zero.labels.forces = Some(exact.forces);
    let (objective, zero_gradient) = gpu.frame_gradient(&zero, &loss, 1).unwrap();
    assert_eq!(objective, 0.0);
    assert!(zero_gradient.iter().all(|x| *x == 0.0));
    let (boundary, record) = boundary_model();
    let mut edge = CudaFullTrainer::new(boundary.clone(), 0).unwrap();
    let (_, a) = edge.frame_gradient(&record, &loss, 1).unwrap();
    let (_, b) = frame_gradient(&boundary, &record, &loss, 1).unwrap();
    let boundary_max = max_diff(&a, &b, 2e-10);
    // Axis>1 exercises all descriptor factors independently of the width boundary.
    let mut axis = m.clone();
    axis.descriptor.axis_neuron = 2;
    for net in &mut axis.fitting {
        let rows = net.layers[0].weights.clone();
        net.layers[0].weights.extend(rows);
    }
    let mut axis_gpu = CudaFullTrainer::new(axis.clone(), 0).unwrap();
    let (_, a) = axis_gpu.frame_gradient(&d.frames[1], &loss, 1).unwrap();
    let (_, b) = frame_gradient(&axis, &d.frames[1], &loss, 1).unwrap();
    let axis_max = max_diff(&a, &b, 2e-10);
    // GPU Adam also matches the CPU independent two-step oracle, with clipping active.
    let mut c = config(2);
    c.gradient_clip_norm = Some(1e-4);
    let state = Checkpoint::start(&m, &d, &c).unwrap();
    let cpu = train_full(state.clone(), &d, &c, |_| Ok(())).unwrap();
    let trained = train_full_with_device(
        state.clone(),
        &d,
        &c,
        TrainingDevice::Cuda { visible_device: 0 },
        |_| Ok(()),
    )
    .unwrap();
    let update_max = max_diff(
        &layout.values(&cpu.checkpoint.model),
        &layout.values(&trained.checkpoint.model),
        2e-10,
    );
    let moment_max = max_diff(
        &cpu.checkpoint.first_moment,
        &trained.checkpoint.first_moment,
        1e-12,
    )
    .max(max_diff(
        &cpu.checkpoint.second_moment,
        &trained.checkpoint.second_moment,
        1e-12,
    ));
    let mut c5 = c.clone();
    c5.steps = 5;
    let continuous = train_full_with_device(
        state,
        &d,
        &c5,
        TrainingDevice::Cuda { visible_device: 0 },
        |_| Ok(()),
    )
    .unwrap();
    let resumed: Checkpoint =
        serde_json::from_slice(&serde_json::to_vec(&trained.checkpoint).unwrap()).unwrap();
    let mut c3 = c.clone();
    c3.steps = 3;
    let split = train_full_with_device(
        resumed.clone(),
        &d,
        &c3,
        TrainingDevice::Cuda { visible_device: 0 },
        |_| Ok(()),
    )
    .unwrap();
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
    assert_eq!(continuous.checkpoint.step, split.checkpoint.step);
    let mut tampered = resumed;
    if let FullExecution::CudaHyperdualF64V1 { program_sha256, .. } = &mut tampered.execution {
        *program_sha256 = "invalid".into();
    }
    assert!(train_full_with_device(
        tampered,
        &d,
        &c3,
        TrainingDevice::Cuda { visible_device: 0 },
        |_| Ok(())
    )
    .err()
    .unwrap()
    .0
    .contains("compiled-program"));
    // Explicit numerical rejection: host snapshot stays valid, instance stops.
    let before = serde_json::to_vec(gpu.model()).unwrap();
    let error = gpu.adam_step(&vec![1e308; values.len()], &c).unwrap_err();
    assert!(error.0.contains("nonfinite CUDA Adam"));
    assert_eq!(before, serde_json::to_vec(gpu.model()).unwrap());
    assert!(gpu
        .frame_gradient(&d.frames[0], &loss, 1)
        .unwrap_err()
        .0
        .contains("earlier failure"));
    println!(
        "{}",
        serde_json::json!({"scope":"CUDA all neural parameter E/F mixed derivatives and Adam; CPU loss, geometry, reduction and assessment",
        "parameters":values.len(),"gradient_max_abs_vs_cpu_reverse_ad":gradient_max,"gradient_max_abs_vs_finite_difference":fd_max,
        "width32_gradient_max_abs":boundary_max,"axis2_gradient_max_abs":axis_max,"two_step_parameter_max_abs":update_max,
        "two_step_moment_max_abs":moment_max,"same_device_serialized_resume_exact":true,"cpu_v1_gpu_v2_checkpoint_guard":true,
        "nonfinite_update_preserves_host_snapshot_and_poisons":true,"execution":continuous.report.execution,
        "kernel_resources":edge.kernel_resources(),"continuous_report":continuous.report})
    );
}

#[test]
#[ignore = "requires an authorized exclusive GPU window"]
fn bounded_64_atom_gradient_parity_and_timing() {
    use std::time::Instant;
    let m = model();
    let layout = ParameterLayout::new(&m).unwrap();
    let mut teacher = m.clone();
    let values: Vec<_> = layout
        .values(&m)
        .iter()
        .enumerate()
        .map(|(i, x)| x + 0.12 * ((i + 1) as f64 * 0.63).sin())
        .collect();
    layout.apply(&mut teacher, &values).unwrap();
    let mut positions = Vec::new();
    let mut species = Vec::new();
    for i in 0..32 {
        let base = [
            (i % 4) as f64 * 6.0,
            ((i / 4) % 4) as f64 * 6.0,
            (i / 16) as f64 * 6.0,
        ];
        positions.push(base);
        species.push("O".into());
        positions.push([
            base[0] + 1.1 + 0.1 * (i as f64).sin(),
            base[1] + 0.2,
            base[2] + 0.1,
        ]);
        species.push("H".into());
    }
    let frame = Frame {
        positions,
        species,
        cell: None,
        units: m.units,
    };
    let e = NativePotential::new(teacher).unwrap().eval(&frame).unwrap();
    let record = DatasetFrame {
        index: 0,
        id: "synthetic-separated-pairs/0".into(),
        frame,
        labels: Labels {
            energy: Some(e.energy),
            forces: Some(e.forces),
            virial: Some(e.virial),
        },
    };
    let loss = config(1).loss();
    let startup = Instant::now();
    let mut gpu = CudaFullTrainer::new(m.clone(), 0).unwrap();
    let startup_ms = startup.elapsed().as_secs_f64() * 1000.;
    let (co, cg) = frame_gradient(&m, &record, &loss, 1).unwrap();
    let (go, gg) = gpu.frame_gradient(&record, &loss, 1).unwrap();
    assert_eq!(co, go);
    let maximum = max_diff(&cg, &gg, 1e-10);
    let repeats = 5;
    let start = Instant::now();
    for _ in 0..repeats {
        std::hint::black_box(frame_gradient(&m, &record, &loss, 1).unwrap());
    }
    let cpu_ms = start.elapsed().as_secs_f64() * 1000.;
    let start = Instant::now();
    for _ in 0..repeats {
        std::hint::black_box(gpu.frame_gradient(&record, &loss, 1).unwrap());
    }
    let cuda_ms = start.elapsed().as_secs_f64() * 1000.;
    println!(
        "{}",
        serde_json::json!({"atoms":64,"parameters":layout.parameters.len(),"repeats":repeats,
        "max_abs_gradient":maximum,"startup_ms":startup_ms,"warmed_cpu_total_ms":cpu_ms,"warmed_cuda_total_ms":cuda_ms,
        "cpu_over_cuda_ratio":cpu_ms/cuda_ms,"execution":gpu.execution(),"kernel_resources":gpu.kernel_resources(),
        "timing_scope":"same fixed 64-atom synthetic frame, E/F full gradient; includes CPU inference/loss, geometry, transfers, kernel and stable reduction; excludes initialization, Adam and checkpoint IO; one bounded run, no production scaling claim"})
    );
}
