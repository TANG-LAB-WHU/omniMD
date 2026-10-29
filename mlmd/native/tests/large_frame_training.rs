#![cfg(feature = "cuda")]
use omnimd_ml_contract::{Evaluation, Frame, Potential};
use omnimd_native::{
    cuda::CudaFullTrainer,
    full_gradient::{frame_gradient_with_virial, ParameterLayout},
    full_training::{Checkpoint, CudaTiling, FullConfig},
    training::TrainingDevice,
    training_data::{Dataset, DatasetFrame, DatasetHeader, Labels, VirialConvention},
    Model, NativePotential,
};
use std::collections::BTreeMap;
fn model() -> Model {
    serde_json::from_str(include_str!("../fixtures/full-tanh-v1.json")).unwrap()
}
fn config() -> FullConfig {
    serde_json::from_value(serde_json::json!({
        "split":{"mode":"explicit","validation_ids":["large/2"]},"steps":2,
        "cuda_gradient":"reverse","cuda_loss":"gpu",
        "cuda_tiling":{"max_frames":4,"max_atoms":1024,"max_atoms_per_frame":256},
        "energy_weight":0.7,"force_weight":0.8,
        "virial_loss":{"weight":1.3,"scale_ev_per_atom":0.4},
        "energy_scale_ev_per_atom":0.3,"force_scale_ev_per_angstrom":0.2,
        "learning_rate":0.003,"beta1":0.9,"beta2":0.999,"epsilon":1e-8,
        "gradient_clip_norm":0.1,"checkpoint_every":1
    }))
    .unwrap()
}
fn base_frame() -> Frame {
    let cell = [[14.4, 0., 0.], [0.4, 7.2, 0.], [0.2, 0.3, 7.2]];
    let mut f = Frame {
        positions: vec![],
        species: vec![],
        cell: Some(cell),
        units: model().units,
    };
    for x in 0..4 {
        for y in 0..2 {
            for z in 0..2 {
                for t in 0..2 {
                    let frac = [x as f64 / 4., y as f64 / 2., z as f64 / 2.];
                    let k = f.positions.len();
                    f.positions.push(std::array::from_fn(|a| {
                        (0..3).map(|b| frac[b] * cell[b][a]).sum::<f64>()
                            + t as f64 * [1.07, 0.21, 0.13][a]
                            + 0.03 * ((k * 3 + a + 1) as f64 * 0.61).sin()
                    }));
                    f.species.push(if t == 0 { "O" } else { "H" }.into());
                }
            }
        }
    }
    f
}
fn replicate(f: &Frame, factors: [usize; 3]) -> Frame {
    let cell = f.cell.unwrap();
    let mut result = f.clone();
    result.positions.clear();
    result.species.clear();
    for x in 0..factors[0] {
        for y in 0..factors[1] {
            for z in 0..factors[2] {
                let count = [x, y, z];
                for (p, s) in f.positions.iter().zip(&f.species) {
                    result.positions.push(std::array::from_fn(|a| {
                        p[a] + (0..3).map(|b| count[b] as f64 * cell[b][a]).sum::<f64>()
                    }));
                    result.species.push(s.clone());
                }
            }
        }
    }
    result.cell = Some(std::array::from_fn(|a| {
        cell[a].map(|x| x * factors[a] as f64)
    }));
    result
}
fn frame83() -> Frame {
    Frame {
        positions: (0..83)
            .map(|i| {
                [
                    (i / 2) as f64 * 3.6 + (i % 2) as f64 * 1.1,
                    (i % 2) as f64 * 0.2,
                    0.1,
                ]
            })
            .collect(),
        species: (0..83)
            .map(|i| if i % 2 == 0 { "O".into() } else { "H".into() })
            .collect(),
        cell: None,
        units: model().units,
    }
}
fn eval(m: &Model, f: &Frame) -> Evaluation {
    NativePotential::new(m.clone()).unwrap().eval(f).unwrap()
}
fn dataset(m: &Model, frames: Vec<Frame>) -> Dataset {
    let layout = ParameterLayout::new(m).unwrap();
    let mut teacher = m.clone();
    let values: Vec<_> = layout
        .values(m)
        .iter()
        .enumerate()
        .map(|(i, v)| v + 0.12 * ((i + 1) as f64 * 0.63).sin())
        .collect();
    layout.apply(&mut teacher, &values).unwrap();
    Dataset {
        header: DatasetHeader {
            schema: "omnimd-dataset-jsonl-v1".into(),
            type_map: m.type_map.clone(),
            units: m.units,
            virial_convention: VirialConvention::MinusDEpsilonRow,
            source: "original periodic larger-frame fixture".into(),
            selected_sets: vec![],
        },
        frames: frames
            .into_iter()
            .enumerate()
            .map(|(i, f)| {
                let e = eval(&teacher, &f);
                DatasetFrame {
                    index: i as u64,
                    id: format!("large/{i}"),
                    frame: f,
                    labels: Labels {
                        energy: Some(e.energy),
                        forces: Some(e.forces),
                        virial: Some(e.virial),
                    },
                }
            })
            .collect(),
        source_sha256: BTreeMap::new(),
        dataset_sha256: "original-large-frame-in-memory".into(),
    }
}
fn close(a: f64, b: f64, t: f64) -> f64 {
    assert!(
        (a - b).abs() <= t * (1. + a.abs().max(b.abs())),
        "{a} vs {b}"
    );
    (a - b).abs()
}
fn compare(a: &[f64], b: &[f64], t: f64) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| close(x, y, t))
        .fold(0., f64::max)
}
fn objective(e: &Evaluation, r: &DatasetFrame, c: &FullConfig) -> f64 {
    let n = r.frame.positions.len() as f64;
    let es = c.energy_weight.sqrt() / n / c.energy_scale_ev_per_atom;
    let fs = (c.force_weight / (3. * n)).sqrt() / c.force_scale_ev_per_angstrom;
    let v = c.virial_loss.as_ref().unwrap();
    let vs = (v.weight / 9.).sqrt() / n / v.scale_ev_per_atom;
    let mut total = ((e.energy - r.labels.energy.unwrap()) * es).powi(2);
    for (x, y) in e
        .forces
        .iter()
        .flatten()
        .zip(r.labels.forces.as_ref().unwrap().iter().flatten())
    {
        total += ((x - y) * fs).powi(2);
    }
    for (x, y) in e
        .virial
        .iter()
        .flatten()
        .zip(r.labels.virial.as_ref().unwrap().iter().flatten())
    {
        total += ((x - y) * vs).powi(2);
    }
    total
}
#[test]
fn large_frame_admission_and_budgets_without_driver() {
    let m = model();
    let c = config();
    let base = base_frame();
    let d = dataset(&m, vec![base.clone(), replicate(&base, [2, 2, 2]), base]);
    let s = Checkpoint::start(&m, &d, &c).unwrap();
    assert_eq!(s.schema, "omnimd-native-full-checkpoint-v9");
    s.validate_for_device(&d, &c, TrainingDevice::Cuda { visible_device: 0 })
        .unwrap();
    assert!(s.validate_for_device(&d, &c, TrainingDevice::Cpu).is_err());
    let cap = c.cuda_tiling.as_ref().unwrap();
    let r = omnimd_native::cuda::device_loss_resources(&m, cap).unwrap();
    assert!(r.effective_max_atoms >= 256);
    assert!(r.estimated_scalar_terms_at_capacity <= 50_000_000);
    assert!(r.device_at_capacity.peak_device_buffer_bytes <= 64 * 1024 * 1024);
    assert!(r.host_tile_payload_at_capacity_bytes <= 32 * 1024 * 1024);
    let old: CudaTiling =
        serde_json::from_value(serde_json::json!({"max_frames":4,"max_atoms":1024})).unwrap();
    assert_eq!(old.max_atoms_per_frame, 64);
    assert!(serde_json::to_value(&old)
        .unwrap()
        .get("max_atoms_per_frame")
        .is_none());
    let mut changed = c.clone();
    changed.cuda_tiling = Some(old);
    assert!(Checkpoint::start(&m, &d, &changed).is_err());
    changed.cuda_tiling.as_mut().unwrap().max_atoms_per_frame = 257;
    assert!(Checkpoint::start(&m, &d, &changed).is_err());
    changed = c.clone();
    changed.cuda_tiling.as_mut().unwrap().max_atoms = 255;
    let s2 = Checkpoint::start(&m, &d, &changed).unwrap();
    assert!(s2
        .validate_for_device(&d, &changed, TrainingDevice::Cuda { visible_device: 0 })
        .is_err());
    // The historical framewise/CPU derivative admission stays at 64.
    assert!(omnimd_native::cuda::full_work_estimate(&m, 128).is_err());
    assert!(
        frame_gradient_with_virial(&m, &d.frames[1], &c.loss(), c.virial_loss.as_ref(), 1).is_err()
    );
    println!(
        "{}",
        serde_json::json!({"large_frame_resources":r,"default_64_and_explicit_bounds_enforced":true})
    );
}
#[test]
#[ignore = "requires an authorized exclusive GPU window"]
fn large_frame_gpu_predictions_gradients_and_preflight() {
    let m = model();
    let c = config();
    let base = base_frame();
    let d = dataset(
        &m,
        vec![
            base.clone(),
            replicate(&base, [2, 2, 1]),
            replicate(&base, [2, 2, 2]),
            frame83(),
        ],
    );
    let cap = c.cuda_tiling.as_ref().unwrap();
    let mut gpu = CudaFullTrainer::new_device_loss(m.clone(), 0, cap).unwrap();
    gpu.set_device_loss_diagnostics(true);
    let mut cpu_loss = CudaFullTrainer::new_tiled(m.clone(), 0, cap).unwrap();
    let (base_obj, base_gradient) =
        frame_gradient_with_virial(&m, &d.frames[0], &c.loss(), c.virial_loss.as_ref(), 1).unwrap();
    let mut max_prediction = 0f64;
    let mut max_gradient = 0f64;
    let mut max_fd = 0f64;
    let mut max_replicated = 0f64;
    for i in [2, 1, 3, 0] {
        // 256 -> 128 -> 83 -> 32 exercises every strided tail and shrinking buffers.
        let r = &d.frames[i];
        let (obj, g) = gpu
            .batch_gradient_with_virial(&d, &[i], &c.loss(), c.virial_loss.as_ref())
            .unwrap();
        let diag = gpu.device_loss_diagnostics().unwrap();
        let pred = eval(&m, &r.frame);
        let mut efv = vec![pred.energy];
        efv.extend(pred.virial.iter().flatten());
        max_prediction = max_prediction.max(compare(&diag.frame_predictions, &efv, 2e-11));
        max_prediction = max_prediction.max(compare(
            &diag.forces,
            &pred.forces.iter().flatten().copied().collect::<Vec<_>>(),
            2e-11,
        ));
        close(obj, objective(&pred, r, &c), 2e-11);
        let (ref_obj, ref_g) = cpu_loss
            .batch_gradient_with_virial(&d, &[i], &c.loss(), c.virial_loss.as_ref())
            .unwrap();
        close(obj, ref_obj, 2e-11);
        max_gradient = max_gradient.max(compare(&g, &ref_g, 3e-10));
        assert!(pred.forces.iter().flatten().any(|f| f.abs() > 1e-5));
        if i < 3 {
            close(obj, base_obj, 3e-10);
            max_replicated = max_replicated.max(compare(&g, &base_gradient, 3e-10));
        }
        if i == 2 {
            let layout = ParameterLayout::new(&m).unwrap();
            let values = layout.values(&m);
            let h = 1e-5;
            for index in [0, 3, 13, 27, 55, 70, 90, 111] {
                let mut plus = m.clone();
                let mut minus = m.clone();
                let mut a = values.clone();
                let mut b = values.clone();
                a[index] += h;
                b[index] -= h;
                layout.apply(&mut plus, &a).unwrap();
                layout.apply(&mut minus, &b).unwrap();
                let fd = (objective(&eval(&plus, &r.frame), r, &c)
                    - objective(&eval(&minus, &r.frame), r, &c))
                    / (2. * h);
                max_fd = max_fd.max(close(g[index], fd, 3e-6));
            }
        }
    }
    let (a, ga) = gpu
        .batch_gradient_with_virial(&d, &[1, 2, 3], &c.loss(), c.virial_loss.as_ref())
        .unwrap();
    let (b, gb) = cpu_loss
        .batch_gradient_with_virial(&d, &[1, 2, 3], &c.loss(), c.virial_loss.as_ref())
        .unwrap();
    close(a, b, 2e-11);
    max_gradient = max_gradient.max(compare(&ga, &gb, 3e-10));
    let profile = gpu.last_profile();
    assert_eq!(profile.max_atoms_per_frame, 256);
    assert_eq!(profile.max_atoms_per_launch, 467);
    let before = ParameterLayout::new(&m).unwrap().values(gpu.model());
    let moments = (gpu.moments().0.to_vec(), gpu.moments().1.to_vec());
    let mut invalid = d;
    invalid.frames[2].frame.positions.push([900., 0., 0.]);
    invalid.frames[2].frame.species.push("O".into());
    invalid.frames[2]
        .labels
        .forces
        .as_mut()
        .unwrap()
        .push([0.; 3]);
    assert!(gpu
        .batch_gradient_with_virial(&invalid, &[0, 2], &c.loss(), c.virial_loss.as_ref())
        .is_err());
    assert_eq!(gpu.last_profile().tiles, 0);
    assert_eq!(gpu.last_profile().peak_device_buffer_bytes, 0);
    assert_eq!(
        before,
        ParameterLayout::new(&m).unwrap().values(gpu.model())
    );
    assert_eq!(moments.0, gpu.moments().0);
    assert_eq!(moments.1, gpu.moments().1);
    assert!(gpu.adam_step(&ga, &c).is_err());
    println!(
        "{}",
        serde_json::json!({"prediction_max_abs":max_prediction,"gradient_vs_cpu_loss_max_abs":max_gradient,
        "normalized_gradient_vs_32_atom_cpu_ad_max_abs":max_replicated,"eight_parameter_cpu_loss_finite_difference_max_abs":max_fd,
        "tested_frame_atoms":[256,128,83,32],"mixed_tile_profile":profile,"late_oversize_rejected_before_any_tile":true,"execution":cpu_loss.execution()})
    );
}
