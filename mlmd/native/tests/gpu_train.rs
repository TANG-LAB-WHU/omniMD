//! The GPU se_e2_a trainer: prediction and loss against the inference path,
//! the loss gradient against finite differences, batching, Adam against a
//! host reference, and a short optimization. Skips without a CUDA driver
//! unless OMNIMD_REQUIRE_GPU=1.
#![cfg(feature = "gpu")]
mod common;
use common::{device, frame, model, values};
use omnimd_ml_contract::{Frame, Potential};
use omnimd_native::gpu_model::GpuPotential;
use omnimd_native::gpu_train::{FrameLabels, GpuTrainer, LossWeights};

struct Labeled {
    frame: Frame,
    energy: f64,
    forces: Vec<[f64; 3]>,
    virial: [[f64; 3]; 3],
}

impl Labeled {
    fn labels(&self) -> FrameLabels<'_> {
        FrameLabels {
            energy: Some(self.energy),
            forces: Some(&self.forces),
            virial: Some(self.virial),
        }
    }
}

/// A frame with labels a fixed distance from the model's own prediction, so
/// every loss term has an O(1) residual.
fn labeled(trainer: &mut GpuTrainer<f64>, seed: u64) -> Labeled {
    let lattice = [[7.0, 0.0, 0.0], [0.5, 7.0, 0.0], [0.0, 0.3, 7.2]];
    let frame = frame(lattice, 3, 2, true, seed);
    let none = FrameLabels {
        energy: None,
        forces: None,
        virial: None,
    };
    let weights = LossWeights {
        energy: 0.0,
        force: 0.0,
        virial: 0.0,
    };
    let (_, prediction) = trainer.accumulate(&frame, &none, &weights, 1.0).unwrap();
    let noise = values(frame.positions.len() * 3 + 9, seed + 1);
    let forces = prediction
        .forces
        .iter()
        .enumerate()
        .map(|(i, f)| std::array::from_fn(|a| f[a] + 0.05 * noise[3 * i + a]))
        .collect();
    let offset = frame.positions.len() * 3;
    let virial = std::array::from_fn(|a| {
        std::array::from_fn(|b| prediction.virial[a][b] + 0.3 * noise[offset + 3 * a + b])
    });
    Labeled {
        frame,
        energy: prediction.energy + 0.37,
        forces,
        virial,
    }
}

const ALL: LossWeights = LossWeights {
    energy: 0.7,
    force: 30.0,
    virial: 0.4,
};

#[test]
fn prediction_and_loss_match_the_inference_path() {
    let Some(device) = device() else { return };
    let model = model(2, vec![40, 80], 4.0, 21);
    let mut trainer = GpuTrainer::<f64>::new(&model, device.clone()).unwrap();
    let data = labeled(&mut trainer, 31);
    let (terms, prediction) = trainer
        .accumulate(&data.frame, &data.labels(), &ALL, 1.0)
        .unwrap();

    let mut reference = GpuPotential::<f64>::new(&model, device.clone()).unwrap();
    let expected = reference.eval(&data.frame).unwrap();
    assert!((prediction.energy - expected.energy).abs() < 1e-10);
    let n = data.frame.positions.len() as f64;
    let (mut l2_force, mut l2_virial) = (0.0, 0.0);
    for (i, f) in expected.forces.iter().enumerate() {
        for a in 0..3 {
            assert!((prediction.forces[i][a] - f[a]).abs() < 1e-10);
            l2_force += (f[a] - data.forces[i][a]).powi(2) / (3.0 * n);
        }
    }
    for a in 0..3 {
        for b in 0..3 {
            assert!((prediction.virial[a][b] - expected.virial[a][b]).abs() < 1e-9);
            l2_virial += (expected.virial[a][b] - data.virial[a][b]).powi(2) / 9.0;
        }
    }
    // DeePMD's ener loss: p_e dE²/N + p_f mean(dF²) + p_v mean(dV²)/N.
    let l2_energy = (expected.energy - data.energy).powi(2);
    let loss = ALL.energy * l2_energy / n + ALL.force * l2_force + ALL.virial * l2_virial / n;
    assert!(
        (terms.loss - loss).abs() < 1e-10 * loss,
        "{} vs {loss}",
        terms.loss
    );
    assert!((terms.rmse_energy_per_atom() - 0.37 / n).abs() < 1e-10);
    assert!((terms.rmse_force() - l2_force.sqrt()).abs() < 1e-12);
    assert!((terms.rmse_virial_per_atom() - l2_virial.sqrt() / n).abs() < 1e-12);

    // Exporting the untouched parameters gives back the same model.
    let exported = trainer.model().unwrap();
    let mut again = GpuPotential::<f64>::new(&exported, device).unwrap();
    assert_eq!(again.eval(&data.frame).unwrap().energy, expected.energy);
    assert_eq!(
        trainer.parameters().unwrap().len(),
        trainer.parameter_count()
    );
}

#[test]
fn loss_gradient_matches_finite_differences() {
    let Some(device) = device() else { return };
    let mut trainer = GpuTrainer::<f64>::new(&model(2, vec![40, 80], 4.0, 22), device).unwrap();
    let data = labeled(&mut trainer, 32);
    let theta = trainer.parameters().unwrap();
    // Each loss term alone, so a wrong term is named, then all together.
    let cases = [
        (
            "energy",
            LossWeights {
                energy: 0.7,
                force: 0.0,
                virial: 0.0,
            },
        ),
        (
            "force",
            LossWeights {
                energy: 0.0,
                force: 30.0,
                virial: 0.0,
            },
        ),
        (
            "virial",
            LossWeights {
                energy: 0.0,
                force: 0.0,
                virial: 0.4,
            },
        ),
        ("all", ALL),
    ];
    for (label, weights) in cases {
        trainer.set_parameters(&theta).unwrap();
        trainer.zero_gradient().unwrap();
        trainer
            .accumulate(&data.frame, &data.labels(), &weights, 1.0)
            .unwrap();
        let gradient = trainer.gradient().unwrap();
        let scale = gradient.iter().fold(0.0_f64, |m, g| m.max(g.abs()));
        assert!(scale > 1e-6, "{label}: gradient is zero");
        let h = 1e-5;
        let mut checked = 0;
        // Every 37th parameter: all embedding and fitting nets, weights,
        // biases and timesteps.
        for k in (0..theta.len()).step_by(37) {
            let mut loss = |shift: f64| {
                let mut moved = theta.clone();
                moved[k] += shift;
                trainer.set_parameters(&moved).unwrap();
                trainer
                    .accumulate(&data.frame, &data.labels(), &weights, 1.0)
                    .unwrap()
                    .0
                    .loss
            };
            let numeric = (loss(h) - loss(-h)) / (2.0 * h);
            assert!(
                (gradient[k] - numeric).abs() < 1e-6 * scale + 1e-5 * numeric.abs(),
                "{label}: parameter {k}: {} vs {numeric} (scale {scale})",
                gradient[k]
            );
            checked += 1;
        }
        assert!(checked > 100, "{label}: only {checked} parameters checked");
    }
}

/// Neighbors of every atom within `rcut`, per neighbor type (types by first
/// appearance of their name, which is not the model's order but enough to
/// tell full slots from empty ones here).
fn neighbor_counts(frame: &Frame, rcut: f64) -> Vec<[usize; 2]> {
    let lattice = frame.cell.unwrap();
    let kind = |i: usize| usize::from(frame.species[i] != frame.species[0]);
    (0..frame.positions.len())
        .map(|i| {
            let mut counts = [0; 2];
            for j in 0..frame.positions.len() {
                for image in 0..27 {
                    let shift = [image / 9, image / 3 % 3, image % 3].map(|k| k as f64 - 1.0);
                    let d: [f64; 3] = std::array::from_fn(|c| {
                        frame.positions[j][c] - frame.positions[i][c]
                            + (0..3).map(|a| shift[a] * lattice[a][c]).sum::<f64>()
                    });
                    let r = d.iter().map(|x| x * x).sum::<f64>().sqrt();
                    if r > 1e-9 && r < rcut {
                        counts[kind(j)] += 1;
                    }
                }
            }
            counts
        })
        .collect()
}

/// The embedding nets run on one row per neighbor and one shared row for the
/// empty slots of each pair of types. With few slots some atoms fill theirs
/// (no shared row to use) or have more neighbors than slots (the list is
/// truncated and its slots are counted one by one); both must give the
/// prediction of the inference path and the gradient of the loss.
#[test]
fn full_and_truncated_neighbor_slots_train_correctly() {
    let Some(device) = device() else { return };
    // The frame `labeled` makes for this seed, and its fewest and most
    // neighbors of one type around one atom.
    let lattice = [[7.0, 0.0, 0.0], [0.5, 7.0, 0.0], [0.0, 0.3, 7.2]];
    let counts = neighbor_counts(&frame(lattice, 3, 2, true, 33), 4.0);
    let fewest = *counts.iter().flatten().min().unwrap();
    let most = *counts.iter().flatten().max().unwrap();
    assert!(fewest >= 3 && most >= fewest + 4, "{fewest}..{most}");
    let middle = (fewest + most) / 2;
    // Slots per neighbor type: never full, always full, and in between.
    for sel in [
        vec![most + 3, most + 5],
        vec![fewest - 1, fewest - 2],
        vec![middle, middle + 1],
    ] {
        let model = model(2, sel.clone(), 4.0, 23);
        let mut trainer = GpuTrainer::<f64>::new(&model, device.clone()).unwrap();
        let data = labeled(&mut trainer, 33);
        let (_, prediction) = trainer
            .accumulate(&data.frame, &data.labels(), &ALL, 1.0)
            .unwrap();
        let expected = GpuPotential::<f64>::new(&model, device.clone())
            .unwrap()
            .eval(&data.frame)
            .unwrap();
        assert!(
            (prediction.energy - expected.energy).abs() < 1e-10,
            "{sel:?}"
        );
        for (ours, theirs) in prediction.forces.iter().zip(&expected.forces) {
            for a in 0..3 {
                assert!((ours[a] - theirs[a]).abs() < 1e-10, "{sel:?}");
            }
        }
        let theta = trainer.parameters().unwrap();
        trainer.zero_gradient().unwrap();
        trainer
            .accumulate(&data.frame, &data.labels(), &ALL, 1.0)
            .unwrap();
        let gradient = trainer.gradient().unwrap();
        let scale = gradient.iter().fold(0.0_f64, |m, g| m.max(g.abs()));
        let (h, mut checked) = (1e-5, 0);
        for k in (0..theta.len()).step_by(61) {
            let mut loss = |shift: f64| {
                let mut moved = theta.clone();
                moved[k] += shift;
                trainer.set_parameters(&moved).unwrap();
                trainer
                    .accumulate(&data.frame, &data.labels(), &ALL, 1.0)
                    .unwrap()
                    .0
                    .loss
            };
            let numeric = (loss(h) - loss(-h)) / (2.0 * h);
            assert!(
                (gradient[k] - numeric).abs() < 1e-6 * scale + 1e-5 * numeric.abs(),
                "{sel:?}: parameter {k}: {} vs {numeric} (scale {scale})",
                gradient[k]
            );
            checked += 1;
        }
        assert!(checked > 60, "{sel:?}: only {checked} parameters checked");
    }
}

/// A frame seen before reuses its neighbor list; the same numbers come out,
/// and a frame that differs in one coordinate is not taken for it.
#[test]
fn a_frame_seen_before_is_not_prepared_again() {
    let Some(device) = device() else { return };
    let model = model(2, vec![40, 80], 4.0, 24);
    let mut trainer = GpuTrainer::<f64>::new(&model, device.clone()).unwrap();
    let data = labeled(&mut trainer, 34);
    // `labeled` evaluated the frame once already.
    assert_eq!(trainer.cached_frames(), 1);
    let mut gradient = |trainer: &mut GpuTrainer<f64>, frame: &Frame| {
        trainer.zero_gradient().unwrap();
        let (terms, _) = trainer
            .accumulate(frame, &data.labels(), &ALL, 1.0)
            .unwrap();
        (terms.loss, trainer.gradient().unwrap())
    };
    let (loss, kept) = gradient(&mut trainer, &data.frame);
    assert_eq!(trainer.cached_frames(), 1);
    // The same frame without the cache.
    let mut fresh = GpuTrainer::<f64>::new(&model, device).unwrap();
    fresh.set_frame_cache(0);
    let (fresh_loss, afresh) = gradient(&mut fresh, &data.frame);
    assert_eq!(fresh.cached_frames(), 0);
    assert!((loss - fresh_loss).abs() < 1e-12 * loss.abs());
    let scale = kept.iter().fold(0.0_f64, |m, g| m.max(g.abs()));
    for (a, b) in kept.iter().zip(&afresh) {
        assert!((a - b).abs() < 1e-11 * scale, "{a} vs {b}");
    }
    // One coordinate moved: another frame, with another loss.
    let mut moved = data.frame.clone();
    moved.positions[5][1] += 0.01;
    let (moved_loss, _) = gradient(&mut trainer, &moved);
    assert_eq!(trainer.cached_frames(), 2);
    assert!((moved_loss - loss).abs() > 1e-6 * loss.abs());
    // And the first one still gives its own.
    let (again, _) = gradient(&mut trainer, &data.frame);
    assert!((again - loss).abs() < 1e-12 * loss.abs());
    assert_eq!(trainer.cached_frames(), 2);
    // Without room nothing more is kept.
    trainer.set_frame_cache(0);
    let mut third = data.frame.clone();
    third.positions[0][0] += 0.02;
    gradient(&mut trainer, &third);
    assert_eq!(trainer.cached_frames(), 2);
}

#[test]
fn batches_average_and_adam_matches_a_host_reference() {
    let Some(device) = device() else { return };
    let mut trainer = GpuTrainer::<f64>::new(&model(2, vec![40, 80], 4.0, 23), device).unwrap();
    let (first, second) = (labeled(&mut trainer, 33), labeled(&mut trainer, 34));
    let single = |trainer: &mut GpuTrainer<f64>, data: &Labeled| {
        trainer.zero_gradient().unwrap();
        trainer
            .accumulate(&data.frame, &data.labels(), &ALL, 1.0)
            .unwrap();
        trainer.gradient().unwrap()
    };
    let (g1, g2) = (single(&mut trainer, &first), single(&mut trainer, &second));
    trainer.zero_gradient().unwrap();
    for data in [&first, &second] {
        trainer
            .accumulate(&data.frame, &data.labels(), &ALL, 0.5)
            .unwrap();
    }
    let batch = trainer.gradient().unwrap();
    let scale = batch.iter().fold(0.0_f64, |m, g| m.max(g.abs()));
    for k in 0..batch.len() {
        assert!((batch[k] - 0.5 * (g1[k] + g2[k])).abs() < 1e-11 * scale);
    }

    // Two Adam updates with that gradient, against torch.optim.Adam's rule.
    let mut expected = trainer.parameters().unwrap();
    let (mut m, mut v) = (vec![0.0; batch.len()], vec![0.0; batch.len()]);
    let lr = 1e-3;
    for step in 1..=2 {
        trainer.adam_step(lr).unwrap();
        for k in 0..batch.len() {
            m[k] = 0.9 * m[k] + 0.1 * batch[k];
            v[k] = 0.999 * v[k] + 0.001 * batch[k] * batch[k];
            let (mh, vh) = (
                m[k] / (1.0 - 0.9_f64.powi(step)),
                v[k] / (1.0 - 0.999_f64.powi(step)),
            );
            expected[k] -= lr * mh / (vh.sqrt() + 1e-8);
        }
    }
    assert_eq!(trainer.optimizer_steps(), 2);
    let got = trainer.parameters().unwrap();
    for k in 0..got.len() {
        assert!((got[k] - expected[k]).abs() < 1e-12, "parameter {k}");
    }
    assert!(trainer.adam_step(f64::NAN).is_err());
}

/// Labels from a teacher model: exactly learnable, unlike [`labeled`].
fn taught(teacher: &mut GpuPotential<f64>, seed: u64) -> Labeled {
    let lattice = [[7.0, 0.0, 0.0], [0.5, 7.0, 0.0], [0.0, 0.3, 7.2]];
    let frame = frame(lattice, 3, 2, true, seed);
    let reference = teacher.eval(&frame).unwrap();
    Labeled {
        frame,
        energy: reference.energy,
        forces: reference.forces,
        virial: reference.virial,
    }
}

#[test]
fn training_fits_a_teacher_in_both_precisions() {
    let Some(device) = device() else { return };
    // The student starts from different random networks than the teacher.
    let mut teacher =
        GpuPotential::<f64>::new(&model(2, vec![40, 80], 4.0, 24), device.clone()).unwrap();
    let data = [taught(&mut teacher, 35), taught(&mut teacher, 36)];
    let student = model(2, vec![40, 80], 4.0, 25);
    fn run<P: omnimd_gpu::Scalar>(trainer: &mut GpuTrainer<P>, data: &[Labeled]) -> (f64, f64) {
        let mut history = Vec::new();
        for _ in 0..300 {
            trainer.zero_gradient().unwrap();
            let mut loss = 0.0;
            for item in data {
                let (terms, _) = trainer
                    .accumulate(&item.frame, &item.labels(), &ALL, 1.0 / data.len() as f64)
                    .unwrap();
                loss += terms.loss / data.len() as f64;
            }
            history.push(loss);
            trainer.adam_step(2e-3).unwrap();
        }
        (history[0], *history.last().unwrap())
    }
    let mut double = GpuTrainer::<f64>::new(&student, device.clone()).unwrap();
    let (start, end) = run(&mut double, &data);
    eprintln!("float64: {start} -> {end}");
    assert!(end < 0.1 * start, "float64: {start} -> {end}");
    // The exported model predicts what the trainer reported.
    let mut potential = GpuPotential::<f64>::new(&double.model().unwrap(), device.clone()).unwrap();
    let (terms, prediction) = double
        .accumulate(&data[0].frame, &data[0].labels(), &ALL, 0.0)
        .unwrap();
    let exported = potential.eval(&data[0].frame).unwrap();
    assert!((exported.energy - prediction.energy).abs() < 1e-9);
    assert!(terms.loss < start);

    let mut single = GpuTrainer::<f32>::new(&student, device).unwrap();
    let (start, end) = run(&mut single, &data);
    eprintln!("float32: {start} -> {end}");
    assert!(end < 0.1 * start, "float32: {start} -> {end}");
}

#[test]
fn a_saved_optimizer_state_continues_the_same_run() {
    let Some(device) = device() else { return };
    let mut teacher =
        GpuPotential::<f64>::new(&model(2, vec![40, 80], 4.0, 26), device.clone()).unwrap();
    let data = taught(&mut teacher, 37);
    let student = model(2, vec![40, 80], 4.0, 27);
    let step = |trainer: &mut GpuTrainer<f64>| {
        trainer.zero_gradient().unwrap();
        trainer
            .accumulate(&data.frame, &data.labels(), &ALL, 1.0)
            .unwrap();
        trainer.adam_step(1e-3).unwrap();
    };
    let mut straight = GpuTrainer::<f64>::new(&student, device.clone()).unwrap();
    assert_eq!(straight.optimizer_state().unwrap().step, 0);
    for _ in 0..6 {
        step(&mut straight);
    }

    let mut first = GpuTrainer::<f64>::new(&student, device.clone()).unwrap();
    for _ in 0..3 {
        step(&mut first);
    }
    let (snapshot, state) = (first.model().unwrap(), first.optimizer_state().unwrap());
    assert_eq!(state.step, 3);
    // A new trainer from the exported model and the saved state.
    let mut second = GpuTrainer::<f64>::new(&snapshot, device.clone()).unwrap();
    second.set_optimizer_state(&state).unwrap();
    for _ in 0..3 {
        step(&mut second);
    }
    assert_eq!(second.optimizer_steps(), 6);
    let (a, b) = (straight.parameters().unwrap(), second.parameters().unwrap());
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    assert!(worst < 1e-11, "restart differs by {worst}");

    // Without the optimizer state the continuation is a different run.
    let mut cold = GpuTrainer::<f64>::new(&snapshot, device).unwrap();
    for _ in 0..3 {
        step(&mut cold);
    }
    let c = cold.parameters().unwrap();
    let apart = a
        .iter()
        .zip(&c)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    assert!(apart > 1e-6);
    let mut bad = state.clone();
    bad.m.pop();
    assert!(cold.set_optimizer_state(&bad).is_err());
}
