//! The GPU trainer: prediction and loss against the inference path, the loss
//! gradient against finite differences, batching, Adam against a host
//! reference, and a short optimization; for se_e2_a, for se_e2_r, for
//! models with one embedding net per neighbor type, and for hybrids of such
//! descriptors. Skips without a CUDA driver unless OMNIMD_REQUIRE_GPU=1.
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
    label(trainer, frame(LATTICE, 3, 2, true, seed), seed)
}

const LATTICE: [[f64; 3]; 3] = [[7.0, 0.0, 0.0], [0.5, 7.0, 0.0], [0.0, 0.3, 7.2]];

/// `frame` with labels a fixed distance from the model's own prediction.
fn label(trainer: &mut GpuTrainer<f64>, frame: Frame, seed: u64) -> Labeled {
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
    let lattice = LATTICE;
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
        let periodic = frame(lattice, 3, 2, true, 33);
        slots_case(device.clone(), &model, periodic, &format!("{sel:?}"));
    }
    // An open cluster with a cutoff just past the nearest neighbors: atoms
    // at its surface have few neighbors, some none of one type.
    let cluster = frame(lattice, 3, 2, false, 35);
    slots_case(device, &model(2, vec![6, 6], 2.7, 25), cluster, "cluster");
}

/// The trainer's prediction for `frame` against the inference path, which
/// has an embedding row for every slot, and its loss gradient against
/// finite differences.
fn slots_case(
    device: std::sync::Arc<omnimd_gpu::Device>,
    model: &omnimd_native::Model,
    frame: Frame,
    case: &str,
) {
    let mut trainer = GpuTrainer::<f64>::new(model, device.clone()).unwrap();
    let data = label(&mut trainer, frame, 33);
    let (_, prediction) = trainer
        .accumulate(&data.frame, &data.labels(), &ALL, 1.0)
        .unwrap();
    let expected = GpuPotential::<f64>::new(model, device)
        .unwrap()
        .eval(&data.frame)
        .unwrap();
    assert!(
        (prediction.energy - expected.energy).abs() < 1e-10,
        "{case}"
    );
    for (ours, theirs) in prediction.forces.iter().zip(&expected.forces) {
        for a in 0..3 {
            assert!((ours[a] - theirs[a]).abs() < 1e-10, "{case}");
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
    for k in (0..theta.len()).step_by(31) {
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
            "{case}: parameter {k}: {} vs {numeric} (scale {scale})",
            gradient[k]
        );
        checked += 1;
    }
    assert!(checked > 60, "{case}: only {checked} parameters checked");
}

#[test]
fn a_frame_seen_before_is_not_prepared_again() {
    let Some(device) = device() else { return };
    let model = model(2, vec![40, 80], 4.0, 24);
    let mut trainer = GpuTrainer::<f64>::new(&model, device.clone()).unwrap();
    let data = labeled(&mut trainer, 34);
    // `labeled` evaluated the frame once already.
    assert_eq!(trainer.cached_frames(), 1);
    let gradient = |trainer: &mut GpuTrainer<f64>, frame: &Frame| {
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

/// Float32 networks keep their parameters in float64: at the end of a
/// learning-rate schedule a step is far below the spacing of float32
/// numbers near the parameters and must not be rounded away.
#[test]
fn float32_networks_keep_steps_below_their_own_precision() {
    let Some(device) = device() else { return };
    let model = model(2, vec![40, 80], 4.0, 26);
    let data = labeled(
        &mut GpuTrainer::<f64>::new(&model, device.clone()).unwrap(),
        36,
    );
    let mut trainer = GpuTrainer::<f32>::new(&model, device.clone()).unwrap();
    let before = trainer.parameters().unwrap();
    trainer.zero_gradient().unwrap();
    trainer
        .accumulate(&data.frame, &data.labels(), &ALL, 1.0)
        .unwrap();
    let gradient = trainer.gradient().unwrap();
    // The first Adam step moves every parameter by the learning rate
    // against the sign of its gradient. Most parameters here are of order
    // 0.1 to 1, where float32 numbers are 1e-8 to 1e-7 apart.
    let rate = 1e-10;
    trainer.adam_step(rate).unwrap();
    let after = trainer.parameters().unwrap();
    let mut checked = 0;
    for ((old, new), g) in before.iter().zip(&after).zip(&gradient) {
        if g.abs() > 1e-4 {
            assert!(
                (new - old + g.signum() * rate).abs() < 1e-3 * rate,
                "{old} to {new} with gradient {g}"
            );
            checked += 1;
        }
    }
    assert!(checked > 1000, "only {checked} parameters checked");
    // The networks compute with the float32 rounding of those parameters:
    // a trainer given them afresh predicts the same.
    let none = FrameLabels {
        energy: None,
        forces: None,
        virial: None,
    };
    let (_, kept) = trainer.accumulate(&data.frame, &none, &ALL, 0.0).unwrap();
    let mut fresh = GpuTrainer::<f32>::new(&model, device).unwrap();
    fresh.set_parameters(&after).unwrap();
    assert_eq!(fresh.parameters().unwrap(), after);
    let (_, again) = fresh.accumulate(&data.frame, &none, &ALL, 0.0).unwrap();
    assert!((kept.energy - again.energy).abs() < 1e-9 * kept.energy.abs().max(1.0));
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

/// The loss gradient of the trainer against central differences of its own
/// loss, for every `stride`-th parameter.
fn gradient_is_the_loss_gradient(
    trainer: &mut GpuTrainer<f64>,
    data: &Labeled,
    weights: &LossWeights,
    stride: usize,
    case: &str,
) -> usize {
    let theta = trainer.parameters().unwrap();
    trainer.zero_gradient().unwrap();
    trainer
        .accumulate(&data.frame, &data.labels(), weights, 1.0)
        .unwrap();
    let gradient = trainer.gradient().unwrap();
    let scale = gradient.iter().fold(0.0_f64, |m, g| m.max(g.abs()));
    assert!(scale > 1e-6, "{case}: gradient is zero");
    let (h, mut checked) = (1e-5, 0);
    for k in (0..theta.len()).step_by(stride) {
        let mut loss = |shift: f64| {
            let mut moved = theta.clone();
            moved[k] += shift;
            trainer.set_parameters(&moved).unwrap();
            trainer
                .accumulate(&data.frame, &data.labels(), weights, 1.0)
                .unwrap()
                .0
                .loss
        };
        let numeric = (loss(h) - loss(-h)) / (2.0 * h);
        assert!(
            (gradient[k] - numeric).abs() < 1e-6 * scale + 1e-5 * numeric.abs(),
            "{case}: parameter {k}: {} vs {numeric} (scale {scale})",
            gradient[k]
        );
        checked += 1;
    }
    trainer.set_parameters(&theta).unwrap();
    checked
}

/// The trainer's prediction for a frame against the inference path, and its
/// loss gradient against finite differences for each loss term alone and for
/// all together.
fn trains_like_the_inference_path(
    device: std::sync::Arc<omnimd_gpu::Device>,
    model: &omnimd_native::Model,
    seed: u64,
    case: &str,
) {
    let mut trainer = GpuTrainer::<f64>::new(model, device.clone()).unwrap();
    let data = labeled(&mut trainer, seed);
    let (_, prediction) = trainer
        .accumulate(&data.frame, &data.labels(), &ALL, 1.0)
        .unwrap();
    let expected = GpuPotential::<f64>::new(model, device)
        .unwrap()
        .eval(&data.frame)
        .unwrap();
    assert!(
        (prediction.energy - expected.energy).abs() < 1e-10,
        "{case}: {} against {}",
        prediction.energy,
        expected.energy
    );
    for (ours, theirs) in prediction.forces.iter().zip(&expected.forces) {
        for a in 0..3 {
            assert!((ours[a] - theirs[a]).abs() < 1e-10, "{case}: force");
        }
    }
    for (ours, theirs) in prediction
        .virial
        .iter()
        .flatten()
        .zip(expected.virial.iter().flatten())
    {
        assert!((ours - theirs).abs() < 1e-9, "{case}: virial");
    }
    let only = |energy: f64, force: f64, virial: f64| LossWeights {
        energy,
        force,
        virial,
    };
    for (term, weights) in [
        ("energy", only(0.7, 0.0, 0.0)),
        ("force", only(0.0, 30.0, 0.0)),
        ("virial", only(0.0, 0.0, 0.4)),
        ("all", ALL),
    ] {
        let checked = gradient_is_the_loss_gradient(
            &mut trainer,
            &data,
            &weights,
            23,
            &format!("{case}, {term}"),
        );
        assert!(
            checked > 100,
            "{case}, {term}: only {checked} parameters checked"
        );
    }
}

/// se_e2_r goes through the passes of se_e2_a with the kernels of its own
/// descriptor; and a model of either descriptor may have one embedding net
/// per neighbor type instead of one per pair of types.
#[test]
fn se_e2_r_and_nets_per_neighbor_type_train_like_the_inference_path() {
    let Some(device) = device() else { return };
    let radial = common::radial_model(2, vec![40, 80], 4.0, 41, 16);
    trains_like_the_inference_path(device.clone(), &radial, 42, "se_e2_r");
    trains_like_the_inference_path(
        device.clone(),
        &common::one_sided(common::radial_model(2, vec![40, 80], 4.0, 43, 16)),
        44,
        "se_e2_r, a net per neighbor type",
    );
    trains_like_the_inference_path(
        device.clone(),
        &common::one_sided(model(2, vec![40, 80], 4.0, 45)),
        46,
        "se_e2_a, a net per neighbor type",
    );
    // An embedding wider than a warp of the descriptor kernels.
    trains_like_the_inference_path(
        device.clone(),
        &common::radial_model(2, vec![40, 80], 4.0, 47, 40),
        48,
        "se_e2_r, wide embedding",
    );
    // Slots never full, always full (the list is truncated), in between,
    // and a cluster whose surface atoms lack neighbors of a type.
    let counts = neighbor_counts(&frame(LATTICE, 3, 2, true, 33), 4.0);
    let fewest = *counts.iter().flatten().min().unwrap();
    let most = *counts.iter().flatten().max().unwrap();
    let middle = (fewest + most) / 2;
    for sel in [
        vec![most + 3, most + 5],
        vec![fewest - 1, fewest - 2],
        vec![middle, middle + 1],
    ] {
        let model = common::radial_model(2, sel.clone(), 4.0, 49, 16);
        let periodic = frame(LATTICE, 3, 2, true, 33);
        slots_case(
            device.clone(),
            &model,
            periodic,
            &format!("se_e2_r {sel:?}"),
        );
        slots_case(
            device.clone(),
            &common::one_sided(model),
            frame(LATTICE, 3, 2, true, 33),
            &format!("se_e2_r, a net per neighbor type, {sel:?}"),
        );
    }
    let cluster = frame(LATTICE, 3, 2, false, 35);
    slots_case(
        device,
        &common::radial_model(2, vec![6, 6], 2.7, 50, 16),
        cluster,
        "se_e2_r cluster",
    );
}

fn parameters_of(network: &omnimd_native::Network) -> usize {
    network
        .layers
        .iter()
        .map(|layer| {
            layer.weights.len() * layer.bias.len()
                + layer.bias.len()
                + layer.timestep.as_ref().map_or(0, Vec::len)
        })
        .sum()
}

/// A net that all central types share has each of its parameters once: its
/// gradient is the sum of the gradients the nets of the pairs would get, and
/// an update keeps the rows of the model alike.
#[test]
fn a_net_per_neighbor_type_takes_the_sum_of_the_gradients_of_its_rows() {
    let Some(device) = device() else { return };
    let cases = [
        ("se_e2_a", model(2, vec![40, 80], 4.0, 51)),
        (
            "se_e2_r",
            common::radial_model(2, vec![40, 80], 4.0, 52, 16),
        ),
    ];
    for (name, base) in cases {
        let ntypes = base.type_map.len();
        // The same numbers twice: with the mark, and as nets of their own.
        let shared = common::one_sided(base);
        let mut apart = shared.clone();
        apart.descriptor.type_one_side = false;
        let net = parameters_of(&shared.descriptor.embedding[0][0]);
        let mut tied = GpuTrainer::<f64>::new(&shared, device.clone()).unwrap();
        let mut free = GpuTrainer::<f64>::new(&apart, device.clone()).unwrap();
        assert_eq!(
            tied.parameter_count() + ntypes * (ntypes - 1) * net,
            free.parameter_count(),
            "{name}"
        );
        let data = labeled(&mut free, 53);
        let gradient = |trainer: &mut GpuTrainer<f64>| {
            trainer.zero_gradient().unwrap();
            let (terms, _) = trainer
                .accumulate(&data.frame, &data.labels(), &ALL, 1.0)
                .unwrap();
            (terms.loss, trainer.gradient().unwrap())
        };
        let (loss_tied, of_tied) = gradient(&mut tied);
        let (loss_free, of_free) = gradient(&mut free);
        assert!(
            (loss_tied - loss_free).abs() < 1e-12 * loss_free.abs(),
            "{name}"
        );
        let scale = of_free.iter().fold(0.0_f64, |m, g| m.max(g.abs()));
        for neighbor in 0..ntypes {
            for k in 0..net {
                let sum: f64 = (0..ntypes)
                    .map(|center| of_free[(center * ntypes + neighbor) * net + k])
                    .sum();
                let ours = of_tied[neighbor * net + k];
                assert!(
                    (ours - sum).abs() < 1e-11 * scale,
                    "{name}: net {neighbor} parameter {k}: {ours} against {sum}"
                );
            }
        }
        // The rows do take different gradients: the sum is not one of them.
        let differ = (0..net)
            .filter(|k| (of_free[*k] - of_free[ntypes * net + *k]).abs() > 1e-6 * scale)
            .count();
        assert!(differ > net / 4, "{name}: {differ} of {net}");
        // The fitting nets follow the embedding nets and are not shared.
        let (tied_fit, free_fit) = (&of_tied[ntypes * net..], &of_free[ntypes * ntypes * net..]);
        assert_eq!(tied_fit.len(), free_fit.len());
        for (a, b) in tied_fit.iter().zip(free_fit) {
            assert!(
                (a - b).abs() < 1e-11 * scale,
                "{name}: fitting {a} against {b}"
            );
        }
        // An update: the shared nets move and stay shared; nets of their
        // own part.
        tied.adam_step(1e-3).unwrap();
        free.adam_step(1e-3).unwrap();
        let exported = tied.model().unwrap();
        assert!(exported.descriptor.type_one_side, "{name}");
        assert_eq!(
            exported.descriptor.embedding[1],
            exported.descriptor.embedding[0]
        );
        assert_ne!(
            exported.descriptor.embedding[0],
            shared.descriptor.embedding[0]
        );
        let parted = free.model().unwrap();
        assert_ne!(
            parted.descriptor.embedding[1],
            parted.descriptor.embedding[0]
        );
        // The exported model predicts what its trainer does.
        let (_, prediction) = tied
            .accumulate(&data.frame, &data.labels(), &ALL, 0.0)
            .unwrap();
        let again = GpuPotential::<f64>::new(&exported, device.clone())
            .unwrap()
            .eval(&data.frame)
            .unwrap();
        assert!((again.energy - prediction.energy).abs() < 1e-9, "{name}");
    }
}

/// Float32 networks give the loss and the gradient of float64 ones to their
/// own precision: for se_e2_r, whose kernels lay out their shared memory by
/// the size of a number, and for nets that the central types share.
#[test]
fn float32_networks_follow_float64_ones_with_se_e2_r_and_shared_nets() {
    let Some(device) = device() else { return };
    let cases = [
        (
            "se_e2_r, wide embedding",
            common::radial_model(2, vec![40, 80], 4.0, 71, 40),
        ),
        (
            "se_e2_r, a net per neighbor type",
            common::one_sided(common::radial_model(2, vec![40, 80], 4.0, 72, 16)),
        ),
        (
            "se_e2_a, a net per neighbor type",
            common::one_sided(model(2, vec![40, 80], 4.0, 73)),
        ),
    ];
    for (case, model) in cases {
        let mut double = GpuTrainer::<f64>::new(&model, device.clone()).unwrap();
        let mut single = GpuTrainer::<f32>::new(&model, device.clone()).unwrap();
        assert_eq!(single.parameter_count(), double.parameter_count());
        let data = labeled(&mut double, 74);
        double.zero_gradient().unwrap();
        single.zero_gradient().unwrap();
        let (exact, _) = double
            .accumulate(&data.frame, &data.labels(), &ALL, 1.0)
            .unwrap();
        let (rounded, _) = single
            .accumulate(&data.frame, &data.labels(), &ALL, 1.0)
            .unwrap();
        assert!(
            (rounded.loss - exact.loss).abs() < 1e-5 * exact.loss,
            "{case}: loss {} against {}",
            rounded.loss,
            exact.loss
        );
        let (of_double, of_single) = (double.gradient().unwrap(), single.gradient().unwrap());
        let scale = of_double.iter().fold(0.0_f64, |m, g| m.max(g.abs()));
        let worst = of_double
            .iter()
            .zip(&of_single)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        eprintln!(
            "{case}: float32 loss off by {:.1e}, gradient by {:.1e} of its largest entry",
            (rounded.loss - exact.loss).abs() / exact.loss,
            worst / scale
        );
        assert!(scale > 1e-6, "{case}: gradient is zero");
        // Measured: 7e-8 to 2e-7 of the largest entry.
        assert!(
            worst < 2e-5 * scale,
            "{case}: {worst} against a largest entry of {scale}"
        );
    }
}

/// Mean loss over `data` at the first and after the last of 300 Adam steps.
fn fit<P: omnimd_gpu::Scalar>(trainer: &mut GpuTrainer<P>, data: &[Labeled]) -> (f64, f64) {
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

#[test]
fn se_e2_r_training_fits_a_teacher() {
    let Some(device) = device() else { return };
    let teacher = common::radial_model(2, vec![40, 80], 4.0, 61, 16);
    let mut teacher = GpuPotential::<f64>::new(&teacher, device.clone()).unwrap();
    let data = [taught(&mut teacher, 62), taught(&mut teacher, 63)];
    // Students with other random networks: nets of the pairs in both
    // precisions, and a net per neighbor type.
    let student = common::radial_model(2, vec![40, 80], 4.0, 64, 16);
    let mut double = GpuTrainer::<f64>::new(&student, device.clone()).unwrap();
    let (start, end) = fit(&mut double, &data);
    eprintln!("se_e2_r float64: {start} -> {end}");
    assert!(end < 0.1 * start, "float64: {start} -> {end}");
    let exported = double.model().unwrap();
    assert_eq!(exported.descriptor.kind, "se_e2_r");
    let mut potential = GpuPotential::<f64>::new(&exported, device.clone()).unwrap();
    let (_, prediction) = double
        .accumulate(&data[0].frame, &data[0].labels(), &ALL, 0.0)
        .unwrap();
    let reloaded = potential.eval(&data[0].frame).unwrap();
    assert!((reloaded.energy - prediction.energy).abs() < 1e-9);

    let mut single = GpuTrainer::<f32>::new(&student, device.clone()).unwrap();
    let (start, end) = fit(&mut single, &data);
    eprintln!("se_e2_r float32: {start} -> {end}");
    assert!(end < 0.1 * start, "float32: {start} -> {end}");

    let shared = common::one_sided(common::radial_model(2, vec![40, 80], 4.0, 65, 16));
    let mut tied = GpuTrainer::<f64>::new(&shared, device).unwrap();
    let (start, end) = fit(&mut tied, &data);
    eprintln!("se_e2_r, a net per neighbor type: {start} -> {end}");
    assert!(
        end < 0.1 * start,
        "a net per neighbor type: {start} -> {end}"
    );
    let exported = tied.model().unwrap();
    assert_eq!(
        exported.descriptor.embedding[1],
        exported.descriptor.embedding[0]
    );
}

// Hybrids: the descriptors of a list side by side, each with its own nets,
// and fitting nets that take the components of all of them.

/// The parameters of a model in the order of the trainer's, by what they
/// belong to: the embedding nets of every descriptor, then the fitting nets.
fn parameter_ranges(model: &omnimd_native::Model) -> Vec<(String, std::ops::Range<usize>)> {
    let mut ranges = Vec::new();
    let mut first = 0;
    for (number, d) in model.descriptor.parts().iter().enumerate() {
        // A net that the central types share is there once.
        let nets: Vec<&omnimd_native::Network> = match d.type_one_side {
            true => d.embedding[0].iter().collect(),
            false => d.embedding.iter().flatten().collect(),
        };
        let count: usize = nets.into_iter().map(parameters_of).sum();
        ranges.push((
            format!("embedding nets of descriptor {number} ({})", d.kind),
            first..first + count,
        ));
        first += count;
    }
    let count: usize = model.fitting.iter().map(parameters_of).sum();
    ranges.push(("fitting nets".to_string(), first..first + count));
    ranges
}

/// The loss gradient of the trainer against central differences of its own
/// loss, for `each` parameters of every range, evenly spread over it. No
/// range may be without a gradient: every descriptor reaches the loss.
fn every_range_has_the_loss_gradient(
    trainer: &mut GpuTrainer<f64>,
    data: &Labeled,
    weights: &LossWeights,
    ranges: &[(String, std::ops::Range<usize>)],
    each: usize,
    case: &str,
) {
    let theta = trainer.parameters().unwrap();
    assert_eq!(ranges.last().unwrap().1.end, theta.len(), "{case}");
    trainer.zero_gradient().unwrap();
    trainer
        .accumulate(&data.frame, &data.labels(), weights, 1.0)
        .unwrap();
    let gradient = trainer.gradient().unwrap();
    let scale = gradient.iter().fold(0.0_f64, |m, g| m.max(g.abs()));
    assert!(scale > 1e-6, "{case}: gradient is zero");
    let h = 1e-5;
    for (name, range) in ranges {
        let largest = gradient[range.clone()]
            .iter()
            .fold(0.0_f64, |m, g| m.max(g.abs()));
        // Measured: the embedding nets of a descriptor have gradients of
        // 4e-3 to 0.7 of the largest of the model.
        assert!(
            largest > 1e-3 * scale,
            "{case}: {name}: largest gradient {largest} of {scale}"
        );
        // An error of a descriptor is seen against its own gradients, not
        // against those of the fitting nets.
        let tolerance = 1e-6 * largest.max(1e-2 * scale);
        let (stride, mut checked) = ((range.len() / each).max(1), 0);
        for k in range.clone().step_by(stride) {
            let mut loss = |shift: f64| {
                let mut moved = theta.clone();
                moved[k] += shift;
                trainer.set_parameters(&moved).unwrap();
                trainer
                    .accumulate(&data.frame, &data.labels(), weights, 1.0)
                    .unwrap()
                    .0
                    .loss
            };
            let numeric = (loss(h) - loss(-h)) / (2.0 * h);
            assert!(
                (gradient[k] - numeric).abs() < tolerance + 1e-5 * numeric.abs(),
                "{case}: {name}: parameter {k}: {} vs {numeric} (largest of the range {largest}, of all {scale})",
                gradient[k]
            );
            checked += 1;
        }
        assert!(
            checked >= each.min(range.len()),
            "{case}: {name}: only {checked} parameters checked"
        );
    }
    trainer.set_parameters(&theta).unwrap();
}

/// Neighbors of the atoms of the frame of [`labeled`] within `rcut`: the
/// fewest and the most of one type around one atom.
fn fewest_and_most(seed: u64, rcut: f64) -> (usize, usize) {
    let counts = neighbor_counts(&frame(LATTICE, 3, 2, true, seed), rcut);
    (
        *counts.iter().flatten().min().unwrap(),
        *counts.iter().flatten().max().unwrap(),
    )
}

/// The model with larger weights in the output layer of its fitting nets:
/// the forces of the hybrids of the helpers are of some thousandths of an
/// eV/A else, which says less than it could.
fn stronger(mut model: omnimd_native::Model) -> omnimd_native::Model {
    for net in &mut model.fitting {
        let output = net.layers.last_mut().unwrap();
        for weight in output.weights.iter_mut().flatten() {
            *weight *= 8.0;
        }
    }
    model
}

/// An angular descriptor of the near neighbors and a radial one of all
/// within the cutoff of the list. The first is narrower than a warp of the
/// kernels and has a table of the list of its own (a shorter cutoff, fewer
/// slots); the second is wider than a warp, has one net per neighbor type
/// and reads the list as it is.
fn near_angular_far_radial(seed: u64) -> omnimd_native::Model {
    stronger(common::hybrid_model(
        &[
            model(2, vec![24, 48], 3.2, seed),
            common::one_sided(common::radial_model(2, vec![40, 80], 4.0, seed + 1, 40)),
        ],
        seed + 2,
    ))
}

/// Three descriptors, so that the middle one has components before and
/// after its own: a radial one on the list, a wide angular one with a net
/// per neighbor type and a narrow angular one, both on tables of their own.
fn three_descriptors(seed: u64) -> omnimd_native::Model {
    stronger(common::hybrid_model(
        &[
            common::radial_model(2, vec![40, 80], 4.0, seed, 16),
            common::one_sided(common::model_of_width(
                2,
                vec![30, 60],
                3.6,
                seed + 1,
                40,
                3,
            )),
            model(2, vec![20, 44], 3.0, seed + 2),
        ],
        seed + 3,
    ))
}

/// A hybrid goes through the passes of a model of one descriptor, each
/// stage once for every descriptor: the prediction is that of the inference
/// path, virial included, and the gradient that of the loss, in the nets of
/// every descriptor and in the fitting nets.
#[test]
fn a_hybrid_trains_like_the_inference_path() {
    let Some(device) = device() else { return };
    // Short of slots, for the frame of seed 133: the list holds fewer
    // neighbors of the first type than some atoms have, and each descriptor
    // has fewer slots of one type than the list (so both read tables of
    // their own). The radial one, with the shorter cutoff, fills its slots
    // of the first type around every atom and never those of the second.
    let (fewest_near, most_near) = fewest_and_most(133, 3.2);
    let (fewest, most) = fewest_and_most(133, 4.0);
    assert!(
        fewest_near >= 1 && most_near > fewest_near && most > fewest + 3,
        "{fewest_near} {most_near} {fewest} {most}"
    );
    let middle = (fewest + most) / 2;
    let roomy = (most_near + 2).max(middle + 1);
    let short = stronger(common::hybrid_model(
        &[
            model(2, vec![middle, middle - 1], 4.0, 131),
            common::radial_model(2, vec![fewest_near, roomy], 3.2, 132, 16),
        ],
        134,
    ));
    assert_eq!(short.descriptor.sel, [middle, roomy]);
    // The slots of the list and a shorter cutoff: the angular descriptor
    // still needs a table of its own, in which the neighbors between the
    // two cutoffs are empty slots (its switch is not zero beyond its cutoff
    // unless they are).
    let same_slots = stronger(common::hybrid_model(
        &[
            common::radial_model(2, vec![40, 80], 4.0, 135, 16),
            model(2, vec![40, 80], 3.2, 136),
        ],
        137,
    ));
    let cases = [
        (
            "near angular, far radial",
            near_angular_far_radial(121),
            124,
        ),
        ("three descriptors", three_descriptors(125), 129),
        ("short of slots", short, 133),
        ("the slots of the list, a shorter cutoff", same_slots, 138),
    ];
    let only = |energy: f64, force: f64, virial: f64| LossWeights {
        energy,
        force,
        virial,
    };
    for (case, hybrid, seed) in cases {
        let mut trainer = GpuTrainer::<f64>::new(&hybrid, device.clone()).unwrap();
        let ranges = parameter_ranges(&hybrid);
        assert_eq!(ranges.len(), hybrid.descriptor.list.len() + 1, "{case}");
        assert_eq!(
            ranges.last().unwrap().1.end,
            trainer.parameter_count(),
            "{case}"
        );
        let data = labeled(&mut trainer, seed);
        let (_, prediction) = trainer
            .accumulate(&data.frame, &data.labels(), &ALL, 1.0)
            .unwrap();
        let expected = GpuPotential::<f64>::new(&hybrid, device.clone())
            .unwrap()
            .eval(&data.frame)
            .unwrap();
        assert!(
            (prediction.energy - expected.energy).abs() < 1e-10,
            "{case}: {} against {}",
            prediction.energy,
            expected.energy
        );
        let mut largest = 0.0_f64;
        for (ours, theirs) in prediction.forces.iter().zip(&expected.forces) {
            for a in 0..3 {
                assert!((ours[a] - theirs[a]).abs() < 1e-10, "{case}: force");
                largest = largest.max(theirs[a].abs());
            }
        }
        eprintln!("{case}: largest force {largest:.2e} eV/A");
        assert!(largest > 3e-3, "{case}: forces of {largest} say little");
        for (ours, theirs) in prediction
            .virial
            .iter()
            .flatten()
            .zip(expected.virial.iter().flatten())
        {
            assert!(
                (ours - theirs).abs() < 1e-9,
                "{case}: virial {ours} against {theirs}"
            );
        }
        for (term, weights) in [
            ("energy", only(0.7, 0.0, 0.0)),
            ("force", only(0.0, 30.0, 0.0)),
            ("virial", only(0.0, 0.0, 0.4)),
            ("all", ALL),
        ] {
            every_range_has_the_loss_gradient(
                &mut trainer,
                &data,
                &weights,
                &ranges,
                40,
                &format!("{case}, {term}"),
            );
        }
        // The exported model is the hybrid it was, with every net in its place.
        let exported = trainer.model().unwrap();
        assert_eq!(
            serde_json::to_value(&exported).unwrap(),
            serde_json::to_value(&hybrid).unwrap(),
            "{case}"
        );
    }
}

/// A list of one descriptor is that descriptor: the same parameters in the
/// same order, the same loss and gradient, and the same nets after an update.
#[test]
fn a_hybrid_of_one_descriptor_trains_as_that_descriptor_alone() {
    let Some(device) = device() else { return };
    let cases = [
        ("se_e2_a", model(2, vec![40, 80], 4.0, 141), 141),
        (
            "se_e2_r, a net per neighbor type",
            common::one_sided(common::radial_model(2, vec![40, 80], 4.0, 142, 40)),
            142,
        ),
    ];
    for (case, alone, seed) in cases {
        // The fitting nets of a hybrid of the helpers are those of the model
        // of the same seed.
        let listed = common::hybrid_model(std::slice::from_ref(&alone), seed);
        assert_eq!(listed.fitting, alone.fitting, "{case}");
        let mut plain = GpuTrainer::<f64>::new(&alone, device.clone()).unwrap();
        let mut hybrid = GpuTrainer::<f64>::new(&listed, device.clone()).unwrap();
        assert_eq!(
            plain.parameters().unwrap(),
            hybrid.parameters().unwrap(),
            "{case}"
        );
        let data = labeled(&mut plain, 143);
        let step = |trainer: &mut GpuTrainer<f64>| {
            trainer.zero_gradient().unwrap();
            let (terms, prediction) = trainer
                .accumulate(&data.frame, &data.labels(), &ALL, 1.0)
                .unwrap();
            (terms.loss, prediction, trainer.gradient().unwrap())
        };
        let (loss, prediction, gradient) = step(&mut plain);
        let (listed_loss, listed_prediction, listed_gradient) = step(&mut hybrid);
        assert!((listed_loss - loss).abs() < 1e-12 * loss.abs(), "{case}");
        assert_eq!(listed_prediction.energy, prediction.energy, "{case}");
        for (ours, theirs) in listed_prediction.forces.iter().zip(&prediction.forces) {
            for a in 0..3 {
                assert!((ours[a] - theirs[a]).abs() < 1e-12, "{case}: force");
            }
        }
        let scale = gradient.iter().fold(0.0_f64, |m, g| m.max(g.abs()));
        assert!(scale > 1e-6, "{case}");
        for (k, (ours, theirs)) in listed_gradient.iter().zip(&gradient).enumerate() {
            assert!(
                (ours - theirs).abs() < 1e-11 * scale,
                "{case}: parameter {k}: {ours} against {theirs}"
            );
        }
        plain.adam_step(1e-3).unwrap();
        hybrid.adam_step(1e-3).unwrap();
        let (moved, listed_moved) = (plain.parameters().unwrap(), hybrid.parameters().unwrap());
        for (ours, theirs) in listed_moved.iter().zip(&moved) {
            assert!((ours - theirs).abs() < 1e-12, "{case}: after an update");
        }
        // The exported hybrid is a hybrid still, with the nets of the model
        // of one descriptor in its list.
        let (exported, listed_exported) = (plain.model().unwrap(), hybrid.model().unwrap());
        assert_eq!(listed_exported.format, "omnimd-native-hybrid", "{case}");
        assert_eq!(listed_exported.descriptor.list.len(), 1, "{case}");
        let part = &listed_exported.descriptor.list[0];
        assert_eq!(part.kind, exported.descriptor.kind, "{case}");
        assert_ne!(part.embedding, alone.descriptor.embedding, "{case}");
        let again = GpuTrainer::<f64>::new(&listed_exported, device.clone()).unwrap();
        assert_eq!(again.parameters().unwrap(), listed_moved, "{case}");
        let again = GpuTrainer::<f64>::new(&exported, device.clone()).unwrap();
        assert_eq!(again.parameters().unwrap(), moved, "{case}");
    }
}

/// A frame seen before keeps what every descriptor reads of it, its tables
/// of the neighbor list included, and those count toward the memory the
/// kept frames may take.
#[test]
fn a_hybrid_keeps_the_tables_of_a_frame_it_has_seen() {
    let Some(device) = device() else { return };
    let hybrid = three_descriptors(151);
    let mut trainer = GpuTrainer::<f64>::new(&hybrid, device.clone()).unwrap();
    let data = labeled(&mut trainer, 155);
    assert_eq!(trainer.cached_frames(), 1);
    let gradient = |trainer: &mut GpuTrainer<f64>, frame: &Frame| {
        trainer.zero_gradient().unwrap();
        let (terms, _) = trainer
            .accumulate(frame, &data.labels(), &ALL, 1.0)
            .unwrap();
        (terms.loss, trainer.gradient().unwrap())
    };
    let (loss, kept) = gradient(&mut trainer, &data.frame);
    assert_eq!(trainer.cached_frames(), 1);
    let mut fresh = GpuTrainer::<f64>::new(&hybrid, device.clone()).unwrap();
    fresh.set_frame_cache(0);
    let (fresh_loss, afresh) = gradient(&mut fresh, &data.frame);
    assert_eq!(fresh.cached_frames(), 0);
    assert!((loss - fresh_loss).abs() < 1e-12 * loss.abs());
    let scale = kept.iter().fold(0.0_f64, |m, g| m.max(g.abs()));
    for (a, b) in kept.iter().zip(&afresh) {
        assert!((a - b).abs() < 1e-11 * scale, "{a} vs {b}");
    }
    // Another frame has other tables: its own loss, and the first one
    // still gives its own.
    let mut moved = data.frame.clone();
    moved.positions[5][1] += 0.01;
    let (moved_loss, _) = gradient(&mut trainer, &moved);
    assert_eq!(trainer.cached_frames(), 2);
    assert!((moved_loss - loss).abs() > 1e-6 * loss.abs());
    let (again, _) = gradient(&mut trainer, &data.frame);
    assert!((again - loss).abs() < 1e-12 * loss.abs());

    // What a kept frame holds on the device, in bytes. The list: the type
    // of every atom, of every slot the neighbor, its vector and distance,
    // of every atom the neighbors of each type and the nearest distance.
    let (atoms, types) = (data.frame.positions.len(), 2);
    let list_slots: usize = hybrid.descriptor.sel.iter().sum();
    let list = atoms * (4 + list_slots * (4 + 24 + 8) + types * 4 + 8);
    // Every descriptor: three numbers of its rows per atom and type, and
    // the two that do not read the list as it is their own tables of it.
    let rows = 3 * atoms * types * 3 * 4;
    let tables: usize = hybrid.descriptor.list[1..]
        .iter()
        .map(|d| atoms * d.sel.iter().sum::<usize>() * (4 + 24 + 8))
        .sum();
    assert!(tables > list / 2, "tables of {tables} bytes beside {list}");
    let bytes = list + rows + tables;
    for (room, held) in [(bytes, 1), (bytes - 1, 0)] {
        let mut limited = GpuTrainer::<f64>::new(&hybrid, device.clone()).unwrap();
        limited.set_frame_cache(room);
        gradient(&mut limited, &data.frame);
        assert_eq!(
            limited.cached_frames(),
            held,
            "room for {room} of {bytes} bytes"
        );
    }
}

/// Float32 networks give the loss and the gradient of float64 ones to their
/// precision, and a hybrid with other random networks learns the labels of
/// a hybrid, in both precisions; the trained model keeps every descriptor.
#[test]
fn a_hybrid_trains_in_float32_and_fits_a_teacher() {
    let Some(device) = device() else { return };
    for (case, hybrid) in [
        ("near angular, far radial", near_angular_far_radial(161)),
        ("three descriptors", three_descriptors(165)),
    ] {
        let mut double = GpuTrainer::<f64>::new(&hybrid, device.clone()).unwrap();
        let mut single = GpuTrainer::<f32>::new(&hybrid, device.clone()).unwrap();
        assert_eq!(single.parameter_count(), double.parameter_count());
        let data = labeled(&mut double, 169);
        double.zero_gradient().unwrap();
        single.zero_gradient().unwrap();
        let (exact, _) = double
            .accumulate(&data.frame, &data.labels(), &ALL, 1.0)
            .unwrap();
        let (rounded, _) = single
            .accumulate(&data.frame, &data.labels(), &ALL, 1.0)
            .unwrap();
        assert!(
            (rounded.loss - exact.loss).abs() < 1e-5 * exact.loss,
            "{case}: loss {} against {}",
            rounded.loss,
            exact.loss
        );
        let (of_double, of_single) = (double.gradient().unwrap(), single.gradient().unwrap());
        let scale = of_double.iter().fold(0.0_f64, |m, g| m.max(g.abs()));
        let worst = of_double
            .iter()
            .zip(&of_single)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        eprintln!(
            "{case}: float32 loss off by {:.1e}, gradient by {:.1e} of its largest entry",
            (rounded.loss - exact.loss).abs() / exact.loss,
            worst / scale
        );
        assert!(scale > 1e-6, "{case}: gradient is zero");
        assert!(
            worst < 2e-5 * scale,
            "{case}: {worst} against a largest entry of {scale}"
        );
    }

    let mut teacher =
        GpuPotential::<f64>::new(&near_angular_far_radial(171), device.clone()).unwrap();
    let data = [taught(&mut teacher, 174), taught(&mut teacher, 175)];
    let student = near_angular_far_radial(176);
    let mut double = GpuTrainer::<f64>::new(&student, device.clone()).unwrap();
    let (start, end) = fit(&mut double, &data);
    eprintln!("hybrid float64: {start} -> {end}");
    assert!(end < 0.1 * start, "float64: {start} -> {end}");
    // The trained model: a hybrid of the same descriptors, every one with
    // nets that have moved, the radial one with its rows still alike; it
    // predicts what its trainer does.
    let exported = double.model().unwrap();
    assert_eq!(exported.format, "omnimd-native-hybrid");
    let kinds: Vec<&str> = exported
        .descriptor
        .list
        .iter()
        .map(|d| d.kind.as_str())
        .collect();
    assert_eq!(kinds, ["se_e2_a", "se_e2_r"]);
    for (trained, initial) in exported
        .descriptor
        .list
        .iter()
        .zip(&student.descriptor.list)
    {
        assert_ne!(trained.embedding, initial.embedding, "{}", trained.kind);
        assert_eq!(trained.sel, initial.sel);
        assert_eq!(trained.type_one_side, initial.type_one_side);
    }
    let radial = &exported.descriptor.list[1];
    assert!(radial.type_one_side && radial.embedding[1] == radial.embedding[0]);
    let angular = &exported.descriptor.list[0];
    assert_ne!(angular.embedding[1], angular.embedding[0]);
    assert_ne!(exported.fitting, student.fitting);
    let mut potential = GpuPotential::<f64>::new(&exported, device.clone()).unwrap();
    let (_, prediction) = double
        .accumulate(&data[0].frame, &data[0].labels(), &ALL, 0.0)
        .unwrap();
    let reloaded = potential.eval(&data[0].frame).unwrap();
    assert!((reloaded.energy - prediction.energy).abs() < 1e-9);
    for (ours, theirs) in reloaded.forces.iter().zip(&prediction.forces) {
        for a in 0..3 {
            assert!((ours[a] - theirs[a]).abs() < 1e-9);
        }
    }

    let mut single = GpuTrainer::<f32>::new(&student, device).unwrap();
    let (start, end) = fit(&mut single, &data);
    eprintln!("hybrid float32: {start} -> {end}");
    assert!(end < 0.1 * start, "float32: {start} -> {end}");
}

/// A run of a hybrid that is stopped continues from its exported model and
/// its optimizer state as if it had gone on.
#[test]
fn a_saved_optimizer_state_continues_the_run_of_a_hybrid() {
    let Some(device) = device() else { return };
    let mut teacher = GpuPotential::<f64>::new(&three_descriptors(181), device.clone()).unwrap();
    let data = taught(&mut teacher, 185);
    let student = three_descriptors(186);
    let step = |trainer: &mut GpuTrainer<f64>| {
        trainer.zero_gradient().unwrap();
        trainer
            .accumulate(&data.frame, &data.labels(), &ALL, 1.0)
            .unwrap();
        trainer.adam_step(1e-3).unwrap();
    };
    let mut straight = GpuTrainer::<f64>::new(&student, device.clone()).unwrap();
    for _ in 0..6 {
        step(&mut straight);
    }
    let mut first = GpuTrainer::<f64>::new(&student, device.clone()).unwrap();
    for _ in 0..3 {
        step(&mut first);
    }
    let (snapshot, state) = (first.model().unwrap(), first.optimizer_state().unwrap());
    assert_eq!(state.step, 3);
    assert_eq!(state.m.len(), first.parameter_count());
    let mut second = GpuTrainer::<f64>::new(&snapshot, device.clone()).unwrap();
    second.set_optimizer_state(&state).unwrap();
    for _ in 0..3 {
        step(&mut second);
    }
    let (a, b) = (straight.parameters().unwrap(), second.parameters().unwrap());
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    assert!(worst < 1e-11, "restart differs by {worst}");
    // Without the optimizer state the continuation is another run.
    let mut cold = GpuTrainer::<f64>::new(&snapshot, device).unwrap();
    for _ in 0..3 {
        step(&mut cold);
    }
    let apart = a
        .iter()
        .zip(&cold.parameters().unwrap())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    assert!(apart > 1e-6);
}

/// What the trainer has no arithmetic for is refused by name, a descriptor
/// of a hybrid by its place in the list.
#[test]
fn a_hybrid_with_a_three_body_descriptor_is_refused() {
    let Some(device) = device() else { return };
    let three_body = common::three_body_model(2, vec![20, 30], 3.0, 191, 8);
    let error = GpuTrainer::<f64>::new(&three_body, device.clone())
        .err()
        .unwrap();
    assert_eq!(
        error.to_string(),
        "the GPU trainer supports the se_e2_a and se_e2_r descriptors and hybrids of them, not se_e3"
    );
    let hybrid = common::hybrid_model(&[model(2, vec![40, 80], 4.0, 192), three_body], 193);
    for (name, error) in [
        (
            "float64",
            GpuTrainer::<f64>::new(&hybrid, device.clone()).err(),
        ),
        ("float32", GpuTrainer::<f32>::new(&hybrid, device).err()),
    ] {
        assert_eq!(
            error.unwrap().to_string(),
            "descriptor.list[1]: the GPU trainer supports the se_e2_a and se_e2_r descriptors and hybrids of them, not se_e3",
            "{name}"
        );
    }
}

/// Two atoms in one place have no direction between them: the trainer
/// refuses the frame by name, for a hybrid as for one descriptor, and keeps
/// nothing of it.
#[test]
fn a_frame_with_two_atoms_in_one_place_is_refused() {
    let Some(device) = device() else { return };
    let none = FrameLabels {
        energy: None,
        forces: None,
        virial: None,
    };
    for (case, model) in [
        ("se_e2_a", model(2, vec![40, 80], 4.0, 201)),
        ("hybrid", three_descriptors(202)),
    ] {
        let mut trainer = GpuTrainer::<f64>::new(&model, device.clone()).unwrap();
        let mut frame = frame(LATTICE, 3, 2, true, 206);
        frame.positions[4] = frame.positions[9];
        let error = trainer
            .accumulate(&frame, &none, &ALL, 1.0)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("coincident"), "{case}: {error}");
        assert_eq!(trainer.cached_frames(), 0, "{case}");
    }
}
