//! Trains a native se_e2_a model on the omnimd-gpu runtime (device 0).
//!
//! ```text
//! omnimd-native-gpu-train REQUEST.json
//! ```
//!
//! The request names a dataset and a new output file, and carries the
//! `learning_rate`, `loss` and `training` sections of a DeePMD input script
//! with the same meaning (see `GPU_TRAINING.md`). `model` is either the path
//! of a native model to continue from, or the `model` section of a DeePMD
//! input script: then the networks are drawn afresh, and the normalization
//! (`davg`, `dstd`), the energy bias and an `"auto"` `sel` are computed from
//! the dataset.
//!
//! ```json
//! {"model": "/abs/initial.native.json", "dataset": "/abs/train.jsonl",
//!  "output": "/abs/trained.native.json", "precision": "float64",
//!  "learning_rate": {"start_lr": 1e-3, "stop_lr": 3e-8, "decay_steps": 5000},
//!  "loss": {"start_pref_e": 0.02, "limit_pref_e": 1, "start_pref_f": 1000,
//!           "limit_pref_f": 1, "start_pref_v": 0, "limit_pref_v": 0},
//!  "training": {"numb_steps": 100000, "batch_size": 1, "seed": 1,
//!               "disp_freq": 100, "disp_file": "/abs/lcurve.out"}}
//! ```
//!
//! Existing output files are never replaced.
use omnimd_gpu::{Device, Scalar};
use omnimd_ml_contract::Frame;
use omnimd_native::gpu_train::{FrameLabels, GpuTrainer, LossTerms, LossWeights};
use omnimd_native::model_init::{initialize, ModelSpec};
use omnimd_native::stats::{energy_bias, env_mat_stat, neighbor_stat};
use omnimd_native::training_data::{Dataset, DatasetFrame};
use omnimd_native::{Model, NativePotential};
use serde::Deserialize;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Copy, Default, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
enum Precision {
    #[default]
    Float64,
    Float32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LearningRate {
    start_lr: f64,
    stop_lr: f64,
    #[serde(default = "default_decay_steps")]
    decay_steps: u64,
}

fn default_decay_steps() -> u64 {
    5000
}

/// DeePMD's `ener` loss prefactors, with its defaults.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Loss {
    #[serde(default = "pref_e_start")]
    start_pref_e: f64,
    #[serde(default = "one")]
    limit_pref_e: f64,
    #[serde(default = "pref_f_start")]
    start_pref_f: f64,
    #[serde(default = "one")]
    limit_pref_f: f64,
    #[serde(default)]
    start_pref_v: f64,
    #[serde(default)]
    limit_pref_v: f64,
}

fn pref_e_start() -> f64 {
    0.02
}

fn pref_f_start() -> f64 {
    1000.0
}

fn one() -> f64 {
    1.0
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Training {
    numb_steps: u64,
    #[serde(default = "one_frame")]
    batch_size: usize,
    #[serde(default)]
    seed: u64,
    #[serde(default = "default_disp_freq")]
    disp_freq: u64,
    disp_file: Option<PathBuf>,
}

fn one_frame() -> usize {
    1
}

fn default_disp_freq() -> u64 {
    100
}

/// A native model file to continue from, or a DeePMD `model` section.
#[derive(Deserialize)]
#[serde(untagged)]
enum ModelSource {
    Path(PathBuf),
    Spec(ModelSpec),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    model: ModelSource,
    dataset: PathBuf,
    output: PathBuf,
    #[serde(default)]
    precision: Precision,
    learning_rate: LearningRate,
    loss: Loss,
    training: Training,
}

/// DeePMD's stepped exponential decay (`LearningRateExp`, no warmup).
struct Schedule {
    start: f64,
    stop: f64,
    decay_steps: u64,
    rate: f64,
}

impl Schedule {
    fn new(lr: &LearningRate, steps: u64) -> Result<Self, String> {
        if !(lr.start_lr.is_finite() && lr.start_lr > 0.0)
            || !(lr.stop_lr.is_finite() && lr.stop_lr > 0.0 && lr.stop_lr <= lr.start_lr)
            || lr.decay_steps == 0
        {
            return Err("learning_rate needs 0 < stop_lr <= start_lr and decay_steps > 0".into());
        }
        // As DeePMD: a decay period no shorter than the run is replaced.
        let decay_steps = if lr.decay_steps >= steps {
            if steps / 10 > 100 {
                100
            } else {
                steps / 100 + 1
            }
        } else {
            lr.decay_steps
        };
        let rate = ((lr.stop_lr.max(1e-10) / lr.start_lr).ln()
            / (steps as f64 / decay_steps as f64))
            .exp();
        Ok(Self {
            start: lr.start_lr,
            stop: lr.stop_lr,
            decay_steps,
            rate,
        })
    }

    fn value(&self, step: u64) -> f64 {
        (self.start * self.rate.powf((step / self.decay_steps) as f64)).max(self.stop)
    }
}

/// SplitMix64, for a reproducible frame order.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn shuffle(&mut self, order: &mut [usize]) {
        for i in (1..order.len()).rev() {
            order.swap(i, (self.next() % (i as u64 + 1)) as usize);
        }
    }
}

fn labels(frame: &DatasetFrame) -> FrameLabels<'_> {
    FrameLabels {
        energy: frame.labels.energy,
        forces: frame.labels.forces.as_deref(),
        virial: frame.labels.virial,
    }
}

/// A fresh model for `spec`, with statistics from the training frames.
fn fresh_model(spec: &ModelSpec, data: &Dataset) -> Result<Model, Box<dyn std::error::Error>> {
    let frames: Vec<&Frame> = data.frames.iter().map(|f| &f.frame).collect();
    let energies: Vec<f64> = data
        .frames
        .iter()
        .map(|f| f.labels.energy.ok_or("frame without an energy label"))
        .collect::<Result<_, _>>()?;
    let d = &spec.descriptor;
    let sel = match (&d.sel, d.sel.auto_factor()?) {
        (_, Some(factor)) => {
            let stat = neighbor_stat(&frames, &spec.type_map, d.rcut)?;
            eprintln!(
                "neighbor statistics: nearest pair {:.4} A, most neighbors per type {:?}",
                stat.min_nbor_dist, stat.max_nbor_size
            );
            stat.auto_sel(factor)
        }
        (omnimd_native::model_init::SelSpec::Counts(counts), None) => counts.clone(),
        _ => unreachable!("auto_factor is None only for explicit counts"),
    };
    let normalization = env_mat_stat(&frames, &spec.type_map, d.rcut, d.rcut_smth, &sel)?;
    let bias = energy_bias(&frames, &energies, &spec.type_map)?;
    eprintln!(
        "sel {sel:?}; energy bias per type {:?} (rank {})",
        bias.bias, bias.rank
    );
    Ok(initialize(spec, sel, normalization, bias.bias)?)
}

fn train<P: Scalar>(
    request: &Request,
    model: &Model,
    data: &Dataset,
    device: Arc<Device>,
) -> Result<(), Box<dyn std::error::Error>> {
    let training = &request.training;
    if training.numb_steps == 0 || training.batch_size == 0 || training.disp_freq == 0 {
        return Err("numb_steps, batch_size and disp_freq must be positive".into());
    }
    let schedule = Schedule::new(&request.learning_rate, training.numb_steps)?;
    let loss = &request.loss;
    let mut trainer = GpuTrainer::<P>::new(model, device)?;
    // Reserve the outputs before any work; never replace existing files.
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&request.output)
        .map_err(|e| format!("output {}: {e}", request.output.display()))?;
    let mut curve = match &training.disp_file {
        Some(path) => {
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .map_err(|e| format!("disp_file {}: {e}", path.display()))?;
            writeln!(
                file,
                "#  step      rmse_e_trn    rmse_f_trn    rmse_v_trn            lr"
            )?;
            Some(file)
        }
        None => None,
    };
    eprintln!(
        "training {} parameters on {} frames, network {}",
        trainer.parameter_count(),
        data.frames.len(),
        P::SUFFIX
    );
    let mut rng = Rng(training.seed);
    let mut order: Vec<usize> = (0..data.frames.len()).collect();
    let mut cursor = order.len();
    let start = Instant::now();
    let mut last = LossTerms::default();
    for step in 0..training.numb_steps {
        let lr = schedule.value(step);
        let coefficient = lr / schedule.start;
        let weights = LossWeights {
            energy: loss.limit_pref_e + (loss.start_pref_e - loss.limit_pref_e) * coefficient,
            force: loss.limit_pref_f + (loss.start_pref_f - loss.limit_pref_f) * coefficient,
            virial: loss.limit_pref_v + (loss.start_pref_v - loss.limit_pref_v) * coefficient,
        };
        trainer.zero_gradient()?;
        // Batch means of the squared errors, as DeePMD reports them.
        let mut batch = LossTerms::default();
        for _ in 0..training.batch_size {
            if cursor == order.len() {
                rng.shuffle(&mut order);
                cursor = 0;
            }
            let frame = &data.frames[order[cursor]];
            cursor += 1;
            let scale = 1.0 / training.batch_size as f64;
            let (terms, _) = trainer.accumulate(&frame.frame, &labels(frame), &weights, scale)?;
            if !terms.loss.is_finite() {
                return Err(format!("nonfinite loss at step {step} on frame {}", frame.id).into());
            }
            batch.loss += terms.loss * scale;
            batch.l2_energy += terms.l2_energy * scale;
            batch.l2_force += terms.l2_force * scale;
            batch.l2_virial += terms.l2_virial * scale;
            batch.natoms = terms.natoms;
        }
        trainer.adam_step(lr)?;
        last = batch;
        if step % training.disp_freq == 0 || step + 1 == training.numb_steps {
            let line = format!(
                "{:>7} {:>15.6e} {:>13.6e} {:>13.6e} {:>13.6e}",
                step,
                batch.rmse_energy_per_atom(),
                batch.rmse_force(),
                batch.rmse_virial_per_atom(),
                lr
            );
            if let Some(file) = &mut curve {
                writeln!(file, "{line}")?;
            }
            eprintln!("{line}");
        }
    }
    let trained = trainer.model()?;
    let elapsed = start.elapsed().as_secs_f64();
    serde_json::to_writer(&mut output, &trained)?;
    output.flush()?;
    println!(
        "{}",
        serde_json::json!({
            "steps": training.numb_steps,
            "batch_size": training.batch_size,
            "frames": data.frames.len(),
            "parameters": trainer.parameter_count(),
            "precision": P::SUFFIX,
            "seconds": elapsed,
            "steps_per_second": training.numb_steps as f64 / elapsed,
            "final_batch": {"loss": last.loss, "rmse_e_per_atom": last.rmse_energy_per_atom(),
                            "rmse_f": last.rmse_force(), "rmse_v_per_atom": last.rmse_virial_per_atom()},
            "output": request.output,
        })
    );
    Ok(())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [path] = args.as_slice() else {
        return Err("usage: omnimd-native-gpu-train REQUEST.json".into());
    };
    let request: Request = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    if request.output.exists() {
        return Err(format!("output {} already exists", request.output.display()).into());
    }
    let data = Dataset::load(&request.dataset)?;
    let declared = match &request.model {
        ModelSource::Path(path) => NativePotential::load(path)?.model().type_map.clone(),
        ModelSource::Spec(spec) => spec.type_map.clone(),
    };
    if data.header.type_map != declared {
        return Err(format!(
            "dataset type map {:?} differs from the model's {declared:?}",
            data.header.type_map
        )
        .into());
    }
    let model = match &request.model {
        ModelSource::Path(path) => NativePotential::load(path)?.model().clone(),
        ModelSource::Spec(spec) => fresh_model(spec, &data)?,
    };
    let device = Arc::new(Device::new(0)?);
    eprintln!("device: {}", device.name());
    match request.precision {
        Precision::Float64 => train::<f64>(&request, &model, &data, device),
        Precision::Float32 => train::<f32>(&request, &model, &data, device),
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("training stopped: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::{LearningRate, Rng, Schedule};

    /// Values printed by DeePMD-kit 3.1.3's `LearningRateExp(...).value(step)`.
    #[test]
    fn learning_rate_matches_deepmd() {
        let cases: [(f64, f64, u64, u64, &[(u64, f64)]); 3] = [
            (
                1e-3,
                3e-8,
                5000,
                100_000,
                &[
                    (0, 0.001),
                    (4999, 0.001),
                    (5000, 0.0005940952263100285),
                    (99_999, 5.049695515369199e-08),
                ],
            ),
            // A decay period longer than the run is replaced by 100 steps.
            (
                1e-3,
                1e-6,
                5000,
                2000,
                &[
                    (0, 0.001),
                    (699, 0.00012589254117941674),
                    (700, 8.912509381337455e-05),
                    (1999, 1.4125375446227544e-06),
                ],
            ),
            (
                2e-4,
                1e-8,
                700,
                300_000,
                &[
                    (699, 0.0002),
                    (700, 0.00019543136212907288),
                    (4999, 0.00017012958360663712),
                    (299_999, 1.0132922164651701e-08),
                ],
            ),
        ];
        for (start_lr, stop_lr, decay_steps, steps, expected) in cases {
            let schedule = Schedule::new(
                &LearningRate {
                    start_lr,
                    stop_lr,
                    decay_steps,
                },
                steps,
            )
            .unwrap();
            for &(step, value) in expected {
                let got = schedule.value(step);
                assert!(
                    (got - value).abs() < 1e-12 * value,
                    "step {step}: {got} vs {value}"
                );
            }
        }
        let bad = LearningRate {
            start_lr: 1e-3,
            stop_lr: 1e-2,
            decay_steps: 10,
        };
        assert!(Schedule::new(&bad, 100).is_err());
    }

    #[test]
    fn frame_order_is_a_reproducible_permutation() {
        let mut order: Vec<usize> = (0..97).collect();
        Rng(5).shuffle(&mut order);
        let mut again: Vec<usize> = (0..97).collect();
        Rng(5).shuffle(&mut again);
        assert_eq!(order, again);
        assert_ne!(order, (0..97).collect::<Vec<_>>());
        let mut sorted = order;
        sorted.sort_unstable();
        assert_eq!(sorted, (0..97).collect::<Vec<_>>());
    }
}
