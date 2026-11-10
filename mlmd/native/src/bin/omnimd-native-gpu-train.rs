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
use omnimd_native::gpu_train::{
    FrameLabels, GpuTrainer, LossTerms, LossWeights, OptimizerState, TrainStageTimes,
};
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

impl Loss {
    /// `(name, start, limit)` of the three terms.
    fn terms(&self) -> [(&'static str, f64, f64); 3] {
        [
            ("energy", self.start_pref_e, self.limit_pref_e),
            ("force", self.start_pref_f, self.limit_pref_f),
            ("virial", self.start_pref_v, self.limit_pref_v),
        ]
    }

    /// The prefactors when the learning rate is `coefficient` times its
    /// starting value. A term is part of the loss only when both of its
    /// prefactors are nonzero: the rule of DeePMD-kit's PyTorch backend,
    /// where `start_pref_v: 0` switches the virial term off whatever
    /// `limit_pref_v` is.
    fn weights(&self, coefficient: f64) -> LossWeights {
        let [energy, force, virial] = self.terms().map(|(_, start, limit)| {
            if start != 0.0 && limit != 0.0 {
                limit + (start - limit) * coefficient
            } else {
                0.0
            }
        });
        LossWeights {
            energy,
            force,
            virial,
        }
    }
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
    /// Write `<output>.step-N` (model) and `<output>.step-N.adam` (optimizer
    /// state) every this many steps.
    save_freq: Option<u64>,
    /// An `.adam` file written by `save_freq`: continue from its step with
    /// its optimizer state; `model` must then be the matching snapshot.
    optimizer_state: Option<PathBuf>,
    /// Frames that are scored, but never trained on, whenever a line of the
    /// learning curve is written.
    validation_data: Option<ValidationData>,
    /// Time every stage of a step and report the means in the summary. The
    /// device is synchronized after each stage, so the run is slower.
    #[serde(default)]
    profile_stages: bool,
}

/// Stage times as milliseconds per step.
fn stages_per_step(times: &TrainStageTimes, steps: u64) -> serde_json::Value {
    let mut value = serde_json::to_value(times).unwrap_or_default();
    if let Some(stages) = value.as_object_mut() {
        stages.remove("frames");
        for time in stages.values_mut() {
            *time = (time.as_f64().unwrap_or(0.0) / steps as f64).into();
        }
    }
    value
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ValidationData {
    dataset: PathBuf,
    /// Frames scored per line of the learning curve.
    #[serde(default = "one_batch")]
    numb_btch: usize,
}

fn one_batch() -> usize {
    1
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

const ADAM_MAGIC: &[u8; 8] = b"OMDADAM1";

/// Optimizer state file: magic, step, count, then the first and second
/// moments as little-endian float64.
fn write_optimizer_state(path: &std::path::Path, state: &OptimizerState) -> std::io::Result<()> {
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    let mut out = std::io::BufWriter::new(file);
    out.write_all(ADAM_MAGIC)?;
    out.write_all(&state.step.to_le_bytes())?;
    out.write_all(&(state.m.len() as u64).to_le_bytes())?;
    for value in state.m.iter().chain(&state.v) {
        out.write_all(&value.to_le_bytes())?;
    }
    out.flush()
}

fn read_optimizer_state(path: &std::path::Path) -> Result<OptimizerState, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let word = |at: usize| -> Option<u64> {
        Some(u64::from_le_bytes(bytes.get(at..at + 8)?.try_into().ok()?))
    };
    let invalid = || format!("{} is not an optimizer state file", path.display());
    if bytes.get(..8) != Some(ADAM_MAGIC.as_slice()) {
        return Err(invalid());
    }
    let (step, count) = (
        word(8).ok_or_else(invalid)?,
        word(16).ok_or_else(invalid)? as usize,
    );
    if count.checked_mul(16).and_then(|n| n.checked_add(24)) != Some(bytes.len()) {
        return Err(invalid());
    }
    let values = |first: usize| -> Vec<f64> {
        (0..count)
            .map(|k| f64::from_bits(word(24 + 8 * (first + k)).unwrap()))
            .collect()
    };
    Ok(OptimizerState {
        step,
        m: values(0),
        v: values(count),
    })
}

fn snapshot_paths(output: &std::path::Path, step: u64) -> (PathBuf, PathBuf) {
    let mut model = output.as_os_str().to_owned();
    model.push(format!(".step-{step}"));
    let mut adam = model.clone();
    adam.push(".adam");
    (PathBuf::from(model), PathBuf::from(adam))
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

thread_local! {
    /// Prefix of this thread's progress lines: the request's file name when
    /// several requests train at once, empty otherwise.
    static TAG: std::cell::RefCell<String> = std::cell::RefCell::new(String::new());
}

/// A progress line on standard error.
fn note(line: std::fmt::Arguments) {
    TAG.with(|tag| eprintln!("{}{line}", tag.borrow()));
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
            note(format_args!(
                "neighbor statistics: nearest pair {:.4} A, most neighbors per type {:?}",
                stat.min_nbor_dist, stat.max_nbor_size
            ));
            stat.auto_sel(factor)
        }
        (omnimd_native::model_init::SelSpec::Counts(counts), None) => counts.clone(),
        _ => unreachable!("auto_factor is None only for explicit counts"),
    };
    let normalization = env_mat_stat(&frames, &spec.type_map, d.rcut, d.rcut_smth, &sel)?;
    let bias = match &spec.fitting_net.atom_ener {
        Some(given) => {
            if given.len() != spec.type_map.len() || given.iter().any(|b| !b.is_finite()) {
                return Err("atom_ener needs one finite energy for every type".into());
            }
            note(format_args!(
                "sel {sel:?}; energy bias per type {given:?} (atom_ener)"
            ));
            given.clone()
        }
        None => {
            let fitted = energy_bias(&frames, &energies, &spec.type_map)?;
            note(format_args!(
                "sel {sel:?}; energy bias per type {:?} (rank {})",
                fitted.bias, fitted.rank
            ));
            fitted.bias
        }
    };
    Ok(initialize(spec, sel, normalization, bias)?)
}

fn train<P: Scalar>(
    request: &Request,
    model: &Model,
    data: &Dataset,
    validation: Option<&Dataset>,
    device: Arc<Device>,
) -> Result<(), Box<dyn std::error::Error>> {
    let training = &request.training;
    if training.numb_steps == 0 || training.batch_size == 0 || training.disp_freq == 0 {
        return Err("numb_steps, batch_size and disp_freq must be positive".into());
    }
    let schedule = Schedule::new(&request.learning_rate, training.numb_steps)?;
    let loss = &request.loss;
    if training.save_freq == Some(0) {
        return Err("save_freq must be positive".into());
    }
    let mut trainer = GpuTrainer::<P>::new(model, device)?;
    let first_step = match &training.optimizer_state {
        Some(path) => {
            let state = read_optimizer_state(path)?;
            if state.step >= training.numb_steps {
                return Err(format!(
                    "optimizer state is at step {}, at or past numb_steps {}",
                    state.step, training.numb_steps
                )
                .into());
            }
            trainer.set_optimizer_state(&state)?;
            note(format_args!("continuing from step {}", state.step));
            state.step
        }
        None => 0,
    };
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
            match validation {
                Some(_) => writeln!(
                    file,
                    "#  step      rmse_e_val    rmse_e_trn    rmse_f_val    rmse_f_trn    rmse_v_val    rmse_v_trn            lr"
                )?,
                None => writeln!(
                    file,
                    "#  step      rmse_e_trn    rmse_f_trn    rmse_v_trn            lr"
                )?,
            }
            Some(file)
        }
        None => None,
    };
    note(format_args!(
        "training {} parameters on {} frames, network {}",
        trainer.parameter_count(),
        data.frames.len(),
        P::SUFFIX
    ));
    for (name, start, limit) in loss.terms() {
        if (start == 0.0) != (limit == 0.0) {
            note(format_args!(
                "the {name} term is off: one of its prefactors is zero ({start} to {limit}), and as in DeePMD-kit's PyTorch backend a term needs both"
            ));
        }
    }
    trainer.set_profiling(training.profile_stages);
    // A continued run draws a new frame order from its starting step.
    let mut rng = Rng(training.seed ^ first_step.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    let mut order: Vec<usize> = (0..data.frames.len()).collect();
    let mut cursor = order.len();
    // Validation frames in an order of their own.
    let batches = training.validation_data.as_ref().map_or(0, |v| v.numb_btch);
    if validation.is_some() && batches == 0 {
        return Err("validation_data numb_btch must be positive".into());
    }
    let mut validation_rng = Rng(training.seed ^ 0x5851_f42d_4c95_7f2d);
    let mut validation_order: Vec<usize> = (0..validation.map_or(0, |v| v.frames.len())).collect();
    let mut validation_cursor = validation_order.len();
    let mut last_validation = None;
    let start = Instant::now();
    let mut last = LossTerms::default();
    for step in first_step..training.numb_steps {
        let lr = schedule.value(step);
        let coefficient = lr / schedule.start;
        let weights = loss.weights(coefficient);
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
        let done = step + 1;
        if training.save_freq.is_some_and(|f| done % f == 0) && done < training.numb_steps {
            let (model_path, adam_path) = snapshot_paths(&request.output, done);
            let file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&model_path)
                .map_err(|e| format!("snapshot {}: {e}", model_path.display()))?;
            serde_json::to_writer(std::io::BufWriter::new(file), &trainer.model()?)?;
            write_optimizer_state(&adam_path, &trainer.optimizer_state()?)
                .map_err(|e| format!("snapshot {}: {e}", adam_path.display()))?;
        }
        if step % training.disp_freq == 0 || step + 1 == training.numb_steps {
            let line = match validation {
                Some(held_out) => {
                    // Scored with the updated parameters; a zero scale leaves
                    // the gradient alone.
                    let mut scored = LossTerms::default();
                    for _ in 0..batches {
                        if validation_cursor == validation_order.len() {
                            validation_rng.shuffle(&mut validation_order);
                            validation_cursor = 0;
                        }
                        let frame = &held_out.frames[validation_order[validation_cursor]];
                        validation_cursor += 1;
                        let (terms, _) =
                            trainer.accumulate(&frame.frame, &labels(frame), &weights, 0.0)?;
                        let share = 1.0 / batches as f64;
                        scored.loss += terms.loss * share;
                        scored.l2_energy += terms.l2_energy * share;
                        scored.l2_force += terms.l2_force * share;
                        scored.l2_virial += terms.l2_virial * share;
                        scored.natoms = terms.natoms;
                    }
                    last_validation = Some(scored);
                    format!(
                        "{:>7} {:>15.6e} {:>13.6e} {:>13.6e} {:>13.6e} {:>13.6e} {:>13.6e} {:>13.6e}",
                        step,
                        scored.rmse_energy_per_atom(),
                        batch.rmse_energy_per_atom(),
                        scored.rmse_force(),
                        batch.rmse_force(),
                        scored.rmse_virial_per_atom(),
                        batch.rmse_virial_per_atom(),
                        lr
                    )
                }
                None => format!(
                    "{:>7} {:>15.6e} {:>13.6e} {:>13.6e} {:>13.6e}",
                    step,
                    batch.rmse_energy_per_atom(),
                    batch.rmse_force(),
                    batch.rmse_virial_per_atom(),
                    lr
                ),
            };
            if let Some(file) = &mut curve {
                writeln!(file, "{line}")?;
            }
            note(format_args!("{line}"));
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
            "first_step": first_step,
            "batch_size": training.batch_size,
            "frames": data.frames.len(),
            "parameters": trainer.parameter_count(),
            "precision": P::SUFFIX,
            "seconds": elapsed,
            "steps_per_second": (training.numb_steps - first_step) as f64 / elapsed,
            "stages_ms_per_step": trainer
                .stage_times()
                .map(|times| stages_per_step(times, training.numb_steps - first_step)),
            "final_batch": {"loss": last.loss, "rmse_e_per_atom": last.rmse_energy_per_atom(),
                            "rmse_f": last.rmse_force(), "rmse_v_per_atom": last.rmse_virial_per_atom()},
            "final_validation": last_validation.map(|v: LossTerms| serde_json::json!({
                "rmse_e_per_atom": v.rmse_energy_per_atom(), "rmse_f": v.rmse_force(),
                "rmse_v_per_atom": v.rmse_virial_per_atom()})),
            "output": request.output,
        })
    );
    Ok(())
}

/// One request, from its file to its trained model, on `device`.
fn train_request(path: &str, device: Device) -> Result<(), Box<dyn std::error::Error>> {
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
    let validation = match &request.training.validation_data {
        Some(held_out) => {
            let frames = Dataset::load(&held_out.dataset)?;
            if frames.header.type_map != declared {
                return Err("validation dataset and model type maps differ".into());
            }
            Some(frames)
        }
        None => None,
    };
    let device = Arc::new(device);
    match request.precision {
        Precision::Float64 => train::<f64>(&request, &model, &data, validation.as_ref(), device),
        Precision::Float32 => train::<f32>(&request, &model, &data, validation.as_ref(), device),
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        return Err("usage: omnimd-native-gpu-train REQUEST.json [REQUEST.json ...]".into());
    }
    let first = Device::new(0)?;
    eprintln!("device: {}", first.name());
    let several = paths.len() > 1;
    // Several requests train at the same time, each on a stream of its own.
    // None of them takes the device's default stream: work on that one waits
    // for the others, which slowed its request to 0.6 of their speed.
    let devices = if several {
        let mut devices = Vec::with_capacity(paths.len());
        for _ in &paths {
            devices.push(first.sibling()?);
        }
        devices
    } else {
        vec![first]
    };
    let failures: Vec<String> = std::thread::scope(|scope| {
        let workers: Vec<_> = paths
            .iter()
            .zip(devices)
            .map(|(path, device)| {
                scope.spawn(move || {
                    if several {
                        let name = std::path::Path::new(path)
                            .file_name()
                            .map_or(path.as_str().into(), |name| name.to_string_lossy());
                        TAG.with(|tag| *tag.borrow_mut() = format!("[{name}] "));
                    }
                    train_request(path, device).map_err(|e| format!("{path}: {e}"))
                })
            })
            .collect();
        workers
            .into_iter()
            .filter_map(|worker| match worker.join() {
                Ok(result) => result.err(),
                Err(_) => Some("a training thread panicked".to_string()),
            })
            .collect()
    });
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; ").into())
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
    use super::{LearningRate, Loss, Rng, Schedule};

    #[test]
    fn a_loss_term_needs_both_prefactors() {
        let loss = |start_pref_v, limit_pref_v| Loss {
            start_pref_e: 0.02,
            limit_pref_e: 1.0,
            start_pref_f: 1000.0,
            limit_pref_f: 1.0,
            start_pref_v,
            limit_pref_v,
        };
        // Between start and limit in proportion to the learning rate.
        let weights = loss(0.5, 0.02).weights(0.25);
        assert!((weights.energy - (1.0 - 0.98 * 0.25)).abs() < 1e-15);
        assert!((weights.force - (1.0 + 999.0 * 0.25)).abs() < 1e-12);
        assert!((weights.virial - (0.02 + 0.48 * 0.25)).abs() < 1e-15);
        // DeePMD-kit 3.1.3, deepmd/pt/loss/ener.py: has_v is
        // `start_pref_v != 0.0 and limit_pref_v != 0.0`.
        for (start, limit) in [(0.0, 0.02), (0.02, 0.0), (0.0, 0.0)] {
            for coefficient in [1.0, 0.3, 0.0] {
                assert_eq!(loss(start, limit).weights(coefficient).virial, 0.0);
            }
        }
        assert!(loss(1e-9, 0.02).weights(0.0).virial == 0.02);
    }

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
    fn optimizer_state_files_round_trip_and_reject_damage() {
        let directory = std::env::temp_dir().join(format!("omnimd-adam-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let (model, adam) = super::snapshot_paths(&directory.join("out.json"), 4000);
        assert!(model.ends_with("out.json.step-4000") && adam.ends_with("out.json.step-4000.adam"));
        let state = super::OptimizerState {
            step: 4000,
            m: vec![0.25, -1.5e-7, 3.0],
            v: vec![1e-12, 2.0, 0.0],
        };
        super::write_optimizer_state(&adam, &state).unwrap();
        assert_eq!(super::read_optimizer_state(&adam).unwrap(), state);
        // Existing files are never replaced.
        assert!(super::write_optimizer_state(&adam, &state).is_err());
        let mut bytes = std::fs::read(&adam).unwrap();
        bytes.pop();
        let damaged = directory.join("damaged.adam");
        std::fs::write(&damaged, &bytes).unwrap();
        assert!(super::read_optimizer_state(&damaged).is_err());
        bytes[0] = b'X';
        std::fs::write(&damaged, &bytes).unwrap();
        assert!(super::read_optimizer_state(&damaged).is_err());
        std::fs::remove_dir_all(directory).unwrap();
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
