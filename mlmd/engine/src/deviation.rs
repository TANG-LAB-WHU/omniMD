//! Absolute population model disagreement. Independent Rust implementation;
//! semantics checked against DeepMD 3.1.3 infer/model_devi.py, not copied code.
use omnimd_ml_contract::{Error, Evaluation, Frame, Potential, Result};
use serde::{Deserialize, Serialize};

pub const MAX_MODELS: usize = 16;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForceWindow {
    pub min_ev_per_angstrom: f64,
    pub max_ev_per_angstrom: f64,
}
impl ForceWindow {
    pub fn validate(&self) -> Result<()> {
        if !self.min_ev_per_angstrom.is_finite()
            || !self.max_ev_per_angstrom.is_finite()
            || self.min_ev_per_angstrom < 0.
            || self.min_ev_per_angstrom >= self.max_ev_per_angstrom
        {
            return Err(Error("force window requires finite 0 <= min < max".into()));
        }
        Ok(())
    }
    pub fn contains(&self, max_force_deviation: f64) -> bool {
        max_force_deviation >= self.min_ev_per_angstrom
            && max_force_deviation < self.max_ev_per_angstrom
    }
}
#[derive(Debug, Serialize)]
pub struct ForceDeviation {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_atom: Option<Vec<f64>>,
}
#[derive(Debug, Serialize)]
pub struct VirialDeviation {
    pub components: [[f64; 3]; 3],
    pub min: f64,
    pub max: f64,
    pub rms: f64,
}
#[derive(Debug, Serialize)]
pub struct ModelDeviation {
    pub energy_per_atom_std_ev: f64,
    pub mean_energy_per_atom_ev: f64,
    pub force_ev_per_angstrom: ForceDeviation,
    pub virial_per_atom_ev: VirialDeviation,
}
#[derive(Clone, Copy, Default)]
struct Moment {
    mean: f64,
    m2: f64,
}
impl Moment {
    fn add(&mut self, value: f64, count: usize) -> Result<()> {
        let delta = value - self.mean;
        self.mean += delta / count as f64;
        self.m2 += delta * (value - self.mean);
        if !self.mean.is_finite() || !self.m2.is_finite() || self.m2 < 0. {
            return Err(Error(
                "nonfinite/negative model-deviation moment (possible numeric overflow)".into(),
            ));
        }
        Ok(())
    }
    fn std(self, count: usize) -> f64 {
        (self.m2 / count as f64).sqrt()
    }
}
/// Frame-local O(N) moments: no model x frame prediction tensor is retained.
struct Accumulator {
    atoms: usize,
    seen: usize,
    energy: Moment,
    forces: Vec<[Moment; 3]>,
    virial: [[Moment; 3]; 3],
}
impl Accumulator {
    fn new(atoms: usize) -> Self {
        Self {
            atoms,
            seen: 0,
            energy: Moment::default(),
            forces: vec![[Moment::default(); 3]; atoms],
            virial: [[Moment::default(); 3]; 3],
        }
    }
    fn add(&mut self, evaluation: &Evaluation) -> Result<()> {
        evaluation.validate(self.atoms)?;
        self.seen += 1;
        self.energy
            .add(evaluation.energy / self.atoms as f64, self.seen)?;
        for (moments, force) in self.forces.iter_mut().zip(&evaluation.forces) {
            for axis in 0..3 {
                moments[axis].add(force[axis], self.seen)?;
            }
        }
        for row in 0..3 {
            for col in 0..3 {
                self.virial[row][col]
                    .add(evaluation.virial[row][col] / self.atoms as f64, self.seen)?;
            }
        }
        Ok(())
    }
    fn finish(self, per_atom: bool) -> Result<ModelDeviation> {
        let mut force = ForceDeviation {
            min: f64::INFINITY,
            max: 0.,
            mean: 0.,
            per_atom: per_atom.then(|| Vec::with_capacity(self.atoms)),
        };
        for moments in self.forces {
            let norm = moments
                .into_iter()
                .fold(0_f64, |s, m| s.hypot(m.std(self.seen)));
            force.min = force.min.min(norm);
            force.max = force.max.max(norm);
            force.mean += norm / self.atoms as f64;
            if let Some(values) = &mut force.per_atom {
                values.push(norm);
            }
        }
        let components = self.virial.map(|row| row.map(|m| m.std(self.seen)));
        let virial = VirialDeviation {
            min: components
                .iter()
                .flatten()
                .copied()
                .fold(f64::INFINITY, f64::min),
            max: components.iter().flatten().copied().fold(0., f64::max),
            rms: components.iter().flatten().fold(0_f64, |s, v| s.hypot(*v)) / 3.,
            components,
        };
        if [
            force.min, force.max, force.mean, virial.min, virial.max, virial.rms,
        ]
        .iter()
        .any(|x| !x.is_finite())
        {
            return Err(Error("model-deviation aggregation overflow".into()));
        }
        Ok(ModelDeviation {
            energy_per_atom_std_ev: self.energy.std(self.seen),
            mean_energy_per_atom_ev: self.energy.mean,
            force_ev_per_angstrom: force,
            virial_per_atom_ev: virial,
        })
    }
}
/// Exactly one combined evaluation per member, no labels, offset fitting or model weights.
/// Callers retain the already-loaded models across frames. Failed frames return no result.
pub fn evaluate_frame<P: Potential>(
    models: &mut [P],
    frame: &Frame,
    per_atom: bool,
) -> Result<ModelDeviation> {
    if !(2..=MAX_MODELS).contains(&models.len()) {
        return Err(Error(format!(
            "model deviation requires 2..={MAX_MODELS} models"
        )));
    }
    let reference = models[0].metadata();
    reference.validate()?;
    frame.type_ids(reference)?;
    for model in &models[1..] {
        model.metadata().validate()?;
        if model.metadata().type_map != reference.type_map
            || model.metadata().units != reference.units
        {
            return Err(Error(
                "ensemble models must have identical ordered type maps and units".into(),
            ));
        }
    }
    let mut accumulator = Accumulator::new(frame.positions.len());
    for (index, model) in models.iter_mut().enumerate() {
        let evaluation = model
            .eval(frame)
            .map_err(|e| Error(format!("ensemble member {index}: {e}")))?;
        accumulator
            .add(&evaluation)
            .map_err(|e| Error(format!("ensemble member {index}: {e}")))?;
    }
    accumulator.finish(per_atom)
}
