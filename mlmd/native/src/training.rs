//! Frozen-hidden final affine readout training only. No full-network optimizer.
use crate::{
    least_squares,
    readout::{ReadoutBasis, ReadoutParameter},
    training_data::{validate_labels, Dataset, DatasetFrame, DatasetHeader},
    Model, NativePotential,
};
use omnimd_ml_contract::{Error, Potential, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum Split {
    Seeded { seed: u64, validation_fraction: f64 },
    Explicit { validation_ids: Vec<String> },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainConfig {
    pub split: Split,
    #[serde(default)]
    pub fit_bias: bool,
    pub energy_weight: f64,
    pub force_weight: f64,
    pub energy_scale_ev_per_atom: f64,
    pub force_scale_ev_per_angstrom: f64,
    #[serde(default)]
    pub ridge_inverse_ev2: f64,
    pub rank_rtol: f64,
}
impl TrainConfig {
    fn validate(&self) -> Result<()> {
        if ![
            self.energy_weight,
            self.force_weight,
            self.ridge_inverse_ev2,
        ]
        .iter()
        .all(|x| x.is_finite() && *x >= 0.0)
            || self.energy_weight + self.force_weight == 0.0
            || ![
                self.energy_scale_ev_per_atom,
                self.force_scale_ev_per_angstrom,
            ]
            .iter()
            .all(|x| x.is_finite() && *x > 0.0)
            || !(0.0 < self.rank_rtol && self.rank_rtol < 1.0)
        {
            return Err(Error(
                "invalid training loss weights, scales, ridge, or rank tolerance".into(),
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Serialize)]
pub struct FrameIdentity {
    pub index: u64,
    pub id: String,
}
#[derive(Debug, Serialize)]
pub struct Metrics {
    pub frames: usize,
    pub energy_rmse_ev: f64,
    pub energy_per_atom_rmse_ev: f64,
    pub force_component_rmse_ev_per_angstrom: f64,
    pub virial_component_rmse_ev: Option<f64>,
    pub normalized_ef_objective: f64,
}
#[derive(Debug, Serialize)]
pub struct TrainReport {
    pub scope: String,
    pub algorithm: String,
    pub feature_method: String,
    pub config: TrainConfig,
    pub dataset_header: DatasetHeader,
    pub dataset_sha256: String,
    pub source_sha256: BTreeMap<String, String>,
    pub training_frames: Vec<FrameIdentity>,
    pub validation_frames: Vec<FrameIdentity>,
    pub parameters: Vec<ReadoutParameter>,
    pub data_rank: usize,
    pub solve_rank: usize,
    pub rank_policy: String,
    pub ridge_penalty: f64,
    pub initial_training: Metrics,
    pub final_training: Metrics,
    pub initial_validation: Metrics,
    pub final_validation: Metrics,
}
pub struct TrainOutcome {
    pub model: Model,
    pub report: TrainReport,
}

fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
fn partition(dataset: &Dataset, split: &Split) -> Result<(Vec<usize>, Vec<usize>)> {
    let n = dataset.frames.len();
    if n < 2 {
        return Err(Error(
            "at least two frames are required for disjoint train/validation sets".into(),
        ));
    }
    let mut validation = HashSet::new();
    match split {
        Split::Seeded {
            seed,
            validation_fraction,
        } => {
            if !(0.0 < *validation_fraction && *validation_fraction < 1.0) {
                return Err(Error(
                    "validation_fraction must be between zero and one".into(),
                ));
            }
            let count = ((n as f64 * validation_fraction).round() as usize).clamp(1, n - 1);
            let mut order: Vec<_> = (0..n).collect();
            let mut state = *seed;
            for i in (1..n).rev() {
                let bound = (i + 1) as u64;
                let threshold = bound.wrapping_neg() % bound;
                let value = loop {
                    let v = next(&mut state);
                    if v >= threshold {
                        break v;
                    }
                };
                order.swap(i, (value % bound) as usize);
            }
            validation.extend(order.into_iter().take(count));
        }
        Split::Explicit { validation_ids } => {
            for id in validation_ids {
                let index = dataset
                    .frames
                    .iter()
                    .position(|f| &f.id == id)
                    .ok_or_else(|| Error(format!("unknown validation frame ID: {id}")))?;
                if !validation.insert(index) {
                    return Err(Error("duplicate validation ID".into()));
                }
            }
            if validation.is_empty() || validation.len() == n {
                return Err(Error("both splits must be nonempty".into()));
            }
        }
    }
    Ok((
        (0..n).filter(|i| !validation.contains(i)).collect(),
        (0..n).filter(|i| validation.contains(i)).collect(),
    ))
}
fn scales(config: &TrainConfig, frames: usize, atoms: usize) -> (f64, f64) {
    (
        (config.energy_weight / frames as f64).sqrt()
            / atoms as f64
            / config.energy_scale_ev_per_atom,
        (config.force_weight / frames as f64).sqrt()
            / (3.0 * atoms as f64).sqrt()
            / config.force_scale_ev_per_angstrom,
    )
}
fn metrics(
    model: &Model,
    dataset: &Dataset,
    indices: &[usize],
    config: &TrainConfig,
) -> Result<Metrics> {
    let mut potential = NativePotential::new(model.clone())?;
    let (mut se, mut sea, mut sf, mut sv, mut objective) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut nf, mut nv) = (0usize, 0usize);
    for &i in indices {
        let record = &dataset.frames[i];
        let prediction = potential.eval(&record.frame)?;
        let n = record.frame.positions.len();
        let de = prediction.energy - record.labels.energy.unwrap();
        se += de * de;
        sea += (de / n as f64).powi(2);
        let mut frame_sf = 0.0;
        for (p, t) in prediction
            .forces
            .iter()
            .flatten()
            .zip(record.labels.forces.as_ref().unwrap().iter().flatten())
        {
            frame_sf += (p - t).powi(2);
            nf += 1;
        }
        sf += frame_sf;
        let (es, fs) = scales(config, indices.len(), n);
        objective += (de * es).powi(2) + frame_sf * fs * fs;
        if let Some(label) = &record.labels.virial {
            for (p, t) in prediction
                .virial
                .iter()
                .flatten()
                .zip(label.iter().flatten())
            {
                sv += (p - t).powi(2);
                nv += 1;
            }
        }
    }
    if ![se, sea, sf, sv, objective].iter().all(|v| v.is_finite()) {
        return Err(Error("training metric overflow".into()));
    }
    Ok(Metrics {
        frames: indices.len(),
        energy_rmse_ev: (se / indices.len() as f64).sqrt(),
        energy_per_atom_rmse_ev: (sea / indices.len() as f64).sqrt(),
        force_component_rmse_ev_per_angstrom: (sf / nf as f64).sqrt(),
        virial_component_rmse_ev: if nv > 0 {
            Some((sv / nv as f64).sqrt())
        } else {
            None
        },
        normalized_ef_objective: objective,
    })
}

pub fn train(initial: &Model, dataset: &Dataset, config: &TrainConfig) -> Result<TrainOutcome> {
    config.validate()?;
    let metadata = initial.validate()?;
    if dataset.header.type_map != initial.type_map {
        return Err(Error(
            "dataset and model ordered type maps must match".into(),
        ));
    }
    let mut ids = HashSet::new();
    for (index, frame) in dataset.frames.iter().enumerate() {
        frame.frame.type_ids(&metadata)?;
        validate_labels(frame)?;
        if frame.index != index as u64 || frame.id.is_empty() || !ids.insert(&frame.id) {
            return Err(Error("invalid dataset frame identities".into()));
        }
    }
    let (training, validation) = partition(dataset, &config.split)?;
    let mut basis = ReadoutBasis::new(initial, config.fit_bias)?;
    let p = basis.parameters.len();
    let rows = training
        .iter()
        .try_fold(0usize, |sum, i| {
            sum.checked_add(1 + 3 * dataset.frames[*i].frame.positions.len())
        })
        .ok_or_else(|| Error("training design size overflow".into()))?;
    if rows
        .checked_add(p)
        .and_then(|r| r.checked_mul(p))
        .is_none_or(|v| v > 4_000_000)
    {
        return Err(Error(
            "small readout trainer design exceeds four million entries".into(),
        ));
    }
    let mut a = Vec::with_capacity(rows + p);
    let mut b = Vec::with_capacity(rows + p);
    let mut original = NativePotential::new(initial.clone())?;
    for &i in &training {
        let record = &dataset.frames[i];
        let feature = basis.features(&record.frame)?;
        let prediction = original.eval(&record.frame)?;
        let n = record.frame.positions.len();
        let (es, fs) = scales(config, training.len(), n);
        a.push(feature.columns.iter().map(|c| c.energy * es).collect());
        b.push((record.labels.energy.unwrap() - prediction.energy) * es);
        for atom in 0..n {
            for axis in 0..3 {
                a.push(
                    feature
                        .columns
                        .iter()
                        .map(|c| c.forces[atom][axis] * fs)
                        .collect(),
                );
                b.push(
                    (record.labels.forces.as_ref().unwrap()[atom][axis]
                        - prediction.forces[atom][axis])
                        * fs,
                );
            }
        }
    }
    let data_solution = least_squares::solve(a.clone(), b.clone(), config.rank_rtol)?;
    let data_rank = data_solution.rank;
    let solution = if config.ridge_inverse_ev2 > 0.0 {
        for i in 0..p {
            let mut row = vec![0.0; p];
            row[i] = config.ridge_inverse_ev2.sqrt();
            a.push(row);
            b.push(0.0);
        }
        least_squares::solve(a, b, config.rank_rtol)?
    } else {
        data_solution
    };
    let coefficients: Vec<_> = basis
        .initial_coefficients
        .iter()
        .zip(&solution.x)
        .map(|(x, dx)| x + dx)
        .collect();
    let model = basis.model_with_coefficients(&coefficients)?;
    let ridge_penalty = config.ridge_inverse_ev2 * solution.x.iter().map(|x| x * x).sum::<f64>();
    if !ridge_penalty.is_finite() {
        return Err(Error("ridge penalty overflow".into()));
    }
    fn identity(record: &DatasetFrame) -> FrameIdentity {
        FrameIdentity {
            index: record.index,
            id: record.id.clone(),
        }
    }
    let report=TrainReport {
        scope:"frozen descriptor and hidden layers; final affine readout only".into(),
        algorithm:"column-pivoted Householder QR; update from initial coefficients".into(),
        feature_method:"analytic one-hot readout basis; no finite-difference parameter gradients".into(),
        config:config.clone(), dataset_header:dataset.header.clone(), dataset_sha256:dataset.dataset_sha256.clone(),
        source_sha256:dataset.source_sha256.clone(),
        training_frames:training.iter().map(|i| identity(&dataset.frames[*i])).collect(),
        validation_frames:validation.iter().map(|i| identity(&dataset.frames[*i])).collect(),
        parameters:basis.parameters, data_rank, solve_rank:solution.rank,
        rank_policy:"relative to leading pivot norm; zero free pivoted updates (basic, not minimum-norm solution)".into(),
        ridge_penalty,
        initial_training:metrics(initial,dataset,&training,config)?,
        final_training:metrics(&model,dataset,&training,config)?,
        initial_validation:metrics(initial,dataset,&validation,config)?,
        final_validation:metrics(&model,dataset,&validation,config)?,
    };
    Ok(TrainOutcome { model, report })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn weighted_rank_deficient_and_ridge_solution() {
        let config = TrainConfig {
            split: Split::Seeded {
                seed: 1,
                validation_fraction: 0.2,
            },
            fit_bias: false,
            energy_weight: 1.0,
            force_weight: 1.0,
            energy_scale_ev_per_atom: 1.0,
            force_scale_ev_per_angstrom: 1.0,
            ridge_inverse_ev2: 0.0,
            rank_rtol: 1e-12,
        };
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for n in [2, 4] {
            let (es, fs) = scales(&config, 2, n);
            a.push(vec![n as f64 * es; 2]);
            b.push(2.0 * n as f64 * es);
            for i in 0..3 * n {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                a.push(vec![sign * fs; 2]);
                b.push(0.0);
            }
        }
        let plain = least_squares::solve(a.clone(), b.clone(), 1e-12).unwrap();
        assert_eq!(plain.rank, 1);
        assert!((plain.x.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(plain.x.contains(&0.0));
        a.extend([vec![1.0, 0.0], vec![0.0, 1.0]]);
        b.extend([0.0, 0.0]);
        let ridge = least_squares::solve(a, b, 1e-12).unwrap();
        assert_eq!(ridge.rank, 2);
        for x in ridge.x {
            assert!((x - 0.4).abs() < 1e-12);
        }
        let zero = least_squares::solve(vec![vec![0.0, 0.0]], vec![1.0], 1e-12).unwrap();
        assert_eq!(zero.rank, 0);
        assert_eq!(zero.x, vec![0.0, 0.0]);
    }
}
