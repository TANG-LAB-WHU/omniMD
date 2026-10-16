//! Full-network E/F plus opt-in row-strain virial loss. No volume conversion.
use crate::{
    neighbor::Neighbor,
    training::{metrics_with, scales, Metrics, TrainConfig},
    training_data::{validate_labels, DatasetAccess, DatasetFrame},
    Model,
};
use omnimd_ml_contract::{Error, Evaluation, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VirialLoss {
    /// Strictly positive; omit the entire option to disable virial fitting.
    pub weight: f64,
    /// Scale of each total-virial component divided by atom count (eV/atom).
    pub scale_ev_per_atom: f64,
}
impl VirialLoss {
    pub(crate) fn validate(&self) -> Result<()> {
        if ![self.weight, self.scale_ev_per_atom]
            .iter()
            .all(|x| x.is_finite() && *x > 0.0)
        {
            return Err(Error("virial_loss weight and scale_ev_per_atom must be finite and positive; omit virial_loss to disable".into()));
        }
        Ok(())
    }
    pub(crate) fn scale(&self, frames: usize, atoms: usize) -> Result<f64> {
        let s = (self.weight / frames as f64 / 9.0).sqrt() / atoms as f64 / self.scale_ev_per_atom;
        let coefficient = 2.0 * s * s;
        if frames == 0
            || atoms == 0
            || !s.is_finite()
            || s <= 0.0
            || !coefficient.is_finite()
            || coefficient <= 0.0
        {
            return Err(Error(
                "virial_loss normalization underflow/overflow for this frame count and atom count"
                    .into(),
            ));
        }
        Ok(s)
    }
}
pub(crate) fn validate_loss(loss: &TrainConfig, virial: Option<&VirialLoss>) -> Result<()> {
    if let Some(v) = virial {
        v.validate()?;
    }
    loss.validate_allow_zero_ef(virial.is_some())
}
pub(crate) fn validate_record(record: &DatasetFrame, virial: Option<&VirialLoss>) -> Result<()> {
    validate_labels(record)?;
    if virial.is_some() && record.labels.virial.is_none() {
        return Err(Error(format!(
            "{}: enabled virial_loss requires all nine finite virial labels on every frame",
            record.id
        )));
    }
    Ok(())
}
pub(crate) struct LossSeeds {
    pub objective: f64,
    pub energy: f64,
    force: Vec<[f64; 3]>,
    virial: Option<[[f64; 3]; 3]>,
}
impl LossSeeds {
    pub fn new(
        predicted: &Evaluation,
        record: &DatasetFrame,
        loss: &TrainConfig,
        virial: Option<&VirialLoss>,
        frames: usize,
    ) -> Result<Self> {
        if frames == 0 {
            return Err(Error(
                "gradient normalization requires positive frame count".into(),
            ));
        }
        let n = record.frame.positions.len();
        let (es, fs) = scales(loss, frames, n);
        let de = predicted.energy - record.labels.energy.unwrap();
        let energy = 2.0 * es * es * de;
        let mut force = vec![[0.0; 3]; n];
        let mut objective = (de * es).powi(2);
        for atom in 0..n {
            for axis in 0..3 {
                let df = predicted.forces[atom][axis]
                    - record.labels.forces.as_ref().unwrap()[atom][axis];
                force[atom][axis] = 2.0 * fs * fs * df;
                objective += (df * fs).powi(2);
            }
        }
        let virial = if let Some(v) = virial {
            let vs = v.scale(frames, n)?;
            let label = record.labels.virial.as_ref().unwrap();
            let mut seed = [[0.0; 3]; 3];
            for a in 0..3 {
                for b in 0..3 {
                    let dv = predicted.virial[a][b] - label[a][b];
                    seed[a][b] = 2.0 * vs * vs * dv;
                    objective += (dv * vs).powi(2);
                }
            }
            Some(seed)
        } else {
            None
        };
        if !objective.is_finite()
            || !energy.is_finite()
            || force.iter().flatten().any(|x| !x.is_finite())
            || virial.iter().flatten().flatten().any(|x| !x.is_finite())
        {
            return Err(Error("nonfinite full loss/cotangent".into()));
        }
        Ok(Self {
            objective,
            energy,
            force,
            virial,
        })
    }
    /// Full image displacement is essential: self-image force directions cancel,
    /// but strain directions do not. Preserve the exact E/F arithmetic when None.
    pub fn tangent(&self, jac: [f64; 3], neighbor: &Neighbor, center: usize) -> f64 {
        (0..3)
            .map(|b| {
                let mut direction = self.force[neighbor.atom][b] - self.force[center][b];
                if let Some(c) = &self.virial {
                    direction += (0..3).map(|a| neighbor.delta[a] * c[a][b]).sum::<f64>();
                }
                jac[b] * direction
            })
            .sum()
    }
}

#[derive(Serialize)]
pub struct FullMetrics {
    #[serde(flatten)]
    pub energy_force: Metrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalized_virial_objective: Option<f64>,
    pub normalized_total_objective: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub virial_per_atom_component_rmse_ev: Option<f64>,
}
impl std::ops::Deref for FullMetrics {
    type Target = Metrics;
    fn deref(&self) -> &Self::Target {
        &self.energy_force
    }
}
pub(crate) fn metrics(
    model: &Model,
    dataset: &dyn DatasetAccess,
    indices: &[usize],
    loss: &TrainConfig,
    virial: Option<&VirialLoss>,
) -> Result<FullMetrics> {
    let (mut objective, mut per_atom, mut count) = (0.0, 0.0, 0usize);
    let energy_force = metrics_with(model, dataset, indices, loss, |record, predicted| {
        if let Some(v) = virial {
            validate_record(record, Some(v))?;
            let n = record.frame.positions.len();
            let vs = v.scale(indices.len(), n)?;
            for (p, t) in predicted
                .virial
                .iter()
                .flatten()
                .zip(record.labels.virial.as_ref().unwrap().iter().flatten())
            {
                objective += ((p - t) * vs).powi(2);
                per_atom += ((p - t) / n as f64).powi(2);
                count += 1;
            }
        }
        Ok(())
    })?;
    let total = energy_force.normalized_ef_objective + objective;
    if ![objective, per_atom, total].iter().all(|x| x.is_finite()) {
        return Err(Error("full training virial metric overflow".into()));
    }
    Ok(FullMetrics {
        energy_force,
        normalized_virial_objective: virial.map(|_| objective),
        normalized_total_objective: total,
        virial_per_atom_component_rmse_ev: virial.map(|_| (per_atom / count as f64).sqrt()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn row_strain_uses_full_displacement_and_ordered_tensor() {
        let seeds = LossSeeds {
            objective: 0.0,
            energy: 0.0,
            force: vec![[1.0, 2.0, 3.0]],
            virial: Some([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        };
        let q = Neighbor {
            atom: 0,
            delta: [2.0, 3.0, 5.0],
            distance: 38.0_f64.sqrt(),
            #[cfg(feature = "cuda")]
            image: [1, 0, 0],
        };
        // d*C = [49,59,69], even when physical neighbor and center are identical.
        assert_eq!(seeds.tangent([1.0, 0.0, 0.0], &q, 0), 49.0);
        assert_eq!(seeds.tangent([0.0, 1.0, 0.0], &q, 0), 59.0);
        assert_eq!(seeds.tangent([0.0, 0.0, 1.0], &q, 0), 69.0);
    }
}
