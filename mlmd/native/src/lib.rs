//! CPU float64 inference for the explicitly versioned native se_e2_a subset.
mod autodiff;
pub mod full_gradient;
pub mod full_training;
mod least_squares;
mod model;
mod neighbor;
mod network;
pub mod readout;
pub mod training;
pub mod training_data;

pub use model::{Activation, Descriptor, Layer, Model, Network, Normalization, Residual};
use neighbor::Neighbor;
use omnimd_ml_contract::{Error, Evaluation, Frame, ModelMetadata, Potential, Result, Units};
use std::path::Path;

/// A validated, immutable set of weights; evaluation owns its temporary buffers.
#[derive(Debug, Clone)]
pub struct NativePotential {
    model: Model,
    metadata: ModelMetadata,
}

impl NativePotential {
    pub fn from_json(json: &str) -> Result<Self> {
        let model: Model =
            serde_json::from_str(json).map_err(|e| Error(format!("native model JSON: {e}")))?;
        Self::new(model)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let json = std::fs::read_to_string(path)
            .map_err(|e| Error(format!("read native model {}: {e}", path.display())))?;
        Self::from_json(&json)
    }

    pub fn new(model: Model) -> Result<Self> {
        let metadata = model.validate()?;
        Ok(Self { model, metadata })
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    fn atom(
        &self,
        center: usize,
        kind: usize,
        neighbors: &[Vec<Neighbor>],
        result: &mut Evaluation,
    ) -> Result<()> {
        let d = &self.model.descriptor;
        let norm = &d.normalization[kind];
        let width = d.embedding[kind][0].output_size();
        let slots = d.sel.iter().sum::<usize>() as f64;
        let mut b = vec![[0.0; 4]; width];
        let mut records = Vec::new();
        for (species, group) in neighbors.iter().enumerate() {
            let net = &d.embedding[kind][species];
            // Empty slots are zero BEFORE normalization; their embedding and
            // radial contribution generally do not vanish after centering.
            for slot in 0..d.sel[species] {
                let neighbor = group.get(slot);
                let (r, jac) = environment(neighbor, norm, d.rcut_smth, d.rcut);
                let (g, dg) = net.scalar_with_derivative(r[0])?;
                for m in 0..width {
                    for a in 0..4 {
                        b[m][a] += g[m] * r[a] / slots;
                    }
                }
                if let Some(n) = neighbor {
                    records.push((n, r, jac, g, dg));
                }
            }
        }
        let mut descriptor = Vec::with_capacity(width * d.axis_neuron);
        for m in 0..width {
            for k in 0..d.axis_neuron {
                descriptor.push((0..4).map(|a| b[m][a] * b[k][a]).sum());
            }
        }
        let (energy, dd) = self.model.fitting[kind].scalar_with_gradient(&descriptor)?;
        result.energy += energy + self.model.atom_energy_bias[kind];
        let mut db = vec![[0.0; 4]; width];
        for m in 0..width {
            for k in 0..d.axis_neuron {
                for a in 0..4 {
                    db[m][a] += dd[m * d.axis_neuron + k] * b[k][a];
                    db[k][a] += dd[m * d.axis_neuron + k] * b[m][a];
                }
            }
        }
        for (n, r, jac, g, dg) in records {
            let mut dr = [0.0; 4];
            for m in 0..width {
                for a in 0..4 {
                    dr[a] += db[m][a] * g[m] / slots;
                    dr[0] += db[m][a] * r[a] * dg[m] / slots;
                }
            }
            let mut grad = [0.0; 3];
            for c in 0..3 {
                grad[c] = (0..4).map(|a| dr[a] * jac[a][c]).sum();
                result.forces[center][c] += grad[c];
                result.forces[n.atom][c] -= grad[c];
                for a in 0..3 {
                    result.virial[a][c] -= n.delta[a] * grad[c];
                }
            }
        }
        if !result.energy.is_finite() {
            return Err(Error(format!("nonfinite native energy at atom {center}")));
        }
        Ok(())
    }
}

impl Potential for NativePotential {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn eval(&mut self, frame: &Frame) -> Result<Evaluation> {
        frame.validate()?;
        self.metadata.validate()?;
        let types = frame.type_ids(&self.metadata)?;
        let geometry = neighbor::Geometry::new(frame, self.model.descriptor.rcut)?;
        let mut result = Evaluation {
            energy: 0.0,
            forces: vec![[0.0; 3]; frame.positions.len()],
            virial: [[0.0; 3]; 3],
            units: Units::EvAngstrom,
        };
        for center in 0..frame.positions.len() {
            let neighbors =
                geometry.neighbors(frame, &types, center, &self.model.descriptor.sel)?;
            self.atom(center, types[center] as usize, &neighbors, &mut result)?;
        }
        result.validate(frame.positions.len())?;
        Ok(result)
    }
}

fn environment(
    n: Option<&Neighbor>,
    norm: &Normalization,
    smooth: f64,
    cutoff: f64,
) -> ([f64; 4], [[f64; 3]; 4]) {
    let mut r = [0.0; 4];
    let mut jac = [[0.0; 3]; 4];
    if let Some(n) = n {
        let radius = n.distance;
        let u = n.delta.map(|x| x / radius);
        let (q, dq) = if radius < smooth {
            (1.0, 0.0)
        } else {
            let t = (radius - smooth) / (cutoff - smooth);
            // Quintic switch: value, first and second derivatives vanish at rc.
            let z = 1.0 - t;
            (
                z * z * z * (1.0 + 3.0 * t + 6.0 * t * t),
                -30.0 * t * t * z * z / (cutoff - smooth),
            )
        };
        let s = q / radius;
        let ds = dq / radius - s / radius;
        r[0] = s;
        for a in 0..3 {
            r[a + 1] = s * u[a];
            jac[0][a] = ds * u[a];
            for c in 0..3 {
                let identity = if a == c { 1.0 } else { 0.0 };
                jac[a + 1][c] = ds * u[a] * u[c] + s / radius * (identity - u[a] * u[c]);
            }
        }
    }
    for a in 0..4 {
        r[a] = (r[a] - norm.mean[a]) / norm.std[a];
        for c in 0..3 {
            jac[a][c] /= norm.std[a];
        }
    }
    (r, jac)
}
