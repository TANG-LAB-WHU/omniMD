//! CPU float64 inference for the explicitly versioned native se_e2_a,
//! se_e2_r and se_e3 subsets and for hybrids of them.
#[cfg(any(test, feature = "gpu"))]
mod adaptive_table;
mod autodiff;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
mod fixed_geometry;
pub mod full_gradient;
mod full_loss;
pub mod full_training;
#[cfg(feature = "gpu")]
pub mod gpu_model;
#[cfg(feature = "gpu")]
pub mod gpu_train;
mod learning_rate;
mod least_squares;
pub mod minibatch;
mod model;
pub mod model_init;
mod neighbor;
mod network;
pub mod readout;
pub mod stats;
pub mod training;
pub mod training_data;

pub use model::{
    Activation, Descriptor, DescriptorKind, Layer, Model, Network, Normalization, Residual,
    HYBRID_FORMAT, HYBRID_KIND, MAX_HYBRID_PARTS,
};
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

    #[cfg(feature = "cuda")]
    pub(crate) fn eval_prepared(
        &self,
        frame: &fixed_geometry::PreparedFrame,
    ) -> Result<Evaluation> {
        let n = frame.types.len();
        let mut result = Evaluation {
            energy: 0.0,
            forces: vec![[0.0; 3]; n],
            virial: [[0.0; 3]; 3],
            units: Units::EvAngstrom,
        };
        for center in 0..n {
            // The prepared rows are those of the one descriptor of the model.
            self.atom(
                center,
                frame.types[center] as usize,
                |_| Ok(frame.atom_rows(center)),
                &mut result,
            )?;
        }
        result.validate(n)?;
        Ok(result)
    }

    /// One atom: the components of every descriptor of the model, one after
    /// the other (a hybrid has several), through the fitting net of the
    /// atom's type, and back through each descriptor to forces and virial.
    /// `rows` gives the slots of the atom's environment as a descriptor
    /// sees them.
    fn atom<'a, R>(
        &'a self,
        center: usize,
        kind: usize,
        mut rows: impl FnMut(&'a Descriptor) -> Result<R>,
        result: &mut Evaluation,
    ) -> Result<()>
    where
        R: IntoIterator<Item = Row<'a>>,
    {
        let parts = self.model.descriptor.parts();
        let mut descriptor = Vec::new();
        let mut tapes = Vec::with_capacity(parts.len());
        for part in parts {
            let first = descriptor.len();
            let tape = components(part, kind, rows(part)?, &mut descriptor)?;
            tapes.push((first, descriptor.len(), tape));
        }
        let (energy, dd) = self.model.fitting[kind].scalar_with_gradient(&descriptor)?;
        result.energy += energy + self.model.atom_energy_bias[kind];
        for (part, (first, end, tape)) in parts.iter().zip(tapes) {
            tape.forces(part, &dd[first..end], center, result);
        }
        if !result.energy.is_finite() {
            return Err(Error(format!("nonfinite native energy at atom {center}")));
        }
        Ok(())
    }
}

/// A slot of the environment of an atom: the neighbor type it is for, its
/// neighbor if it has one, the normalized row and the Jacobian of the row
/// with respect to the neighbor vector.
type Row<'a> = (usize, Option<&'a Neighbor>, [f64; 4], [[f64; 3]; 4]);

/// What a descriptor keeps of an atom between its components and the forces
/// they give once dE/dD is known.
enum Tape<'a> {
    /// se_e2_a: B, and of every neighbor its row, the Jacobian of the row,
    /// the embedding of its radial entry and the tangent of that.
    Angular {
        b: Vec<[f64; 4]>,
        records: Vec<(&'a Neighbor, [f64; 4], [[f64; 3]; 4], Vec<f64>, Vec<f64>)>,
    },
    /// se_e2_r: the scale of the mean, and of every neighbor the Jacobian
    /// of its radial entry and the tangent of its embedding.
    Radial {
        scale: f64,
        records: Vec<(&'a Neighbor, [f64; 3], Vec<f64>)>,
    },
    /// se_e3: the neighbors (type, direction entries and their Jacobian),
    /// and of every pair of them the two neighbors, the weight, θ, the
    /// embedding of θ and its tangent.
    ThreeBody {
        near: Vec<(usize, &'a Neighbor, [f64; 3], [[f64; 3]; 3])>,
        pairs: Vec<(usize, usize, f64, f64, Vec<f64>, Vec<f64>)>,
    },
}

/// The components of the descriptor `d` (one of a validated model, not a
/// hybrid) for an atom of type `kind` with the slots `rows`, added to the
/// end of `descriptor`; and what [`Tape::forces`] needs of them.
fn components<'a>(
    d: &Descriptor,
    kind: usize,
    rows: impl IntoIterator<Item = Row<'a>>,
    descriptor: &mut Vec<f64>,
) -> Result<Tape<'a>> {
    let width = d.embedding[kind][0].output_size();
    match d.descriptor_kind() {
        // D = B·B[:axis]ᵀ with B the mean over the slots of the embedding
        // row times the environment row.
        Some(DescriptorKind::SeE2A) => {
            let slots = d.sel.iter().sum::<usize>() as f64;
            let mut b = vec![[0.0; 4]; width];
            let mut records = Vec::new();
            for (species, neighbor, r, jac) in rows {
                let (g, dg) = d.embedding[kind][species].scalar_with_derivative(r[0])?;
                for m in 0..width {
                    for a in 0..4 {
                        b[m][a] += g[m] * r[a] / slots;
                    }
                }
                if let Some(n) = neighbor {
                    records.push((n, r, jac, g, dg));
                }
            }
            for m in 0..width {
                for k in 0..d.axis_neuron {
                    descriptor.push((0..4).map(|a| b[m][a] * b[k][a]).sum());
                }
            }
            Ok(Tape::Angular { b, records })
        }
        // D[m] = RADIAL_RESCALE / slots · sum over the slots of G[slot][m],
        // the embedding of the slot's normalized radial entry (an empty slot
        // has the normalized zero), as DeePMD-kit averages the rows of each
        // neighbor type and weights the types by their slots. Only the
        // radial entry of a row and its Jacobian are used.
        Some(DescriptorKind::SeE2R) => {
            let scale = RADIAL_RESCALE / d.sel.iter().sum::<usize>() as f64;
            let mut mean = vec![0.0; width];
            let mut records = Vec::new();
            for (species, neighbor, r, jac) in rows {
                let (g, dg) = d.embedding[kind][species].scalar_with_derivative(r[0])?;
                for m in 0..width {
                    mean[m] += scale * g[m];
                }
                if let Some(n) = neighbor {
                    records.push((n, jac[0], dg));
                }
            }
            descriptor.extend(mean);
            Ok(Tape::Radial { scale, records })
        }
        // With a_s the direction entries of the normalized environment row
        // of slot s, DeePMD-kit sums over the pairs of neighbor types t <= u
        // and over every slot j of type t and k of type u: θ = a_j·a_k goes
        // through the net of (t, u), and D[m] is the sum of θ·G[m](θ) /
        // (sel[t]·sel[u]). An empty slot has a zero direction and adds
        // nothing, so only neighbors are paired here, each pair once: within
        // one type DeePMD-kit's sum has a pair of two neighbors in both
        // orders, which is twice the pair, and a neighbor with itself once.
        Some(DescriptorKind::SeE3) => {
            // The neighbors: type, neighbor, direction entries and their Jacobian.
            let near: Vec<(usize, &Neighbor, [f64; 3], [[f64; 3]; 3])> = rows
                .into_iter()
                .filter_map(|(species, neighbor, r, jac)| {
                    neighbor.map(|n| (species, n, [r[1], r[2], r[3]], [jac[1], jac[2], jac[3]]))
                })
                .collect();
            let mut sum = vec![0.0; width];
            let mut pairs = Vec::new();
            for (j, &(tj, _, aj, _)) in near.iter().enumerate() {
                for (k, &(tk, _, ak, _)) in near.iter().enumerate().skip(j) {
                    let weight = three_body_weight(j == k, tj == tk, d.sel[tj], d.sel[tk]);
                    let theta: f64 = (0..3).map(|a| aj[a] * ak[a]).sum();
                    let (g, dg) = d.embedding[tj][tk].scalar_with_derivative(theta)?;
                    for m in 0..width {
                        sum[m] += weight * theta * g[m];
                    }
                    pairs.push((j, k, weight, theta, g, dg));
                }
            }
            descriptor.extend(sum);
            Ok(Tape::ThreeBody { near, pairs })
        }
        None => Err(Error(format!(
            "no arithmetic for the descriptor {:?}",
            d.kind
        ))),
    }
}

impl Tape<'_> {
    /// Adds to `result` the forces and the virial of the atom `center` that
    /// `dd` gives, dE/d(the components this tape was recorded for) of the
    /// descriptor `d`.
    fn forces(self, d: &Descriptor, dd: &[f64], center: usize, result: &mut Evaluation) {
        match self {
            Tape::Angular { b, records } => {
                let width = b.len();
                let slots = d.sel.iter().sum::<usize>() as f64;
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
            }
            Tape::Radial { scale, records } => {
                for (n, jac, dg) in records {
                    // dE/d(radial entry) of this neighbor, then along its vector.
                    let ds: f64 = dd.iter().zip(&dg).map(|(w, t)| w * t).sum::<f64>() * scale;
                    for c in 0..3 {
                        let grad = ds * jac[c];
                        result.forces[center][c] += grad;
                        result.forces[n.atom][c] -= grad;
                        for a in 0..3 {
                            result.virial[a][c] -= n.delta[a] * grad;
                        }
                    }
                }
            }
            Tape::ThreeBody { near, pairs } => {
                // dE/d(direction entries) of every neighbor: θ changes by a_k
                // when a_j moves, and by a_j when a_k does (twice a_j for a
                // pair of j with itself).
                let mut adjoint = vec![[0.0; 3]; near.len()];
                for (j, k, weight, theta, g, dg) in pairs {
                    let w = weight
                        * (0..dd.len())
                            .map(|m| dd[m] * (g[m] + theta * dg[m]))
                            .sum::<f64>();
                    for a in 0..3 {
                        adjoint[j][a] += w * near[k].2[a];
                        adjoint[k][a] += w * near[j].2[a];
                    }
                }
                for ((_, n, _, jac), da) in near.iter().zip(&adjoint) {
                    for c in 0..3 {
                        let grad: f64 = (0..3).map(|a| da[a] * jac[a][c]).sum();
                        result.forces[center][c] += grad;
                        result.forces[n.atom][c] -= grad;
                        for a in 0..3 {
                            result.virial[a][c] -= n.delta[a] * grad;
                        }
                    }
                }
            }
        }
    }
}

/// DeePMD-kit's `res_rescale` of se_e2_r: the mean of the embedding rows is
/// divided by five.
pub const RADIAL_RESCALE: f64 = 0.2;

/// What a pair of neighbors of an se_e3 atom counts in its descriptor: once
/// over the slots of its two types, and twice that for two different
/// neighbors of one type, which DeePMD-kit's sum holds in both orders.
pub fn three_body_weight(
    same_slot: bool,
    same_type: bool,
    sel_first: usize,
    sel_second: usize,
) -> f64 {
    let orders = if same_type && !same_slot { 2.0 } else { 1.0 };
    orders / (sel_first as f64 * sel_second as f64)
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
            // The neighbors of each type, nearest first: one list for all
            // descriptors of a hybrid, with the cutoff and the slots of the
            // model's descriptor.
            let neighbors =
                geometry.neighbors(frame, &types, center, &self.model.descriptor.sel)?;
            let neighbors = &neighbors;
            let kind = types[center] as usize;
            self.atom(
                center,
                kind,
                |d| {
                    // A descriptor of a hybrid has the nearest neighbors of each
                    // type in its own slots, and those at or beyond its own
                    // cutoff are empty slots to it. Like the list itself, it
                    // does not drop a neighbor within its cutoff here.
                    for (species, group) in neighbors.iter().enumerate() {
                        if group
                            .get(d.sel[species])
                            .is_some_and(|n| n.distance < d.rcut)
                        {
                            return Err(Error(format!(
                                "neighbor capacity overflow at atom {center}, type {}: more than sel={} within {} of a descriptor of the hybrid",
                                self.model.type_map[species], d.sel[species], d.rcut
                            )));
                        }
                    }
                    Ok(neighbors.iter().enumerate().flat_map(move |(species, group)| {
                        (0..d.sel[species]).map(move |slot| {
                            let n = group.get(slot).filter(|n| n.distance < d.rcut);
                            let (r, jac) = environment(n, &d.normalization[kind], d.rcut_smth, d.rcut);
                            (species, n, r, jac)
                        })
                    }))
                },
                &mut result,
            )?;
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

pub mod indexed_dataset;
