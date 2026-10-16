//! Exact joint energy/force/optional-virial parameter gradient for tanh se_e2_a networks.
use crate::{
    autodiff::{Dual, Id, Tape},
    full_loss::{validate_loss, validate_record, LossSeeds, VirialLoss},
    neighbor::Geometry,
    training::TrainConfig,
    training_data::DatasetFrame,
    Activation, Model, NativePotential, Network, Residual,
};
use omnimd_ml_contract::{Error, Potential, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "network", rename_all = "snake_case", deny_unknown_fields)]
pub enum NetworkLocation {
    Embedding { center: usize, neighbor: usize },
    Fitting { species: usize },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "field", rename_all = "snake_case", deny_unknown_fields)]
pub enum ParameterField {
    Weight { input: usize, output: usize },
    Bias { output: usize },
    Timestep { output: usize },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Parameter {
    pub location: NetworkLocation,
    pub layer: usize,
    pub field: ParameterField,
}
impl Parameter {
    fn network<'a>(&self, model: &'a Model) -> &'a Network {
        match self.location {
            NetworkLocation::Embedding { center, neighbor } => {
                &model.descriptor.embedding[center][neighbor]
            }
            NetworkLocation::Fitting { species } => &model.fitting[species],
        }
    }
    pub fn value(&self, model: &Model) -> f64 {
        let layer = &self.network(model).layers[self.layer];
        match self.field {
            ParameterField::Weight { input, output } => layer.weights[input][output],
            ParameterField::Bias { output } => layer.bias[output],
            ParameterField::Timestep { output } => layer.timestep.as_ref().unwrap()[output],
        }
    }
    fn set(&self, model: &mut Model, value: f64) {
        let net = match self.location {
            NetworkLocation::Embedding { center, neighbor } => {
                &mut model.descriptor.embedding[center][neighbor]
            }
            NetworkLocation::Fitting { species } => &mut model.fitting[species],
        };
        let layer = &mut net.layers[self.layer];
        match self.field {
            ParameterField::Weight { input, output } => layer.weights[input][output] = value,
            ParameterField::Bias { output } => layer.bias[output] = value,
            ParameterField::Timestep { output } => layer.timestep.as_mut().unwrap()[output] = value,
        }
    }
}
#[derive(Clone)]
pub struct ParameterLayout {
    pub parameters: Vec<Parameter>,
}
impl ParameterLayout {
    pub fn new(model: &Model) -> Result<Self> {
        model.validate()?;
        if model.descriptor.sel.iter().sum::<usize>() > 128 {
            return Err(Error("full trainer supports sum(sel)<=128".into()));
        }
        let mut parameters = Vec::new();
        let mut add = |location: NetworkLocation, net: &Network, fitting: bool| -> Result<()> {
            for (i, layer) in net.layers.iter().enumerate() {
                let final_linear = fitting && i + 1 == net.layers.len();
                if !(matches!(layer.activation, Activation::Tanh) && !final_linear
                    || matches!(layer.activation, Activation::Linear) && final_linear)
                {
                    return Err(Error("full trainer requires tanh embedding/hidden layers and a linear fitting output".into()));
                }
                for (input, row) in layer.weights.iter().enumerate() {
                    for output in 0..row.len() {
                        parameters.push(Parameter {
                            location: location.clone(),
                            layer: i,
                            field: ParameterField::Weight { input, output },
                        });
                    }
                }
                for output in 0..layer.bias.len() {
                    parameters.push(Parameter {
                        location: location.clone(),
                        layer: i,
                        field: ParameterField::Bias { output },
                    });
                }
                if let Some(dt) = &layer.timestep {
                    for output in 0..dt.len() {
                        parameters.push(Parameter {
                            location: location.clone(),
                            layer: i,
                            field: ParameterField::Timestep { output },
                        });
                    }
                }
            }
            Ok(())
        };
        for (center, row) in model.descriptor.embedding.iter().enumerate() {
            for (neighbor, net) in row.iter().enumerate() {
                add(NetworkLocation::Embedding { center, neighbor }, net, false)?;
            }
        }
        for (species, net) in model.fitting.iter().enumerate() {
            add(NetworkLocation::Fitting { species }, net, true)?;
        }
        if parameters.len() > 10_000 {
            return Err(Error(
                "full trainer supports at most 10000 neural parameters".into(),
            ));
        }
        Ok(Self { parameters })
    }
    pub fn values(&self, model: &Model) -> Vec<f64> {
        self.parameters.iter().map(|p| p.value(model)).collect()
    }
    pub fn apply(&self, model: &mut Model, values: &[f64]) -> Result<()> {
        if values.len() != self.parameters.len() || values.iter().any(|x| !x.is_finite()) {
            return Err(Error("invalid full parameter vector".into()));
        }
        for (p, &x) in self.parameters.iter().zip(values) {
            p.set(model, x);
        }
        model.validate()?;
        Ok(())
    }
    pub fn sha256(&self) -> String {
        format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&self.parameters).unwrap())
        )
    }
}
struct LayerIds {
    weights: Vec<Vec<Id>>,
    bias: Vec<Id>,
    timestep: Option<Vec<Id>>,
    activation: Activation,
    residual: Residual,
}
struct NetIds {
    layers: Vec<LayerIds>,
}
impl NetIds {
    fn build(net: &Network, ids: &[Id], cursor: &mut usize) -> Self {
        let mut layers = Vec::new();
        for layer in &net.layers {
            let mut weights = Vec::new();
            for row in &layer.weights {
                let mut w = Vec::new();
                for _ in row {
                    w.push(ids[*cursor]);
                    *cursor += 1;
                }
                weights.push(w);
            }
            let bias = (0..layer.bias.len())
                .map(|_| {
                    let id = ids[*cursor];
                    *cursor += 1;
                    id
                })
                .collect();
            let timestep = layer.timestep.as_ref().map(|dt| {
                (0..dt.len())
                    .map(|_| {
                        let id = ids[*cursor];
                        *cursor += 1;
                        id
                    })
                    .collect()
            });
            layers.push(LayerIds {
                weights,
                bias,
                timestep,
                activation: layer.activation,
                residual: layer.residual,
            });
        }
        Self { layers }
    }
    fn forward(&self, tape: &mut Tape, input: &[Dual]) -> Vec<Dual> {
        let mut x = input.to_vec();
        for layer in &self.layers {
            let mut y = Vec::new();
            for j in 0..layer.bias.len() {
                let mut z = Dual {
                    value: layer.bias[j],
                    tangent: 0,
                };
                for (i, value) in x.iter().enumerate() {
                    let product = tape.dmul(
                        *value,
                        Dual {
                            value: layer.weights[i][j],
                            tangent: 0,
                        },
                    );
                    z = tape.dadd(z, product);
                }
                if matches!(layer.activation, Activation::Tanh) {
                    z = tape.dtanh(z);
                }
                if let Some(dt) = &layer.timestep {
                    z = tape.dmul(
                        z,
                        Dual {
                            value: dt[j],
                            tangent: 0,
                        },
                    );
                }
                if !matches!(layer.residual, Residual::None) {
                    z = tape.dadd(z, x[j % x.len()]);
                }
                y.push(z);
            }
            x = y;
        }
        x
    }
}

/// Seeds are detached loss cotangents. Environment tangent is jac*(v_j-v_i).
/// Reverse over this forward tangent returns a E_theta - (v.grad_r E)_theta.
pub fn frame_gradient(
    model: &Model,
    record: &DatasetFrame,
    loss: &TrainConfig,
    normalizing_frames: usize,
) -> Result<(f64, Vec<f64>)> {
    frame_gradient_with_virial(model, record, loss, None, normalizing_frames)
}
/// Adds the full row-strain virial cotangent to the environment direction.
pub fn frame_gradient_with_virial(
    model: &Model,
    record: &DatasetFrame,
    loss: &TrainConfig,
    virial: Option<&VirialLoss>,
    normalizing_frames: usize,
) -> Result<(f64, Vec<f64>)> {
    validate_loss(loss, virial)?;
    validate_record(record, virial)?;
    if normalizing_frames == 0 {
        return Err(Error(
            "gradient normalization requires positive frame count".into(),
        ));
    }
    let layout = ParameterLayout::new(model)?;
    let mut potential = NativePotential::new(model.clone())?;
    let predicted = potential.eval(&record.frame)?;
    let n = record.frame.positions.len();
    if n > 64 {
        return Err(Error(
            "full trainer supports at most 64 atoms per frame".into(),
        ));
    }
    let seeds = LossSeeds::new(&predicted, record, loss, virial, normalizing_frames)?;
    let mut tape = Tape::new();
    let parameter_ids: Vec<_> = layout
        .values(model)
        .into_iter()
        .map(|x| tape.leaf(x))
        .collect();
    let mut cursor = 0;
    let embedding: Vec<Vec<_>> = model
        .descriptor
        .embedding
        .iter()
        .map(|row| {
            row.iter()
                .map(|net| NetIds::build(net, &parameter_ids, &mut cursor))
                .collect()
        })
        .collect();
    let fitting: Vec<_> = model
        .fitting
        .iter()
        .map(|net| NetIds::build(net, &parameter_ids, &mut cursor))
        .collect();
    debug_assert_eq!(cursor, parameter_ids.len());
    let d = &model.descriptor;
    let types = record.frame.type_ids(potential.metadata())?;
    let geometry = Geometry::new(&record.frame, d.rcut)?;
    let slots = d.sel.iter().sum::<usize>() as f64;
    let width = d.embedding[0][0].output_size();
    let mut total = Dual {
        value: 0,
        tangent: 0,
    };
    for center in 0..n {
        let kind = types[center] as usize;
        let neighbors = geometry.neighbors(&record.frame, &types, center, &d.sel)?;
        let mut b = vec![
            [Dual {
                value: 0,
                tangent: 0
            }; 4];
            width
        ];
        for (species, group) in neighbors.iter().enumerate() {
            for slot in 0..d.sel[species] {
                let neighbor = group.get(slot);
                let (r, jac) =
                    crate::environment(neighbor, &d.normalization[kind], d.rcut_smth, d.rcut);
                let mut row = [Dual {
                    value: 0,
                    tangent: 0,
                }; 4];
                for a in 0..4 {
                    let tangent = neighbor.map_or(0.0, |q| seeds.tangent(jac[a], q, center));
                    row[a] = tape.dual_constant(r[a], tangent);
                }
                let g = embedding[kind][species].forward(&mut tape, &[row[0]]);
                for m in 0..width {
                    for a in 0..4 {
                        let product = tape.dmul(g[m], row[a]);
                        let scaled = tape.dscale(product, 1.0 / slots);
                        b[m][a] = tape.dadd(b[m][a], scaled);
                    }
                }
            }
        }
        let mut descriptor = Vec::with_capacity(width * d.axis_neuron);
        for m in 0..width {
            for k in 0..d.axis_neuron {
                let mut value = Dual {
                    value: 0,
                    tangent: 0,
                };
                for a in 0..4 {
                    let product = tape.dmul(b[m][a], b[k][a]);
                    value = tape.dadd(value, product);
                }
                descriptor.push(value);
            }
        }
        let atomic = fitting[kind].forward(&mut tape, &descriptor)[0];
        let offset = tape.dual_constant(model.atom_energy_bias[kind], 0.0);
        let energy = tape.dadd(atomic, offset);
        total = tape.dadd(total, energy);
    }
    if (tape.value(total.value) - predicted.energy).abs() > 1e-11 * (1.0 + predicted.energy.abs()) {
        return Err(Error("AD primal disagrees with native inference".into()));
    }
    let adj = tape.reverse(&[(total.value, seeds.energy), (total.tangent, -1.0)])?;
    let gradient = parameter_ids.iter().map(|id| adj[*id]).collect();
    if !seeds.objective.is_finite() {
        return Err(Error("nonfinite full-network objective".into()));
    }
    Ok((seeds.objective, gradient))
}
