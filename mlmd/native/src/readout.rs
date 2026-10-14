//! Exact affine readout features using analytic native basis evaluations.
//! This is not finite-difference parameter differentiation.
use crate::{Model, NativePotential};
use omnimd_ml_contract::{Error, Evaluation, Frame, Potential, Result, Units};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "coefficient", rename_all = "snake_case")]
pub enum ReadoutParameter {
    Weight { species: usize, input: usize },
    Bias { species: usize },
}
impl ReadoutParameter {
    fn get(&self, model: &Model) -> f64 {
        match *self {
            Self::Weight { species, input } => {
                model.fitting[species].layers.last().unwrap().weights[input][0]
            }
            Self::Bias { species } => model.fitting[species].layers.last().unwrap().bias[0],
        }
    }
    fn set(&self, model: &mut Model, value: f64) {
        match *self {
            Self::Weight { species, input } => {
                model.fitting[species].layers.last_mut().unwrap().weights[input][0] = value
            }
            Self::Bias { species } => {
                model.fitting[species].layers.last_mut().unwrap().bias[0] = value
            }
        }
    }
}

/// One column per final-layer coefficient. Offsets are separate to avoid
/// subtracting large atomic energies when forming small hidden features.
#[derive(Debug, Clone)]
pub struct ReadoutFeatures {
    pub fixed_energy: f64,
    pub columns: Vec<Evaluation>,
}
impl ReadoutFeatures {
    pub fn evaluate(&self, coefficients: &[f64]) -> Result<Evaluation> {
        if self.columns.is_empty()
            || coefficients.len() != self.columns.len()
            || coefficients.iter().any(|x| !x.is_finite())
        {
            return Err(Error("invalid readout coefficient vector".into()));
        }
        let natoms = self.columns[0].forces.len();
        let mut result = Evaluation {
            energy: self.fixed_energy,
            forces: vec![[0.0; 3]; natoms],
            virial: [[0.0; 3]; 3],
            units: Units::EvAngstrom,
        };
        for (column, weight) in self.columns.iter().zip(coefficients) {
            column.validate(natoms)?;
            result.energy += weight * column.energy;
            for i in 0..natoms {
                for a in 0..3 {
                    result.forces[i][a] += weight * column.forces[i][a];
                }
            }
            for a in 0..3 {
                for b in 0..3 {
                    result.virial[a][b] += weight * column.virial[a][b];
                }
            }
        }
        result.validate(natoms)?;
        Ok(result)
    }
}

pub struct ReadoutBasis {
    original: Model,
    basis: NativePotential,
    offsets: Vec<f64>,
    pub parameters: Vec<ReadoutParameter>,
    pub initial_coefficients: Vec<f64>,
}
impl ReadoutBasis {
    pub fn new(model: &Model, fit_bias: bool) -> Result<Self> {
        model.validate()?;
        let mut parameters = Vec::new();
        let mut zero = model.clone();
        let mut offsets = model.atom_energy_bias.clone();
        for (kind, net) in zero.fitting.iter_mut().enumerate() {
            let last = net.layers.last_mut().unwrap();
            for (input, row) in last.weights.iter_mut().enumerate() {
                parameters.push(ReadoutParameter::Weight {
                    species: kind,
                    input,
                });
                row[0] = 0.0;
            }
            if fit_bias {
                parameters.push(ReadoutParameter::Bias { species: kind });
            } else {
                offsets[kind] += last.bias[0];
            }
            last.bias[0] = 0.0;
        }
        if parameters.len() > 256 {
            return Err(Error(
                "small readout trainer supports at most 256 coefficients".into(),
            ));
        }
        zero.atom_energy_bias.fill(0.0);
        let initial_coefficients = parameters.iter().map(|p| p.get(model)).collect();
        Ok(Self {
            original: model.clone(),
            basis: NativePotential::new(zero)?,
            offsets,
            parameters,
            initial_coefficients,
        })
    }
    pub fn features(&mut self, frame: &Frame) -> Result<ReadoutFeatures> {
        let types = frame.type_ids(self.basis.metadata())?;
        let fixed_energy: f64 = types.iter().map(|t| self.offsets[*t as usize]).sum();
        if !fixed_energy.is_finite() {
            return Err(Error("nonfinite fixed readout offset".into()));
        }
        let mut columns = Vec::with_capacity(self.parameters.len());
        for parameter in &self.parameters {
            parameter.set(&mut self.basis.model, 1.0);
            let column = self.basis.eval(frame);
            parameter.set(&mut self.basis.model, 0.0); // also reset after failed eval
            columns.push(column?);
        }
        Ok(ReadoutFeatures {
            fixed_energy,
            columns,
        })
    }
    pub fn model_with_coefficients(&self, coefficients: &[f64]) -> Result<Model> {
        if coefficients.len() != self.parameters.len() {
            return Err(Error("readout coefficient count mismatch".into()));
        }
        let mut model = self.original.clone();
        for (parameter, value) in self.parameters.iter().zip(coefficients) {
            parameter.set(&mut model, *value);
        }
        model.validate()?;
        Ok(model)
    }
}
