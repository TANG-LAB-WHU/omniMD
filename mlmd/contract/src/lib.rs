//! Strict, unit-explicit model evaluation shared by the FFI and native backends.
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, fmt};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Units {
    EvAngstrom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Frame {
    pub positions: Vec<[f64; 3]>,
    pub species: Vec<String>,
    pub cell: Option<[[f64; 3]; 3]>,
    pub units: Units,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Evaluation {
    pub energy: f64,
    pub forces: Vec<[f64; 3]>,
    pub virial: [[f64; 3]; 3],
    pub units: Units,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelMetadata {
    pub type_map: Vec<String>,
    pub cutoff: f64,
    pub backend: String,
    pub precision: String,
    pub units: Units,
}

pub trait Potential {
    fn metadata(&self) -> &ModelMetadata;
    fn eval(&mut self, frame: &Frame) -> Result<Evaluation>;
}

const ELEMENTS: &str = "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og";

impl ModelMetadata {
    pub fn validate(&self) -> Result<()> {
        if self.type_map.is_empty() || self.type_map.len() > i32::MAX as usize {
            return Err(Error("model type map is empty or too large".into()));
        }
        let mut seen = HashSet::new();
        for name in &self.type_map {
            if !ELEMENTS.split_whitespace().any(|s| s == name) || !seen.insert(name) {
                return Err(Error(format!(
                    "invalid or duplicate element in type map: {name}"
                )));
            }
        }
        if !self.cutoff.is_finite() || self.cutoff <= 0.0 {
            return Err(Error("model cutoff must be finite and positive".into()));
        }
        Ok(())
    }
}

impl Frame {
    pub fn validate(&self) -> Result<()> {
        let n = self.positions.len();
        if n == 0 || n > (i32::MAX as usize) / 9 || n.checked_mul(3).is_none() {
            return Err(Error(
                "atom count must be positive and all C buffer sizes must fit C int".into(),
            ));
        }
        if n != self.species.len() {
            return Err(Error("positions/species count mismatch".into()));
        }
        if self.positions.iter().flatten().any(|v| !v.is_finite()) {
            return Err(Error("coordinates contain nonfinite values".into()));
        }
        if let Some(h) = &self.cell {
            if h.iter().flatten().any(|v| !v.is_finite()) {
                return Err(Error("cell contains nonfinite values".into()));
            }
            let scale = h.iter().flatten().fold(0.0_f64, |s, v| s.max(v.abs()));
            if scale == 0.0 {
                return Err(Error("cell is singular".into()));
            }
            let a = h.map(|r| r.map(|v| v / scale));
            let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
            if !det.is_finite() || det <= 1e-12 {
                return Err(Error(
                    "cell must be right handed and well conditioned".into(),
                ));
            }
        }
        Ok(())
    }

    pub fn type_ids(&self, model: &ModelMetadata) -> Result<Vec<i32>> {
        self.validate()?;
        model.validate()?;
        self.species
            .iter()
            .map(|name| {
                model
                    .type_map
                    .iter()
                    .position(|s| s == name)
                    .map(|i| i as i32)
                    .ok_or_else(|| Error(format!("element {name:?} is not in model type map")))
            })
            .collect()
    }
}

impl Evaluation {
    pub fn validate(&self, natoms: usize) -> Result<()> {
        if self.forces.len() != natoms {
            return Err(Error("force shape does not match atom count".into()));
        }
        if !self.energy.is_finite()
            || self
                .forces
                .iter()
                .flatten()
                .chain(self.virial.iter().flatten())
                .any(|v| !v.is_finite())
        {
            return Err(Error("model output is missing or nonfinite".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn frame() -> Frame {
        Frame {
            positions: vec![[0.0; 3]; 2],
            species: vec!["H".into(), "O".into()],
            cell: None,
            units: Units::EvAngstrom,
        }
    }
    #[test]
    fn mapping_is_model_order_and_unknown_is_error() {
        let m = ModelMetadata {
            type_map: vec!["O".into(), "H".into()],
            cutoff: 6.0,
            backend: "test".into(),
            precision: "float64".into(),
            units: Units::EvAngstrom,
        };
        assert_eq!(frame().type_ids(&m).unwrap(), vec![1, 0]);
        let mut f = frame();
        f.species[0] = "X".into();
        assert!(f.type_ids(&m).is_err());
    }
    #[test]
    fn rejects_invalid_shapes_cells_and_outputs() {
        let mut f = frame();
        f.positions[0][0] = f64::NAN;
        assert!(f.validate().is_err());
        f = frame();
        f.cell = Some([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]);
        assert!(f.validate().is_err());
        f.cell = Some([[2.0, 0.0, 0.0], [0.2, 2.0, 0.0], [0.1, 0.3, 2.0]]);
        assert!(f.validate().is_ok());
        let e = Evaluation {
            energy: 0.0,
            forces: vec![[0.0; 3]; 2],
            virial: [[f64::NAN; 3]; 3],
            units: Units::EvAngstrom,
        };
        assert!(e.validate(2).is_err());
    }
}
