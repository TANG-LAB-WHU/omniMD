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
    /// One evaluation per frame, each as [`Self::eval`] gives it. A backend
    /// that can share work between frames (one device pass for a batch)
    /// overrides this; the frames need not be alike. An error of any frame
    /// fails the batch.
    fn eval_batch(&mut self, frames: &[&Frame]) -> Result<Vec<Evaluation>> {
        frames.iter().map(|frame| self.eval(frame)).collect()
    }
}

/// Several models of one type map that are evaluated together, for their
/// disagreement on a frame (model deviation). As a [`Potential`] a committee
/// is its first member.
pub trait Committee: Potential {
    /// Number of models.
    fn members(&self) -> usize;
    /// The evaluation of every frame by every member: `[frame][member]`,
    /// each as the member alone would give it.
    fn eval_members(&mut self, frames: &[&Frame]) -> Result<Vec<Vec<Evaluation>>>;
}

/// Models that are evaluated one after the other: the committee of any
/// backend.
pub struct Separate<P: Potential>(pub Vec<P>);

impl<P: Potential> Potential for Separate<P> {
    fn metadata(&self) -> &ModelMetadata {
        self.0[0].metadata()
    }
    fn eval(&mut self, frame: &Frame) -> Result<Evaluation> {
        self.0[0].eval(frame)
    }
    fn eval_batch(&mut self, frames: &[&Frame]) -> Result<Vec<Evaluation>> {
        self.0[0].eval_batch(frames)
    }
}

impl<P: Potential> Separate<P> {
    /// At least one model; the models must share the ordered type map and
    /// the units.
    pub fn new(models: Vec<P>) -> Result<Self> {
        let Some(first) = models.first() else {
            return Err(Error("a committee needs at least one model".into()));
        };
        let reference = first.metadata();
        reference.validate()?;
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
        Ok(Self(models))
    }
}

impl<P: Potential> Committee for Separate<P> {
    fn members(&self) -> usize {
        self.0.len()
    }
    fn eval_members(&mut self, frames: &[&Frame]) -> Result<Vec<Vec<Evaluation>>> {
        let mut all: Vec<Vec<Evaluation>> = frames
            .iter()
            .map(|_| Vec::with_capacity(self.0.len()))
            .collect();
        for (index, model) in self.0.iter_mut().enumerate() {
            let evaluations = model
                .eval_batch(frames)
                .map_err(|e| Error(format!("ensemble member {index}: {e}")))?;
            if evaluations.len() != frames.len() {
                return Err(Error(format!(
                    "ensemble member {index} returned {} evaluations for {} frames",
                    evaluations.len(),
                    frames.len()
                )));
            }
            for (members, evaluation) in all.iter_mut().zip(evaluations) {
                members.push(evaluation);
            }
        }
        Ok(all)
    }
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
    /// Energy = its offset plus the first coordinate; counts its calls.
    struct Shifted {
        metadata: ModelMetadata,
        offset: f64,
        calls: usize,
    }
    impl Potential for Shifted {
        fn metadata(&self) -> &ModelMetadata {
            &self.metadata
        }
        fn eval(&mut self, frame: &Frame) -> Result<Evaluation> {
            self.calls += 1;
            let x = frame.positions[0][0];
            if x < 0.0 {
                return Err(Error(format!("negative coordinate {x}")));
            }
            Ok(Evaluation {
                energy: self.offset + x,
                forces: vec![[0.0; 3]; frame.positions.len()],
                virial: [[0.0; 3]; 3],
                units: Units::EvAngstrom,
            })
        }
    }
    fn shifted(offset: f64, types: &[&str]) -> Shifted {
        Shifted {
            metadata: ModelMetadata {
                type_map: types.iter().map(|s| s.to_string()).collect(),
                cutoff: 6.0,
                backend: "test".into(),
                precision: "float64".into(),
                units: Units::EvAngstrom,
            },
            offset,
            calls: 0,
        }
    }
    fn frames_at(xs: &[f64]) -> Vec<Frame> {
        xs.iter()
            .map(|&x| {
                let mut f = frame();
                f.positions[0][0] = x;
                f
            })
            .collect()
    }
    #[test]
    fn a_batch_is_each_frame_in_turn_unless_a_backend_says_otherwise() {
        let frames = frames_at(&[1.0, 2.0, 4.0]);
        let batch: Vec<&Frame> = frames.iter().collect();
        let mut model = shifted(10.0, &["H", "O"]);
        let energies: Vec<f64> = model
            .eval_batch(&batch)
            .unwrap()
            .iter()
            .map(|e| e.energy)
            .collect();
        assert_eq!(energies, [11.0, 12.0, 14.0]);
        assert_eq!(model.calls, 3);
        assert!(model.eval_batch(&[]).unwrap().is_empty());
        // The first error ends the batch.
        let bad = frames_at(&[1.0, -1.0, 4.0]);
        let batch: Vec<&Frame> = bad.iter().collect();
        let error = model.eval_batch(&batch).unwrap_err();
        assert!(error.0.contains("negative coordinate"));
        assert_eq!(model.calls, 5);
    }
    #[test]
    fn separate_models_are_a_committee_frame_by_member() {
        let types = ["H", "O"];
        let models = vec![
            shifted(0.0, &types),
            shifted(10.0, &types),
            shifted(20.0, &types),
        ];
        let mut committee = Separate::new(models).unwrap();
        assert_eq!(committee.members(), 3);
        let frames = frames_at(&[1.0, 2.0]);
        let batch: Vec<&Frame> = frames.iter().collect();
        let all = committee.eval_members(&batch).unwrap();
        let energies: Vec<Vec<f64>> = all
            .iter()
            .map(|members| members.iter().map(|e| e.energy).collect())
            .collect();
        assert_eq!(energies, [[1.0, 11.0, 21.0], [2.0, 12.0, 22.0]]);
        assert!(committee.0.iter().all(|model| model.calls == 2));
        assert!(committee.eval_members(&[]).unwrap().is_empty());
        // As a potential it is its first member.
        assert_eq!(committee.eval(&frames[1]).unwrap().energy, 2.0);
        assert_eq!(committee.eval_batch(&batch).unwrap()[0].energy, 1.0);
        assert_eq!(committee.metadata().backend, "test");
        assert_eq!(
            committee.0.iter().map(|m| m.calls).collect::<Vec<_>>(),
            [5, 2, 2]
        );
        // An error names the member.
        let bad = frames_at(&[-3.0]);
        let error = committee.eval_members(&[&bad[0]]).unwrap_err();
        assert!(error.0.starts_with("ensemble member 0:"), "{error}");
        // The members share type map and units; there is at least one.
        assert!(Separate::new(vec![shifted(0.0, &types), shifted(1.0, &["O", "H"])]).is_err());
        assert!(Separate::new(vec![shifted(0.0, &types), shifted(1.0, &["H", "Xx"])]).is_err());
        assert!(Separate::<Shifted>::new(Vec::new()).is_err());
        assert_eq!(
            Separate::new(vec![shifted(0.0, &types)]).unwrap().members(),
            1
        );
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
