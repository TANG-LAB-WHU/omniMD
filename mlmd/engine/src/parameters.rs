//! Versioned engine context; the geometry/EFV shared contract stays unchanged.
use omnimd_ml_contract::{Error, Frame, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
pub const DATASET_SCHEMA: &str = "omnimd-dataset-jsonl-v2";
const MAX_DIM: usize = 1024;
const MAX_VALUES: usize = 1_000_000;
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Field {
    pub name: String,
    pub unit: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Layout {
    pub fparam: Vec<Field>,
    pub aparam: Vec<Field>,
}
impl Layout {
    pub fn validate(&self) -> Result<()> {
        if self.fparam.len() > MAX_DIM
            || self.aparam.len() > MAX_DIM
            || self.fparam.len() + self.aparam.len() == 0
        {
            return Err(Error("parameter layout requires nonzero total dimension and at most 1024 fields per category".into()));
        }
        for fields in [&self.fparam, &self.aparam] {
            let mut names = HashSet::new();
            for field in fields {
                if field.name.trim().is_empty()
                    || field.unit.trim().is_empty()
                    || field.name.len() > 128
                    || field.unit.len() > 128
                    || !names.insert(&field.name)
                {
                    return Err(Error("parameter names must be unique within each category; names/units nonempty and at most 128 bytes".into()));
                }
            }
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FramePolicy {
    #[default]
    Explicit,
    ModelDefault,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Values {
    #[serde(default)]
    pub frame_parameter_policy: FramePolicy,
    pub fparam: Vec<f64>,
    pub aparam: Vec<Vec<f64>>,
}
impl Values {
    pub fn validate(&self, layout: &Layout, natoms: usize) -> Result<()> {
        layout.validate()?;
        let df = layout.fparam.len();
        let da = layout.aparam.len();
        let expected = match self.frame_parameter_policy {
            FramePolicy::Explicit => df,
            FramePolicy::ModelDefault if df > 0 => 0,
            _ => {
                return Err(Error(
                    "model_default requires a positive declared frame-parameter dimension".into(),
                ))
            }
        };
        if self.fparam.len() != expected || self.fparam.iter().any(|v| !v.is_finite()) {
            return Err(Error("frame-parameter dimension/nonfinite error; defaults require empty fparam and explicit policy".into()));
        }
        let count = natoms
            .checked_mul(da)
            .and_then(|n| n.checked_add(df))
            .ok_or_else(|| Error("parameter size overflow".into()))?;
        if count > MAX_VALUES {
            return Err(Error(
                "parameter values exceed 1000000 per-frame bound".into(),
            ));
        }
        if (da == 0 && !self.aparam.is_empty())
            || (da > 0
                && (self.aparam.len() != natoms
                    || self
                        .aparam
                        .iter()
                        .any(|row| row.len() != da || row.iter().any(|v| !v.is_finite()))))
        {
            return Err(Error("atomic parameters require one finite row per input atom and exact declared width; no broadcast/defaults".into()));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixedParameters {
    pub schema: String,
    pub layout: Layout,
    pub values: Values,
    pub atom_species: Vec<String>,
}
impl FixedParameters {
    pub fn new(layout: Layout, values: Values, frame: &Frame) -> Result<Self> {
        let result = Self {
            schema: "omnimd-fixed-parameters-v1".into(),
            layout,
            values,
            atom_species: frame.species.clone(),
        };
        result.validate(frame)?;
        Ok(result)
    }
    pub fn validate(&self, frame: &Frame) -> Result<()> {
        if self.schema != "omnimd-fixed-parameters-v1" {
            return Err(Error("unsupported fixed-parameter schema".into()));
        }
        frame.validate()?;
        if self.atom_species != frame.species {
            return Err(Error(
                "fixed atomic parameter binding requires unchanged atom count and ordered species"
                    .into(),
            ));
        }
        self.values.validate(&self.layout, frame.positions.len())
    }
}
