use omnimd_ml_contract::{Error, ModelMetadata, Result, Units};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub format: String,
    pub version: u32,
    pub precision: String,
    pub units: Units,
    pub type_map: Vec<String>,
    pub descriptor: Descriptor,
    pub fitting: Vec<Network>,
    pub atom_energy_bias: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Descriptor {
    pub kind: String,
    pub rcut: f64,
    pub rcut_smth: f64,
    pub sel: Vec<usize>,
    pub axis_neuron: usize,
    /// Per-center-type, uniform across neighbor slots.
    pub normalization: Vec<Normalization>,
    /// Ordered [central type][neighbor type]. One-side models repeat rows.
    pub embedding: Vec<Vec<Network>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Normalization {
    pub mean: [f64; 4],
    pub std: [f64; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Network {
    pub layers: Vec<Layer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Layer {
    /// Input-major matrix: weights[input][output].
    pub weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
    pub activation: Activation,
    pub residual: Residual,
    /// Multiplies the activated branch, before the residual addition.
    pub timestep: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Linear,
    Tanh,
    Sigmoid,
    Softplus,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Residual {
    None,
    Identity,
    Concat2,
}

fn require(ok: bool, text: impl Into<String>) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error(text.into()))
    }
}
fn finite(mut values: impl Iterator<Item = f64>) -> bool {
    values.all(f64::is_finite)
}

impl Model {
    pub fn validate(&self) -> Result<ModelMetadata> {
        require(
            self.format == "omnimd-native-se-e2-a" && self.version == 1,
            "unsupported native format/version (expected omnimd-native-se-e2-a v1)",
        )?;
        require(
            self.precision == "float64",
            "native v1 supports only float64 precision",
        )?;
        let d = &self.descriptor;
        require(
            d.kind == "se_e2_a",
            "unsupported descriptor: native v1 requires se_e2_a",
        )?;
        let metadata = ModelMetadata {
            type_map: self.type_map.clone(),
            cutoff: d.rcut,
            backend: "native-se-e2-a-v1".into(),
            precision: self.precision.clone(),
            units: self.units,
        };
        metadata.validate()?;
        let n = self.type_map.len();
        require(n <= 256, "native v1 supports at most 256 types")?;
        require(
            d.rcut_smth.is_finite()
                && d.rcut_smth >= 0.0
                && d.rcut_smth < d.rcut
                && (d.rcut * d.rcut).is_finite(),
            "invalid smoothing/cutoff radii",
        )?;
        require(d.sel.len() == n, "sel must have one entry per ordered type")?;
        let slots = d
            .sel
            .iter()
            .try_fold(0usize, |a, b| a.checked_add(*b))
            .ok_or_else(|| Error("sel sum overflow".into()))?;
        require(slots > 0 && slots <= 65536, "sum(sel) must be in 1..=65536")?;
        require(
            d.normalization.len() == n,
            "normalization must have one entry per central type",
        )?;
        for (i, norm) in d.normalization.iter().enumerate() {
            require(
                finite(norm.mean.into_iter()) && norm.std.iter().all(|s| s.is_finite() && *s > 0.0),
                format!("normalization[{i}] must be finite with positive std"),
            )?;
            require(
                norm.mean[1..] == [0.0; 3]
                    && norm.std[1] == norm.std[2]
                    && norm.std[2] == norm.std[3],
                format!("normalization[{i}]: angular mean must be zero and angular std isotropic"),
            )?;
        }
        require(
            d.embedding.len() == n,
            "embedding must have one row per central type",
        )?;
        let mut width = None;
        for (i, row) in d.embedding.iter().enumerate() {
            require(
                row.len() == n,
                format!("embedding[{i}] must have one network per neighbor type"),
            )?;
            for (j, net) in row.iter().enumerate() {
                let out = net.validate(1, &format!("embedding[{i}][{j}]"))?;
                if let Some(expected) = width {
                    require(
                        out == expected,
                        "all embedding networks must have the same output width",
                    )?;
                } else {
                    width = Some(out);
                }
            }
        }
        let width = width.unwrap();
        require(
            d.axis_neuron > 0 && d.axis_neuron <= width,
            "axis_neuron must be in 1..=embedding width",
        )?;
        let input = width
            .checked_mul(d.axis_neuron)
            .ok_or_else(|| Error("descriptor width overflow".into()))?;
        require(
            self.fitting.len() == n && self.atom_energy_bias.len() == n,
            "fitting and atom_energy_bias must have one entry per central type",
        )?;
        require(
            finite(self.atom_energy_bias.iter().copied()),
            "nonfinite atomic energy bias",
        )?;
        for (i, net) in self.fitting.iter().enumerate() {
            require(
                net.validate(input, &format!("fitting[{i}]"))? == 1,
                "fitting output must be scalar",
            )?;
            let last = net.layers.last().unwrap();
            require(
                matches!(last.activation, Activation::Linear)
                    && matches!(last.residual, Residual::None)
                    && last.timestep.is_none(),
                "fitting output must be linear with no residual or timestep",
            )?;
        }
        Ok(metadata)
    }
}

impl Network {
    pub(crate) fn output_size(&self) -> usize {
        self.layers.last().unwrap().bias.len()
    }

    fn validate(&self, mut input: usize, context: &str) -> Result<usize> {
        require(
            !self.layers.is_empty() && self.layers.len() <= 128,
            format!("{context}: layer count must be in 1..=128"),
        )?;
        for (i, layer) in self.layers.iter().enumerate() {
            let output = layer.bias.len();
            require(
                output > 0
                    && output <= 65536
                    && layer.weights.len() == input
                    && layer.weights.iter().all(|row| row.len() == output),
                format!(
                    "{context} layer {i}: expected weights[{input}][{output}] and nonempty bias"
                ),
            )?;
            require(
                finite(layer.weights.iter().flatten().copied())
                    && finite(layer.bias.iter().copied()),
                format!("{context} layer {i}: nonfinite weights/bias"),
            )?;
            require(
                match layer.residual {
                    Residual::None => true,
                    Residual::Identity => input == output,
                    Residual::Concat2 => input.checked_mul(2) == Some(output),
                },
                format!("{context} layer {i}: residual dimensions do not match"),
            )?;
            if let Some(timestep) = &layer.timestep {
                require(
                    timestep.len() == output && finite(timestep.iter().copied()),
                    format!("{context} layer {i}: invalid timestep vector"),
                )?;
            }
            input = output;
        }
        Ok(input)
    }
}
