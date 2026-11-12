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
    /// `se_e2_a` or `se_e2_r`, with the `format` of the model that goes
    /// with it (see [`DescriptorKind`]).
    pub kind: String,
    pub rcut: f64,
    pub rcut_smth: f64,
    pub sel: Vec<usize>,
    /// se_e2_a: the first so many embedding outputs are the axis neurons.
    /// se_e2_r has none: zero, or left out.
    #[serde(default)]
    pub axis_neuron: usize,
    /// Per-center-type, uniform across neighbor slots. se_e2_r has the
    /// radial entry only: its other means are zero and its other widths one.
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

/// The descriptors a native model can have. Both take the environment of an
/// atom through one embedding net per pair of types, fed the normalized
/// radial entry s(r) = switch(r) / r of every neighbor slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorKind {
    /// Radial and angular information: with B the mean over the slots of
    /// the embedding row times the environment row, D = B·B[:axis]ᵀ.
    SeE2A,
    /// Radial information only: D is a fifth of the mean of the embedding
    /// rows over the slots (DeePMD-kit's `se_e2_r`, also called `se_r`).
    SeE2R,
}

impl DescriptorKind {
    /// What `descriptor.kind` says for it.
    pub fn name(self) -> &'static str {
        match self {
            Self::SeE2A => "se_e2_a",
            Self::SeE2R => "se_e2_r",
        }
    }

    /// The `format` of a native model with this descriptor.
    pub fn format(self) -> &'static str {
        match self {
            Self::SeE2A => "omnimd-native-se-e2-a",
            Self::SeE2R => "omnimd-native-se-e2-r",
        }
    }
}

impl Descriptor {
    /// The descriptor `kind` names, if the format has it.
    pub fn descriptor_kind(&self) -> Option<DescriptorKind> {
        [DescriptorKind::SeE2A, DescriptorKind::SeE2R]
            .into_iter()
            .find(|kind| kind.name() == self.kind)
    }

    /// Outputs of an embedding net (of a validated model: all alike).
    pub fn width(&self) -> usize {
        self.embedding[0][0].output_size()
    }

    /// Components of the descriptor, the input of a fitting net (of a
    /// validated model).
    pub fn features(&self) -> usize {
        match self.descriptor_kind() {
            Some(DescriptorKind::SeE2R) => self.width(),
            _ => self.width() * self.axis_neuron,
        }
    }
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
    /// The descriptor of a validated model.
    pub fn descriptor_kind(&self) -> DescriptorKind {
        self.descriptor
            .descriptor_kind()
            .unwrap_or(DescriptorKind::SeE2A)
    }

    /// Fails for a model with another descriptor than se_e2_a. `what` names
    /// the caller, whose arithmetic is that descriptor's only.
    pub fn require_se_e2_a(&self, what: &str) -> Result<()> {
        require(
            self.descriptor.kind == DescriptorKind::SeE2A.name(),
            format!(
                "{what} supports the se_e2_a descriptor only, not {}",
                self.descriptor.kind
            ),
        )
    }

    pub fn validate(&self) -> Result<ModelMetadata> {
        let formats = [DescriptorKind::SeE2A, DescriptorKind::SeE2R].map(DescriptorKind::format);
        require(
            formats.contains(&self.format.as_str()) && self.version == 1,
            "unsupported native format/version (expected omnimd-native-se-e2-a or omnimd-native-se-e2-r, v1)",
        )?;
        require(
            self.precision == "float64",
            "native v1 supports only float64 precision",
        )?;
        let d = &self.descriptor;
        // A format names its descriptor: the two must agree.
        let kind = d
            .descriptor_kind()
            .filter(|kind| kind.format() == self.format)
            .ok_or_else(|| {
                Error(format!(
                    "unsupported descriptor {:?} for native format {}: omnimd-native-se-e2-a has se_e2_a, omnimd-native-se-e2-r has se_e2_r",
                    d.kind, self.format
                ))
            })?;
        let metadata = ModelMetadata {
            type_map: self.type_map.clone(),
            cutoff: d.rcut,
            backend: match kind {
                DescriptorKind::SeE2A => "native-se-e2-a-v1".into(),
                DescriptorKind::SeE2R => "native-se-e2-r-v1".into(),
            },
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
            require(
                kind != DescriptorKind::SeE2R || norm.std[1] == 1.0,
                format!("normalization[{i}]: se_e2_r has a radial entry only; the other widths must be one"),
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
        let input = match kind {
            DescriptorKind::SeE2A => {
                require(
                    d.axis_neuron > 0 && d.axis_neuron <= width,
                    "axis_neuron must be in 1..=embedding width",
                )?;
                width
                    .checked_mul(d.axis_neuron)
                    .ok_or_else(|| Error("descriptor width overflow".into()))?
            }
            DescriptorKind::SeE2R => {
                require(
                    d.axis_neuron == 0,
                    "se_e2_r has no axis neurons: axis_neuron must be zero or left out",
                )?;
                width
            }
        };
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
