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
    /// `se_e2_a`, `se_e2_r` or `se_e3`, with the `format` of the model that
    /// goes with it (see [`DescriptorKind`]), or `hybrid` ([`HYBRID_KIND`]):
    /// the descriptors of `list` one after the other.
    pub kind: String,
    /// hybrid: the neighbor list that every descriptor of `list` reads has
    /// the largest cutoff of them and, of every type, the most slots
    /// (`rcut_smth` is the smallest of theirs; nothing reads it).
    pub rcut: f64,
    pub rcut_smth: f64,
    pub sel: Vec<usize>,
    /// se_e2_a: the first so many embedding outputs are the axis neurons.
    /// se_e2_r and se_e3 have none: zero, or left out.
    #[serde(default)]
    pub axis_neuron: usize,
    /// Per-center-type, uniform across neighbor slots. se_e2_r has the
    /// radial entry only: its other means are zero and its other widths one.
    /// se_e3 has the direction entries only: its radial mean is zero and its
    /// radial width one. A hybrid has none of its own.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub normalization: Vec<Normalization>,
    /// The embedding nets depend on the neighbor type only (DeePMD's
    /// `type_one_side`): every row of `embedding` is then the same, and the
    /// GPU trainer keeps it so (the bounded trainers train the net of every
    /// pair on its own and drop the mark). Left out or false, the nets of
    /// every pair of types are their own, alike or not. Not for se_e3,
    /// whose nets do not depend on the central type at all.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub type_one_side: bool,
    /// Ordered [central type][neighbor type]. One-side models repeat rows.
    /// se_e3: [neighbor type][neighbor type], the net of a pair of neighbor
    /// types, the same in either order. A hybrid has none of its own.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub embedding: Vec<Vec<Network>>,
    /// hybrid: the descriptors whose components, in this order, are the
    /// components of this one. Each is an se_e2_a, se_e2_r or se_e3
    /// descriptor with cutoffs, slots, normalization and nets of its own.
    /// Empty, or left out, for the others.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub list: Vec<Descriptor>,
}

/// `descriptor.kind` of a hybrid (DeePMD-kit's `hybrid`): the components of
/// several descriptors, one after the other, are the input of the fitting
/// nets. All of them read one neighbor list, made with the largest of their
/// cutoffs and, of every type, the most slots any of them has: a descriptor
/// takes the nearest `sel[t]` of the list's neighbors of type `t` (its own
/// `sel`), and those at or beyond its own cutoff are empty slots to it.
pub const HYBRID_KIND: &str = "hybrid";
/// The `format` of a native model with a hybrid descriptor.
pub const HYBRID_FORMAT: &str = "omnimd-native-hybrid";
/// The most descriptors a hybrid can have.
pub const MAX_HYBRID_PARTS: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Normalization {
    pub mean: [f64; 4],
    pub std: [f64; 4],
}

/// The descriptors a native model can have. The first two take the
/// environment of an atom through one embedding net per pair of central and
/// neighbor type, fed the normalized radial entry s(r) = switch(r) / r of
/// every neighbor slot; the third through one net per pair of neighbor
/// types, fed a product of two neighbors' directions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorKind {
    /// Radial and angular information: with B the mean over the slots of
    /// the embedding row times the environment row, D = B·B[:axis]ᵀ.
    SeE2A,
    /// Radial information only: D is a fifth of the mean of the embedding
    /// rows over the slots (DeePMD-kit's `se_e2_r`, also called `se_r`).
    SeE2R,
    /// Three-body information: with a the direction entries s(r)·d/r of a
    /// normalized environment row, every pair of an atom's neighbor slots
    /// gives θ = a_j·a_k to the net of its pair of neighbor types, and D is
    /// the sum of θ·G(θ) over the pairs, the pairs of types t <= u each over
    /// sel[t]·sel[u] (DeePMD-kit's `se_e3`, also called `se_at` and
    /// `se_a_3be`).
    SeE3,
}

impl DescriptorKind {
    /// Every descriptor a native model can have.
    pub const ALL: [Self; 3] = [Self::SeE2A, Self::SeE2R, Self::SeE3];

    /// What `descriptor.kind` says for it.
    pub fn name(self) -> &'static str {
        match self {
            Self::SeE2A => "se_e2_a",
            Self::SeE2R => "se_e2_r",
            Self::SeE3 => "se_e3",
        }
    }

    /// The `format` of a native model with this descriptor.
    pub fn format(self) -> &'static str {
        match self {
            Self::SeE2A => "omnimd-native-se-e2-a",
            Self::SeE2R => "omnimd-native-se-e2-r",
            Self::SeE3 => "omnimd-native-se-e3",
        }
    }
}

impl Descriptor {
    /// The descriptor `kind` names, if the format has it. A hybrid is none
    /// of them: see [`Self::parts`].
    pub fn descriptor_kind(&self) -> Option<DescriptorKind> {
        DescriptorKind::ALL
            .into_iter()
            .find(|kind| kind.name() == self.kind)
    }

    /// Whether this is a hybrid: the descriptors of `list` together.
    pub fn is_hybrid(&self) -> bool {
        self.kind == HYBRID_KIND
    }

    /// The descriptors whose components make up this one, in their order:
    /// those of the list of a hybrid, or this descriptor alone. Of a
    /// validated model each has a [`DescriptorKind`].
    pub fn parts(&self) -> &[Descriptor] {
        match self.is_hybrid() {
            true => &self.list,
            false => std::slice::from_ref(self),
        }
    }

    /// Outputs of an embedding net (of a validated model: all alike). Not
    /// for a hybrid, whose descriptors each have their own.
    pub fn width(&self) -> usize {
        self.embedding[0][0].output_size()
    }

    /// Components of the descriptor, the input of a fitting net (of a
    /// validated model): those of all its descriptors for a hybrid.
    pub fn features(&self) -> usize {
        match self.descriptor_kind() {
            Some(DescriptorKind::SeE2A) => self.width() * self.axis_neuron,
            Some(DescriptorKind::SeE2R | DescriptorKind::SeE3) => self.width(),
            None => self.list.iter().map(Descriptor::features).sum(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Network {
    pub layers: Vec<Layer>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Linear,
    Tanh,
    Sigmoid,
    Softplus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
    /// The descriptor of a validated model that has one; none for a hybrid,
    /// which has several ([`Descriptor::parts`]).
    pub fn descriptor_kind(&self) -> Option<DescriptorKind> {
        self.descriptor.descriptor_kind()
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
        let formats = DescriptorKind::ALL.map(DescriptorKind::format);
        require(
            (formats.contains(&self.format.as_str()) || self.format == HYBRID_FORMAT)
                && self.version == 1,
            "unsupported native format/version (expected omnimd-native-se-e2-a, omnimd-native-se-e2-r, omnimd-native-se-e3 or omnimd-native-hybrid, v1)",
        )?;
        require(
            self.precision == "float64",
            "native v1 supports only float64 precision",
        )?;
        let d = &self.descriptor;
        // A format names its descriptor: the two must agree.
        let kind = d
            .descriptor_kind()
            .filter(|kind| kind.format() == self.format);
        if kind.is_none() && !(d.is_hybrid() && self.format == HYBRID_FORMAT) {
            return Err(Error(format!(
                "unsupported descriptor {:?} for native format {}: omnimd-native-se-e2-a has se_e2_a, omnimd-native-se-e2-r has se_e2_r, omnimd-native-se-e3 has se_e3, omnimd-native-hybrid has hybrid",
                d.kind, self.format
            )));
        }
        let metadata = ModelMetadata {
            type_map: self.type_map.clone(),
            cutoff: d.rcut,
            backend: match kind {
                Some(DescriptorKind::SeE2A) => "native-se-e2-a-v1".into(),
                Some(DescriptorKind::SeE2R) => "native-se-e2-r-v1".into(),
                Some(DescriptorKind::SeE3) => "native-se-e3-v1".into(),
                None => "native-hybrid-v1".into(),
            },
            precision: self.precision.clone(),
            units: self.units,
        };
        metadata.validate()?;
        let n = self.type_map.len();
        require(n <= 256, "native v1 supports at most 256 types")?;
        let input = match kind {
            Some(kind) => {
                require(
                    d.list.is_empty(),
                    "only a hybrid descriptor has a list of descriptors",
                )?;
                d.validate_part(kind, n, "")?
            }
            None => d.validate_hybrid(n)?,
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

impl Descriptor {
    /// Checks an se_e2_a, se_e2_r or se_e3 descriptor for `n` types and
    /// returns the number of its components. `context` goes before every
    /// message: nothing for the descriptor of a model, the place in the list
    /// for a descriptor of a hybrid.
    fn validate_part(&self, kind: DescriptorKind, n: usize, context: &str) -> Result<usize> {
        let d = self;
        require(
            d.rcut_smth.is_finite()
                && d.rcut_smth >= 0.0
                && d.rcut_smth < d.rcut
                && (d.rcut * d.rcut).is_finite(),
            format!("{context}invalid smoothing/cutoff radii"),
        )?;
        require(
            d.sel.len() == n,
            format!("{context}sel must have one entry per ordered type"),
        )?;
        let slots = d
            .sel
            .iter()
            .try_fold(0usize, |a, b| a.checked_add(*b))
            .ok_or_else(|| Error(format!("{context}sel sum overflow")))?;
        require(
            slots > 0 && slots <= 65536,
            format!("{context}sum(sel) must be in 1..=65536"),
        )?;
        require(
            d.normalization.len() == n,
            format!("{context}normalization must have one entry per central type"),
        )?;
        for (i, norm) in d.normalization.iter().enumerate() {
            require(
                finite(norm.mean.into_iter()) && norm.std.iter().all(|s| s.is_finite() && *s > 0.0),
                format!("{context}normalization[{i}] must be finite with positive std"),
            )?;
            require(
                norm.mean[1..] == [0.0; 3]
                    && norm.std[1] == norm.std[2]
                    && norm.std[2] == norm.std[3],
                format!("{context}normalization[{i}]: angular mean must be zero and angular std isotropic"),
            )?;
            require(
                kind != DescriptorKind::SeE2R || norm.std[1] == 1.0,
                format!("{context}normalization[{i}]: se_e2_r has a radial entry only; the other widths must be one"),
            )?;
            require(
                kind != DescriptorKind::SeE3 || (norm.mean[0] == 0.0 && norm.std[0] == 1.0),
                format!("{context}normalization[{i}]: se_e3 has direction entries only; the radial mean must be zero and the radial width one"),
            )?;
        }
        require(
            d.embedding.len() == n,
            format!("{context}embedding must have one row per type"),
        )?;
        let mut width = None;
        for (i, row) in d.embedding.iter().enumerate() {
            require(
                row.len() == n,
                format!("{context}embedding[{i}] must have one network per neighbor type"),
            )?;
            for (j, net) in row.iter().enumerate() {
                let out = net.validate(1, &format!("{context}embedding[{i}][{j}]"))?;
                if let Some(expected) = width {
                    require(
                        out == expected,
                        format!("{context}all embedding networks must have the same output width"),
                    )?;
                } else {
                    width = Some(out);
                }
            }
        }
        let width = width.ok_or_else(|| Error(format!("{context}no embedding networks")))?;
        require(
            !d.type_one_side || d.embedding.iter().all(|row| *row == d.embedding[0]),
            format!("{context}type_one_side: the embedding nets of a neighbor type must be the same for every central type"),
        )?;
        if kind == DescriptorKind::SeE3 {
            require(
                !d.type_one_side,
                format!("{context}se_e3 has one embedding net per pair of neighbor types: type_one_side does not apply"),
            )?;
            // The net of a pair of neighbor types, whichever comes first.
            let symmetric = (0..n).all(|i| (0..i).all(|j| d.embedding[i][j] == d.embedding[j][i]));
            require(
                symmetric,
                format!("{context}se_e3: the embedding net of a pair of neighbor types must be the same in either order"),
            )?;
        }
        match kind {
            DescriptorKind::SeE2A => {
                require(
                    d.axis_neuron > 0 && d.axis_neuron <= width,
                    format!("{context}axis_neuron must be in 1..=embedding width"),
                )?;
                width
                    .checked_mul(d.axis_neuron)
                    .ok_or_else(|| Error(format!("{context}descriptor width overflow")))
            }
            DescriptorKind::SeE2R | DescriptorKind::SeE3 => {
                require(
                    d.axis_neuron == 0,
                    format!(
                        "{context}{} has no axis neurons: axis_neuron must be zero or left out",
                        kind.name()
                    ),
                )?;
                Ok(width)
            }
        }
    }

    /// Checks a hybrid descriptor for `n` types and returns the number of
    /// its components, those of the descriptors in its list together.
    fn validate_hybrid(&self, n: usize) -> Result<usize> {
        require(
            !self.list.is_empty() && self.list.len() <= MAX_HYBRID_PARTS,
            format!("a hybrid descriptor has 1..={MAX_HYBRID_PARTS} descriptors in its list"),
        )?;
        require(
            self.axis_neuron == 0
                && !self.type_one_side
                && self.normalization.is_empty()
                && self.embedding.is_empty(),
            "a hybrid descriptor has no axis neurons, normalization or embedding nets of its own: those are in its list",
        )?;
        let mut features = 0usize;
        for (index, part) in self.list.iter().enumerate() {
            let context = format!("descriptor.list[{index}]: ");
            let kind = part.descriptor_kind().ok_or_else(|| {
                Error(format!(
                    "{context}unsupported descriptor {:?}: a hybrid is made of se_e2_a, se_e2_r and se_e3 descriptors",
                    part.kind
                ))
            })?;
            require(
                part.list.is_empty(),
                format!("{context}only a hybrid descriptor has a list of descriptors"),
            )?;
            features = features
                .checked_add(part.validate_part(kind, n, &context)?)
                .ok_or_else(|| Error("descriptor width overflow".into()))?;
        }
        // The neighbor list that all of them read.
        let rcut = self.list.iter().map(|part| part.rcut).fold(0.0, f64::max);
        let smooth = self
            .list
            .iter()
            .map(|part| part.rcut_smth)
            .fold(f64::INFINITY, f64::min);
        let sel: Vec<usize> = (0..n)
            .map(|t| self.list.iter().map(|part| part.sel[t]).max().unwrap_or(0))
            .collect();
        require(
            self.rcut == rcut && self.rcut_smth == smooth && self.sel == sel,
            format!(
                "a hybrid descriptor has the largest rcut ({rcut}), the smallest rcut_smth ({smooth}) and of every type the largest sel ({sel:?}) of the descriptors in its list"
            ),
        )?;
        let slots = sel
            .iter()
            .try_fold(0usize, |a, b| a.checked_add(*b))
            .ok_or_else(|| Error("sel sum overflow".into()))?;
        require(slots <= 65536, "sum(sel) must be in 1..=65536")?;
        Ok(features)
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
