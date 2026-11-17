//! A fresh se_e2_a or se_e2_r model, or a hybrid of such descriptors, from
//! DeePMD's `model` input section (plan W2): the network shapes and residual
//! rules of DeePMD-kit, its random initialization distributions, and
//! normalization and energy bias supplied by the caller (see
//! [`crate::stats`]).
//!
//! Initialization follows DeePMD-kit's PyTorch backend (`MLPLayer`): weights
//! are normal with standard deviation `1/sqrt(inputs + outputs)`, biases
//! standard normal, timesteps normal around 0.1 with standard deviation
//! 0.001. A hidden layer is residual when it keeps or doubles its width, and
//! has a timestep when it is residual and `resnet_dt` is set. The random
//! stream is omniMD's own, so a seed does not reproduce DeePMD's numbers.
//!
//! A hybrid (`{"type": "hybrid", "list": [...]}`) has the descriptors of its
//! list side by side: each with its own cutoffs, selection, nets, seed and
//! normalization, and fitting nets that take the components of all of them.
use crate::{
    Activation, Descriptor, DescriptorKind, Layer, Model, Network, Normalization, Residual,
    HYBRID_FORMAT, HYBRID_KIND, MAX_HYBRID_PARTS,
};
use omnimd_ml_contract::{Error, Result, Units};
use serde::Deserialize;

/// `sel` as in a DeePMD input: explicit counts, or `"auto"` / `"auto:1.1"`.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum SelSpec {
    Counts(Vec<usize>),
    Auto(String),
}

impl SelSpec {
    /// The factor of `"auto[:factor]"` (DeePMD's default is 1.1), or `None`
    /// for explicit counts.
    pub fn auto_factor(&self) -> Result<Option<f64>> {
        match self {
            Self::Counts(_) => Ok(None),
            Self::Auto(text) => {
                let factor = match text.strip_prefix("auto") {
                    Some("") => Some(1.1),
                    Some(rest) => rest.strip_prefix(':').and_then(|f| f.parse::<f64>().ok()),
                    None => None,
                };
                factor
                    .filter(|f| f.is_finite() && *f > 0.0)
                    .map(Some)
                    .ok_or_else(|| {
                        Error(format!(
                            "sel must be counts, \"auto\" or \"auto:FACTOR\", got {text:?}"
                        ))
                    })
            }
        }
    }
}

fn tanh_name() -> String {
    "tanh".into()
}

fn default_precision() -> String {
    "default".into()
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DescriptorSpec {
    #[serde(rename = "type")]
    pub kind: String,
    pub sel: SelSpec,
    pub rcut: f64,
    pub rcut_smth: f64,
    pub neuron: Vec<usize>,
    /// se_e2_a only; se_e2_r has no axis neurons.
    #[serde(default)]
    pub axis_neuron: usize,
    #[serde(default)]
    pub resnet_dt: bool,
    /// One embedding net per neighbor type instead of one per pair of
    /// types. DeePMD-kit's input default is false for both descriptors; its
    /// PyTorch backend (3.1.3) gives se_e2_r a net per neighbor type
    /// whatever this says.
    #[serde(default)]
    pub type_one_side: bool,
    #[serde(default = "tanh_name")]
    pub activation_function: String,
    #[serde(default = "default_precision")]
    pub precision: String,
    #[serde(default)]
    pub seed: u64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FittingSpec {
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    pub neuron: Vec<usize>,
    #[serde(default = "resnet_dt_default")]
    pub resnet_dt: bool,
    #[serde(default = "tanh_name")]
    pub activation_function: String,
    #[serde(default = "default_precision")]
    pub precision: String,
    #[serde(default)]
    pub seed: u64,
    /// The energy bias of every type, in the order of the type map, to use
    /// instead of the one fitted to the training energies (DeePMD's
    /// `atom_ener`, here for all types or none).
    #[serde(default)]
    pub atom_ener: Option<Vec<f64>>,
}

fn resnet_dt_default() -> bool {
    true
}

/// The `descriptor` section of a DeePMD input script: one descriptor, or a
/// hybrid, `{"type": "hybrid", "list": [...]}`, of several.
#[derive(Clone, Debug)]
pub enum DescriptorSection {
    One(DescriptorSpec),
    Hybrid(Vec<DescriptorSpec>),
}

impl DescriptorSection {
    /// The descriptors whose components the fitting nets take, in their
    /// order: those of the list of a hybrid, or the one descriptor.
    pub fn parts(&self) -> &[DescriptorSpec] {
        match self {
            Self::One(descriptor) => std::slice::from_ref(descriptor),
            Self::Hybrid(list) => list,
        }
    }

    /// [`Self::parts`], to change them.
    pub fn parts_mut(&mut self) -> &mut [DescriptorSpec] {
        match self {
            Self::One(descriptor) => std::slice::from_mut(descriptor),
            Self::Hybrid(list) => list,
        }
    }

    pub fn is_hybrid(&self) -> bool {
        matches!(self, Self::Hybrid(_))
    }

    /// What goes before the message about descriptor `number`: nothing for
    /// the descriptor of a model, its place in the list for one of a hybrid.
    pub fn context(&self, number: usize) -> String {
        match self {
            Self::One(_) => String::new(),
            Self::Hybrid(_) => format!("descriptor.list[{number}]: "),
        }
    }
}

/// The section is read by its `type`, so that a key a descriptor does not
/// have is named in the error, and for a hybrid with its place in the list.
impl<'de> Deserialize<'de> for DescriptorSection {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        use serde::de::Error as _;
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Hybrid {
            #[serde(rename = "type")]
            _kind: String,
            list: Vec<serde_json::Value>,
        }
        let is_hybrid = |value: &serde_json::Value| {
            value.get("type").and_then(serde_json::Value::as_str) == Some(HYBRID_KIND)
        };
        let value = serde_json::Value::deserialize(deserializer)?;
        if !is_hybrid(&value) {
            return serde_json::from_value(value)
                .map(Self::One)
                .map_err(|e| D::Error::custom(format!("descriptor: {e}")));
        }
        let hybrid: Hybrid = serde_json::from_value(value)
            .map_err(|e| D::Error::custom(format!("descriptor: {e}")))?;
        let mut list = Vec::with_capacity(hybrid.list.len());
        for (number, entry) in hybrid.list.into_iter().enumerate() {
            if is_hybrid(&entry) {
                return Err(D::Error::custom(format!(
                    "descriptor.list[{number}]: a hybrid cannot hold another hybrid"
                )));
            }
            list.push(
                serde_json::from_value(entry)
                    .map_err(|e| D::Error::custom(format!("descriptor.list[{number}]: {e}")))?,
            );
        }
        Ok(Self::Hybrid(list))
    }
}

/// The `model` section of a DeePMD input script: se_e2_a or se_e2_r (also
/// written `se_r`), or a hybrid of such descriptors, with an energy fitting
/// net.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    pub type_map: Vec<String>,
    pub descriptor: DescriptorSection,
    pub fitting_net: FittingSpec,
}

/// What the training data give one descriptor: its neighbor selection (the
/// counts of the script, or those of the neighbor statistics) and its
/// normalization, both with one entry per type.
#[derive(Clone, Debug)]
pub struct DescriptorInput {
    pub sel: Vec<usize>,
    pub normalization: Vec<Normalization>,
}

/// SplitMix64 with Box–Muller normals.
struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        // In (0, 1): never zero, so the logarithm below is finite.
        (((z ^ (z >> 31)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self, mean: f64, std: f64) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        mean + std * (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

fn layer(rng: &mut Rng, input: usize, output: usize, hidden: bool, resnet_dt: bool) -> Layer {
    let residual = match (hidden, output == input, output == 2 * input) {
        (true, true, _) => Residual::Identity,
        (true, _, true) => Residual::Concat2,
        _ => Residual::None,
    };
    let std = 1.0 / ((input + output) as f64).sqrt();
    let weights = (0..input)
        .map(|_| (0..output).map(|_| rng.normal(0.0, std)).collect())
        .collect();
    let bias = (0..output).map(|_| rng.normal(0.0, 1.0)).collect();
    let timestep = (resnet_dt && !matches!(residual, Residual::None))
        .then(|| (0..output).map(|_| rng.normal(0.1, 0.001)).collect());
    Layer {
        weights,
        bias,
        activation: if hidden {
            Activation::Tanh
        } else {
            Activation::Linear
        },
        residual,
        timestep,
    }
}

/// The descriptor a spec names: `se_e2_a`, or `se_e2_r` under either of
/// DeePMD-kit's names for it.
fn spec_kind(descriptor: &DescriptorSpec) -> Option<DescriptorKind> {
    match descriptor.kind.as_str() {
        "se_e2_a" => Some(DescriptorKind::SeE2A),
        "se_e2_r" | "se_r" => Some(DescriptorKind::SeE2R),
        _ => None,
    }
}

const UNSUPPORTED: &str = "only se_e2_a and se_e2_r descriptors, alone or in a hybrid, with tanh activations, an ener fitting net and float64 parameters can be initialized";
const SHAPES: &str = "model needs a type map, one sel per type and nonempty positive neuron lists";

/// Whether a `precision` key asks for float64 parameters, the only ones a
/// native model has.
fn is_float64(precision: &str) -> bool {
    precision == "default" || precision == "float64"
}

/// One descriptor of the script with the selection it gets, for `ntypes`
/// types. `context` goes before every message.
fn check_descriptor(
    d: &DescriptorSpec,
    ntypes: usize,
    sel: &[usize],
    context: &str,
) -> Result<DescriptorKind> {
    let supported = d.activation_function == "tanh" && is_float64(&d.precision);
    let Some(kind) = spec_kind(d).filter(|_| supported) else {
        return Err(Error(format!("{context}{UNSUPPORTED}")));
    };
    if ntypes == 0 || sel.len() != ntypes || d.neuron.is_empty() || d.neuron.contains(&0) {
        return Err(Error(format!("{context}{SHAPES}")));
    }
    let axis_fits = match kind {
        DescriptorKind::SeE2A => d.axis_neuron > 0 && d.axis_neuron <= *d.neuron.last().unwrap(),
        // (A script cannot name se_e3 here: see `spec_kind`.)
        DescriptorKind::SeE2R | DescriptorKind::SeE3 => d.axis_neuron == 0,
    };
    if !axis_fits {
        return Err(Error(format!(
            "{context}se_e2_a needs 0 < axis_neuron <= the embedding width; se_e2_r has no axis_neuron"
        )));
    }
    Ok(kind)
}

/// Builds the model with freshly drawn parameters. `inputs` has, for every
/// descriptor of the script (one, or those of the list of a hybrid, in its
/// order), the resolved neighbor selection and the normalization;
/// `atom_energy_bias` has one entry per type. Of the normalization an
/// se_e2_r descriptor keeps the radial entry, which is all it reads.
pub fn initialize(
    spec: &ModelSpec,
    inputs: Vec<DescriptorInput>,
    atom_energy_bias: Vec<f64>,
) -> Result<Model> {
    let section = &spec.descriptor;
    let parts = section.parts();
    let ntypes = spec.type_map.len();
    let f = &spec.fitting_net;
    let fitting_supported = f.activation_function == "tanh"
        && f.kind.as_deref().is_none_or(|k| k == "ener")
        && is_float64(&f.precision);
    if !fitting_supported {
        return Err(Error(UNSUPPORTED.into()));
    }
    if section.is_hybrid() && !(1..=MAX_HYBRID_PARTS).contains(&parts.len()) {
        return Err(Error(format!(
            "a hybrid descriptor has 1..={MAX_HYBRID_PARTS} descriptors in its list, got {}",
            parts.len()
        )));
    }
    if inputs.len() != parts.len() {
        return Err(Error(format!(
            "expected the selection and normalization of {} descriptors, got {}",
            parts.len(),
            inputs.len()
        )));
    }
    let mut kinds = Vec::with_capacity(parts.len());
    for (number, (d, input)) in parts.iter().zip(&inputs).enumerate() {
        let context = section.context(number);
        kinds.push(check_descriptor(d, ntypes, &input.sel, &context)?);
        if input.normalization.len() != ntypes || atom_energy_bias.len() != ntypes {
            return Err(Error(format!(
                "{context}expected {ntypes} normalizations and energy biases, got {} and {}",
                input.normalization.len(),
                atom_energy_bias.len()
            )));
        }
    }
    if f.neuron.is_empty() || f.neuron.contains(&0) {
        return Err(Error(SHAPES.into()));
    }
    let network = |rng: &mut Rng,
                   mut input: usize,
                   neuron: &[usize],
                   resnet_dt: bool,
                   output: Option<usize>| {
        let mut layers = Vec::with_capacity(neuron.len() + 1);
        for &width in neuron {
            layers.push(layer(rng, input, width, true, resnet_dt));
            input = width;
        }
        if let Some(output) = output {
            layers.push(layer(rng, input, output, false, false));
        }
        Network { layers }
    };
    // One independent stream per network, derived from the section's seed.
    // The descriptors of a hybrid may carry the same seed: the place of a
    // descriptor in the list enters, so that their nets still differ. The
    // first one draws what the same descriptor draws alone.
    let stream = |seed: u64, part: usize, index: usize| {
        Rng(
            (seed.wrapping_mul(0x2545_f491_4f6c_dd1d) ^ ((index as u64 + 1) << 32))
                .wrapping_add((part as u64).wrapping_mul(0xd6e8_feb8_6659_fd93)),
        )
    };
    let mut features = 0;
    let mut descriptors = Vec::with_capacity(parts.len());
    for (number, ((d, input), kind)) in parts
        .iter()
        .zip(inputs)
        .zip(kinds.iter().copied())
        .enumerate()
    {
        let mut normalization = input.normalization;
        if kind == DescriptorKind::SeE2R {
            for entry in &mut normalization {
                entry.mean = [entry.mean[0], 0.0, 0.0, 0.0];
                entry.std = [entry.std[0], 1.0, 1.0, 1.0];
            }
        }
        // A net for every pair of types, or with `type_one_side` one for
        // every neighbor type, which all central types then have.
        let embedding = (0..ntypes)
            .map(|i| {
                (0..ntypes)
                    .map(|j| {
                        let index = match d.type_one_side {
                            true => j,
                            false => i * ntypes + j,
                        };
                        let mut rng = stream(d.seed, number, index);
                        network(&mut rng, 1, &d.neuron, d.resnet_dt, None)
                    })
                    .collect()
            })
            .collect();
        features += match kind {
            DescriptorKind::SeE2A => d.neuron.last().unwrap() * d.axis_neuron,
            DescriptorKind::SeE2R | DescriptorKind::SeE3 => *d.neuron.last().unwrap(),
        };
        descriptors.push(Descriptor {
            kind: kind.name().into(),
            rcut: d.rcut,
            rcut_smth: d.rcut_smth,
            sel: input.sel,
            axis_neuron: d.axis_neuron,
            normalization,
            type_one_side: d.type_one_side,
            embedding,
            list: Vec::new(),
        });
    }
    let fitting = (0..ntypes)
        .map(|i| {
            network(
                &mut stream(f.seed ^ 0x5eed, 0, i),
                features,
                &f.neuron,
                f.resnet_dt,
                Some(1),
            )
        })
        .collect();
    let (format, descriptor) = match section {
        DescriptorSection::One(_) => (kinds[0].format(), descriptors.remove(0)),
        // The neighbor list of a hybrid: the largest cutoff, the smallest
        // smoothing radius and of every type the most slots.
        DescriptorSection::Hybrid(_) => {
            let sel = (0..ntypes)
                .map(|t| descriptors.iter().map(|d| d.sel[t]).max().unwrap_or(0))
                .collect();
            let hybrid = Descriptor {
                kind: HYBRID_KIND.into(),
                rcut: descriptors.iter().map(|d| d.rcut).fold(0.0, f64::max),
                rcut_smth: descriptors
                    .iter()
                    .map(|d| d.rcut_smth)
                    .fold(f64::INFINITY, f64::min),
                sel,
                axis_neuron: 0,
                normalization: Vec::new(),
                type_one_side: false,
                embedding: Vec::new(),
                list: descriptors,
            };
            (HYBRID_FORMAT, hybrid)
        }
    };
    let model = Model {
        format: format.into(),
        version: 1,
        precision: "float64".into(),
        units: Units::EvAngstrom,
        type_map: spec.type_map.clone(),
        descriptor,
        fitting,
        atom_energy_bias,
    };
    model.validate()?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> ModelSpec {
        serde_json::from_str(
            r#"{"type_map": ["Ca", "P", "O"],
                "descriptor": {"type": "se_e2_a", "sel": "auto", "rcut_smth": 0.5, "rcut": 6.0,
                               "neuron": [25, 50, 100], "axis_neuron": 16, "resnet_dt": false,
                               "type_one_side": false, "seed": 11},
                "fitting_net": {"neuron": [240, 240, 240], "resnet_dt": true, "seed": 11}}"#,
        )
        .unwrap()
    }

    fn unit_normalization(n: usize) -> Vec<Normalization> {
        vec![
            Normalization {
                mean: [0.1, 0.0, 0.0, 0.0],
                std: [0.2, 0.1, 0.1, 0.1],
            };
            n
        ]
    }

    /// The selection and the normalization of a model's one descriptor.
    fn one(sel: Vec<usize>, normalization: Vec<Normalization>) -> Vec<DescriptorInput> {
        vec![DescriptorInput { sel, normalization }]
    }

    #[test]
    fn shapes_residuals_and_distributions_follow_deepmd() {
        let spec = spec();
        assert_eq!(
            spec.descriptor.parts()[0].sel.auto_factor().unwrap(),
            Some(1.1)
        );
        let model = initialize(
            &spec,
            one(vec![44, 16, 60], unit_normalization(3)),
            vec![-1.0, -2.0, -3.0],
        )
        .unwrap();
        assert_eq!(model.descriptor.embedding.len(), 3);
        let net = &model.descriptor.embedding[2][1];
        let shapes: Vec<_> = net
            .layers
            .iter()
            .map(|l| (l.weights.len(), l.bias.len()))
            .collect();
        assert_eq!(shapes, [(1, 25), (25, 50), (50, 100)]);
        // 1 -> 25 cannot be residual; the doubling layers concatenate; no
        // timestep without resnet_dt.
        assert!(matches!(net.layers[0].residual, Residual::None));
        assert!(net.layers[1..]
            .iter()
            .all(|l| matches!(l.residual, Residual::Concat2)));
        assert!(net.layers.iter().all(|l| l.timestep.is_none()));

        let fit = &model.fitting[0];
        let shapes: Vec<_> = fit
            .layers
            .iter()
            .map(|l| (l.weights.len(), l.bias.len()))
            .collect();
        assert_eq!(shapes, [(1600, 240), (240, 240), (240, 240), (240, 1)]);
        assert!(
            matches!(fit.layers[0].residual, Residual::None) && fit.layers[0].timestep.is_none()
        );
        for hidden in &fit.layers[1..3] {
            assert!(matches!(hidden.residual, Residual::Identity));
            let timestep = hidden.timestep.as_ref().unwrap();
            assert!(timestep.iter().all(|t| (t - 0.1).abs() < 0.01));
        }
        let last = &fit.layers[3];
        assert!(matches!(last.activation, Activation::Linear) && last.timestep.is_none());

        // 1600 x 240 weights: mean 0, standard deviation 1/sqrt(1840).
        let weights: Vec<f64> = fit.layers[0].weights.iter().flatten().copied().collect();
        let mean = weights.iter().sum::<f64>() / weights.len() as f64;
        let std =
            (weights.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / weights.len() as f64).sqrt();
        let expected = 1.0 / 1840f64.sqrt();
        assert!(
            mean.abs() < 1e-4 && (std / expected - 1.0).abs() < 0.01,
            "{mean} {std}"
        );
        let bias = &fit.layers[1].bias;
        let bias_std = (bias.iter().map(|b| b * b).sum::<f64>() / bias.len() as f64).sqrt();
        assert!((bias_std - 1.0).abs() < 0.2);

        // Networks differ from each other, and a seed reproduces the model.
        assert_ne!(
            model.fitting[0].layers[0].bias,
            model.fitting[1].layers[0].bias
        );
        assert_ne!(
            model.descriptor.embedding[0][1].layers[0].bias,
            model.descriptor.embedding[1][0].layers[0].bias
        );
        let again = initialize(
            &spec,
            one(vec![44, 16, 60], unit_normalization(3)),
            vec![-1.0, -2.0, -3.0],
        )
        .unwrap();
        assert_eq!(
            again.fitting[2].layers[2].bias,
            model.fitting[2].layers[2].bias
        );
    }

    #[test]
    fn one_net_per_neighbor_type_and_the_radial_descriptor() {
        // se_e2_a with one embedding net per neighbor type: every central
        // type has the same row, and the model says so.
        let mut one_side = spec();
        one_side.descriptor.parts_mut()[0].type_one_side = true;
        let model = initialize(
            &one_side,
            one(vec![44, 16, 60], unit_normalization(3)),
            vec![0.0; 3],
        )
        .unwrap();
        assert!(model.descriptor.type_one_side);
        let rows = &model.descriptor.embedding;
        assert!(rows[1] == rows[0] && rows[2] == rows[0]);
        assert_ne!(rows[0][0], rows[0][1]);
        // The nets of the pairs differ, and that model does not carry the mark.
        let pairs = initialize(
            &spec(),
            one(vec![44, 16, 60], unit_normalization(3)),
            vec![0.0; 3],
        )
        .unwrap();
        assert!(!pairs.descriptor.type_one_side);
        assert_ne!(pairs.descriptor.embedding[1], pairs.descriptor.embedding[0]);
        assert!(!serde_json::to_string(&pairs)
            .unwrap()
            .contains("type_one_side"));
        assert!(serde_json::to_string(&model)
            .unwrap()
            .contains("\"type_one_side\":true"));

        // se_e2_r under both of its names: no axis neurons, fitting nets on
        // the embedding width, the radial normalization only.
        for (name, shared) in [("se_e2_r", false), ("se_r", true)] {
            let mut radial = spec();
            let descriptor = &mut radial.descriptor.parts_mut()[0];
            descriptor.kind = name.into();
            descriptor.axis_neuron = 0;
            descriptor.type_one_side = shared;
            let model = initialize(
                &radial,
                one(vec![44, 16, 60], unit_normalization(3)),
                vec![-1.0, -2.0, -3.0],
            )
            .unwrap();
            assert_eq!(model.format, "omnimd-native-se-e2-r");
            assert_eq!(model.descriptor.kind, "se_e2_r");
            assert_eq!(model.descriptor_kind(), Some(DescriptorKind::SeE2R));
            assert_eq!(model.descriptor.axis_neuron, 0);
            assert_eq!(model.descriptor.type_one_side, shared);
            assert_eq!(
                model.descriptor.embedding[1] == model.descriptor.embedding[0],
                shared
            );
            for entry in &model.descriptor.normalization {
                assert_eq!(entry.mean, [0.1, 0.0, 0.0, 0.0]);
                assert_eq!(entry.std, [0.2, 1.0, 1.0, 1.0]);
            }
            let shapes: Vec<_> = model.fitting[1]
                .layers
                .iter()
                .map(|l| (l.weights.len(), l.bias.len()))
                .collect();
            assert_eq!(shapes, [(100, 240), (240, 240), (240, 240), (240, 1)]);
            assert_eq!(model.atom_energy_bias, [-1.0, -2.0, -3.0]);
            // An axis neuron has no meaning for it.
            radial.descriptor.parts_mut()[0].axis_neuron = 16;
            assert!(initialize(
                &radial,
                one(vec![44, 16, 60], unit_normalization(3)),
                vec![0.0; 3]
            )
            .is_err());
        }
        // A script of se_e2_r may leave axis_neuron out; one of se_e2_a not.
        let text = r#"{"type_map": ["O", "H"],
            "descriptor": {"type": "se_e2_r", "sel": [4, 8], "rcut_smth": 0.5, "rcut": 4.0,
                           "neuron": [4, 8]},
            "fitting_net": {"neuron": [6, 6]}}"#;
        let radial: ModelSpec = serde_json::from_str(text).unwrap();
        assert!(initialize(
            &radial,
            one(vec![4, 8], unit_normalization(2)),
            vec![0.0; 2]
        )
        .is_ok());
        let angular: ModelSpec = serde_json::from_str(&text.replace("se_e2_r", "se_e2_a")).unwrap();
        assert!(initialize(
            &angular,
            one(vec![4, 8], unit_normalization(2)),
            vec![0.0; 2]
        )
        .is_err());
    }

    #[test]
    fn unsupported_models_are_rejected() {
        let mut bad = spec();
        // (A hybrid is not a descriptor of this kind: it is a section of its
        // own, read by its type.)
        for unsupported in ["se_e3", "se_atten", "hybrid"] {
            bad.descriptor.parts_mut()[0].kind = unsupported.into();
            let error = initialize(
                &bad,
                one(vec![1, 1, 1], unit_normalization(3)),
                vec![0.0; 3],
            )
            .unwrap_err()
            .to_string();
            assert!(
                error.contains("only se_e2_a and se_e2_r"),
                "{unsupported}: {error}"
            );
        }
        assert!(initialize(
            &spec(),
            one(vec![1, 1], unit_normalization(3)),
            vec![0.0; 3]
        )
        .is_err());
        assert!(initialize(
            &spec(),
            one(vec![1, 1, 1], unit_normalization(2)),
            vec![0.0; 3]
        )
        .is_err());
        for text in ["auto:", "auto:x", "automatic", "auto:-1"] {
            assert!(SelSpec::Auto(text.into()).auto_factor().is_err(), "{text}");
        }
        assert_eq!(
            SelSpec::Auto("auto:1.5".into()).auto_factor().unwrap(),
            Some(1.5)
        );
        assert_eq!(SelSpec::Counts(vec![1]).auto_factor().unwrap(), None);
    }

    const HYBRID: &str = r#"{"type_map": ["Ca", "P", "O"],
        "descriptor": {"type": "hybrid", "list": [
            {"type": "se_e2_a", "sel": [20, 8, 28], "rcut_smth": 0.5, "rcut": 4.0,
             "neuron": [25, 50, 100], "axis_neuron": 16, "seed": 11},
            {"type": "se_e2_r", "sel": "auto", "rcut_smth": 1.0, "rcut": 6.0,
             "neuron": [25, 50, 100], "type_one_side": true, "seed": 11}]},
        "fitting_net": {"neuron": [240, 240, 240], "resnet_dt": true, "seed": 11}}"#;

    /// What the data would give the two descriptors of [`HYBRID`].
    fn hybrid_inputs() -> Vec<DescriptorInput> {
        let far = Normalization {
            mean: [0.3, 0.0, 0.0, 0.0],
            std: [0.4, 0.2, 0.2, 0.2],
        };
        vec![
            DescriptorInput {
                sel: vec![20, 8, 28],
                normalization: unit_normalization(3),
            },
            DescriptorInput {
                sel: vec![44, 16, 24],
                normalization: vec![far; 3],
            },
        ]
    }

    #[test]
    fn a_hybrid_has_the_descriptors_of_its_list_side_by_side() {
        let spec: ModelSpec = serde_json::from_str(HYBRID).unwrap();
        assert!(spec.descriptor.is_hybrid());
        let parts = spec.descriptor.parts();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].sel.auto_factor().unwrap(), None);
        assert_eq!(parts[1].sel.auto_factor().unwrap(), Some(1.1));
        assert_eq!(spec.descriptor.context(1), "descriptor.list[1]: ");

        let model = initialize(&spec, hybrid_inputs(), vec![-1.0, -2.0, -3.0]).unwrap();
        assert_eq!(model.format, "omnimd-native-hybrid");
        assert_eq!(model.descriptor_kind(), None);
        let d = &model.descriptor;
        assert_eq!(d.kind, "hybrid");
        // The neighbor list of both: the largest cutoff, the smallest
        // smoothing radius and of every type the most slots.
        assert_eq!((d.rcut, d.rcut_smth), (6.0, 0.5));
        assert_eq!(d.sel, [44, 16, 28]);
        assert!(d.embedding.is_empty() && d.normalization.is_empty());
        assert!(d.axis_neuron == 0 && !d.type_one_side);
        let (angular, radial) = (&d.list[0], &d.list[1]);
        assert_eq!(
            (angular.kind.as_str(), radial.kind.as_str()),
            ("se_e2_a", "se_e2_r")
        );
        assert_eq!(
            (angular.rcut, angular.rcut_smth, angular.axis_neuron),
            (4.0, 0.5, 16)
        );
        assert_eq!(
            (radial.rcut, radial.rcut_smth, radial.axis_neuron),
            (6.0, 1.0, 0)
        );
        assert_eq!(angular.sel, [20, 8, 28]);
        assert_eq!(radial.sel, [44, 16, 24]);
        assert!(!angular.type_one_side && radial.type_one_side);
        // Each has the normalization it was given; se_e2_r keeps the
        // radial entry of it.
        assert_eq!(angular.normalization[1].mean, [0.1, 0.0, 0.0, 0.0]);
        assert_eq!(angular.normalization[1].std, [0.2, 0.1, 0.1, 0.1]);
        assert_eq!(radial.normalization[1].mean, [0.3, 0.0, 0.0, 0.0]);
        assert_eq!(radial.normalization[1].std, [0.4, 1.0, 1.0, 1.0]);
        // The fitting nets take the components of both: 100 · 16 + 100.
        assert_eq!(d.features(), 1700);
        assert_eq!(model.fitting[2].layers[0].weights.len(), 1700);
        assert_eq!(model.atom_energy_bias, [-1.0, -2.0, -3.0]);

        // The first descriptor draws what it draws as the descriptor of a
        // model. The second has the seed and the shape of the first, and
        // the same place among the nets for a neighbor of type 1 around a
        // center of type 0: its place in the list makes the nets differ.
        let alone = ModelSpec {
            descriptor: DescriptorSection::One(parts[0].clone()),
            ..spec.clone()
        };
        let inputs = hybrid_inputs();
        let alone = initialize(&alone, vec![inputs[0].clone()], vec![0.0; 3]).unwrap();
        assert_eq!(alone.format, "omnimd-native-se-e2-a");
        assert_eq!(alone.descriptor.embedding, angular.embedding);
        assert_ne!(radial.embedding[0][1], angular.embedding[0][1]);
        assert_ne!(
            radial.embedding[0][1].layers[0].bias,
            angular.embedding[0][1].layers[0].bias
        );
        // The same request gives the same model.
        let again = initialize(&spec, hybrid_inputs(), vec![-1.0, -2.0, -3.0]).unwrap();
        assert_eq!(again.descriptor.list[1].embedding, radial.embedding);
        assert_eq!(again.fitting, model.fitting);

        // A list of one descriptor is still a hybrid, with the nets of that
        // descriptor and fitting nets of its width.
        let single = ModelSpec {
            descriptor: DescriptorSection::Hybrid(vec![parts[1].clone()]),
            ..spec.clone()
        };
        let single = initialize(&single, vec![inputs[1].clone()], vec![0.0; 3]).unwrap();
        assert_eq!(single.format, "omnimd-native-hybrid");
        assert_eq!(single.descriptor.list.len(), 1);
        assert_eq!(
            (single.descriptor.rcut, single.descriptor.rcut_smth),
            (6.0, 1.0)
        );
        assert_eq!(single.descriptor.sel, [44, 16, 24]);
        assert_eq!(single.fitting[0].layers[0].weights.len(), 100);
    }

    #[test]
    fn a_hybrid_that_cannot_be_built_names_the_descriptor() {
        let spec: ModelSpec = serde_json::from_str(HYBRID).unwrap();
        let build = |spec: &ModelSpec, inputs: Vec<DescriptorInput>| {
            initialize(spec, inputs, vec![0.0; 3])
                .unwrap_err()
                .to_string()
        };
        // A descriptor that cannot be trained, by its place in the list.
        let mut three_body = spec.clone();
        three_body.descriptor.parts_mut()[1].kind = "se_e3".into();
        let error = build(&three_body, hybrid_inputs());
        assert!(
            error.starts_with("descriptor.list[1]: only se_e2_a and se_e2_r descriptors"),
            "{error}"
        );
        let mut axis = spec.clone();
        axis.descriptor.parts_mut()[1].axis_neuron = 4;
        let error = build(&axis, hybrid_inputs());
        assert!(
            error.starts_with("descriptor.list[1]: se_e2_a needs 0 < axis_neuron"),
            "{error}"
        );
        // The selection of one descriptor for another number of types.
        let mut short = hybrid_inputs();
        short[1].sel.pop();
        let error = build(&spec, short);
        assert!(
            error.starts_with("descriptor.list[1]: model needs a type map, one sel per type"),
            "{error}"
        );
        let mut fewer = hybrid_inputs();
        fewer[0].normalization.pop();
        let error = build(&spec, fewer);
        assert!(
            error.starts_with("descriptor.list[0]: expected 3 normalizations"),
            "{error}"
        );
        // Statistics for one descriptor only, and a list without or with too
        // many descriptors.
        let mut one_input = hybrid_inputs();
        one_input.pop();
        let error = build(&spec, one_input);
        assert_eq!(
            error,
            "expected the selection and normalization of 2 descriptors, got 1"
        );
        let empty = ModelSpec {
            descriptor: DescriptorSection::Hybrid(Vec::new()),
            ..spec.clone()
        };
        let error = build(&empty, Vec::new());
        assert_eq!(
            error,
            "a hybrid descriptor has 1..=16 descriptors in its list, got 0"
        );
        let part = spec.descriptor.parts()[1].clone();
        let many = ModelSpec {
            descriptor: DescriptorSection::Hybrid(vec![part; 17]),
            ..spec.clone()
        };
        let error = build(&many, vec![hybrid_inputs().remove(1); 17]);
        assert!(error.contains("got 17"), "{error}");
        let sixteen = ModelSpec {
            descriptor: DescriptorSection::Hybrid(vec![spec.descriptor.parts()[1].clone(); 16]),
            ..spec.clone()
        };
        let sixteen =
            initialize(&sixteen, vec![hybrid_inputs().remove(1); 16], vec![0.0; 3]).unwrap();
        assert_eq!(sixteen.descriptor.features(), 1600);
        // A cutoff no larger than its smoothing radius is the model's own
        // check, with the place of the descriptor.
        let mut radii = spec.clone();
        radii.descriptor.parts_mut()[0].rcut_smth = 4.0;
        let error = build(&radii, hybrid_inputs());
        assert!(
            error.contains("descriptor.list[0]: invalid smoothing/cutoff radii"),
            "{error}"
        );
    }

    #[test]
    fn a_descriptor_section_is_read_by_its_type() {
        let read = |descriptor: &str| {
            let text = format!(
                r#"{{"type_map": ["O", "H"], "descriptor": {descriptor},
                    "fitting_net": {{"neuron": [6, 6]}}}}"#
            );
            serde_json::from_str::<ModelSpec>(&text).map_err(|e| e.to_string())
        };
        let part = r#"{"type": "se_e2_r", "sel": [4, 8], "rcut_smth": 0.5, "rcut": 4.0,
                       "neuron": [4, 8]}"#;
        assert!(!read(part).unwrap().descriptor.is_hybrid());
        let hybrid = read(&format!(
            r#"{{"type": "hybrid", "list": [{part}, {part}]}}"#
        ))
        .unwrap();
        assert!(hybrid.descriptor.is_hybrid());
        assert_eq!(hybrid.descriptor.parts().len(), 2);
        assert_eq!(hybrid.descriptor.parts()[1].kind, "se_e2_r");
        // A hybrid has a list and nothing else; its descriptors have the
        // keys they have alone; a hybrid is not one of them.
        let error = read(&format!(
            r#"{{"type": "hybrid", "list": [{part}], "rcut": 6.0}}"#
        ))
        .unwrap_err();
        assert!(
            error.contains("descriptor: unknown field `rcut`"),
            "{error}"
        );
        let error = read(r#"{"type": "hybrid"}"#).unwrap_err();
        assert!(
            error.contains("descriptor: missing field `list`"),
            "{error}"
        );
        let typo = part.replace("rcut_smth", "rcut_smooth");
        let error = read(&format!(
            r#"{{"type": "hybrid", "list": [{part}, {typo}]}}"#
        ))
        .unwrap_err();
        assert!(
            error.contains("descriptor.list[1]: unknown field `rcut_smooth`"),
            "{error}"
        );
        let error = read(&typo).unwrap_err();
        assert!(
            error.contains("descriptor: unknown field `rcut_smooth`"),
            "{error}"
        );
        let nested = format!(
            r#"{{"type": "hybrid", "list": [{part}, {{"type": "hybrid", "list": [{part}]}}]}}"#
        );
        let error = read(&nested).unwrap_err();
        assert!(
            error.contains("descriptor.list[1]: a hybrid cannot hold another hybrid"),
            "{error}"
        );
    }
}
