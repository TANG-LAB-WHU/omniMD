//! A fresh se_e2_a model from DeePMD's `model` input section (plan W2): the
//! network shapes and residual rules of DeePMD-kit, its random initialization
//! distributions, and normalization and energy bias supplied by the caller
//! (see [`crate::stats`]).
//!
//! Initialization follows DeePMD-kit's PyTorch backend (`MLPLayer`): weights
//! are normal with standard deviation `1/sqrt(inputs + outputs)`, biases
//! standard normal, timesteps normal around 0.1 with standard deviation
//! 0.001. A hidden layer is residual when it keeps or doubles its width, and
//! has a timestep when it is residual and `resnet_dt` is set. The random
//! stream is omniMD's own, so a seed does not reproduce DeePMD's numbers.
use crate::{Activation, Descriptor, Layer, Model, Network, Normalization, Residual};
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
    pub axis_neuron: usize,
    #[serde(default)]
    pub resnet_dt: bool,
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
}

fn resnet_dt_default() -> bool {
    true
}

/// The `model` section of a DeePMD input script, se_e2_a with an energy
/// fitting net.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    pub type_map: Vec<String>,
    pub descriptor: DescriptorSpec,
    pub fitting_net: FittingSpec,
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

fn check_spec(spec: &ModelSpec, sel: &[usize]) -> Result<()> {
    let d = &spec.descriptor;
    let f = &spec.fitting_net;
    let supported = d.kind == "se_e2_a"
        && !d.type_one_side
        && d.activation_function == "tanh"
        && f.activation_function == "tanh"
        && f.kind.as_deref().is_none_or(|k| k == "ener")
        && [&d.precision, &f.precision]
            .iter()
            .all(|p| *p == "default" || *p == "float64");
    if !supported {
        return Err(Error(
            "only se_e2_a with type_one_side=false, tanh activations, an ener fitting net and float64 parameters can be initialized".into(),
        ));
    }
    if spec.type_map.is_empty()
        || sel.len() != spec.type_map.len()
        || d.neuron.is_empty()
        || f.neuron.is_empty()
        || d.neuron.iter().chain(&f.neuron).any(|&n| n == 0)
        || d.axis_neuron == 0
        || d.axis_neuron > *d.neuron.last().unwrap()
    {
        return Err(Error(
            "model needs a type map, one sel per type, nonempty positive neuron lists and 0 < axis_neuron <= the embedding width".into(),
        ));
    }
    Ok(())
}

/// Builds the model with freshly drawn parameters. `sel` is the resolved
/// neighbor selection (from the spec or from neighbor statistics);
/// `normalization` and `atom_energy_bias` have one entry per type.
pub fn initialize(
    spec: &ModelSpec,
    sel: Vec<usize>,
    normalization: Vec<Normalization>,
    atom_energy_bias: Vec<f64>,
) -> Result<Model> {
    check_spec(spec, &sel)?;
    let ntypes = spec.type_map.len();
    if normalization.len() != ntypes || atom_energy_bias.len() != ntypes {
        return Err(Error(format!(
            "expected {ntypes} normalizations and energy biases, got {} and {}",
            normalization.len(),
            atom_energy_bias.len()
        )));
    }
    let d = &spec.descriptor;
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
    let stream = |seed: u64, index: usize| {
        Rng(seed.wrapping_mul(0x2545_f491_4f6c_dd1d) ^ ((index as u64 + 1) << 32))
    };
    let embedding = (0..ntypes)
        .map(|i| {
            (0..ntypes)
                .map(|j| {
                    network(
                        &mut stream(d.seed, i * ntypes + j),
                        1,
                        &d.neuron,
                        d.resnet_dt,
                        None,
                    )
                })
                .collect()
        })
        .collect();
    let features = d.neuron.last().unwrap() * d.axis_neuron;
    let f = &spec.fitting_net;
    let fitting = (0..ntypes)
        .map(|i| {
            network(
                &mut stream(f.seed ^ 0x5eed, i),
                features,
                &f.neuron,
                f.resnet_dt,
                Some(1),
            )
        })
        .collect();
    let model = Model {
        format: "omnimd-native-se-e2-a".into(),
        version: 1,
        precision: "float64".into(),
        units: Units::EvAngstrom,
        type_map: spec.type_map.clone(),
        descriptor: Descriptor {
            kind: "se_e2_a".into(),
            rcut: d.rcut,
            rcut_smth: d.rcut_smth,
            sel,
            axis_neuron: d.axis_neuron,
            normalization,
            embedding,
        },
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

    #[test]
    fn shapes_residuals_and_distributions_follow_deepmd() {
        let spec = spec();
        assert_eq!(spec.descriptor.sel.auto_factor().unwrap(), Some(1.1));
        let model = initialize(
            &spec,
            vec![44, 16, 60],
            unit_normalization(3),
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
            vec![44, 16, 60],
            unit_normalization(3),
            vec![-1.0, -2.0, -3.0],
        )
        .unwrap();
        assert_eq!(
            again.fitting[2].layers[2].bias,
            model.fitting[2].layers[2].bias
        );
    }

    #[test]
    fn unsupported_models_are_rejected() {
        let mut bad = spec();
        bad.descriptor.type_one_side = true;
        assert!(initialize(&bad, vec![1, 1, 1], unit_normalization(3), vec![0.0; 3]).is_err());
        let mut bad = spec();
        bad.descriptor.kind = "se_e3".into();
        assert!(initialize(&bad, vec![1, 1, 1], unit_normalization(3), vec![0.0; 3]).is_err());
        assert!(initialize(&spec(), vec![1, 1], unit_normalization(3), vec![0.0; 3]).is_err());
        assert!(initialize(&spec(), vec![1, 1, 1], unit_normalization(2), vec![0.0; 3]).is_err());
        for text in ["auto:", "auto:x", "automatic", "auto:-1"] {
            assert!(SelSpec::Auto(text.into()).auto_factor().is_err(), "{text}");
        }
        assert_eq!(
            SelSpec::Auto("auto:1.5".into()).auto_factor().unwrap(),
            Some(1.5)
        );
        assert_eq!(SelSpec::Counts(vec![1]).auto_factor().unwrap(), None);
    }
}
