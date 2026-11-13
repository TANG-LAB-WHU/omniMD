//! Builders shared by the GPU tests: a device that skips without a driver
//! (unless OMNIMD_REQUIRE_GPU is set), small random se_e2_a models and
//! jittered-lattice frames.
#![allow(dead_code)]
use omnimd_gpu::Device;
use omnimd_ml_contract::{Frame, Units};
use omnimd_native::{Activation, Descriptor, Layer, Model, Network, Normalization, Residual};
use std::sync::{Arc, OnceLock};

pub fn device() -> Option<Arc<Device>> {
    static DEVICE: OnceLock<Option<Arc<Device>>> = OnceLock::new();
    DEVICE
        .get_or_init(|| match Device::new(0) {
            Ok(device) => Some(Arc::new(device)),
            Err(error) if std::env::var_os("OMNIMD_REQUIRE_GPU").is_some() => {
                panic!("OMNIMD_REQUIRE_GPU is set but no device: {error}")
            }
            Err(error) => {
                eprintln!("skipping GPU tests: {error}");
                None
            }
        })
        .clone()
}

/// Deterministic values in (-1, 1).
pub fn values(count: usize, seed: u64) -> Vec<f64> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        })
        .collect()
}

pub fn layer(
    input: usize,
    output: usize,
    activation: Activation,
    residual: Residual,
    timestep: bool,
    seed: u64,
) -> Layer {
    let scale = 1.0 / (input as f64).sqrt();
    let w = values(input * output, seed);
    Layer {
        weights: (0..input)
            .map(|i| {
                w[i * output..][..output]
                    .iter()
                    .map(|v| v * scale)
                    .collect()
            })
            .collect(),
        bias: values(output, seed + 1).iter().map(|v| 0.3 * v).collect(),
        activation,
        residual,
        timestep: timestep.then(|| {
            values(output, seed + 2)
                .iter()
                .map(|v| 0.1 + 0.02 * v)
                .collect()
        }),
    }
}

/// Small se_e2_a with every layer feature: concat and identity residuals,
/// timesteps, nonzero radial mean.
pub fn model(ntypes: usize, sel: Vec<usize>, rcut: f64, seed: u64) -> Model {
    model_of_width(ntypes, sel, rcut, seed, 16, 4)
}

/// The same with an embedding of `width` outputs (even) and `axis` of them
/// as axis neurons. More than 32 outputs take more than one warp of the
/// kernels that have a thread for each.
#[allow(dead_code)]
pub fn model_of_width(
    ntypes: usize,
    sel: Vec<usize>,
    rcut: f64,
    seed: u64,
    width: usize,
    axis: usize,
) -> Model {
    let half = width / 2;
    let embedding = (0..ntypes)
        .map(|i| {
            (0..ntypes)
                .map(|j| {
                    let s = seed + 100 * (i * ntypes + j) as u64;
                    Network {
                        layers: vec![
                            layer(1, half, Activation::Tanh, Residual::None, false, s),
                            layer(
                                half,
                                width,
                                Activation::Tanh,
                                Residual::Concat2,
                                false,
                                s + 10,
                            ),
                            layer(
                                width,
                                width,
                                Activation::Tanh,
                                Residual::Identity,
                                true,
                                s + 20,
                            ),
                        ],
                    }
                })
                .collect()
        })
        .collect();
    let fitting = (0..ntypes)
        .map(|i| {
            let s = seed + 5000 + 100 * i as u64;
            Network {
                layers: vec![
                    layer(width * axis, 20, Activation::Tanh, Residual::None, false, s),
                    layer(20, 20, Activation::Tanh, Residual::Identity, true, s + 10),
                    layer(20, 1, Activation::Linear, Residual::None, false, s + 20),
                ],
            }
        })
        .collect();
    Model {
        format: "omnimd-native-se-e2-a".into(),
        version: 1,
        precision: "float64".into(),
        units: Units::EvAngstrom,
        type_map: ["O", "H", "C"][..ntypes]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        descriptor: Descriptor {
            kind: "se_e2_a".into(),
            rcut,
            rcut_smth: 0.5,
            sel,
            axis_neuron: axis,
            normalization: (0..ntypes)
                .map(|i| Normalization {
                    mean: [0.05 + 0.01 * i as f64, 0.0, 0.0, 0.0],
                    std: [0.3, 0.2, 0.2, 0.2],
                })
                .collect(),
            type_one_side: false,
            embedding,
        },
        fitting,
        atom_energy_bias: (0..ntypes).map(|i| -3.0 - i as f64).collect(),
    }
}

/// The same networks around an se_e2_r descriptor: no axis neurons, the
/// radial normalization only, fitting nets that take the embedding width.
#[allow(dead_code)]
pub fn radial_model(ntypes: usize, sel: Vec<usize>, rcut: f64, seed: u64, width: usize) -> Model {
    let mut model = model_of_width(ntypes, sel, rcut, seed, width, 1);
    model.format = "omnimd-native-se-e2-r".into();
    model.descriptor.kind = "se_e2_r".into();
    model.descriptor.axis_neuron = 0;
    for normalization in &mut model.descriptor.normalization {
        normalization.std = [normalization.std[0], 1.0, 1.0, 1.0];
    }
    model
}

/// The model with the embedding nets of its first central type for every
/// central type, and the mark that says so (`type_one_side`).
#[allow(dead_code)]
pub fn one_sided(mut model: Model) -> Model {
    let first = model.descriptor.embedding[0].clone();
    for row in &mut model.descriptor.embedding {
        *row = first.clone();
    }
    model.descriptor.type_one_side = true;
    model
}

/// Jittered lattice so no two atoms come closer than ~1 Å.
pub fn frame(
    lattice: [[f64; 3]; 3],
    grid: usize,
    ntypes: usize,
    periodic: bool,
    seed: u64,
) -> Frame {
    let jitter = values(grid * grid * grid * 3, seed);
    let mut positions = Vec::new();
    for x in 0..grid {
        for y in 0..grid {
            for z in 0..grid {
                let k = positions.len();
                let frac = [x, y, z].map(|c| c as f64 / grid as f64);
                let f: [f64; 3] =
                    std::array::from_fn(|a| frac[a] + 0.15 / grid as f64 * jitter[3 * k + a]);
                positions.push(std::array::from_fn(|c| {
                    (0..3).map(|a| f[a] * lattice[a][c]).sum()
                }));
            }
        }
    }
    let names = ["O", "H", "C"];
    Frame {
        species: (0..positions.len())
            .map(|i| names[(i * 7 + i / 3) % ntypes].to_string())
            .collect(),
        positions,
        cell: periodic.then_some(lattice),
        units: Units::EvAngstrom,
    }
}
