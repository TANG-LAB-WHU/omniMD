//! GPU se_e2_a (omnimd-gpu) against the CPU reference `NativePotential`, which
//! is itself checked against DeePMD. Skips without a CUDA driver unless
//! OMNIMD_REQUIRE_GPU=1.
#![cfg(feature = "gpu")]
use omnimd_gpu::Device;
use omnimd_ml_contract::{Evaluation, Frame, Potential, Units};
use omnimd_native::gpu_model::GpuPotential;
use omnimd_native::{
    Activation, Descriptor, Layer, Model, NativePotential, Network, Normalization, Residual,
};
use std::sync::{Arc, OnceLock};

fn device() -> Option<Arc<Device>> {
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
fn values(count: usize, seed: u64) -> Vec<f64> {
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

fn layer(
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
fn model(ntypes: usize, sel: Vec<usize>, rcut: f64, seed: u64) -> Model {
    let embedding = (0..ntypes)
        .map(|i| {
            (0..ntypes)
                .map(|j| {
                    let s = seed + 100 * (i * ntypes + j) as u64;
                    Network {
                        layers: vec![
                            layer(1, 8, Activation::Tanh, Residual::None, false, s),
                            layer(8, 16, Activation::Tanh, Residual::Concat2, false, s + 10),
                            layer(16, 16, Activation::Tanh, Residual::Identity, true, s + 20),
                        ],
                    }
                })
                .collect()
        })
        .collect();
    let axis = 4;
    let fitting = (0..ntypes)
        .map(|i| {
            let s = seed + 5000 + 100 * i as u64;
            Network {
                layers: vec![
                    layer(16 * axis, 20, Activation::Tanh, Residual::None, false, s),
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
            embedding,
        },
        fitting,
        atom_energy_bias: (0..ntypes).map(|i| -3.0 - i as f64).collect(),
    }
}

/// Jittered lattice so no two atoms come closer than ~1 Å.
fn frame(lattice: [[f64; 3]; 3], grid: usize, ntypes: usize, periodic: bool, seed: u64) -> Frame {
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

/// Tolerances: relative energy, then forces/virial relative to their scale.
fn compare(label: &str, gpu: &Evaluation, cpu: &Evaluation, tolerance: (f64, f64)) {
    let energy = (gpu.energy - cpu.energy).abs() / cpu.energy.abs().max(1.0);
    assert!(
        energy < tolerance.0,
        "{label}: energy {} vs {}",
        gpu.energy,
        cpu.energy
    );
    let scale = cpu
        .forces
        .iter()
        .flatten()
        .fold(1e-3, |m: f64, v| m.max(v.abs()));
    let force = gpu
        .forces
        .iter()
        .flatten()
        .zip(cpu.forces.iter().flatten())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0, f64::max);
    assert!(
        force / scale < tolerance.1,
        "{label}: force error {force} (scale {scale})"
    );
    let vscale = cpu
        .virial
        .iter()
        .flatten()
        .fold(1e-3, |m: f64, v| m.max(v.abs()));
    let virial = gpu
        .virial
        .iter()
        .flatten()
        .zip(cpu.virial.iter().flatten())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0, f64::max);
    assert!(
        virial / vscale < tolerance.1,
        "{label}: virial error {virial} (scale {vscale})"
    );
}

const FLOAT64: (f64, f64) = (1e-12, 1e-10);

fn check(label: &str, model: Model, frame: &Frame) {
    let Some(device) = device() else { return };
    let mut gpu = GpuPotential::<f64>::new(&model, device).unwrap();
    let mut cpu = NativePotential::new(model).unwrap();
    let (g, c) = (gpu.eval(frame).unwrap(), cpu.eval(frame).unwrap());
    assert!(
        !gpu.last_truncated(),
        "{label}: sel too small for the test frame"
    );
    compare(label, &g, &c, FLOAT64);
    // Re-evaluation reuses cached kernels and must give the same answer.
    compare(label, &gpu.eval(frame).unwrap(), &c, FLOAT64);
    // Chunks that split type blocks unevenly must not change anything.
    gpu.set_chunk_atoms(7);
    compare(
        &format!("{label} (chunks of 7)"),
        &gpu.eval(frame).unwrap(),
        &c,
        FLOAT64,
    );
}

#[test]
fn float32_networks_stay_close_to_float64() {
    let Some(device) = device() else { return };
    let lattice = [[9.0, 0.0, 0.0], [1.5, 8.5, 0.0], [-1.0, 0.8, 9.2]];
    let model = model(2, vec![80, 160], 4.5, 5);
    let frame = frame(lattice, 5, 2, true, 15);
    let mut cpu = NativePotential::new(model.clone()).unwrap();
    let mut gpu = GpuPotential::<f32>::new(&model, device).unwrap();
    let c = cpu.eval(&frame).unwrap();
    // Float32 networks with float64 geometry and reductions: about single
    // precision on the network outputs, nothing worse.
    compare("float32", &gpu.eval(&frame).unwrap(), &c, (1e-6, 1e-4));
    gpu.set_chunk_atoms(9);
    compare(
        "float32 chunks",
        &gpu.eval(&frame).unwrap(),
        &c,
        (1e-6, 1e-4),
    );
}

#[test]
fn two_types_triclinic_matches_cpu() {
    let lattice = [[9.0, 0.0, 0.0], [1.5, 8.5, 0.0], [-1.0, 0.8, 9.2]];
    check(
        "triclinic",
        model(2, vec![80, 160], 4.5, 1),
        &frame(lattice, 5, 2, true, 11),
    );
}

#[test]
fn small_periodic_box_with_multiple_images_matches_cpu() {
    // 4.2 Å box with a 4.5 Å cutoff: neighbors include several images.
    let lattice = [[4.2, 0.0, 0.0], [0.0, 4.2, 0.0], [0.0, 0.0, 4.2]];
    check(
        "small box",
        model(2, vec![60, 120], 4.5, 2),
        &frame(lattice, 2, 2, true, 12),
    );
}

#[test]
fn open_boundary_and_three_types_match_cpu() {
    let lattice = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]];
    check(
        "open",
        model(3, vec![30, 30, 30], 4.0, 3),
        &frame(lattice, 4, 3, false, 13),
    );
}

#[test]
fn forces_are_the_energy_gradient() {
    let Some(device) = device() else { return };
    let lattice = [[7.0, 0.0, 0.0], [0.5, 7.0, 0.0], [0.0, 0.0, 7.0]];
    let base = frame(lattice, 3, 2, true, 14);
    let mut gpu = GpuPotential::<f64>::new(&model(2, vec![40, 80], 4.0, 4), device).unwrap();
    let forces = gpu.eval(&base).unwrap().forces;
    let h = 1e-5;
    for atom in [0, 7, 19] {
        for axis in 0..3 {
            let mut shifted = |s: f64| {
                let mut f = base.clone();
                f.positions[atom][axis] += s;
                gpu.eval(&f).unwrap().energy
            };
            let fd = -(shifted(h) - shifted(-h)) / (2.0 * h);
            let error = (fd - forces[atom][axis]).abs();
            assert!(
                error < 1e-6 * forces[atom][axis].abs().max(1.0),
                "atom {atom} axis {axis}: {fd} vs {}",
                forces[atom][axis]
            );
        }
    }
}

#[derive(serde::Deserialize)]
struct DeepmdReference {
    frame: Frame,
    energy: f64,
    forces: Vec<[f64; 3]>,
    virial: [[f64; 3]; 3],
}

/// `sel` = [4, 6] against ~27 neighbors per atom: the result depends on
/// which neighbors are kept. Reference from DeePMD-kit 3.1.3 itself
/// (`tools/make_truncation_fixture.py`); the CPU `NativePotential` rejects
/// such frames, so this is the only check of the selection rule end to end.
#[test]
fn tight_sel_matches_deepmd_reference() {
    let Some(device) = device() else { return };
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
    let model = NativePotential::load(root.join("truncation-model.native.json")).unwrap();
    let reference: DeepmdReference = serde_json::from_str(
        &std::fs::read_to_string(root.join("truncation-deepmd-reference.json")).unwrap(),
    )
    .unwrap();
    let expected = Evaluation {
        energy: reference.energy,
        forces: reference.forces,
        virial: reference.virial,
        units: Units::EvAngstrom,
    };
    let mut gpu = GpuPotential::<f64>::new(model.model(), device).unwrap();
    let got = gpu.eval(&reference.frame).unwrap();
    assert!(gpu.last_truncated(), "the fixture must truncate");
    compare("tight sel vs DeePMD", &got, &expected, FLOAT64);
    gpu.set_chunk_atoms(5);
    compare(
        "tight sel vs DeePMD (chunks)",
        &gpu.eval(&reference.frame).unwrap(),
        &expected,
        FLOAT64,
    );
}
