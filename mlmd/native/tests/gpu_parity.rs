//! GPU se_e2_a (omnimd-gpu) against the CPU reference `NativePotential`, which
//! is itself checked against DeePMD. Skips without a CUDA driver unless
//! OMNIMD_REQUIRE_GPU=1.
#![cfg(feature = "gpu")]
mod common;
use common::{device, frame, model};
use omnimd_ml_contract::{Evaluation, Frame, Potential, Units};
use omnimd_native::gpu_model::GpuPotential;
use omnimd_native::{Model, NativePotential};

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
    // The layer-by-layer cuBLAS fitting path (used for large chunks) gives
    // the same answer as the fused kernel used above.
    gpu.set_fused_fitting_max_atoms(0);
    compare(
        &format!("{label} (cuBLAS fitting)"),
        &gpu.eval(frame).unwrap(),
        &c,
        FLOAT64,
    );
    gpu.set_fused_fitting_max_atoms(usize::MAX);
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
    gpu.set_fused_fitting_max_atoms(0);
    compare(
        "float32 cuBLAS fitting",
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

#[test]
fn compressed_embedding_matches_cpu_and_falls_back_outside_the_table() {
    let Some(device) = device() else { return };
    let lattice = [[9.0, 0.0, 0.0], [1.5, 8.5, 0.0], [-1.0, 0.8, 9.2]];
    let model = model(2, vec![80, 160], 4.5, 6);
    let frame = frame(lattice, 5, 2, true, 16);
    let mut cpu = NativePotential::new(model.clone()).unwrap();
    let expected = cpu.eval(&frame).unwrap();

    // Tables down to 0.8 Å cover this frame (nearest pair ~1.2 Å). Quintic
    // interpolation at step 0.01 is good to ~1e-12; allow 1e-9.
    let mut gpu = GpuPotential::<f64>::new(&model, device.clone()).unwrap();
    assert!(!gpu.is_compressed());
    gpu.compress(0.8, 0.01).unwrap();
    assert!(gpu.is_compressed());
    compare(
        "compressed",
        &gpu.eval(&frame).unwrap(),
        &expected,
        (1e-11, 1e-9),
    );
    gpu.set_chunk_atoms(11);
    compare(
        "compressed chunks",
        &gpu.eval(&frame).unwrap(),
        &expected,
        (1e-11, 1e-9),
    );
    assert_eq!(gpu.table_fallbacks(), 0);

    // A coarse table is visibly worse, so the fine one is really in use.
    let mut coarse = GpuPotential::<f64>::new(&model, device.clone()).unwrap();
    coarse.compress(0.8, 0.5).unwrap();
    let rough = coarse.eval(&frame).unwrap();
    let difference = rough
        .forces
        .iter()
        .flatten()
        .zip(expected.forces.iter().flatten())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(difference > 1e-9, "coarse table too accurate: {difference}");

    // Fine tables that stop at 2 Å still cover this frame through the coarse
    // tail (ten times the step, down to 0.4 Å): no fallback, lower accuracy.
    let mut short = GpuPotential::<f64>::new(&model, device.clone()).unwrap();
    short.compress(2.0, 0.01).unwrap();
    compare(
        "tail",
        &short.eval(&frame).unwrap(),
        &expected,
        (1e-9, 1e-6),
    );
    assert_eq!(short.table_fallbacks(), 0);
    // A pair at 0.3 Å is closer than the tail reaches: that frame is
    // evaluated with the exact networks, same answer as without compression.
    let mut close = frame.clone();
    close.positions[1] = [
        close.positions[0][0] + 0.3,
        close.positions[0][1],
        close.positions[0][2],
    ];
    let expected_close = cpu.eval(&close).unwrap();
    compare(
        "fallback",
        &short.eval(&close).unwrap(),
        &expected_close,
        FLOAT64,
    );
    assert_eq!(short.table_fallbacks(), 1);

    let mut single = GpuPotential::<f32>::new(&model, device).unwrap();
    single.compress(0.8, 0.01).unwrap();
    compare(
        "compressed float32",
        &single.eval(&frame).unwrap(),
        &expected,
        (1e-6, 1e-4),
    );
    assert_eq!(single.table_fallbacks(), 0);
    assert!(single.compress(0.0, 0.01).is_err());
    assert!(single.compress(0.8, 0.0).is_err());
}
