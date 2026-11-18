//! GPU se_e2_a, se_e2_r, se_e3 and hybrids of them (omnimd-gpu) against the
//! CPU reference `NativePotential`, which is itself checked against DeePMD.
//! Skips without a CUDA driver unless OMNIMD_REQUIRE_GPU=1.
#![cfg(feature = "gpu")]
mod common;
use common::{device, frame, model};
use omnimd_ml_contract::{Committee, Evaluation, Frame, Potential, Units};
use omnimd_native::gpu_model::GpuPotential;
use omnimd_native::{Activation, Model, NativePotential, Residual};

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
    // The fused kernel with its first layer in tiles (the default only for
    // large float32 nets) gives the same answer again.
    assert!(gpu.set_tiled_fitting(true), "{label}: no tiles");
    compare(
        &format!("{label} (tiled fitting)"),
        &gpu.eval(frame).unwrap(),
        &c,
        FLOAT64,
    );
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
    // Float32 from the stored environment rows to their adjoint, float64
    // geometry around it: about single precision on the result, nothing
    // worse.
    compare("float32", &gpu.eval(&frame).unwrap(), &c, (1e-6, 1e-4));
    assert!(gpu.set_tiled_fitting(true));
    compare(
        "float32 tiled fitting",
        &gpu.eval(&frame).unwrap(),
        &c,
        (1e-6, 1e-4),
    );
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
fn one_potential_serves_frames_of_different_sizes_in_turn() {
    let Some(device) = device() else { return };
    // The potential keeps its neighbor search and its buffers between
    // evaluations: a frame of another size, cell or boundary must see
    // nothing of the one before.
    let model = model(2, vec![80, 160], 4.5, 6);
    let triclinic = [[9.0, 0.0, 0.0], [1.5, 8.5, 0.0], [-1.0, 0.8, 9.2]];
    let cubic = [[6.3, 0.0, 0.0], [0.0, 6.3, 0.0], [0.0, 0.0, 6.3]];
    let large = frame(triclinic, 5, 2, true, 16);
    let small = frame(cubic, 3, 2, true, 17);
    let open = frame(triclinic, 4, 2, false, 18);
    let mut cpu = NativePotential::new(model.clone()).unwrap();
    let mut gpu = GpuPotential::<f64>::new(&model, device.clone()).unwrap();
    let mut single = GpuPotential::<f32>::new(&model, device).unwrap();
    let turns = [
        ("large", &large),
        ("small", &small),
        ("open", &open),
        ("large again", &large),
        ("small again", &small),
    ];
    for (label, frame) in turns {
        let c = cpu.eval(frame).unwrap();
        compare(label, &gpu.eval(frame).unwrap(), &c, FLOAT64);
        let label = format!("{label}, float32");
        compare(&label, &single.eval(frame).unwrap(), &c, (1e-6, 1e-4));
    }
}

#[test]
fn coincident_atoms_are_an_error() {
    let Some(device) = device() else { return };
    let lattice = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]];
    let model = model(2, vec![60, 120], 4.0, 7);
    let mut gpu = GpuPotential::<f64>::new(&model, device).unwrap();
    let good = frame(lattice, 4, 2, true, 19);
    gpu.eval(&good).unwrap();
    let mut bad = good.clone();
    bad.positions[9] = bad.positions[40];
    let error = gpu.eval(&bad).unwrap_err().to_string();
    assert!(error.contains("coincident"), "{error}");
    // The potential is as usable afterwards as before.
    let mut cpu = NativePotential::new(model).unwrap();
    compare(
        "after an error",
        &gpu.eval(&good).unwrap(),
        &cpu.eval(&good).unwrap(),
        FLOAT64,
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

/// An embedding wider than a warp: the descriptor and force kernels have a
/// thread for each of its outputs, and what one warp of them writes to
/// shared memory another may still be reading.
#[test]
fn an_embedding_wider_than_a_warp_matches_cpu() {
    let lattice = [[7.0, 0.0, 0.0], [0.5, 7.0, 0.0], [0.0, 0.0, 7.0]];
    check(
        "wide embedding",
        common::model_of_width(2, vec![40, 80], 4.0, 31, 40, 6),
        &frame(lattice, 3, 2, true, 32),
    );
    // Wider than the block of the kernels as well: threads take turns.
    check(
        "embedding wider than a block",
        common::model_of_width(2, vec![40, 80], 4.0, 33, 136, 5),
        &frame(lattice, 3, 2, true, 34),
    );
}

/// The descriptor counts the neighbors of each type before it sums them: a
/// type without slots has none, wherever it stands among the types.
#[test]
fn a_type_without_slots_matches_cpu() {
    let lattice = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]];
    // The last type has no slots, and the frame none of its atoms.
    let two = frame(lattice, 4, 2, false, 22);
    check(
        "no slots for the last type",
        model(3, vec![30, 30, 0], 4.0, 21),
        &two,
    );
    // The same for the first type: oxygen becomes carbon.
    let mut renamed = two.clone();
    for name in &mut renamed.species {
        if name == "O" {
            *name = "C".into();
        }
    }
    check(
        "no slots for the first type",
        model(3, vec![0, 30, 30], 4.0, 23),
        &renamed,
    );
    // And in a periodic cell, where every atom has neighbors of both others.
    let cell = [[7.0, 0.0, 0.0], [0.5, 7.0, 0.0], [0.0, 0.0, 7.0]];
    let mut periodic = frame(cell, 3, 2, true, 24);
    for name in &mut periodic.species {
        if name == "H" {
            *name = "C".into();
        }
    }
    check(
        "no slots for the middle type",
        model(3, vec![60, 0, 60], 4.0, 25),
        &periodic,
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

/// Four frames that share nothing but the model: sizes, cells and boundary.
fn assorted_frames() -> Vec<Frame> {
    let triclinic = [[9.0, 0.0, 0.0], [1.5, 8.5, 0.0], [-1.0, 0.8, 9.2]];
    // Narrower than the cutoff: neighbors include several images.
    let small = [[4.2, 0.0, 0.0], [0.0, 4.2, 0.0], [0.0, 0.0, 4.2]];
    let cubic = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]];
    vec![
        frame(triclinic, 5, 2, true, 31),
        frame(small, 2, 2, true, 32),
        frame(cubic, 4, 2, false, 33),
        frame(triclinic, 4, 2, true, 34),
    ]
}

#[test]
fn a_batch_of_frames_matches_each_frame_alone() {
    let Some(device) = device() else { return };
    let model = model(2, vec![80, 160], 4.5, 21);
    let frames = assorted_frames();
    let batch: Vec<&Frame> = frames.iter().collect();
    let mut cpu = NativePotential::new(model.clone()).unwrap();
    let expected: Vec<Evaluation> = frames.iter().map(|f| cpu.eval(f).unwrap()).collect();
    let against = |label: &str, got: &[Evaluation], tolerance: (f64, f64)| {
        assert_eq!(got.len(), expected.len(), "{label}");
        for (index, (g, c)) in got.iter().zip(&expected).enumerate() {
            compare(&format!("{label}, frame {index}"), g, c, tolerance);
        }
    };
    let mut gpu = GpuPotential::<f64>::new(&model, device.clone()).unwrap();
    assert_eq!(gpu.models(), 1);
    let together = gpu.eval_batch(&batch).unwrap();
    against("batch", &together, FLOAT64);
    // Nothing of a frame leaks into another: each is what it is alone, to
    // the rounding of sums that may run in another order.
    for (index, (frame, got)) in frames.iter().zip(&together).enumerate() {
        let alone = gpu.eval(frame).unwrap();
        compare(&format!("frame {index} alone"), got, &alone, (1e-13, 1e-11));
    }
    // The order of the frames is the order of the results.
    let reversed: Vec<&Frame> = frames.iter().rev().collect();
    let mut backwards = gpu.eval_batch(&reversed).unwrap();
    backwards.reverse();
    against("reversed batch", &backwards, FLOAT64);
    // One frame twice is two results.
    let twice = gpu.eval_batch(&[&frames[1], &frames[1]]).unwrap();
    compare("twice, first", &twice[0], &expected[1], FLOAT64);
    compare("twice, second", &twice[1], &expected[1], FLOAT64);
    assert!(gpu.eval_batch(&[]).unwrap().is_empty());
    // Every path of the single frame, over the batch: chunks that cut
    // through frames and types, cuBLAS fitting, tiles, tables.
    gpu.set_chunk_atoms(7);
    against("batch in chunks", &gpu.eval_batch(&batch).unwrap(), FLOAT64);
    gpu.set_chunk_atoms(100_000);
    gpu.set_fused_fitting_max_atoms(0);
    against("batch, cuBLAS", &gpu.eval_batch(&batch).unwrap(), FLOAT64);
    gpu.set_fused_fitting_max_atoms(usize::MAX);
    assert!(gpu.set_tiled_fitting(true));
    against("batch, tiles", &gpu.eval_batch(&batch).unwrap(), FLOAT64);
    gpu.compress(0.8, 0.01).unwrap();
    let fine = gpu.eval_batch(&batch).unwrap();
    against("batch, tables", &fine, (1e-11, 1e-9));
    assert_eq!(gpu.table_fallbacks(), 0);
    // Coarse tables, to tell their results from those of the networks.
    gpu.compress(0.8, 0.5).unwrap();
    let tabulated = gpu.eval_batch(&batch).unwrap();
    for index in [0, 3] {
        let (rough, exact) = (&tabulated[index].forces, &expected[index].forces);
        let apart = rough
            .iter()
            .flatten()
            .zip(exact.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(apart > 1e-9, "coarse tables too accurate: {apart}");
    }

    // A pair closer than the tables reach, in one frame of the batch: that
    // frame alone goes through the exact networks, the others keep the
    // tables and give what they gave before.
    let mut close = frames[2].clone();
    close.positions[1] = close.positions[0];
    close.positions[1][0] += 0.1;
    let with_close = [&frames[0], &close, &frames[3]];
    let got = gpu.eval_batch(&with_close).unwrap();
    assert_eq!(gpu.table_fallbacks(), 1);
    compare(
        "exact in a batch",
        &got[1],
        &cpu.eval(&close).unwrap(),
        FLOAT64,
    );
    compare("tables kept, first", &got[0], &tabulated[0], (1e-13, 1e-11));
    compare("tables kept, last", &got[2], &tabulated[3], (1e-13, 1e-11));

    // An error in one frame fails the batch and names the frame.
    let mut bad = frames[3].clone();
    bad.positions[9] = bad.positions[40];
    let error = gpu
        .eval_batch(&[&frames[0], &frames[1], &bad])
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("frame 2 of the batch") && error.contains("coincident"),
        "{error}"
    );
    let mut unknown = frames[1].clone();
    unknown.species[3] = "Xe".into();
    let error = gpu.eval_batch(&[&unknown, &frames[0]]).unwrap_err();
    assert!(
        error.to_string().contains("frame 0 of the batch"),
        "{error}"
    );
    // A single frame is named by nobody.
    let error = gpu.eval_batch(&[&bad]).unwrap_err().to_string();
    assert!(
        !error.contains("batch") && error.contains("coincident"),
        "{error}"
    );
    // The potential is as usable afterwards as before.
    gpu.compress(0.8, 0.01).unwrap();
    against(
        "after errors",
        &gpu.eval_batch(&batch).unwrap(),
        (1e-11, 1e-9),
    );

    let mut single = GpuPotential::<f32>::new(&model, device).unwrap();
    against(
        "float32 batch",
        &single.eval_batch(&batch).unwrap(),
        (1e-6, 1e-4),
    );
    single.set_chunk_atoms(9);
    single.set_fused_fitting_max_atoms(0);
    let chunked = single.eval_batch(&batch).unwrap();
    against("float32 batch, chunks and cuBLAS", &chunked, (1e-6, 1e-4));
}

#[test]
fn truncation_in_a_batch_is_reported_for_the_batch() {
    let Some(device) = device() else { return };
    // Room for 4 and 6 neighbors against some twenty within the cutoff.
    let model = model(2, vec![4, 6], 4.0, 22);
    let lattice = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]];
    let dense = frame(lattice, 4, 2, true, 35);
    // Two atoms far apart in the open: no neighbors at all.
    let mut sparse = frame(lattice, 2, 2, false, 36);
    sparse.positions.truncate(2);
    sparse.species.truncate(2);
    sparse.positions[1] = [30.0, 30.0, 30.0];
    let mut gpu = GpuPotential::<f64>::new(&model, device).unwrap();
    let alone = [gpu.eval(&sparse).unwrap(), gpu.eval(&dense).unwrap()];
    assert!(gpu.last_truncated());
    gpu.eval(&sparse).unwrap();
    assert!(!gpu.last_truncated());
    let together = gpu.eval_batch(&[&sparse, &dense]).unwrap();
    assert!(gpu.last_truncated(), "one frame of the batch truncates");
    // The neighbors kept are those kept for the frame alone.
    compare("sparse", &together[0], &alone[0], (1e-13, 1e-11));
    compare("dense", &together[1], &alone[1], (1e-13, 1e-11));
    gpu.eval_batch(&[&sparse, &sparse]).unwrap();
    assert!(!gpu.last_truncated());
}

/// Models of one architecture with weights, statistics and biases of their
/// own, as an ensemble trained from different seeds on different data.
fn committee_models(count: usize) -> Vec<Model> {
    (0..count)
        .map(|member| {
            let mut model = model(2, vec![80, 160], 4.5, 40 + 7 * member as u64);
            for (t, normalization) in model.descriptor.normalization.iter_mut().enumerate() {
                normalization.mean[0] += 0.013 * member as f64 - 0.004 * t as f64;
                normalization.std[0] *= 1.0 + 0.05 * member as f64;
                // The angular widths stay equal, as the format asks.
                for width in &mut normalization.std[1..] {
                    *width *= 1.0 - 0.03 * member as f64;
                }
            }
            for bias in &mut model.atom_energy_bias {
                *bias += 0.37 * member as f64;
            }
            model
        })
        .collect()
}

#[test]
fn a_committee_matches_its_members_alone() {
    let Some(device) = device() else { return };
    let models = committee_models(3);
    let members: Vec<&Model> = models.iter().collect();
    let frames = assorted_frames();
    let batch: Vec<&Frame> = frames.iter().collect();
    // [frame][member]
    let expected: Vec<Vec<Evaluation>> = frames
        .iter()
        .map(|frame| {
            models
                .iter()
                .map(|model| {
                    NativePotential::new(model.clone())
                        .unwrap()
                        .eval(frame)
                        .unwrap()
                })
                .collect()
        })
        .collect();
    // The members do differ.
    assert!((expected[0][0].energy - expected[0][1].energy).abs() > 1e-3);
    let against = |label: &str, got: &[Vec<Evaluation>], count: usize, tolerance: (f64, f64)| {
        assert_eq!(got.len(), frames.len(), "{label}");
        for (index, (g, c)) in got.iter().zip(&expected).enumerate() {
            assert_eq!(g.len(), count, "{label}");
            for (member, (g, c)) in g.iter().zip(c).enumerate() {
                compare(
                    &format!("{label}, frame {index} member {member}"),
                    g,
                    c,
                    tolerance,
                );
            }
        }
    };
    let mut gpu = GpuPotential::<f64>::committee(&members, device.clone()).unwrap();
    assert_eq!((gpu.models(), gpu.members()), (3, 3));
    against("committee", &gpu.eval_members(&batch).unwrap(), 3, FLOAT64);
    // Fewer members are the first ones; as a potential it is the first.
    against(
        "two members",
        &gpu.evaluate_frames(&batch, 2).unwrap(),
        2,
        FLOAT64,
    );
    against(
        "one member",
        &gpu.evaluate_frames(&batch, 1).unwrap(),
        1,
        FLOAT64,
    );
    compare(
        "as a potential",
        &gpu.eval(&frames[0]).unwrap(),
        &expected[0][0],
        FLOAT64,
    );
    let first: Vec<Evaluation> = gpu.eval_batch(&batch).unwrap();
    for (index, (g, c)) in first.iter().zip(&expected).enumerate() {
        compare(&format!("first member, frame {index}"), g, &c[0], FLOAT64);
    }
    assert!(gpu.evaluate_frames(&batch, 0).is_err());
    assert!(gpu.evaluate_frames(&batch, 4).is_err());
    assert!(gpu.evaluate_frames(&[], 3).unwrap().is_empty());
    // One frame, as model deviation along a trajectory asks for it.
    let one = gpu.eval_members(&[&frames[1]]).unwrap();
    for (member, (g, c)) in one[0].iter().zip(&expected[1]).enumerate() {
        compare(&format!("one frame, member {member}"), g, c, FLOAT64);
    }
    // The paths of the single model, with three models deep.
    gpu.set_chunk_atoms(7);
    against(
        "committee in chunks",
        &gpu.eval_members(&batch).unwrap(),
        3,
        FLOAT64,
    );
    gpu.set_chunk_atoms(100_000);
    gpu.set_fused_fitting_max_atoms(0);
    against(
        "committee, cuBLAS",
        &gpu.eval_members(&batch).unwrap(),
        3,
        FLOAT64,
    );
    gpu.set_fused_fitting_max_atoms(usize::MAX);
    assert!(gpu.set_tiled_fitting(true));
    against(
        "committee, tiles",
        &gpu.eval_members(&batch).unwrap(),
        3,
        FLOAT64,
    );
    // Each member's tables follow its own statistics.
    gpu.compress(0.8, 0.01).unwrap();
    against(
        "committee, tables",
        &gpu.eval_members(&batch).unwrap(),
        3,
        (1e-11, 1e-9),
    );
    assert_eq!(gpu.table_fallbacks(), 0);

    let mut single = GpuPotential::<f32>::committee(&members, device.clone()).unwrap();
    against(
        "float32 committee",
        &single.eval_members(&batch).unwrap(),
        3,
        (1e-6, 1e-4),
    );
    single.set_chunk_atoms(10);
    single.set_fused_fitting_max_atoms(0);
    let chunked = single.eval_members(&batch).unwrap();
    against(
        "float32 committee, chunks and cuBLAS",
        &chunked,
        3,
        (1e-6, 1e-4),
    );

    // Models that cannot share the device pass are refused with the reason.
    use omnimd_native::gpu_model::committee_mismatch;
    assert_eq!(committee_mismatch(&members), None);
    assert!(committee_mismatch(&[]).is_some());
    let refused = |change: fn(&mut Model), what: &str| {
        let mut other = models[1].clone();
        change(&mut other);
        let mixed = [&models[0], &other];
        let reason = committee_mismatch(&mixed).unwrap();
        assert!(reason.contains(what), "{reason}");
        let error = GpuPotential::<f64>::committee(&mixed, device.clone())
            .err()
            .unwrap_or_else(|| panic!("a committee with another {what}"));
        assert!(error.to_string().contains("committee"), "{error}");
    };
    refused(|m| m.descriptor.sel = vec![80, 150], "sel");
    refused(|m| m.descriptor.rcut = 4.4, "cutoffs");
    refused(|m| m.descriptor.rcut_smth = 0.6, "cutoffs");
    refused(|m| m.type_map.swap(0, 1), "type map");
    refused(
        |m| {
            // A wider second fitting layer.
            let network = &mut m.fitting[1];
            network.layers[0] = common::layer(64, 24, Activation::Tanh, Residual::None, false, 1);
            network.layers[1] =
                common::layer(24, 24, Activation::Tanh, Residual::Identity, true, 2);
            network.layers[2] = common::layer(24, 1, Activation::Linear, Residual::None, false, 3);
        },
        "fitting nets",
    );
    refused(
        |m| {
            let layer = &mut m.descriptor.embedding[1][0].layers[2];
            layer.activation = Activation::Softplus;
        },
        "embedding nets",
    );
    // Other axis neurons, or another width of the embedding: named as the
    // layout of the descriptor, not as nets of another shape further on.
    let layout = |other: Model, what: &str| {
        let mixed = [&models[0], &other];
        assert_eq!(
            committee_mismatch(&mixed).unwrap(),
            format!("model 1 differs from the first in its {what}")
        );
        let error = GpuPotential::<f64>::committee(&mixed, device.clone())
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains("differs from the first in its descriptor, type map, cutoffs, sel, axis_neuron or embedding width"),
            "{what}: {error}"
        );
    };
    layout(
        common::model_of_width(2, vec![80, 160], 4.5, 47, 16, 2),
        "axis_neuron",
    );
    layout(
        common::model_of_width(2, vec![80, 160], 4.5, 47, 8, 4),
        "embedding nets",
    );
    assert!(GpuPotential::<f64>::committee(&[], device).is_err());
}

/// The se_e2_a test model's networks around an se_e2_r descriptor.
fn radial(ntypes: usize, sel: Vec<usize>, rcut: f64, seed: u64) -> Model {
    common::radial_model(ntypes, sel, rcut, seed, 16)
}

/// se_e2_r on the device against the CPU reference, on the kinds of frames
/// the se_e2_a checks use.
#[test]
fn se_e2_r_matches_cpu() {
    let triclinic = [[9.0, 0.0, 0.0], [1.5, 8.5, 0.0], [-1.0, 0.8, 9.2]];
    check(
        "radial, triclinic",
        radial(2, vec![80, 160], 4.5, 51),
        &frame(triclinic, 5, 2, true, 52),
    );
    // Narrower than the cutoff: neighbors include several images.
    let small = [[4.2, 0.0, 0.0], [0.0, 4.2, 0.0], [0.0, 0.0, 4.2]];
    check(
        "radial, several images",
        radial(2, vec![60, 120], 4.5, 53),
        &frame(small, 2, 2, true, 54),
    );
    let cubic = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]];
    check(
        "radial, open",
        radial(3, vec![30, 30, 30], 4.0, 55),
        &frame(cubic, 4, 3, false, 56),
    );
    // An embedding wider than a warp, and wider than a block of the kernels.
    let lattice = [[7.0, 0.0, 0.0], [0.5, 7.0, 0.0], [0.0, 0.0, 7.0]];
    check(
        "radial, wide embedding",
        common::radial_model(2, vec![40, 80], 4.0, 57, 40),
        &frame(lattice, 3, 2, true, 58),
    );
    check(
        "radial, embedding wider than a block",
        common::radial_model(2, vec![40, 80], 4.0, 59, 136),
        &frame(lattice, 3, 2, true, 60),
    );
    // A type without slots, and without atoms in the frame.
    check(
        "radial, no slots for the last type",
        radial(3, vec![30, 30, 0], 4.0, 61),
        &frame(cubic, 4, 2, false, 62),
    );
}

#[derive(serde::Deserialize)]
struct RadialCase {
    model: String,
    frame: Frame,
    energy: f64,
    forces: Vec<[f64; 3]>,
    virial: [[f64; 3]; 3],
}

#[derive(serde::Deserialize)]
struct RadialReference {
    cases: Vec<RadialCase>,
}

/// se_e2_r against DeePMD-kit 3.1.3 itself (`tools/make_se_r_fixture.py`):
/// the model whose `sel` holds every neighbor, with and without a cell, and
/// the one with `sel` = [4, 6] against some 23 neighbors, where the result
/// depends on which neighbors are kept.
#[test]
fn se_e2_r_matches_deepmd_reference() {
    let Some(device) = device() else { return };
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
    let reference: RadialReference = serde_json::from_str(
        &std::fs::read_to_string(root.join("se-r-deepmd-reference.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(reference.cases.len(), 3);
    for (index, case) in reference.cases.into_iter().enumerate() {
        let model = NativePotential::load(root.join(&case.model)).unwrap();
        let expected = Evaluation {
            energy: case.energy,
            forces: case.forces,
            virial: case.virial,
            units: Units::EvAngstrom,
        };
        let label = format!("{} (case {index})", case.model);
        let mut gpu = GpuPotential::<f64>::new(model.model(), device.clone()).unwrap();
        assert_eq!(gpu.metadata().backend, "native-gpu-se-e2-r");
        let got = gpu.eval(&case.frame).unwrap();
        assert_eq!(
            gpu.last_truncated(),
            case.model == "se-r-tight-model.native.json",
            "{label}"
        );
        compare(&label, &got, &expected, FLOAT64);
        gpu.set_chunk_atoms(5);
        compare(
            &format!("{label}, chunks"),
            &gpu.eval(&case.frame).unwrap(),
            &expected,
            FLOAT64,
        );
    }
}

/// Float32, the embedding tables, batches and committees take an se_e2_r
/// model through the same stages as an se_e2_a one: each against the CPU.
#[test]
fn se_e2_r_in_float32_compressed_in_batches_and_committees() {
    let Some(device) = device() else { return };
    let triclinic = [[9.0, 0.0, 0.0], [1.5, 8.5, 0.0], [-1.0, 0.8, 9.2]];
    let cubic = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]];
    let frames = [
        frame(triclinic, 5, 2, true, 71),
        frame(cubic, 4, 2, false, 72),
        frame(triclinic, 4, 2, true, 73),
    ];
    let models: Vec<Model> = (0..3)
        .map(|member| {
            let mut model = radial(2, vec![80, 160], 4.5, 74 + 7 * member as u64);
            for (t, normalization) in model.descriptor.normalization.iter_mut().enumerate() {
                normalization.mean[0] += 0.013 * member as f64 - 0.004 * t as f64;
                normalization.std[0] *= 1.0 + 0.05 * member as f64;
            }
            for bias in &mut model.atom_energy_bias {
                *bias += 0.37 * member as f64;
            }
            model
        })
        .collect();
    let expected: Vec<Vec<Evaluation>> = frames
        .iter()
        .map(|frame| {
            models
                .iter()
                .map(|model| {
                    NativePotential::new(model.clone())
                        .unwrap()
                        .eval(frame)
                        .unwrap()
                })
                .collect()
        })
        .collect();
    let batch: Vec<&Frame> = frames.iter().collect();
    let members: Vec<&Model> = models.iter().collect();

    // Float64: a batch for one model, then for the committee.
    let mut one = GpuPotential::<f64>::new(&models[0], device.clone()).unwrap();
    for (index, got) in one.eval_batch(&batch).unwrap().iter().enumerate() {
        compare(
            &format!("radial batch, frame {index}"),
            got,
            &expected[index][0],
            FLOAT64,
        );
    }
    let mut committee = GpuPotential::<f64>::committee(&members, device.clone()).unwrap();
    assert_eq!(committee.members(), 3);
    for (index, of_frame) in committee.eval_members(&batch).unwrap().iter().enumerate() {
        for (member, got) in of_frame.iter().enumerate() {
            compare(
                &format!("radial committee, frame {index}, member {member}"),
                got,
                &expected[index][member],
                FLOAT64,
            );
        }
    }
    // The tables: quintic interpolation at step 0.01, as for se_e2_a.
    committee.compress(0.8, 0.01).unwrap();
    for (index, of_frame) in committee.eval_members(&batch).unwrap().iter().enumerate() {
        for (member, got) in of_frame.iter().enumerate() {
            compare(
                &format!("radial tables, frame {index}, member {member}"),
                got,
                &expected[index][member],
                (1e-9, 1e-7),
            );
        }
    }
    assert_eq!(committee.table_fallbacks(), 0);
    // Float32 networks and sums: about single precision, nothing worse.
    let mut single = GpuPotential::<f32>::committee(&members, device.clone()).unwrap();
    for compressed in [false, true] {
        if compressed {
            single.compress(0.8, 0.01).unwrap();
        }
        for (index, of_frame) in single.eval_members(&batch).unwrap().iter().enumerate() {
            for (member, got) in of_frame.iter().enumerate() {
                compare(
                    &format!(
                        "radial float32 (tables: {compressed}), frame {index}, member {member}"
                    ),
                    got,
                    &expected[index][member],
                    (1e-6, 1e-4),
                );
            }
        }
    }
    // The forces are the gradient of the energy the device gives.
    let h = 1e-5;
    let base = &frames[0];
    let forces = one.eval(base).unwrap().forces;
    for atom in [0, 7, 19] {
        for axis in 0..3 {
            let mut shifted = |s: f64| {
                let mut f = base.clone();
                f.positions[atom][axis] += s;
                one.eval(&f).unwrap().energy
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
    // Models of the two descriptors are not one committee.
    let other = model(2, vec![80, 160], 4.5, 6);
    let mixed = [&models[0], &other];
    let error = GpuPotential::<f64>::committee(&mixed, device)
        .err()
        .unwrap();
    assert!(
        error
            .to_string()
            .contains("differs from the first in its descriptor"),
        "{error}"
    );
    assert_eq!(
        omnimd_native::gpu_model::committee_mismatch(&mixed).unwrap(),
        "model 1 differs from the first in its descriptor"
    );
    assert!(omnimd_native::gpu_model::committee_mismatch(&members).is_none());
}

/// The se_e2_a test model's networks around an se_e3 descriptor.
fn three_body(ntypes: usize, sel: Vec<usize>, rcut: f64, seed: u64) -> Model {
    common::three_body_model(ntypes, sel, rcut, seed, 16)
}

/// se_e3 on the device against the CPU reference, on the kinds of frames the
/// other descriptors' checks use, and in chunks that end by their rows.
#[test]
fn se_e3_matches_cpu() {
    let triclinic = [[9.0, 0.0, 0.0], [1.5, 8.5, 0.0], [-1.0, 0.8, 9.2]];
    check(
        "three-body, triclinic",
        three_body(2, vec![80, 160], 4.5, 81),
        &frame(triclinic, 5, 2, true, 82),
    );
    // Narrower than the cutoff: neighbors include several images.
    let small = [[4.2, 0.0, 0.0], [0.0, 4.2, 0.0], [0.0, 0.0, 4.2]];
    check(
        "three-body, several images",
        three_body(2, vec![60, 120], 4.5, 83),
        &frame(small, 2, 2, true, 84),
    );
    let cubic = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]];
    check(
        "three-body, open",
        three_body(3, vec![30, 30, 30], 4.0, 85),
        &frame(cubic, 4, 3, false, 86),
    );
    // An embedding wider than a warp, and wider than a block of the kernels.
    let lattice = [[7.0, 0.0, 0.0], [0.5, 7.0, 0.0], [0.0, 0.0, 7.0]];
    check(
        "three-body, wide embedding",
        common::three_body_model(2, vec![40, 80], 4.0, 87, 40),
        &frame(lattice, 3, 2, true, 88),
    );
    check(
        "three-body, embedding wider than a block",
        common::three_body_model(2, vec![40, 80], 4.0, 89, 136),
        &frame(lattice, 3, 2, true, 90),
    );
    // A type without slots, and without atoms in the frame.
    check(
        "three-body, no slots for the last type",
        three_body(3, vec![30, 30, 0], 4.0, 91),
        &frame(cubic, 4, 2, false, 92),
    );

    // Chunks that end where their rows (pairs of neighbors) are too many,
    // down to one atom a chunk, give what one chunk gives.
    let Some(device) = device() else { return };
    let model = three_body(3, vec![30, 30, 30], 4.0, 93);
    let open = frame(cubic, 4, 3, false, 94);
    let expected = NativePotential::new(model.clone())
        .unwrap()
        .eval(&open)
        .unwrap();
    let mut gpu = GpuPotential::<f64>::new(&model, device.clone()).unwrap();
    assert!(gpu.chunk_pairs() > 1_000_000, "{}", gpu.chunk_pairs());
    for rows in [1, 40, 333, 5000] {
        gpu.set_chunk_pairs(rows);
        assert_eq!(gpu.chunk_pairs(), rows);
        compare(
            &format!("three-body, {rows} rows a chunk"),
            &gpu.eval(&open).unwrap(),
            &expected,
            FLOAT64,
        );
    }
    // Atoms without neighbors have no row; alone in a chunk they have no
    // work for the embedding nets at all. One pair, and one atom far away.
    let sparse = Frame {
        positions: vec![[0.0, 0.0, 0.0], [20.0, 1.0, 3.0], [1.3, 0.4, -0.2]],
        species: vec!["H".into(), "O".into(), "C".into()],
        cell: None,
        units: Units::EvAngstrom,
    };
    let expected = NativePotential::new(model.clone())
        .unwrap()
        .eval(&sparse)
        .unwrap();
    assert!(expected.forces[0].iter().any(|f| f.abs() > 1e-6));
    assert_eq!(expected.forces[1], [0.0; 3]);
    let mut gpu = GpuPotential::<f64>::new(&model, device).unwrap();
    compare(
        "three-body, an atom alone",
        &gpu.eval(&sparse).unwrap(),
        &expected,
        FLOAT64,
    );
    gpu.set_chunk_atoms(1);
    compare(
        "three-body, an atom alone in its chunk",
        &gpu.eval(&sparse).unwrap(),
        &expected,
        FLOAT64,
    );
}

/// se_e3 against DeePMD-kit 3.1.3 itself (`tools/make_se_e3_fixture.py`): the
/// model whose `sel` holds every neighbor, with and without a cell, and the
/// one with `sel` = [3, 4, 2] against some 27 neighbors, where the result
/// depends on which neighbors are kept.
#[test]
fn se_e3_matches_deepmd_reference() {
    let Some(device) = device() else { return };
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
    let reference: RadialReference = serde_json::from_str(
        &std::fs::read_to_string(root.join("se-e3-deepmd-reference.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(reference.cases.len(), 3);
    let mut worst = [0.0f64; 3];
    for (index, case) in reference.cases.into_iter().enumerate() {
        let model = NativePotential::load(root.join(&case.model)).unwrap();
        let expected = Evaluation {
            energy: case.energy,
            forces: case.forces,
            virial: case.virial,
            units: Units::EvAngstrom,
        };
        let label = format!("{} (case {index})", case.model);
        let mut gpu = GpuPotential::<f64>::new(model.model(), device.clone()).unwrap();
        assert_eq!(gpu.metadata().backend, "native-gpu-se-e3");
        let got = gpu.eval(&case.frame).unwrap();
        assert_eq!(
            gpu.last_truncated(),
            case.model == "se-e3-tight-model.native.json",
            "{label}"
        );
        compare(&label, &got, &expected, FLOAT64);
        worst[0] = worst[0].max((got.energy - expected.energy).abs());
        for (a, b) in got
            .forces
            .iter()
            .flatten()
            .zip(expected.forces.iter().flatten())
        {
            worst[1] = worst[1].max((a - b).abs());
        }
        for (a, b) in got
            .virial
            .iter()
            .flatten()
            .zip(expected.virial.iter().flatten())
        {
            worst[2] = worst[2].max((a - b).abs());
        }
        gpu.set_chunk_atoms(5);
        compare(
            &format!("{label}, chunks"),
            &gpu.eval(&case.frame).unwrap(),
            &expected,
            FLOAT64,
        );
        gpu.set_chunk_pairs(100);
        compare(
            &format!("{label}, chunks by rows"),
            &gpu.eval(&case.frame).unwrap(),
            &expected,
            FLOAT64,
        );
    }
    eprintln!(
        "se_e3 on the device against DeePMD-kit: energy within {:.1e} eV, forces {:.1e} eV/A, virial {:.1e} eV",
        worst[0], worst[1], worst[2]
    );
}

/// Float32, batches, committees and tables take an se_e3 model through the
/// same stages as the other descriptors: each against the CPU.
#[test]
fn se_e3_in_float32_in_batches_and_committees() {
    let Some(device) = device() else { return };
    let triclinic = [[9.0, 0.0, 0.0], [1.5, 8.5, 0.0], [-1.0, 0.8, 9.2]];
    let cubic = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]];
    let frames = [
        frame(triclinic, 5, 2, true, 101),
        frame(cubic, 4, 2, false, 102),
        frame(triclinic, 4, 2, true, 103),
    ];
    let models: Vec<Model> = (0..3)
        .map(|member| {
            let mut model = three_body(2, vec![80, 160], 4.5, 104 + 7 * member as u64);
            // Each member with widths of its own, and each type of each.
            for (t, normalization) in model.descriptor.normalization.iter_mut().enumerate() {
                let direction =
                    normalization.std[1] * (1.0 + 0.05 * member as f64 + 0.02 * t as f64);
                normalization.std = [1.0, direction, direction, direction];
            }
            for bias in &mut model.atom_energy_bias {
                *bias += 0.37 * member as f64;
            }
            model
        })
        .collect();
    let expected: Vec<Vec<Evaluation>> = frames
        .iter()
        .map(|frame| {
            models
                .iter()
                .map(|model| {
                    NativePotential::new(model.clone())
                        .unwrap()
                        .eval(frame)
                        .unwrap()
                })
                .collect()
        })
        .collect();
    let batch: Vec<&Frame> = frames.iter().collect();
    let members: Vec<&Model> = models.iter().collect();
    // The forces are not small: the comparisons below mean something.
    let largest = expected
        .iter()
        .flatten()
        .flat_map(|evaluation| evaluation.forces.iter().flatten())
        .fold(0.0f64, |m, f| m.max(f.abs()));
    eprintln!("se_e3 test models: largest force {largest:.3e} eV/A");
    assert!(largest > 1e-2, "{largest}");

    // Float64: a batch for one model, then for the committee.
    let mut one = GpuPotential::<f64>::new(&models[0], device.clone()).unwrap();
    for (index, got) in one.eval_batch(&batch).unwrap().iter().enumerate() {
        compare(
            &format!("three-body batch, frame {index}"),
            got,
            &expected[index][0],
            FLOAT64,
        );
    }
    let mut committee = GpuPotential::<f64>::committee(&members, device.clone()).unwrap();
    assert_eq!(committee.members(), 3);
    for chunked in [false, true] {
        if chunked {
            // The members share the rows of a chunk as they share its atoms.
            committee.set_chunk_atoms(50);
            committee.set_chunk_pairs(20_000);
        }
        for (index, of_frame) in committee.eval_members(&batch).unwrap().iter().enumerate() {
            for (member, got) in of_frame.iter().enumerate() {
                compare(
                    &format!(
                        "three-body committee (chunks: {chunked}), frame {index}, member {member}"
                    ),
                    got,
                    &expected[index][member],
                    FLOAT64,
                );
            }
        }
    }
    // Tables for the nets of every member, each over the range of θ that
    // the widths of its own central types give (the members differ in them,
    // and so do the types of one member).
    assert!(!committee.is_compressed());
    committee.compress(0.8, 0.01).unwrap();
    assert!(committee.is_compressed());
    for (index, of_frame) in committee.eval_members(&batch).unwrap().iter().enumerate() {
        for (member, got) in of_frame.iter().enumerate() {
            compare(
                &format!("three-body committee with tables, frame {index}, member {member}"),
                got,
                &expected[index][member],
                (1e-11, 1e-9),
            );
        }
    }
    assert_eq!(committee.table_fallbacks(), 0);
    // Float32 networks and sums: about single precision, nothing worse;
    // with tables as well.
    let mut single = GpuPotential::<f32>::committee(&members, device.clone()).unwrap();
    let mut worst = (0.0f64, 0.0f64);
    for tables in [false, true] {
        if tables {
            single.compress(0.8, 0.01).unwrap();
        }
        for (index, of_frame) in single.eval_members(&batch).unwrap().iter().enumerate() {
            for (member, got) in of_frame.iter().enumerate() {
                let reference = &expected[index][member];
                compare(
                    &format!(
                        "three-body float32 (tables: {tables}), frame {index}, member {member}"
                    ),
                    got,
                    reference,
                    (1e-6, 1e-4),
                );
                worst.0 = worst.0.max(
                    (got.energy - reference.energy).abs() / frames[index].positions.len() as f64,
                );
                for (a, b) in got
                    .forces
                    .iter()
                    .flatten()
                    .zip(reference.forces.iter().flatten())
                {
                    worst.1 = worst.1.max((a - b).abs());
                }
            }
        }
    }
    assert_eq!(single.table_fallbacks(), 0);
    eprintln!(
        "se_e3 in float32 against the CPU: energy within {:.1e} eV per atom, forces {:.1e} eV/A",
        worst.0, worst.1
    );
    // The forces are the gradient of the energy the device gives.
    let h = 1e-5;
    let base = &frames[0];
    let forces = one.eval(base).unwrap().forces;
    for atom in [0, 7, 19] {
        for axis in 0..3 {
            let mut shifted = |s: f64| {
                let mut f = base.clone();
                f.positions[atom][axis] += s;
                one.eval(&f).unwrap().energy
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
    // Models of two descriptors are not one committee.
    let other = radial(2, vec![80, 160], 4.5, 6);
    let mixed = [&models[0], &other];
    let error = GpuPotential::<f64>::committee(&mixed, device.clone())
        .err()
        .unwrap();
    assert!(
        error
            .to_string()
            .contains("differs from the first in its descriptor"),
        "{error}"
    );
    assert_eq!(
        omnimd_native::gpu_model::committee_mismatch(&mixed).unwrap(),
        "model 1 differs from the first in its descriptor"
    );
    assert!(omnimd_native::gpu_model::committee_mismatch(&members).is_none());
    // The GPU trainer takes the descriptor (its tests are in gpu_train.rs).
    // Among its parameters the net of a pair of neighbor types is there
    // once, whichever of the two types comes first.
    let trainer = omnimd_native::gpu_train::GpuTrainer::<f64>::new(&models[0], device).unwrap();
    let size = |network: &omnimd_native::Network| {
        let layer = |l: &omnimd_native::Layer| {
            l.weights.iter().map(Vec::len).sum::<usize>()
                + l.bias.len()
                + l.timestep.as_ref().map_or(0, Vec::len)
        };
        network.layers.iter().map(layer).sum::<usize>()
    };
    let d = &models[0].descriptor;
    let ntypes = d.sel.len();
    let nets: usize = (0..ntypes)
        .flat_map(|t| (t..ntypes).map(move |u| (t, u)))
        .map(|(t, u)| size(&d.embedding[t][u]))
        .sum();
    let fitting: usize = models[0].fitting.iter().map(size).sum();
    assert_eq!(trainer.parameter_count(), nets + fitting);
}

/// The largest difference of a force component.
fn force_difference(a: &Evaluation, b: &Evaluation) -> f64 {
    a.forces
        .iter()
        .flatten()
        .zip(b.forces.iter().flatten())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// The embedding nets of se_e3 as tables in pieces of uneven length: against
/// the nets (the CPU reference), in chunks and batches, for atoms without a
/// pair, in float32, up to the ends of the tables and beyond them, where a
/// frame goes back to the nets.
#[test]
fn se_e3_tables_follow_the_nets_and_give_way_outside_their_range() {
    let Some(device) = device() else { return };
    const TABLES: (f64, f64) = (1e-11, 1e-9);
    let triclinic = [[9.0, 0.0, 0.0], [1.5, 8.5, 0.0], [-1.0, 0.8, 9.2]];
    let cubic = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]];
    let model = three_body(2, vec![80, 160], 4.5, 301);
    let frames = [
        frame(triclinic, 5, 2, true, 302),
        frame(cubic, 4, 2, false, 303),
    ];
    let mut cpu = NativePotential::new(model.clone()).unwrap();
    let expected: Vec<Evaluation> = frames.iter().map(|f| cpu.eval(f).unwrap()).collect();

    let mut gpu = GpuPotential::<f64>::new(&model, device.clone()).unwrap();
    assert_eq!(gpu.table_size(), (0, 0));
    gpu.compress(0.8, 0.01).unwrap();
    assert!(gpu.is_compressed());
    // Three nets (two types of neighbors), of sixteen outputs.
    let (bytes, pieces) = gpu.table_size();
    assert!((300..100_000).contains(&pieces), "{pieces} pieces");
    assert!(bytes >= pieces * 16 * 6 * 8, "{bytes} bytes");
    let mut worst = 0.0f64;
    for (index, (frame, exact)) in frames.iter().zip(&expected).enumerate() {
        let got = gpu.eval(frame).unwrap();
        compare(
            &format!("three-body tables, frame {index}"),
            &got,
            exact,
            TABLES,
        );
        worst = worst.max(force_difference(&got, exact));
    }
    eprintln!(
        "se_e3 tables in float64 against the nets: forces within {worst:.1e} eV/A; {pieces} pieces, {bytes} bytes"
    );
    // Both frames in one pass.
    let batch: Vec<&Frame> = frames.iter().collect();
    for (index, got) in gpu.eval_batch(&batch).unwrap().iter().enumerate() {
        compare(
            &format!("three-body tables in a batch, frame {index}"),
            got,
            &expected[index],
            TABLES,
        );
    }
    // Chunks that end by their atoms or by their rows, down to an atom a
    // chunk (a row of a table takes a tenth of the memory of a stored row
    // here, and a chunk holds as many times more of them).
    for (atoms, rows) in [(7, 1_000_000), (1000, 1), (1000, 40), (13, 333)] {
        gpu.set_chunk_atoms(atoms);
        gpu.set_chunk_pairs(rows);
        compare(
            &format!("three-body tables, chunks of {atoms} atoms and {rows} rows"),
            &gpu.eval(&frames[0]).unwrap(),
            &expected[0],
            TABLES,
        );
    }
    assert_eq!(gpu.table_fallbacks(), 0);
    // That factor shows in the chunks. With so few rows a chunk that the
    // nets take the frame in six chunks or more, the tables take it in a
    // sixth of them at most.
    let mut nets = GpuPotential::<f64>::new(&model, device.clone()).unwrap();
    let chunks_of = |potential: &mut GpuPotential<f64>, rows: usize| {
        potential.set_chunk_atoms(1000);
        potential.set_chunk_pairs(rows);
        potential.set_profiling(true);
        potential.eval(&frames[0]).unwrap();
        let chunks = potential.stage_times().unwrap().chunks;
        potential.set_profiling(false);
        chunks
    };
    let mut rows = 1 << 22;
    while chunks_of(&mut nets, rows) < 6 {
        rows /= 2;
        assert!(rows > 16, "the frame has too few pairs for six chunks");
    }
    let (with_nets, with_tables) = (chunks_of(&mut nets, rows), chunks_of(&mut gpu, rows));
    assert!(
        with_nets >= 6 && with_tables * 6 <= with_nets,
        "{with_nets} chunks with nets, {with_tables} with tables"
    );

    // A coarse step is a loose tolerance: fewer pieces, visibly worse. So the
    // tables are what is evaluated.
    let mut coarse = GpuPotential::<f64>::new(&model, device.clone()).unwrap();
    coarse.compress(0.8, 0.5).unwrap();
    assert!(
        coarse.table_size().1 * 3 < pieces,
        "{:?}",
        coarse.table_size()
    );
    let rough = force_difference(&coarse.eval(&frames[0]).unwrap(), &expected[0]);
    assert!(rough > 1e-7, "coarse tables too close: {rough}");
    assert!(rough < 1.0, "coarse tables too far: {rough}");

    // Float32: the tables are as fine as single precision can use, fewer
    // pieces than in float64, and the result is within its rounding.
    let mut single = GpuPotential::<f32>::new(&model, device.clone()).unwrap();
    single.compress(0.8, 0.01).unwrap();
    assert!(single.table_size().1 < pieces);
    for (index, (frame, exact)) in frames.iter().zip(&expected).enumerate() {
        for atoms in [1000, 7] {
            single.set_chunk_atoms(atoms);
            compare(
                &format!("three-body tables in float32, frame {index}, chunks of {atoms}"),
                &single.eval(frame).unwrap(),
                exact,
                (1e-6, 1e-4),
            );
        }
    }
    assert_eq!(single.table_fallbacks(), 0);

    // Atoms without a pair have no row, with tables as without: a pair and an
    // atom far away, each atom in a chunk of its own, and an atom alone.
    let wide = three_body(3, vec![30, 30, 30], 4.0, 304);
    let mut wide_cpu = NativePotential::new(wide.clone()).unwrap();
    let mut wide_gpu = GpuPotential::<f64>::new(&wide, device.clone()).unwrap();
    wide_gpu.compress(0.8, 0.01).unwrap();
    let sparse = Frame {
        positions: vec![[0.0, 0.0, 0.0], [20.0, 1.0, 3.0], [1.3, 0.4, -0.2]],
        species: vec!["H".into(), "O".into(), "C".into()],
        cell: None,
        units: Units::EvAngstrom,
    };
    let alone = Frame {
        positions: vec![[0.3, -0.1, 0.2]],
        species: vec!["C".into()],
        cell: None,
        units: Units::EvAngstrom,
    };
    for (label, frame) in [("a pair and an atom", &sparse), ("an atom alone", &alone)] {
        let exact = wide_cpu.eval(frame).unwrap();
        for atoms in [1000, 1] {
            wide_gpu.set_chunk_atoms(atoms);
            compare(
                &format!("three-body tables, {label}, chunks of {atoms}"),
                &wide_gpu.eval(frame).unwrap(),
                &exact,
                TABLES,
            );
        }
    }
    assert_eq!(wide_gpu.table_fallbacks(), 0);

    // Embeddings wider than a warp, and than a block of the kernel that has
    // a warp for a row (float32: three rounds of its lanes for 136 outputs).
    // In float64 a thread has a row and sums of its own in shared memory,
    // for which 136 outputs leave one warp of threads; with 200 not one
    // warp fits, and float64 goes by warps as well.
    let lattice = [[7.0, 0.0, 0.0], [0.5, 7.0, 0.0], [0.0, 0.0, 7.0]];
    let small = frame(lattice, 3, 2, true, 305);
    for width in [40, 136, 200] {
        let model = common::three_body_model(2, vec![40, 80], 4.0, 306 + width as u64, width);
        let exact = NativePotential::new(model.clone())
            .unwrap()
            .eval(&small)
            .unwrap();
        let mut double = GpuPotential::<f64>::new(&model, device.clone()).unwrap();
        double.compress(0.8, 0.01).unwrap();
        compare(
            &format!("three-body tables, embedding of {width}"),
            &double.eval(&small).unwrap(),
            &exact,
            TABLES,
        );
        let mut single = GpuPotential::<f32>::new(&model, device.clone()).unwrap();
        single.compress(0.8, 0.01).unwrap();
        compare(
            &format!("three-body tables in float32, embedding of {width}"),
            &single.eval(&small).unwrap(),
            &exact,
            (1e-6, 1e-4),
        );
        assert_eq!(double.table_fallbacks() + single.table_fallbacks(), 0);
    }

    // The ends of the tables. Tables from 2 Å reach to a fifth of that: the
    // direction of a neighbor at 0.4 Å (inside the smoothing radius, where
    // the entry is 1/r) over the narrowest width of a central type gives the
    // largest θ, and a table ends at the power of two from there on. Two
    // atoms of the type with that width have one pair each, a neighbor with
    // itself, whose θ is the square of its direction.
    let width = model
        .descriptor
        .normalization
        .iter()
        .map(|n| n.std[1])
        .fold(f64::INFINITY, f64::min);
    assert_eq!(width, model.descriptor.normalization[0].std[1]);
    let end = 2f64.powf((1.0 / (0.4 * width)).powi(2).log2().ceil());
    // The distance at which θ is the end of the table.
    let edge = 1.0 / (width * end.sqrt());
    assert!(edge > 0.2 && edge < 0.4, "{edge}");
    let pair = |r: f64| Frame {
        positions: vec![[0.0, 0.0, 0.0], [r, 0.0, 0.0]],
        species: vec!["O".into(), "O".into()],
        cell: None,
        units: Units::EvAngstrom,
    };
    let mut short = GpuPotential::<f64>::new(&model, device.clone()).unwrap();
    short.compress(2.0, 0.01).unwrap();
    for (factor, fallbacks) in [(1.3, 0), (1.001, 0), (0.999, 1), (0.8, 2)] {
        let frame = pair(edge * factor);
        let exact = cpu.eval(&frame).unwrap();
        assert!(exact.forces[0][0].abs() > 1e-3, "{:?}", exact.forces);
        compare(
            &format!("three-body tables, a pair at {factor} of the end"),
            &short.eval(&frame).unwrap(),
            &exact,
            TABLES,
        );
        assert_eq!(short.table_fallbacks(), fallbacks, "at {factor} of the end");
    }
    // Every member of a committee has the range of its own widths. A second
    // member whose directions are divided by half as much has four times
    // the θ: for a pair just inside the end of the first member's tables
    // that is far outside them and inside its own, so nothing goes back to
    // the nets, and each member is its model.
    let mut narrow = model.clone();
    for normalization in &mut narrow.descriptor.normalization {
        let direction = normalization.std[1] / 2.0;
        normalization.std = [1.0, direction, direction, direction];
    }
    let mut both = GpuPotential::<f64>::committee(&[&model, &narrow], device.clone()).unwrap();
    both.compress(2.0, 0.01).unwrap();
    let inside = pair(edge * 1.001);
    let of_members = both.eval_members(&[&inside]).unwrap().remove(0);
    for (member, (got, of)) in of_members.iter().zip([&model, &narrow]).enumerate() {
        let exact = NativePotential::new(of.clone())
            .unwrap()
            .eval(&inside)
            .unwrap();
        compare(
            &format!("three-body tables, member {member} at the end of the first member's"),
            got,
            &exact,
            TABLES,
        );
    }
    assert_eq!(both.table_fallbacks(), 0);
    // The frames above are within these tables as well, at the tolerance of
    // any (there is no coarser part).
    compare(
        "three-body tables from 2 A",
        &short.eval(&frames[0]).unwrap(),
        &expected[0],
        TABLES,
    );
    assert_eq!(short.table_fallbacks(), 2);
    // One frame of a batch outside the tables: that frame alone goes back to
    // the nets, in chunks as well.
    let mut close = frames[0].clone();
    close.positions[1] = [
        close.positions[0][0] + 0.25,
        close.positions[0][1],
        close.positions[0][2],
    ];
    let exact_close = cpu.eval(&close).unwrap();
    for atoms in [1000, 9] {
        short.set_chunk_atoms(atoms);
        let before = short.table_fallbacks();
        let got = short.eval_batch(&[&frames[0], &close, &frames[1]]).unwrap();
        compare("a batch, the frame before", &got[0], &expected[0], TABLES);
        compare("a batch, the frame outside", &got[1], &exact_close, FLOAT64);
        compare("a batch, the frame after", &got[2], &expected[1], TABLES);
        assert_eq!(short.table_fallbacks(), before + 1);
    }

    // What is refused, and leaves the potential as it was.
    for (min_distance, step) in [(4.5, 0.01), (0.0, 0.01), (0.8, 0.0), (0.8, 2.0)] {
        let error = short.compress(min_distance, step).unwrap_err().to_string();
        assert!(
            error.contains("min_distance < rcut") || error.contains("table step"),
            "{error}"
        );
    }
    assert!(short.is_compressed());
    // A net that turns within a thousandth of a unit at θ = 200, where a
    // float32 has steps of 1.5e-5: too few for a piece. The potential says
    // which net and where, and goes on with its nets.
    let mut sharp = model.clone();
    let first = &mut sharp.descriptor.embedding[0][0].layers[0];
    first.weights[0][0] = 2000.0;
    first.bias[0] = -2000.0 * 200.0;
    let mut single = GpuPotential::<f32>::new(&sharp, device).unwrap();
    let error = single.compress(0.8, 0.01).unwrap_err().to_string();
    assert!(
        error.contains("se_e3 for neighbors of types 0 and 0") && error.contains("turns too fast"),
        "{error}"
    );
    let place: f64 = error
        .split("near ")
        .nth(1)
        .and_then(|rest| rest.split(' ').next())
        .and_then(|number| number.parse().ok())
        .unwrap_or_else(|| panic!("no place in {error:?}"));
    assert!((place - 200.0).abs() < 0.1, "{error}");
    assert!(!single.is_compressed());
    assert_eq!(single.table_size(), (0, 0));
    let exact = NativePotential::new(sharp)
        .unwrap()
        .eval(&frames[0])
        .unwrap();
    compare(
        "three-body, a net too fast for float32 tables",
        &single.eval(&frames[0]).unwrap(),
        &exact,
        (1e-6, 1e-4),
    );
}

/// A descriptor for a hybrid, as a model of its own, with a smoothing radius
/// and widths that the others of the list do not have.
fn tuned(mut model: Model, rcut_smth: f64, factor: f64) -> Model {
    model.descriptor.rcut_smth = rcut_smth;
    for normalization in &mut model.descriptor.normalization {
        for width in &mut normalization.std {
            // A width of one stands for an entry the descriptor does not read.
            if *width != 1.0 {
                *width *= factor;
            }
        }
    }
    model
}

/// An se_e2_a, an se_e2_r and an se_e3 descriptor of two types. The second
/// has the cutoff and the slots of the neighbor list and reads it as it is;
/// the others have their own tables of it.
fn three_kinds(seed: u64) -> Model {
    let mut model = common::hybrid_model(
        &[
            tuned(
                common::model_of_width(2, vec![60, 90], 3.8, seed, 16, 4),
                0.9,
                1.1,
            ),
            tuned(
                common::radial_model(2, vec![80, 160], 4.5, seed + 1, 24),
                1.4,
                0.9,
            ),
            tuned(
                common::three_body_model(2, vec![40, 60], 3.2, seed + 2, 8),
                0.3,
                1.2,
            ),
        ],
        seed + 3,
    );
    // Larger output weights, for forces of some hundredths of an eV/A.
    for net in &mut model.fitting {
        let output = net.layers.last_mut().unwrap();
        for weight in output.weights.iter_mut().flatten() {
            *weight *= 4.0;
        }
    }
    model
}

/// A hybrid on the device against the CPU reference, on the kinds of frames
/// the descriptors' own checks use.
#[test]
fn hybrid_matches_cpu() {
    let triclinic = [[9.0, 0.0, 0.0], [1.5, 8.5, 0.0], [-1.0, 0.8, 9.2]];
    let model = three_kinds(201);
    assert_eq!(model.descriptor.sel, [80, 160]);
    assert_eq!(
        (model.descriptor.rcut, model.descriptor.rcut_smth),
        (4.5, 0.3)
    );
    assert_eq!(model.descriptor.features(), 16 * 4 + 24 + 8);
    check(
        "hybrid, three kinds",
        model,
        &frame(triclinic, 5, 2, true, 202),
    );
    // No descriptor reads the list as it is: the first has its cutoff and
    // fewer slots of the second type, the second all slots of that type and
    // a shorter cutoff.
    let model = common::hybrid_model(
        &[
            model_of(2, vec![80, 100], 4.5, 203),
            tuned(
                common::radial_model(2, vec![50, 160], 3.5, 204, 16),
                0.7,
                1.3,
            ),
        ],
        205,
    );
    assert_eq!(model.descriptor.sel, [80, 160]);
    check(
        "hybrid, two tables",
        model,
        &frame(triclinic, 5, 2, true, 206),
    );
    // The slots of the list and a shorter cutoff: the second descriptor
    // still has a table of its own, without the neighbors beyond 3.5.
    let model = common::hybrid_model(
        &[
            model_of(2, vec![80, 160], 4.5, 261),
            tuned(
                common::radial_model(2, vec![80, 160], 3.5, 262, 16),
                0.7,
                1.3,
            ),
        ],
        263,
    );
    assert_eq!(model.descriptor.list[1].sel, model.descriptor.sel);
    check(
        "hybrid, a shorter cutoff only",
        model,
        &frame(triclinic, 5, 2, true, 264),
    );
    // Narrower than the cutoff: neighbors include several images.
    let small = [[4.2, 0.0, 0.0], [0.0, 4.2, 0.0], [0.0, 0.0, 4.2]];
    check(
        "hybrid, several images",
        common::hybrid_model(
            &[
                three_body(2, vec![60, 120], 4.5, 207),
                tuned(model_of(2, vec![60, 120], 4.0, 208), 1.1, 0.8),
                tuned(radial(2, vec![30, 60], 3.0, 209), 0.2, 1.1),
            ],
            210,
        ),
        &frame(small, 2, 2, true, 211),
    );
    let cubic = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]];
    let open = |seed: u64| {
        common::hybrid_model(
            &[
                model_of(3, vec![30, 30, 30], 4.0, seed),
                tuned(three_body(3, vec![20, 20, 20], 3.0, seed + 1), 1.0, 0.9),
                tuned(radial(3, vec![30, 30, 30], 3.5, seed + 2), 0.2, 1.2),
            ],
            seed + 3,
        )
    };
    check("hybrid, open", open(212), &frame(cubic, 4, 3, false, 216));
    // Embeddings wider than a warp, and one wider than a block of the kernels.
    let lattice = [[7.0, 0.0, 0.0], [0.5, 7.0, 0.0], [0.0, 0.0, 7.0]];
    check(
        "hybrid, wide embeddings",
        common::hybrid_model(
            &[
                common::model_of_width(2, vec![40, 80], 4.0, 217, 40, 6),
                common::radial_model(2, vec![40, 80], 4.0, 218, 136),
                common::three_body_model(2, vec![40, 80], 3.5, 219, 40),
            ],
            220,
        ),
        &frame(lattice, 3, 2, true, 221),
    );
    // A type without slots in any descriptor, and without atoms in the frame.
    check(
        "hybrid, no slots for the last type",
        common::hybrid_model(
            &[
                model_of(3, vec![30, 30, 0], 4.0, 222),
                three_body(3, vec![20, 20, 0], 3.0, 223),
            ],
            224,
        ),
        &frame(cubic, 4, 2, false, 225),
    );
    // Two descriptors of one kind, and descriptors that are in the list twice.
    let first = model_of(3, vec![30, 30, 30], 4.0, 226);
    let second = tuned(radial(3, vec![30, 30, 30], 3.5, 227), 0.8, 1.1);
    check(
        "hybrid, descriptors of one kind",
        common::hybrid_model(
            &[
                first.clone(),
                tuned(model_of(3, vec![20, 20, 20], 3.0, 228), 1.2, 0.7),
                second.clone(),
                first,
                second,
            ],
            229,
        ),
        &frame(cubic, 4, 3, false, 230),
    );

    let Some(device) = device() else { return };
    // The order of the list is the order of the components: the same
    // descriptors the other way round are another model, on the device as
    // on the CPU.
    let forward = open(231);
    let mut backward = forward.clone();
    backward.descriptor.list.reverse();
    let cluster = frame(cubic, 4, 3, false, 232);
    let mut energies = Vec::new();
    for (label, model) in [("forward", &forward), ("backward", &backward)] {
        let expected = NativePotential::new(model.clone())
            .unwrap()
            .eval(&cluster)
            .unwrap();
        let mut gpu = GpuPotential::<f64>::new(model, device.clone()).unwrap();
        assert_eq!(gpu.metadata().backend, "native-gpu-hybrid");
        let got = gpu.eval(&cluster).unwrap();
        compare(&format!("hybrid, {label}"), &got, &expected, FLOAT64);
        energies.push(got.energy);
    }
    assert!((energies[0] - energies[1]).abs() > 1e-3, "{energies:?}");

    // Two se_e3 descriptors: chunks end where the rows of either are too
    // many, down to one atom a chunk, and give what one chunk gives.
    let model = common::hybrid_model(
        &[
            three_body(3, vec![30, 30, 30], 4.0, 233),
            tuned(radial(3, vec![30, 30, 30], 3.5, 234), 0.6, 1.1),
            tuned(
                common::three_body_model(3, vec![20, 20, 20], 3.0, 235, 8),
                1.1,
                0.8,
            ),
        ],
        236,
    );
    let expected = NativePotential::new(model.clone())
        .unwrap()
        .eval(&cluster)
        .unwrap();
    let mut gpu = GpuPotential::<f64>::new(&model, device.clone()).unwrap();
    assert!(gpu.chunk_pairs() > 1_000_000, "{}", gpu.chunk_pairs());
    for rows in [1, 40, 333, 5000] {
        gpu.set_chunk_pairs(rows);
        compare(
            &format!("hybrid, {rows} rows a chunk"),
            &gpu.eval(&cluster).unwrap(),
            &expected,
            FLOAT64,
        );
    }
    gpu.set_chunk_atoms(3);
    compare(
        "hybrid, three atoms a chunk",
        &gpu.eval(&cluster).unwrap(),
        &expected,
        FLOAT64,
    );
    // Atoms without neighbors: one pair, and one atom far away, which has
    // no row in any descriptor.
    let sparse = Frame {
        positions: vec![[0.0, 0.0, 0.0], [20.0, 1.0, 3.0], [1.3, 0.4, -0.2]],
        species: vec!["H".into(), "O".into(), "C".into()],
        cell: None,
        units: Units::EvAngstrom,
    };
    let expected = NativePotential::new(model.clone())
        .unwrap()
        .eval(&sparse)
        .unwrap();
    assert!(expected.forces[0].iter().any(|f| f.abs() > 1e-6));
    assert_eq!(expected.forces[1], [0.0; 3]);
    let mut gpu = GpuPotential::<f64>::new(&model, device.clone()).unwrap();
    compare(
        "hybrid, an atom alone",
        &gpu.eval(&sparse).unwrap(),
        &expected,
        FLOAT64,
    );
    gpu.set_chunk_atoms(1);
    compare(
        "hybrid, an atom alone in its chunk",
        &gpu.eval(&sparse).unwrap(),
        &expected,
        FLOAT64,
    );

    // One potential serves frames of different sizes in turn: the tables of
    // its descriptors grow with the frame.
    let model = open(265);
    let mut cpu = NativePotential::new(model.clone()).unwrap();
    let mut gpu = GpuPotential::<f64>::new(&model, device.clone()).unwrap();
    let sizes = [
        frame(cubic, 2, 3, false, 266),
        frame(cubic, 4, 3, false, 267),
        frame(cubic, 3, 3, false, 268),
        frame(cubic, 4, 3, false, 269),
    ];
    for (index, f) in sizes.iter().enumerate() {
        compare(
            &format!("hybrid, frame {index} of {} atoms", f.positions.len()),
            &gpu.eval(f).unwrap(),
            &cpu.eval(f).unwrap(),
            FLOAT64,
        );
    }

    // One descriptor in a list is that descriptor: the same launches with
    // the same numbers. (Not the same bits: the sums of the forces and of
    // the virial are added to in whatever order the blocks come.)
    for (label, plain) in [
        ("se_e2_a", model_of(3, vec![30, 30, 30], 4.0, 237)),
        ("se_e2_r", radial(3, vec![30, 30, 30], 4.0, 238)),
        ("se_e3", three_body(3, vec![30, 30, 30], 4.0, 239)),
    ] {
        let mut listed = common::hybrid_model(std::slice::from_ref(&plain), 240);
        listed.fitting = plain.fitting.clone();
        let alone = GpuPotential::<f64>::new(&plain, device.clone())
            .unwrap()
            .eval(&cluster)
            .unwrap();
        let in_list = GpuPotential::<f64>::new(&listed, device.clone())
            .unwrap()
            .eval(&cluster)
            .unwrap();
        assert_eq!(alone.energy, in_list.energy, "{label}");
        compare(
            &format!("hybrid of one {label}"),
            &in_list,
            &alone,
            (1e-15, 1e-12),
        );
    }
}

/// The se_e2_a test model under the name the other builders have.
fn model_of(ntypes: usize, sel: Vec<usize>, rcut: f64, seed: u64) -> Model {
    model(ntypes, sel, rcut, seed)
}

/// A hybrid against DeePMD-kit 3.1.3 itself (`tools/make_hybrid_fixture.py`):
/// the model whose descriptors hold every neighbor within their cutoffs,
/// with and without a cell, and the one whose slots are far below the
/// neighbor counts in the list and in every descriptor. There the result
/// depends on which neighbors are kept, and a descriptor of a hybrid does
/// not keep those it would keep alone: for 16, 22 and 18 of the 27 atoms the
/// fixture's three descriptors have other neighbors than their own
/// selection would give them.
#[test]
fn hybrid_matches_deepmd_reference() {
    let Some(device) = device() else { return };
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
    let text = std::fs::read_to_string(root.join("hybrid-deepmd-reference.json")).unwrap();
    let reference: RadialReference = serde_json::from_str(&text).unwrap();
    assert_eq!(reference.cases.len(), 3);
    // What the script found: the list of the tight model is cut for every
    // atom, and each descriptor differs from its own selection for some.
    let notes: serde_json::Value = serde_json::from_str(&text).unwrap();
    let tight = &notes["models"]["hybrid-tight-model.native.json"];
    assert_eq!(tight["atoms_whose_list_is_cut"], 27);
    for count in tight["atoms_where_a_part_differs_from_its_own_selection"]
        .as_array()
        .unwrap()
    {
        assert!(count.as_u64().unwrap() > 0, "{count}");
    }
    let mut worst = [0.0f64; 3];
    for (index, case) in reference.cases.into_iter().enumerate() {
        let model = NativePotential::load(root.join(&case.model)).unwrap();
        let expected = Evaluation {
            energy: case.energy,
            forces: case.forces,
            virial: case.virial,
            units: Units::EvAngstrom,
        };
        let label = format!("{} (case {index})", case.model);
        let mut gpu = GpuPotential::<f64>::new(model.model(), device.clone()).unwrap();
        assert_eq!(gpu.metadata().backend, "native-gpu-hybrid");
        let got = gpu.eval(&case.frame).unwrap();
        assert_eq!(
            gpu.last_truncated(),
            case.model == "hybrid-tight-model.native.json",
            "{label}"
        );
        compare(&label, &got, &expected, FLOAT64);
        worst[0] = worst[0].max((got.energy - expected.energy).abs());
        for (a, b) in got
            .forces
            .iter()
            .flatten()
            .zip(expected.forces.iter().flatten())
        {
            worst[1] = worst[1].max((a - b).abs());
        }
        for (a, b) in got
            .virial
            .iter()
            .flatten()
            .zip(expected.virial.iter().flatten())
        {
            worst[2] = worst[2].max((a - b).abs());
        }
        gpu.set_chunk_atoms(5);
        compare(
            &format!("{label}, chunks"),
            &gpu.eval(&case.frame).unwrap(),
            &expected,
            FLOAT64,
        );
        gpu.set_chunk_pairs(30);
        compare(
            &format!("{label}, chunks by rows"),
            &gpu.eval(&case.frame).unwrap(),
            &expected,
            FLOAT64,
        );
        // In float32 too, within what single precision gives.
        let mut single = GpuPotential::<f32>::new(model.model(), device.clone()).unwrap();
        compare(
            &format!("{label}, float32"),
            &single.eval(&case.frame).unwrap(),
            &expected,
            (1e-6, 1e-4),
        );
    }
    eprintln!(
        "hybrid on the device against DeePMD-kit: energy within {:.1e} eV, forces {:.1e} eV/A, virial {:.1e} eV",
        worst[0], worst[1], worst[2]
    );
}

/// The neighbor search reports a list that had to drop neighbors. A
/// descriptor of a hybrid can drop neighbors of a list that holds them all:
/// that is reported the same way.
#[test]
fn a_hybrid_reports_the_neighbors_a_descriptor_drops() {
    let Some(device) = device() else { return };
    let cubic = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]];
    let cluster = frame(cubic, 4, 3, false, 241);
    let hybrid = |sel: Vec<usize>, rcut: f64| {
        common::hybrid_model(
            &[
                model_of(3, vec![30, 30, 30], 4.0, 242),
                radial(3, sel, rcut, 243),
            ],
            244,
        )
    };
    let eval = |model: &Model| {
        let mut gpu = GpuPotential::<f64>::new(model, device.clone()).unwrap();
        let result = gpu.eval(&cluster).unwrap();
        (result, gpu.last_truncated())
    };
    // Room for all: nothing is dropped, and the CPU agrees.
    let roomy = hybrid(vec![30, 30, 30], 3.5);
    let (result, dropped) = eval(&roomy);
    assert!(!dropped);
    let expected = NativePotential::new(roomy).unwrap().eval(&cluster).unwrap();
    compare("hybrid, room for all", &result, &expected, FLOAT64);
    // Two slots for the second type, where atoms have more such neighbors
    // within 3.5: the descriptor keeps the nearest two, the list holds all.
    let (result, dropped) = eval(&hybrid(vec![30, 2, 30], 3.5));
    assert!(dropped);
    assert!(result.energy.is_finite());
    // No slot at all for a type that has neighbors.
    assert!(eval(&hybrid(vec![30, 30, 0], 3.5)).1);
    // The same few slots, and a cutoff so short that they are enough: it is
    // the neighbors within the descriptor's own cutoff that count.
    let near = hybrid(vec![30, 2, 30], 1.6);
    let (result, dropped) = eval(&near);
    assert!(!dropped);
    let expected = NativePotential::new(near).unwrap().eval(&cluster).unwrap();
    compare("hybrid, a short cutoff", &result, &expected, FLOAT64);
    // One neighbor more than slots is enough: an O with two H within the
    // cutoff of a descriptor that has one slot for them, and no other atom
    // with two neighbors of a type.
    let water = Frame {
        positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.1, 0.0]],
        species: vec!["O".into(), "H".into(), "H".into()],
        cell: None,
        units: Units::EvAngstrom,
    };
    let drops = |sel: Vec<usize>| {
        let model = hybrid(sel, 1.3);
        let mut gpu = GpuPotential::<f64>::new(&model, device.clone()).unwrap();
        gpu.eval(&water).unwrap();
        gpu.last_truncated()
    };
    assert!(drops(vec![1, 1, 1]));
    assert!(!drops(vec![1, 2, 1]));
    // Atoms in one place are an error also where a descriptor drops
    // neighbors: the two reports are kept apart. In a periodic box every
    // atom has neighbors of the second type within 3.5, and this descriptor
    // no slot for them, so both atoms of the pair carry both reports.
    let bulk = frame(cubic, 4, 3, true, 245);
    let none = hybrid(vec![30, 0, 30], 3.5);
    let mut gpu = GpuPotential::<f64>::new(&none, device.clone()).unwrap();
    gpu.eval(&bulk).unwrap();
    assert!(gpu.last_truncated());
    let mut touching = bulk.clone();
    touching.positions[22] = touching.positions[21];
    let error = gpu.eval(&touching).unwrap_err().to_string();
    assert!(error.contains("coincident"), "{error}");
    // The CPU reference refuses what the device reports.
    let error = NativePotential::new(hybrid(vec![30, 2, 30], 3.5))
        .unwrap()
        .eval(&cluster)
        .unwrap_err()
        .to_string();
    assert!(error.contains("of a descriptor of the hybrid"), "{error}");
}

/// Float32, batches, committees and embedding tables take a hybrid through
/// the same stages as a model of one descriptor: each against the CPU.
#[test]
fn hybrid_in_float32_compressed_in_batches_and_committees() {
    let Some(device) = device() else { return };
    let triclinic = [[9.0, 0.0, 0.0], [1.5, 8.5, 0.0], [-1.0, 0.8, 9.2]];
    let cubic = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]];
    let frames = [
        frame(triclinic, 5, 2, true, 251),
        frame(cubic, 4, 2, false, 252),
        frame(triclinic, 4, 2, true, 253),
    ];
    let models: Vec<Model> = (0..3)
        .map(|member| {
            let mut model = three_kinds(254 + 11 * member as u64);
            // Each member with statistics of its own in every descriptor.
            for part in &mut model.descriptor.list {
                for (t, normalization) in part.normalization.iter_mut().enumerate() {
                    for width in &mut normalization.std {
                        if *width != 1.0 {
                            *width *= 1.0 + 0.05 * member as f64 + 0.02 * t as f64;
                        }
                    }
                }
            }
            for bias in &mut model.atom_energy_bias {
                *bias += 0.37 * member as f64;
            }
            model
        })
        .collect();
    let expected: Vec<Vec<Evaluation>> = frames
        .iter()
        .map(|frame| {
            models
                .iter()
                .map(|model| {
                    NativePotential::new(model.clone())
                        .unwrap()
                        .eval(frame)
                        .unwrap()
                })
                .collect()
        })
        .collect();
    let batch: Vec<&Frame> = frames.iter().collect();
    let members: Vec<&Model> = models.iter().collect();
    // The forces are not small: the comparisons below mean something.
    let largest = expected
        .iter()
        .flatten()
        .flat_map(|evaluation| evaluation.forces.iter().flatten())
        .fold(0.0f64, |m, f| m.max(f.abs()));
    eprintln!("hybrid test models: largest force {largest:.3e} eV/A");
    assert!(largest > 1e-2, "{largest}");

    // Float64: a batch for one model, then for the committee.
    let mut one = GpuPotential::<f64>::new(&models[0], device.clone()).unwrap();
    for (index, got) in one.eval_batch(&batch).unwrap().iter().enumerate() {
        compare(
            &format!("hybrid batch, frame {index}"),
            got,
            &expected[index][0],
            FLOAT64,
        );
    }
    let mut committee = GpuPotential::<f64>::committee(&members, device.clone()).unwrap();
    assert_eq!(committee.members(), 3);
    for chunked in [false, true] {
        if chunked {
            committee.set_chunk_atoms(50);
            committee.set_chunk_pairs(20_000);
        }
        for (index, of_frame) in committee.eval_members(&batch).unwrap().iter().enumerate() {
            for (member, got) in of_frame.iter().enumerate() {
                compare(
                    &format!(
                        "hybrid committee (chunks: {chunked}), frame {index}, member {member}"
                    ),
                    got,
                    &expected[index][member],
                    FLOAT64,
                );
            }
        }
    }
    // Tables for every descriptor, each over its own range: of one step for
    // se_e2_a and se_e2_r, in pieces of uneven length for se_e3. Quintic
    // interpolation at step 0.01 is good to ~1e-12 for these nets; allow 1e-9.
    assert!(!committee.is_compressed());
    committee.compress(0.8, 0.01).unwrap();
    assert!(committee.is_compressed());
    for (index, of_frame) in committee.eval_members(&batch).unwrap().iter().enumerate() {
        for (member, got) in of_frame.iter().enumerate() {
            compare(
                &format!("hybrid committee with tables, frame {index}, member {member}"),
                got,
                &expected[index][member],
                (1e-11, 1e-9),
            );
        }
    }
    assert_eq!(committee.table_fallbacks(), 0);
    // A coarse table is visibly worse, so the tables are really in use, and
    // for each of the three descriptors: with the weights of the two others
    // set to zero in the fitting nets it still shows.
    let silenced = |parts: [usize; 2]| {
        let mut model = models[0].clone();
        for part in parts {
            let list = &model.descriptor.list;
            let first: usize = list[..part].iter().map(|d| d.features()).sum();
            let count = list[part].features();
            for net in &mut model.fitting {
                for row in &mut net.layers[0].weights[first..first + count] {
                    row.fill(0.0);
                }
            }
        }
        model
    };
    for (label, model) in [
        ("se_e2_a alone", silenced([1, 2])),
        ("se_e2_r alone", silenced([0, 2])),
        ("se_e3 alone", silenced([0, 1])),
    ] {
        let exact = NativePotential::new(model.clone())
            .unwrap()
            .eval(&frames[0])
            .unwrap();
        let mut coarse = GpuPotential::<f64>::new(&model, device.clone()).unwrap();
        coarse.compress(0.8, 0.5).unwrap();
        let rough = coarse.eval(&frames[0]).unwrap();
        let difference = rough
            .forces
            .iter()
            .flatten()
            .zip(exact.forces.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            difference > 1e-9,
            "{label}: coarse table too accurate: {difference}"
        );
        // And with fine tables it is the model again.
        let mut fine = GpuPotential::<f64>::new(&model, device.clone()).unwrap();
        fine.compress(0.8, 0.01).unwrap();
        compare(
            label,
            &fine.eval(&frames[0]).unwrap(),
            &exact,
            (1e-11, 1e-9),
        );
    }
    // A pair closer than the tables reach: that frame is evaluated with the
    // exact networks, same answer as without tables.
    let mut close = frames[0].clone();
    close.positions[1] = [
        close.positions[0][0] + 0.3,
        close.positions[0][1],
        close.positions[0][2],
    ];
    let mut short = GpuPotential::<f64>::new(&models[0], device.clone()).unwrap();
    short.compress(2.0, 0.01).unwrap();
    let expected_close = NativePotential::new(models[0].clone())
        .unwrap()
        .eval(&close)
        .unwrap();
    compare(
        "hybrid fallback",
        &short.eval(&close).unwrap(),
        &expected_close,
        FLOAT64,
    );
    assert_eq!(short.table_fallbacks(), 1);
    // The tables reach from a distance inside the shortest cutoff of the
    // descriptors (the 3.2 of the se_e3 descriptor here).
    assert!(short.compress(3.1, 0.01).is_ok());
    let error = short.compress(3.3, 0.01).unwrap_err().to_string();
    assert!(
        error.contains("compression needs 0 < min_distance < rcut"),
        "{error}"
    );
    // Nothing but se_e3 descriptors, of two cutoffs and embedding widths:
    // each has its tables, and the rows of a chunk are those of both.
    let pairs_only = common::hybrid_model(
        &[
            three_body(2, vec![80, 160], 4.5, 291),
            common::three_body_model(2, vec![40, 60], 3.2, 292, 8),
        ],
        293,
    );
    let exact = NativePotential::new(pairs_only.clone())
        .unwrap()
        .eval(&frames[0])
        .unwrap();
    let mut gpu = GpuPotential::<f64>::new(&pairs_only, device.clone()).unwrap();
    gpu.compress(0.8, 0.01).unwrap();
    assert!(gpu.is_compressed());
    for rows in [1_000_000, 50] {
        gpu.set_chunk_pairs(rows);
        compare(
            &format!("two se_e3 descriptors with tables, {rows} rows a chunk"),
            &gpu.eval(&frames[0]).unwrap(),
            &exact,
            (1e-11, 1e-9),
        );
    }
    assert_eq!(gpu.table_fallbacks(), 0);

    // Float32 networks and sums: about single precision, nothing worse;
    // with tables as well.
    let mut single = GpuPotential::<f32>::committee(&members, device.clone()).unwrap();
    let mut worst = (0.0f64, 0.0f64);
    for tables in [false, true] {
        if tables {
            single.compress(0.8, 0.01).unwrap();
        }
        for (index, of_frame) in single.eval_members(&batch).unwrap().iter().enumerate() {
            for (member, got) in of_frame.iter().enumerate() {
                let reference = &expected[index][member];
                compare(
                    &format!("hybrid float32 (tables: {tables}), frame {index}, member {member}"),
                    got,
                    reference,
                    (1e-6, 1e-4),
                );
                worst.0 = worst.0.max(
                    (got.energy - reference.energy).abs() / frames[index].positions.len() as f64,
                );
                for (a, b) in got
                    .forces
                    .iter()
                    .flatten()
                    .zip(reference.forces.iter().flatten())
                {
                    worst.1 = worst.1.max((a - b).abs());
                }
            }
        }
    }
    eprintln!(
        "hybrid in float32 against the CPU: energy within {:.1e} eV per atom, forces {:.1e} eV/A",
        worst.0, worst.1
    );
    // The forces are the gradient of the energy the device gives.
    let h = 1e-5;
    let base = &frames[0];
    let forces = one.eval(base).unwrap().forces;
    for atom in [0, 7, 19] {
        for axis in 0..3 {
            let mut shifted = |s: f64| {
                let mut f = base.clone();
                f.positions[atom][axis] += s;
                one.eval(&f).unwrap().energy
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

    // What makes hybrids one committee: the same descriptors in the same
    // order, laid out alike.
    let mismatch = |change: &dyn Fn(&mut Model)| {
        let mut other = models[1].clone();
        change(&mut other);
        assert!(
            NativePotential::new(other.clone()).is_ok(),
            "the changed model must be a model"
        );
        let pair = [&models[0], &other];
        let error = GpuPotential::<f64>::committee(&pair, device.clone())
            .err()
            .map(|e| e.to_string());
        (omnimd_native::gpu_model::committee_mismatch(&pair), error)
    };
    assert_eq!(mismatch(&|_| ()), (None, None));
    // The list in another order (the fitting nets take the same number of
    // components either way).
    let (reason, error) = mismatch(&|m| m.descriptor.list.swap(0, 2));
    assert_eq!(
        reason.unwrap(),
        "model 1 differs from the first in its descriptor"
    );
    assert!(error.unwrap().contains("differs from the first"));
    // Another smoothing radius in one descriptor.
    let (reason, error) = mismatch(&|m| {
        m.descriptor.list[1].rcut_smth = 1.5;
    });
    assert_eq!(
        reason.unwrap(),
        "model 1 differs from the first in its cutoffs"
    );
    assert!(error.unwrap().contains("differs from the first"));
    // Fewer slots in the descriptor that does not set the list's.
    let (reason, error) = mismatch(&|m| m.descriptor.list[2].sel[0] -= 1);
    assert_eq!(reason.unwrap(), "model 1 differs from the first in its sel");
    assert!(error.unwrap().contains("differs from the first"));
    // A descriptor less (and fitting nets for the components that are left).
    let parts: Vec<Model> = models[1].descriptor.list[..2]
        .iter()
        .map(|part| {
            let mut plain = models[1].clone();
            plain.descriptor = part.clone();
            plain
        })
        .collect();
    let shorter = common::hybrid_model(&parts, 297);
    assert!(NativePotential::new(shorter.clone()).is_ok());
    for pair in [[&models[0], &shorter], [&shorter, &models[0]]] {
        assert_eq!(
            omnimd_native::gpu_model::committee_mismatch(&pair).unwrap(),
            "model 1 differs from the first in its descriptor"
        );
        let error = GpuPotential::<f64>::committee(&pair, device.clone())
            .err()
            .unwrap();
        assert!(
            error.to_string().contains("differs from the first"),
            "{error}"
        );
    }
    // A model of one descriptor and a hybrid of that descriptor.
    let plain = model_of(2, vec![60, 90], 3.8, 295);
    let mut listed = common::hybrid_model(std::slice::from_ref(&plain), 296);
    listed.fitting = plain.fitting.clone();
    let pair = [&plain, &listed];
    assert_eq!(
        omnimd_native::gpu_model::committee_mismatch(&pair).unwrap(),
        "model 1 differs from the first in its descriptor"
    );
    assert!(GpuPotential::<f64>::committee(&pair, device.clone()).is_err());
    // The GPU trainer takes a hybrid of the three descriptors (its tests
    // are in gpu_train.rs).
    assert!(omnimd_native::gpu_train::GpuTrainer::<f64>::new(&models[0], device).is_ok());
}
