//! GPU se_e2_a (omnimd-gpu) against the CPU reference `NativePotential`, which
//! is itself checked against DeePMD. Skips without a CUDA driver unless
//! OMNIMD_REQUIRE_GPU=1.
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
    assert!(GpuPotential::<f64>::committee(&[], device).is_err());
}
