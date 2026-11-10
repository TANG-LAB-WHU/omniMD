//! Dataset statistics: closed forms, and DeePMD-kit 3.1.3 reference values
//! for a small mixed-composition dataset (`tools/make_stats_fixture.py`).
use omnimd_ml_contract::{Frame, Units};
use omnimd_native::stats::{energy_bias, env_mat_stat, neighbor_stat};
use omnimd_native::training_data::Dataset;

fn frame(species: &[&str], positions: &[[f64; 3]], cell: Option<[[f64; 3]; 3]>) -> Frame {
    Frame {
        species: species.iter().map(|s| s.to_string()).collect(),
        positions: positions.to_vec(),
        cell,
        units: Units::EvAngstrom,
    }
}

fn types(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| s.to_string()).collect()
}

fn cube(length: f64) -> Option<[[f64; 3]; 3]> {
    Some([[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]])
}

#[test]
fn neighbor_stat_counts_periodic_images() {
    // One atom in a 3 Å cube, cutoff 3.5 Å: its six nearest images.
    let single = frame(&["O"], &[[0.2, 0.3, 0.4]], cube(3.0));
    let stat = neighbor_stat(&[&single], &types(&["O", "H"]), 3.5).unwrap();
    assert_eq!(stat.max_nbor_size, [6, 0]);
    assert!((stat.min_nbor_dist - 3.0).abs() < 1e-12);
    // Cutoff past the face diagonal (4.24 Å): 6 + 12 images.
    let stat = neighbor_stat(&[&single], &types(&["O", "H"]), 4.3).unwrap();
    assert_eq!(stat.max_nbor_size, [18, 0]);

    // The largest count and the smallest distance are taken over all frames.
    let dimer = frame(&["O", "H"], &[[0.0; 3], [0.0, 0.0, 0.9]], None);
    let stat = neighbor_stat(&[&single, &dimer], &types(&["O", "H"]), 3.5).unwrap();
    assert_eq!(stat.max_nbor_size, [6, 1]);
    assert!((stat.min_nbor_dist - 0.9).abs() < 1e-12);
    // 6 and 1 neighbors at factor 1.1: 6.6 -> 6 -> 8, 1.1 -> 1 -> 4.
    assert_eq!(stat.auto_sel(1.1), [8, 4]);

    // No pair within the cutoff: nothing to report, not an error.
    let far = frame(&["O", "H"], &[[0.0; 3], [0.0, 0.0, 9.0]], None);
    let stat = neighbor_stat(&[&far], &types(&["O", "H"]), 3.5).unwrap();
    assert_eq!(stat.max_nbor_size, [0, 0]);
    assert!(stat.min_nbor_dist.is_infinite());

    assert!(neighbor_stat(&[&dimer], &types(&["O"]), 3.5).is_err());
    assert!(neighbor_stat(&[&dimer], &types(&["O", "H"]), 0.0).is_err());
}

#[test]
fn env_mat_stat_matches_the_dimer_closed_form() {
    // O-H along x at 2 Å, rcut 4, rcut_smth 1: u = 1/3.
    let dimer = frame(&["O", "H"], &[[0.0; 3], [2.0, 0.0, 0.0]], None);
    let u: f64 = 1.0 / 3.0;
    let s = (u.powi(3) * (-6.0 * u * u + 15.0 * u - 10.0) + 1.0) / 2.0;
    // Three slots per atom (sel [1, 2]), one filled: s for the radial part,
    // (±s, 0, 0) for the angular part, zeros elsewhere.
    let stats = env_mat_stat(&[&dimer], &types(&["O", "H"]), 4.0, 1.0, &[1, 2]).unwrap();
    for normalization in &stats {
        let mean = s / 3.0;
        assert!((normalization.mean[0] - mean).abs() < 1e-15);
        assert_eq!(normalization.mean[1..], [0.0; 3]);
        let radial = (s * s / 3.0 - mean * mean).sqrt();
        assert!((normalization.std[0] - radial).abs() < 1e-15);
        let angular = (s * s / 9.0 - (s / 9.0).powi(2)).sqrt();
        for a in 1..4 {
            assert!((normalization.std[a] - angular).abs() < 1e-15);
        }
    }

    // A type with no atoms gets DeePMD's defaults; tiny spreads are clamped.
    let stats = env_mat_stat(&[&dimer], &types(&["O", "H", "C"]), 4.0, 1.0, &[1, 1, 1]).unwrap();
    assert_eq!((stats[2].mean, stats[2].std), ([0.0; 4], [0.1; 4]));
    let lonely = frame(&["O"], &[[0.0; 3]], None);
    let stats = env_mat_stat(&[&lonely], &types(&["O"]), 4.0, 1.0, &[4]).unwrap();
    assert_eq!((stats[0].mean, stats[0].std), ([0.0; 4], [0.01; 4]));

    // sel keeps the nearest: with sel [0, 1] only the H at 1 Å counts for O.
    let trimer = frame(
        &["O", "H", "H"],
        &[[0.0; 3], [1.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
        None,
    );
    let near = env_mat_stat(&[&trimer], &types(&["O", "H"]), 4.0, 0.5, &[0, 1]).unwrap();
    let u: f64 = 0.5 / 3.5;
    let s = u.powi(3) * (-6.0 * u * u + 15.0 * u - 10.0) + 1.0;
    assert!((near[0].mean[0] - s).abs() < 1e-15);

    // DeePMD-kit 3 selection: the nearest sum(sel) = 2 neighbors of C are the
    // two H, so the O at 2 Å is crowded out although sel[O] = 1 has room, and
    // the per-type cap then keeps only the H at 1 Å.
    let crowded = frame(
        &["C", "H", "H", "O"],
        &[[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 2.0]],
        None,
    );
    let stats = env_mat_stat(&[&crowded], &types(&["C", "H", "O"]), 4.0, 0.5, &[0, 1, 1]).unwrap();
    assert!((stats[0].mean[0] - s / 2.0).abs() < 1e-15);

    assert!(env_mat_stat(&[&dimer], &types(&["O", "H"]), 4.0, 4.0, &[1, 1]).is_err());
    assert!(env_mat_stat(&[&dimer], &types(&["O", "H"]), 4.0, 1.0, &[1]).is_err());
}

#[test]
fn energy_bias_fits_type_counts() {
    let names = types(&["O", "H"]);
    let make = |o: usize, h: usize| {
        let species: Vec<&str> = std::iter::repeat_n("O", o)
            .chain(std::iter::repeat_n("H", h))
            .collect();
        let positions: Vec<[f64; 3]> = (0..o + h).map(|i| [i as f64 * 2.0, 0.0, 0.0]).collect();
        frame(&species, &positions, None)
    };
    // Independent compositions: the bias is recovered and nothing is left.
    let frames = [make(2, 1), make(1, 3), make(4, 4)];
    let energies: Vec<f64> = [(2.0, 1.0), (1.0, 3.0), (4.0, 4.0)]
        .iter()
        .map(|(o, h)| -430.0 * o - 13.6 * h)
        .collect();
    let refs: Vec<&Frame> = frames.iter().collect();
    let fit = energy_bias(&refs, &energies, &names).unwrap();
    assert_eq!(fit.rank, 2);
    assert!((fit.bias[0] + 430.0).abs() < 1e-10 && (fit.bias[1] + 13.6).abs() < 1e-10);
    assert!(fit.std < 1e-10);

    // Water only: counts cannot separate O from H, so the minimum-norm
    // split along (1, 2) is returned and the residual spread is kept.
    let frames = [make(1, 2), make(2, 4), make(3, 6)];
    let refs: Vec<&Frame> = frames.iter().collect();
    let fit = energy_bias(&refs, &[-10.0, -20.0, -33.0], &names).unwrap();
    assert_eq!(fit.rank, 1);
    assert!((fit.bias[1] - 2.0 * fit.bias[0]).abs() < 1e-12);
    // Least squares on e = 5 b·n with n = 1, 2, 3: b = (10 + 40 + 99) / (5·14).
    assert!((fit.bias[0] + 149.0 / 70.0).abs() < 1e-12);
    assert!(fit.std > 0.1);

    assert!(energy_bias(&refs, &[1.0], &names).is_err());
}

#[derive(serde::Deserialize)]
struct Reference {
    rcut: f64,
    rcut_smth: f64,
    sel: Vec<usize>,
    min_nbor_dist: f64,
    max_nbor_size: Vec<usize>,
    davg: Vec<[f64; 4]>,
    dstd: Vec<[f64; 4]>,
    energy_bias: Vec<f64>,
    energy_std: f64,
}

#[test]
fn statistics_match_deepmd_reference() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
    let dataset = Dataset::load(root.join("stats-dataset.jsonl")).unwrap();
    let reference: Reference = serde_json::from_str(
        &std::fs::read_to_string(root.join("stats-deepmd-reference.json")).unwrap(),
    )
    .unwrap();
    let names = &dataset.header.type_map;
    let frames: Vec<&Frame> = dataset.frames.iter().map(|f| &f.frame).collect();
    let energies: Vec<f64> = dataset
        .frames
        .iter()
        .map(|f| f.labels.energy.unwrap())
        .collect();
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-8 * b.abs().max(1.0);

    let neighbors = neighbor_stat(&frames, names, reference.rcut).unwrap();
    assert_eq!(neighbors.max_nbor_size, reference.max_nbor_size);
    assert!(close(neighbors.min_nbor_dist, reference.min_nbor_dist));

    let stats = env_mat_stat(
        &frames,
        names,
        reference.rcut,
        reference.rcut_smth,
        &reference.sel,
    )
    .unwrap();
    assert_eq!(stats.len(), reference.davg.len());
    for (t, normalization) in stats.iter().enumerate() {
        for a in 0..4 {
            assert!(
                close(normalization.mean[a], reference.davg[t][a]),
                "davg[{t}][{a}]: {} vs {}",
                normalization.mean[a],
                reference.davg[t][a]
            );
            assert!(
                close(normalization.std[a], reference.dstd[t][a]),
                "dstd[{t}][{a}]: {} vs {}",
                normalization.std[a],
                reference.dstd[t][a]
            );
        }
    }

    let fit = energy_bias(&frames, &energies, names).unwrap();
    for (ours, theirs) in fit.bias.iter().zip(&reference.energy_bias) {
        assert!(close(*ours, *theirs), "bias {ours} vs {theirs}");
    }
    assert!(close(fit.std, reference.energy_std));
}
