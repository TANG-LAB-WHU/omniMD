//! Constant-pressure dynamics: the cell move of the integrator, the
//! stochastic cell rescaling ensemble, its checkpoints and the CLI.
use omnimd_ml_contract::{Evaluation, Frame, ModelMetadata, Potential, Result, Units};
use omnimd_mlmd::{
    barostat::BAR_PER_EV_PER_ANGSTROM3, initialize_velocities, system_from_frame, CellCoupling,
    Checkpoint, Ensemble, EnsembleState, MlipDynamics, VelocityInitialization,
};
use omnimd_sim::md::Barostat;
use serde_json::{json, Value};
use std::{
    path::{Path, PathBuf},
    process::Command,
};

type Matrix = [[f64; 3]; 3];

/// Atoms that do not interact, in a cell with the shape energy
/// `kappa (sum |a_i|^2 / V^(2/3) - 3)` over the lattice vectors `a_i`.
struct CellGas {
    meta: ModelMetadata,
    kappa: f64,
}

impl CellGas {
    fn new(kappa: f64) -> Self {
        Self {
            meta: ModelMetadata {
                type_map: vec!["Ar".into()],
                cutoff: 4.,
                backend: "synthetic-cell-gas".into(),
                precision: "float64".into(),
                units: Units::EvAngstrom,
            },
            kappa,
        }
    }
}

fn determinant(a: &Matrix) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

/// `sum_i a_i a_i^T / V^(2/3)` for the rows `a_i` of the cell.
fn metric(cell: &Matrix) -> Matrix {
    let scale = determinant(cell).powf(-2. / 3.);
    std::array::from_fn(|a| {
        std::array::from_fn(|b| (0..3).map(|i| cell[i][a] * cell[i][b]).sum::<f64>() * scale)
    })
}

impl Potential for CellGas {
    fn metadata(&self) -> &ModelMetadata {
        &self.meta
    }
    fn eval(&mut self, frame: &Frame) -> Result<Evaluation> {
        let m = metric(&frame.cell.expect("periodic test frame"));
        let trace = m[0][0] + m[1][1] + m[2][2];
        Ok(Evaluation {
            energy: self.kappa * (trace - 3.),
            forces: vec![[0.; 3]; frame.positions.len()],
            // Minus the strain derivative of the energy.
            virial: std::array::from_fn(|a| {
                std::array::from_fn(|b| {
                    -self.kappa * (2. * m[a][b] - f64::from(a == b) * 2. / 3. * trace)
                })
            }),
            units: Units::EvAngstrom,
        })
    }
}

const CELL: Matrix = [[11., 0., 0.], [1.5, 10., 0.], [-0.8, 0.6, 9.]];

fn gas_frame(atoms: usize, cell: Matrix) -> Frame {
    Frame {
        positions: (0..atoms)
            .map(|i| {
                let t = i as f64;
                [
                    0.37 * t + 0.2,
                    1.3 * (0.7 * t).sin() + 4.,
                    0.11 * t * t % 7.,
                ]
            })
            .collect(),
        species: vec!["Ar".into(); atoms],
        cell: Some(cell),
        units: Units::EvAngstrom,
    }
}

fn gas(atoms: usize, cell: Matrix, kappa: f64, timestep: f64) -> MlipDynamics<CellGas> {
    let frame = gas_frame(atoms, cell);
    let mut system =
        system_from_frame(&frame, &vec![39.948; atoms], &vec![[0.; 3]; atoms]).unwrap();
    let settings = VelocityInitialization {
        temperature_kelvin: 300.,
        seed: 7,
        rescale_to_temperature: true,
    };
    initialize_velocities(&mut system, &settings, false).unwrap();
    MlipDynamics::new(system, CellGas::new(kappa), timestep).unwrap()
}

fn velocities<P: Potential>(md: &MlipDynamics<P>) -> Vec<[f64; 3]> {
    md.system()
        .particles()
        .velocity
        .iter()
        .map(|v| [v[0], v[1], v[2]])
        .collect()
}

fn apply(m: &Matrix, v: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|a| (0..3).map(|b| m[a][b] * v[b]).sum())
}

fn transpose(m: &Matrix) -> Matrix {
    std::array::from_fn(|a| std::array::from_fn(|b| m[b][a]))
}

fn inverse(a: &Matrix) -> Matrix {
    let d = determinant(a);
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let (r0, r1) = ((j + 1) % 3, (j + 2) % 3);
            let (c0, c1) = ((i + 1) % 3, (i + 2) % 3);
            (a[r0][c0] * a[r1][c1] - a[r0][c1] * a[r1][c0]) / d
        })
    })
}

fn assert_close(a: &[f64], b: &[f64], tolerance: f64) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= tolerance, "{x} vs {y}");
    }
}

/// A barostat that tries one fixed deformation and records what it is shown.
struct Fixed {
    deformation: Matrix,
    accept: bool,
    before: Option<(Matrix, f64)>,
    after: Option<(Matrix, f64, f64)>,
}

impl Barostat for Fixed {
    fn deformation(&mut self, pressure: &Matrix, volume: f64) -> Result<Matrix> {
        self.before = Some((*pressure, volume));
        Ok(self.deformation)
    }
    fn accept(&mut self, pressure: &Matrix, volume: f64, energy_change: f64) -> Result<bool> {
        self.after = Some((*pressure, volume, energy_change));
        Ok(self.accept)
    }
}

/// The pressure tensor from its definition, with the kinetic part scaled to
/// eV through the kinetic energy the integrator reports.
fn pressure_by_hand(md: &MlipDynamics<CellGas>) -> (Matrix, f64) {
    let particles = md.system().particles();
    let mut tensor = [[0.; 3]; 3];
    let mut twice_kinetic = 0.;
    for (mass, v) in particles.mass.iter().zip(particles.velocity.iter()) {
        for a in 0..3 {
            for b in 0..3 {
                tensor[a][b] += mass * v[a] * v[b];
            }
            twice_kinetic += mass * v[a] * v[a];
        }
    }
    let to_ev = 2. * md.kinetic_energy_ev() / twice_kinetic;
    let volume = determinant(&md.frame().cell.unwrap());
    let virial = md.evaluation().virial;
    (
        std::array::from_fn(|a| {
            std::array::from_fn(|b| (tensor[a][b] * to_ev + virial[a][b]) / volume)
        }),
        volume,
    )
}

#[test]
fn a_cell_move_maps_cell_positions_and_velocities_and_reports_its_work() {
    let deformation = [
        [1.02, 0.01, -0.004],
        [0.003, 0.97, 0.006],
        [0.0, -0.002, 1.01],
    ];
    let timestep = 0.5;
    for accept in [true, false] {
        let mut md = gas(6, CELL, 2.5, timestep);
        let start = md.frame();
        let start_velocities = velocities(&md);
        let start_energy = md.total_energy_ev();
        // Without forces the step itself is a free flight.
        let flown: Vec<[f64; 3]> = start
            .positions
            .iter()
            .zip(&start_velocities)
            .map(|(x, v)| std::array::from_fn(|a| x[a] + timestep * v[a]))
            .collect();
        let (pressure, volume) = pressure_by_hand(&md);
        let reported = md.pressure_tensor().unwrap();
        assert_close(&reported.0.concat(), &pressure.concat(), 1e-15);
        assert!((reported.1 - volume).abs() < 1e-9);

        let mut barostat = Fixed {
            deformation,
            accept,
            before: None,
            after: None,
        };
        let exchange = md.step_with_controls(None, Some(&mut barostat)).unwrap();
        assert_eq!(exchange.cell_move, Some(accept));
        assert_eq!(exchange.heat, 0.);
        assert_eq!(md.system().step, 1);
        // The barostat saw the state after the flight, which has the
        // pressure of the start (velocities and cell did not change).
        let (seen, seen_volume) = barostat.before.unwrap();
        assert_close(&seen.concat(), &pressure.concat(), 1e-15);
        assert!((seen_volume - volume).abs() < 1e-9);

        let (cell, positions, moved_velocities): (Matrix, Vec<[f64; 3]>, Vec<[f64; 3]>) = if accept
        {
            let map = transpose(&inverse(&deformation));
            (
                CELL.map(|row| apply(&deformation, &row)),
                flown.iter().map(|x| apply(&deformation, x)).collect(),
                start_velocities.iter().map(|v| apply(&map, v)).collect(),
            )
        } else {
            (CELL, flown.clone(), start_velocities.clone())
        };
        assert_close(&md.frame().cell.unwrap().concat(), &cell.concat(), 1e-12);
        assert_close(&md.frame().positions.concat(), &positions.concat(), 1e-12);
        assert_close(&velocities(&md).concat(), &moved_velocities.concat(), 1e-14);

        // The acceptance test is shown the moved state and its energy change,
        // whether or not the move stands.
        let (after, after_volume, change) = barostat.after.unwrap();
        assert!((after_volume - volume * determinant(&deformation)).abs() < 1e-8);
        if accept {
            let (by_hand, _) = pressure_by_hand(&md);
            assert_close(&after.concat(), &by_hand.concat(), 1e-15);
            assert!((exchange.work - change).abs() < 1e-15);
            assert!((md.total_energy_ev() - start_energy - change).abs() < 1e-12);
            assert!(change.abs() > 1e-3, "the move should change the energy");
        } else {
            assert_eq!(exchange.work, 0.);
            assert!((md.total_energy_ev() - start_energy).abs() < 1e-13);
        }
        // The cached evaluation belongs to the published geometry.
        let mut model = CellGas::new(2.5);
        let fresh = model.eval(&md.frame()).unwrap();
        assert_eq!(fresh.energy, md.evaluation().energy);
    }
}

#[test]
fn a_failing_cell_move_leaves_the_state_untouched() {
    struct Failing;
    impl Barostat for Failing {
        fn deformation(&mut self, _: &Matrix, _: f64) -> Result<Matrix> {
            // A reflection is not a deformation.
            Ok([[1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
        }
        fn accept(&mut self, _: &Matrix, _: f64, _: f64) -> Result<bool> {
            Ok(true)
        }
    }
    let mut md = gas(4, CELL, 1., 0.5);
    let (frame, before) = (md.frame(), velocities(&md));
    assert!(md.step_with_controls(None, Some(&mut Failing)).is_err());
    assert_eq!(md.system().step, 0);
    assert_eq!(md.frame().positions, frame.positions);
    assert_eq!(md.frame().cell, frame.cell);
    assert_eq!(velocities(&md), before);
}

fn npt(coupling: CellCoupling, pressure_bar: f64, compressibility_per_bar: f64) -> Ensemble {
    Ensemble::NptCsvrCrescale {
        temperature_kelvin: 300.,
        tau_fs: 50.,
        seed: 5,
        pressure_bar,
        barostat_tau_fs: 1000.,
        compressibility_per_bar,
        coupling,
        barostat_every: 2,
    }
}

#[test]
fn npt_settings_are_validated_and_have_defaults() {
    let parsed: Ensemble = serde_json::from_value(json!({
        "kind": "npt_csvr_crescale", "temperature_kelvin": 300.0, "tau_fs": 100.0, "seed": 1,
        "pressure_bar": 1.0, "barostat_tau_fs": 500.0, "compressibility_per_bar": 4.5e-5
    }))
    .unwrap();
    let Ensemble::NptCsvrCrescale {
        coupling,
        barostat_every,
        ..
    } = parsed
    else {
        panic!("not an NPT ensemble")
    };
    assert_eq!((coupling, barostat_every), (CellCoupling::Isotropic, 10));
    assert!(parsed.is_variable_cell() && !Ensemble::Nve.is_variable_cell());
    parsed.validate(0.5).unwrap();
    for (key, value) in [
        ("compressibility_per_bar", json!(0.0)),
        ("compressibility_per_bar", json!(-1e-5)),
        ("barostat_every", json!(0)),
        // Shorter than the ten steps between moves.
        ("barostat_tau_fs", json!(4.0)),
        ("temperature_kelvin", json!(0.0)),
        ("tau_fs", json!(0.1)),
    ] {
        let mut request = serde_json::to_value(&parsed).unwrap();
        request[key] = value;
        let ensemble: Ensemble = serde_json::from_value(request).unwrap();
        assert!(ensemble.validate(0.5).is_err(), "{key} accepted");
    }
    for bad in [
        json!({"coupling": "semi_isotropic"}),
        json!({"pressure": 1.0}),
    ] {
        let mut request = serde_json::to_value(&parsed).unwrap();
        for (key, value) in bad.as_object().unwrap() {
            request[key] = value.clone();
        }
        assert!(serde_json::from_value::<Ensemble>(request).is_err());
    }
    for name in ["isotropic", "anisotropic", "triclinic"] {
        let mut request = serde_json::to_value(&parsed).unwrap();
        request["coupling"] = json!(name);
        serde_json::from_value::<Ensemble>(request).unwrap();
    }
    // A cell is required.
    let state = EnsembleState::new(parsed, 0.5).unwrap();
    let mut frame = gas_frame(3, CELL);
    frame.cell = None;
    let mut open = system_from_frame(&frame, &[39.948; 3], &[[0.01; 3]; 3]).unwrap();
    assert!(state.validate_system(&open).is_err());
    open = system_from_frame(&gas_frame(3, CELL), &[39.948; 3], &[[0.01; 3]; 3]).unwrap();
    state.validate_system(&open).unwrap();
}

#[test]
fn couplings_move_only_their_strain_components() {
    let orthorhombic = [[10., 0., 0.], [0., 11., 0.], [0., 0., 12.]];
    for coupling in [
        CellCoupling::Isotropic,
        CellCoupling::Anisotropic,
        CellCoupling::Triclinic,
    ] {
        let mut md = gas(8, orthorhombic, 0.3, 0.5);
        // About the pressure of eight atoms in this cell at 300 K.
        let mut state = EnsembleState::new(npt(coupling, 250., 4e-3), 0.5).unwrap();
        for _ in 0..200 {
            state.advance(&mut md).unwrap();
        }
        assert_eq!(state.barostat_moves, 100);
        assert!(state.barostat_refused < 20, "{}", state.barostat_refused);
        assert!(state.barostat_work_ev != 0.);
        let cell = md.frame().cell.unwrap();
        let off: f64 = (0..3)
            .flat_map(|a| (0..3).filter(move |b| *b != a).map(move |b| (a, b)))
            .map(|(a, b)| cell[a][b].abs())
            .sum();
        let ratios = [cell[1][1] / cell[0][0], cell[2][2] / cell[0][0]];
        match coupling {
            CellCoupling::Isotropic => {
                assert_eq!(off, 0.);
                assert_close(&ratios, &[1.1, 1.2], 1e-12);
                assert!((cell[0][0] - 10.).abs() > 1e-3);
            }
            CellCoupling::Anisotropic => {
                assert_eq!(off, 0.);
                assert!((ratios[0] - 1.1).abs() > 1e-4 && (ratios[1] - 1.2).abs() > 1e-4);
            }
            CellCoupling::Triclinic => {
                assert!(off > 1e-3);
            }
        }
        // Thermostat heat and barostat work account for every energy change
        // of this force-free system.
        let drift = md.total_energy_ev() - state.heat_ev - state.barostat_work_ev;
        let initial = gas(8, orthorhombic, 0.3, 0.5).total_energy_ev();
        assert!(
            (drift - initial).abs() < 1e-10,
            "{coupling:?}: {drift} vs {initial}"
        );
    }
}

#[test]
fn npt_checkpoint_restores_cell_rng_and_work_exactly() {
    let digest = "b".repeat(64);
    let ensemble = npt(CellCoupling::Triclinic, 250., 4e-3);
    let mut uninterrupted = gas(8, CELL, 0.3, 0.5);
    let mut state = EnsembleState::new(ensemble.clone(), 0.5).unwrap();
    for _ in 0..15 {
        state.advance(&mut uninterrupted).unwrap();
    }
    let mut checkpoint = Checkpoint::capture(&uninterrupted, digest.clone());
    checkpoint.ensemble_state = Some(state.clone());
    // A constant-pressure state needs version 4, and only it may use it.
    assert!(checkpoint.restore(&digest, 0.5).is_err());
    checkpoint.schema_version = 4;
    let encoded = serde_json::to_string(&checkpoint).unwrap();
    let decoded: Checkpoint = serde_json::from_str(&encoded).unwrap();
    let mut resumed = MlipDynamics::new(
        decoded.restore(&digest, 0.5).unwrap(),
        CellGas::new(0.3),
        0.5,
    )
    .unwrap();
    let mut resumed_state = decoded.ensemble_state.clone().unwrap();
    for _ in 0..15 {
        state.advance(&mut uninterrupted).unwrap();
        resumed_state.advance(&mut resumed).unwrap();
    }
    // Step 15 is odd: the schedule of moves follows the absolute step.
    assert_eq!(state.barostat_moves, 15);
    assert_eq!(uninterrupted.frame().cell, resumed.frame().cell);
    assert_eq!(uninterrupted.frame().positions, resumed.frame().positions);
    assert_eq!(velocities(&uninterrupted), velocities(&resumed));
    assert_eq!(
        serde_json::to_string(&state).unwrap(),
        serde_json::to_string(&resumed_state).unwrap()
    );
    assert_ne!(uninterrupted.frame().cell, Some(CELL));

    let mut fixed_cell = Checkpoint::capture(&uninterrupted, digest.clone());
    fixed_cell.ensemble_state = Some(
        EnsembleState::new(
            Ensemble::NvtCsvr {
                temperature_kelvin: 300.,
                tau_fs: 50.,
                seed: 5,
            },
            0.5,
        )
        .unwrap(),
    );
    fixed_cell.restore(&digest, 0.5).unwrap();
    fixed_cell.schema_version = 4;
    assert!(fixed_cell.restore(&digest, 0.5).is_err());
    // Fixed-cell states do not carry barostat fields, so older readers accept them.
    let text = serde_json::to_string(&fixed_cell.ensemble_state).unwrap();
    assert!(!text.contains("barostat"), "{text}");
}

#[test]
fn npt_brings_an_ideal_gas_to_its_equation_of_state() {
    // Thirty atoms at 300 K and the pressure at which the mean volume of the
    // isothermal-isobaric ensemble, (N + 1) kT / P, is 20000 cubic angstrom,
    // started from a quarter of that volume.
    let atoms = 30;
    let side = 5000f64.cbrt();
    let mut md = gas(
        atoms,
        [[side, 0., 0.], [0., side, 0.], [0., 0., side]],
        0.,
        1.,
    );
    let thermal = md.thermal_energy_ev(300.);
    let pressure = (atoms as f64 + 1.) * thermal / 20000.;
    let ensemble = Ensemble::NptCsvrCrescale {
        temperature_kelvin: 300.,
        tau_fs: 20.,
        seed: 3,
        pressure_bar: pressure * BAR_PER_EV_PER_ANGSTROM3,
        barostat_tau_fs: 100.,
        // The compressibility of the gas, 1 / P.
        compressibility_per_bar: 1. / (pressure * BAR_PER_EV_PER_ANGSTROM3),
        coupling: CellCoupling::Isotropic,
        barostat_every: 1,
    };
    let mut state = EnsembleState::new(ensemble, 1.).unwrap();
    let initial = md.total_energy_ev();
    let (mut sum, mut samples) = (0., 0);
    for step in 0..24_000 {
        state.advance(&mut md).unwrap();
        if step >= 4_000 {
            sum += md.pressure_tensor().unwrap().1;
            samples += 1;
        }
    }
    let mean = sum / f64::from(samples);
    // The volume fluctuates by 18 percent over about 100 steps; the mean of
    // 20000 steps is good to about 2 percent.
    assert!((mean / 20000. - 1.).abs() < 0.08, "mean volume {mean}");
    assert!(state.barostat_refused * 20 < state.barostat_moves);
    let effective = md.total_energy_ev() - state.heat_ev - state.barostat_work_ev;
    assert!((effective - initial).abs() < 1e-9);
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../native/fixtures")
        .join(name)
        .canonicalize()
        .unwrap()
}

/// Four atoms of the synthetic O/H model in a triclinic cell.
fn water_frame() -> Frame {
    Frame {
        positions: vec![
            [0.2, 0.3, 0.4],
            [1.1, 0.5, 0.7],
            [0.6, 1.4, 0.9],
            [2.4, 2.0, 2.2],
        ],
        species: ["O", "H", "H", "O"].map(String::from).to_vec(),
        cell: Some([[3.7, 0., 0.], [1.2, 3.4, 0.], [0.6, 0.5, 3.8]]),
        units: Units::EvAngstrom,
    }
}

#[test]
fn the_model_virial_is_minus_the_strain_derivative_of_the_energy() {
    // The sign and layout the pressure tensor relies on, for the native model.
    let mut model = omnimd_native::NativePotential::load(fixture("synthetic-v1.json")).unwrap();
    let frame = water_frame();
    let virial = model.eval(&frame).unwrap().virial;
    let step = 1e-5;
    let mut largest: f64 = 0.;
    for a in 0..3 {
        for b in a..3 {
            let mut energies = [0.; 2];
            for (slot, sign) in [(0, 1.), (1, -1.)] {
                let mut d = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
                d[a][b] += 0.5 * sign * step;
                d[b][a] += 0.5 * sign * step;
                let mut strained = frame.clone();
                strained
                    .positions
                    .iter_mut()
                    .for_each(|x| *x = apply(&d, x));
                strained.cell = frame.cell.map(|cell| cell.map(|row| apply(&d, &row)));
                energies[slot] = model.eval(&strained).unwrap().energy;
            }
            let derivative = (energies[0] - energies[1]) / (2. * step);
            let symmetric = 0.5 * (virial[a][b] + virial[b][a]);
            assert!(
                (derivative + symmetric).abs() < 1e-9,
                "({a}, {b}): dE/de {derivative} vs virial {symmetric}"
            );
            largest = largest.max(symmetric.abs());
        }
    }
    // The components of this small model are a few 1e-4 eV.
    assert!(
        largest > 1e-4,
        "virial too small to test the sign: {largest}"
    );
}

/// Runs `omnimd-mlmd run request output` and returns the output records.
fn engine(directory: &Path, name: &str, request: &Value) -> Vec<Value> {
    let input = directory.join(format!("{name}.json"));
    let output = directory.join(format!("{name}.jsonl"));
    std::fs::write(&input, request.to_string()).unwrap();
    let run = Command::new(env!("CARGO_BIN_EXE_omnimd-mlmd"))
        .arg("run")
        .arg(&input)
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{name}: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    std::fs::read_to_string(output)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn the_cli_runs_npt_writes_pressure_and_restarts_exactly() {
    let directory = std::env::temp_dir().join(format!("omnimd-npt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let ensemble = json!({
        "kind": "npt_csvr_crescale", "temperature_kelvin": 300.0, "tau_fs": 20.0, "seed": 4,
        "pressure_bar": 1.0, "barostat_tau_fs": 200.0, "compressibility_per_bar": 1e-5,
        "coupling": "triclinic", "barostat_every": 4
    });
    let fresh = |steps: u64| {
        json!({
            "options": {"native_model": fixture("synthetic-v1.json")},
            "frame": water_frame(),
            "masses": [15.999, 1.008, 1.008, 15.999],
            "initialize_velocities":
                {"temperature_kelvin": 300.0, "seed": 7, "rescale_to_temperature": true},
            "remove_center_of_mass": true,
            "ensemble": ensemble,
            "timestep_fs": 0.5,
            "steps": steps,
            "sample_every": 8
        })
    };
    let whole = engine(&directory, "whole", &fresh(40));
    assert_eq!(whole[0]["schema"], "omnimd-mlmd-jsonl-v4");
    assert_eq!(whole[0]["ensemble"]["coupling"], "triclinic");
    let frames: Vec<&Value> = whole.iter().filter(|r| r["kind"] == "frame").collect();
    assert_eq!(frames.len(), 6, "steps 0, 8, ..., 40");
    for frame in &frames {
        let cell: Matrix = serde_json::from_value(frame["frame"]["cell"].clone()).unwrap();
        let volume = frame["volume_angstrom3"].as_f64().unwrap();
        assert!((volume - determinant(&cell)).abs() < 1e-9 * volume);
        let tensor: Matrix = serde_json::from_value(frame["pressure_tensor_bar"].clone()).unwrap();
        let mean = (tensor[0][0] + tensor[1][1] + tensor[2][2]) / 3.;
        assert!((frame["pressure_bar"].as_f64().unwrap() - mean).abs() < 1e-9 * mean.abs());
        // Energy bookkeeping: total minus heat minus work.
        let effective = frame["total_energy_ev"].as_f64().unwrap()
            - frame["thermostat_heat_ev"].as_f64().unwrap()
            - frame["barostat_work_ev"].as_f64().unwrap();
        assert!((frame["effective_energy_ev"].as_f64().unwrap() - effective).abs() < 1e-12);
    }
    assert_ne!(frames[0]["frame"]["cell"], frames[5]["frame"]["cell"]);
    let completed = whole.last().unwrap();
    assert_eq!(completed["kind"], "completed");
    assert_eq!(completed["barostat_moves"], 10);
    assert!(completed["barostat_refused"].as_u64().unwrap() <= 3);
    // The integrator alone moves the effective energy; over 40 steps of
    // this small model that stays far below the thermal energy.
    assert!(
        completed["max_absolute_effective_energy_drift_ev"]
            .as_f64()
            .unwrap()
            < 1e-4
    );

    // The same run in two halves.
    let first = engine(&directory, "first", &fresh(24));
    let checkpoint = directory.join("first.jsonl.checkpoint.json");
    let saved: Value =
        serde_json::from_str(&std::fs::read_to_string(&checkpoint).unwrap()).unwrap();
    assert_eq!(saved["schema_version"], 4);
    assert_eq!(first.last().unwrap()["barostat_moves"], 6);
    let second = engine(
        &directory,
        "second",
        &json!({
            "options": {"native_model": fixture("synthetic-v1.json")},
            "restart": checkpoint, "timestep_fs": 0.5, "steps": 16, "sample_every": 8
        }),
    );
    let last = |records: &[Value]| -> Value {
        records
            .iter()
            .rev()
            .find(|r| r["kind"] == "frame")
            .unwrap()
            .clone()
    };
    assert_eq!(last(&second), last(&whole));
    assert_eq!(second.last().unwrap()["barostat_moves"], 10);
    // A fixed-cell run of the same system still writes pressure and volume.
    let mut nvt = fresh(8);
    nvt["ensemble"] =
        json!({"kind": "nvt_csvr", "temperature_kelvin": 300.0, "tau_fs": 20.0, "seed": 4});
    let fixed = engine(&directory, "nvt", &nvt);
    assert_eq!(fixed[0]["schema"], "omnimd-mlmd-jsonl-v2");
    assert!(fixed[1]["pressure_bar"].is_number() && fixed[1]["volume_angstrom3"].is_number());
    assert!(fixed[1].get("barostat_work_ev").is_none());
    std::fs::remove_dir_all(directory).unwrap();
}
