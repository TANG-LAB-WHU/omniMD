//! What a step of MLIP dynamics costs on the host, apart from the model: a
//! probe, not a test. The model here does next to nothing, so the time is
//! that of the integrator, its checks and its bookkeeping.
//!
//!     cargo test --release -p omnimd-mlmd --test step_cost -- --ignored --nocapture
use omnimd_ml_contract::{Evaluation, Frame, ModelMetadata, Potential, Result, Units};
use omnimd_mlmd::{
    initialize_velocities, system_from_frame, Ensemble, EnsembleState, MlipDynamics,
    VelocityInitialization,
};
use std::time::Instant;

/// Every atom on a spring to where it started.
struct Springs {
    metadata: ModelMetadata,
    home: Vec<[f64; 3]>,
}

impl Potential for Springs {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }
    fn eval(&mut self, frame: &Frame) -> Result<Evaluation> {
        let mut energy = 0.0;
        let forces = frame
            .positions
            .iter()
            .zip(&self.home)
            .map(|(r, home)| {
                let d: [f64; 3] = std::array::from_fn(|a| r[a] - home[a]);
                energy += 0.5 * d.iter().map(|x| x * x).sum::<f64>();
                d.map(|x| -x)
            })
            .collect();
        Ok(Evaluation {
            energy,
            forces,
            virial: [[0.0; 3]; 3],
            units: Units::EvAngstrom,
        })
    }
}

const ELEMENTS: [&str; 5] = ["Ca", "P", "O", "H", "C"];

fn dynamics(atoms: usize) -> MlipDynamics<Springs> {
    let edge = (atoms as f64 * 12.0).cbrt();
    let side = (atoms as f64).cbrt().ceil() as usize;
    let positions: Vec<[f64; 3]> = (0..atoms)
        .map(|i| {
            [i % side, i / side % side, i / (side * side)].map(|k| k as f64 * edge / side as f64)
        })
        .collect();
    let frame = Frame {
        species: (0..atoms).map(|i| ELEMENTS[i % 5].to_string()).collect(),
        cell: Some([[edge, 0.0, 0.0], [0.0, edge, 0.0], [0.0, 0.0, edge]]),
        positions: positions.clone(),
        units: Units::EvAngstrom,
    };
    let mut system = system_from_frame(&frame, &vec![16.0; atoms], &vec![[0.0; 3]; atoms]).unwrap();
    let settings: VelocityInitialization = serde_json::from_value(serde_json::json!(
        {"temperature_kelvin": 300.0, "seed": 3, "rescale_to_temperature": true}
    ))
    .unwrap();
    initialize_velocities(&mut system, &settings, true).unwrap();
    let model = Springs {
        metadata: ModelMetadata {
            type_map: ELEMENTS.iter().map(|s| s.to_string()).collect(),
            cutoff: 6.0,
            backend: "springs".into(),
            precision: "float64".into(),
            units: Units::EvAngstrom,
        },
        home: positions,
    };
    MlipDynamics::new(system, model, 0.5).unwrap()
}

/// Microseconds per step, the least of five rounds.
fn time(steps: usize, mut step: impl FnMut()) -> f64 {
    (0..5)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..steps {
                step();
            }
            start.elapsed().as_secs_f64() * 1e6 / steps as f64
        })
        .fold(f64::INFINITY, f64::min)
}

#[test]
#[ignore = "a timing probe"]
fn step_costs() {
    for atoms in [88, 224, 1024] {
        let mut md = dynamics(atoms);
        let nve = time(20_000, || {
            md.step().unwrap();
        });
        let mut md = dynamics(atoms);
        let mut state = EnsembleState::new(
            Ensemble::NvtCsvr {
                temperature_kelvin: 300.0,
                tau_fs: 50.0,
                seed: 1,
            },
            0.5,
        )
        .unwrap();
        let nvt = time(20_000, || state.advance(&mut md).unwrap());
        println!(
            "{atoms:>5} atoms: {nve:6.2} us per NVE step, {nvt:6.2} us per NVT step (host, model apart)"
        );
    }
}
