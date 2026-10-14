use omnimd_core::{consts::K_BOLTZMANN, units, DegreesOfFreedom};
use omnimd_ml_contract::{Evaluation, Frame, ModelMetadata, Potential, Result, Units};
use omnimd_mlmd::{
    initialize_velocities, system_from_frame, Checkpoint, Ensemble, EnsembleState, MlipDynamics,
    VelocityInitialization,
};
use omnimd_sim::md::{CSVRThermostat, Thermostat};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn frame(n: usize) -> Frame {
    Frame {
        positions: (0..n)
            .map(|i| [0.1 + i as f64 * 0.02, -0.08, 0.07])
            .collect(),
        species: vec!["H".into(); n],
        cell: None,
        units: Units::EvAngstrom,
    }
}
fn init(seed: u64) -> VelocityInitialization {
    VelocityInitialization {
        temperature_kelvin: 300.,
        seed,
        rescale_to_temperature: true,
    }
}
struct Oscillator {
    meta: ModelMetadata,
}
impl Oscillator {
    fn new() -> Self {
        Self {
            meta: ModelMetadata {
                type_map: vec!["H".into()],
                cutoff: 4.,
                backend: "synthetic-anisotropic-oscillator".into(),
                precision: "float64".into(),
                units: Units::EvAngstrom,
            },
        }
    }
}
impl Potential for Oscillator {
    fn metadata(&self) -> &ModelMetadata {
        &self.meta
    }
    fn eval(&mut self, frame: &Frame) -> Result<Evaluation> {
        let stiffness = [1., 1.7, 2.3];
        Ok(Evaluation {
            energy: frame
                .positions
                .iter()
                .map(|x| {
                    (0..3)
                        .map(|a| 0.5 * stiffness[a] * x[a] * x[a])
                        .sum::<f64>()
                })
                .sum(),
            forces: frame
                .positions
                .iter()
                .map(|x| std::array::from_fn(|a| -stiffness[a] * x[a]))
                .collect(),
            virial: std::array::from_fn(|a| {
                std::array::from_fn(|b| {
                    -frame
                        .positions
                        .iter()
                        .map(|x| x[a] * stiffness[b] * x[b])
                        .sum::<f64>()
                })
            }),
            units: Units::EvAngstrom,
        })
    }
}
fn dynamics(n: usize) -> MlipDynamics<Oscillator> {
    let mut system = system_from_frame(&frame(n), &vec![1.; n], &vec![[0.; 3]; n]).unwrap();
    initialize_velocities(&mut system, &init(17), false).unwrap();
    MlipDynamics::new(system, Oscillator::new(), 0.1).unwrap()
}
fn ensemble() -> Ensemble {
    Ensemble::NvtCsvr {
        temperature_kelvin: 300.,
        tau_fs: 0.5,
        seed: 987,
    }
}

#[test]
fn thermal_initialization_is_seeded_and_counts_com_dof() {
    let f = frame(3);
    let mut a = system_from_frame(&f, &[1., 2., 3.], &[[0.; 3]; 3]).unwrap();
    let mut b = a.clone();
    initialize_velocities(&mut a, &init(42), true).unwrap();
    initialize_velocities(&mut b, &init(42), true).unwrap();
    assert_eq!(a.particles().velocity, b.particles().velocity);
    assert_eq!(a.degrees_of_freedom(), 6);
    assert!((a.temperature() - 300.).abs() < 1e-9);
    for axis in 0..3 {
        let momentum: f64 = a
            .particles()
            .iter()
            .map(|p| p.mass * p.velocity[axis])
            .sum();
        assert!(momentum.abs() < 1e-15);
    }
    initialize_velocities(&mut b, &init(43), true).unwrap();
    assert_ne!(a.particles().velocity, b.particles().velocity);
    let mut one = system_from_frame(&frame(1), &[1.], &[[0.; 3]]).unwrap();
    assert!(initialize_velocities(&mut one, &init(1), true).is_err());
    let bad = VelocityInitialization {
        temperature_kelvin: 0.,
        ..init(1)
    };
    assert!(initialize_velocities(&mut one, &bad, false).is_err());
    assert!(Ensemble::NvtCsvr {
        temperature_kelvin: 300.,
        tau_fs: 0.01,
        seed: 1
    }
    .validate(0.1)
    .is_err());
}

#[test]
fn csvr_one_step_kinetic_moments_and_negative_scale() {
    // Independent one-step trials detect correlated R1/chi-square noise; no autocorrelation.
    let theta = K_BOLTZMANN * 300.;
    let c = (-1.0_f64).exp();
    let b = (1. - c) * theta / 2.;
    for dof in [1, 2, 3, 6] {
        let n = (dof + 2) / 3;
        let mut system = system_from_frame(&frame(n), &vec![1.; n], &vec![[0.; 3]; n]).unwrap();
        system.simulated_degrees_of_freedom = DegreesOfFreedom::Frozen(3 * n - dof);
        let initial = dof as f64 * theta / 2.;
        let speed = (2. * initial).sqrt();
        let mut thermostat =
            CSVRThermostat::from_rng(300., 1., Box::new(ChaCha8Rng::seed_from_u64(1234)));
        let mut sum = 0.;
        let mut sum2 = 0.;
        let mut negative = 0;
        let samples = 25000;
        for _ in 0..samples {
            for v in system.particles_mut().velocity {
                *v = [0.; 3].into();
            }
            system.particles_mut().velocity[0][0] = speed;
            thermostat.apply(&mut system);
            let k = system.kinetic_energy();
            sum += k;
            sum2 += k * k;
            if system.particles().velocity[0][0] < 0. {
                negative += 1;
            }
        }
        let mean = sum / samples as f64;
        let variance = sum2 / samples as f64 - mean * mean;
        let expected_mean = c * initial + dof as f64 * b;
        let expected_var = 4. * c * initial * b + 2. * dof as f64 * b * b;
        println!(
            "CSVR DOF {dof}: mean ratio {}, conditional variance ratio {}",
            mean / expected_mean,
            variance / expected_var
        );
        assert!((mean / expected_mean - 1.).abs() < 0.02);
        assert!((variance / expected_var - 1.).abs() < 0.06);
        if dof == 1 {
            assert!(negative > 100);
        }
    }
}

#[test]
fn csvr_checkpoint_restores_rng_and_heat_exactly() {
    let mut uninterrupted = dynamics(2);
    let mut state = EnsembleState::new(ensemble(), 0.1).unwrap();
    for _ in 0..20 {
        state.advance(&mut uninterrupted).unwrap();
    }
    let digest = "a".repeat(64);
    let mut checkpoint = Checkpoint::capture(&uninterrupted, digest.clone());
    checkpoint.ensemble_state = Some(state.clone());
    let encoded = serde_json::to_string(&checkpoint).unwrap();
    let decoded: Checkpoint = serde_json::from_str(&encoded).unwrap();
    let mut resumed = MlipDynamics::new(
        decoded.restore(&digest, 0.1).unwrap(),
        Oscillator::new(),
        0.1,
    )
    .unwrap();
    let mut resumed_state = decoded.ensemble_state.unwrap();
    for _ in 0..20 {
        state.advance(&mut uninterrupted).unwrap();
        resumed_state.advance(&mut resumed).unwrap();
    }
    assert_eq!(uninterrupted.frame().positions, resumed.frame().positions);
    assert_eq!(
        uninterrupted.system().particles().velocity,
        resumed.system().particles().velocity
    );
    assert_eq!(state.heat_ev, resumed_state.heat_ev);
    assert_eq!(
        serde_json::to_string(&state).unwrap(),
        serde_json::to_string(&resumed_state).unwrap()
    );
}

#[test]
fn csvr_anisotropic_oscillator_has_thermal_energy_and_configurational_scale() {
    let mut md = dynamics(1);
    let mut state = EnsembleState::new(ensemble(), 0.1).unwrap();
    let theta = K_BOLTZMANN * 300. / units::from(1., "eV").unwrap();
    let initial_effective = md.total_energy_ev();
    let mut max_effective_drift = 0.0_f64;
    let mut k = 0.;
    let mut k2 = 0.;
    let mut x2 = [0.; 3];
    let mut count = 0;
    for step in 0..100000 {
        state.advance(&mut md).unwrap();
        max_effective_drift = max_effective_drift
            .max((md.total_energy_ev() - state.heat_ev - initial_effective).abs());
        if step >= 10000 && step % 10 == 0 {
            count += 1;
            k += md.kinetic_energy_ev();
            k2 += md.kinetic_energy_ev().powi(2);
            for a in 0..3 {
                x2[a] += md.frame().positions[0][a].powi(2);
            }
        }
    }
    let mean = k / count as f64;
    let variance = k2 / count as f64 - mean * mean;
    let ratios: [f64; 3] =
        std::array::from_fn(|a| x2[a] / count as f64 * [1., 1.7, 2.3][a] / theta);
    println!("Oscillator kinetic mean ratio {}, variance ratio {}, position second-moment ratios {:?}, effective drift {} eV",
        mean/(1.5*theta),variance/(1.5*theta*theta),ratios,max_effective_drift);
    assert!((mean / (1.5 * theta) - 1.).abs() < 0.1);
    assert!((variance / (1.5 * theta * theta) - 1.).abs() < 0.15);
    for ratio in ratios {
        assert!((ratio - 1.).abs() < 0.2);
    }
    assert!(max_effective_drift < 0.001);
}
