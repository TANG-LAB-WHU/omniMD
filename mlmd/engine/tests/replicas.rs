//! Replicas advanced together against the same systems run alone: on the CPU
//! the two are the same arithmetic, so everything must agree exactly.
use omnimd_ml_contract::{Error, Evaluation, Frame, ModelMetadata, Potential, Result, Units};
use omnimd_mlmd::{
    advance_replicas, initialize_velocities, system_from_frame, Controls, Ensemble, EnsembleState,
    MlipDynamics, MlipReplicas, StateView, VelocityInitialization,
};
use omnimd_sim::md::{Barostat, Thermostat};
use serde_json::{json, Value};
use std::{
    cell::Cell,
    path::{Path, PathBuf},
    process::Command,
    rc::Rc,
};

type Matrix = [[f64; 3]; 3];

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../native/fixtures")
        .join(name)
        .canonicalize()
        .unwrap()
}

fn native() -> omnimd_native::NativePotential {
    omnimd_native::NativePotential::load(fixture("synthetic-v1.json")).unwrap()
}

/// Atoms of the synthetic O/H model, in a triclinic cell unless `open`;
/// `variant` moves them a little.
fn water(atoms: usize, open: bool, variant: usize) -> Frame {
    let shift = 0.03 * variant as f64;
    let all = [
        [0.2 + shift, 0.3, 0.4],
        [1.1, 0.5 - shift, 0.7],
        [0.6, 1.4, 0.9 + shift],
        [2.4 - shift, 2.0, 2.2],
    ];
    Frame {
        positions: all[..atoms].to_vec(),
        species: ["O", "H", "H", "O"][..atoms]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        cell: (!open).then_some([[3.7, 0., 0.], [1.2, 3.4, 0.], [0.6, 0.5, 3.8]]),
        units: Units::EvAngstrom,
    }
}

const MASSES: [f64; 4] = [15.999, 1.008, 1.008, 15.999];

fn thermal_system(frame: &Frame, seed: u64) -> omnimd_core::System {
    let atoms = frame.positions.len();
    let mut system = system_from_frame(frame, &MASSES[..atoms], &vec![[0.; 3]; atoms]).unwrap();
    let settings: VelocityInitialization = serde_json::from_value(
        json!({"temperature_kelvin": 300.0, "seed": seed, "rescale_to_temperature": true}),
    )
    .unwrap();
    initialize_velocities(&mut system, &settings, true).unwrap();
    system
}

fn nvt(seed: u64) -> Ensemble {
    Ensemble::NvtCsvr {
        temperature_kelvin: 300.,
        tau_fs: 20.,
        seed,
    }
}

fn npt(seed: u64, every: u64) -> Ensemble {
    serde_json::from_value(json!({
        "kind": "npt_csvr_crescale", "temperature_kelvin": 300.0, "tau_fs": 20.0, "seed": seed,
        "pressure_bar": 1.0, "barostat_tau_fs": 200.0, "compressibility_per_bar": 1e-5,
        "coupling": "triclinic", "barostat_every": every
    }))
    .unwrap()
}

/// Everything a step can change, to compare exactly.
fn snapshot(md: StateView<'_>) -> (Frame, Vec<[f64; 3]>, Evaluation, u64) {
    let velocities = md
        .system()
        .particles()
        .velocity
        .iter()
        .map(|v| [v[0], v[1], v[2]])
        .collect();
    (
        md.frame(),
        velocities,
        md.evaluation().clone(),
        md.system().step,
    )
}

fn same(a: StateView<'_>, b: StateView<'_>, what: &str) {
    let (x, y) = (snapshot(a), snapshot(b));
    assert_eq!(x.0.positions, y.0.positions, "{what}: positions");
    assert_eq!(x.0.cell, y.0.cell, "{what}: cell");
    assert_eq!(x.1, y.1, "{what}: velocities");
    assert_eq!(x.2.energy.to_bits(), y.2.energy.to_bits(), "{what}: energy");
    assert_eq!(x.2.forces, y.2.forces, "{what}: forces");
    assert_eq!(x.2.virial, y.2.virial, "{what}: virial");
    assert_eq!(x.3, y.3, "{what}: step");
}

#[test]
fn replicas_under_their_ensembles_go_as_each_goes_alone() {
    // Four systems that share nothing but the model: sizes, boundary,
    // ensemble. The third moves its cell every other step.
    let frames = [
        water(4, false, 0),
        water(4, false, 1),
        water(4, false, 2),
        water(3, true, 3),
    ];
    let ensembles = [Ensemble::Nve, nvt(4), npt(5, 2), nvt(6)];
    let systems = || -> Vec<omnimd_core::System> {
        frames
            .iter()
            .enumerate()
            .map(|(index, frame)| thermal_system(frame, 10 + index as u64))
            .collect()
    };
    let states = || -> Vec<EnsembleState> {
        ensembles
            .iter()
            .map(|ensemble| EnsembleState::new(ensemble.clone(), 0.5).unwrap())
            .collect()
    };
    let mut alone: Vec<MlipDynamics<_>> = systems()
        .into_iter()
        .map(|system| MlipDynamics::new(system, native(), 0.5).unwrap())
        .collect();
    let mut alone_states = states();
    let mut together = MlipReplicas::new(systems(), native(), 0.5).unwrap();
    let mut together_states = states();
    assert_eq!((together.len(), together.is_empty()), (4, false));
    assert_eq!(together.timestep_fs(), 0.5);
    assert_eq!(together.metadata().type_map, ["O", "H"]);
    for step in 0..=24 {
        for index in 0..4 {
            let what = format!("step {step}, replica {index}");
            same(together.view(index), alone[index].view(), &what);
            assert_eq!(
                serde_json::to_string(&together_states[index]).unwrap(),
                serde_json::to_string(&alone_states[index]).unwrap(),
                "{what}: ensemble state"
            );
        }
        advance_replicas(&mut together_states, &mut together).unwrap();
        for (state, md) in alone_states.iter_mut().zip(&mut alone) {
            state.advance(md).unwrap();
        }
    }
    // The controls did their work: heat everywhere but in NVE, cell moves
    // only where there is a barostat.
    assert_eq!(together_states[0].heat_ev, 0.);
    assert!(together_states[1].heat_ev != 0. && together_states[3].heat_ev != 0.);
    let moves: Vec<u64> = together_states.iter().map(|s| s.barostat_moves).collect();
    assert_eq!(moves, [0, 0, 12, 0]);
    assert_ne!(together.view(2).frame().cell, frames[2].cell);
    assert_eq!(together.view(1).frame().cell, frames[1].cell);
    assert_eq!(
        together.view(1).temperature_kelvin(),
        alone[1].temperature_kelvin()
    );
    assert_eq!(
        together.view(2).pressure_tensor().unwrap(),
        alone[2].pressure_tensor().unwrap()
    );
    assert!(together.view(3).pressure_tensor().is_err());

    // As many states as systems.
    assert!(advance_replicas(&mut together_states[..3], &mut together).is_err());
    let mut plain = [Controls::default(), Controls::default()];
    assert!(together.step(&mut plain).is_err());
    same(together.view(0), alone[0].view(), "after refused steps");
    // An ensemble that does not fit its system is refused for all: constant
    // pressure without a cell.
    together_states[3] = EnsembleState::new(npt(7, 2), 0.5).unwrap();
    let error = advance_replicas(&mut together_states, &mut together).unwrap_err();
    let expected = "replica 3: NPT requires a periodic cell";
    assert!(error.to_string().contains(expected), "{error}");
    same(
        together.view(1),
        alone[1].view(),
        "after a refused ensemble",
    );
    assert!(MlipReplicas::new(Vec::new(), native(), 0.5).is_err());
    assert!(MlipReplicas::new(systems(), native(), 0.).is_err());
}

/// The native model, failing on request and counting its calls.
struct Flaky {
    model: omnimd_native::NativePotential,
    calls: Rc<Cell<usize>>,
    fail_at: Rc<Cell<usize>>,
}

impl Potential for Flaky {
    fn metadata(&self) -> &ModelMetadata {
        self.model.metadata()
    }
    fn eval(&mut self, frame: &Frame) -> Result<Evaluation> {
        self.calls.set(self.calls.get() + 1);
        if self.calls.get() == self.fail_at.get() {
            return Err(Error("injected model failure".into()));
        }
        self.model.eval(frame)
    }
}

/// Moves an atom: not a thermostat's business.
struct Mover;

impl Thermostat for Mover {
    fn apply(&mut self, system: &mut omnimd_core::System) {
        system.particles_mut().position[0][0] += 0.25;
    }
}

/// Scales velocities by a factor.
struct Scale(f64);

impl Thermostat for Scale {
    fn apply(&mut self, system: &mut omnimd_core::System) {
        for velocity in system.particles_mut().velocity {
            *velocity = self.0 * *velocity;
        }
    }
}

/// Tries one fixed deformation.
struct Fixed {
    deformation: Matrix,
    accept: bool,
    asked: usize,
}

impl Barostat for Fixed {
    fn deformation(&mut self, _: &Matrix, _: f64) -> Result<Matrix> {
        self.asked += 1;
        Ok(self.deformation)
    }
    fn accept(&mut self, _: &Matrix, _: f64, _: f64) -> Result<bool> {
        Ok(self.accept)
    }
}

#[test]
fn a_step_that_fails_for_one_system_moves_none() {
    let frames = [water(4, false, 0), water(4, false, 1), water(3, true, 2)];
    let systems = || -> Vec<omnimd_core::System> {
        frames
            .iter()
            .enumerate()
            .map(|(index, frame)| thermal_system(frame, 20 + index as u64))
            .collect()
    };
    let (calls, fail_at) = (Rc::new(Cell::new(0)), Rc::new(Cell::new(usize::MAX)));
    let flaky = Flaky {
        model: native(),
        calls: calls.clone(),
        fail_at: fail_at.clone(),
    };
    let mut md = MlipReplicas::new(systems(), flaky, 0.5).unwrap();
    assert_eq!(calls.get(), 3, "one evaluation per system to start");
    // Each call with a lifetime of its own, so that the controls of one
    // step do not stay borrowed for the next.
    fn plain<'a>() -> [Controls<'a>; 3] {
        [
            Controls::default(),
            Controls::default(),
            Controls::default(),
        ]
    }
    md.step(&mut plain()).unwrap();
    assert_eq!(calls.get(), 6);
    let mut reference: Vec<MlipDynamics<_>> = systems()
        .into_iter()
        .map(|system| MlipDynamics::new(system, native(), 0.5).unwrap())
        .collect();
    for alone in &mut reference {
        alone.step().unwrap();
    }
    fn unchanged(
        md: &MlipReplicas<Flaky>,
        reference: &[MlipDynamics<omnimd_native::NativePotential>],
        what: &str,
    ) {
        for (index, alone) in reference.iter().enumerate() {
            same(
                md.view(index),
                alone.view(),
                &format!("{what}, replica {index}"),
            );
        }
    }
    unchanged(&md, &reference, "one step");

    // The model fails for the second system: the first has been evaluated
    // and the third not yet.
    fail_at.set(calls.get() + 2);
    let error = md.step(&mut plain()).unwrap_err();
    assert!(error.to_string().contains("injected"), "{error}");
    unchanged(&md, &reference, "after a model failure");
    // A thermostat that moves an atom of the third system: the others have
    // had their second kick by then.
    let (mut scale, mut mover) = (Scale(0.9), Mover);
    let mut controls = plain();
    controls[0].thermostat = Some(&mut scale);
    controls[2].thermostat = Some(&mut mover);
    let error = md.step(&mut controls).unwrap_err().to_string();
    assert!(
        error.contains("replica 2") && error.contains("thermostat changed more"),
        "{error}"
    );
    unchanged(&md, &reference, "after a thermostat failure");
    // A cell move for a system without a cell, after the first system's
    // cell has moved.
    let stretch = [[1.01, 0., 0.], [0., 1., 0.], [0., 0., 0.99]];
    let barostat = |accept| Fixed {
        deformation: stretch,
        accept,
        asked: 0,
    };
    let (mut first, mut last) = (barostat(true), barostat(true));
    let mut controls = plain();
    controls[0].barostat = Some(&mut first);
    controls[2].barostat = Some(&mut last);
    let error = md.step(&mut controls).unwrap_err().to_string();
    assert!(
        error.contains("replica 2") && error.contains("periodic cell"),
        "{error}"
    );
    assert_eq!((first.asked, last.asked), (1, 0));
    unchanged(&md, &reference, "after a cell move failure");
    // The model failing on a moved cell: the step and the move are undone.
    let mut first = barostat(true);
    let mut controls = plain();
    controls[0].barostat = Some(&mut first);
    fail_at.set(calls.get() + 4);
    assert!(md.step(&mut controls).is_err());
    unchanged(&md, &reference, "after a failure on the moved cell");

    // The same controls with nothing failing: what each system does alone.
    let (mut scale, mut accepted, mut refused) = (Scale(0.9), barostat(true), barostat(false));
    let mut controls = plain();
    controls[0].barostat = Some(&mut accepted);
    controls[1].barostat = Some(&mut refused);
    controls[1].thermostat = Some(&mut scale);
    let before = calls.get();
    let exchanges = md.step(&mut controls).unwrap();
    assert_eq!(calls.get() - before, 5, "three systems and two moved cells");
    let (mut scale, mut accepted, mut refused) = (Scale(0.9), barostat(true), barostat(false));
    let alone = [
        reference[0]
            .step_with_controls(None, Some(&mut accepted))
            .unwrap(),
        reference[1]
            .step_with_controls(Some(&mut scale), Some(&mut refused))
            .unwrap(),
        reference[2].step_with_controls(None, None).unwrap(),
    ];
    assert_eq!(exchanges.as_slice(), alone.as_slice());
    assert_eq!(exchanges[0].cell_move, Some(true));
    assert_eq!(exchanges[1].cell_move, Some(false));
    assert!(exchanges[0].work != 0. && exchanges[1].work == 0. && exchanges[1].heat < 0.);
    assert_eq!(exchanges[2], Default::default());
    unchanged(&md, &reference, "after a step with controls");
    assert_ne!(md.view(0).frame().cell, frames[0].cell);
    assert_eq!(md.view(1).frame().cell, frames[1].cell);
}

#[test]
fn a_system_that_cannot_begin_a_step_stops_the_others_where_they_are() {
    // The second system is one step short of the last step number.
    let frames = [water(4, false, 0), water(4, false, 1), water(3, true, 2)];
    let systems = || -> Vec<omnimd_core::System> {
        let mut systems: Vec<_> = frames
            .iter()
            .enumerate()
            .map(|(index, frame)| thermal_system(frame, 30 + index as u64))
            .collect();
        systems[1].step = u64::MAX - 1;
        systems
    };
    fn plain<'a>() -> [Controls<'a>; 3] {
        [
            Controls::default(),
            Controls::default(),
            Controls::default(),
        ]
    }
    let mut md = MlipReplicas::new(systems(), native(), 0.5).unwrap();
    let mut reference: Vec<MlipDynamics<_>> = systems()
        .into_iter()
        .map(|system| MlipDynamics::new(system, native(), 0.5).unwrap())
        .collect();
    md.step(&mut plain()).unwrap();
    for alone in &mut reference {
        alone.step().unwrap();
    }
    assert_eq!(md.view(1).system().step, u64::MAX);
    // No step is left for it. The system before it had begun the step and
    // the one after it had not: both stay where the last step left them,
    // however often the step is tried.
    for attempt in 0..2 {
        let error = md.step(&mut plain()).unwrap_err().to_string();
        assert!(
            error.contains("replica 1") && error.contains("step counter overflow"),
            "{error}"
        );
        for (index, alone) in reference.iter().enumerate() {
            let what = format!("attempt {attempt}, replica {index}");
            same(md.view(index), alone.view(), &what);
        }
    }
    assert!(reference[1].step().is_err());
}

/// A model that answers a batch with one evaluation too few.
struct Short(omnimd_native::NativePotential);

impl Potential for Short {
    fn metadata(&self) -> &ModelMetadata {
        self.0.metadata()
    }
    fn eval(&mut self, frame: &Frame) -> Result<Evaluation> {
        self.0.eval(frame)
    }
    fn eval_batch(&mut self, frames: &[&Frame]) -> Result<Vec<Evaluation>> {
        let mut all = self.0.eval_batch(frames)?;
        all.pop();
        Ok(all)
    }
}

#[test]
fn a_model_must_answer_for_every_system() {
    let systems = vec![
        thermal_system(&water(4, false, 0), 40),
        thermal_system(&water(3, true, 1), 41),
    ];
    let error = MlipReplicas::new(systems, Short(native()), 0.5)
        .err()
        .expect("one evaluation for two systems")
        .to_string();
    assert!(error.contains("1 evaluations for 2 frames"), "{error}");
}

/// Runs an engine command that must succeed.
fn engine(command: &str, input: &Path, output: &Path) {
    let run = Command::new(env!("CARGO_BIN_EXE_omnimd-mlmd"))
        .arg(command)
        .arg(input)
        .arg(output)
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{command} {}: {}",
        input.display(),
        String::from_utf8_lossy(&run.stderr)
    );
}

fn records(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn read(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn the_cli_runs_replicas_as_it_runs_each_alone_and_restarts_them() {
    let directory = std::env::temp_dir().join(format!("omnimd-replicas-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let model = fixture("synthetic-v1.json");
    let thermal = |seed: u64| json!({"temperature_kelvin": 300.0, "seed": seed, "rescale_to_temperature": true});
    // What a `run` request says of each system.
    let replicas = [
        (
            "plain",
            json!({"frame": water(4, false, 0), "masses": MASSES,
            "initialize_velocities": thermal(7), "remove_center_of_mass": true}),
        ),
        (
            "warm.300K",
            json!({"frame": water(4, false, 1), "masses": MASSES,
            "initialize_velocities": thermal(8), "remove_center_of_mass": true,
            "ensemble": {"kind": "nvt_csvr", "temperature_kelvin": 300.0, "tau_fs": 20.0, "seed": 4}}),
        ),
        (
            "pressed_1bar",
            json!({"frame": water(4, false, 2), "masses": MASSES,
            "initialize_velocities": thermal(9), "remove_center_of_mass": true,
            "ensemble": {"kind": "npt_csvr_crescale", "temperature_kelvin": 300.0, "tau_fs": 20.0,
                "seed": 5, "pressure_bar": 1.0, "barostat_tau_fs": 200.0,
                "compressibility_per_bar": 1e-5, "coupling": "triclinic", "barostat_every": 4}}),
        ),
        (
            "open-3",
            json!({"frame": water(3, true, 3), "masses": MASSES[..3],
            "velocities": [[0.001, 0.0, 0.0], [0.0, -0.002, 0.0], [0.0, 0.0, 0.001]]}),
        ),
    ];
    let shared = |steps: u64| {
        json!({"options": {"native_model": model}, "timestep_fs": 0.5,
        "steps": steps, "sample_every": 8, "checkpoint_every": 16})
    };
    let request = |steps: u64, of: &dyn Fn(usize) -> Value| {
        let mut request = shared(steps);
        request["schema"] = json!("omnimd-replicas-v1");
        request["replicas"] = replicas
            .iter()
            .enumerate()
            .map(|(index, (name, _))| {
                let mut replica = of(index);
                replica["name"] = json!(name);
                replica
            })
            .collect();
        request
    };
    let fresh = |index: usize| replicas[index].1.clone();
    let run_replicas = |name: &str, request: &Value| -> PathBuf {
        let input = directory.join(format!("{name}.json"));
        std::fs::write(&input, request.to_string()).unwrap();
        let output = directory.join(name);
        engine("run-replicas", &input, &output);
        output
    };
    let together = run_replicas("together", &request(40, &fresh));
    let summary = read(&together.join("replicas.json"));
    assert_eq!(summary["schema"], "omnimd-replicas-run-v1");
    assert_eq!(summary["replicas"].as_array().unwrap().len(), 4);
    assert_eq!(summary["steps"], 40);
    for (index, (name, replica)) in replicas.iter().enumerate() {
        // The same system through `run`.
        let mut single = shared(40);
        for (key, value) in replica.as_object().unwrap() {
            single[key] = value.clone();
        }
        let input = directory.join(format!("alone-{name}.json"));
        std::fs::write(&input, single.to_string()).unwrap();
        let output = directory.join(format!("alone-{name}.jsonl"));
        engine("run", &input, &output);
        let trajectory = together.join(format!("{name}.jsonl"));
        let (mut got, mut expected) = (records(&trajectory), records(&output));
        assert_eq!(got.len(), expected.len(), "{name}");
        // The header names the replica; the completed records name their
        // own checkpoints.
        assert_eq!(
            got[0]["replica"],
            json!({"name": name, "index": index, "of": 4})
        );
        got[0].as_object_mut().unwrap().remove("replica");
        let checkpoint = |records: &mut Vec<Value>| -> PathBuf {
            let last = records.last_mut().unwrap().as_object_mut().unwrap();
            serde_json::from_value(last.remove("checkpoint").unwrap()).unwrap()
        };
        let (ours, theirs) = (checkpoint(&mut got), checkpoint(&mut expected));
        assert_eq!(ours, together.join(format!("{name}.jsonl.checkpoint.json")));
        for (line, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert_eq!(g, e, "{name}: record {line}");
        }
        assert_eq!(read(&ours), read(&theirs), "{name}: final checkpoint");
        for step in [16, 32] {
            let at = |path: &Path| {
                let mut name = path.as_os_str().to_os_string();
                name.push(format!(".step-{step}.checkpoint.json"));
                read(Path::new(&name))
            };
            assert_eq!(
                at(&trajectory),
                at(&output),
                "{name}: checkpoint of step {step}"
            );
        }
        let entry = &summary["replicas"][index];
        assert_eq!(entry["name"], *name);
        assert_eq!(entry["final_step"], 40);
        assert_eq!(entry["trajectory"], json!(trajectory));
        assert_eq!(entry["checkpoint"], json!(ours));
        assert_eq!(got[0]["start_step"], 0);
        assert_eq!(got.last().unwrap()["final_step"], 40);
    }
    // The replicas are not copies of each other.
    let ends: Vec<Value> = replicas
        .iter()
        .map(|(name, _)| {
            let all = records(&together.join(format!("{name}.jsonl")));
            all[all.len() - 2]["evaluation"]["energy"].clone()
        })
        .collect();
    assert!(ends[0] != ends[1] && ends[1] != ends[2] && ends[2] != ends[3]);
    let pressed = records(&together.join("pressed_1bar.jsonl"));
    assert_eq!(pressed[0]["schema"], "omnimd-mlmd-jsonl-v4");
    assert_eq!(pressed.last().unwrap()["barostat_moves"], 10);

    // In two halves: 24 steps, then 16 more from the checkpoints.
    let first = run_replicas("first", &request(24, &fresh));
    let resume = |index: usize| {
        let name = replicas[index].0;
        json!({"restart": first.join(format!("{name}.jsonl.checkpoint.json"))})
    };
    let mut resumed = request(16, &resume);
    resumed["checkpoint_every"] = json!(8);
    let second = run_replicas("second", &resumed);
    for (name, _) in &replicas {
        let whole = records(&together.join(format!("{name}.jsonl")));
        let resumed = records(&second.join(format!("{name}.jsonl")));
        assert_eq!(resumed[0]["start_step"], 24, "{name}");
        // Checkpoints on the way are named by the absolute step.
        let between = second.join(format!("{name}.jsonl.step-32.checkpoint.json"));
        assert_eq!(read(&between)["step"], 32, "{name}");
        // The last frame and what the completed record says of the run.
        assert_eq!(resumed[resumed.len() - 2], whole[whole.len() - 2], "{name}");
        for key in [
            "final_step",
            "thermostat_heat_ev",
            "barostat_moves",
            "final_total_energy_ev",
        ] {
            assert_eq!(
                resumed.last().unwrap()[key],
                whole.last().unwrap()[key],
                "{name}: {key}"
            );
        }
    }

    // Steps that are not multiples of the intervals: the last step is
    // always sampled, and has the final checkpoint only.
    let mut odd = request(6, &fresh);
    odd["sample_every"] = json!(4);
    odd["checkpoint_every"] = json!(3);
    let odd = run_replicas("odd", &odd);
    for (name, _) in &replicas {
        let all = records(&odd.join(format!("{name}.jsonl")));
        let steps: Vec<u64> = all
            .iter()
            .filter(|record| record["kind"] == "frame")
            .map(|record| record["step"].as_u64().unwrap())
            .collect();
        assert_eq!(steps, [0, 4, 6], "{name}");
        assert_eq!(all.last().unwrap()["final_step"], 6);
        let exists = |suffix: &str| odd.join(format!("{name}.jsonl{suffix}")).exists();
        assert!(
            exists(".step-3.checkpoint.json") && exists(".checkpoint.json"),
            "{name}"
        );
        assert!(!exists(".step-6.checkpoint.json"), "{name}");
        assert_eq!(
            read(&odd.join(format!("{name}.jsonl.checkpoint.json")))["step"],
            6
        );
    }
    assert!(
        read(&odd.join("replicas.json"))["stepping_seconds"]
            .as_f64()
            .unwrap()
            > 0.
    );

    // What is refused, before any directory is made.
    let refused = |name: &str, request: Value, why: &str| {
        let input = directory.join(format!("{name}.json"));
        std::fs::write(&input, request.to_string()).unwrap();
        let output = directory.join(name);
        let run = Command::new(env!("CARGO_BIN_EXE_omnimd-mlmd"))
            .arg("run-replicas")
            .arg(&input)
            .arg(&output)
            .output()
            .unwrap();
        let error = String::from_utf8_lossy(&run.stderr).into_owned();
        assert!(!run.status.success(), "{name} should be refused");
        assert!(error.contains(why), "{name}: {error}");
        assert!(!output.exists(), "{name}: no output for a refused request");
    };
    let good = || request(8, &fresh);
    let renamed = |index: usize, name: &str| {
        let mut request = good();
        request["replicas"][index]["name"] = json!(name);
        request
    };
    refused(
        "twice",
        renamed(1, "PLAIN"),
        "distinct without regard to case",
    );
    refused("slash", renamed(0, "a/b"), "characters of");
    refused("dot", renamed(0, ".hidden"), "not starting with a dot");
    refused("empty-name", renamed(0, ""), "names are");
    let mut none = good();
    none["replicas"] = json!([]);
    refused("none", none, "1..=1024 replicas");
    let mut schema = good();
    schema["schema"] = json!("omnimd-replicas-v0");
    refused("schema", schema, "unsupported replicas request schema");
    let mut sampled = good();
    sampled["sample_every"] = json!(0);
    refused("sampling", sampled, "must be positive");
    let mut parameters = good();
    parameters["replicas"][0]["fixed_parameters"] = json!({});
    refused("parameters", parameters, "unknown field");
    let mut massless = good();
    massless["replicas"][2]
        .as_object_mut()
        .unwrap()
        .remove("masses");
    refused(
        "massless",
        massless,
        "replica pressed_1bar: fresh run requires masses",
    );
    let mut both = good();
    both["replicas"][3]["restart"] = json!(first.join("open-3.jsonl.checkpoint.json"));
    refused("both", both, "replica open-3: restart cannot be combined");
    // An existing directory is never written into.
    let input = directory.join("together.json");
    let again = Command::new(env!("CARGO_BIN_EXE_omnimd-mlmd"))
        .arg("run-replicas")
        .arg(&input)
        .arg(&together)
        .output()
        .unwrap();
    assert!(!again.status.success());
    assert!(String::from_utf8_lossy(&again.stderr).contains("already exists"));
    std::fs::remove_dir_all(directory).unwrap();
}
