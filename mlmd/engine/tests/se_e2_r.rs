//! An se_e2_r model behind the engine CLI: the commands that take a native
//! model take it as they take an se_e2_a one.
use serde_json::{json, Value};
use std::{
    path::{Path, PathBuf},
    process::Command,
};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../native/fixtures")
        .join(name)
        .canonicalize()
        .unwrap()
}

fn work_directory(name: &str) -> PathBuf {
    let directory =
        std::env::temp_dir().join(format!("omnimd-se-e2-r-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

fn engine(command: &str, directory: &Path, name: &str, request: &Value) -> Vec<Value> {
    let input = directory.join(format!("{name}.json"));
    let output = directory.join(format!("{name}.jsonl"));
    std::fs::write(&input, request.to_string()).unwrap();
    let run = Command::new(env!("CARGO_BIN_EXE_omnimd-mlmd"))
        .arg(command)
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

/// The periodic frame of the DeePMD reference and the energy DeePMD-kit
/// gives for it.
fn reference() -> (Value, f64) {
    let reference: Value = serde_json::from_str(
        &std::fs::read_to_string(fixture("se-r-deepmd-reference.json")).unwrap(),
    )
    .unwrap();
    let case = &reference["cases"][0];
    assert!(case["frame"]["cell"].is_array());
    (case["frame"].clone(), case["energy"].as_f64().unwrap())
}

fn masses(frame: &Value) -> Vec<f64> {
    frame["species"]
        .as_array()
        .unwrap()
        .iter()
        .map(|name| match name.as_str().unwrap() {
            "O" => 15.999,
            "H" => 1.008,
            other => panic!("{other}"),
        })
        .collect()
}

#[test]
fn nve_with_an_se_e2_r_model_conserves_its_energy() {
    let directory = work_directory("nve");
    let (frame, energy) = reference();
    // 100 fs in steps of `timestep_fs`, five frames.
    let request = |timestep_fs: f64| {
        let steps = (100.0 / timestep_fs).round() as u64;
        json!({
            "options": {"native_model": fixture("se-r-model.native.json")},
            "masses": masses(&frame),
            "frame": frame,
            "initialize_velocities":
                {"temperature_kelvin": 300.0, "seed": 11, "rescale_to_temperature": true},
            "remove_center_of_mass": true,
            "ensemble": {"kind": "nve"},
            "timestep_fs": timestep_fs,
            "steps": steps,
            "sample_every": steps / 4
        })
    };
    let records = engine("run", &directory, "nve", &request(0.5));
    assert_eq!(records[0]["metadata"]["backend"], "native-se-e2-r-v1");
    let frames: Vec<&Value> = records.iter().filter(|r| r["kind"] == "frame").collect();
    assert_eq!(frames.len(), 5, "steps 0, 50, ..., 200");
    assert_eq!(frames[4]["step"], 200);
    // The first frame is the reference frame: DeePMD-kit's energy for it.
    let first = frames[0]["evaluation"]["energy"].as_f64().unwrap();
    assert!((first - energy).abs() < 1e-11, "{first} against {energy}");
    // The atoms move, and the total energy stays.
    let moved = frames[4]["frame"]["positions"][0][0].as_f64().unwrap()
        - frames[0]["frame"]["positions"][0][0].as_f64().unwrap();
    assert!(moved.abs() > 1e-3, "{moved}");
    // The integrator is of second order when the forces are the gradient of
    // the energy: half the step, a quarter of the error of the total energy.
    let drift = |records: &[Value]| {
        let completed = records.last().unwrap();
        assert_eq!(completed["kind"], "completed");
        completed["max_absolute_energy_drift_ev"].as_f64().unwrap()
    };
    let (coarse, fine) = (
        drift(&records),
        drift(&engine("run", &directory, "half", &request(0.25))),
    );
    assert!(coarse < 1e-4, "drift {coarse} eV over 100 fs");
    assert!(
        (3.0..5.0).contains(&(coarse / fine)),
        "drift {coarse} eV at 0.5 fs and {fine} eV at 0.25 fs"
    );
    std::fs::remove_dir_all(directory).unwrap();
}

/// The same run on the GPU runtime follows the CPU trajectory.
#[cfg(feature = "native-gpu")]
#[test]
fn nve_with_an_se_e2_r_model_on_the_gpu_runtime_follows_the_cpu() {
    let model = fixture("se-r-model.native.json");
    let loaded = omnimd_native::NativePotential::load(&model).unwrap();
    match omnimd_native::gpu_model::open(loaded.model(), 0, Default::default()) {
        Ok(_) => {}
        Err(error) if std::env::var_os("OMNIMD_REQUIRE_GPU").is_some() => {
            panic!("OMNIMD_REQUIRE_GPU is set but no device: {error}")
        }
        Err(error) => {
            eprintln!("skipping the se_e2_r engine GPU test: {error}");
            return;
        }
    }
    let directory = work_directory("gpu");
    let (frame, _) = reference();
    let request = |options: Value| {
        json!({
            "options": options,
            "masses": masses(&frame),
            "frame": frame,
            "initialize_velocities":
                {"temperature_kelvin": 300.0, "seed": 11, "rescale_to_temperature": true},
            "remove_center_of_mass": true,
            "ensemble": {"kind": "nve"},
            "timestep_fs": 0.5,
            "steps": 60,
            "sample_every": 20
        })
    };
    let samples = |records: &[Value]| -> Vec<Value> {
        records
            .iter()
            .filter(|r| r["kind"] == "frame")
            .cloned()
            .collect()
    };
    let cpu = samples(&engine(
        "run",
        &directory,
        "cpu",
        &request(json!({"native_model": model})),
    ));
    assert_eq!(cpu.len(), 4);
    let gpu = engine(
        "run",
        &directory,
        "gpu",
        &request(json!({"native_model": model, "device": "cuda:0", "gpu_runtime": {}})),
    );
    assert_eq!(gpu[0]["metadata"]["backend"], "native-gpu-se-e2-r");
    for (got, expected) in samples(&gpu).iter().zip(&cpu) {
        let energies = |record: &Value| {
            [&record["evaluation"]["energy"], &record["total_energy_ev"]]
                .map(|value| value.as_f64().unwrap())
        };
        for (ours, theirs) in energies(got).iter().zip(energies(expected)) {
            assert!(
                (ours - theirs).abs() < 1e-10,
                "step {}: energies {ours} and {theirs}",
                expected["step"]
            );
        }
        let positions = |record: &Value| -> Vec<f64> {
            record["frame"]["positions"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|p| p.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()))
                .collect()
        };
        let worst = positions(got)
            .iter()
            .zip(positions(expected))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            worst < 1e-10,
            "step {}: positions differ by {worst}",
            expected["step"]
        );
    }
    // Float32 with tables runs too and says what it is.
    let fast = engine(
        "run",
        &directory,
        "fast",
        &request(json!({"native_model": model, "device": "cuda:0",
            "gpu_runtime": {"precision": "float32", "compress_min_distance": 0.5}})),
    );
    assert_eq!(fast[0]["metadata"]["precision"], "float32");
    let difference = samples(&fast)[3]["evaluation"]["energy"].as_f64().unwrap()
        - cpu[3]["evaluation"]["energy"].as_f64().unwrap();
    assert!(difference.abs() < 1e-4, "{difference}");
    std::fs::remove_dir_all(directory).unwrap();
}
