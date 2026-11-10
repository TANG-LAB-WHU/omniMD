//! `import-trajectory`: the frames of a finished run as an unlabeled dataset,
//! and the chain run -> import-trajectory -> model-deviation.
use omnimd_mlmd::{
    dataset::DatasetReader,
    trajectory::{import_trajectory, TrajectoryRequest, REQUEST_SCHEMA},
};
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
        std::env::temp_dir().join(format!("omnimd-trajectory-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

fn engine(command: &str, input: &Path, output: &Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_omnimd-mlmd"))
        .arg(command)
        .arg(input)
        .arg(output)
        .output()
        .unwrap()
}

fn records(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

/// A 40-step constant-pressure run of the synthetic model, sampled every 8 steps.
fn run(directory: &Path) -> PathBuf {
    let request = json!({
        "options": {"native_model": fixture("synthetic-v1.json")},
        "frame": {
            "positions": [[0.2, 0.3, 0.4], [1.1, 0.5, 0.7], [0.6, 1.4, 0.9], [2.4, 2.0, 2.2]],
            "species": ["O", "H", "H", "O"],
            "cell": [[3.7, 0.0, 0.0], [1.2, 3.4, 0.0], [0.6, 0.5, 3.8]],
            "units": "EvAngstrom"
        },
        "masses": [15.999, 1.008, 1.008, 15.999],
        "initialize_velocities":
            {"temperature_kelvin": 300.0, "seed": 7, "rescale_to_temperature": true},
        "remove_center_of_mass": true,
        "ensemble": {
            "kind": "npt_csvr_crescale", "temperature_kelvin": 300.0, "tau_fs": 20.0, "seed": 4,
            "pressure_bar": 1.0, "barostat_tau_fs": 200.0, "compressibility_per_bar": 1e-5,
            "barostat_every": 4
        },
        "timestep_fs": 0.5,
        "steps": 40,
        "sample_every": 8
    });
    let (input, output) = (directory.join("run.json"), directory.join("run.jsonl"));
    std::fs::write(&input, request.to_string()).unwrap();
    let result = engine("run", &input, &output);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    output
}

/// Force window (eV/angstrom) of the deviation test, placed inside the
/// spread of the six frames so all three classes can occur.
const WINDOW: (f64, f64) = (1.25, 1.42);

fn request(trajectory: &Path, first_step: u64, stride: u64) -> TrajectoryRequest {
    serde_json::from_value(json!({
        "schema": REQUEST_SCHEMA, "trajectory": trajectory, "id_prefix": "water-npt",
        "first_step": first_step, "stride": stride
    }))
    .unwrap()
}

#[test]
fn a_finished_run_becomes_an_unlabeled_dataset() {
    let directory = work_directory("import");
    let trajectory = run(&directory);
    let source: Vec<Value> = records(&trajectory)
        .into_iter()
        .filter(|r| r["kind"] == "frame")
        .collect();
    assert_eq!(source.len(), 6);

    let all = directory.join("all.jsonl");
    assert_eq!(
        import_trajectory(&request(&trajectory, 0, 1), &all).unwrap(),
        6
    );
    let mut reader = DatasetReader::open_hashed(&all).unwrap();
    assert_eq!(reader.header.type_map, ["O", "H"]);
    for (index, expected) in source.iter().enumerate() {
        let data = reader.next_frame().unwrap().unwrap();
        assert_eq!(data.index, index as u64);
        assert_eq!(data.id, format!("water-npt/step-{}", expected["step"]));
        // The geometry of the run, cell included, and no labels.
        assert_eq!(
            serde_json::to_value(&data.frame).unwrap(),
            expected["frame"]
        );
        assert!(data.labels.energy.is_none() && data.labels.forces.is_none());
        assert!(data.labels.virial.is_none());
    }
    assert!(reader.next_frame().unwrap().is_none());
    let provenance = reader.completed_provenance().unwrap();
    assert_eq!(provenance.declared_source_sha256.len(), 1);

    // Steps 16, 32: from step 16 on, every second sampled frame.
    let some = directory.join("some.jsonl");
    assert_eq!(
        import_trajectory(&request(&trajectory, 16, 2), &some).unwrap(),
        2
    );
    let ids: Vec<String> = records(&some)
        .iter()
        .filter(|r| r["kind"] == "frame")
        .map(|r| r["data"]["id"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(ids, ["water-npt/step-16", "water-npt/step-32"]);

    // Nothing is overwritten, and bad requests are refused.
    assert!(import_trajectory(&request(&trajectory, 0, 1), &all).is_err());
    let fresh = |name: &str| directory.join(name);
    assert!(import_trajectory(&request(&trajectory, 0, 0), &fresh("a.jsonl")).is_err());
    assert!(import_trajectory(&request(&trajectory, 41, 1), &fresh("b.jsonl")).is_err());
    let mut bad = request(&trajectory, 0, 1);
    bad.id_prefix = "has/slash".into();
    assert!(import_trajectory(&bad, &fresh("c.jsonl")).is_err());
    bad = request(Path::new("relative.jsonl"), 0, 1);
    assert!(import_trajectory(&bad, &fresh("d.jsonl")).is_err());

    // A run without its completed record is not imported.
    let text = std::fs::read_to_string(&trajectory).unwrap();
    let cut = directory.join("cut.jsonl");
    let lines: Vec<&str> = text.lines().collect();
    let mut partial = lines[..lines.len() - 1].join("\n");
    partial.push('\n');
    std::fs::write(&cut, partial).unwrap();
    let error = import_trajectory(&request(&cut, 0, 1), &fresh("e.jsonl")).unwrap_err();
    assert!(error.to_string().contains("completed"), "{error}");
    // The unfinished output is not a dataset.
    let unfinished = DatasetReader::open_hashed(&fresh("e.jsonl")).and_then(|mut reader| {
        while reader.next_frame()?.is_some() {}
        reader.completed_provenance().map(|_| ())
    });
    assert!(unfinished.is_err());
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn the_cli_chains_run_import_and_model_deviation() {
    let directory = work_directory("chain");
    let trajectory = run(&directory);
    let (import, dataset) = (
        directory.join("import.json"),
        directory.join("explored.jsonl"),
    );
    std::fs::write(
        &import,
        json!({"schema": REQUEST_SCHEMA, "trajectory": trajectory, "id_prefix": "water-npt"})
            .to_string(),
    )
    .unwrap();
    let result = engine("import-trajectory", &import, &dataset);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );

    let (deviation, output) = (
        directory.join("deviation.json"),
        directory.join("deviation.jsonl"),
    );
    std::fs::write(
        &deviation,
        json!({
            "schema": "omnimd-model-deviation-v1", "dataset": dataset,
            "models": [
                {"id": "a", "options": {"native_model": fixture("synthetic-v1.json")}},
                {"id": "b", "options": {"native_model": fixture("full-tanh-v1.json")}}
            ],
            "force_window": {"min_ev_per_angstrom": WINDOW.0, "max_ev_per_angstrom": WINDOW.1}
        })
        .to_string(),
    )
    .unwrap();
    let result = engine("model-deviation", &deviation, &output);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let lines = records(&output);
    let frames: Vec<&Value> = lines.iter().filter(|r| r["kind"] == "deviation").collect();
    assert_eq!(frames.len(), 6);
    assert_eq!(frames[5]["id"], "water-npt/step-40");
    assert!(frames
        .iter()
        .all(|r| r["statistics"]["force_ev_per_angstrom"]["max"]
            .as_f64()
            .unwrap()
            > 0.0));
    // The window splits the frames into accurate, candidate and failed.
    let completed = lines.last().unwrap();
    assert_eq!(completed["frames"], 6);
    let count = |low: f64, high: f64| {
        frames
            .iter()
            .filter(|r| {
                let max = r["statistics"]["force_ev_per_angstrom"]["max"]
                    .as_f64()
                    .unwrap();
                low <= max && max < high
            })
            .count() as u64
    };
    assert_eq!(
        (count(0.0, WINDOW.0), count(WINDOW.0, WINDOW.1)),
        (1, 4),
        "the window should split the frames"
    );
    assert_eq!(completed["accurate_frames"], count(0.0, WINDOW.0));
    assert_eq!(completed["candidate_frames"], count(WINDOW.0, WINDOW.1));
    assert_eq!(completed["failed_frames"], count(WINDOW.1, f64::INFINITY));
    let candidates = frames.iter().filter(|r| r["candidate"] == true).count() as u64;
    assert_eq!(completed["candidate_frames"], candidates);
    std::fs::remove_dir_all(directory).unwrap();
}
