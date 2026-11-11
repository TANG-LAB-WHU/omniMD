//! Model deviation along a run: the sampled frames carry what the members of
//! the model's ensemble say of them. On the CPU that is exactly what
//! `model-deviation` says of the imported trajectory afterwards.
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
    let directory = std::env::temp_dir().join(format!(
        "omnimd-run-deviation-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

/// The synthetic model with every fitting bias shifted, written to
/// `directory`: another member of its ensemble.
fn member(directory: &Path, name: &str, shift: f64) -> PathBuf {
    let mut model: Value =
        serde_json::from_str(&std::fs::read_to_string(fixture("synthetic-v1.json")).unwrap())
            .unwrap();
    for network in model["fitting"].as_array_mut().unwrap() {
        for layer in network["layers"].as_array_mut().unwrap() {
            for bias in layer["bias"].as_array_mut().unwrap() {
                *bias = json!(bias.as_f64().unwrap() + shift);
            }
        }
    }
    let path = directory.join(name);
    std::fs::write(&path, model.to_string()).unwrap();
    path
}

fn water(variant: usize) -> Value {
    let shift = 0.03 * variant as f64;
    json!({
        "positions": [[0.2 + shift, 0.3, 0.4], [1.1, 0.5 - shift, 0.7], [0.6, 1.4, 0.9 + shift],
                      [2.4 - shift, 2.0, 2.2]],
        "species": ["O", "H", "H", "O"],
        "cell": [[3.7, 0.0, 0.0], [1.2, 3.4, 0.0], [0.6, 0.5, 3.8]],
        "units": "EvAngstrom"
    })
}

/// What a `run` request says of a system: NVT from a seeded start.
fn system(variant: usize) -> Value {
    json!({
        "frame": water(variant),
        "masses": [15.999, 1.008, 1.008, 15.999],
        "initialize_velocities": {"temperature_kelvin": 600.0, "seed": 7 + variant,
            "rescale_to_temperature": true},
        "remove_center_of_mass": true,
        "ensemble": {"kind": "nvt_csvr", "temperature_kelvin": 600.0, "tau_fs": 20.0,
            "seed": 4 + variant}
    })
}

fn merged(mut base: Value, more: &Value) -> Value {
    for (key, value) in more.as_object().unwrap() {
        base[key] = value.clone();
    }
    base
}

/// Runs an engine command; its records (of the file, or of `NAME.jsonl`
/// files for a directory are read by the caller) or what it said when it
/// failed.
fn engine(command: &str, directory: &Path, name: &str, request: &Value) -> Result<PathBuf, String> {
    let input = directory.join(format!("{name}.json"));
    let output = directory.join(match command {
        "run-replicas" | "select-dataset" => name.to_string(),
        _ => format!("{name}.jsonl"),
    });
    std::fs::write(&input, request.to_string()).unwrap();
    let run = Command::new(env!("CARGO_BIN_EXE_omnimd-mlmd"))
        .arg(command)
        .arg(&input)
        .arg(&output)
        .output()
        .unwrap();
    match run.status.success() {
        true => Ok(output),
        false => Err(String::from_utf8_lossy(&run.stderr).into_owned()),
    }
}

fn records(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn frames(records: &[Value]) -> Vec<Value> {
    records
        .iter()
        .filter(|record| record["kind"] == "frame")
        .cloned()
        .collect()
}

/// The largest force deviation of each frame of a trajectory.
fn largest(frames: &[Value]) -> Vec<f64> {
    frames
        .iter()
        .map(|frame| {
            frame["model_deviation"]["force_ev_per_angstrom"]["max"]
                .as_f64()
                .unwrap()
        })
        .collect()
}

#[test]
fn a_run_with_deviation_marks_its_frames_as_model_deviation_does() {
    let directory = work_directory("run");
    let first = fixture("synthetic-v1.json");
    let others = [
        member(&directory, "second.json", 0.05),
        member(&directory, "third.json", -0.08),
    ];
    let run = |steps: u64, more: Value| {
        merged(
            merged(
                json!({"options": {"native_model": first}, "timestep_fs": 0.5, "steps": steps,
                    "sample_every": 5}),
                &system(0),
            ),
            &more,
        )
    };
    let deviation = |window: Option<(f64, f64)>| {
        let mut section = json!({"models": [{"native_model": others[0]},
            {"native_model": others[1]}]});
        if let Some((low, high)) = window {
            section["force_window"] =
                json!({"min_ev_per_angstrom": low, "max_ev_per_angstrom": high});
        }
        json!({"deviation": section})
    };
    // Without a window: the statistics, and no candidates.
    let plain = records(&engine("run", &directory, "plain", &run(40, json!({}))).unwrap());
    let open = records(&engine("run", &directory, "open", &run(40, deviation(None))).unwrap());
    let sampled = frames(&open);
    assert_eq!(sampled.len(), 9, "steps 0, 5, ..., 40");
    assert!(sampled.iter().all(|frame| frame["candidate"].is_null()));
    let spread = largest(&sampled);
    assert!(
        spread.iter().all(|&d| d > 1e-6),
        "the members differ: {spread:?}"
    );
    let summary = &open.last().unwrap()["deviation"];
    assert_eq!(summary["frames"], 9);
    assert!(summary["candidate_frames"].is_null() && summary["accurate_frames"].is_null());
    let top = spread.iter().copied().fold(0.0, f64::max);
    assert_eq!(summary["max_force_deviation_ev_per_angstrom"], json!(top));
    let header = &open[0]["deviation"];
    assert_eq!(header["evaluation"], "member by member, frame by frame");
    assert_eq!(header["models"].as_array().unwrap().len(), 3);
    assert_eq!(header["models"][0]["model"], json!(first));
    assert_eq!(header["models"][0]["model_sha256"], open[0]["model_sha256"]);
    assert_ne!(
        header["models"][1]["model_sha256"],
        header["models"][2]["model_sha256"]
    );

    // The members only watch: the run is the run without them.
    assert_eq!(plain.len(), open.len());
    for (with, without) in sampled.iter().zip(&frames(&plain)) {
        let mut with = with.clone();
        let fields = with.as_object_mut().unwrap();
        assert!(fields.remove("model_deviation").is_some() && fields.remove("candidate").is_some());
        assert_eq!(&with, without);
        assert!(without.get("model_deviation").is_none() && without.get("candidate").is_none());
    }
    assert!(
        plain[0].get("deviation").is_none() && plain.last().unwrap().get("deviation").is_none()
    );

    // A window inside the spread of the frames, so that some are below it,
    // some in it and some above.
    let mut sorted = spread.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(sorted[1] < sorted[2] && sorted[5] < sorted[6], "{sorted:?}");
    let window = (0.5 * (sorted[1] + sorted[2]), 0.5 * (sorted[5] + sorted[6]));
    let marked = records(
        &engine(
            "run",
            &directory,
            "marked",
            &run(40, deviation(Some(window))),
        )
        .unwrap(),
    );
    let sampled = frames(&marked);
    assert_eq!(largest(&sampled), spread);
    let flags: Vec<bool> = sampled
        .iter()
        .map(|frame| frame["candidate"].as_bool().unwrap())
        .collect();
    let expected: Vec<bool> = spread
        .iter()
        .map(|&d| window.0 <= d && d < window.1)
        .collect();
    assert_eq!(flags, expected);
    let summary = &marked.last().unwrap()["deviation"];
    assert_eq!(
        (
            &summary["accurate_frames"],
            &summary["candidate_frames"],
            &summary["failed_frames"]
        ),
        (&json!(2), &json!(4), &json!(3))
    );
    assert_eq!(
        marked[0]["deviation"]["force_window"]["min_ev_per_angstrom"],
        json!(window.0)
    );

    // The same from the finished trajectory: import it, compare the members.
    let trajectory = directory.join("marked.jsonl");
    let import = |name: &str, more: Value| {
        engine(
            "import-trajectory",
            &directory,
            name,
            &merged(
                json!({"schema": "omnimd-trajectory-import-v1", "trajectory": trajectory,
                    "id_prefix": "water"}),
                &more,
            ),
        )
    };
    let dataset = import("all", json!({})).unwrap();
    let members: Vec<Value> = std::iter::once(&first)
        .chain(&others)
        .enumerate()
        .map(
            |(index, model)| json!({"id": format!("m{index}"), "options": {"native_model": model}}),
        )
        .collect();
    let afterwards = records(
        &engine(
            "model-deviation",
            &directory,
            "afterwards",
            &json!({"schema": "omnimd-model-deviation-v1", "models": members, "dataset": dataset,
                "force_window": {"min_ev_per_angstrom": window.0, "max_ev_per_angstrom": window.1}}),
        )
        .unwrap(),
    );
    let compared: Vec<&Value> = afterwards
        .iter()
        .filter(|record| record["kind"] == "deviation")
        .collect();
    assert_eq!(compared.len(), 9);
    for (frame, record) in sampled.iter().zip(&compared) {
        assert_eq!(record["id"], format!("water/step-{}", frame["step"]));
        assert_eq!(
            frame["model_deviation"], record["statistics"],
            "step {}",
            frame["step"]
        );
        assert_eq!(frame["candidate"], record["candidate"]);
    }
    let (done, summary) = (
        afterwards.last().unwrap(),
        &marked.last().unwrap()["deviation"],
    );
    for count in ["accurate_frames", "candidate_frames", "failed_frames"] {
        assert_eq!(done[count], summary[count], "{count}");
    }
    assert_eq!(done["frames"], summary["frames"]);
    assert_eq!(
        done["max_force_deviation_ev_per_angstrom"],
        summary["max_force_deviation_ev_per_angstrom"]
    );

    // Only the candidates, as a dataset to label.
    let chosen = records(&import("candidates", json!({"candidates_only": true})).unwrap());
    let ids: Vec<&str> = chosen
        .iter()
        .filter(|record| record["kind"] == "frame")
        .map(|record| record["data"]["id"].as_str().unwrap())
        .collect();
    let expected: Vec<String> = sampled
        .iter()
        .filter(|frame| frame["candidate"] == true)
        .map(|frame| format!("water/step-{}", frame["step"]))
        .collect();
    assert_eq!(ids, expected);
    assert_eq!(chosen.last().unwrap()["frames"], 4);
    // After the first of them, every second one.
    let later = import(
        "later",
        json!({"candidates_only": true, "stride": 2,
            "first_step": sampled.iter().find(|f| f["candidate"] == true).unwrap()["step"]}),
    );
    assert_eq!(records(&later.unwrap()).last().unwrap()["frames"], 2);
    // Candidates need marks, and at least one of them.
    let unmarked = |name: &str, trajectory: &str| {
        engine(
            "import-trajectory",
            &directory,
            name,
            &json!({"schema": "omnimd-trajectory-import-v1", "id_prefix": "water",
                "trajectory": directory.join(trajectory), "candidates_only": true}),
        )
        .unwrap_err()
    };
    assert!(unmarked("no-window", "open.jsonl").contains("carries no candidate mark"));
    assert!(unmarked("no-members", "plain.jsonl").contains("carries no candidate mark"));
    let none = run(40, deviation(Some((10.0 * top, 20.0 * top))));
    let quiet = records(&engine("run", &directory, "quiet", &none).unwrap());
    assert_eq!(quiet.last().unwrap()["deviation"]["accurate_frames"], 9);
    assert!(unmarked("no-candidates", "quiet.jsonl").contains("marked no candidate frame"));

    // In two halves: the members are named again, the frames are the same.
    engine("run", &directory, "half", &run(25, deviation(Some(window)))).unwrap();
    let resumed = records(
        &engine(
            "run",
            &directory,
            "rest",
            &merged(
                json!({"options": {"native_model": first}, "timestep_fs": 0.5, "steps": 15,
                    "sample_every": 5,
                    "restart": directory.join("half.jsonl.checkpoint.json")}),
                &deviation(Some(window)),
            ),
        )
        .unwrap(),
    );
    assert_eq!(frames(&resumed)[1..], sampled[6..], "steps 30, 35, 40");
    assert_eq!(resumed.last().unwrap()["deviation"]["frames"], 4);

    // What is refused.
    let refused = |name: &str, section: Value, why: &str| {
        let error = engine(
            "run",
            &directory,
            name,
            &run(5, json!({"deviation": section})),
        )
        .expect_err(name);
        assert!(error.contains(why), "{name}: {error}");
        assert!(!directory.join(format!("{name}.jsonl")).exists());
    };
    refused("alone", json!({"models": []}), "1..=15 models besides");
    let crowd: Vec<Value> = (0..16)
        .map(|_| json!({"native_model": others[0]}))
        .collect();
    refused("crowd", json!({"models": crowd}), "1..=15 models besides");
    refused(
        "window",
        json!({"models": [{"native_model": others[0]}],
            "force_window": {"min_ev_per_angstrom": 0.3, "max_ev_per_angstrom": 0.1}}),
        "force window",
    );
    refused(
        "elsewhere",
        json!({"models": [{"native_model": others[0], "device": "cuda:0", "gpu_runtime": {}}]}),
        "same cuda:N with gpu_runtime",
    );
    refused(
        "missing",
        json!({"models": [{"native_model": directory.join("absent.json")}]}),
        "existing absolute file path",
    );
    refused(
        "unknown",
        json!({"models": [{"native_model": others[0]}], "every": 2}),
        "unknown field",
    );
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn replicas_with_deviation_carry_what_each_run_carries_alone() {
    let directory = work_directory("replicas");
    let first = fixture("synthetic-v1.json");
    let other = member(&directory, "second.json", 0.07);
    let shared = json!({"options": {"native_model": first}, "timestep_fs": 0.5, "steps": 20,
        "sample_every": 4,
        "deviation": {"models": [{"native_model": other}],
            "force_window": {"min_ev_per_angstrom": 0.0, "max_ev_per_angstrom": 1e3}}});
    let replicas: Vec<Value> = (0..3)
        .map(|variant| merged(json!({"name": format!("w{variant}")}), &system(variant)))
        .collect();
    let together = engine(
        "run-replicas",
        &directory,
        "together",
        &merged(
            shared.clone(),
            &json!({"schema": "omnimd-replicas-v1", "replicas": replicas}),
        ),
    )
    .unwrap();
    let mut seen = Vec::new();
    for variant in 0..3 {
        let alone = records(
            &engine(
                "run",
                &directory,
                &format!("alone-{variant}"),
                &merged(shared.clone(), &system(variant)),
            )
            .unwrap(),
        );
        let replica = records(&together.join(format!("w{variant}.jsonl")));
        assert_eq!(frames(&replica), frames(&alone), "replica {variant}");
        assert_eq!(frames(&replica).len(), 6);
        assert!(frames(&replica)
            .iter()
            .all(|frame| frame["candidate"] == true));
        assert_eq!(replica[0]["deviation"], alone[0]["deviation"]);
        assert_eq!(
            replica.last().unwrap()["deviation"],
            alone.last().unwrap()["deviation"]
        );
        assert_eq!(replica.last().unwrap()["deviation"]["candidate_frames"], 6);
        seen.push(largest(&frames(&replica)));
    }
    // The replicas are different systems with different deviations.
    assert!(seen[0] != seen[1] && seen[1] != seen[2]);
    std::fs::remove_dir_all(directory).unwrap();
}
