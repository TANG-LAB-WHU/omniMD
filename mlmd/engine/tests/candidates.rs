//! `collect-candidates`: the candidates of runs with model deviation as one
//! dataset, at most so many of a trajectory, and the summary of the
//! exploration.
use omnimd_mlmd::{
    candidates::{collect_candidates, CandidatesRequest, DATASET, REQUEST_SCHEMA, SUMMARY},
    dataset::{self, DatasetReader},
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
        std::env::temp_dir().join(format!("omnimd-candidates-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

fn window(low: f64, high: f64) -> Value {
    json!({"min_ev_per_angstrom": low, "max_ev_per_angstrom": high})
}

/// The geometry of the frame at `step` of a written trajectory.
fn geometry(step: u64) -> Value {
    json!({"positions": [[0.001 * step as f64, 0.0, 0.0], [0.96, 0.0, 0.0]],
        "species": ["O", "H"],
        "cell": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        "units": "EvAngstrom"})
}

/// The records of a trajectory as a run with deviation writes them: a frame
/// every ten steps with these largest force deviations, marked by the window
/// of the run when it has one.
fn trajectory(deviations: &[f64], marked_by: Option<(f64, f64)>) -> Vec<Value> {
    let mut records = vec![json!({"kind": "header", "schema": "omnimd-mlmd-jsonl-v2",
        "ensemble": {"kind": "nvt_csvr", "temperature_kelvin": 300.0, "tau_fs": 20.0, "seed": 1},
        "metadata": {"type_map": ["O", "H"], "cutoff": 3.0, "backend": "written by a test",
            "precision": "float64", "units": "EvAngstrom"},
        "timestep_fs": 0.5, "model_sha256": "ab".repeat(32),
        "deviation": {"models": [], "evaluation": "member by member, frame by frame",
            "force_window": marked_by.map(|(low, high)| window(low, high))}})];
    for (index, &deviation) in deviations.iter().enumerate() {
        let step = 10 * index as u64;
        records.push(
            json!({"kind": "frame", "step": step, "time_fs": 0.5 * step as f64,
            "frame": geometry(step),
            "model_deviation": {"energy_per_atom_std_ev": 0.001,
                "force_ev_per_angstrom": {"min": 0.0, "max": deviation, "mean": 0.5 * deviation}},
            "candidate": marked_by.map(|(low, high)| low <= deviation && deviation < high)}),
        );
    }
    records.push(
        json!({"kind": "completed", "steps": 10 * (deviations.len() - 1),
        "final_step": 10 * (deviations.len() - 1)}),
    );
    records
}

fn write(directory: &Path, name: &str, records: &[Value]) -> PathBuf {
    let path = directory.join(format!("{name}.jsonl"));
    let text: String = records.iter().map(|record| format!("{record}\n")).collect();
    std::fs::write(&path, text).unwrap();
    path
}

/// A request for these trajectories and ID prefixes, with further entries.
fn listed(sources: &[(&Path, &str)], more: Value) -> Value {
    let trajectories: Vec<Value> = sources
        .iter()
        .map(|(path, prefix)| json!({"trajectory": path, "id_prefix": prefix}))
        .collect();
    let mut request = json!({"schema": REQUEST_SCHEMA, "trajectories": trajectories});
    for (key, value) in more.as_object().unwrap() {
        request[key] = value.clone();
    }
    request
}

fn request(sources: &[(&Path, &str)], more: Value) -> CandidatesRequest {
    serde_json::from_value(listed(sources, more)).unwrap()
}

fn records(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

/// The frames of a dataset: ID and geometry.
fn dataset_frames(path: &Path) -> Vec<(String, Value)> {
    let mut reader = DatasetReader::open_hashed(path).unwrap();
    let mut frames = Vec::new();
    while let Some(data) = reader.next_frame().unwrap() {
        assert_eq!(data.index, frames.len() as u64);
        assert!(data.labels.energy.is_none() && data.labels.forces.is_none());
        assert!(data.labels.virial.is_none());
        frames.push((data.id, serde_json::to_value(&data.frame).unwrap()));
    }
    frames
}

/// Ten frames: two below the window 0.1..0.3, six in it, two at or above.
const MIXED: [f64; 10] = [0.05, 0.12, 0.31, 0.15, 0.2, 0.09, 0.25, 0.11, 0.29, 0.3];

#[test]
fn candidates_are_counted_and_picked_evenly_in_time() {
    let directory = work_directory("picked");
    let run_window = Some((0.1, 0.3));
    let mixed = write(&directory, "mixed", &trajectory(&MIXED, run_window));
    let quiet = write(
        &directory,
        "quiet",
        &trajectory(&[0.01, 0.02, 0.03], run_window),
    );
    let rough = write(&directory, "rough", &trajectory(&[0.5, 0.2], run_window));
    let sources = [
        (mixed.as_path(), "mixed"),
        (quiet.as_path(), "quiet"),
        (rough.as_path(), "rough"),
    ];

    // At most three of a trajectory: of the six candidates of the first (at
    // steps 10, 30, 40, 60, 70, 80) the first, the last and the one at the
    // even position next to the middle.
    let output = directory.join("three");
    let summary = collect_candidates(
        &request(&sources, json!({"max_per_trajectory": 3})),
        &output,
    )
    .unwrap();
    let on_disk: Value =
        serde_json::from_str(&std::fs::read_to_string(output.join(SUMMARY)).unwrap()).unwrap();
    let summary = serde_json::to_value(&summary).unwrap();
    assert_eq!(summary, on_disk);
    assert_eq!(summary["schema"], "omnimd-candidates-summary-v1");
    assert_eq!(
        (&summary["first_step"], &summary["max_per_trajectory"]),
        (&json!(0), &json!(3))
    );
    assert!(
        summary["force_window"].is_null(),
        "every trajectory by its own"
    );
    let counts = |of: &Value| {
        [
            "frames",
            "accurate_frames",
            "candidate_frames",
            "failed_frames",
        ]
        .map(|name| of[name].as_u64().unwrap())
    };
    assert_eq!(counts(&summary), [15, 5, 7, 3]);
    assert_eq!(summary["accurate_fraction"], json!(5.0 / 15.0));
    assert_eq!(
        summary["mean_accurate_fraction"],
        json!((0.2 + 1.0 + 0.0) / 3.0)
    );
    assert_eq!(summary["picked_frames"], 4);
    let of = summary["trajectories"].as_array().unwrap();
    assert_eq!(of.len(), 3);
    assert_eq!(counts(&of[0]), [10, 2, 6, 2]);
    assert_eq!(counts(&of[1]), [3, 3, 0, 0]);
    assert_eq!(counts(&of[2]), [2, 0, 1, 1]);
    assert_eq!(
        [
            &of[0]["accurate_fraction"],
            &of[1]["accurate_fraction"],
            &of[2]["accurate_fraction"]
        ],
        [&json!(0.2), &json!(1.0), &json!(0.0)]
    );
    assert_eq!(
        [
            &of[0]["max_force_deviation_ev_per_angstrom"],
            &of[1]["max_force_deviation_ev_per_angstrom"],
            &of[2]["max_force_deviation_ev_per_angstrom"]
        ],
        [&json!(0.31), &json!(0.03), &json!(0.5)]
    );
    assert_eq!(
        of[0]["picked"],
        json!([
            {"id": "mixed/step-10", "step": 10, "max_force_deviation_ev_per_angstrom": 0.12},
            {"id": "mixed/step-40", "step": 40, "max_force_deviation_ev_per_angstrom": 0.2},
            {"id": "mixed/step-80", "step": 80, "max_force_deviation_ev_per_angstrom": 0.29}
        ])
    );
    assert_eq!(of[1]["picked"], json!([]));
    assert_eq!(
        of[2]["picked"],
        json!([{"id": "rough/step-10", "step": 10, "max_force_deviation_ev_per_angstrom": 0.2}])
    );
    // What the summary says of where the frames came from.
    assert_eq!(of[0]["id_prefix"], "mixed");
    assert_eq!(of[0]["trajectory"], json!(mixed));
    assert_eq!(
        of[0]["trajectory_sha256"],
        json!(dataset::sha256(&mixed).unwrap())
    );
    assert_eq!(of[0]["model_sha256"], json!("ab".repeat(32)));
    assert_eq!(of[0]["ensemble"]["kind"], "nvt_csvr");
    assert_eq!(of[0]["force_window"], window(0.1, 0.3));
    // The dataset: those frames in the order of the request, then of time.
    assert_eq!(summary["dataset"], DATASET);
    let dataset_path = output.join(DATASET);
    assert_eq!(
        summary["dataset_sha256"],
        json!(dataset::sha256(&dataset_path).unwrap())
    );
    assert_eq!(
        dataset_frames(&dataset_path),
        [
            ("mixed/step-10".to_string(), geometry(10)),
            ("mixed/step-40".to_string(), geometry(40)),
            ("mixed/step-80".to_string(), geometry(80)),
            ("rough/step-10".to_string(), geometry(10)),
        ]
    );
    let written = records(&dataset_path);
    assert_eq!(written[0]["header"]["type_map"], json!(["O", "H"]));
    // The trajectories that gave frames are the sources of the dataset.
    assert_eq!(
        written.last().unwrap()["source_sha256"],
        json!({"mixed.jsonl": dataset::sha256(&mixed).unwrap(),
            "rough.jsonl": dataset::sha256(&rough).unwrap()})
    );

    // Without a most: every candidate.
    let output = directory.join("all");
    let summary = collect_candidates(&request(&sources, json!({})), &output).unwrap();
    assert_eq!(summary.picked_frames, 7);
    assert!(summary.max_per_trajectory.is_none());
    let ids: Vec<String> = dataset_frames(&output.join(DATASET))
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(
        ids,
        [10, 30, 40, 60, 70, 80]
            .map(|step| format!("mixed/step-{step}"))
            .into_iter()
            .chain(["rough/step-10".to_string()])
            .collect::<Vec<_>>()
    );

    // One of a trajectory: its first candidate. Two: the first and the last.
    for (most, steps) in [
        (1, vec![10]),
        (2, vec![10, 80]),
        (5, vec![10, 30, 40, 70, 80]),
    ] {
        let output = directory.join(format!("most-{most}"));
        let summary = collect_candidates(
            &request(&sources[..1], json!({"max_per_trajectory": most})),
            &output,
        )
        .unwrap();
        let picked: Vec<u64> = summary.trajectories[0]
            .picked
            .iter()
            .map(|pick| pick.step)
            .collect();
        assert_eq!(picked, steps, "at most {most}");
    }

    // From step 35 on, the frames before it are neither counted nor picked.
    let output = directory.join("later");
    let summary =
        collect_candidates(&request(&sources[..1], json!({"first_step": 35})), &output).unwrap();
    let later = &summary.trajectories[0];
    assert_eq!(
        (
            later.frames,
            later.accurate_frames,
            later.candidate_frames,
            later.failed_frames
        ),
        (6, 1, 4, 1)
    );
    assert_eq!(later.max_force_deviation_ev_per_angstrom, 0.3);
    let steps: Vec<u64> = later.picked.iter().map(|pick| pick.step).collect();
    assert_eq!(steps, [40, 60, 70, 80]);

    // A window of the request judges the frames anew, whatever the runs
    // marked: 0.2..0.31 has the frames of steps 40, 60, 80 and 90 inside.
    let unmarked = write(&directory, "unmarked", &trajectory(&MIXED, None));
    for (name, path) in [("anew", &mixed), ("unmarked", &unmarked)] {
        let output = directory.join(format!("window-{name}"));
        let summary = collect_candidates(
            &request(
                &[(path.as_path(), "run")],
                json!({"force_window": window(0.2, 0.31)}),
            ),
            &output,
        )
        .unwrap();
        let judged = &summary.trajectories[0];
        assert_eq!(
            (
                judged.accurate_frames,
                judged.candidate_frames,
                judged.failed_frames
            ),
            (5, 4, 1),
            "{name}"
        );
        let steps: Vec<u64> = judged.picked.iter().map(|pick| pick.step).collect();
        assert_eq!(steps, [40, 60, 80, 90], "{name}");
        let summary = serde_json::to_value(&summary).unwrap();
        assert_eq!(summary["force_window"], window(0.2, 0.31));
        assert_eq!(
            summary["trajectories"][0]["force_window"],
            window(0.2, 0.31)
        );
    }
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn an_exploration_without_candidates_is_a_result() {
    let directory = work_directory("none");
    let quiet = write(
        &directory,
        "quiet",
        &trajectory(&[0.01, 0.02, 0.03], Some((0.1, 0.3))),
    );
    let output = directory.join("collected");
    let summary = collect_candidates(
        &request(
            &[(quiet.as_path(), "quiet")],
            json!({"max_per_trajectory": 30}),
        ),
        &output,
    )
    .unwrap();
    assert_eq!(
        (
            summary.frames,
            summary.accurate_frames,
            summary.picked_frames
        ),
        (3, 3, 0)
    );
    assert_eq!(
        (summary.accurate_fraction, summary.mean_accurate_fraction),
        (1.0, 1.0)
    );
    assert!(summary.dataset.is_none() && summary.dataset_sha256.is_none());
    // The summary alone: a dataset holds at least one frame.
    let files: Vec<String> = std::fs::read_dir(&output)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    assert_eq!(files, [SUMMARY]);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn what_cannot_be_collected_is_refused_before_anything_is_written() {
    let directory = work_directory("refused");
    let run_window = Some((0.1, 0.3));
    let good = write(&directory, "good", &trajectory(&MIXED, run_window));
    let refused = |name: &str, request: Value, output: &Path, why: &str| {
        let request: CandidatesRequest = serde_json::from_value(request).unwrap();
        let error = collect_candidates(&request, output)
            .expect_err(name)
            .to_string();
        assert!(error.contains(why), "{name}: {error}");
    };
    let fresh = directory.join("fresh");
    let one = |path: &Path| listed(&[(path, "run")], json!({}));

    // The request.
    let mut other_schema = one(&good);
    other_schema["schema"] = json!("omnimd-candidates-v0");
    refused(
        "schema",
        other_schema,
        &fresh,
        "unsupported candidates request schema",
    );
    refused(
        "at most none",
        listed(&[(&good, "run")], json!({"max_per_trajectory": 0})),
        &fresh,
        "max_per_trajectory must be positive",
    );
    refused(
        "window",
        listed(&[(&good, "run")], json!({"force_window": window(0.3, 0.1)})),
        &fresh,
        "force window",
    );
    refused(
        "nothing listed",
        json!({"schema": REQUEST_SCHEMA}),
        &fresh,
        "1..=4096 trajectories",
    );
    refused(
        "relative trajectory",
        one(Path::new("good.jsonl")),
        &fresh,
        "trajectory paths must be absolute",
    );
    refused(
        "relative run",
        json!({"schema": REQUEST_SCHEMA, "replica_runs": ["runs"]}),
        &fresh,
        "replica run directories must be absolute",
    );
    refused(
        "no replica run",
        json!({"schema": REQUEST_SCHEMA, "replica_runs": [directory.join("absent")]}),
        &fresh,
        "replicas.json",
    );
    let foreign = directory.join("foreign");
    std::fs::create_dir(&foreign).unwrap();
    std::fs::write(
        foreign.join("replicas.json"),
        json!({"schema": "omnimd-replicas-run-v0", "replicas": [{"name": "good"}]}).to_string(),
    )
    .unwrap();
    refused(
        "another record",
        json!({"schema": REQUEST_SCHEMA, "replica_runs": [foreign]}),
        &fresh,
        "not the record of a replica run",
    );
    refused(
        "prefix",
        listed(&[(&good, "has/slash")], json!({})),
        &fresh,
        "prefixes are 1..=128 characters",
    );
    let copy = write(&directory, "copy", &trajectory(&MIXED, run_window));
    refused(
        "one prefix twice",
        listed(&[(&good, "run"), (&copy, "run")], json!({})),
        &fresh,
        "occurs twice",
    );
    refused(
        "one trajectory twice",
        listed(&[(&good, "a"), (&good, "b")], json!({})),
        &fresh,
        "is listed twice",
    );
    refused(
        "relative output",
        one(&good),
        Path::new("fresh"),
        "new absolute output directory",
    );
    refused(
        "output exists",
        one(&good),
        &directory,
        "new absolute output directory",
    );
    assert!(serde_json::from_value::<CandidatesRequest>(
        json!({"schema": REQUEST_SCHEMA, "trajectories": [], "stride": 2})
    )
    .is_err());

    // The trajectories. Each is named in what is said of it.
    let broken = |name: &str, change: &dyn Fn(&mut Vec<Value>), why: &str| {
        let mut records = trajectory(&MIXED, run_window);
        change(&mut records);
        let path = write(&directory, name, &records);
        let request: CandidatesRequest = serde_json::from_value(one(&path)).unwrap();
        let error = collect_candidates(&request, &fresh)
            .expect_err(name)
            .to_string();
        assert!(error.contains(why), "{name}: {error}");
        assert!(
            error.contains(&format!("{name}.jsonl")),
            "{name} is named: {error}"
        );
    };
    broken(
        "without-members",
        &|records| {
            records[0].as_object_mut().unwrap().remove("deviation");
        },
        "made without deviation",
    );
    broken(
        "without-window",
        &|records| records[0]["deviation"]["force_window"] = Value::Null,
        "made without a force window",
    );
    broken(
        "bare-frame",
        &|records| {
            records[4]
                .as_object_mut()
                .unwrap()
                .remove("model_deviation");
        },
        "the frame of step 30 carries no model deviation",
    );
    broken(
        "wrong-mark",
        &|records| records[2]["candidate"] = json!(false),
        "the candidate mark of the frame of step 10 contradicts",
    );
    broken(
        "no-mark",
        &|records| records[1]["candidate"] = Value::Null,
        "the candidate mark of the frame of step 0 contradicts",
    );
    broken(
        "negative",
        &|records| records[3]["model_deviation"]["force_ev_per_angstrom"]["max"] = json!(-0.2),
        "not a finite, nonnegative number",
    );
    broken(
        "unfinished",
        &|records| {
            records.pop();
        },
        "no completed record",
    );
    broken(
        "cut-short",
        &|records| {
            let _ = records.remove(10);
        },
        "no completed record",
    );
    broken(
        "back-in-time",
        &|records| records.swap(3, 4),
        "steps must increase",
    );
    broken(
        "one-step-twice",
        &|records| records.insert(3, records[3].clone()),
        "steps must increase",
    );
    broken(
        "window-of-the-run",
        &|records| records[0]["deviation"]["force_window"] = window(0.3, 0.1),
        "force window",
    );
    broken(
        "metadata",
        &|records| records[0]["metadata"]["cutoff"] = json!(-3.0),
        "cutoff must be finite and positive",
    );
    broken(
        "conditioned",
        &|records| records[0]["schema"] = json!("omnimd-mlmd-jsonl-v3"),
        "is not supported",
    );
    broken(
        "foreign-species",
        &|records| records[5]["frame"]["species"] = json!(["O", "C"]),
        "is not in model type map",
    );
    broken(
        "headless",
        &|records| {
            let _ = records.remove(0);
        },
        "must start with its header",
    );
    broken(
        "two-headers",
        &|records| records.insert(3, records[0].clone()),
        "second header",
    );
    broken(
        "goes-on",
        &|records| records.push(records[10].clone()),
        "continues after its completed record",
    );
    // A window that leaves no frame at or after first_step.
    let request: CandidatesRequest =
        serde_json::from_value(listed(&[(&good, "run")], json!({"first_step": 91}))).unwrap();
    let error = collect_candidates(&request, &fresh)
        .unwrap_err()
        .to_string();
    assert!(error.contains("no trajectory frame at or after first_step"));

    // Trajectories of other elements do not go into one dataset.
    let mut other = trajectory(&MIXED, run_window);
    other[0]["metadata"]["type_map"] = json!(["H", "O"]);
    let other = write(&directory, "other-types", &other);
    refused(
        "other types",
        listed(&[(&good, "a"), (&other, "b")], json!({})),
        &fresh,
        "type map and units must equal those of the first trajectory",
    );
    assert!(!fresh.exists(), "nothing was written");
    std::fs::remove_dir_all(directory).unwrap();
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

/// A replica of the run: water at 600 K from a seeded start.
fn replica(variant: usize) -> Value {
    let shift = 0.03 * variant as f64;
    json!({
        "name": format!("w{variant}"),
        "frame": {
            "positions": [[0.2 + shift, 0.3, 0.4], [1.1, 0.5 - shift, 0.7],
                          [0.6, 1.4, 0.9 + shift], [2.4 - shift, 2.0, 2.2]],
            "species": ["O", "H", "H", "O"],
            "cell": [[3.7, 0.0, 0.0], [1.2, 3.4, 0.0], [0.6, 0.5, 3.8]],
            "units": "EvAngstrom"
        },
        "masses": [15.999, 1.008, 1.008, 15.999],
        "initialize_velocities": {"temperature_kelvin": 600.0, "seed": 7 + variant,
            "rescale_to_temperature": true},
        "remove_center_of_mass": true,
        "ensemble": {"kind": "nvt_csvr", "temperature_kelvin": 600.0, "tau_fs": 20.0,
            "seed": 4 + variant}
    })
}

/// Runs an engine command that writes to a new path; what it printed, or
/// what it said when it failed.
fn engine(command: &str, input: &Path, request: &Value, output: &Path) -> Result<String, String> {
    std::fs::write(input, request.to_string()).unwrap();
    let run = Command::new(env!("CARGO_BIN_EXE_omnimd-mlmd"))
        .arg(command)
        .arg(input)
        .arg(output)
        .output()
        .unwrap();
    match run.status.success() {
        true => Ok(String::from_utf8_lossy(&run.stdout).into_owned()),
        false => Err(String::from_utf8_lossy(&run.stderr).into_owned()),
    }
}

#[test]
fn the_cli_collects_the_candidates_of_a_replica_run() {
    let directory = work_directory("cli");
    let first = fixture("synthetic-v1.json");
    let others = [
        member(&directory, "second.json", 0.05),
        member(&directory, "third.json", -0.08),
    ];
    let explore = |name: &str, force_window: Option<(f64, f64)>| {
        let mut deviation = json!({"models": [{"native_model": others[0]},
            {"native_model": others[1]}]});
        if let Some((low, high)) = force_window {
            deviation["force_window"] = window(low, high);
        }
        let request = json!({"schema": "omnimd-replicas-v1",
            "options": {"native_model": first}, "timestep_fs": 0.5, "steps": 60,
            "sample_every": 5, "deviation": deviation,
            "replicas": (0..3).map(replica).collect::<Vec<_>>()});
        let output = directory.join(name);
        engine(
            "run-replicas",
            &directory.join(format!("{name}.json")),
            &request,
            &output,
        )
        .unwrap();
        output
    };
    let frames = |run: &Path, name: &str| -> Vec<Value> {
        records(&run.join(format!("{name}.jsonl")))
            .into_iter()
            .filter(|record| record["kind"] == "frame")
            .collect()
    };
    let largest = |frame: &Value| {
        frame["model_deviation"]["force_ev_per_angstrom"]["max"]
            .as_f64()
            .unwrap()
    };

    // A first exploration without a window shows how far the members
    // disagree; the window is then laid inside that range, so that frames
    // fall below it, into it and above it.
    let open = explore("open", None);
    let mut seen: Vec<f64> = ["w0", "w1", "w2"]
        .iter()
        .flat_map(|name| frames(&open, name))
        .map(|frame| largest(&frame))
        .collect();
    assert_eq!(seen.len(), 39, "three replicas, steps 0, 5, ..., 60");
    seen.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(seen[9] < seen[10] && seen[29] < seen[30], "{seen:?}");
    let (low, high) = (0.5 * (seen[9] + seen[10]), 0.5 * (seen[29] + seen[30]));
    let marked = explore("marked", Some((low, high)));

    // The candidates of the whole run, at most two of a replica.
    let collected = directory.join("collected");
    let said = engine(
        "collect-candidates",
        &directory.join("collect.json"),
        &json!({"schema": REQUEST_SCHEMA, "replica_runs": [marked], "max_per_trajectory": 2}),
        &collected,
    )
    .unwrap();
    let summary: Value =
        serde_json::from_str(&std::fs::read_to_string(collected.join(SUMMARY)).unwrap()).unwrap();
    assert_eq!(
        (
            &summary["frames"],
            &summary["accurate_frames"],
            &summary["candidate_frames"],
            &summary["failed_frames"]
        ),
        (&json!(39), &json!(10), &json!(20), &json!(9))
    );
    let mut expected = Vec::new();
    for (index, name) in ["w0", "w1", "w2"].iter().enumerate() {
        let of = &summary["trajectories"][index];
        assert_eq!(of["id_prefix"], *name);
        assert_eq!(
            of["trajectory"],
            json!(marked.join(format!("{name}.jsonl")))
        );
        assert_eq!(of["ensemble"]["kind"], "nvt_csvr");
        // The counts of the run itself.
        let sampled = frames(&marked, name);
        let completed = records(&marked.join(format!("{name}.jsonl")))
            .pop()
            .unwrap();
        for count in [
            "frames",
            "accurate_frames",
            "candidate_frames",
            "failed_frames",
            "max_force_deviation_ev_per_angstrom",
        ] {
            assert_eq!(of[count], completed["deviation"][count], "{name} {count}");
        }
        // Its first candidate and its last.
        let candidates: Vec<&Value> = sampled
            .iter()
            .filter(|frame| frame["candidate"] == true)
            .collect();
        assert!(candidates.len() > 2, "{name}: {}", candidates.len());
        let ends = [candidates[0], candidates[candidates.len() - 1]];
        let picked = of["picked"].as_array().unwrap();
        assert_eq!(picked.len(), 2);
        for (pick, frame) in picked.iter().zip(ends) {
            assert_eq!(pick["step"], frame["step"]);
            assert_eq!(
                pick["max_force_deviation_ev_per_angstrom"],
                json!(largest(frame))
            );
            expected.push((
                format!("{name}/step-{}", frame["step"]),
                frame["frame"].clone(),
            ));
        }
    }
    assert_eq!(summary["picked_frames"], 6);
    assert_eq!(dataset_frames(&collected.join(DATASET)), expected);
    assert_eq!(
        said.trim(),
        format!(
            "Picked 6 of 20 candidate frames of 3 trajectories; the models agree on 10 of 39 frames ({:.1}%).",
            100.0 * 10.0 / 39.0
        )
    );

    // The run without a window, judged by the window of the request, gives
    // the same frames: the dynamics do not depend on the window.
    let judged = directory.join("judged");
    engine(
        "collect-candidates",
        &directory.join("judge.json"),
        &json!({"schema": REQUEST_SCHEMA, "replica_runs": [open], "max_per_trajectory": 2,
            "force_window": window(low, high)}),
        &judged,
    )
    .unwrap();
    assert_eq!(dataset_frames(&judged.join(DATASET)), expected);

    // Two runs with replicas of the same names need prefixes of their own.
    let error = engine(
        "collect-candidates",
        &directory.join("both.json"),
        &json!({"schema": REQUEST_SCHEMA, "replica_runs": [marked, open],
            "force_window": window(low, high)}),
        &directory.join("both"),
    )
    .unwrap_err();
    assert!(error.contains("occurs twice"), "{error}");
    let both = directory.join("renamed");
    engine(
        "collect-candidates",
        &directory.join("renamed.json"),
        &json!({"schema": REQUEST_SCHEMA, "replica_runs": [marked],
            "trajectories": [{"trajectory": open.join("w1.jsonl"), "id_prefix": "again-w1"}],
            "force_window": window(low, high)}),
        &both,
    )
    .unwrap();
    let ids: Vec<String> = dataset_frames(&both.join(DATASET))
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    // The listed trajectory first, then those of the run.
    assert!(ids[0].starts_with("again-w1/step-") && ids.last().unwrap().starts_with("w2/step-"));
    assert_eq!(
        ids.len(),
        20 + frames(&marked, "w1")
            .iter()
            .filter(|f| f["candidate"] == true)
            .count()
    );

    // Nothing is overwritten, and an unfinished run is not collected.
    let error = engine(
        "collect-candidates",
        &directory.join("collect.json"),
        &json!({"schema": REQUEST_SCHEMA, "replica_runs": [marked]}),
        &collected,
    )
    .unwrap_err();
    assert!(error.contains("new absolute output directory"), "{error}");
    std::fs::remove_file(marked.join("replicas.json")).unwrap();
    let error = engine(
        "collect-candidates",
        &directory.join("unfinished.json"),
        &json!({"schema": REQUEST_SCHEMA, "replica_runs": [marked]}),
        &directory.join("unfinished"),
    )
    .unwrap_err();
    assert!(error.contains("replicas.json"), "{error}");
    std::fs::remove_dir_all(directory).unwrap();
}
