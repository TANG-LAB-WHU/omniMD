//! The omnimd-gpu runtime behind the engine CLI: NVE dynamics and
//! model-deviation on `cuda:0` must reproduce the CPU backend. Skips without a
//! CUDA device unless OMNIMD_REQUIRE_GPU is set.
#![cfg(feature = "native-gpu")]
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

fn gpu_available() -> bool {
    let model = omnimd_native::NativePotential::load(fixture("synthetic-v1.json")).unwrap();
    match omnimd_native::gpu_model::open(model.model(), 0, Default::default()) {
        Ok(_) => true,
        Err(error) if std::env::var_os("OMNIMD_REQUIRE_GPU").is_some() => {
            panic!("OMNIMD_REQUIRE_GPU is set but no device: {error}")
        }
        Err(error) => {
            eprintln!("skipping engine GPU tests: {error}");
            false
        }
    }
}

/// Runs `omnimd-mlmd COMMAND request output` and returns the output records.
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

fn numbers(value: &Value, out: &mut Vec<f64>) {
    match value {
        Value::Number(n) => out.push(n.as_f64().unwrap()),
        Value::Array(items) => items.iter().for_each(|item| numbers(item, out)),
        Value::Object(map) => map.values().for_each(|item| numbers(item, out)),
        _ => {}
    }
}

/// Largest difference between the numbers of two records of the same shape.
fn difference(a: &Value, b: &Value) -> f64 {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    numbers(a, &mut x);
    numbers(b, &mut y);
    assert_eq!(x.len(), y.len(), "records differ in shape");
    x.iter()
        .zip(&y)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f64::max)
}

fn work_directory(name: &str) -> PathBuf {
    let directory =
        std::env::temp_dir().join(format!("omnimd-gpu-runtime-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

fn water_frame() -> Value {
    json!({
        "positions": [[0.2, 0.3, 0.4], [1.1, 0.5, 0.7], [0.6, 1.4, 0.9], [1.7, 1.3, 1.1],
                      [2.6, 2.4, 0.3], [2.9, 0.4, 2.2], [0.5, 2.8, 2.6], [2.2, 2.9, 2.7]],
        "species": ["O", "H", "H", "O", "H", "O", "H", "H"],
        "cell": [[3.7, 0.0, 0.0], [1.2, 3.4, 0.0], [0.6, 0.5, 3.8]],
        "units": "EvAngstrom"
    })
}

#[test]
fn nve_on_the_gpu_runtime_follows_the_cpu_trajectory() {
    if !gpu_available() {
        return;
    }
    let directory = work_directory("nve");
    let request = |options: Value| {
        json!({
            "options": options,
            "frame": water_frame(),
            "masses": [15.999, 1.008, 1.008, 15.999, 1.008, 15.999, 1.008, 1.008],
            "initialize_velocities":
                {"temperature_kelvin": 300.0, "seed": 7, "rescale_to_temperature": true},
            "remove_center_of_mass": true,
            "ensemble": {"kind": "nve"},
            "timestep_fs": 0.5,
            "steps": 60,
            "sample_every": 20
        })
    };
    let model = fixture("synthetic-v1.json");
    let cpu = engine(
        "run",
        &directory,
        "cpu",
        &request(json!({"native_model": model})),
    );
    let samples = |records: &[Value]| -> Vec<Value> {
        records
            .iter()
            .filter(|r| r.get("step").is_some())
            .cloned()
            .collect()
    };
    let reference = samples(&cpu);
    assert_eq!(reference.len(), 4, "steps 0, 20, 40, 60");

    // Float64 networks: the same trajectory to rounding.
    let gpu = engine(
        "run",
        &directory,
        "gpu",
        &request(json!({"native_model": model, "device": "cuda:0", "gpu_runtime": {}})),
    );
    assert_eq!(gpu[0]["metadata"]["backend"], "native-gpu-se-e2-a");
    for (got, expected) in samples(&gpu).iter().zip(&reference) {
        let error = difference(got, expected);
        assert!(error < 1e-10, "float64 step {}: {error}", expected["step"]);
    }

    // Float32 networks with compressed embeddings stay close over 60 steps.
    let fast = engine(
        "run",
        &directory,
        "fast",
        &request(json!({"native_model": model, "device": "cuda:0",
            "gpu_runtime": {"precision": "float32", "compress_min_distance": 0.5}})),
    );
    assert_eq!(fast[0]["metadata"]["precision"], "float32");
    // The pressure is the virial in bar (1 eV/angstrom^3 is 1.6e6 bar), so it
    // is compared on its own scale.
    for (got, expected) in samples(&fast).iter().zip(&reference) {
        let (mut got, mut expected) = (got.clone(), expected.clone());
        let pressure =
            got["pressure_bar"].as_f64().unwrap() - expected["pressure_bar"].as_f64().unwrap();
        for record in [&mut got, &mut expected] {
            let fields = record.as_object_mut().unwrap();
            fields.remove("pressure_tensor_bar");
            fields.remove("pressure_bar");
        }
        let error = difference(&got, &expected);
        assert!(error < 1e-4, "float32 step {}: {error}", expected["step"]);
        assert!(
            pressure.abs() < 0.01,
            "float32 step {}: pressure differs by {pressure} bar",
            expected["step"]
        );
    }
    std::fs::remove_dir_all(directory).unwrap();
}

/// `count` frames around the water frame: jittered positions, and among
/// them frames with fewer atoms, without a cell and with a stretched cell.
fn assorted_frames(count: usize) -> Vec<Value> {
    let mut state = 88172645463325252u64;
    let mut jitter = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 * 0.1 - 0.05
    };
    (0..count)
        .map(|index| {
            let mut frame = water_frame();
            let positions = frame["positions"].as_array_mut().unwrap();
            for position in positions.iter_mut() {
                for value in position.as_array_mut().unwrap() {
                    *value = json!(value.as_f64().unwrap() + jitter());
                }
            }
            match index % 5 {
                1 => {
                    // Two atoms fewer.
                    positions.truncate(6);
                    frame["species"].as_array_mut().unwrap().truncate(6);
                }
                2 => frame["cell"] = Value::Null,
                3 => {
                    for value in positions.iter_mut().flat_map(|p| p.as_array_mut().unwrap()) {
                        *value = json!(value.as_f64().unwrap() * 1.04);
                    }
                    let cell = frame["cell"].as_array_mut().unwrap();
                    for value in cell.iter_mut().flat_map(|row| row.as_array_mut().unwrap()) {
                        *value = json!(value.as_f64().unwrap() * 1.04);
                    }
                }
                _ => {}
            }
            frame
        })
        .collect()
}

/// A dataset of `frames` in the engine's JSONL format, with zero labels.
fn write_dataset(path: &Path, frames: &[Value]) {
    let mut text = format!(
        "{}\n",
        json!({"kind": "header", "header": {"schema": "omnimd-dataset-jsonl-v1",
            "type_map": ["O", "H"], "units": "EvAngstrom",
            "virial_convention": "MinusDEpsilonRow", "source": "gpu_runtime test",
            "selected_sets": []}})
    );
    for (index, frame) in frames.iter().enumerate() {
        let atoms = frame["positions"].as_array().unwrap().len();
        let labels = json!({"energy": 0.0, "forces": vec![[0.0; 3]; atoms], "virial": null});
        let data = json!({"index": index, "id": format!("frame-{index}"), "frame": frame,
            "labels": labels});
        text += &format!("{}\n", json!({"kind": "frame", "data": data}));
    }
    text += &format!(
        "{}\n",
        json!({"kind": "completed", "frames": frames.len(), "source_sha256": {}})
    );
    std::fs::write(path, text).unwrap();
}

/// The synthetic model with something changed, written to `directory`.
fn variant(directory: &Path, name: &str, change: impl FnOnce(&mut Value)) -> PathBuf {
    let mut model: Value =
        serde_json::from_str(&std::fs::read_to_string(fixture("synthetic-v1.json")).unwrap())
            .unwrap();
    change(&mut model);
    let path = directory.join(name);
    std::fs::write(&path, model.to_string()).unwrap();
    path
}

/// Shifts every bias of the fitting nets.
fn shift_fitting_biases(model: &mut Value, by: f64) {
    for network in model["fitting"].as_array_mut().unwrap() {
        for layer in network["layers"].as_array_mut().unwrap() {
            for bias in layer["bias"].as_array_mut().unwrap() {
                *bias = json!(bias.as_f64().unwrap() + by);
            }
        }
    }
}

/// Runs a command that must fail and returns what it wrote to stderr.
fn engine_error(command: &str, directory: &Path, name: &str, request: &Value) -> String {
    let input = directory.join(format!("{name}.json"));
    std::fs::write(&input, request.to_string()).unwrap();
    let run = Command::new(env!("CARGO_BIN_EXE_omnimd-mlmd"))
        .arg(command)
        .arg(&input)
        .arg(directory.join(format!("{name}.jsonl")))
        .output()
        .unwrap();
    assert!(!run.status.success(), "{name} should fail");
    String::from_utf8_lossy(&run.stderr).into_owned()
}

fn of_kind(records: &[Value], kind: &str) -> Vec<Value> {
    records
        .iter()
        .filter(|record| record["kind"] == kind)
        .cloned()
        .collect()
}

#[test]
fn model_deviation_on_the_gpu_runtime_matches_cpu() {
    if !gpu_available() {
        return;
    }
    let directory = work_directory("deviation");
    // More frames than one device pass takes.
    let frames = assorted_frames(70);
    let dataset = directory.join("dataset.jsonl");
    write_dataset(&dataset, &frames);
    // Ensemble members: the model, and the model with other fitting biases.
    let first = fixture("synthetic-v1.json");
    let second = variant(&directory, "second.json", |m| shift_fitting_biases(m, 0.05));
    let third = variant(&directory, "third.json", |m| shift_fitting_biases(m, -0.08));
    // A member that cannot share a device pass with the others: its
    // descriptor is switched off over another range.
    let other = variant(&directory, "other.json", |m| {
        shift_fitting_biases(m, 0.11);
        m["descriptor"]["rcut_smth"] = json!(1.9);
    });
    let request = |models: &[&Path], extra: Value| {
        let members: Vec<Value> = models
            .iter()
            .enumerate()
            .map(|(index, model)| {
                let mut options = json!({"native_model": model});
                for (key, value) in extra.as_object().unwrap() {
                    options[key] = value.clone();
                }
                json!({"id": format!("member-{index}"), "options": options})
            })
            .collect();
        json!({
            "schema": "omnimd-model-deviation-v1",
            "models": members,
            "dataset": dataset,
            "per_atom": true,
            "force_window": {"min_ev_per_angstrom": 0.01, "max_ev_per_angstrom": 0.5}
        })
    };
    let on_gpu = json!({"device": "cuda:0", "gpu_runtime": {}});
    let alike: [&Path; 3] = [&first, &second, &third];
    let mixed: [&Path; 3] = [&first, &other, &second];
    let compare = |label: &str, cpu: &[Value], gpu: &[Value], tolerance: f64, how: &str| {
        assert_eq!(cpu.len(), gpu.len(), "{label}");
        let (expected, got) = (of_kind(cpu, "deviation"), of_kind(gpu, "deviation"));
        assert_eq!(expected.len(), 70, "{label}");
        for (g, e) in got.iter().zip(&expected) {
            assert_eq!((&g["index"], &g["id"]), (&e["index"], &e["id"]), "{label}");
            let apart = difference(&g["statistics"], &e["statistics"]);
            assert!(apart < tolerance, "{label}, frame {}: {apart}", e["index"]);
        }
        let header = &gpu[0]["evaluation"];
        assert!(
            header.as_str().unwrap().starts_with(how),
            "{label}: {header}"
        );
        assert_eq!(cpu[0]["evaluation"], "member by member, frame by frame");
        let (done, expected) = (gpu.last().unwrap(), cpu.last().unwrap());
        assert_eq!(done["combined_evaluations"], 210, "{label}");
        assert_eq!(done["model_loads"], 3, "{label}");
        // The same frames fall in the force window.
        for count in ["candidate_frames", "accurate_frames", "failed_frames"] {
            assert_eq!(done[count], expected[count], "{label}: {count}");
        }
    };
    let cpu = engine(
        "model-deviation",
        &directory,
        "cpu",
        &request(&alike, json!({})),
    );
    let gpu = engine(
        "model-deviation",
        &directory,
        "gpu",
        &request(&alike, on_gpu.clone()),
    );
    compare("committee", &cpu, &gpu, 1e-10, "all members together");
    // The members do disagree, and not on every frame alike.
    let spread: Vec<f64> = of_kind(&cpu, "deviation")
        .iter()
        .map(|record| {
            record["statistics"]["force_ev_per_angstrom"]["max"]
                .as_f64()
                .unwrap()
        })
        .collect();
    assert!(spread.iter().all(|&s| s > 1e-6) && spread[0] != spread[4]);
    // Float32 networks: the same statistics to single precision.
    let single = json!({"device": "cuda:0", "gpu_runtime": {"precision": "float32"}});
    let gpu = engine(
        "model-deviation",
        &directory,
        "gpu-float32",
        &request(&alike, single),
    );
    let window = |records: &[Value]| records.last().unwrap()["candidate_frames"].clone();
    assert_eq!(window(&gpu), window(&cpu));
    for (g, e) in of_kind(&gpu, "deviation")
        .iter()
        .zip(&of_kind(&cpu, "deviation"))
    {
        assert!(difference(&g["statistics"], &e["statistics"]) < 1e-4);
    }
    // Members of different architectures are evaluated one after the other,
    // each over a batch of frames.
    let cpu = engine(
        "model-deviation",
        &directory,
        "cpu-mixed",
        &request(&mixed, json!({})),
    );
    let gpu = engine(
        "model-deviation",
        &directory,
        "gpu-mixed",
        &request(&mixed, on_gpu.clone()),
    );
    compare(
        "mixed",
        &cpu,
        &gpu,
        1e-10,
        "member by member, a batch of frames",
    );
    // So are members with different runtime options.
    let mut unlike = request(&alike, on_gpu.clone());
    unlike["models"][2]["options"]["gpu_runtime"] = json!({"compress_min_distance": 0.4});
    let gpu = engine("model-deviation", &directory, "gpu-unlike", &unlike);
    let cpu = engine(
        "model-deviation",
        &directory,
        "cpu-alike",
        &request(&alike, json!({})),
    );
    compare(
        "unlike options",
        &cpu,
        &gpu,
        1e-7,
        "member by member, a batch of frames",
    );

    // A frame that cannot be evaluated is named by its place in the dataset.
    let mut broken = frames.clone();
    broken[37]["positions"][1] = broken[37]["positions"][0].clone();
    let broken_dataset = directory.join("broken-dataset.jsonl");
    write_dataset(&broken_dataset, &broken);
    for (name, models) in [("broken", &alike), ("broken-mixed", &mixed)] {
        let mut request = request(models, on_gpu.clone());
        request["dataset"] = json!(broken_dataset);
        let error = engine_error("model-deviation", &directory, name, &request);
        assert!(
            error.contains("dataset frame 37 (frame-37)") && error.contains("coincident"),
            "{name}: {error}"
        );
    }

    // Members on different devices are rejected before any worker starts.
    let mut apart = request(&alike, on_gpu);
    apart["models"][1]["options"] = json!({"native_model": second});
    let error = engine_error("model-deviation", &directory, "apart", &apart);
    assert!(error.contains("same cuda:N with gpu_runtime"), "{error}");
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn batch_eval_on_the_gpu_runtime_matches_cpu() {
    if !gpu_available() {
        return;
    }
    let directory = work_directory("batch-eval");
    let frames = assorted_frames(70);
    let dataset = directory.join("dataset.jsonl");
    write_dataset(&dataset, &frames);
    let model = fixture("synthetic-v1.json");
    let request = |extra: Value, dataset: &Path| {
        let mut options = json!({"native_model": model});
        for (key, value) in extra.as_object().unwrap() {
            options[key] = value.clone();
        }
        json!({"options": options, "dataset": dataset})
    };
    let on_gpu = json!({"device": "cuda:0", "gpu_runtime": {}});
    let cpu = engine(
        "batch-eval",
        &directory,
        "cpu",
        &request(json!({}), &dataset),
    );
    let gpu = engine(
        "batch-eval",
        &directory,
        "gpu",
        &request(on_gpu.clone(), &dataset),
    );
    assert_eq!(cpu.len(), gpu.len());
    let (expected, got) = (of_kind(&cpu, "prediction"), of_kind(&gpu, "prediction"));
    assert_eq!(expected.len(), 70);
    for (g, e) in got.iter().zip(&expected) {
        assert_eq!(
            (&g["index"], &g["id"], &g["natoms"]),
            (&e["index"], &e["id"], &e["natoms"])
        );
        let apart = difference(&g["evaluation"], &e["evaluation"]);
        assert!(apart < 1e-9, "frame {}: {apart}", e["index"]);
    }
    // The frames are not all alike.
    assert_ne!(expected[0]["natoms"], expected[1]["natoms"]);
    assert_ne!(
        expected[0]["evaluation"]["energy"],
        expected[5]["evaluation"]["energy"]
    );
    let (done, reference) = (gpu.last().unwrap(), cpu.last().unwrap());
    assert_eq!(done["frames"], 70);
    assert!(difference(&done["metrics"], &reference["metrics"]) < 1e-9);

    // A frame that cannot be evaluated is named by its place in the dataset.
    let mut broken = frames.clone();
    broken[41]["positions"][1] = broken[41]["positions"][0].clone();
    let broken_dataset = directory.join("broken-dataset.jsonl");
    write_dataset(&broken_dataset, &broken);
    let error = engine_error(
        "batch-eval",
        &directory,
        "broken",
        &request(on_gpu, &broken_dataset),
    );
    assert!(
        error.contains("dataset frame 41 (frame-41)") && error.contains("coincident"),
        "{error}"
    );
    std::fs::remove_dir_all(directory).unwrap();
}

/// Runs `omnimd-mlmd run-replicas` and returns the records of each replica.
fn replicas(directory: &Path, name: &str, request: &Value) -> Vec<Vec<Value>> {
    let input = directory.join(format!("{name}.json"));
    let output = directory.join(name);
    std::fs::write(&input, request.to_string()).unwrap();
    let run = Command::new(env!("CARGO_BIN_EXE_omnimd-mlmd"))
        .arg("run-replicas")
        .arg(&input)
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{name}: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    request["replicas"]
        .as_array()
        .unwrap()
        .iter()
        .map(|replica| {
            let trajectory = output.join(format!("{}.jsonl", replica["name"].as_str().unwrap()));
            std::fs::read_to_string(trajectory)
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect()
        })
        .collect()
}

#[test]
fn replicas_on_the_gpu_runtime_follow_the_cpu_replicas() {
    if !gpu_available() {
        return;
    }
    let directory = work_directory("replicas");
    // Three systems of different size, boundary and ensemble; the second
    // moves its cell every fourth step.
    let frames = assorted_frames(3);
    let ensembles = [
        json!({"kind": "nve"}),
        json!({"kind": "npt_csvr_crescale", "temperature_kelvin": 300.0, "tau_fs": 20.0,
            "seed": 5, "pressure_bar": 1.0, "barostat_tau_fs": 200.0,
            "compressibility_per_bar": 1e-5, "coupling": "anisotropic", "barostat_every": 4}),
        json!({"kind": "nvt_csvr", "temperature_kelvin": 300.0, "tau_fs": 20.0, "seed": 4}),
    ];
    let request = |options: Value| {
        let replicas: Vec<Value> = frames
            .iter()
            .zip(&ensembles)
            .enumerate()
            .map(|(index, (frame, ensemble))| {
                let masses: Vec<f64> = frame["species"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|species| if species == "O" { 15.999 } else { 1.008 })
                    .collect();
                json!({"name": format!("r{index}"), "frame": frame, "masses": masses,
                    "initialize_velocities": {"temperature_kelvin": 300.0,
                        "seed": 7 + index as u64, "rescale_to_temperature": true},
                    "remove_center_of_mass": true, "ensemble": ensemble})
            })
            .collect();
        json!({"schema": "omnimd-replicas-v1", "options": options, "replicas": replicas,
            "timestep_fs": 0.5, "steps": 40, "sample_every": 10})
    };
    let model = fixture("synthetic-v1.json");
    let cpu = replicas(&directory, "cpu", &request(json!({"native_model": model})));
    let gpu = replicas(
        &directory,
        "gpu",
        &request(json!({"native_model": model, "device": "cuda:0", "gpu_runtime": {}})),
    );
    assert_eq!((cpu.len(), gpu.len()), (3, 3));
    for (index, (got, expected)) in gpu.iter().zip(&cpu).enumerate() {
        assert_eq!(got[0]["metadata"]["backend"], "native-gpu-se-e2-a");
        assert_eq!(got[0]["replica"]["name"], format!("r{index}"));
        assert_eq!(got.len(), expected.len(), "replica {index}");
        // Header first, completed last; five sampled frames between.
        assert_eq!(got.len(), 7, "replica {index}");
        for (g, e) in got[1..6].iter().zip(&expected[1..6]) {
            let error = difference(g, e);
            assert!(error < 1e-9, "replica {index} step {}: {error}", e["step"]);
        }
        let (done, reference) = (got.last().unwrap(), expected.last().unwrap());
        for key in ["final_step", "barostat_moves", "barostat_refused"] {
            assert_eq!(done[key], reference[key], "replica {index}: {key}");
        }
    }
    assert_eq!(gpu[1].last().unwrap()["barostat_moves"], 10);
    assert_ne!(gpu[1][1]["frame"]["cell"], gpu[1][5]["frame"]["cell"]);
    // Float32 networks with compressed embeddings: the same runs, close.
    let fast = replicas(
        &directory,
        "fast",
        &request(json!({"native_model": model, "device": "cuda:0",
            "gpu_runtime": {"precision": "float32", "compress_min_distance": 0.5}})),
    );
    for (index, (got, expected)) in fast.iter().zip(&cpu).enumerate() {
        let energy = |record: &Value| record["evaluation"]["energy"].as_f64().unwrap();
        for (g, e) in got[1..6].iter().zip(&expected[1..6]) {
            assert!(
                (energy(g) - energy(e)).abs() < 1e-4,
                "replica {index} step {}",
                e["step"]
            );
        }
    }
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn deviation_along_a_run_on_the_gpu_runtime_matches_cpu() {
    if !gpu_available() {
        return;
    }
    let directory = work_directory("run-deviation");
    let first = fixture("synthetic-v1.json");
    let second = variant(&directory, "second.json", |m| shift_fitting_biases(m, 0.05));
    let third = variant(&directory, "third.json", |m| shift_fitting_biases(m, -0.08));
    // A member of another architecture: the members then go one by one.
    let other = variant(&directory, "other.json", |m| {
        shift_fitting_biases(m, 0.11);
        m["descriptor"]["rcut_smth"] = json!(1.9);
    });
    let request = |extra: &Value, others: &[&Path]| {
        let with = |model: &Path| {
            let mut options = json!({"native_model": model});
            for (key, value) in extra.as_object().unwrap() {
                options[key] = value.clone();
            }
            options
        };
        json!({
            "options": with(&first),
            "frame": water_frame(),
            "masses": [15.999, 1.008, 1.008, 15.999, 1.008, 15.999, 1.008, 1.008],
            "initialize_velocities":
                {"temperature_kelvin": 300.0, "seed": 7, "rescale_to_temperature": true},
            "remove_center_of_mass": true,
            "ensemble": {"kind": "nvt_csvr", "temperature_kelvin": 300.0, "tau_fs": 20.0, "seed": 3},
            "timestep_fs": 0.5,
            "steps": 30,
            "sample_every": 10,
            "deviation": {"models": others.iter().map(|model| with(model)).collect::<Vec<_>>(),
                "force_window": {"min_ev_per_angstrom": 0.0, "max_ev_per_angstrom": 100.0}}
        })
    };
    let on_gpu = json!({"device": "cuda:0", "gpu_runtime": {}});
    let alike: [&Path; 2] = [&second, &third];
    let mixed: [&Path; 2] = [&other, &second];
    for (label, others, how) in [
        ("alike", &alike, "all members together"),
        ("mixed", &mixed, "member by member, a batch of frames"),
    ] {
        let cpu = engine(
            "run",
            &directory,
            &format!("cpu-{label}"),
            &request(&json!({}), others),
        );
        let gpu = engine(
            "run",
            &directory,
            &format!("gpu-{label}"),
            &request(&on_gpu, others),
        );
        let how_it_went = gpu[0]["deviation"]["evaluation"].as_str().unwrap();
        assert!(how_it_went.starts_with(how), "{label}: {how_it_went}");
        assert_eq!(
            cpu[0]["deviation"]["evaluation"],
            "member by member, frame by frame"
        );
        let (got, expected) = (of_kind(&gpu, "frame"), of_kind(&cpu, "frame"));
        assert_eq!((got.len(), expected.len()), (4, 4), "{label}");
        for (g, e) in got.iter().zip(&expected) {
            let apart = difference(&g["model_deviation"], &e["model_deviation"]);
            assert!(apart < 1e-9, "{label}, step {}: {apart}", e["step"]);
            assert_eq!(g["candidate"], true);
            // The run itself follows the CPU run, as without the members.
            let apart = difference(&g["evaluation"], &e["evaluation"]);
            assert!(apart < 1e-9, "{label}, step {}: {apart}", e["step"]);
        }
        let spread = expected[0]["model_deviation"]["force_ev_per_angstrom"]["max"]
            .as_f64()
            .unwrap();
        assert!(spread > 1e-6, "{label}: the members differ");
        assert_eq!(gpu.last().unwrap()["deviation"]["candidate_frames"], 4);
    }
    // Replicas with their members: one pass for all frames and members.
    let replicas_request = |extra: &Value| {
        let mut request = request(extra, &alike);
        let fields = request.as_object_mut().unwrap();
        let system: Vec<(String, Value)> = [
            "frame",
            "masses",
            "initialize_velocities",
            "remove_center_of_mass",
            "ensemble",
        ]
        .iter()
        .map(|key| (key.to_string(), fields.remove(*key).unwrap()))
        .collect();
        let replicas: Vec<Value> = (0..3)
            .map(|index| {
                let mut replica: Value = system
                    .iter()
                    .cloned()
                    .collect::<serde_json::Map<_, _>>()
                    .into();
                replica["name"] = json!(format!("r{index}"));
                replica["initialize_velocities"]["seed"] = json!(20 + index);
                replica
            })
            .collect();
        request["schema"] = json!("omnimd-replicas-v1");
        request["replicas"] = json!(replicas);
        request
    };
    let cpu = replicas(&directory, "cpu-replicas", &replicas_request(&json!({})));
    let gpu = replicas(&directory, "gpu-replicas", &replicas_request(&on_gpu));
    for (index, (got, expected)) in gpu.iter().zip(&cpu).enumerate() {
        assert!(got[0]["deviation"]["evaluation"]
            .as_str()
            .unwrap()
            .starts_with("all members together"));
        for (g, e) in of_kind(got, "frame")
            .iter()
            .zip(&of_kind(expected, "frame"))
        {
            let apart = difference(&g["model_deviation"], &e["model_deviation"]);
            assert!(apart < 1e-9, "replica {index}, step {}: {apart}", e["step"]);
        }
    }
    assert_ne!(
        gpu[0][2]["model_deviation"]["force_ev_per_angstrom"]["max"],
        gpu[1][2]["model_deviation"]["force_ev_per_angstrom"]["max"]
    );
    std::fs::remove_dir_all(directory).unwrap();
}
