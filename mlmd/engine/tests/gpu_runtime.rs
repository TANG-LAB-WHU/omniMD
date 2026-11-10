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

#[test]
fn model_deviation_on_the_gpu_runtime_matches_cpu() {
    if !gpu_available() {
        return;
    }
    let directory = work_directory("deviation");
    // A two-frame dataset in the engine's JSONL format.
    let mut second = water_frame();
    second["positions"][0] = json!([0.35, 0.2, 0.5]);
    let zero_forces = vec![[0.0; 3]; 8];
    let labels = json!({"energy": 0.0, "forces": zero_forces, "virial": null});
    let records = [
        json!({"kind": "header", "header": {"schema": "omnimd-dataset-jsonl-v1",
            "type_map": ["O", "H"], "units": "EvAngstrom",
            "virial_convention": "MinusDEpsilonRow", "source": "gpu_runtime test",
            "selected_sets": []}}),
        json!({"kind": "frame", "data": {"index": 0, "id": "a", "frame": water_frame(),
            "labels": labels}}),
        json!({"kind": "frame", "data": {"index": 1, "id": "b", "frame": second,
            "labels": labels}}),
        json!({"kind": "completed", "frames": 2, "source_sha256": {}}),
    ];
    let dataset = directory.join("dataset.jsonl");
    let text: String = records.iter().map(|r| format!("{r}\n")).collect();
    std::fs::write(&dataset, text).unwrap();

    // The second ensemble member: the same model with perturbed fitting biases.
    let first = fixture("synthetic-v1.json");
    let mut perturbed: Value =
        serde_json::from_str(&std::fs::read_to_string(&first).unwrap()).unwrap();
    for network in perturbed["fitting"].as_array_mut().unwrap() {
        for layer in network["layers"].as_array_mut().unwrap() {
            for bias in layer["bias"].as_array_mut().unwrap() {
                *bias = json!(bias.as_f64().unwrap() + 0.05);
            }
        }
    }
    let second_model = directory.join("perturbed.json");
    std::fs::write(&second_model, perturbed.to_string()).unwrap();

    let request = |extra: Value| {
        let member = |id: &str, model: &Path| {
            let mut options = json!({"native_model": model});
            for (key, value) in extra.as_object().unwrap() {
                options[key] = value.clone();
            }
            json!({"id": id, "options": options})
        };
        json!({
            "schema": "omnimd-model-deviation-v1",
            "models": [member("a", &first), member("b", &second_model)],
            "dataset": dataset,
            "per_atom": true
        })
    };
    let cpu = engine("model-deviation", &directory, "cpu", &request(json!({})));
    let gpu = engine(
        "model-deviation",
        &directory,
        "gpu",
        &request(json!({"device": "cuda:0", "gpu_runtime": {}})),
    );
    assert_eq!(cpu.len(), gpu.len());
    let frames = |records: &[Value]| -> Vec<Value> {
        records
            .iter()
            .filter(|r| r["kind"] == "deviation")
            .cloned()
            .collect()
    };
    let (expected, got) = (frames(&cpu), frames(&gpu));
    assert!(
        !expected.is_empty(),
        "no per-frame deviation records: {cpu:?}"
    );
    for (g, e) in got.iter().zip(&expected) {
        assert!(difference(g, e) < 1e-10);
    }

    // Members on different devices are rejected before any worker starts.
    let mut mixed = request(json!({"device": "cuda:0", "gpu_runtime": {}}));
    mixed["models"][1]["options"] = json!({"native_model": second_model});
    let input = directory.join("mixed.json");
    std::fs::write(&input, mixed.to_string()).unwrap();
    let run = Command::new(env!("CARGO_BIN_EXE_omnimd-mlmd"))
        .arg("model-deviation")
        .arg(&input)
        .arg(directory.join("mixed.jsonl"))
        .output()
        .unwrap();
    assert!(!run.status.success());
    assert!(String::from_utf8_lossy(&run.stderr).contains("same cuda:N with gpu_runtime"));
    std::fs::remove_dir_all(directory).unwrap();
}
