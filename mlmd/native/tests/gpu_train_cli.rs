//! `omnimd-native-gpu-train` as a program: one request, and several
//! requests at once on streams of one device. Skips without a CUDA device
//! unless OMNIMD_REQUIRE_GPU is set.
#![cfg(feature = "gpu")]
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn gpu_available() -> bool {
    match omnimd_gpu::Device::new(0) {
        Ok(_) => true,
        Err(error) if std::env::var_os("OMNIMD_REQUIRE_GPU").is_some() => {
            panic!("OMNIMD_REQUIRE_GPU is set but no device: {error}")
        }
        Err(error) => {
            eprintln!("skipping trainer CLI tests: {error}");
            false
        }
    }
}

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/stats-dataset.jsonl")
        .canonicalize()
        .unwrap()
}

/// A twelve-step request on the statistics fixture (six frames of O, H, C).
fn request(directory: &Path, name: &str, seed: u64) -> PathBuf {
    write_request(directory, name, seed, None)
}

/// The same, optionally scoring `validation` frames per learning-curve line.
fn write_request(directory: &Path, name: &str, seed: u64, validation: Option<usize>) -> PathBuf {
    let dataset = fixture();
    let mut request = json!({
        "model": {
            "type_map": ["O", "H", "C"],
            "descriptor": {"type": "se_e2_a", "sel": "auto", "rcut": 4.0, "rcut_smth": 0.5,
                           "neuron": [4, 8], "axis_neuron": 2, "resnet_dt": false,
                           "type_one_side": false, "seed": seed},
            "fitting_net": {"neuron": [8, 8], "resnet_dt": true, "seed": seed}
        },
        "dataset": dataset,
        "output": directory.join(format!("{name}.native.json")),
        "precision": "float64",
        "learning_rate": {"start_lr": 1e-3, "stop_lr": 1e-4, "decay_steps": 4},
        "loss": {},
        "training": {"numb_steps": 12, "batch_size": 1, "seed": seed, "disp_freq": 4,
                     "disp_file": directory.join(format!("{name}.lcurve"))}
    });
    if let Some(frames) = validation {
        request["training"]["validation_data"] = json!({"dataset": fixture(), "numb_btch": frames});
    }
    let path = directory.join(format!("{name}.json"));
    std::fs::write(&path, request.to_string()).unwrap();
    path
}

fn train(requests: &[&Path]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_omnimd-native-gpu-train"))
        .args(requests)
        .output()
        .unwrap()
}

fn numbers(value: &Value, out: &mut Vec<f64>) {
    match value {
        Value::Number(n) => out.push(n.as_f64().unwrap()),
        Value::Array(items) => items.iter().for_each(|item| numbers(item, out)),
        Value::Object(map) => map.values().for_each(|item| numbers(item, out)),
        _ => {}
    }
}

/// Largest difference between the numbers of two trained models.
fn difference(a: &Path, b: &Path) -> f64 {
    let read = |path: &Path| -> Vec<f64> {
        let model: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let mut values = Vec::new();
        numbers(&model, &mut values);
        values
    };
    let (x, y) = (read(a), read(b));
    assert_eq!(x.len(), y.len(), "models differ in shape");
    assert!(x.len() > 500, "expected a model's worth of numbers");
    x.iter()
        .zip(&y)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f64::max)
}

#[test]
fn several_requests_train_at_once_like_one_by_one() {
    if !gpu_available() {
        return;
    }
    let root = std::env::temp_dir().join(format!("omnimd-train-cli-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let (alone, together) = (root.join("alone"), root.join("together"));
    for directory in [&alone, &together] {
        std::fs::create_dir_all(directory).unwrap();
    }
    // Each request by itself.
    for (name, seed) in [("a", 5), ("b", 6)] {
        let run = train(&[&request(&alone, name, seed)]);
        assert!(
            run.status.success(),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
        let summary: Value = serde_json::from_slice(&run.stdout).unwrap();
        assert_eq!(
            (summary["steps"].as_u64(), summary["frames"].as_u64()),
            (Some(12), Some(6))
        );
    }
    // Both in one process: one summary line each, and the same models (the
    // order of atomic additions on the device varies, so not bit for bit).
    let (a, b) = (request(&together, "a", 5), request(&together, "b", 6));
    let run = train(&[&a, &b]);
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let summaries: Vec<Value> = String::from_utf8(run.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(summaries.len(), 2);
    for name in ["a", "b"] {
        let file = format!("{name}.native.json");
        let apart = difference(&alone.join(&file), &together.join(&file));
        assert!(apart < 1e-8, "{name}: models differ by {apart}");
        assert!(together.join(format!("{name}.lcurve")).is_file());
    }
    // Different seeds give different models.
    assert!(difference(&alone.join("a.native.json"), &alone.join("b.native.json")) > 1e-3);

    // A request that cannot run fails the process and is named; the other
    // one still trains.
    let third = root.join("third");
    std::fs::create_dir_all(&third).unwrap();
    let good = request(&third, "good", 7);
    let bad = third.join("missing.json");
    let run = train(&[&good, &bad]);
    assert!(!run.status.success());
    assert!(String::from_utf8_lossy(&run.stderr).contains("missing.json"));
    assert!(third.join("good.native.json").is_file());
    // No request at all is a usage error.
    assert!(!train(&[]).status.success());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn validation_frames_are_scored_but_not_trained_on() {
    if !gpu_available() {
        return;
    }
    let root = std::env::temp_dir().join(format!("omnimd-train-val-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let plain = train(&[&request(&root, "plain", 5)]);
    assert!(
        plain.status.success(),
        "{}",
        String::from_utf8_lossy(&plain.stderr)
    );
    let scored = train(&[&write_request(&root, "scored", 5, Some(2))]);
    assert!(
        scored.status.success(),
        "{}",
        String::from_utf8_lossy(&scored.stderr)
    );
    // The same model with and without validation.
    let apart = difference(
        &root.join("plain.native.json"),
        &root.join("scored.native.json"),
    );
    assert!(apart < 1e-8, "validation changed the model by {apart}");
    // Lines at steps 0, 4, 8 and 11 with the validation columns first.
    let curve = std::fs::read_to_string(root.join("scored.lcurve")).unwrap();
    let mut lines = curve.lines();
    let header: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
    assert_eq!(
        header,
        [
            "#",
            "step",
            "rmse_e_val",
            "rmse_e_trn",
            "rmse_f_val",
            "rmse_f_trn",
            "rmse_v_val",
            "rmse_v_trn",
            "lr"
        ]
    );
    let rows: Vec<Vec<f64>> = lines
        .map(|line| {
            line.split_whitespace()
                .map(|x| x.parse().unwrap())
                .collect()
        })
        .collect();
    assert_eq!(
        rows.iter().map(|r| r[0] as u64).collect::<Vec<_>>(),
        [0, 4, 8, 11]
    );
    assert!(rows
        .iter()
        .all(|r| r.len() == 8 && r[1] > 0.0 && r[3] > 0.0));
    let plain_curve = std::fs::read_to_string(root.join("plain.lcurve")).unwrap();
    assert_eq!(
        plain_curve
            .lines()
            .next()
            .unwrap()
            .split_whitespace()
            .count(),
        6
    );
    let summary: Value = serde_json::from_slice(&scored.stdout).unwrap();
    assert!(summary["final_validation"]["rmse_f"].as_f64().unwrap() > 0.0);
    let summary: Value = serde_json::from_slice(&plain.stdout).unwrap();
    assert!(summary["final_validation"].is_null());
    // A validation set with another type map, or no frame per line, is refused.
    let mut bad: Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("scored.json")).unwrap()).unwrap();
    bad["output"] = json!(root.join("bad.native.json"));
    bad["training"]["disp_file"] = json!(root.join("bad.lcurve"));
    bad["training"]["validation_data"]["numb_btch"] = json!(0);
    let path = root.join("bad.json");
    std::fs::write(&path, bad.to_string()).unwrap();
    assert!(!train(&[&path]).status.success());
    std::fs::remove_dir_all(root).unwrap();
}

/// `atom_ener` gives the energy bias of every type instead of the fit to the
/// training energies, and the loss terms follow DeePMD-kit's PyTorch rule.
#[test]
fn a_given_energy_bias_is_used_and_a_zero_prefactor_turns_a_term_off() {
    if !gpu_available() {
        return;
    }
    let directory =
        std::env::temp_dir().join(format!("omnimd-train-cli-bias-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let edit = |name: &str, change: &dyn Fn(&mut Value)| -> PathBuf {
        let path = request(&directory, name, 5);
        let mut body: Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        change(&mut body);
        std::fs::write(&path, body.to_string()).unwrap();
        path
    };
    let given = edit("given", &|body| {
        body["model"]["fitting_net"]["atom_ener"] = json!([-1.5, 0.25, 3.0]);
        body["loss"] = json!({"start_pref_v": 0.0, "limit_pref_v": 0.02});
    });
    let output = train(&[&given]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let model: Value = serde_json::from_str(
        &std::fs::read_to_string(directory.join("given.native.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(model["atom_energy_bias"], json!([-1.5, 0.25, 3.0]));
    let log = String::from_utf8_lossy(&output.stderr);
    assert!(log.contains("(atom_ener)"), "{log}");
    assert!(log.contains("the virial term is off"), "{log}");
    // One energy per type, or none.
    let short = edit("short", &|body| {
        body["model"]["fitting_net"]["atom_ener"] = json!([-1.5, 0.25]);
    });
    let output = train(&[&short]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("atom_ener"));
    std::fs::remove_dir_all(directory).unwrap();
}
