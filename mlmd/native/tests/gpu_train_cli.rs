//! `omnimd-native-gpu-train` as a program: one request, several requests at
//! once on streams of one device, and the descriptors a request may ask
//! for. Skips without a CUDA device unless OMNIMD_REQUIRE_GPU is set.
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

/// A request may name the radial descriptor under either of DeePMD-kit's
/// names for it, and either descriptor with one embedding net per neighbor
/// type; and a model trained so is a model to continue from.
#[test]
fn se_e2_r_and_nets_per_neighbor_type_train_from_a_request() {
    if !gpu_available() {
        return;
    }
    let directory =
        std::env::temp_dir().join(format!("omnimd-train-cli-radial-{}", std::process::id()));
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
    let radial = |body: &mut Value, name: &str, one_side: bool| {
        let descriptor = body["model"]["descriptor"].as_object_mut().unwrap();
        descriptor.insert("type".into(), json!(name));
        descriptor.remove("axis_neuron");
        descriptor.insert("type_one_side".into(), json!(one_side));
    };
    // What the trainer said of DeePMD-kit's PyTorch backend, which gives the
    // radial descriptor a net per neighbor type whatever the script says.
    let said = std::cell::RefCell::new(Vec::new());
    let trained = |name: &str, request: &Path| -> Value {
        let run = train(&[request]);
        let notes = String::from_utf8_lossy(&run.stderr).into_owned();
        assert!(run.status.success(), "{name}: {notes}");
        if notes.contains("one net per neighbor type whatever") {
            said.borrow_mut().push(name.to_string());
        }
        let summary: Value = serde_json::from_slice(&run.stdout).unwrap();
        assert_eq!(summary["steps"].as_u64(), Some(12), "{name}");
        let path = directory.join(format!("{name}.native.json"));
        // The model that was written is one the CPU reference takes.
        omnimd_native::NativePotential::load(&path).unwrap();
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    };
    let rows_alike = |model: &Value| {
        let rows = model["descriptor"]["embedding"].as_array().unwrap();
        rows.iter().all(|row| row == &rows[0])
    };

    // The radial descriptor with a net for every pair of types.
    let pairs = trained(
        "pairs",
        &edit("pairs", &|body| radial(body, "se_e2_r", false)),
    );
    assert_eq!(pairs["format"], "omnimd-native-se-e2-r");
    assert_eq!(pairs["descriptor"]["kind"], "se_e2_r");
    assert_eq!(pairs["descriptor"]["axis_neuron"], 0);
    assert!(pairs["descriptor"].get("type_one_side").is_none());
    assert!(!rows_alike(&pairs));
    for entry in pairs["descriptor"]["normalization"].as_array().unwrap() {
        assert_eq!(
            entry["std"].as_array().unwrap()[1..],
            vec![json!(1.0); 3][..]
        );
        assert_eq!(
            entry["mean"].as_array().unwrap()[1..],
            vec![json!(0.0); 3][..]
        );
    }
    // Its fitting nets take the embedding width.
    let first = &pairs["fitting"][0]["layers"][0]["weights"];
    assert_eq!(first.as_array().unwrap().len(), 8);

    // Under its other name, with a net per neighbor type: after twelve
    // updates the rows of the model are still alike.
    let shared = trained(
        "shared",
        &edit("shared", &|body| radial(body, "se_r", true)),
    );
    assert_eq!(shared["descriptor"]["kind"], "se_e2_r");
    assert_eq!(shared["descriptor"]["type_one_side"], true);
    assert!(rows_alike(&shared));
    // se_e2_a with a net per neighbor type.
    let angular = trained(
        "angular",
        &edit("angular", &|body| {
            body["model"]["descriptor"]["type_one_side"] = json!(true)
        }),
    );
    assert_eq!(angular["format"], "omnimd-native-se-e2-a");
    assert_eq!(angular["descriptor"]["type_one_side"], true);
    assert!(rows_alike(&angular));

    // Continuing from the trained models: the descriptor and the shared
    // nets are the model's own.
    for (name, source) in [("pairs", &pairs), ("shared", &shared)] {
        let next = format!("{name}-more");
        let more = trained(
            &next,
            &edit(&next, &|body| {
                body["model"] = json!(directory.join(format!("{name}.native.json")))
            }),
        );
        assert_eq!(more["descriptor"]["kind"], "se_e2_r");
        assert_eq!(rows_alike(&more), name == "shared");
        assert_ne!(
            more["fitting"], source["fitting"],
            "{name}: nothing was learned"
        );
    }
    // Only the fresh radial model with a net per pair of types differs from
    // what DeePMD-kit's PyTorch backend would build from the same script;
    // se_e2_a with a net per pair, the plain request, does not.
    trained("plain", &request(&directory, "plain", 5));
    assert_eq!(*said.borrow(), ["pairs"]);

    // What a request cannot ask for.
    let refused = |name: &str, change: &dyn Fn(&mut Value)| {
        let run = train(&[&edit(name, change)]);
        assert!(!run.status.success(), "{name}");
        assert!(
            !directory.join(format!("{name}.native.json")).exists(),
            "{name}"
        );
        String::from_utf8_lossy(&run.stderr).into_owned()
    };
    let error = refused("axis", &|body| {
        radial(body, "se_e2_r", false);
        body["model"]["descriptor"]["axis_neuron"] = json!(2);
    });
    assert!(error.contains("se_e2_r has no axis_neuron"), "{error}");
    let error = refused("unknown", &|body| radial(body, "se_e3", false));
    assert!(error.contains("only se_e2_a and se_e2_r"), "{error}");
    std::fs::remove_dir_all(directory).unwrap();
}

/// The numbers of a part of a model, in the order they are written.
fn numbers_of(value: &Value) -> Vec<f64> {
    let mut out = Vec::new();
    numbers(value, &mut out);
    out
}

/// Largest difference between two lists of as many numbers.
fn apart(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "the parts differ in shape");
    assert!(!a.is_empty());
    a.iter()
        .zip(b)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f64::max)
}

/// A request may ask for a hybrid, `{"type": "hybrid", "list": [...]}`: every
/// descriptor of the list with its own cutoffs, selection, nets and seed.
/// Each gets the statistics it gets as the descriptor of a model, and the
/// trained model is a hybrid to evaluate and to continue from.
#[test]
fn a_hybrid_trains_from_a_request() {
    if !gpu_available() {
        return;
    }
    let directory =
        std::env::temp_dir().join(format!("omnimd-train-cli-hybrid-{}", std::process::id()));
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
    // The descriptors of the hybrid: an angular one of the near neighbors
    // and a radial one, with wider nets, of all within 4 A.
    let angular = json!({"type": "se_e2_a", "sel": "auto", "rcut": 3.0, "rcut_smth": 0.5,
                         "neuron": [4, 8], "axis_neuron": 2, "seed": 5});
    let radial = json!({"type": "se_r", "sel": "auto:1.5", "rcut": 4.0, "rcut_smth": 1.0,
                        "neuron": [6, 12], "type_one_side": true, "seed": 5});
    let trained = |name: &str, request: &Path| -> (Value, String) {
        let run = train(&[request]);
        let notes = String::from_utf8_lossy(&run.stderr).into_owned();
        assert!(run.status.success(), "{name}: {notes}");
        let summary: Value = serde_json::from_slice(&run.stdout).unwrap();
        assert_eq!(summary["steps"].as_u64(), Some(12), "{name}");
        let path = directory.join(format!("{name}.native.json"));
        // The model that was written is one the CPU reference takes.
        omnimd_native::NativePotential::load(&path).unwrap();
        let model = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        (model, notes)
    };
    let list = json!([angular, radial]);
    let (hybrid, notes) = trained(
        "hybrid",
        &edit("hybrid", &|body| {
            body["model"]["descriptor"] = json!({"type": "hybrid", "list": list})
        }),
    );
    assert_eq!(hybrid["format"], "omnimd-native-hybrid");
    let d = &hybrid["descriptor"];
    assert_eq!(d["kind"], "hybrid");
    assert_eq!((&d["rcut"], &d["rcut_smth"]), (&json!(4.0), &json!(0.5)));
    let parts = d["list"].as_array().unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(
        (&parts[0]["kind"], &parts[1]["kind"]),
        (&json!("se_e2_a"), &json!("se_e2_r"))
    );
    assert_eq!(
        (&parts[0]["rcut"], &parts[1]["rcut"]),
        (&json!(3.0), &json!(4.0))
    );
    assert_eq!(parts[1]["type_one_side"], true);
    assert!(parts[0].get("type_one_side").is_none());
    // The list has of every type the most slots of the two.
    let sel = |value: &Value| -> Vec<u64> {
        value["sel"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_u64().unwrap())
            .collect()
    };
    let (near, far) = (sel(&parts[0]), sel(&parts[1]));
    assert_ne!(near, far);
    let most: Vec<u64> = near.iter().zip(&far).map(|(a, b)| *a.max(b)).collect();
    assert_eq!(sel(d), most);
    // The fitting nets take the components of both: 8 · 2 + 12.
    let first = &hybrid["fitting"][0]["layers"][0]["weights"];
    assert_eq!(first.as_array().unwrap().len(), 28);
    // What the trainer said: of every descriptor its own statistics.
    for line in [
        "descriptor.list[0]: neighbor statistics:",
        "descriptor.list[1]: neighbor statistics:",
        "descriptor.list[0]: se_e2_a within 3 A, sel [",
        "descriptor.list[1]: se_r within 4 A, sel [",
    ] {
        assert!(notes.contains(line), "{line}: {notes}");
    }
    assert!(
        !notes.contains("one net per neighbor type whatever"),
        "{notes}"
    );
    // The line about the energy bias names no selection: every descriptor
    // has had a line with its own.
    assert!(
        notes
            .lines()
            .any(|line| line.starts_with("energy bias per type")),
        "{notes}"
    );

    // Each descriptor alone, as the descriptor of a model, has the
    // selection and the normalization it has in the hybrid.
    let (alone, _) = trained(
        "angular",
        &edit("angular", &|body| {
            body["model"]["descriptor"] = angular.clone()
        }),
    );
    assert_eq!(alone["format"], "omnimd-native-se-e2-a");
    assert_eq!(sel(&alone["descriptor"]), near);
    assert_eq!(
        alone["descriptor"]["normalization"],
        parts[0]["normalization"]
    );
    let (alone, _) = trained(
        "radial",
        &edit("radial", &|body| {
            body["model"]["descriptor"] = radial.clone()
        }),
    );
    assert_eq!(alone["format"], "omnimd-native-se-e2-r");
    assert_eq!(sel(&alone["descriptor"]), far);
    assert_eq!(
        alone["descriptor"]["normalization"],
        parts[1]["normalization"]
    );

    // A list of one descriptor trains to the nets of that descriptor alone.
    let (single, _) = trained(
        "single",
        &edit("single", &|body| {
            body["model"]["descriptor"] = json!({"type": "hybrid", "list": [radial]})
        }),
    );
    assert_eq!(single["format"], "omnimd-native-hybrid");
    let part = &single["descriptor"]["list"][0];
    let difference = apart(
        &numbers_of(&part["embedding"]),
        &numbers_of(&alone["descriptor"]["embedding"]),
    );
    assert!(difference < 1e-8, "embedding nets differ by {difference}");
    let difference = apart(
        &numbers_of(&single["fitting"]),
        &numbers_of(&alone["fitting"]),
    );
    assert!(difference < 1e-8, "fitting nets differ by {difference}");
    // In the hybrid of two the same descriptor has learned other nets.
    let other = apart(
        &numbers_of(&parts[1]["embedding"]),
        &numbers_of(&alone["descriptor"]["embedding"]),
    );
    assert!(other > 1e-3, "{other}");

    // Continuing from the trained hybrid: the descriptors are the model's.
    let (more, _) = trained(
        "more",
        &edit("more", &|body| {
            body["model"] = json!(directory.join("hybrid.native.json"))
        }),
    );
    assert_eq!(more["format"], "omnimd-native-hybrid");
    let more_parts = more["descriptor"]["list"].as_array().unwrap();
    assert_eq!(more_parts.len(), 2);
    for (number, (after, before)) in more_parts.iter().zip(parts).enumerate() {
        assert_eq!(after["sel"], before["sel"], "descriptor {number}");
        assert_eq!(after["normalization"], before["normalization"]);
        assert_ne!(
            after["embedding"], before["embedding"],
            "descriptor {number}: nothing was learned"
        );
    }
    let rows = more_parts[1]["embedding"].as_array().unwrap();
    assert!(rows.iter().all(|row| row == &rows[0]));
    assert_ne!(more["fitting"], hybrid["fitting"]);

    // The radial descriptor with a net for every pair of types: the note
    // about DeePMD-kit's PyTorch backend names it by its place.
    let mut pairs = radial.clone();
    pairs["type_one_side"] = json!(false);
    let (_, notes) = trained(
        "pairs",
        &edit("pairs", &|body| {
            body["model"]["descriptor"] = json!({"type": "hybrid", "list": [angular, pairs]})
        }),
    );
    assert!(
        notes.contains("descriptor.list[1]: se_e2_r with an embedding net for every pair"),
        "{notes}"
    );

    // What a request cannot ask for is said, with the place in the list.
    let refused = |name: &str, descriptor: Value| {
        let run = train(&[&edit(name, &|body| {
            body["model"]["descriptor"] = descriptor.clone()
        })]);
        assert!(!run.status.success(), "{name}");
        assert!(
            !directory.join(format!("{name}.native.json")).exists(),
            "{name}"
        );
        String::from_utf8_lossy(&run.stderr).into_owned()
    };
    let mut three_body = radial.clone();
    three_body["type"] = json!("se_e3");
    let error = refused(
        "three-body",
        json!({"type": "hybrid", "list": [angular, three_body]}),
    );
    assert!(
        error.contains("descriptor.list[1]: only se_e2_a and se_e2_r descriptors"),
        "{error}"
    );
    let mut typo = angular.clone();
    typo["rcut_smooth"] = json!(0.5);
    let error = refused(
        "typo",
        json!({"type": "hybrid", "list": [typo.clone(), radial]}),
    );
    assert!(
        error.contains("model: descriptor.list[0]: unknown field `rcut_smooth`"),
        "{error}"
    );
    let error = refused("plain-typo", typo);
    assert!(
        error.contains("model: descriptor: unknown field `rcut_smooth`"),
        "{error}"
    );
    let error = refused(
        "nested",
        json!({"type": "hybrid", "list": [angular, {"type": "hybrid", "list": [radial]}]}),
    );
    assert!(
        error.contains("descriptor.list[1]: a hybrid cannot hold another hybrid"),
        "{error}"
    );
    let error = refused("empty", json!({"type": "hybrid", "list": []}));
    assert!(
        error.contains("a hybrid descriptor has 1..=16 descriptors in its list, got 0"),
        "{error}"
    );
    let error = refused(
        "keys",
        json!({"type": "hybrid", "list": [angular], "rcut": 4.0}),
    );
    assert!(
        error.contains("model: descriptor: unknown field `rcut`"),
        "{error}"
    );
    std::fs::remove_dir_all(directory).unwrap();
}
