use omnimd_mlmd::{
    dataset::{sha256, DatasetReader},
    labeling::{import_labels, prepare_labels},
};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
static COUNT: AtomicU64 = AtomicU64::new(0);
fn write(path: &Path, rows: &[Value]) {
    fs::write(
        path,
        rows.iter().map(|v| format!("{v}\n")).collect::<String>(),
    )
    .unwrap();
}
fn read(path: &Path) -> Vec<Value> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect()
}
fn fixture(v2: bool) -> (PathBuf, PathBuf, Vec<Value>) {
    let root = std::env::temp_dir().join(format!(
        "omnimd-label-{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        COUNT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&root).unwrap();
    let mut header = json!({"schema":"omnimd-dataset-jsonl-v1","type_map":["O","H"],"units":"EvAngstrom","virial_convention":"MinusDEpsilonRow","source":"original label test","selected_sets":[]});
    if v2 {
        header["schema"] = "omnimd-dataset-jsonl-v2".into();
        header["parameter_layout"] = json!({"fparam":[{"name":"temperature","unit":"K"}],"aparam":[{"name":"atomic_condition","unit":"fixture"}]});
    }
    let mut rows = vec![json!({"kind":"header","header":header})];
    for i in 0..3 {
        let mut data = json!({"index":i,"id":format!("frame/{i}"),"frame":{"positions":[[0.0,0.,0.],[1.2345678901234567,0.1,0.2]],"species":["O","O"],"cell":null,"units":"EvAngstrom"},"labels":{"energy":if i==0 {Some(1.25)} else {None},"forces":null,"virial":if i==0 {Some([[2.0,0.,0.],[0.,2.,0.],[0.,0.,2.]])} else {None}}});
        if i == 1 {
            data["frame"]["cell"] = json!([[4., 0., 0.], [0.1, 5., 0.], [0.2, 0.3, 6.]]);
            data["labels"]["forces"] = json!([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]);
        }
        if v2 {
            data["parameters"] = json!({"frame_parameter_policy":if i==2 {"model_default"} else {"explicit"},"fparam":if i==2 {vec![]} else {vec![300.0]},"aparam":[[0.125],[0.875]]});
        }
        rows.push(json!({"kind":"frame","data":data}));
    }
    rows.push(json!({"kind":"completed","frames":3,"source_sha256":{"original":"ab".repeat(32)}}));
    let data = root.join("input.jsonl");
    write(&data, &rows);
    (root, data, rows)
}
fn request(root: &Path, data: &Path, labels: Option<&Path>, name: &str) -> PathBuf {
    let mut r = json!({"schema":"omnimd-prepare-labels-v1","dataset":data,"expected_source_sha256":sha256(data).unwrap()});
    if let Some(labels) = labels {
        r["schema"] = "omnimd-import-labels-v1".into();
        r["labels"] = json!(labels);
        r["expected_labels_sha256"] = sha256(labels).unwrap().into();
    }
    let out = root.join(name);
    fs::write(&out, serde_json::to_vec(&r).unwrap()).unwrap();
    out
}
fn returned(root: &Path, data: &Path) -> Vec<Value> {
    let req = request(root, data, None, "prepare.json");
    let bundle = root.join("job");
    let receipt = prepare_labels(&req, &bundle).unwrap();
    assert_eq!(
        receipt.output_sha256,
        sha256(&bundle.join("job.jsonl")).unwrap()
    );
    let mut rows = read(&bundle.join("job.jsonl"));
    rows[0]["schema"] = "omnimd-label-return-v1".into();
    rows[0]["provenance"] = json!({"provider":"test calculator","method":"manufactured E/F","source_sha256":{"input_dataset":"cd".repeat(32)}});
    for row in &mut rows[1..4] {
        let mut labels = row
            .as_object_mut()
            .unwrap()
            .remove("existing_labels")
            .unwrap();
        if labels["energy"].is_null() {
            labels["energy"] = json!(1.25);
        }
        labels["forces"] = json!([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]);
        labels["virial"] = Value::Null;
        row["labels"] = labels;
    }
    rows.swap(1, 3);
    rows
}
#[test]
fn complete_out_of_order_returns_preserve_science_v1_v2_and_fill_only_missing() {
    for v2 in [false, true] {
        let (root, data, original) = fixture(v2);
        let rows = returned(&root, &data);
        let labels = root.join("returned.jsonl");
        write(&labels, &rows);
        let req = request(&root, &data, Some(&labels), "import.json");
        let out = root.join("labeled");
        let report = import_labels(&req, &out).unwrap();
        assert_eq!(report.frames, 3);
        assert_eq!(
            report.output_sha256,
            sha256(&out.join("dataset.jsonl")).unwrap()
        );
        assert_eq!(
            fs::read(&labels).unwrap(),
            fs::read(out.join("labels.jsonl")).unwrap()
        );
        assert_eq!(
            fs::read(&req).unwrap(),
            fs::read(out.join("request.json")).unwrap()
        );
        let actual = read(&out.join("dataset.jsonl"));
        assert_eq!(actual[0], original[0]);
        for i in 1..4 {
            for key in ["index", "id", "frame", "parameters"] {
                assert_eq!(actual[i]["data"][key], original[i]["data"][key]);
            }
        }
        assert_eq!(
            actual[1]["data"]["labels"]["virial"],
            original[1]["data"]["labels"]["virial"]
        );
        assert_eq!(
            actual[4]["source_sha256"]["input_dataset"],
            sha256(&data).unwrap()
        );
        assert_eq!(
            actual[4]["source_sha256"]["calculator:input_dataset"],
            "cd".repeat(32)
        );
        assert_eq!(report.bindings[0].added_fields, vec!["forces"]);
        assert_eq!(report.bindings[1].added_fields, vec!["energy"]);
        let mut reader = DatasetReader::open(&out.join("dataset.jsonl")).unwrap();
        while reader.next_frame().unwrap().is_some() {}
        let old = sha256(&out.join("dataset.jsonl")).unwrap();
        assert!(import_labels(&req, &out).is_err());
        assert_eq!(old, sha256(&out.join("dataset.jsonl")).unwrap());
        assert!(prepare_labels(&root.join("prepare.json"), &root.join("job")).is_err());
    }
}
#[test]
fn identity_units_shape_duplicate_missing_and_conflict_errors_create_no_bundle() {
    let (root, data, _) = fixture(true);
    let baseline = returned(&root, &data);
    for case in 0..24 {
        let mut rows = baseline.clone();
        match case {
            0 => rows[0]["source_dataset_sha256"] = "00".repeat(32).into(),
            1 => rows[0]["context"]["units"] = "HartreeBohr".into(),
            2 => rows[0]["context"]["virial_convention"] = "Stress".into(),
            3 => rows[0]["context"]["type_map"] = json!(["H", "O"]),
            4 => rows[1]["identity"]["frame"]["positions"][0][0] = json!(0.0000000000000001),
            5 => {
                rows[1]["identity"]["frame"]["positions"] =
                    json!([[1.2345678901234567, 0.1, 0.2], [0., 0., 0.]])
            }
            6 => rows[1]["identity"]["parameters"]["aparam"] = json!([[0.875], [0.125]]),
            7 => {
                rows[1]["identity"]
                    .as_object_mut()
                    .unwrap()
                    .remove("parameters");
            }
            8 => rows[1]["identity"]["index"] = 0.into(),
            9 => rows[1]["identity"]["id"] = "other".into(),
            10 => rows[1]["identity"]["identity_sha256"] = "00".repeat(32).into(),
            11 => rows[1]["labels"]["energy"] = Value::Null,
            12 => rows[1]["labels"]["forces"] = json!([[0., 0., 0.]]),
            13 => rows[1]["labels"]["virial"] = json!([[0., 0.]]),
            14 => rows[3]["labels"]["energy"] = 99.into(),
            15 => rows[2]["labels"]["forces"][0][0] = 99.into(),
            16 => rows[3]["labels"]["virial"] = json!([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
            17 => rows.insert(2, rows[1].clone()),
            18 => {
                rows.remove(1);
            }
            19 => {
                rows.pop();
            }
            20 => rows.push(json!({"kind":"completed","frames":3})),
            21 => rows[4]["frames"] = 2.into(),
            22 => rows[1]["identity"]["frame"]["positions"][0][0] = json!(-0.0),
            _ => rows[0]["context"]["parameter_layout"]["aparam"][0]["unit"] = "changed".into(),
        }
        let labels = root.join(format!("bad-{case}.jsonl"));
        write(&labels, &rows);
        let req = request(&root, &data, Some(&labels), &format!("r-{case}.json"));
        let out = root.join(format!("out-{case}"));
        assert!(import_labels(&req, &out).is_err(), "case {case}");
        assert!(!out.exists(), "case {case}");
    }
}
#[test]
fn exact_byte_hashes_truncation_and_duplicate_json_fields_reject() {
    let (root, data, _) = fixture(false);
    let rows = returned(&root, &data);
    let labels = root.join("return.jsonl");
    write(&labels, &rows);
    let req = request(&root, &data, Some(&labels), "import.json");
    let out = root.join("out");
    let mut r: Value = serde_json::from_slice(&fs::read(&req).unwrap()).unwrap();
    r["expected_labels_sha256"] = "ab".repeat(32).into();
    fs::write(&req, r.to_string()).unwrap();
    assert!(import_labels(&req, &out).is_err());
    assert!(!out.exists());
    for (i, text) in [
        fs::read_to_string(&labels).unwrap().trim_end().to_owned(),
        fs::read_to_string(&labels).unwrap().replace(
            "\"provider\":\"test calculator\"",
            "\"provider\":\"test calculator\",\"provider\":\"duplicate\"",
        ),
        fs::read_to_string(&labels).unwrap().replace(
            &format!("\"input_dataset\":\"{}\"", "cd".repeat(32)),
            &format!(
                "\"same\":\"{}\",\"same\":\"{}\"",
                "cd".repeat(32),
                "ab".repeat(32)
            ),
        ),
        fs::read_to_string(&labels)
            .unwrap()
            .replace("\"energy\":1.25", "\"energy\":1e999"),
    ]
    .iter()
    .enumerate()
    {
        let bad = root.join(format!("malformed-{i}.jsonl"));
        fs::write(&bad, text).unwrap();
        let r = request(&root, &data, Some(&bad), &format!("m-{i}.json"));
        assert!(import_labels(&r, &out).is_err(), "case {i}");
        assert!(!out.exists());
    }
    let prepare = request(&root, &data, None, "wrong-source.json");
    let mut r: Value = serde_json::from_slice(&fs::read(&prepare).unwrap()).unwrap();
    r["expected_source_sha256"] = "00".repeat(32).into();
    fs::write(&prepare, r.to_string()).unwrap();
    assert!(prepare_labels(&prepare, &root.join("bad-job")).is_err());
    assert!(!root.join("bad-job").exists());
}
#[test]
fn bounded_reader_enforces_exact_input_limit_including_trailing_bytes() {
    let (_root, data, _) = fixture(false);
    let size = fs::metadata(&data).unwrap().len();
    let mut exact = DatasetReader::open_hashed_bounded(&data, size).unwrap();
    while exact.next_frame().unwrap().is_some() {}
    assert_eq!(
        exact.completed_provenance().unwrap().dataset_sha256,
        sha256(&data).unwrap()
    );
    let result = (|| {
        let mut short = DatasetReader::open_hashed_bounded(&data, size - 1)?;
        while short.next_frame()?.is_some() {}
        short.completed_provenance()
    })();
    assert!(result.is_err());
}
