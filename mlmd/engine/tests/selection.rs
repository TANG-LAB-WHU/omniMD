use omnimd_mlmd::{
    dataset::{sha256, DatasetReader},
    selection::select_dataset,
};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
static COUNT: AtomicU64 = AtomicU64::new(0);
fn fixture() -> (PathBuf, PathBuf, Vec<Value>) {
    let root = std::env::temp_dir().join(format!(
        "omnimd-select-{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        COUNT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&root).unwrap();
    let mut records = vec![
        json!({"kind":"header","header":{"schema":"omnimd-dataset-jsonl-v1","type_map":["O","H"],"units":"EvAngstrom","virial_convention":"MinusDEpsilonRow","source":"original selection test","selected_sets":[]}}),
    ];
    for (i, id) in ["a", "b", "c"].iter().enumerate() {
        records.push(json!({"kind":"frame","data":{"index":i,"id":id,"frame":{"positions":[[0.12345678901234566,0.,0.],[1.,0.,0.]],"species":["O","H"],"cell":null,"units":"EvAngstrom"},"labels":{"energy":if i==1 {Some(1.25)} else {None},"forces":null,"virial":null}}}));
    }
    records
        .push(json!({"kind":"completed","frames":3,"source_sha256":{"fixture":"ab".repeat(32)}}));
    let input = root.join("input.jsonl");
    save_records(&input, &records);
    (root, input, records)
}
fn save_records(path: &Path, values: &[Value]) {
    fs::write(
        path,
        values
            .iter()
            .map(|v| format!("{v}\r\n"))
            .collect::<String>(),
    )
    .unwrap();
}
fn request(root: &Path, input: &Path, ids: Value, name: &str) -> PathBuf {
    let path = root.join(name);
    fs::write(&path,serde_json::to_vec(&json!({"schema":"omnimd-select-dataset-v1","dataset":input,"expected_source_sha256":sha256(input).unwrap(),"ids":ids})).unwrap()).unwrap();
    path
}
#[test]
fn selection_preserves_content_order_identity_and_immediate_ancestry() {
    let (root, input, records) = fixture();
    let req = request(&root, &input, json!(["c", "a"]), "request.json");
    let out = root.join("selected");
    let report = select_dataset(&req, &out).unwrap();
    assert_eq!(report.input_dataset_sha256, sha256(&input).unwrap());
    assert_eq!(
        report.output_dataset_sha256,
        sha256(&out.join("dataset.jsonl")).unwrap()
    );
    assert_eq!(
        report.request_sha256,
        sha256(&out.join("request.json")).unwrap()
    );
    assert_eq!(
        fs::read(&req).unwrap(),
        fs::read(out.join("request.json")).unwrap()
    );
    assert_eq!(
        report
            .mapping
            .iter()
            .map(|m| (m.id.as_str(), m.source_index, m.output_index))
            .collect::<Vec<_>>(),
        vec![("a", 0, 0), ("c", 2, 1)]
    );
    let output: Vec<Value> = fs::read_to_string(out.join("dataset.jsonl"))
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    for (new, old) in [(1, 1), (2, 3)] {
        assert_eq!(output[new]["data"]["frame"], records[old]["data"]["frame"]);
        assert_eq!(
            output[new]["data"]["labels"],
            records[old]["data"]["labels"]
        );
    }
    assert_eq!(
        output[3]["source_sha256"]["upstream:fixture"],
        "ab".repeat(32)
    );
    let mut reader = DatasetReader::open(&out.join("dataset.jsonl")).unwrap();
    while reader.next_frame().unwrap().is_some() {}
    let again = request(
        &root,
        &out.join("dataset.jsonl"),
        json!(["c"]),
        "again.json",
    );
    let repeated = select_dataset(&again, &root.join("again")).unwrap();
    assert_eq!(repeated.mapping[0].source_index, 1);
    assert_eq!(repeated.mapping[0].id, "c");
    assert!(select_dataset(&req, &out).is_err());
    assert_eq!(
        report.output_dataset_sha256,
        sha256(&out.join("dataset.jsonl")).unwrap()
    );
}
#[test]
fn incorrect_hash_ids_and_corrupt_unselected_suffix_create_no_bundle() {
    for (i, ids) in [
        json!([]),
        json!(["a", "a"]),
        json!([""]),
        json!(["missing"]),
    ]
    .iter()
    .enumerate()
    {
        let (root, input, _) = fixture();
        let req = request(&root, &input, ids.clone(), "r.json");
        let out = root.join(format!("out-{i}"));
        assert!(select_dataset(&req, &out).is_err());
        assert!(!out.exists());
    }
    let (root, input, mut records) = fixture();
    let req = request(&root, &input, json!(["a"]), "r.json");
    let out = root.join("out");
    let mut value: Value = serde_json::from_slice(&fs::read(&req).unwrap()).unwrap();
    value["expected_source_sha256"] = json!("00".repeat(32));
    fs::write(&req, value.to_string()).unwrap();
    assert!(select_dataset(&req, &out)
        .err()
        .unwrap()
        .to_string()
        .contains("SHA256"));
    assert!(!out.exists());
    // The only requested frame is valid, but a later unselected frame is invalid.
    records[3]["data"]["index"] = 99.into();
    save_records(&input, &records);
    let req = request(&root, &input, json!(["a"]), "suffix.json");
    assert!(select_dataset(&req, &out).is_err());
    assert!(!out.exists());
}
