//! `merge-datasets`: frames of several datasets as one.
use omnimd_ml_contract::{Frame, Units};
use omnimd_mlmd::{
    dataset::{
        self, DatasetFrame, DatasetHeader, DatasetReader, Labels, Record, VirialConvention, SCHEMA,
    },
    merge::{merge_datasets, MergeRequest, REQUEST_SCHEMA},
};
use std::{
    collections::BTreeMap,
    io::Write,
    path::{Path, PathBuf},
};

fn directory(name: &str) -> PathBuf {
    let directory =
        std::env::temp_dir().join(format!("omnimd-merge-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

/// A dataset of `count` two-atom frames with the IDs `PREFIX/k`; every other
/// frame has an energy label.
fn write(path: &Path, type_map: &[&str], prefix: &str, count: u64) {
    let mut writer = dataset::new_output(path).unwrap();
    let header = DatasetHeader {
        parameter_layout: None,
        schema: SCHEMA.into(),
        type_map: type_map.iter().map(|s| s.to_string()).collect(),
        units: Units::EvAngstrom,
        virial_convention: VirialConvention::MinusDEpsilonRow,
        source: format!("test {prefix}"),
        selected_sets: vec![],
    };
    dataset::write_line(&mut writer, &Record::Header { header }).unwrap();
    for index in 0..count {
        let shift = index as f64 * 0.1;
        let data = DatasetFrame {
            parameters: None,
            index,
            id: format!("{prefix}/{index}"),
            frame: Frame {
                positions: vec![[shift, 0., 0.], [0.9 + shift, 0.3, 0.]],
                species: vec![type_map[0].into(), type_map[1].into()],
                cell: Some([[6., 0., 0.], [0., 6., 0.], [0., 0., 6.]]),
                units: Units::EvAngstrom,
            },
            labels: Labels {
                energy: (index % 2 == 0).then_some(-1.5 - shift),
                forces: None,
                virial: None,
            },
        };
        dataset::write_line(&mut writer, &Record::Frame { data }).unwrap();
    }
    let completed = Record::Completed {
        frames: count,
        source_sha256: BTreeMap::new(),
    };
    dataset::write_line(&mut writer, &completed).unwrap();
    writer.flush().unwrap();
}

fn request(datasets: &[&Path]) -> MergeRequest {
    serde_json::from_value(serde_json::json!({"schema": REQUEST_SCHEMA, "datasets": datasets}))
        .unwrap()
}

#[test]
fn datasets_merge_in_order_with_new_indices() {
    let root = directory("order");
    let (a, b, c) = (
        root.join("a.jsonl"),
        root.join("b.jsonl"),
        root.join("c.jsonl"),
    );
    write(&a, &["O", "H"], "first", 3);
    write(&b, &["O", "H"], "second", 2);
    write(&c, &["O", "H"], "third", 4);
    let merged = root.join("merged.jsonl");
    assert_eq!(merge_datasets(&request(&[&a, &b, &c]), &merged).unwrap(), 9);
    let mut reader = DatasetReader::open_hashed(&merged).unwrap();
    assert_eq!(reader.header.type_map, ["O", "H"]);
    let expected: Vec<String> = [("first", 3), ("second", 2), ("third", 4)]
        .iter()
        .flat_map(|(prefix, count)| (0..*count).map(move |k| format!("{prefix}/{k}")))
        .collect();
    for (index, id) in expected.iter().enumerate() {
        let data = reader.next_frame().unwrap().unwrap();
        assert_eq!((data.index, &data.id), (index as u64, id));
        // Geometry and labels are those of the source frame.
        let source: u64 = id.rsplit('/').next().unwrap().parse().unwrap();
        assert_eq!(data.frame.positions[0][0], source as f64 * 0.1);
        assert_eq!(data.labels.energy.is_some(), source % 2 == 0);
    }
    assert!(reader.next_frame().unwrap().is_none());
    // The inputs are recorded by their hashes.
    let sources = reader
        .completed_provenance()
        .unwrap()
        .declared_source_sha256;
    assert_eq!(sources.len(), 3);
    assert_eq!(sources["dataset-1"], dataset::sha256(&b).unwrap());

    // Refused: an existing output, one dataset, a repeated dataset, repeated
    // IDs, another type map, a relative path, a wrong schema.
    let fresh = |name: &str| root.join(name);
    assert!(merge_datasets(&request(&[&a, &b]), &merged).is_err());
    assert!(merge_datasets(&request(&[&a]), &fresh("one.jsonl")).is_err());
    assert!(merge_datasets(&request(&[&a, &a]), &fresh("twice.jsonl")).is_err());
    let same_ids = root.join("same.jsonl");
    write(&same_ids, &["O", "H"], "first", 1);
    let error = merge_datasets(&request(&[&a, &same_ids]), &fresh("ids.jsonl")).unwrap_err();
    assert!(error.to_string().contains("first/0"), "{error}");
    let other_types = root.join("types.jsonl");
    write(&other_types, &["H", "O"], "swapped", 1);
    assert!(merge_datasets(&request(&[&a, &other_types]), &fresh("types-out.jsonl")).is_err());
    assert!(
        !fresh("types-out.jsonl").exists(),
        "headers are checked before writing"
    );
    assert!(merge_datasets(&request(&[&a, Path::new("b.jsonl")]), &fresh("rel.jsonl")).is_err());
    let mut wrong = request(&[&a, &b]);
    wrong.schema = "omnimd-dataset-merge-v0".into();
    assert!(merge_datasets(&wrong, &fresh("schema.jsonl")).is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn the_cli_merges_datasets() {
    let root = directory("cli");
    let (a, b) = (root.join("a.jsonl"), root.join("b.jsonl"));
    write(&a, &["O", "H"], "first", 2);
    write(&b, &["O", "H"], "second", 3);
    let (input, output) = (root.join("merge.json"), root.join("merged.jsonl"));
    std::fs::write(
        &input,
        serde_json::json!({"schema": REQUEST_SCHEMA, "datasets": [a, b]}).to_string(),
    )
    .unwrap();
    let run = std::process::Command::new(env!("CARGO_BIN_EXE_omnimd-mlmd"))
        .arg("merge-datasets")
        .arg(&input)
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(String::from_utf8_lossy(&run.stdout).contains("Merged 5 frames of 2 datasets"));
    let mut reader = DatasetReader::open_hashed(&output).unwrap();
    let mut frames = 0;
    while reader.next_frame().unwrap().is_some() {
        frames += 1;
    }
    assert_eq!(frames, 5);
    std::fs::remove_dir_all(root).unwrap();
}
