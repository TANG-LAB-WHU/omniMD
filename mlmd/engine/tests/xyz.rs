use omnimd_ml_contract::Units;
use omnimd_mlmd::{
    dataset::{DatasetReader, Pbc},
    xyz::{
        export_extxyz, export_xyz, import_xyz, XyzExportRequest, XyzFormat, XyzRequest,
        EXPORT_SCHEMA,
    },
};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
fn root() -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "omnimd-xyz-{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&p).unwrap();
    p
}
const VALID:&str="2\nProperties=pos:R:3:force:R:3:species:S:1 pbc=\"T T T\" Lattice=\"8 0 0 1 9 0 2 3 10\" energy=-5.0 virial=\"1 4 7 2 5 8 3 6 9\"\n1 2 3 0.1 0.2 0.3 H\n4 5 6 -0.1 -0.2 -0.3 O\n";
fn request(input: PathBuf) -> XyzRequest {
    XyzRequest {
        input,
        format: XyzFormat::Extxyz,
        units: Units::EvAngstrom,
        pbc: Pbc::Periodic,
        type_map: vec!["O".into(), "H".into()],
        cell: None,
    }
}
#[test]
fn extxyz_reorders_columns_and_converts_standard_tensor_order_once() {
    let root = root();
    let input = root.join("source.xyz");
    fs::write(&input, VALID).unwrap();
    let output = root.join("data.jsonl");
    import_xyz(&request(input), &output).unwrap();
    let mut reader = DatasetReader::open(&output).unwrap();
    let data = reader.next_frame().unwrap().unwrap();
    assert_eq!(data.frame.species, ["H", "O"]);
    assert_eq!(data.frame.cell.unwrap()[2], [2., 3., 10.]);
    assert_eq!(data.labels.virial.unwrap()[1], [4., 5., 6.]);
    assert_eq!(data.labels.energy, Some(-5.));
    assert!(reader.next_frame().unwrap().is_none());
    let mut export = Vec::new();
    export_extxyz(&mut export, &data.frame, "converged").unwrap();
    let file = root.join("export.xyz");
    fs::write(&file, export).unwrap();
    let output = root.join("again.jsonl");
    import_xyz(&request(file), &output).unwrap();
    let data = DatasetReader::open(&output)
        .unwrap()
        .next_frame()
        .unwrap()
        .unwrap();
    assert!(
        data.labels.energy.is_none()
            && data.labels.forces.is_none()
            && data.labels.virial.is_none()
    );
}
#[test]
fn xyz_is_explicit_and_extxyz_malformed_inputs_fail() {
    let root = root();
    let input = root.join("simple.xyz");
    fs::write(&input, "1\nplain comment\nH 1D0 2 3\n").unwrap();
    let mut req = request(input);
    req.format = XyzFormat::Xyz;
    req.pbc = Pbc::NonPeriodic;
    let output = root.join("plain.jsonl");
    import_xyz(&req, &output).unwrap();
    let data = DatasetReader::open(&output)
        .unwrap()
        .next_frame()
        .unwrap()
        .unwrap();
    assert_eq!(data.frame.positions[0], [1., 2., 3.]);
    assert!(data.frame.cell.is_none());
    let bad = [
        VALID.replace("T T T", "T F T"),
        VALID.replace("energy=-5.0", "energy=NaN"),
        VALID.replace("energy=-5.0", "stress=\"1 2 3 4 5 6\""),
        VALID.replace("force:R:3", "vel:R:3"),
        VALID.replace("energy=-5.0", "energy=-5 energy=-4"),
        VALID.replace("pbc=\"T T T\"", "pbc=\"F F F\""),
        VALID.replace("H\n4", "X\n4"),
        VALID.replace(" -0.3 O", " O"),
        VALID.replace("energy=-5.0", "units=HartreeBohr"),
    ];
    for (i, text) in bad.iter().enumerate() {
        let path = root.join(format!("bad{i}.xyz"));
        fs::write(&path, text).unwrap();
        assert!(
            import_xyz(&request(path), &root.join(format!("bad{i}.jsonl"))).is_err(),
            "case {i}"
        );
    }
}

/// A dataset written by hand: the first frame with every label, the second
/// with an energy only, the third with none. `cell` is each frame's cell.
fn dataset(root: &std::path::Path, name: &str, cell: serde_json::Value) -> PathBuf {
    use serde_json::json;
    let third = 1.0 / 3.0;
    let frame = |shift: f64| {
        json!({"positions": [[0.1 + shift, -0.2, third], [1.0e-7, 2.5, -123.45678901234568],
                             [3.0, 0.30000000000000004, 0.7]],
            "species": ["O", "H", "H"], "cell": cell, "units": "EvAngstrom"})
    };
    let rows = [
        json!({"kind": "header", "header": {"schema": "omnimd-dataset-jsonl-v1",
            "type_map": ["O", "H"], "units": "EvAngstrom",
            "virial_convention": "MinusDEpsilonRow", "source": "an export test",
            "selected_sets": []}}),
        json!({"kind": "frame", "data": {"index": 0, "id": "run/step-10", "frame": frame(0.0),
            "labels": {"energy": -5.123456789012345,
                "forces": [[0.1, -0.2, third], [1.0e-9, 0.0, 2.0], [-0.1, 0.2, -2.0 - third]],
                "virial": [[third, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.000000000000002]]}}}),
        json!({"kind": "frame", "data": {"index": 1, "id": "a \"quoted\" \\ id", "frame": frame(0.5),
            "labels": {"energy": 0.25, "forces": null, "virial": null}}}),
        json!({"kind": "frame", "data": {"index": 2, "id": "plain", "frame": frame(1.0),
            "labels": {"energy": null, "forces": null, "virial": null}}}),
        json!({"kind": "completed", "frames": 3, "source_sha256": {}}),
    ];
    let path = root.join(name);
    let text: String = rows.iter().map(|row| format!("{row}\n")).collect();
    fs::write(&path, text).unwrap();
    path
}

fn frames(path: &std::path::Path) -> Vec<omnimd_mlmd::dataset::DatasetFrame> {
    let mut reader = DatasetReader::open(path).unwrap();
    let mut all = Vec::new();
    while let Some(data) = reader.next_frame().unwrap() {
        all.push(data);
    }
    all
}

fn export_request(dataset: &std::path::Path) -> XyzExportRequest {
    serde_json::from_value(serde_json::json!({"schema": EXPORT_SCHEMA, "dataset": dataset}))
        .unwrap()
}

#[test]
fn a_dataset_goes_to_extended_xyz_and_back_unchanged() {
    use serde_json::{json, to_value};
    let root = root();
    for (name, cell, pbc) in [
        (
            "periodic",
            json!([
                [8.0, 0.0, 0.0],
                [1.1, 9.0, 0.0],
                [2.0, 0.30000000000000004, 10.0]
            ]),
            Pbc::Periodic,
        ),
        ("open", json!(null), Pbc::NonPeriodic),
    ] {
        let source = dataset(&root, &format!("{name}.jsonl"), cell);
        let exported = root.join(format!("{name}.extxyz"));
        assert_eq!(export_xyz(&export_request(&source), &exported).unwrap(), 3);
        let text = fs::read_to_string(&exported).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3 * 5, "three atoms, two more lines a frame");
        // The comment lines: what each frame has, and its ID.
        let comments = [lines[1], lines[6], lines[11]];
        assert!(comments[0].contains("Properties=species:S:1:pos:R:3:forces:R:3 "));
        assert!(comments[0].contains(" energy=-5.12345678901234"));
        // The virial in the column-major order of extended XYZ.
        let virial: Vec<f64> = comments[0]
            .split("virial=\"")
            .nth(1)
            .unwrap()
            .split('"')
            .next()
            .unwrap()
            .split(' ')
            .map(|number| number.parse().unwrap())
            .collect();
        assert_eq!(
            virial,
            [
                1.0 / 3.0,
                4.0,
                7.0,
                2.0,
                5.0,
                8.0,
                3.0,
                6.0,
                9.000000000000002
            ]
        );
        assert!(comments[0].ends_with(" comment=\"run/step-10\""));
        assert!(comments[1].contains("Properties=species:S:1:pos:R:3 "));
        assert!(comments[1].contains(" energy=") && !comments[1].contains("virial="));
        assert!(comments[1].ends_with(r#" comment="a \"quoted\" \\ id""#));
        assert!(!comments[2].contains("energy=") && !comments[2].contains("forces"));
        assert!(comments[2].ends_with(" comment=\"plain\""));
        for comment in comments {
            assert!(comment.contains(" units=EvAngstrom"));
            assert_eq!(comment.starts_with("Lattice=\""), pbc == Pbc::Periodic);
            assert_eq!(comment.contains("pbc=\"T T T\""), pbc == Pbc::Periodic);
            assert_eq!(comment.contains("pbc=\"F F F\""), pbc == Pbc::NonPeriodic);
        }
        // Atom lines: a force for every atom of the first frame, none else.
        assert_eq!(lines[2].split(' ').count(), 7);
        assert_eq!(lines[7].split(' ').count(), 4);

        // Back through import-xyz: the same numbers to the last digit.
        let mut back = request(exported.clone());
        back.pbc = pbc;
        let again = root.join(format!("{name}-again.jsonl"));
        import_xyz(&back, &again).unwrap();
        let (before, after) = (frames(&source), frames(&again));
        assert_eq!(after.len(), 3);
        for (index, (was, is)) in before.iter().zip(&after).enumerate() {
            assert_eq!(to_value(&is.frame).unwrap(), to_value(&was.frame).unwrap());
            assert_eq!(
                to_value(&is.labels).unwrap(),
                to_value(&was.labels).unwrap()
            );
            assert_eq!(
                is.id,
                format!("xyz/{index}"),
                "import names the frames anew"
            );
        }

        // Nothing is overwritten.
        assert!(export_xyz(&export_request(&source), &exported).is_err());
    }

    // What is refused.
    let source = dataset(&root, "source.jsonl", json!(null));
    let refused = |name: &str, request: XyzExportRequest, why: &str| {
        let output = root.join(format!("{name}.extxyz"));
        let error = export_xyz(&request, &output).expect_err(name).to_string();
        assert!(error.contains(why), "{name}: {error}");
    };
    let mut other = export_request(&source);
    other.schema = "omnimd-xyz-export-v0".into();
    refused("schema", other, "unsupported XYZ export request schema");
    refused(
        "relative",
        export_request(std::path::Path::new("source.jsonl")),
        "dataset path must be absolute",
    );
    assert!(serde_json::from_value::<XyzExportRequest>(
        json!({"schema": EXPORT_SCHEMA, "dataset": source, "stride": 2})
    )
    .is_err());
    let rewritten = |name: &str, change: &dyn Fn(&mut Vec<serde_json::Value>)| {
        let mut rows: Vec<serde_json::Value> = fs::read_to_string(&source)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        change(&mut rows);
        let path = root.join(format!("{name}.jsonl"));
        let text: String = rows.iter().map(|row| format!("{row}\n")).collect();
        fs::write(&path, text).unwrap();
        path
    };
    let broken_line = rewritten("line-break", &|rows| {
        rows[2]["data"]["id"] = json!("two\nlines");
    });
    refused(
        "line-break",
        export_request(&broken_line),
        "frame 1: an ID with control characters",
    );
    let unfinished = rewritten("unfinished", &|rows| {
        rows.pop();
    });
    refused(
        "unfinished",
        export_request(&unfinished),
        "completed record required",
    );
    let conditioned = rewritten("conditioned", &|rows| {
        rows[0]["header"]["schema"] = json!("omnimd-dataset-jsonl-v2");
        rows[0]["header"]["parameter_layout"] =
            json!({"fparam": [{"name": "temperature", "unit": "K"}], "aparam": []});
        for row in &mut rows[1..4] {
            row["data"]["parameters"] = json!({"fparam": [300.0], "aparam": [[], [], []]});
        }
    });
    refused(
        "conditioned",
        export_request(&conditioned),
        "a parameterized dataset is not exported",
    );
    assert!(!root.join("conditioned.extxyz").exists());
}

#[test]
fn the_cli_exports_a_dataset() {
    let root = root();
    let source = dataset(
        &root,
        "source.jsonl",
        serde_json::json!([[8.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 10.0]]),
    );
    let by_library = root.join("library.extxyz");
    export_xyz(&export_request(&source), &by_library).unwrap();
    let request = root.join("export.json");
    fs::write(
        &request,
        serde_json::json!({"schema": EXPORT_SCHEMA, "dataset": source}).to_string(),
    )
    .unwrap();
    let by_cli = root.join("cli.extxyz");
    let run = || {
        std::process::Command::new(env!("CARGO_BIN_EXE_omnimd-mlmd"))
            .arg("export-xyz")
            .arg(&request)
            .arg(&by_cli)
            .output()
            .unwrap()
    };
    let first = run();
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert!(String::from_utf8_lossy(&first.stdout).starts_with("Exported 3 frames"));
    assert_eq!(fs::read(&by_cli).unwrap(), fs::read(&by_library).unwrap());
    // A second time the file is in the way.
    assert!(!run().status.success());
}
