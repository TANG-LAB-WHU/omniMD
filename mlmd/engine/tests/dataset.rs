use omnimd_ml_contract::{Evaluation, Units};
use omnimd_mlmd::dataset::*;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
static COUNTER: AtomicU64 = AtomicU64::new(0);
fn temp() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "omnimd-dataset-{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&path).unwrap();
    path
}
fn npy(path: &Path, shape: &[u64], values: &[f64], dtype: &str, fortran: bool) {
    let dims = shape
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let mut header = format!(
        "{{'descr': '{dtype}', 'fortran_order': {}, 'shape': ({dims},), }}",
        if fortran { "True" } else { "False" }
    );
    while (header.len() + 11) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for x in values {
        bytes.extend_from_slice(&if dtype == ">f8" {
            x.to_be_bytes()
        } else {
            x.to_le_bytes()
        });
    }
    fs::write(path, bytes).unwrap();
}
fn system(periodic: bool) -> (PathBuf, ImportRequest) {
    let root = temp();
    fs::write(root.join("type_map.raw"), "O H\n").unwrap();
    fs::write(root.join("type.raw"), "1 0\n").unwrap();
    if !periodic {
        fs::write(root.join("nopbc"), "").unwrap();
    }
    for set in ["set.000", "set.001"] {
        fs::create_dir(root.join(set)).unwrap();
        npy(
            &root.join(set).join("coord.npy"),
            &[1, 6],
            &[1., 2., 3., 4., 5., 6.],
            "<f8",
            false,
        );
        if periodic {
            npy(
                &root.join(set).join("box.npy"),
                &[1, 9],
                &[8., 0., 0., 1., 9., 0., 2., 3., 10.],
                "<f8",
                false,
            );
        }
    }
    let request = ImportRequest {
        system: root.clone(),
        sets: vec!["set.001".into(), "set.000".into()],
        units: Units::EvAngstrom,
        pbc: if periodic {
            Pbc::Periodic
        } else {
            Pbc::NonPeriodic
        },
    };
    (root, request)
}
#[test]
fn import_preserves_explicit_order_symbols_tensor_layout_and_optional_labels() {
    let (root, request) = system(true);
    npy(&root.join("set.001/energy.npy"), &[1], &[4.], ">f8", false);
    npy(
        &root.join("set.000/energy.npy"),
        &[1, 1],
        &[6.],
        "<f8",
        false,
    );
    npy(
        &root.join("set.001/virial.npy"),
        &[1, 9],
        &[1., 2., 3., 4., 5., 6., 7., 8., 9.],
        "<f8",
        false,
    );
    let output = root.join("dataset.jsonl");
    assert_eq!(import_deepmd(&request, &output).unwrap(), 2);
    let mut reader = DatasetReader::open(&output).unwrap();
    let first = reader.next_frame().unwrap().unwrap();
    assert_eq!(first.id, "set.001/0");
    assert_eq!(first.frame.species, ["H", "O"]);
    assert_eq!(first.frame.cell.unwrap()[2], [2., 3., 10.]);
    assert_eq!(first.labels.virial.unwrap()[1], [4., 5., 6.]);
    assert_eq!(first.labels.energy, Some(4.));
    assert!(first.labels.forces.is_none());
    assert_eq!(
        reader.next_frame().unwrap().unwrap().labels.energy,
        Some(6.)
    );
    assert!(reader.next_frame().unwrap().is_none());
    assert!(import_deepmd(&request, &output).is_err());
}
#[test]
fn importer_rejects_wrong_pbc_dtype_shape_values_types_and_fields() {
    let mutations: Vec<Box<dyn Fn(&Path, &mut ImportRequest)>> = vec![
        Box::new(|_, r| r.pbc = Pbc::Periodic),
        Box::new(|p, _| {
            npy(
                &p.join("set.001/coord.npy"),
                &[u64::MAX, u64::MAX],
                &[],
                "<f8",
                false,
            )
        }),
        Box::new(|p, _| {
            npy(
                &p.join("set.001/coord.npy"),
                &[1, 6],
                &[0.; 7],
                "<f8",
                false,
            )
        }),
        Box::new(|_, r| r.sets.push(r.sets[0].clone())),
        Box::new(|p, _| fs::write(p.join("type.raw"), "2 0").unwrap()),
        Box::new(|p, _| fs::write(p.join("type.raw"), "").unwrap()),
        Box::new(|p, _| fs::write(p.join("type_map.raw"), "O O").unwrap()),
        Box::new(|p, _| npy(&p.join("set.001/coord.npy"), &[1, 6], &[0.; 6], "<f8", true)),
        Box::new(|p, _| {
            npy(
                &p.join("set.001/coord.npy"),
                &[1, 6],
                &[0.; 6],
                "<i8",
                false,
            )
        }),
        Box::new(|p, _| {
            npy(
                &p.join("set.001/coord.npy"),
                &[1, 6],
                &[f64::NAN; 6],
                "<f8",
                false,
            )
        }),
        Box::new(|p, _| {
            npy(
                &p.join("set.001/coord.npy"),
                &[1, 6],
                &[0.; 5],
                "<f8",
                false,
            )
        }),
        Box::new(|p, _| {
            npy(
                &p.join("set.001/force.npy"),
                &[2, 6],
                &[0.; 12],
                "<f8",
                false,
            )
        }),
        Box::new(|p, _| fs::write(p.join("set.001/real_atom_types.npy"), "").unwrap()),
        Box::new(|p, _| fs::write(p.join("fparam.raw"), "").unwrap()),
        Box::new(|p, _| npy(&p.join("set.001/box.npy"), &[1, 9], &[0.; 9], "<f8", false)),
    ];
    for (i, mutate) in mutations.iter().enumerate() {
        let (root, mut request) = system(false);
        mutate(&root, &mut request);
        assert!(
            import_deepmd(&request, &root.join("data.jsonl")).is_err(),
            "case {i}"
        );
    }
    let (root, request) = system(true);
    npy(
        &root.join("set.001/box.npy"),
        &[1, 9],
        &[0.; 9],
        "<f8",
        false,
    );
    assert!(import_deepmd(&request, &root.join("data.jsonl")).is_err());
    let (root, request) = system(false);
    let output = root.join("data.jsonl");
    import_deepmd(&request, &output).unwrap();
    let reference = FrameReference {
        dataset: output,
        index: 1,
    };
    assert!(reference.resolve().unwrap().cell.is_none());
}
#[test]
fn dataset_requires_exact_terminal_record_and_unique_frame_identity() {
    let (root, request) = system(false);
    let output = root.join("data.jsonl");
    import_deepmd(&request, &output).unwrap();
    let text = fs::read_to_string(output).unwrap();
    let lines: Vec<_> = text.lines().collect();
    let duplicate = lines[2].replace("set.000/0", "set.001/0");
    let corruptions = [
        format!("{}\n{}\n", lines[0], lines[1]),
        format!("{}\n{}\n{}\n{}\n", lines[0], lines[1], duplicate, lines[3]),
        format!("{}\n{}\n", text.trim_end(), lines[1]),
        text.replace("\"frames\":2", "\"frames\":3"),
        text.trim_end().to_owned(),
    ];
    for (i, contents) in corruptions.iter().enumerate() {
        let path = root.join(format!("bad{i}.jsonl"));
        fs::write(&path, contents).unwrap();
        let result = (|| -> Result<()> {
            let mut reader = DatasetReader::open(&path)?;
            while reader.next_frame()?.is_some() {}
            Ok(())
        })();
        assert!(result.is_err(), "case {i}");
    }
}
#[test]
fn metrics_use_actual_label_counts_atom_normalization_and_all_tensor_entries() {
    let mut metrics = Metrics::default();
    for n in [1, 2] {
        let prediction = Evaluation {
            energy: 4.,
            forces: vec![[1., 2., 3.]; n],
            virial: [[2.; 3]; 3],
            units: Units::EvAngstrom,
        };
        let labels = Labels {
            energy: Some(2.),
            forces: (n == 2).then(|| vec![[0.; 3]; n]),
            virial: None,
        };
        metrics.add(&prediction, &labels, n).unwrap();
    }
    let report = metrics.report();
    assert_eq!(report["energy_total"]["count"], 2);
    assert_eq!(report["energy_total"]["mae"], 2.);
    assert_eq!(report["energy_per_atom"]["mae"], 1.5);
    assert!((report["energy_per_atom"]["rmse"].as_f64().unwrap() - 2.5f64.sqrt()).abs() < 1e-14);
    assert_eq!(report["force_components"]["count"], 6);
    assert_eq!(report["force_components"]["mae"], 2.);
    assert!(report["virial_components"]["rmse"].is_null());
    assert_eq!(report["virial_components"]["count"], 0);
    let mut huge = Metric::default();
    huge.add(1e300).unwrap();
    huge.add(-1e300).unwrap();
    assert_eq!(huge.report("eV")["rmse"].as_f64().unwrap(), 1e300);
    assert!(huge.add(f64::INFINITY).is_err());
}

#[test]
fn inspection_hashes_original_bytes_and_reports_mixed_label_coverage() {
    let (root, request) = system(false);
    let input = root.join("inspect.jsonl");
    import_deepmd(&request, &input).unwrap();
    let text = fs::read_to_string(&input).unwrap();
    let mut records: Vec<serde_json::Value> = text
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    records[1]["data"]["labels"] =
        serde_json::json!({"energy": 2., "forces": [[0.,0.,0.],[0.,0.,0.]], "virial": null});
    records[2]["data"]["frame"]["cell"] =
        serde_json::json!([[8., 0., 0.], [1., 9., 0.], [0., 0., 10.]]);
    records[2]["data"]["labels"] = serde_json::json!({"energy": null,"forces":null,"virial":[[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]});
    // Deliberately preserve CRLF and whitespace: reserialized JSON hashes differ.
    let bytes = records
        .iter()
        .map(|r| format!("  {}  \r\n", serde_json::to_string(r).unwrap()))
        .collect::<String>();
    fs::write(&input, &bytes).unwrap();
    let report = inspect_dataset(&input).unwrap();
    assert_eq!(report.dataset_sha256, sha256(&input).unwrap());
    assert_eq!(report.frames, 2);
    assert_eq!(
        (report.min_atoms_per_frame, report.max_atoms_per_frame),
        (2, 2)
    );
    assert_eq!(report.atom_occurrences, 4);
    assert_eq!(report.species_atom_occurrences["H"], 2);
    assert_eq!((report.periodic_frames, report.nonperiodic_frames), (1, 1));
    assert_eq!(report.labels.energy_frames, 1);
    assert_eq!(report.labels.force_frames, 1);
    assert_eq!(report.labels.virial_frames, 1);
    assert_eq!(report.labels.energy_and_force_frames, 1);
    assert!(!report.labels.all_frames_have_energy_and_forces);
    assert_eq!(report.labels.missing_examples[0].id, "set.000/0");
    assert!(!report.labels.missing_examples_truncated);
    assert_eq!(
        serde_json::to_value(&report.declared_source_sha256).unwrap(),
        records[3]["source_sha256"]
    );
    assert!(!report.source_hashes_verified);
    records[2]["data"]["labels"] = records[1]["data"]["labels"].clone();
    fs::write(
        &input,
        records.iter().map(|r| format!("{r}\n")).collect::<String>(),
    )
    .unwrap();
    assert!(
        inspect_dataset(&input)
            .unwrap()
            .labels
            .all_frames_have_energy_and_forces
    );
}

#[test]
fn inspection_bounds_missing_examples_and_cli_does_not_write_on_invalid_input() {
    let (root, request) = system(false);
    let input = root.join("inspect.jsonl");
    import_deepmd(&request, &input).unwrap();
    let records: Vec<serde_json::Value> = fs::read_to_string(&input)
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    let mut text = format!("{}\n", records[0]);
    for index in 0..40 {
        let mut r = records[1].clone();
        r["data"]["index"] = index.into();
        r["data"]["id"] = format!("example/{index}").into();
        text.push_str(&format!("{r}\n"));
    }
    let mut completed = records[3].clone();
    completed["frames"] = 40.into();
    fs::write(&input, format!("{text}{completed}\n")).unwrap();
    let report = inspect_dataset(&input).unwrap();
    assert_eq!(report.labels.missing_energy_or_force_frames, 40);
    assert_eq!(report.labels.missing_examples.len(), 32);
    assert!(report.labels.missing_examples_truncated);
    let output = root.join("report.json");
    let cli = env!("CARGO_BIN_EXE_omnimd-mlmd");
    let valid = std::process::Command::new(cli)
        .arg("inspect-dataset")
        .arg(&input)
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        valid.status.success(),
        "{}",
        String::from_utf8_lossy(&valid.stderr)
    );
    let original = fs::read(&output).unwrap();
    assert!(!std::process::Command::new(cli)
        .arg("inspect-dataset")
        .arg(&input)
        .arg(&output)
        .output()
        .unwrap()
        .status
        .success());
    assert_eq!(fs::read(&output).unwrap(), original);
    // Truncated and trailing files must not produce a misleading success report.
    for (i, invalid) in [
        text,
        format!("{}{{}}\n", fs::read_to_string(&input).unwrap()),
    ]
    .iter()
    .enumerate()
    {
        let bad = root.join(format!("invalid-{i}.jsonl"));
        let absent = root.join(format!("absent-{i}.json"));
        fs::write(&bad, invalid).unwrap();
        let result = std::process::Command::new(cli)
            .arg("inspect-dataset")
            .arg(bad)
            .arg(&absent)
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(!absent.exists());
    }
}
