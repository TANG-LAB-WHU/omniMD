use omnimd_ml_contract::Units;
use omnimd_mlmd::{
    dataset::{DatasetReader, Pbc},
    xyz::{export_extxyz, import_xyz, XyzFormat, XyzRequest},
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
