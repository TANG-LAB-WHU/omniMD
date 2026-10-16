use omnimd_ml_contract::{Frame, Potential};
use omnimd_native::{
    full_gradient::ParameterLayout,
    full_training::{train_full, BatchOrder, Checkpoint, FullConfig, MiniBatchConfig, VirialLoss},
    indexed_dataset::IndexedDataset,
    training::{Split, TrainingDevice},
    training_data::{
        Dataset, DatasetAccess, DatasetFrame, DatasetHeader, Labels, VirialConvention,
    },
    Model, NativePotential,
};
use std::{collections::BTreeMap, fs, io::Write, path::PathBuf};
static NEXT_TEMP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "omnimd-indexed-{}-{}-{}",
            NEXT_TEMP.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        for e in fs::read_dir(&self.0).unwrap() {
            let p = e.unwrap().path();
            if p.is_file() {
                let _ = fs::remove_file(p);
            }
        }
        let _ = fs::remove_dir(&self.0);
    }
}
fn write_data(path: &std::path::Path, d: &Dataset) {
    let mut f = fs::File::create(path).unwrap();
    writeln!(
        f,
        "{}",
        serde_json::json!({"kind":"header","header":d.header})
    )
    .unwrap();
    for r in &d.frames {
        writeln!(f, "{}", serde_json::json!({"kind":"frame","data":r})).unwrap();
    }
    writeln!(f,"{}",serde_json::json!({"kind":"completed","frames":d.frames.len(),"source_sha256":d.source_sha256})).unwrap();
}
fn state(s: &Checkpoint) -> serde_json::Value {
    let mut v = serde_json::to_value(s).unwrap();
    v["config"].as_object_mut().unwrap().remove("steps");
    v["config"]
        .as_object_mut()
        .unwrap()
        .remove("fixed_geometry_cache");
    v
}
fn model() -> Model {
    serde_json::from_str(include_str!("../fixtures/full-tanh-v1.json")).unwrap()
}
fn config(steps: u64) -> FullConfig {
    FullConfig {
        minibatch: Some(MiniBatchConfig {
            batch_size: 2,
            seed: 20261016,
            order: BatchOrder::ShuffleEachEpoch,
        }),
        fixed_geometry_cache: false,
        cuda_gradient: Default::default(),
        split: Split::Explicit {
            validation_ids: vec!["set.000/2".into()],
        },
        steps,
        energy_weight: 1.0,
        force_weight: 1.0,
        virial_loss: Some(VirialLoss {
            weight: 0.7,
            scale_ev_per_atom: 0.2,
        }),
        energy_scale_ev_per_atom: 0.1,
        force_scale_ev_per_angstrom: 0.1,
        learning_rate: 0.005,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        gradient_clip_norm: Some(10.0),
        checkpoint_every: 1,
    }
}
fn data(count: usize) -> (Model, Dataset) {
    let initial = model();
    let layout = ParameterLayout::new(&initial).unwrap();
    let mut teacher = initial.clone();
    let values: Vec<_> = layout
        .values(&initial)
        .iter()
        .enumerate()
        .map(|(i, x)| x + 0.12 * ((i + 1) as f64 * 0.63).sin())
        .collect();
    layout.apply(&mut teacher, &values).unwrap();
    let mut potential = NativePotential::new(teacher).unwrap();
    let mut frames = Vec::new();
    for i in 0..count {
        let mut f: Frame = serde_json::from_str(include_str!("../fixtures/frame.jsonl")).unwrap();
        if i % 2 == 0 {
            f.cell = None;
        } else {
            f.cell = Some([[3.8, 0.0, 0.0], [0.3, 3.9, 0.0], [0.1, 0.2, 4.0]]);
        }
        for (atom, r) in f.positions.iter_mut().enumerate() {
            for (axis, x) in r.iter_mut().enumerate() {
                *x += 0.19 * ((i + 1) as f64 * (atom * 3 + axis + 1) as f64 * 0.618).sin();
            }
        }
        if i % 3 == 0 {
            f.positions.pop();
            f.species.pop();
        }
        let e = potential.eval(&f).unwrap();
        frames.push(DatasetFrame {
            index: i as u64,
            id: format!("set.000/{i}"),
            frame: f,
            labels: Labels {
                energy: Some(e.energy),
                forces: Some(e.forces),
                virial: Some(e.virial),
            },
        });
    }
    let header = DatasetHeader {
        schema: "omnimd-dataset-jsonl-v1".into(),
        type_map: initial.type_map.clone(),
        units: initial.units,
        virial_convention: VirialConvention::MinusDEpsilonRow,
        source: "original synthetic full-network teacher".into(),
        selected_sets: vec!["set.000".into()],
    };
    (
        initial,
        Dataset {
            header,
            frames,
            source_sha256: BTreeMap::new(),
            dataset_sha256: "in-memory-test".into(),
        },
    )
}

#[test]
fn indexed_memory_exact_metrics_updates_and_bidirectional_resume() {
    let tmp = Temp::new();
    let path = tmp.0.join("data.jsonl");
    let (m, d) = data(6);
    write_data(&path, &d);
    let memory = Dataset::load(&path).unwrap();
    let indexed = IndexedDataset::open(&path).unwrap();
    assert_eq!(memory.dataset_sha256, indexed.dataset_sha256());
    for i in [4, 0, 5, 1, 3, 2] {
        assert_eq!(
            serde_json::to_value(indexed.frame(i).unwrap().as_ref()).unwrap(),
            serde_json::to_value(&memory.frames[i]).unwrap()
        );
    }
    let c = config(7);
    let mut saved = Vec::new();
    let a = train_full(
        Checkpoint::start(&m, &memory, &c).unwrap(),
        &memory,
        &c,
        |s| {
            saved.push(
                serde_json::from_slice::<Checkpoint>(&serde_json::to_vec(s).unwrap()).unwrap(),
            );
            Ok(())
        },
    )
    .unwrap();
    let b = train_full(
        Checkpoint::start(&m, &indexed, &c).unwrap(),
        &indexed,
        &c,
        |_| Ok(()),
    )
    .unwrap();
    assert_eq!(state(&a.checkpoint), state(&b.checkpoint));
    for (x, y) in [
        (&a.report.initial_training, &b.report.initial_training),
        (&a.report.final_training, &b.report.final_training),
        (&a.report.initial_validation, &b.report.initial_validation),
        (&a.report.final_validation, &b.report.final_validation),
    ] {
        assert_eq!(
            serde_json::to_value(x).unwrap(),
            serde_json::to_value(y).unwrap()
        );
    }
    for split in [2, 3, 4] {
        let mut r = c.clone();
        r.steps = 7 - split;
        let out = train_full(saved[split as usize - 1].clone(), &indexed, &r, |_| Ok(())).unwrap();
        assert_eq!(state(&a.checkpoint), state(&out.checkpoint));
    }
    let mut r = c.clone();
    r.steps = 2;
    let start = train_full(
        Checkpoint::start(&m, &indexed, &r).unwrap(),
        &indexed,
        &r,
        |_| Ok(()),
    )
    .unwrap();
    r.steps = 5;
    let resumed = train_full(start.checkpoint, &memory, &r, |_| Ok(())).unwrap();
    assert_eq!(state(&a.checkpoint), state(&resumed.checkpoint));
    let stats = indexed.access_report().unwrap();
    assert_eq!(stats.frames_decoded_per_read, 1);
    assert!(stats.index_payload_bytes < 64 * 1024);
    assert!(stats.peak_decoded_frame_payload_bytes < 4096);
    assert!(stats.integrity_verifications >= 7);
    // Changing only the storage policy never changes checkpoint identity.
    drop(indexed);
    let reopened = IndexedDataset::open(&path).unwrap();
    a.checkpoint
        .validate_for_device(&reopened, &c, TrainingDevice::Cpu)
        .unwrap();
    let mut legacy = c.clone();
    legacy.minibatch = None;
    assert!(Checkpoint::start(&m, &reopened, &legacy).is_err());
}
#[test]
fn indexed_trainer_validates_once_and_rejects_invalid_state_before_callback() {
    let tmp = Temp::new();
    let path = tmp.0.join("data.jsonl");
    let (m, d) = data(6);
    write_data(&path, &d);
    let indexed = IndexedDataset::open(&path).unwrap();
    let c = config(1);
    let start = Checkpoint::start(&m, &indexed, &c).unwrap();
    let reads_before = indexed.access_report().unwrap().frame_reads;
    let result = train_full(start.clone(), &indexed, &c, |_| Ok(())).unwrap();
    // One complete trainer validation plus initial train/validation assessment.
    // Constructor, index scan, updates and final assessment are separate work.
    assert_eq!(
        result.report.timing.startup_frame_reads.unwrap() - reads_before,
        2 * d.frames.len() as u64
    );
    assert_eq!(indexed.access_report().unwrap().integrity_verifications, 2);
    for case in 0..5 {
        let mut state = start.clone();
        let mut config = c.clone();
        match case {
            0 => state.first_moment[0] = 1.0,
            1 => state.dataset_sha256 = "changed".into(),
            2 => state.sampler.as_mut().unwrap().cursor = 1,
            3 => config.fixed_geometry_cache = true,
            _ => config.steps = 0,
        }
        let mut called = false;
        assert!(
            train_full(state, &indexed, &config, |_| {
                called = true;
                Ok(())
            })
            .is_err(),
            "case {case}"
        );
        assert!(!called, "case {case} published an invalid checkpoint");
    }
}

#[test]
fn indexed_strict_record_and_completion_rules() {
    let tmp = Temp::new();
    let path = tmp.0.join("data.jsonl");
    let (_, d) = data(6);
    write_data(&path, &d);
    let original = fs::read_to_string(&path).unwrap();
    let records: Vec<serde_json::Value> = original
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    for case in 0..8 {
        let mut lines = records.clone();
        match case {
            0 => {
                lines.pop();
            }
            1 => lines.push(serde_json::json!({"kind":"completed","frames":6,"source_sha256":{}})),
            2 => lines[2]["data"]["id"] = lines[1]["data"]["id"].clone(),
            3 => lines[2]["data"]["index"] = serde_json::json!(9),
            4 => lines[1]["data"]["frame"]["positions"] = serde_json::json!(vec![[0., 0., 0.]; 65]),
            5 => lines[1]["data"]["labels"]["forces"] = serde_json::Value::Null,
            6 => lines[0]["header"]["units"] = serde_json::json!("atomic"),
            _ => lines[1]["data"]["unknown"] = serde_json::json!(true),
        }
        fs::write(
            &path,
            lines.iter().map(|v| format!("{v}\n")).collect::<String>(),
        )
        .unwrap();
        assert!(IndexedDataset::open(&path).is_err(), "case {case}");
    }
    fs::write(&path, original.trim_end()).unwrap();
    assert!(IndexedDataset::open(&path).is_err());
    fs::write(
        &path,
        vec![b' '; omnimd_native::indexed_dataset::MAX_RECORD_BYTES + 1],
    )
    .unwrap();
    assert!(IndexedDataset::open(&path).is_err());
}
#[cfg(feature = "cuda")]
#[test]
fn indexed_geometry_pool_budget_rejected_before_driver() {
    let tmp = Temp::new();
    let path = tmp.0.join("data.jsonl");
    let (m, mut d) = data(6);
    let template = d.frames[1].clone();
    d.frames = (0..6000)
        .map(|i| {
            let mut f = template.clone();
            f.index = i;
            f.id = format!("set.000/{i}");
            f
        })
        .collect();
    write_data(&path, &d);
    drop(d);
    let index = IndexedDataset::open(&path).unwrap();
    let mut c = config(1);
    c.fixed_geometry_cache = true;
    let s = Checkpoint::start(&m, &index, &c).unwrap();
    let e = s
        .validate_for_device(&index, &c, TrainingDevice::Cuda { visible_device: 0 })
        .unwrap_err();
    assert!(e.0.contains("payload"), "{}", e.0);
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an explicitly authorized GPU window"]
fn indexed_cuda_cache_and_storage_resume_are_exact() {
    use omnimd_native::full_training::{train_full_with_device, CudaGradient};
    let tmp = Temp::new();
    let path = tmp.0.join("data.jsonl");
    let (m, d) = data(6);
    write_data(&path, &d);
    let memory = Dataset::load(&path).unwrap();
    let indexed = IndexedDataset::open(&path).unwrap();
    for mode in [CudaGradient::Hyperdual, CudaGradient::Reverse] {
        let mut c = config(7);
        c.cuda_gradient = mode;
        c.fixed_geometry_cache = true;
        let device = TrainingDevice::Cuda { visible_device: 0 };
        let a = train_full_with_device(
            Checkpoint::start(&m, &memory, &c).unwrap(),
            &memory,
            &c,
            device,
            |_| Ok(()),
        )
        .unwrap();
        let mut snapshots = Vec::new();
        let b = train_full_with_device(
            Checkpoint::start(&m, &indexed, &c).unwrap(),
            &indexed,
            &c,
            device,
            |s| {
                snapshots.push(
                    serde_json::from_slice::<Checkpoint>(&serde_json::to_vec(s).unwrap()).unwrap(),
                );
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(state(&a.checkpoint), state(&b.checkpoint));
        let mut r = c.clone();
        r.steps = 5;
        r.fixed_geometry_cache = false;
        let resumed =
            train_full_with_device(snapshots[1].clone(), &memory, &r, device, |_| Ok(())).unwrap();
        assert_eq!(state(&a.checkpoint), state(&resumed.checkpoint));
        assert_eq!(b.report.fixed_geometry_cache.unwrap().hits, 12);
    }
}
