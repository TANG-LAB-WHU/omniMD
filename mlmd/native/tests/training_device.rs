//! Default tests never create a CUDA context. The single opt-in test requires
//! an explicitly authorized exclusive GPU window.
use omnimd_ml_contract::{Evaluation, Frame, Potential};
use omnimd_native::{
    readout::ReadoutBasis,
    training::{train, train_with_device, Split, TrainConfig, TrainingDevice},
    training_data::{Dataset, DatasetFrame, DatasetHeader, Labels, VirialConvention},
    Model, NativePotential,
};
use std::collections::BTreeMap;

fn config(fit_bias: bool) -> TrainConfig {
    TrainConfig {
        split: Split::Explicit {
            validation_ids: (24..32).map(|i| format!("synthetic/{i}")).collect(),
        },
        fit_bias,
        energy_weight: 1.0,
        force_weight: 1.0,
        energy_scale_ev_per_atom: 0.1,
        force_scale_ev_per_angstrom: 0.01,
        ridge_inverse_ev2: 0.0,
        rank_rtol: 1e-12,
    }
}
fn fixture(fit_bias: bool) -> (Model, Dataset) {
    let initial: Model =
        serde_json::from_str(include_str!("../fixtures/full-tanh-v1.json")).unwrap();
    let basis = ReadoutBasis::new(&initial, fit_bias).unwrap();
    let values: Vec<_> = basis
        .initial_coefficients
        .iter()
        .enumerate()
        .map(|(i, x)| x + 0.3 * ((i + 1) as f64 * 0.83).sin())
        .collect();
    let mut teacher =
        NativePotential::new(basis.model_with_coefficients(&values).unwrap()).unwrap();
    let mut frames = Vec::new();
    for index in 0..32 {
        let mut frame: Frame =
            serde_json::from_str(include_str!("../fixtures/frame.jsonl")).unwrap();
        if index % 2 == 0 {
            frame.cell = None;
        }
        if index % 3 == 0 {
            frame.positions.pop();
            frame.species.pop();
        }
        for (atom, r) in frame.positions.iter_mut().enumerate() {
            for (axis, x) in r.iter_mut().enumerate() {
                *x += 0.19 * ((index + 1) as f64 * (atom * 3 + axis + 1) as f64 * 0.618).sin();
            }
        }
        let value = teacher.eval(&frame).unwrap();
        frames.push(DatasetFrame {
            index: index as u64,
            id: format!("synthetic/{index}"),
            frame,
            labels: Labels {
                energy: Some(value.energy),
                forces: Some(value.forces),
                virial: Some(value.virial),
            },
        });
    }
    let header = DatasetHeader {
        schema: "omnimd-dataset-jsonl-v1".into(),
        type_map: initial.type_map.clone(),
        units: initial.units,
        virial_convention: VirialConvention::MinusDEpsilonRow,
        source: "original deterministic mixed-composition readout test".into(),
        selected_sets: vec!["synthetic".into()],
    };
    (
        initial,
        Dataset {
            header,
            frames,
            source_sha256: BTreeMap::new(),
            dataset_sha256: "in-memory fixture".into(),
        },
    )
}
fn compare(a: &Evaluation, b: &Evaluation, maximum: &mut [f64; 3], tolerance: f64) {
    assert_eq!(a.forces.len(), b.forces.len());
    for (k, (x, y)) in [
        (vec![a.energy], vec![b.energy]),
        (
            a.forces.iter().flatten().copied().collect(),
            b.forces.iter().flatten().copied().collect(),
        ),
        (
            a.virial.iter().flatten().copied().collect(),
            b.virial.iter().flatten().copied().collect(),
        ),
    ]
    .iter()
    .enumerate()
    {
        for (x, y) in x.iter().zip(y) {
            maximum[k] = maximum[k].max((x - y).abs());
            assert!(
                (x - y).abs() <= tolerance * (1.0 + x.abs()),
                "component {k}: {x} vs {y}"
            );
        }
    }
}
#[test]
fn explicit_devices_are_strict_and_full_gpu_is_rejected() {
    for (text, expected) in [
        ("cpu", TrainingDevice::Cpu),
        ("cuda:0", TrainingDevice::Cuda { visible_device: 0 }),
        ("cuda:12", TrainingDevice::Cuda { visible_device: 12 }),
    ] {
        let value: TrainingDevice = serde_json::from_value(text.into()).unwrap();
        assert_eq!(value, expected);
        assert_eq!(
            serde_json::to_value(value).unwrap(),
            serde_json::Value::String(text.into())
        );
    }
    for text in [
        "auto",
        "gpu",
        "cuda",
        "CUDA:0",
        "cuda:-1",
        "cuda:+1",
        "cuda: 0",
        "cuda:",
        "cuda:2147483648",
    ] {
        assert!(serde_json::from_value::<TrainingDevice>(text.into()).is_err());
    }
    assert!(serde_json::from_value::<TrainingDevice>(serde_json::json!(0)).is_err());
    assert!(TrainingDevice::Cuda { visible_device: -1 }
        .validate_readout()
        .is_err());
    assert!(TrainingDevice::Cuda { visible_device: 0 }
        .validate_full()
        .unwrap_err()
        .0
        .contains("cpu only"));
    #[cfg(not(feature = "cuda"))]
    assert!(TrainingDevice::Cuda { visible_device: 0 }
        .validate_readout()
        .unwrap_err()
        .0
        .contains("--features cuda"));
}
#[test]
fn explicit_cpu_preserves_nonzero_delta_fit_and_repeatability() {
    let (initial, dataset) = fixture(true);
    let c = config(true);
    let implicit = train(&initial, &dataset, &c).unwrap();
    let explicit = train_with_device(&initial, &dataset, &c, TrainingDevice::Cpu).unwrap();
    assert_eq!(
        serde_json::to_vec(&implicit.model).unwrap(),
        serde_json::to_vec(&explicit.model).unwrap()
    );
    assert_eq!(
        explicit.report.execution.requested_device,
        TrainingDevice::Cpu
    );
    assert!(explicit.report.final_validation.normalized_ef_objective < 1e-20);
    let mut model = NativePotential::new(explicit.model).unwrap();
    for record in &dataset.frames {
        let expected = Evaluation {
            energy: record.labels.energy.unwrap(),
            forces: record.labels.forces.clone().unwrap(),
            virial: record.labels.virial.unwrap(),
            units: record.frame.units,
        };
        compare(
            &expected,
            &model.eval(&record.frame).unwrap(),
            &mut [0.0; 3],
            1e-9,
        );
    }
}
#[cfg(feature = "cuda")]
#[test]
fn unsupported_cuda_inputs_fail_before_any_driver_creation() {
    use omnimd_native::Activation;
    let (mut initial, mut dataset) = fixture(false);
    let c = config(false);
    let device = TrainingDevice::Cuda { visible_device: 0 };
    dataset.frames[0].labels.energy = None;
    let error = train_with_device(&initial, &dataset, &c, device)
        .err()
        .unwrap();
    assert!(error.0.contains("requires finite energy"), "{error}");
    dataset.frames[0].labels.energy = Some(0.0);
    initial.descriptor.embedding[0][0].layers[0].activation = Activation::Softplus;
    let error = train_with_device(&initial, &dataset, &c, device)
        .err()
        .unwrap();
    assert!(error.0.contains("only tanh/linear"), "{error}");
    initial.descriptor.embedding[0][0].layers[0].activation = Activation::Tanh;
    let record = &mut dataset.frames[0];
    record.frame.positions = vec![[0.0; 3]; 2049];
    record.frame.species = vec!["O".into(); 2049];
    record.frame.cell = None;
    record.labels.forces = Some(vec![[0.0; 3]; 2049]);
    let error = train_with_device(&initial, &dataset, &c, device)
        .err()
        .unwrap();
    assert!(error.0.contains("at most 2048 atoms"), "{error}");
    record_design_overflow(&initial, &mut dataset, &c, device);
}
#[cfg(feature = "cuda")]
fn record_design_overflow(
    initial: &Model,
    dataset: &mut Dataset,
    config: &TrainConfig,
    device: TrainingDevice,
) {
    let record = &mut dataset.frames[0];
    record.frame.positions = vec![[0.0; 3]; 230_000];
    record.frame.species = vec!["O".into(); 230_000];
    record.labels.forces = Some(vec![[0.0; 3]; 230_000]);
    let error = train_with_device(initial, dataset, config, device)
        .err()
        .unwrap();
    assert!(error.0.contains("four million entries"), "{error}");
}

#[test]
fn full_cuda_cli_rejects_before_dataset_io_and_output_creation() {
    let suffix = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let root = std::env::temp_dir().join(format!("omnimd-full-cuda-rejection-{suffix}"));
    std::fs::create_dir(&root).unwrap();
    let request = serde_json::json!({"schema":"omnimd-native-full-train-v1", "device":"cuda:0",
        "initial_model":"missing-model.json", "dataset":"missing-dataset.jsonl", "output_directory":"must-not-exist",
        "config":{"split":{"mode":"explicit","validation_ids":["test/1"]}, "steps":1,
        "energy_weight":1.0,"force_weight":1.0,"energy_scale_ev_per_atom":1.0,"force_scale_ev_per_angstrom":1.0,
        "learning_rate":0.001,"beta1":0.9,"beta2":0.999,"epsilon":1e-8,"gradient_clip_norm":null,"checkpoint_every":1}});
    let path = root.join("request.json");
    std::fs::write(&path, serde_json::to_vec(&request).unwrap()).unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_omnimd-native-train"))
        .arg(&path)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("cpu only"));
    assert!(output.stdout.is_empty());
    assert!(!root.join("must-not-exist").exists());
    std::fs::remove_file(path).unwrap();
    std::fs::remove_dir(root).unwrap();
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an explicitly authorized exclusive GPU window"]
fn cuda_columns_fits_and_repeatability_match_cpu() {
    use omnimd_native::cuda::CudaReadoutBasis;
    let mut reports = Vec::new();
    for fit_bias in [false, true] {
        let (initial, dataset) = fixture(fit_bias);
        let mut offset_model = initial.clone();
        offset_model.atom_energy_bias = vec![-9e4, 3e4];
        let mut cpu_basis = ReadoutBasis::new(&offset_model, fit_bias).unwrap();
        let mut gpu_basis = CudaReadoutBasis::new(&offset_model, fit_bias, 0).unwrap();
        let mut column_max = [0.0; 3];
        // Changes of atom count, type sequence, cell and geometry, then reuse.
        for index in [0, 1, 3, 2, 0, 0] {
            let frame = &dataset.frames[index].frame;
            let cpu = cpu_basis.features(frame).unwrap();
            let gpu = gpu_basis.features(frame).unwrap();
            assert_eq!(cpu.fixed_energy, gpu.fixed_energy);
            for (a, b) in cpu.columns.iter().zip(&gpu.columns) {
                compare(a, b, &mut column_max, 1e-10);
            }
            compare(
                &cpu.evaluate(&cpu_basis.initial_coefficients).unwrap(),
                &gpu.evaluate(&cpu_basis.initial_coefficients).unwrap(),
                &mut [0.0; 3],
                1e-10,
            );
        }
        let c = config(fit_bias);
        let cpu = train(&initial, &dataset, &c).unwrap();
        let gpu = train_with_device(
            &initial,
            &dataset,
            &c,
            TrainingDevice::Cuda { visible_device: 0 },
        )
        .unwrap();
        let repeated = train_with_device(
            &initial,
            &dataset,
            &c,
            TrainingDevice::Cuda { visible_device: 0 },
        )
        .unwrap();
        assert_eq!(
            serde_json::to_vec(&gpu.model).unwrap(),
            serde_json::to_vec(&repeated.model).unwrap()
        );
        assert_eq!(
            serde_json::to_value(&initial.descriptor).unwrap(),
            serde_json::to_value(&gpu.model.descriptor).unwrap()
        );
        assert_eq!(initial.atom_energy_bias, gpu.model.atom_energy_bias);
        for (before, after) in initial.fitting.iter().zip(&gpu.model.fitting) {
            assert_eq!(
                serde_json::to_value(&before.layers[..before.layers.len() - 1]).unwrap(),
                serde_json::to_value(&after.layers[..after.layers.len() - 1]).unwrap()
            );
            if !fit_bias {
                assert_eq!(
                    before.layers.last().unwrap().bias,
                    after.layers.last().unwrap().bias
                );
            }
        }
        assert_eq!(cpu.report.data_rank, gpu.report.data_rank);
        assert!(gpu.report.final_validation.normalized_ef_objective < 1e-20);
        let mut cpu_p = NativePotential::new(cpu.model).unwrap();
        let mut gpu_p = NativePotential::new(gpu.model).unwrap();
        let mut prediction_max = [0.0; 3];
        for record in &dataset.frames {
            compare(
                &cpu_p.eval(&record.frame).unwrap(),
                &gpu_p.eval(&record.frame).unwrap(),
                &mut prediction_max,
                1e-9,
            );
        }
        reports.push(
            serde_json::json!({"fit_bias":fit_bias,"max_abs_column_efv":column_max,
            "max_abs_fitted_prediction_efv":prediction_max,"same_device_repeat_model_exact":true,
            "cpu_report":cpu.report,"cuda_report":gpu.report}),
        );
    }
    println!("{}",serde_json::to_string(&serde_json::json!({"scope":"opt-in synthetic CUDA readout feature/fit check; no full-network GPU training","results":reports})).unwrap());
}
