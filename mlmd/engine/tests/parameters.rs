use omnimd_ml_contract::{Frame, Units};
use omnimd_mlmd::{
    parameters::{Field, FixedParameters, FramePolicy, Layout, Values},
    Checkpoint,
};
use serde_json::json;
fn frame() -> Frame {
    Frame {
        positions: vec![[0., 0., 0.], [1., 0., 0.]],
        species: vec!["O".into(), "H".into()],
        cell: None,
        units: Units::EvAngstrom,
    }
}
fn fixed() -> FixedParameters {
    FixedParameters::new(
        Layout {
            fparam: vec![Field {
                name: "condition".into(),
                unit: "fixture-unit".into(),
            }],
            aparam: vec![Field {
                name: "atomic_condition".into(),
                unit: "fixture-unit".into(),
            }],
        },
        Values {
            frame_parameter_policy: FramePolicy::Explicit,
            fparam: vec![0.2],
            aparam: vec![vec![0.1], vec![0.8]],
        },
        &frame(),
    )
    .unwrap()
}
#[test]
fn strict_values_defaults_and_atom_binding() {
    let original = fixed();
    original.validate(&frame()).unwrap();
    let mut f = frame();
    f.species.reverse();
    assert!(original.validate(&f).is_err());
    for i in 0..5 {
        let mut p = original.clone();
        match i {
            0 => p.values.fparam.clear(),
            1 => p.values.aparam.pop().map(|_| ()).unwrap(),
            2 => p.values.aparam[1].push(0.),
            3 => p.values.aparam[0][0] = f64::NAN,
            _ => p.layout.fparam[0].unit.clear(),
        };
        assert!(p.validate(&frame()).is_err());
    }
    let mut p = original.clone();
    p.values.frame_parameter_policy = FramePolicy::ModelDefault;
    assert!(p.validate(&frame()).is_err());
    p.values.fparam.clear();
    p.validate(&frame()).unwrap();
    p.layout.fparam.clear();
    assert!(p.validate(&frame()).is_err());
    p = original.clone();
    p.layout.aparam.clear();
    assert!(p.validate(&frame()).is_err());
    p.values.aparam.clear();
    p.validate(&frame()).unwrap();
    let values: Values =
        serde_json::from_value(json!({"fparam":[0.2],"aparam":[[0.1],[0.8]]})).unwrap();
    assert_eq!(values.frame_parameter_policy, FramePolicy::Explicit);
    assert!(serde_json::from_value::<Values>(
        json!({"frame_parameter_policy":"auto","fparam":[],"aparam":[]})
    )
    .is_err());
}
#[test]
fn v3_restart_cannot_drop_change_or_downgrade_fixed_parameters() {
    let f = frame();
    let p = fixed();
    let hash = "ab".repeat(32);
    let checkpoint:Checkpoint=serde_json::from_value(json!({"schema_version":3,"frame":f,"masses":[16.,1.],"velocities":[[0.,0.,0.],[0.,0.,0.]],"step":7,"timestep_fs":0.01,"model_sha256":hash,"fixed_parameters":p})).unwrap();
    assert!(checkpoint.restore(&hash, 0.01).is_err());
    assert_eq!(
        checkpoint
            .restore_with_parameters(&hash, 0.01, Some(&p))
            .unwrap()
            .step,
        7
    );
    let mut changed = p.clone();
    changed.values.aparam.swap(0, 1);
    assert!(checkpoint
        .restore_with_parameters(&hash, 0.01, Some(&changed))
        .is_err());
    changed = p.clone();
    changed.layout.aparam[0].unit = "different".into();
    assert!(checkpoint
        .restore_with_parameters(&hash, 0.01, Some(&changed))
        .is_err());
    let mut old = checkpoint.clone();
    old.schema_version = 2;
    assert!(old.restore_with_parameters(&hash, 0.01, Some(&p)).is_err());
    old.fixed_parameters = None;
    old.restore(&hash, 0.01).unwrap();
    assert!(old.restore_with_parameters(&hash, 0.01, Some(&p)).is_err());
    let mut absent = checkpoint.clone();
    absent.fixed_parameters = None;
    assert!(absent.restore(&hash, 0.01).is_err());
    let encoded = serde_json::to_string(&checkpoint).unwrap();
    let reloaded: Checkpoint = serde_json::from_str(&encoded).unwrap();
    assert_eq!(reloaded.fixed_parameters, Some(p));
}
