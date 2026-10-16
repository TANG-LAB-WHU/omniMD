use super::*;
use std::cell::RefCell;

#[derive(Default)]
struct Counts {
    loads: usize,
    drops: usize,
    strings: usize,
    frees: usize,
    computes: usize,
    default_queries: usize,
    fparam_was_null: bool,
    types: Vec<i32>,
    coords: Vec<f64>,
    cell: Option<Vec<f64>>,
    frames: i32,
    optional_were_null: bool,
    fparam: Vec<f64>,
    aparam: Vec<f64>,
}
thread_local! { static COUNTS: RefCell<Counts> = RefCell::new(Counts::default()); }
struct Mock {
    mode: String,
    error: String,
}

unsafe fn state<'a>(h: Handle) -> &'a mut Mock {
    unsafe { &mut *h.cast::<Mock>() }
}
unsafe extern "C" fn new(
    path: *const c_char,
    rank: i32,
    content: *const c_char,
    size: i32,
) -> Handle {
    COUNTS.with(|c| c.borrow_mut().loads += 1);
    let mode = unsafe { CStr::from_ptr(path) }
        .to_string_lossy()
        .into_owned();
    if mode == "null" {
        return ptr::null_mut();
    }
    let error = if mode == "load-error" {
        "bad frozen model"
    } else if rank != 0 || content.is_null() || size != 0 {
        "bad constructor arguments"
    } else {
        ""
    }
    .into();
    Box::into_raw(Box::new(Mock { mode, error })).cast()
}
unsafe extern "C" fn delete(h: Handle) {
    COUNTS.with(|c| c.borrow_mut().drops += 1);
    drop(unsafe { Box::from_raw(h.cast::<Mock>()) });
}
fn ctext(s: &str) -> *const c_char {
    COUNTS.with(|c| c.borrow_mut().strings += 1);
    CString::new(s).unwrap().into_raw()
}
unsafe extern "C" fn check(h: Handle) -> *const c_char {
    ctext(&unsafe { state(h) }.error)
}
unsafe extern "C" fn map(h: Handle) -> *const c_char {
    if unsafe { state(h) }.mode == "empty-map" {
        ctext("")
    } else {
        ctext("O H")
    }
}
unsafe extern "C" fn delete_text(s: *const c_char) {
    COUNTS.with(|c| c.borrow_mut().frees += 1);
    drop(unsafe { CString::from_raw(s.cast_mut()) });
}
unsafe extern "C" fn ntypes(h: Handle) -> i32 {
    if unsafe { state(h) }.mode == "bad-ntypes" {
        3
    } else {
        2
    }
}
unsafe extern "C" fn nspin(h: Handle) -> i32 {
    i32::from(unsafe { state(h) }.mode == "spin")
}
unsafe extern "C" fn fparam(h: Handle) -> i32 {
    match unsafe { state(h) }.mode.as_str() {
        "fparam" => 1,
        "parameters" | "default-fparam" | "default-parameters" | "default-compute-error" => 2,
        "bad-dim" => -1,
        _ => 0,
    }
}
unsafe extern "C" fn aparam(h: Handle) -> i32 {
    i32::from(matches!(
        unsafe { state(h) }.mode.as_str(),
        "aparam" | "parameters" | "default-parameters"
    ))
}
unsafe extern "C" fn charge_spin(h: Handle) -> i32 {
    i32::from(unsafe { state(h) }.mode == "charge-spin")
}
unsafe extern "C" fn has_default_fparam(h: Handle) -> bool {
    COUNTS.with(|c| c.borrow_mut().default_queries += 1);
    let m = unsafe { state(h) };
    if m.mode == "default-query-error" {
        m.error = "injected default metadata error".into();
    }
    matches!(
        m.mode.as_str(),
        "default-fparam" | "default-parameters" | "default-compute-error"
    )
}
unsafe extern "C" fn cutoff(h: Handle) -> f64 {
    if unsafe { state(h) }.mode == "bad-cutoff" {
        f64::NAN
    } else {
        6.0
    }
}
unsafe extern "C" fn compute(
    h: Handle,
    nf: i32,
    n: i32,
    x: *const f64,
    t: *const i32,
    cell: *const f64,
    fp: *const f64,
    ap: *const f64,
    energy: *mut f64,
    force: *mut f64,
    virial: *mut f64,
    ae: *mut f64,
    av: *mut f64,
) {
    let count = nf as usize;
    let n = n as usize;
    let df = unsafe { fparam(h) } as usize;
    let da = unsafe { aparam(h) } as usize;
    let coords = unsafe { std::slice::from_raw_parts(x, count * n * 3) };
    COUNTS.with(|c| {
        let mut c = c.borrow_mut();
        c.computes += 1;
        c.fparam_was_null = fp.is_null();
        c.frames = nf;
        c.coords = coords.to_vec();
        c.types = unsafe { std::slice::from_raw_parts(t, n) }.to_vec();
        c.cell = if cell.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(cell, count * 9) }.to_vec())
        };
        c.fparam = if fp.is_null() {
            vec![]
        } else {
            unsafe { std::slice::from_raw_parts(fp, count * df) }.to_vec()
        };
        c.aparam = if ap.is_null() {
            vec![]
        } else {
            unsafe { std::slice::from_raw_parts(ap, count * n * da) }.to_vec()
        };
        c.optional_were_null = fp.is_null() && ap.is_null() && ae.is_null() && av.is_null();
    });
    let mock = unsafe { state(h) };
    if matches!(
        mock.mode.as_str(),
        "compute-error" | "default-compute-error"
    ) {
        mock.error = "injected compute error".into();
        return;
    }
    for f in 0..count {
        let offset = f * n * 3;
        let e = coords[offset..offset + n * 3]
            .iter()
            .map(|v| v * v * 0.5)
            .sum();
        unsafe {
            *energy.add(f) = e;
        }
        for i in 0..n * 3 {
            unsafe {
                *force.add(offset + i) = -coords[offset + i];
            }
        }
        if mock.mode != "missing-virial" {
            for i in 0..9 {
                unsafe {
                    *virial.add(f * 9 + i) = (i + 1) as f64;
                }
            }
        }
        if !ae.is_null() && mock.mode != "missing-atomic" {
            for atom in 0..n {
                unsafe {
                    *ae.add(f * n + atom) = coords[offset + atom * 3..offset + (atom + 1) * 3]
                        .iter()
                        .map(|v| v * v * 0.5)
                        .sum();
                }
            }
        }
        if !av.is_null() && mock.mode != "missing-atomic" {
            for atom in 0..n {
                for i in 0..9 {
                    unsafe {
                        *av.add((f * n + atom) * 9 + i) = (i + 1) as f64 / n as f64;
                    }
                }
            }
        }
    }
    if mock.mode == "nonfinite-force" {
        unsafe {
            *force = f64::INFINITY;
        }
    }
    if mock.mode == "nonfinite-last-frame" {
        unsafe {
            *energy.add(count - 1) = f64::NAN;
        }
    }
}

fn api() -> Api {
    Api {
        new,
        delete,
        check,
        type_map: map,
        delete_text,
        ntypes,
        nspin,
        fparam,
        aparam,
        charge_spin: Some(charge_spin),
        has_default_fparam: Some(has_default_fparam),
        cutoff,
        compute,
        _library: None,
    }
}
fn model(mode: &str) -> Result<DeepMd> {
    DeepMd::initialize(
        api(),
        &CString::new(mode).unwrap(),
        &["O".into(), "H".into()],
        Units::EvAngstrom,
    )
}
fn reset() {
    COUNTS.with(|c| *c.borrow_mut() = Counts::default());
}
fn frame() -> Frame {
    Frame {
        positions: vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        species: vec!["H".into(), "O".into()],
        cell: None,
        units: Units::EvAngstrom,
    }
}

#[test]
fn single_load_combined_eval_row_cell_and_owned_strings() {
    reset();
    let mut m = model("ok").unwrap();
    let first = m.eval(&frame()).unwrap();
    assert_eq!(first.energy, 45.5);
    assert_eq!(first.forces, vec![[-1., -2., -3.], [-4., -5., -6.]]);
    assert_eq!(first.virial, [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
    COUNTS.with(|c| assert!(c.borrow().cell.is_none()));
    let mut f = frame();
    f.cell = Some([[8., 0., 0.], [1., 9., 0.], [2., 3., 10.]]);
    m.eval(&f).unwrap();
    drop(m);
    COUNTS.with(|c| {
        let c = c.borrow();
        assert_eq!((c.loads, c.drops, c.computes, c.frames), (1, 1, 2, 1));
        assert_eq!(c.types, vec![1, 0]);
        assert_eq!(c.coords, vec![1., 2., 3., 4., 5., 6.]);
        assert_eq!(c.cell, Some(vec![8., 0., 0., 1., 9., 0., 2., 3., 10.]));
        assert!(c.optional_were_null);
        assert_eq!(c.strings, c.frees);
    });
}

#[test]
fn failed_initialization_releases_model_and_strings() {
    for mode in [
        "load-error",
        "empty-map",
        "bad-ntypes",
        "spin",
        "bad-dim",
        "charge-spin",
        "bad-cutoff",
        "default-query-error",
    ] {
        reset();
        assert!(model(mode).is_err(), "{mode}");
        COUNTS.with(|c| {
            let c = c.borrow();
            assert_eq!((c.loads, c.drops), (1, 1), "{mode}");
            assert_eq!(c.strings, c.frees, "{mode}");
        });
    }
    reset();
    assert!(model("null").is_err());
    COUNTS.with(|c| assert_eq!(c.borrow().drops, 0));
}

#[test]
fn type_order_mismatch_is_not_silently_permuted() {
    reset();
    let result = DeepMd::initialize(
        api(),
        &CString::new("ok").unwrap(),
        &["H".into(), "O".into()],
        Units::EvAngstrom,
    );
    assert!(result.err().unwrap().0.contains("differs"));
    COUNTS.with(|c| assert_eq!(c.borrow().drops, 1));
}

#[test]
fn rejects_invalid_frames_before_calling_runtime() {
    reset();
    let mut m = model("ok").unwrap();
    let mut f = frame();
    f.species[0] = "Ne".into();
    assert!(m.eval(&f).is_err());
    f = frame();
    f.positions[0][1] = f64::NAN;
    assert!(m.eval(&f).is_err());
    f = frame();
    f.species.pop();
    assert!(m.eval(&f).is_err());
    COUNTS.with(|c| assert_eq!(c.borrow().computes, 0));
}

#[test]
fn missing_nonfinite_and_runtime_errors_never_return_results() {
    for (mode, message) in [
        ("missing-virial", "nonfinite"),
        ("nonfinite-force", "nonfinite"),
        ("compute-error", "injected compute error"),
    ] {
        let mut m = model(mode).unwrap();
        let error = m.eval(&frame()).unwrap_err();
        assert!(error.0.contains(message), "{error}");
    }
}

#[test]
fn omitted_device_defaults_to_cpu_without_cuda_dependencies() {
    let options: LoadOptions = serde_json::from_value(serde_json::json!({
        "library":"/libdeepmd.so", "model":"/model.pb", "type_map":["O","H"], "units":"EvAngstrom"
    }))
    .unwrap();
    assert_eq!(options.device, Device::Cpu);
    assert!(options.cuda.is_none());
}

#[test]
fn combined_batch_preserves_frame_atom_cell_order_and_atomic_outputs() {
    reset();
    let mut model = model("parameters").unwrap();
    assert_eq!(model.details().frame_parameter_dim, 2);
    assert_eq!(model.details().atom_parameter_dim, 1);
    let mut f0 = frame();
    f0.cell = Some([[8., 0., 0.], [1., 9., 0.], [2., 3., 10.]]);
    let mut f1 = f0.clone();
    f1.positions[0] = [7., 8., 9.];
    f1.cell.as_mut().unwrap()[0][0] = 11.;
    let params = [
        EvalParameters {
            fparam: vec![300., 1.],
            aparam: vec![vec![2.], vec![3.]],
        },
        EvalParameters {
            fparam: vec![600., 4.],
            aparam: vec![vec![5.], vec![6.]],
        },
    ];
    let outputs = model.evaluate(&[f0, f1], Some(&params), true).unwrap();
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].evaluation.energy, 45.5);
    assert_eq!(outputs[1].evaluation.energy, 135.5);
    assert_eq!(outputs[1].evaluation.forces[0], [-7., -8., -9.]);
    for result in &outputs {
        let atomic = result.atomic.as_ref().unwrap();
        assert_eq!(
            atomic.energies.iter().sum::<f64>(),
            result.evaluation.energy
        );
        assert_eq!(atomic.virials.len(), 2);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(
                    atomic.virials.iter().map(|a| a[i][j]).sum::<f64>(),
                    result.evaluation.virial[i][j]
                );
            }
        }
    }
    drop(model);
    COUNTS.with(|c| {
        let c = c.borrow();
        assert_eq!((c.loads, c.computes, c.frames, c.drops), (1, 1, 2, 1));
        assert_eq!(
            c.coords,
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 4., 5., 6.]
        );
        assert_eq!(c.types, vec![1, 0]);
        assert_eq!(c.fparam, vec![300., 1., 600., 4.]);
        assert_eq!(c.aparam, vec![2., 3., 5., 6.]);
        assert_eq!(
            c.cell.as_ref().unwrap(),
            &vec![8., 0., 0., 1., 9., 0., 2., 3., 10., 11., 0., 0., 1., 9., 0., 2., 3., 10.]
        );
        assert_eq!(c.strings, c.frees);
    });
}

#[test]
fn parameter_lengths_and_finiteness_fail_before_ffi() {
    reset();
    let mut m = model("parameters").unwrap();
    assert!(m
        .eval(&frame())
        .unwrap_err()
        .0
        .contains("explicit parameters"));
    let good = EvalParameters {
        fparam: vec![300., 1.],
        aparam: vec![vec![2.], vec![3.]],
    };
    assert!(m.evaluate(&[frame()], Some(&[]), false).is_err());
    for bad in [
        EvalParameters {
            fparam: vec![300.],
            ..good.clone()
        },
        EvalParameters {
            fparam: vec![300., f64::NAN],
            ..good.clone()
        },
        EvalParameters {
            aparam: vec![vec![2.]],
            ..good.clone()
        },
        EvalParameters {
            aparam: vec![vec![2.], vec![3., 4.]],
            ..good.clone()
        },
        EvalParameters {
            aparam: vec![vec![2.], vec![f64::INFINITY]],
            ..good.clone()
        },
    ] {
        assert!(m.evaluate(&[frame()], Some(&[bad]), false).is_err());
    }
    COUNTS.with(|c| assert_eq!(c.borrow().computes, 0));
    assert!(m.evaluate(&[frame()], Some(&[good]), false).is_ok());
}

#[test]
fn heterogeneous_batches_and_overflow_fail_before_ffi() {
    reset();
    let mut m = model("ok").unwrap();
    assert!(m.evaluate(&[], None, false).is_err());
    let mut changed = frame();
    changed.species.swap(0, 1);
    assert!(m.evaluate(&[frame(), changed], None, false).is_err());
    let mut changed = frame();
    changed.positions.pop();
    changed.species.pop();
    assert!(m.evaluate(&[frame(), changed], None, false).is_err());
    let mut changed = frame();
    changed.cell = Some([[8., 0., 0.], [0., 8., 0.], [0., 0., 8.]]);
    assert!(m.evaluate(&[frame(), changed], None, false).is_err());
    assert!(c_product(&[2, c_int::MAX as usize]).is_err());
    assert!(c_product(&[usize::MAX, 2]).is_err());
    assert_eq!(c_product(&[2, 3, 9]).unwrap(), 54);
    COUNTS.with(|c| assert_eq!(c.borrow().computes, 0));
}

#[test]
fn whole_batch_and_atomic_outputs_are_validated_before_return() {
    let mut m = model("nonfinite-last-frame").unwrap();
    assert!(m.evaluate(&[frame(), frame()], None, false).is_err());
    let mut m = model("missing-atomic").unwrap();
    assert!(m
        .evaluate(&[frame()], None, true)
        .unwrap_err()
        .0
        .contains("atomic outputs"));
    let mut m = model("ok").unwrap();
    m.evaluate(&[frame()], None, true).unwrap();
    let result = m.evaluate(&[frame(), frame()], None, false).unwrap();
    assert!(result.iter().all(|r| r.atomic.is_none()));
    COUNTS.with(|c| assert!(c.borrow().optional_were_null));
}

#[test]
fn legacy_runtime_without_optional_charge_metadata_still_loads() {
    let mut api = api();
    api.charge_spin = None;
    let mut m = DeepMd::initialize(
        api,
        &CString::new("ok").unwrap(),
        &["O".into(), "H".into()],
        Units::EvAngstrom,
    )
    .unwrap();
    assert_eq!(m.eval(&frame()).unwrap().energy, 45.5);
}

fn default_options(atomic: bool) -> EvaluationOptions {
    EvaluationOptions {
        atomic,
        frame_parameter_policy: FrameParameterPolicy::ModelDefault,
    }
}

#[test]
fn optional_default_metadata_is_cached_and_legacy_json_stays_readable() {
    reset();
    let old: ModelDetails =
        serde_json::from_str(r#"{"frame_parameter_dim":2,"atom_parameter_dim":0}"#).unwrap();
    assert_eq!(old.has_default_fparam, None);
    let mut api = api();
    api.has_default_fparam = None;
    let mut legacy = DeepMd::initialize(
        api,
        &CString::new("default-fparam").unwrap(),
        &["O".into(), "H".into()],
        Units::EvAngstrom,
    )
    .unwrap();
    assert_eq!(legacy.details().has_default_fparam, None);
    assert!(legacy
        .evaluate_with_options(&[frame()], None, default_options(false))
        .unwrap_err()
        .0
        .contains("unknown"));
    assert_eq!(
        model("fparam").unwrap().details().has_default_fparam,
        Some(false)
    );
    let mut known = model("default-fparam").unwrap();
    assert_eq!(known.details().has_default_fparam, Some(true));
    known
        .evaluate_with_options(&[frame()], None, default_options(false))
        .unwrap();
    known
        .evaluate_with_options(&[frame()], None, default_options(false))
        .unwrap();
    COUNTS.with(|c| assert_eq!(c.borrow().default_queries, 2));
}

#[test]
fn existing_apis_stay_strict_even_when_model_has_defaults() {
    reset();
    let mut m = model("default-fparam").unwrap();
    assert!(m.eval(&frame()).is_err());
    assert!(m.evaluate(&[frame()], None, false).is_err());
    assert!(m
        .evaluate_with_options(&[frame()], None, EvaluationOptions::default())
        .is_err());
    COUNTS.with(|c| assert_eq!(c.borrow().computes, 0));
    let p = EvalParameters {
        fparam: vec![3., 4.],
        aparam: vec![],
    };
    m.evaluate(&[frame()], Some(&[p]), false).unwrap();
    COUNTS.with(|c| {
        assert!(!c.borrow().fparam_was_null);
        assert_eq!(c.borrow().fparam, vec![3., 4.]);
    });
    // A later explicit opt-in must clear the reused explicit parameter buffer.
    m.evaluate_with_options(&[frame()], None, default_options(false))
        .unwrap();
    COUNTS.with(|c| assert!(c.borrow().fparam_was_null));
}

#[test]
fn default_policy_preserves_atomic_parameters_and_combined_atomic_call() {
    reset();
    let mut m = model("default-parameters").unwrap();
    let params = EvalParameters {
        fparam: vec![],
        aparam: vec![vec![2.], vec![5.]],
    };
    let out = m
        .evaluate_with_options(&[frame()], Some(&[params]), default_options(true))
        .unwrap();
    assert_eq!(out[0].evaluation.energy, 45.5);
    assert!(out[0].atomic.is_some());
    drop(m);
    COUNTS.with(|c| {
        let c = c.borrow();
        assert!(c.fparam_was_null);
        assert_eq!(c.aparam, vec![2., 5.]);
        assert_eq!(
            (c.loads, c.drops, c.computes, c.frames, c.default_queries),
            (1, 1, 1, 1, 1)
        );
        assert_eq!(c.strings, c.frees);
    });
}

#[test]
fn default_policy_rejects_conflicts_missing_defaults_and_batches_before_ffi() {
    reset();
    let mut m = model("fparam").unwrap();
    assert!(m
        .evaluate_with_options(&[frame()], None, default_options(false))
        .unwrap_err()
        .0
        .contains("no stored"));
    let mut m = model("ok").unwrap();
    assert!(m
        .evaluate_with_options(&[frame()], None, default_options(false))
        .unwrap_err()
        .0
        .contains("positive"));
    let mut m = model("default-fparam").unwrap();
    assert!(m
        .evaluate_with_options(&[frame(), frame()], None, default_options(false))
        .unwrap_err()
        .0
        .contains("one frame"));
    let p = EvalParameters {
        fparam: vec![1., 2.],
        aparam: vec![],
    };
    assert!(m
        .evaluate_with_options(&[frame()], Some(&[p]), default_options(false))
        .unwrap_err()
        .0
        .contains("conflicts"));
    let mut m = model("default-parameters").unwrap();
    assert!(m
        .evaluate_with_options(&[frame()], None, default_options(false))
        .unwrap_err()
        .0
        .contains("aparam"));
    for rows in [
        vec![],
        vec![vec![1.]],
        vec![vec![1.], vec![f64::NAN]],
        vec![vec![1.], vec![2., 3.]],
    ] {
        let p = EvalParameters {
            fparam: vec![],
            aparam: rows,
        };
        assert!(m
            .evaluate_with_options(&[frame()], Some(&[p]), default_options(false))
            .is_err());
    }
    COUNTS.with(|c| assert_eq!(c.borrow().computes, 0));
}

#[test]
fn default_runtime_error_does_not_retry_or_publish_results() {
    reset();
    let mut m = model("default-compute-error").unwrap();
    assert!(m
        .evaluate_with_options(&[frame()], None, default_options(true))
        .unwrap_err()
        .0
        .contains("injected compute error"));
    drop(m);
    COUNTS.with(|c| {
        let c = c.borrow();
        assert_eq!((c.computes, c.loads, c.drops), (1, 1, 1));
        assert_eq!(c.strings, c.frees);
    });
}

#[test]
fn pytorch_single_frame_limit_rejects_explicit_batch_before_ffi() {
    reset();
    let parameters = EvalParameters {
        fparam: vec![300., 1.],
        aparam: vec![vec![2.], vec![3.]],
    };
    let mut pytorch = model("parameters").unwrap();
    pytorch.configure_runtime_batch_limit(Path::new("frozen.pth"));
    let error = pytorch
        .evaluate(
            &[frame(), frame()],
            Some(&[parameters.clone(), parameters.clone()]),
            true,
        )
        .unwrap_err();
    assert!(error.0.contains("PyTorch .pth"));
    assert!(error.0.contains("batch_size=1"));
    COUNTS.with(|c| assert_eq!(c.borrow().computes, 0));
    let outputs = pytorch
        .evaluate(&[frame()], Some(&[parameters.clone()]), true)
        .unwrap();
    assert_eq!(outputs[0].evaluation.energy, 45.5);
    assert!(outputs[0].atomic.is_some());
    COUNTS.with(|c| assert_eq!((c.borrow().computes, c.borrow().frames), (1, 1)));

    // The suffix restriction must not disable TensorFlow's existing batching.
    let mut tensorflow = model("parameters").unwrap();
    tensorflow.configure_runtime_batch_limit(Path::new("frozen.pb"));
    let outputs = tensorflow
        .evaluate(
            &[frame(), frame()],
            Some(&[parameters.clone(), parameters]),
            false,
        )
        .unwrap();
    assert_eq!(outputs.len(), 2);
    COUNTS.with(|c| assert_eq!((c.borrow().computes, c.borrow().frames), (2, 2)));
}
