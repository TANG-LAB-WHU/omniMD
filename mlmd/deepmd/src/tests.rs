use super::*;
use std::cell::RefCell;

#[derive(Default)]
struct Counts {
    loads: usize,
    drops: usize,
    strings: usize,
    frees: usize,
    computes: usize,
    types: Vec<i32>,
    coords: Vec<f64>,
    cell: Option<Vec<f64>>,
    frames: i32,
    optional_were_null: bool,
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
    i32::from(unsafe { state(h) }.mode == "fparam")
}
unsafe extern "C" fn aparam(h: Handle) -> i32 {
    i32::from(unsafe { state(h) }.mode == "aparam")
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
    let coords = unsafe { std::slice::from_raw_parts(x, n as usize * 3) };
    COUNTS.with(|c| {
        let mut c = c.borrow_mut();
        c.computes += 1;
        c.frames = nf;
        c.coords = coords.to_vec();
        c.types = unsafe { std::slice::from_raw_parts(t, n as usize) }.to_vec();
        c.cell = if cell.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(cell, 9) }.to_vec())
        };
        c.optional_were_null = fp.is_null() && ap.is_null() && ae.is_null() && av.is_null();
    });
    let mock = unsafe { state(h) };
    if mock.mode == "compute-error" {
        mock.error = "injected compute error".into();
        return;
    }
    unsafe {
        *energy = coords.iter().map(|v| v * v * 0.5).sum();
    }
    for (i, x) in coords.iter().enumerate() {
        unsafe {
            *force.add(i) = -*x;
        }
    }
    if mock.mode != "missing-virial" {
        for i in 0..9 {
            unsafe {
                *virial.add(i) = (i + 1) as f64;
            }
        }
    }
    if mock.mode == "nonfinite-force" {
        unsafe {
            *force = f64::INFINITY;
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
        "fparam",
        "aparam",
        "bad-cutoff",
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
