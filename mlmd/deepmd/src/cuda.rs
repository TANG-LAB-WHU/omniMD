//! Optional CUDA execution evidence for the official TensorFlow C API path.
//!
//! Requires trusted, explicitly selected CUDA runtime and CUPTI libraries at
//! runtime only. It does not link against a CUDA SDK or initialize CUDA for CPU
//! requests. The isolated worker must expose exactly one selected CUDA device.
//!
//! Every evaluation must issue a successful direct CUDA kernel launch and all
//! observed contexts must synchronize successfully. This proves some GPU work
//! occurred during the call, not that every model operation ran on the GPU.
//! CUDA graphs and stream capture are deliberately unsupported: a graph launch
//! may contain no kernels, and a captured direct launch need not execute at all.
//! No CUDA API is called from the CUPTI callback.
use libloading::Library;
use omnimd_ml_contract::{Error, Result};
use std::{
    ffi::{c_char, c_int, c_void, CStr},
    marker::PhantomData,
    path::Path,
    ptr,
    rc::Rc,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
};

type Context = *mut c_void;
type Subscriber = *mut c_void;
type Callback = unsafe extern "system" fn(*mut c_void, c_int, u32, *const c_void);
type GetDeviceCount = unsafe extern "system" fn(*mut c_int) -> c_int;
type SetDevice = unsafe extern "system" fn(c_int) -> c_int;
type Free = unsafe extern "system" fn(*mut c_void) -> c_int;
type ErrorString = unsafe extern "system" fn(c_int) -> *const c_char;
type PushContext = unsafe extern "system" fn(Context) -> c_int;
type SynchronizeContext = unsafe extern "system" fn() -> c_int;
type PopContext = unsafe extern "system" fn(*mut Context) -> c_int;
type Subscribe = unsafe extern "system" fn(*mut Subscriber, Callback, *mut c_void) -> c_int;
type EnableDomain = unsafe extern "system" fn(u32, Subscriber, c_int) -> c_int;
type Unsubscribe = unsafe extern "system" fn(Subscriber) -> c_int;
type ResultString = unsafe extern "system" fn(c_int, *mut *const c_char) -> c_int;

const DRIVER_DOMAIN: c_int = 1;
const API_EXIT: c_int = 1;
const MAX_CONTEXTS: usize = 8;

// Matches CUDA 12.8 cupti_callbacks.h. CUPTIAPI is __stdcall on Windows,
// hence extern "system" above. Callback data is borrowed only during callback.
#[repr(C)]
struct CallbackData {
    callback_site: c_int,
    function_name: *const c_char,
    function_params: *const c_void,
    function_return_value: *const c_void,
    symbol_name: *const c_char,
    context: Context,
    context_uid: u32,
    correlation_data: *mut u64,
    correlation_id: u32,
}

#[derive(Default)]
struct Evidence {
    active: bool,
    launches: u64,
    graphs: u64,
    capture: bool,
    launch_error: Option<c_int>,
    malformed: bool,
    overflow: bool,
    contexts: Vec<usize>,
}

struct CallbackState {
    evidence: Mutex<Evidence>,
    poisoned: AtomicBool,
}

impl CallbackState {
    fn new() -> Self {
        Self {
            evidence: Mutex::new(Evidence {
                contexts: Vec::with_capacity(MAX_CONTEXTS),
                ..Evidence::default()
            }),
            poisoned: AtomicBool::new(false),
        }
    }

    fn begin(&self) -> Result<()> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(Error("CUDA callback state is poisoned".into()));
        }
        let mut evidence = self
            .evidence
            .lock()
            .map_err(|_| Error("CUDA callback state lock is poisoned".into()))?;
        if evidence.active {
            return Err(Error("CUDA evaluation evidence is already active".into()));
        }
        if evidence.capture {
            return Err(Error("CUDA stream capture was observed in this worker; this direct-kernel execution verifier does not support CUDA graphs".into()));
        }
        evidence.launches = 0;
        evidence.graphs = 0;
        evidence.launch_error = None;
        evidence.malformed = false;
        evidence.overflow = false;
        evidence.contexts.clear();
        evidence.active = true;
        Ok(())
    }

    fn end(&self) -> Result<(u64, Vec<usize>)> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(Error("CUDA callback state is poisoned".into()));
        }
        let mut evidence = self
            .evidence
            .lock()
            .map_err(|_| Error("CUDA callback state lock is poisoned".into()))?;
        if !evidence.active {
            return Err(Error("CUDA evaluation evidence was not started".into()));
        }
        evidence.active = false;
        if let Some(code) = evidence.launch_error {
            return Err(Error(format!("CUDA driver kernel/graph/capture operation failed with code {code}; GPU execution stopped")));
        }
        if evidence.malformed || evidence.overflow {
            return Err(Error("CUDA execution evidence is incomplete (missing context/return value or too many contexts); GPU execution stopped".into()));
        }
        if evidence.capture || evidence.graphs != 0 {
            return Err(Error("CUDA graph launch or stream capture observed; this direct-kernel execution verifier does not support CUDA graphs, and cannot attest this evaluation".into()));
        }
        if evidence.launches == 0 {
            return Err(Error("CUDA requested, but this model evaluation issued no successful direct CUDA kernel launches; CPU-only runtime or CPU placement cannot be reported as GPU execution".into()));
        }
        Ok((evidence.launches, evidence.contexts.clone()))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CallKind {
    Kernel,
    Graph,
    Capture,
}

fn call_kind(name: &[u8]) -> Option<CallKind> {
    match name {
        b"cuLaunchKernel"
        | b"cuLaunchKernel_ptsz"
        | b"cuLaunchKernelEx"
        | b"cuLaunchKernelEx_ptsz"
        | b"cuLaunchCooperativeKernel"
        | b"cuLaunchCooperativeKernel_ptsz" => Some(CallKind::Kernel),
        b"cuGraphLaunch" | b"cuGraphLaunch_ptsz" => Some(CallKind::Graph),
        _ if name.starts_with(b"cuStreamBeginCapture")
            || name.starts_with(b"cuStreamEndCapture") =>
        {
            Some(CallKind::Capture)
        }
        _ => None,
    }
}

unsafe extern "system" fn callback(
    userdata: *mut c_void,
    domain: c_int,
    _id: u32,
    data: *const c_void,
) {
    if domain != DRIVER_DOMAIN || userdata.is_null() || data.is_null() {
        return;
    }
    let data = unsafe { &*data.cast::<CallbackData>() };
    if data.callback_site != API_EXIT || data.function_name.is_null() {
        return;
    }
    let name = unsafe { CStr::from_ptr(data.function_name) }.to_bytes();
    let Some(kind) = call_kind(name) else {
        return;
    };
    let state = unsafe { &*userdata.cast::<CallbackState>() };
    // TensorFlow may launch from multiple worker threads. Keep one short,
    // process-wide critical section; do not use thread-local evidence.
    let Ok(mut evidence) = state.evidence.lock() else {
        state.poisoned.store(true, Ordering::Release);
        return;
    };
    // Capture may begin during model loading and continue into evaluation.
    // Remember it for the entire guard lifetime, including inactive periods.
    if !evidence.active && kind != CallKind::Capture {
        return;
    }
    if data.function_return_value.is_null() {
        evidence.malformed = true;
        return;
    }
    let result = unsafe { *data.function_return_value.cast::<c_int>() };
    if result != 0 {
        evidence.launch_error.get_or_insert(result);
        return;
    }
    match kind {
        CallKind::Graph => {
            evidence.graphs = evidence.graphs.saturating_add(1);
        }
        CallKind::Capture => {
            evidence.capture = true;
        }
        CallKind::Kernel => {
            if data.context.is_null() {
                evidence.malformed = true;
                return;
            }
            let context = data.context as usize;
            if !evidence.contexts.contains(&context) {
                if evidence.contexts.len() == MAX_CONTEXTS {
                    evidence.overflow = true;
                    return;
                }
                evidence.contexts.push(context);
            }
            evidence.launches = evidence.launches.saturating_add(1);
        }
    }
}

struct Api {
    get_device_count: GetDeviceCount,
    set_device: SetDevice,
    free: Free,
    error_string: ErrorString,
    push_context: PushContext,
    synchronize_context: SynchronizeContext,
    pop_context: PopContext,
    subscribe: Subscribe,
    enable_domain: EnableDomain,
    unsubscribe: Unsubscribe,
    result_string: ResultString,
    _runtime: Library,
    _driver: Library,
    _cupti: Library,
}

unsafe fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T> {
    unsafe { library.get::<T>(name) }
        .map(|value| *value)
        .map_err(|error| Error(format!("required CUDA/CUPTI symbol missing: {error}")))
}

fn trusted_library(path: &Path, label: &str) -> Result<Library> {
    if !path.is_absolute() || !path.is_file() {
        return Err(Error(format!(
            "{label} must be an absolute path to an existing trusted library: {}",
            path.display()
        )));
    }
    unsafe { Library::new(path) }
        .map_err(|error| Error(format!("load {label} {}: {error}", path.display())))
}

impl Api {
    fn load(runtime: &Path, cupti: &Path) -> Result<Self> {
        let runtime = trusted_library(runtime, "CUDA runtime")?;
        let cupti = trusted_library(cupti, "CUPTI library")?;
        let driver_name = if cfg!(windows) {
            "nvcuda.dll"
        } else {
            "libcuda.so.1"
        };
        let driver = unsafe { Library::new(driver_name) }
            .map_err(|error| Error(format!("load CUDA driver {driver_name}: {error}")))?;
        unsafe {
            Ok(Self {
                get_device_count: symbol(&runtime, b"cudaGetDeviceCount\0")?,
                set_device: symbol(&runtime, b"cudaSetDevice\0")?,
                free: symbol(&runtime, b"cudaFree\0")?,
                error_string: symbol(&runtime, b"cudaGetErrorString\0")?,
                push_context: symbol(&driver, b"cuCtxPushCurrent_v2\0")?,
                synchronize_context: symbol(&driver, b"cuCtxSynchronize\0")?,
                pop_context: symbol(&driver, b"cuCtxPopCurrent_v2\0")?,
                subscribe: symbol(&cupti, b"cuptiSubscribe\0")?,
                enable_domain: symbol(&cupti, b"cuptiEnableDomain\0")?,
                unsubscribe: symbol(&cupti, b"cuptiUnsubscribe\0")?,
                result_string: symbol(&cupti, b"cuptiGetResultString\0")?,
                _runtime: runtime,
                _driver: driver,
                _cupti: cupti,
            })
        }
    }

    fn runtime_check(&self, code: c_int, action: &str) -> Result<()> {
        if code == 0 {
            return Ok(());
        }
        let message = unsafe { (self.error_string)(code) };
        let message = if message.is_null() {
            "unknown error".into()
        } else {
            unsafe { CStr::from_ptr(message) }.to_string_lossy()
        };
        Err(Error(format!("{action}: CUDA error {code}: {message}")))
    }

    fn cupti_check(&self, code: c_int, action: &str) -> Result<()> {
        if code == 0 {
            return Ok(());
        }
        let mut message = ptr::null();
        let text_status = unsafe { (self.result_string)(code, &mut message) };
        let message = if text_status != 0 || message.is_null() {
            "unknown error".into()
        } else {
            unsafe { CStr::from_ptr(message) }.to_string_lossy()
        };
        Err(Error(format!(
            "{action}: CUPTI error {code}: {message}; CUDA execution cannot be verified"
        )))
    }

    fn synchronize(&self, context: usize) -> Result<()> {
        synchronize_with(
            context,
            |context| unsafe { (self.push_context)(context as Context) },
            || unsafe { (self.synchronize_context)() },
            |popped| unsafe { (self.pop_context)(popped) },
        )
    }
}

fn synchronize_with(
    context: usize,
    push: impl FnOnce(usize) -> c_int,
    synchronize: impl FnOnce() -> c_int,
    pop: impl FnOnce(&mut Context) -> c_int,
) -> Result<()> {
    let status = push(context);
    if status != 0 {
        return Err(Error(format!(
            "cuCtxPushCurrent_v2 for observed kernel context failed: CUDA driver error {status}"
        )));
    }
    let synchronized = synchronize();
    // Restore caller context even on synchronization failure. Never reset or
    // release a context belonging to the model runtime.
    let mut popped = ptr::null_mut();
    let restored = pop(&mut popped);
    if synchronized != 0 || restored != 0 || popped as usize != context {
        return Err(Error(format!("observed kernel context synchronization failed: cuCtxSynchronize={synchronized}, cuCtxPopCurrent_v2={restored}, restored_context_matches={}; GPU execution stopped", popped as usize == context)));
    }
    Ok(())
}

/// One subscriber and one model per fresh isolated worker. Intentionally not
/// Send/Sync; only the callback's evidence storage is shared between threads.
pub(crate) struct CudaGuard {
    api: Option<Api>,
    state: Option<Box<CallbackState>>,
    subscriber: Subscriber,
    failed: bool,
    _single_thread: PhantomData<Rc<()>>,
}

impl CudaGuard {
    /// Call only inside the trusted-library boundary, before loading DeepMD,
    /// in a fresh worker whose CUDA visibility selects exactly one GPU.
    ///
    /// # Safety
    /// Both paths must identify trusted matching NVIDIA libraries implementing
    /// these C ABIs; the caller must guarantee the fresh-worker initialization
    /// order and must not mutate accelerator visibility after initialization.
    pub(crate) unsafe fn load(runtime: &Path, cupti: &Path) -> Result<Self> {
        let api = Api::load(runtime, cupti)?;
        let mut count = 0;
        api.runtime_check(
            unsafe { (api.get_device_count)(&mut count) },
            "cudaGetDeviceCount",
        )?;
        if count != 1 {
            return Err(Error(format!(
                "CUDA worker must expose exactly one selected GPU; CUDA runtime reports {count}"
            )));
        }
        api.runtime_check(unsafe { (api.set_device)(0) }, "cudaSetDevice(0)")?;
        api.runtime_check(
            unsafe { (api.free)(ptr::null_mut()) },
            "initialize CUDA context with cudaFree(NULL)",
        )?;
        let mut guard = Self {
            api: Some(api),
            state: Some(Box::new(CallbackState::new())),
            subscriber: ptr::null_mut(),
            failed: false,
            _single_thread: PhantomData,
        };
        let api = guard.api.as_ref().expect("new guard API");
        let state = guard.state.as_deref_mut().expect("new guard state") as *mut CallbackState;
        let status = unsafe { (api.subscribe)(&mut guard.subscriber, callback, state.cast()) };
        if status != 0 {
            // CUPTI requires stopping after subscription failure, particularly
            // when another profiler already owns the sole subscriber slot.
            guard.subscriber = ptr::null_mut();
            return api.cupti_check(status, "cuptiSubscribe").map(|()| guard);
        }
        if guard.subscriber.is_null() {
            return Err(Error("cuptiSubscribe returned a null subscriber".into()));
        }
        api.cupti_check(
            unsafe { (api.enable_domain)(1, guard.subscriber, DRIVER_DOMAIN) },
            "enable CUDA driver callbacks",
        )?;
        Ok(guard)
    }

    pub(crate) fn begin(&mut self) -> Result<()> {
        if self.failed {
            return Err(Error(
                "CUDA guard is stopped after an earlier error; GPU retries are disabled".into(),
            ));
        }
        let result = self.state.as_ref().expect("live guard state").begin();
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    /// Call after DP_DeepPotCompute2 and before publishing its E/F/virial. The
    /// count is evidence of direct kernel launches, not a GPU operation ratio.
    pub(crate) fn finish(&mut self) -> Result<u64> {
        if self.failed {
            return Err(Error(
                "CUDA guard is stopped after an earlier error; GPU retries are disabled".into(),
            ));
        }
        let result = (|| {
            let (launches, contexts) = self.state.as_ref().expect("live guard state").end()?;
            for context in contexts {
                self.api
                    .as_ref()
                    .expect("live guard API")
                    .synchronize(context)?;
            }
            Ok(launches)
        })();
        if result.is_err() {
            self.failed = true;
        }
        result
    }
}

impl Drop for CudaGuard {
    fn drop(&mut self) {
        if self.subscriber.is_null() {
            return;
        }
        if let Some(state) = self.state.as_ref() {
            if let Ok(mut evidence) = state.evidence.lock() {
                evidence.active = false;
            }
        }
        // Unsubscribe before freeing state or unloading callback dependencies.
        let api = self.api.as_ref().expect("live guard API");
        let status = unsafe { (api.unsubscribe)(self.subscriber) };
        if status != 0 {
            // An uncertain detach must never leave CUPTI with dangling userdata
            // or unloaded dependencies. The isolated process reclaims this on exit.
            eprintln!("CUDA verifier: cuptiUnsubscribe failed ({status}); retaining callback state and libraries until process exit");
            if let Some(state) = self.state.take() {
                std::mem::forget(state);
            }
            if let Some(api) = self.api.take() {
                std::mem::forget(api);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    fn event(state: &CallbackState, name: &str, code: c_int, context: usize, site: c_int) {
        let name = CString::new(name).unwrap();
        let data = CallbackData {
            callback_site: site,
            function_name: name.as_ptr(),
            function_params: ptr::null(),
            function_return_value: (&code as *const c_int).cast(),
            symbol_name: ptr::null(),
            context: context as Context,
            context_uid: 0,
            correlation_data: ptr::null_mut(),
            correlation_id: 0,
        };
        unsafe {
            callback(
                (state as *const CallbackState).cast_mut().cast(),
                DRIVER_DOMAIN,
                0,
                (&data as *const CallbackData).cast(),
            );
        }
    }

    #[test]
    fn records_only_active_successful_exit_launches_and_resets() {
        let state = CallbackState::new();
        event(&state, "cuLaunchKernel", 0, 1, API_EXIT);
        state.begin().unwrap();
        event(&state, "cuLaunchKernel", 0, 1, 0);
        event(&state, "cuMemcpyHtoD_v2", 0, 1, API_EXIT);
        event(&state, "cuLaunchKernel", 0, 1, API_EXIT);
        event(&state, "cuLaunchKernel_ptsz", 0, 1, API_EXIT);
        event(&state, "cuLaunchCooperativeKernel", 0, 2, API_EXIT);
        assert_eq!(state.end().unwrap(), (3, vec![1, 2]));
        state.begin().unwrap();
        assert!(state.end().unwrap_err().0.contains("no successful"));
    }

    #[test]
    fn graph_capture_failed_launch_and_context_overflow_fail_closed() {
        for name in ["cuGraphLaunch", "cuStreamBeginCapture_v2_ptsz"] {
            let state = CallbackState::new();
            state.begin().unwrap();
            event(&state, "cuLaunchKernel", 0, 1, API_EXIT);
            event(&state, name, 0, 1, API_EXIT);
            assert!(state
                .end()
                .unwrap_err()
                .0
                .contains("does not support CUDA graphs"));
        }
        let state = CallbackState::new();
        state.begin().unwrap();
        event(&state, "cuLaunchKernel", 0, 1, API_EXIT);
        event(&state, "cuLaunchKernel", 719, 1, API_EXIT);
        assert!(state.end().unwrap_err().0.contains("code 719"));
        state.begin().unwrap();
        for context in 1..=MAX_CONTEXTS + 1 {
            event(&state, "cuLaunchKernel", 0, context, API_EXIT);
        }
        assert!(state.end().unwrap_err().0.contains("incomplete"));
    }

    #[test]
    fn callback_collects_across_runtime_worker_threads() {
        let state = CallbackState::new();
        state.begin().unwrap();
        std::thread::scope(|scope| {
            for context in 1..=4 {
                let state = &state;
                scope.spawn(move || {
                    for _ in 0..8 {
                        event(state, "cuLaunchKernelEx_ptsz", 0, context, API_EXIT);
                    }
                });
            }
        });
        let (launches, contexts) = state.end().unwrap();
        assert_eq!(launches, 32);
        assert_eq!(contexts.len(), 4);
    }

    #[test]
    fn context_is_popped_even_after_sync_error_and_not_after_push_error() {
        use std::cell::Cell;
        let pops = Cell::new(0);
        let error = synchronize_with(
            7,
            |_| 0,
            || 719,
            |out| {
                pops.set(pops.get() + 1);
                *out = 7usize as Context;
                0
            },
        )
        .unwrap_err();
        assert!(error.0.contains("cuCtxSynchronize=719"));
        assert_eq!(pops.get(), 1);
        assert!(synchronize_with(
            7,
            |_| 201,
            || panic!("must not synchronize"),
            |_| panic!("must not pop")
        )
        .is_err());
    }
}
