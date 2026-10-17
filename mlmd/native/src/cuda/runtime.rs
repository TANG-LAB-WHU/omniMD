//! Minimal dynamic NVIDIA Driver/NVRTC bindings. No link-time CUDA dependency.
#[path = "device_runtime.rs"]
mod device_loss;
use omnimd_ml_contract::{Error, Result};
use std::{
    ffi::{c_char, c_void, CStr, CString},
    marker::PhantomData,
    path::{Path, PathBuf},
    ptr,
    rc::Rc,
};
type Handle = *mut c_void;
struct Library(Handle);
#[cfg(windows)]
mod os {
    use super::*;
    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryExW(path: *const u16, file: Handle, flags: u32) -> Handle;
        fn GetProcAddress(module: Handle, name: *const c_char) -> Handle;
        fn FreeLibrary(module: Handle) -> i32;
    }
    pub unsafe fn open(path: &Path) -> Handle {
        use std::os::windows::ffi::OsStrExt;
        let wide: Vec<_> = path.as_os_str().encode_wide().chain(Some(0)).collect();
        LoadLibraryExW(wide.as_ptr(), ptr::null_mut(), 0x1100)
    }
    pub unsafe fn symbol(lib: Handle, name: *const c_char) -> Handle {
        GetProcAddress(lib, name)
    }
    pub unsafe fn close(lib: Handle) {
        FreeLibrary(lib);
    }
}
#[cfg(unix)]
mod os {
    use super::*;
    #[link(name = "dl")]
    extern "C" {
        fn dlopen(file: *const c_char, flags: i32) -> Handle;
        fn dlsym(lib: Handle, name: *const c_char) -> Handle;
        fn dlclose(lib: Handle) -> i32;
    }
    pub unsafe fn open(path: &Path) -> Handle {
        use std::os::unix::ffi::OsStrExt;
        let value = CString::new(path.as_os_str().as_bytes()).unwrap();
        dlopen(value.as_ptr(), 2)
    }
    pub unsafe fn symbol(lib: Handle, name: *const c_char) -> Handle {
        dlsym(lib, name)
    }
    pub unsafe fn close(lib: Handle) {
        dlclose(lib);
    }
}
impl Library {
    fn open(path: &Path) -> Result<Self> {
        let handle = unsafe { os::open(path) };
        if handle.is_null() {
            Err(Error(format!("cannot load NVIDIA library {}; install/configure an official matching CUDA runtime explicitly",path.display())))
        } else {
            Ok(Self(handle))
        }
    }
    unsafe fn symbol<T: Copy>(&self, name: &str) -> Result<T> {
        let text = CString::new(name).unwrap();
        let address = os::symbol(self.0, text.as_ptr());
        if address.is_null() {
            return Err(Error(format!("missing CUDA symbol {name}")));
        }
        assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<Handle>());
        Ok(std::mem::transmute_copy(&address))
    }
}
impl Drop for Library {
    fn drop(&mut self) {
        unsafe { os::close(self.0) }
    }
}
fn nvrtc_path() -> Result<PathBuf> {
    if let Some(value) = std::env::var_os("OMNIMD_NVRTC_LIBRARY") {
        let path = PathBuf::from(value);
        if !path.is_absolute() {
            return Err(Error(
                "OMNIMD_NVRTC_LIBRARY must be an absolute path to official NVIDIA NVRTC".into(),
            ));
        }
        return Ok(path);
    }
    let root = std::env::var_os("CUDA_PATH").map(PathBuf::from);
    #[cfg(windows)]
    {
        Ok(root.ok_or_else(||Error("set CUDA_PATH or absolute OMNIMD_NVRTC_LIBRARY for the installed official toolkit".into()))?.join("bin/nvrtc64_120_0.dll"))
    }
    #[cfg(unix)]
    {
        Ok(root
            .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"))
            .join("lib64/libnvrtc.so"))
    }
}
/// Compile without touching the CUDA driver or creating a GPU context.
pub fn compile(source: &str, compute: i32) -> Result<Vec<u8>> {
    unsafe {
        let library = Library::open(&nvrtc_path()?)?;
        let create: unsafe extern "C" fn(
            *mut Handle,
            *const c_char,
            *const c_char,
            i32,
            *const *const c_char,
            *const *const c_char,
        ) -> i32 = library.symbol("nvrtcCreateProgram")?;
        let compile: unsafe extern "C" fn(Handle, i32, *const *const c_char) -> i32 =
            library.symbol("nvrtcCompileProgram")?;
        let log_size: unsafe extern "C" fn(Handle, *mut usize) -> i32 =
            library.symbol("nvrtcGetProgramLogSize")?;
        let log_get: unsafe extern "C" fn(Handle, *mut c_char) -> i32 =
            library.symbol("nvrtcGetProgramLog")?;
        let ptx_size: unsafe extern "C" fn(Handle, *mut usize) -> i32 =
            library.symbol("nvrtcGetPTXSize")?;
        let ptx_get: unsafe extern "C" fn(Handle, *mut c_char) -> i32 =
            library.symbol("nvrtcGetPTX")?;
        let destroy: unsafe extern "C" fn(*mut Handle) -> i32 =
            library.symbol("nvrtcDestroyProgram")?;
        let source = CString::new(source).map_err(|e| Error(e.to_string()))?;
        let name = CString::new("omnimd_native.cu").unwrap();
        let mut program = ptr::null_mut();
        check(
            create(
                &mut program,
                source.as_ptr(),
                name.as_ptr(),
                0,
                ptr::null(),
                ptr::null(),
            ),
            "nvrtcCreateProgram",
        )?;
        struct Program {
            handle: Handle,
            destroy: unsafe extern "C" fn(*mut Handle) -> i32,
        }
        impl Drop for Program {
            fn drop(&mut self) {
                unsafe {
                    (self.destroy)(&mut self.handle);
                }
            }
        }
        let program = Program {
            handle: program,
            destroy,
        };
        let opts = [
            format!("--gpu-architecture=compute_{compute}"),
            "--std=c++11".into(),
            "--fmad=false".into(),
        ];
        let strings: Vec<_> = opts
            .iter()
            .map(|x| CString::new(x.as_str()).unwrap())
            .collect();
        let options: Vec<_> = strings.iter().map(|x| x.as_ptr()).collect();
        let status = compile(program.handle, options.len() as i32, options.as_ptr());
        if status != 0 {
            let mut size = 0;
            log_size(program.handle, &mut size);
            let mut log = vec![0; size.min(65536).max(1)];
            if size <= 65536 {
                log_get(program.handle, log.as_mut_ptr() as *mut c_char);
            }
            return Err(Error(format!(
                "nvrtcCompileProgram failed ({status}): {}",
                String::from_utf8_lossy(&log)
            )));
        }
        let mut size = 0;
        check(ptx_size(program.handle, &mut size), "nvrtcGetPTXSize")?;
        let mut ptx = vec![0; size];
        check(
            ptx_get(program.handle, ptx.as_mut_ptr() as *mut c_char),
            "nvrtcGetPTX",
        )?;
        Ok(ptx)
    }
}
fn check(code: i32, operation: &str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(Error(format!("{operation} failed with NVIDIA status {code}; stop this CUDA instance, no automatic retry")))
    }
}
macro_rules! api {
    ($($field:ident:$ty:ty=>$name:literal),* $(,)?) => {
        #[derive(Clone,Copy)] struct Api {$( $field:$ty, )*}
        impl Api {unsafe fn load(lib:&Library)->Result<Self> {Ok(Self {$( $field:lib.symbol($name)?, )*})}}
    }
}
api! {
    init:unsafe extern "system" fn(u32)->i32=>"cuInit",
    driver_version:unsafe extern "system" fn(*mut i32)->i32=>"cuDriverGetVersion",
    get_device:unsafe extern "system" fn(*mut i32,i32)->i32=>"cuDeviceGet",
    name:unsafe extern "system" fn(*mut c_char,i32,i32)->i32=>"cuDeviceGetName",
    capability:unsafe extern "system" fn(*mut i32,*mut i32,i32)->i32=>"cuDeviceComputeCapability",
    ctx_create:unsafe extern "system" fn(*mut Handle,u32,i32)->i32=>"cuCtxCreate_v2",
    ctx_destroy:unsafe extern "system" fn(Handle)->i32=>"cuCtxDestroy_v2",
    ctx_current:unsafe extern "system" fn(*mut Handle)->i32=>"cuCtxGetCurrent",
    ctx_set:unsafe extern "system" fn(Handle)->i32=>"cuCtxSetCurrent",
    module_load:unsafe extern "system" fn(*mut Handle,*const c_void)->i32=>"cuModuleLoadData",
    module_unload:unsafe extern "system" fn(Handle)->i32=>"cuModuleUnload",
    function:unsafe extern "system" fn(*mut Handle,Handle,*const c_char)->i32=>"cuModuleGetFunction",
    attribute:unsafe extern "system" fn(*mut i32,i32,Handle)->i32=>"cuFuncGetAttribute",
    alloc:unsafe extern "system" fn(*mut u64,usize)->i32=>"cuMemAlloc_v2",
    free:unsafe extern "system" fn(u64)->i32=>"cuMemFree_v2",
    h2d:unsafe extern "system" fn(u64,*const c_void,usize)->i32=>"cuMemcpyHtoD_v2",
    d2h:unsafe extern "system" fn(*mut c_void,u64,usize)->i32=>"cuMemcpyDtoH_v2",
    launch:unsafe extern "system" fn(Handle,u32,u32,u32,u32,u32,u32,u32,Handle,*mut *mut c_void,*mut *mut c_void)->i32=>"cuLaunchKernel",
    sync:unsafe extern "system" fn()->i32=>"cuCtxSynchronize",
}
struct Current {
    api: Api,
    previous: Handle,
}
impl Current {
    unsafe fn enter(api: Api, context: Handle) -> Result<Self> {
        let mut previous = ptr::null_mut();
        check((api.ctx_current)(&mut previous), "cuCtxGetCurrent")?;
        check((api.ctx_set)(context), "cuCtxSetCurrent")?;
        Ok(Self { api, previous })
    }
}
impl Drop for Current {
    fn drop(&mut self) {
        unsafe {
            (self.api.ctx_set)(self.previous);
        }
    }
}
#[derive(Debug, serde::Serialize)]
pub struct KernelResources {
    pub name: String,
    pub registers_per_thread: i32,
    pub static_shared_bytes: i32,
    pub local_bytes_per_thread: i32,
}
pub struct Runtime {
    api: Api,
    context: Handle,
    module: Handle,
    functions: Vec<Handle>,
    buffers: Vec<(u64, usize)>,
    pub device_info: String,
    pub driver_version: i32,
    pub program_sha256: String,
    pub kernel_resources: Vec<KernelResources>,
    pub last_timing_ms: [f64; 3],
    pub type_uploads: usize,
    pub last_upload_bytes: usize,
    cached_types: Vec<i32>,
    packed_input: Vec<u8>,
    packed_input_index: Option<usize>,
    pub energies: Vec<f64>,
    pub gradients: Vec<f64>,
    _library: Library,
    _not_send: PhantomData<Rc<()>>,
}
impl Runtime {
    pub fn new(source: &str, ordinal: i32) -> Result<Self> {
        Self::with_functions(source, ordinal, &["embed", "fit", "pullback"], 11)
    }
    pub(super) fn with_functions(
        source: &str,
        ordinal: i32,
        names: &[&str],
        buffers: usize,
    ) -> Result<Self> {
        unsafe {
            #[cfg(windows)]
            let driver = PathBuf::from(
                std::env::var_os("SystemRoot").unwrap_or_else(|| "C:\\Windows".into()),
            )
            .join("System32/nvcuda.dll");
            #[cfg(unix)]
            let driver = PathBuf::from("libcuda.so.1");
            let library = Library::open(&driver)?;
            let api = Api::load(&library)?;
            check((api.init)(0), "cuInit")?;
            let mut driver_version = 0;
            check(
                (api.driver_version)(&mut driver_version),
                "cuDriverGetVersion",
            )?;
            let mut device = 0;
            check((api.get_device)(&mut device, ordinal), "cuDeviceGet")?;
            let (mut major, mut minor) = (0, 0);
            check(
                (api.capability)(&mut major, &mut minor, device),
                "cuDeviceComputeCapability",
            )?;
            let mut name = [0_i8; 256];
            check(
                (api.name)(name.as_mut_ptr(), 256, device),
                "cuDeviceGetName",
            )?;
            let device_info = format!(
                "{}; compute {}.{}; FP64; device {}",
                CStr::from_ptr(name.as_ptr()).to_string_lossy(),
                major,
                minor,
                ordinal
            );
            let ptx = compile(source, major * 10 + minor)?;
            use sha2::{Digest, Sha256};
            let program_sha256 = format!("{:x}", Sha256::digest(&ptx));
            let mut previous = ptr::null_mut();
            check((api.ctx_current)(&mut previous), "cuCtxGetCurrent")?;
            let mut context = ptr::null_mut();
            check((api.ctx_create)(&mut context, 0, device), "cuCtxCreate_v2")?;
            let restore = Current { api, previous };
            let mut rt = Self {
                api,
                context,
                module: ptr::null_mut(),
                functions: vec![ptr::null_mut(); names.len()],
                buffers: vec![(0, 0); buffers],
                device_info,
                driver_version,
                program_sha256,
                kernel_resources: Vec::new(),
                last_timing_ms: [0.0; 3],
                type_uploads: 0,
                last_upload_bytes: 0,
                cached_types: Vec::new(),
                packed_input: Vec::new(),
                packed_input_index: None,
                energies: Vec::new(),
                gradients: Vec::new(),
                _library: library,
                _not_send: PhantomData,
            };
            check(
                (api.module_load)(&mut rt.module, ptx.as_ptr() as *const c_void),
                "cuModuleLoadData",
            )?;
            for (i, name) in names.iter().enumerate() {
                let name = CString::new(*name).unwrap();
                check(
                    (api.function)(&mut rt.functions[i], rt.module, name.as_ptr()),
                    "cuModuleGetFunction",
                )?;
            }
            for (i, name) in names.iter().enumerate() {
                let mut values = [0; 3];
                for (value, attribute) in values.iter_mut().zip([4, 1, 3]) {
                    check(
                        (api.attribute)(value, attribute, rt.functions[i]),
                        "cuFuncGetAttribute",
                    )?;
                }
                rt.kernel_resources.push(KernelResources {
                    name: (*name).into(),
                    registers_per_thread: values[0],
                    static_shared_bytes: values[1],
                    local_bytes_per_thread: values[2],
                });
            }
            drop(restore);
            Ok(rt)
        }
    }
    pub(super) fn enable_packed_inputs(&mut self) {
        assert!(self.packed_input_index.is_none());
        self.packed_input_index = Some(self.buffers.len());
        self.buffers.push((0, 0));
    }
    pub(super) fn reserve_packed_inputs(
        &mut self,
        bytes: usize,
        host_budget: usize,
    ) -> Result<usize> {
        if self.packed_input.capacity().max(bytes) > host_budget {
            return Err(Error("packed CUDA input host capacity exceeded".into()));
        }
        if self.packed_input.capacity() < bytes {
            self.packed_input = Vec::new();
            self.packed_input = Vec::with_capacity(bytes);
        }
        // Vec is allowed to provide excess capacity: count the actual retained payload.
        if self.packed_input.capacity() > host_budget {
            self.packed_input = Vec::new();
            return Err(Error(
                "packed CUDA allocation exceeded host capacity".into(),
            ));
        }
        Ok(self.packed_input.capacity())
    }
    // Caller holds this runtime's current context; copies are synchronous.
    // Only the arena base is owned in buffers. Interior pointers are call-local.
    fn upload_packed_inputs(&mut self, parts: &[(usize, &[u8])]) -> Result<[u64; 31]> {
        let layout = super::transfer::Layout::new(parts.iter().map(|(i, x)| (*i, x.len())))?;
        if layout.bytes > 32 * 1024 * 1024 {
            return Err(Error("packed CUDA inputs exceed host policy".into()));
        }
        let index = self
            .packed_input_index
            .ok_or_else(|| Error("packed CUDA inputs are not enabled".into()))?;
        if self.packed_input.capacity() < layout.bytes {
            return Err(Error(
                "packed CUDA inputs were not reserved within host budget".into(),
            ));
        }
        self.packed_input.resize(layout.bytes, 0);
        for &(i, bytes) in parts {
            let start = layout.offsets[i];
            self.packed_input[start..start + bytes.len()].copy_from_slice(bytes);
        }
        unsafe {
            if self.buffers[index].1 < layout.bytes && self.buffers[index].0 != 0 {
                check(
                    (self.api.free)(self.buffers[index].0),
                    "cuMemFree_v2(packed grow)",
                )?;
                self.buffers[index] = (0, 0);
            }
            let base = self.ensure(index, layout.bytes)?;
            check(
                (self.api.h2d)(base, self.packed_input.as_ptr().cast(), layout.bytes),
                "cuMemcpyHtoD_v2(packed)",
            )?;
            let mut address = std::array::from_fn(|i| self.buffers.get(i).map_or(0, |b| b.0));
            for &(i, _) in parts {
                address[i] = base
                    .checked_add(layout.offsets[i] as u64)
                    .ok_or_else(|| Error("packed CUDA pointer overflow".into()))?;
            }
            Ok(address)
        }
    }
    fn ensure(&mut self, index: usize, bytes: usize) -> Result<u64> {
        unsafe {
            if self.buffers[index].1 < bytes {
                let mut pointer = 0;
                check((self.api.alloc)(&mut pointer, bytes), "cuMemAlloc_v2")?;
                let old = self.buffers[index];
                self.buffers[index] = (pointer, bytes);
                if old.0 != 0 {
                    check((self.api.free)(old.0), "cuMemFree_v2")?;
                }
            }
            Ok(self.buffers[index].0)
        }
    }
    pub fn upload<T: Copy>(&mut self, index: usize, data: &[T]) -> Result<()> {
        let _current = unsafe { Current::enter(self.api, self.context)? };
        self.upload_current(index, data)
    }
    fn upload_current<T: Copy>(&mut self, index: usize, data: &[T]) -> Result<()> {
        let bytes = std::mem::size_of_val(data);
        let pointer = self.ensure(index, bytes)?;
        unsafe {
            check(
                (self.api.h2d)(pointer, data.as_ptr() as *const c_void, bytes),
                "cuMemcpyHtoD_v2",
            )
        }
    }
    unsafe fn download_into<T: Copy>(api: Api, pointer: u64, output: &mut [T]) -> Result<()> {
        check(
            (api.d2h)(
                output.as_mut_ptr() as *mut c_void,
                pointer,
                std::mem::size_of_val(output),
            ),
            "cuMemcpyDtoH_v2",
        )
    }
    /// Select one final-readout basis coefficient in private training weights.
    /// Both indices refer to the resident packed weight array. No graph/model
    /// reload or host-wide weight copy is necessary.
    pub(super) fn select_readout_basis(
        &mut self,
        previous: Option<usize>,
        next: usize,
    ) -> Result<()> {
        let count = self.buffers[1].1 / std::mem::size_of::<f64>();
        if next >= count || previous.is_some_and(|p| p >= count) {
            return Err(Error(
                "readout basis weight index outside resident buffer".into(),
            ));
        }
        if previous == Some(next) {
            return Ok(());
        }
        let _current = unsafe { Current::enter(self.api, self.context)? };
        let copy = |offset: usize, values: &[f64]| unsafe {
            check(
                (self.api.h2d)(
                    self.buffers[1].0 + (offset * 8) as u64,
                    values.as_ptr() as *const c_void,
                    std::mem::size_of_val(values),
                ),
                "cuMemcpyHtoD_v2(readout basis)",
            )
        };
        if previous.is_some_and(|p| p.checked_add(1) == Some(next)) {
            copy(previous.unwrap(), &[0.0, 1.0])?;
        } else {
            if let Some(previous) = previous {
                copy(previous, &[0.0])?;
            }
            copy(next, &[1.0])?;
        }
        Ok(())
    }
    pub fn evaluate(
        &mut self,
        rows: &[f64],
        types: &[i32],
        slots: usize,
        width: usize,
    ) -> Result<()> {
        unsafe {
            let _current = Current::enter(self.api, self.context)?;
            let n = types.len();
            let q = n * slots;
            let phase = std::time::Instant::now();
            self.upload_current(2, rows)?;
            self.last_upload_bytes = std::mem::size_of_val(rows) + 4;
            if self.cached_types != types {
                self.upload_current(3, types)?;
                self.cached_types.clear();
                self.cached_types.extend_from_slice(types);
                self.type_uploads += 1;
                self.last_upload_bytes += std::mem::size_of_val(types);
            }
            self.upload_current(10, &[0_i32])?;
            self.ensure(5, q * width * 8)?;
            self.ensure(6, q * width * 8)?;
            self.ensure(7, n * width * 4 * 8)?;
            self.ensure(8, n * 8)?;
            self.ensure(9, q * 4 * 8)?;
            self.last_timing_ms[0] = phase.elapsed().as_secs_f64() * 1000.0;
            let phase = std::time::Instant::now();
            let mut addresses: [u64; 11] = std::array::from_fn(|i| self.buffers[i].0);
            let mut atoms = n as i32;
            let pointer = |p: &mut u64| p as *mut u64 as *mut c_void;
            let mut embed = [
                pointer(&mut addresses[0]),
                pointer(&mut addresses[1]),
                pointer(&mut addresses[2]),
                pointer(&mut addresses[3]),
                pointer(&mut addresses[4]),
                &mut atoms as *mut i32 as *mut c_void,
                pointer(&mut addresses[5]),
                pointer(&mut addresses[6]),
                pointer(&mut addresses[10]),
            ];
            self.launch(0, q, &mut embed)?;
            let mut fit = [
                pointer(&mut addresses[0]),
                pointer(&mut addresses[1]),
                pointer(&mut addresses[2]),
                pointer(&mut addresses[3]),
                &mut atoms as *mut i32 as *mut c_void,
                pointer(&mut addresses[5]),
                pointer(&mut addresses[7]),
                pointer(&mut addresses[8]),
                pointer(&mut addresses[10]),
            ];
            self.launch(1, n, &mut fit)?;
            let mut pull = [
                pointer(&mut addresses[2]),
                &mut atoms as *mut i32 as *mut c_void,
                pointer(&mut addresses[5]),
                pointer(&mut addresses[6]),
                pointer(&mut addresses[7]),
                pointer(&mut addresses[9]),
                pointer(&mut addresses[10]),
            ];
            self.launch(2, q, &mut pull)?;
            check((self.api.sync)(), "cuCtxSynchronize")?;
            self.last_timing_ms[1] = phase.elapsed().as_secs_f64() * 1000.0;
            let phase = std::time::Instant::now();
            let mut status = [0_i32];
            Self::download_into(self.api, self.buffers[10].0, &mut status)?;
            if status[0] != 0 {
                return Err(Error(
                    "nonfinite CUDA network/contraction intermediate".into(),
                ));
            }
            self.energies.resize(n, 0.0);
            self.gradients.resize(q * 4, 0.0);
            Self::download_into(self.api, self.buffers[8].0, &mut self.energies)?;
            Self::download_into(self.api, self.buffers[9].0, &mut self.gradients)?;
            self.last_timing_ms[2] = phase.elapsed().as_secs_f64() * 1000.0;
            Ok(())
        }
    }
    pub(super) fn full_frame(
        &mut self,
        rows: &[f64],
        tangent: &[f64],
        types: &[i32],
        parameters: usize,
        mut energy_seed: f64,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        unsafe {
            let _current = Current::enter(self.api, self.context)?;
            let phase = std::time::Instant::now();
            self.upload_current(2, rows)?;
            self.upload_current(3, types)?;
            self.upload_current(5, tangent)?;
            self.upload_current(8, &[0_i32])?;
            self.ensure(6, types.len() * parameters * 8)?;
            self.ensure(7, types.len() * 8)?;
            let mut address: [u64; 12] = std::array::from_fn(|i| self.buffers[i].0);
            let pointer = |p: &mut u64| p as *mut u64 as *mut c_void;
            let mut atoms = types.len() as i32;
            let mut count = parameters as i32;
            let mut args = [
                pointer(&mut address[0]),
                pointer(&mut address[1]),
                pointer(&mut address[2]),
                pointer(&mut address[5]),
                pointer(&mut address[3]),
                pointer(&mut address[4]),
                &mut atoms as *mut i32 as *mut c_void,
                &mut count as *mut i32 as *mut c_void,
                &mut energy_seed as *mut f64 as *mut c_void,
                pointer(&mut address[6]),
                pointer(&mut address[7]),
                pointer(&mut address[8]),
            ];
            self.last_timing_ms[0] = phase.elapsed().as_secs_f64() * 1000.;
            let phase = std::time::Instant::now();
            self.launch_grid(0, (types.len() * parameters).div_ceil(64), 64, &mut args)?;
            check((self.api.sync)(), "cuCtxSynchronize(full gradient)")?;
            self.last_timing_ms[1] = phase.elapsed().as_secs_f64() * 1000.;
            let phase = std::time::Instant::now();
            let mut status = [0_i32];
            Self::download_into(self.api, self.buffers[8].0, &mut status)?;
            if status[0] != 0 {
                return Err(Error("nonfinite CUDA full-gradient intermediate".into()));
            }
            let mut gradient = vec![0.0; types.len() * parameters];
            let mut energy = vec![0.0; types.len()];
            Self::download_into(self.api, self.buffers[6].0, &mut gradient)?;
            Self::download_into(self.api, self.buffers[7].0, &mut energy)?;
            self.last_timing_ms[2] = phase.elapsed().as_secs_f64() * 1000.;
            Ok((gradient, energy))
        }
    }
    pub(super) fn reverse_frame(
        &mut self,
        rows: &[f64],
        tangent: &[f64],
        types: &[i32],
        parameters: usize,
        mut energy_seed: f64,
        resources: &super::ReverseResources,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        unsafe {
            let _current = Current::enter(self.api, self.context)?;
            let phase = std::time::Instant::now();
            // Scratch has no persistent state. Release every growing allocation first,
            // so old+new buffers never exceed the planner's resident payload bound.
            let required = [
                (2, std::mem::size_of_val(rows)),
                (3, std::mem::size_of_val(types)),
                (5, std::mem::size_of_val(tangent)),
                (6, types.len() * parameters * 8),
                (7, types.len() * 8),
                (12, resources.cache_bytes),
                (13, resources.descriptor_adjoint_bytes),
                (14, resources.slot_gradient_bytes),
            ];
            for (index, bytes) in required {
                if self.buffers[index].1 < bytes && self.buffers[index].0 != 0 {
                    check(
                        (self.api.free)(self.buffers[index].0),
                        "cuMemFree_v2(reverse grow)",
                    )?;
                    self.buffers[index] = (0, 0);
                }
            }
            self.upload_current(2, rows)?;
            self.upload_current(3, types)?;
            self.upload_current(5, tangent)?;
            self.upload_current(8, &[0_i32])?;
            self.ensure(6, types.len() * parameters * 8)?;
            self.ensure(7, types.len() * 8)?;
            self.ensure(12, resources.cache_bytes)?;
            self.ensure(13, resources.descriptor_adjoint_bytes)?;
            self.ensure(14, resources.slot_gradient_bytes)?;
            let mut address: [u64; 16] = std::array::from_fn(|i| self.buffers[i].0);
            let pointer = |p: &mut u64| p as *mut u64 as *mut c_void;
            let mut atoms = types.len() as i32;
            let mut count = parameters as i32;
            let mut args = [
                pointer(&mut address[0]),
                pointer(&mut address[1]),
                pointer(&mut address[2]),
                pointer(&mut address[5]),
                pointer(&mut address[3]),
                pointer(&mut address[4]),
                &mut atoms as *mut i32 as *mut c_void,
                &mut count as *mut i32 as *mut c_void,
                &mut energy_seed as *mut f64 as *mut c_void,
                pointer(&mut address[6]),
                pointer(&mut address[7]),
                pointer(&mut address[8]),
                pointer(&mut address[12]),
                pointer(&mut address[13]),
                pointer(&mut address[14]),
                pointer(&mut address[15]),
            ];
            self.last_timing_ms[0] = phase.elapsed().as_secs_f64() * 1000.;
            let phase = std::time::Instant::now();
            let slots = rows.len() / 4;
            self.launch_grid(0, slots.div_ceil(64), 64, &mut args)?;
            self.launch_grid(2, types.len().div_ceil(32), 32, &mut args)?;
            self.launch_grid(3, slots.div_ceil(64), 64, &mut args)?;
            self.launch_grid(4, (types.len() * parameters).div_ceil(64), 64, &mut args)?;
            check((self.api.sync)(), "cuCtxSynchronize(reverse gradient)")?;
            self.last_timing_ms[1] = phase.elapsed().as_secs_f64() * 1000.;
            let phase = std::time::Instant::now();
            let mut status = [0_i32];
            Self::download_into(self.api, self.buffers[8].0, &mut status)?;
            if status[0] != 0 {
                return Err(Error("nonfinite CUDA reverse-gradient intermediate".into()));
            }
            let mut gradient = vec![0.; types.len() * parameters];
            let mut energy = vec![0.; types.len()];
            Self::download_into(self.api, self.buffers[6].0, &mut gradient)?;
            Self::download_into(self.api, self.buffers[7].0, &mut energy)?;
            self.last_timing_ms[2] = phase.elapsed().as_secs_f64() * 1000.;
            Ok((gradient, energy))
        }
    }
    pub(super) fn reverse_tile(
        &mut self,
        rows: &[f64],
        tangent: &[f64],
        types: &[i32],
        parameters: usize,
        energy_seed: &[f64],
        resources: &super::ReverseResources,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        if energy_seed.len() != types.len() {
            return Err(Error("CUDA tile energy seed shape mismatch".into()));
        }
        unsafe {
            let _current = Current::enter(self.api, self.context)?;
            let phase = std::time::Instant::now();
            // Scratch has no persistent state. Release every growing allocation first,
            // so old+new buffers never exceed the planner's resident payload bound.
            let required = [
                (6, types.len() * parameters * 8),
                (7, types.len() * 8),
                (12, resources.cache_bytes),
                (13, resources.descriptor_adjoint_bytes),
                (14, resources.slot_gradient_bytes),
            ];
            for (index, bytes) in required {
                if self.buffers[index].1 < bytes && self.buffers[index].0 != 0 {
                    check(
                        (self.api.free)(self.buffers[index].0),
                        "cuMemFree_v2(reverse grow)",
                    )?;
                    self.buffers[index] = (0, 0);
                }
            }
            self.upload_current(8, &[0_i32])?;
            self.ensure(6, types.len() * parameters * 8)?;
            self.ensure(7, types.len() * 8)?;
            self.ensure(12, resources.cache_bytes)?;
            self.ensure(13, resources.descriptor_adjoint_bytes)?;
            self.ensure(14, resources.slot_gradient_bytes)?;
            use super::transfer::{f64_bytes, i32_bytes};
            let mut address = self.upload_packed_inputs(&[
                (2, f64_bytes(rows)),
                (3, i32_bytes(types)),
                (5, f64_bytes(tangent)),
                (16, f64_bytes(energy_seed)),
            ])?;
            let pointer = |p: &mut u64| p as *mut u64 as *mut c_void;
            let mut atoms = types.len() as i32;
            let mut count = parameters as i32;
            let mut args = [
                pointer(&mut address[0]),
                pointer(&mut address[1]),
                pointer(&mut address[2]),
                pointer(&mut address[5]),
                pointer(&mut address[3]),
                pointer(&mut address[4]),
                &mut atoms as *mut i32 as *mut c_void,
                &mut count as *mut i32 as *mut c_void,
                pointer(&mut address[16]),
                pointer(&mut address[6]),
                pointer(&mut address[7]),
                pointer(&mut address[8]),
                pointer(&mut address[12]),
                pointer(&mut address[13]),
                pointer(&mut address[14]),
                pointer(&mut address[15]),
            ];
            self.last_timing_ms[0] = phase.elapsed().as_secs_f64() * 1000.;
            let phase = std::time::Instant::now();
            let slots = rows.len() / 4;
            self.launch_grid(0, slots.div_ceil(64), 64, &mut args)?;
            self.launch_grid(2, types.len().div_ceil(32), 32, &mut args)?;
            self.launch_grid(3, slots.div_ceil(64), 64, &mut args)?;
            self.launch_grid(4, (types.len() * parameters).div_ceil(64), 64, &mut args)?;
            check((self.api.sync)(), "cuCtxSynchronize(reverse gradient)")?;
            self.last_timing_ms[1] = phase.elapsed().as_secs_f64() * 1000.;
            let phase = std::time::Instant::now();
            let mut status = [0_i32];
            Self::download_into(self.api, self.buffers[8].0, &mut status)?;
            if status[0] != 0 {
                return Err(Error("nonfinite CUDA reverse-gradient intermediate".into()));
            }
            let mut gradient = vec![0.; types.len() * parameters];
            let mut energy = vec![0.; types.len()];
            Self::download_into(self.api, self.buffers[6].0, &mut gradient)?;
            Self::download_into(self.api, self.buffers[7].0, &mut energy)?;
            self.last_timing_ms[2] = phase.elapsed().as_secs_f64() * 1000.;
            Ok((gradient, energy))
        }
    }
    pub(super) fn resident_buffer_bytes(&self) -> usize {
        self.buffers.iter().map(|b| b.1).sum()
    }
    pub(super) fn full_adam(
        &mut self,
        gradient: &[f64],
        mut beta1: f64,
        mut beta2: f64,
        mut c1: f64,
        mut c2: f64,
        mut rate: f64,
        mut epsilon: f64,
        snapshot: bool,
    ) -> Result<Option<(Vec<f64>, Vec<f64>, Vec<f64>)>> {
        unsafe {
            let _current = Current::enter(self.api, self.context)?;
            self.upload_current(11, gradient)?;
            self.upload_current(8, &[0_i32])?;
            let mut address: [u64; 12] = std::array::from_fn(|i| self.buffers[i].0);
            let pointer = |p: &mut u64| p as *mut u64 as *mut c_void;
            let scalar = |p: &mut f64| p as *mut f64 as *mut c_void;
            let mut count = gradient.len() as i32;
            let mut args = [
                pointer(&mut address[1]),
                pointer(&mut address[9]),
                pointer(&mut address[10]),
                pointer(&mut address[11]),
                &mut count as *mut i32 as *mut c_void,
                scalar(&mut beta1),
                scalar(&mut beta2),
                scalar(&mut c1),
                scalar(&mut c2),
                scalar(&mut rate),
                scalar(&mut epsilon),
                pointer(&mut address[8]),
            ];
            self.launch_grid(1, gradient.len().div_ceil(64), 64, &mut args)?;
            check((self.api.sync)(), "cuCtxSynchronize(full Adam)")?;
            let mut status = [0_i32];
            Self::download_into(self.api, self.buffers[8].0, &mut status)?;
            if status[0] != 0 {
                return Err(Error(
                    "nonfinite CUDA Adam update; last completed checkpoint remains usable".into(),
                ));
            }
            if snapshot {
                self.download_adam_current(gradient.len()).map(Some)
            } else {
                Ok(None)
            }
        }
    }
    pub(super) fn adam_snapshot(
        &mut self,
        parameters: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        unsafe {
            let _current = Current::enter(self.api, self.context)?;
            self.download_adam_current(parameters)
        }
    }
    unsafe fn download_adam_current(
        &self,
        parameters: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let mut values = vec![0.0; parameters];
        let mut first = values.clone();
        let mut second = values.clone();
        Self::download_into(self.api, self.buffers[1].0, &mut values)?;
        Self::download_into(self.api, self.buffers[9].0, &mut first)?;
        Self::download_into(self.api, self.buffers[10].0, &mut second)?;
        Ok((values, first, second))
    }
    unsafe fn launch(&self, index: usize, threads: usize, args: &mut [*mut c_void]) -> Result<()> {
        self.launch_grid(
            index,
            if index == 1 {
                threads
            } else {
                threads.div_ceil(64)
            },
            if index == 1 { 128 } else { 64 },
            args,
        )
    }
    unsafe fn launch_grid(
        &self,
        index: usize,
        blocks: usize,
        threads: usize,
        args: &mut [*mut c_void],
    ) -> Result<()> {
        check(
            (self.api.launch)(
                self.functions[index],
                blocks as u32,
                1,
                1,
                threads as u32,
                1,
                1,
                0,
                ptr::null_mut(),
                args.as_mut_ptr(),
                ptr::null_mut(),
            ),
            "cuLaunchKernel",
        )
    }
}
impl Drop for Runtime {
    fn drop(&mut self) {
        unsafe {
            // Best-effort cleanup only. No reset, retry loop, or global device mutation.
            if let Ok(_current) = Current::enter(self.api, self.context) {
                for &(pointer, _) in &self.buffers {
                    if pointer != 0 {
                        (self.api.free)(pointer);
                    }
                }
                if !self.module.is_null() {
                    (self.api.module_unload)(self.module);
                }
                (self.api.ctx_destroy)(self.context);
            }
        }
    }
}
