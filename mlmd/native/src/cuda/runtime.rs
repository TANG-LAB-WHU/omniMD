//! Minimal dynamic NVIDIA Driver/NVRTC bindings. No link-time CUDA dependency.
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
    functions: [Handle; 3],
    buffers: Vec<(u64, usize)>,
    pub device_info: String,
    pub kernel_resources: Vec<KernelResources>,
    pub last_timing_ms: [f64; 3],
    pub type_uploads: usize,
    pub last_upload_bytes: usize,
    cached_types: Vec<i32>,
    pub energies: Vec<f64>,
    pub gradients: Vec<f64>,
    _library: Library,
    _not_send: PhantomData<Rc<()>>,
}
impl Runtime {
    pub fn new(source: &str, ordinal: i32) -> Result<Self> {
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
            let mut previous = ptr::null_mut();
            check((api.ctx_current)(&mut previous), "cuCtxGetCurrent")?;
            let mut context = ptr::null_mut();
            check((api.ctx_create)(&mut context, 0, device), "cuCtxCreate_v2")?;
            let restore = Current { api, previous };
            let mut rt = Self {
                api,
                context,
                module: ptr::null_mut(),
                functions: [ptr::null_mut(); 3],
                buffers: vec![(0, 0); 11],
                device_info,
                kernel_resources: Vec::new(),
                last_timing_ms: [0.0; 3],
                type_uploads: 0,
                last_upload_bytes: 0,
                cached_types: Vec::new(),
                energies: Vec::new(),
                gradients: Vec::new(),
                _library: library,
                _not_send: PhantomData,
            };
            check(
                (api.module_load)(&mut rt.module, ptx.as_ptr() as *const c_void),
                "cuModuleLoadData",
            )?;
            for (i, name) in ["embed", "fit", "pullback"].iter().enumerate() {
                let name = CString::new(*name).unwrap();
                check(
                    (api.function)(&mut rt.functions[i], rt.module, name.as_ptr()),
                    "cuModuleGetFunction",
                )?;
            }
            for (i, name) in ["embed", "fit", "pullback"].iter().enumerate() {
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
    unsafe fn launch(&self, index: usize, threads: usize, args: &mut [*mut c_void]) -> Result<()> {
        check(
            (self.api.launch)(
                self.functions[index],
                if index == 1 {
                    threads as u32
                } else {
                    threads.div_ceil(64) as u32
                },
                1,
                1,
                if index == 1 { 128 } else { 64 },
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
