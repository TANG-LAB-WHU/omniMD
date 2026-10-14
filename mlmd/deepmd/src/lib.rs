//! A fallible, double-I/O wrapper around the official DeePMD-kit C API.
//! This crate does not implement the model in Rust. See README.md for scope.
use libloading::Library;
mod cuda;
mod device;
pub use device::Device;
use omnimd_ml_contract::{Error, Evaluation, Frame, ModelMetadata, Potential, Result, Units};
use serde::{Deserialize, Serialize};
use std::{
    ffi::{c_char, c_int, c_void, CStr, CString},
    marker::PhantomData,
    path::{Path, PathBuf},
    ptr::{self, NonNull},
    rc::Rc,
};

type Handle = *mut c_void;
type New = unsafe extern "C" fn(*const c_char, c_int, *const c_char, c_int) -> Handle;
type Delete = unsafe extern "C" fn(Handle);
type Text = unsafe extern "C" fn(Handle) -> *const c_char;
type DeleteText = unsafe extern "C" fn(*const c_char);
type Int = unsafe extern "C" fn(Handle) -> c_int;
type Cutoff = unsafe extern "C" fn(Handle) -> f64;
type Compute = unsafe extern "C" fn(
    Handle,
    c_int,
    c_int,
    *const f64,
    *const c_int,
    *const f64,
    *const f64,
    *const f64,
    *mut f64,
    *mut f64,
    *mut f64,
    *mut f64,
    *mut f64,
);

struct Api {
    new: New,
    delete: Delete,
    check: Text,
    type_map: Text,
    delete_text: DeleteText,
    ntypes: Int,
    nspin: Int,
    fparam: Int,
    aparam: Int,
    cutoff: Cutoff,
    compute: Compute,
    // Keep the code and its dependencies alive until after the model is deleted.
    _library: Option<Library>,
}

impl Api {
    unsafe fn load(path: &Path) -> Result<Self> {
        let library = unsafe { Library::new(path) }
            .map_err(|e| Error(format!("load DeepMD C library {}: {e}", path.display())))?;
        unsafe fn symbol<T: Copy>(lib: &Library, name: &[u8]) -> Result<T> {
            unsafe { lib.get::<T>(name) }
                .map(|s| *s)
                .map_err(|e| Error(format!("required DeepMD C API symbol missing: {e}")))
        }
        // These declarations match the official v3.1.3 C API v26. No C++ ABI is used.
        unsafe {
            Ok(Self {
                new: symbol(&library, b"DP_NewDeepPotWithParam2\0")?,
                delete: symbol(&library, b"DP_DeleteDeepPot\0")?,
                check: symbol(&library, b"DP_DeepPotCheckOK\0")?,
                type_map: symbol(&library, b"DP_DeepPotGetTypeMap\0")?,
                delete_text: symbol(&library, b"DP_DeleteChar\0")?,
                ntypes: symbol(&library, b"DP_DeepPotGetNumbTypes\0")?,
                nspin: symbol(&library, b"DP_DeepPotGetNumbTypesSpin\0")?,
                fparam: symbol(&library, b"DP_DeepPotGetDimFParam\0")?,
                aparam: symbol(&library, b"DP_DeepPotGetDimAParam\0")?,
                cutoff: symbol(&library, b"DP_DeepPotGetCutoff\0")?,
                compute: symbol(&library, b"DP_DeepPotCompute2\0")?,
                _library: Some(library),
            })
        }
    }

    fn text(&self, value: *const c_char, context: &str) -> Result<String> {
        if value.is_null() {
            return Err(Error(format!("{context}: null C string")));
        }
        // Both metadata and CheckOK return new[] allocations in the official API.
        // Copy before freeing, including on a UTF-8 error; never use libc free().
        let result = unsafe { CStr::from_ptr(value) }
            .to_str()
            .map(str::to_owned)
            .map_err(|e| Error(format!("{context}: invalid UTF-8: {e}")));
        unsafe { (self.delete_text)(value) };
        result
    }

    fn check(&self, handle: Handle, context: &str) -> Result<()> {
        let error = self.text(unsafe { (self.check)(handle) }, context)?;
        if error.is_empty() {
            Ok(())
        } else {
            Err(Error(format!("{context}: {error}")))
        }
    }
}

/// Trusted runtime libraries used only by an explicit CUDA request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CudaLibraries {
    pub runtime: PathBuf,
    pub cupti: PathBuf,
}

/// All paths and model type order are explicit. The library must be trusted.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoadOptions {
    pub library: PathBuf,
    pub model: PathBuf,
    #[serde(default)]
    pub device: Device,
    #[serde(default)]
    pub cuda: Option<CudaLibraries>,
    pub type_map: Vec<String>,
    pub units: Units,
}

/// The model is loaded once, released once and deliberately not Send or Sync.
pub struct DeepMd {
    api: Api,
    handle: NonNull<c_void>,
    metadata: ModelMetadata,
    cuda: Option<cuda::CudaGuard>,
    cuda_kernel_launches: u64,
    coords: Vec<f64>,
    forces: Vec<f64>,
    _single_thread: PhantomData<Rc<()>>,
}

impl DeepMd {
    /// Successful direct GPU launches observed during completed evaluations.
    /// None means CPU; zero means the CUDA model has not yet been evaluated.
    pub fn cuda_kernel_launches(&self) -> Option<u64> {
        self.cuda.as_ref().map(|_| self.cuda_kernel_launches)
    }

    /// Load a trusted official DeepMD shared library and one frozen model.
    ///
    /// # Safety
    /// The library must implement the official C ABI correctly. Start this in a
    /// fresh process configured with Device::configure_worker *before* any
    /// accelerator/runtime initialization. Do not subsequently mutate device
    /// visibility. Use matching official CUDA/CUPTI libraries for CUDA requests.
    /// The CLI provides process isolation. Loading arbitrary native code
    /// is not a sandbox.
    pub unsafe fn load(options: &LoadOptions) -> Result<Self> {
        options.device.validate_environment()?;
        for (name, path) in [("library", &options.library), ("model", &options.model)] {
            if !path.is_absolute() || !path.is_file() {
                return Err(Error(format!(
                    "{name} must be an absolute path to an existing file: {}",
                    path.display()
                )));
            }
        }
        let filename = CString::new(
            options
                .model
                .to_str()
                .ok_or_else(|| Error("model path must be valid UTF-8".into()))?,
        )
        .map_err(|_| Error("model path contains NUL".into()))?;
        let cuda = match options.device {
            Device::Cpu => None,
            Device::Cuda(_) => {
                if options.model.extension().and_then(|s| s.to_str()) != Some("pb") {
                    return Err(Error("CUDA C API currently supports TensorFlow frozen .pb models only; select cpu for other supported official backends".into()));
                }
                let libs = options.cuda.as_ref().ok_or_else(|| Error("CUDA requires explicit cuda.runtime and cuda.cupti library paths; no automatic CPU fallback, select device=cpu to run on CPU".into()))?;
                Some(unsafe { cuda::CudaGuard::load(&libs.runtime, &libs.cupti) }?)
            }
        };
        let api = unsafe { Api::load(&options.library) }?;
        let mut model = Self::initialize(api, &filename, &options.type_map, options.units)?;
        model.cuda = cuda;
        if model.cuda.is_some() {
            model.metadata.backend = "deepmd-c-api-cuda-pending-evaluation".into();
        }
        Ok(model)
    }

    fn initialize(api: Api, filename: &CStr, expected: &[String], units: Units) -> Result<Self> {
        let handle =
            NonNull::new(unsafe { (api.new)(filename.as_ptr(), 0, b"\0".as_ptr().cast(), 0) })
                .ok_or_else(|| Error("DeepMD constructor returned a null model".into()))?;
        // Construct RAII owner before fallible metadata queries so all failures drop.
        let mut model = Self {
            api,
            handle,
            metadata: ModelMetadata {
                type_map: vec![],
                cutoff: f64::NAN,
                backend: "deepmd-c-api".into(),
                precision: "float64-io".into(),
                units,
            },
            _single_thread: PhantomData,
            cuda: None,
            cuda_kernel_launches: 0,
            coords: Vec::new(),
            forces: Vec::new(),
        };
        let h = model.handle.as_ptr();
        model.api.check(h, "load model")?;
        let ntypes = unsafe { (model.api.ntypes)(h) };
        let nspin = unsafe { (model.api.nspin)(h) };
        let fparam = unsafe { (model.api.fparam)(h) };
        let aparam = unsafe { (model.api.aparam)(h) };
        model.metadata.cutoff = unsafe { (model.api.cutoff)(h) };
        model.api.check(h, "read model metadata")?;
        if nspin != 0 || fparam != 0 || aparam != 0 {
            return Err(Error(format!("unsupported model: spin types={nspin}, frame parameters={fparam}, atom parameters={aparam}; all must be zero")));
        }
        let map = model
            .api
            .text(unsafe { (model.api.type_map)(h) }, "read model type map")?;
        model.api.check(h, "read model type map")?;
        model.metadata.type_map = map.split_whitespace().map(str::to_owned).collect();
        model.metadata.validate()?;
        if ntypes <= 0 || model.metadata.type_map.len() != ntypes as usize {
            return Err(Error(
                "model type map length disagrees with model type count".into(),
            ));
        }
        if model.metadata.type_map != expected {
            return Err(Error(format!(
                "requested type order {expected:?} differs from model {:?}",
                model.metadata.type_map
            )));
        }
        Ok(model)
    }
}

impl Potential for DeepMd {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn eval(&mut self, frame: &Frame) -> Result<Evaluation> {
        let types = frame.type_ids(&self.metadata)?;
        let n = frame.positions.len();
        self.coords.clear();
        self.coords
            .extend(frame.positions.iter().flatten().copied());
        let cell = frame
            .cell
            .map(|h| std::array::from_fn::<_, 9, _>(|i| h[i / 3][i % 3]));
        let mut energy = f64::NAN;
        self.forces.resize(self.coords.len(), f64::NAN);
        self.forces.fill(f64::NAN);
        let mut virial = [f64::NAN; 9];
        let h = self.handle.as_ptr();
        self.api.check(h, "before evaluation")?;
        if let Some(cuda) = &mut self.cuda {
            cuda.begin()?;
        }
        unsafe {
            // Exactly one call for energy + force + virial; nframes is always one.
            (self.api.compute)(
                h,
                1,
                n as c_int,
                self.coords.as_ptr(),
                types.as_ptr(),
                cell.as_ref().map_or(ptr::null(), |v| v.as_ptr()),
                ptr::null(),
                ptr::null(),
                &mut energy,
                self.forces.as_mut_ptr(),
                virial.as_mut_ptr(),
                ptr::null_mut(),
                ptr::null_mut(),
            );
        }
        let runtime_result = self.api.check(h, "evaluate model");
        // Synchronize and verify GPU work even when the runtime reported an error.
        let gpu_result = self.cuda.as_mut().map(|cuda| cuda.finish()).transpose();
        runtime_result?;
        let launches = gpu_result?;
        let output = Evaluation {
            energy,
            forces: self
                .forces
                .chunks_exact(3)
                .map(|v| [v[0], v[1], v[2]])
                .collect(),
            virial: std::array::from_fn(|i| std::array::from_fn(|j| virial[3 * i + j])),
            units: Units::EvAngstrom,
        };
        output.validate(n)?;
        if let Some(launches) = launches {
            self.cuda_kernel_launches = self.cuda_kernel_launches.saturating_add(launches);
        }
        if self.cuda.is_some() {
            self.metadata.backend = "deepmd-c-api-cuda-kernels-verified".into();
        }
        Ok(output)
    }
}

impl Drop for DeepMd {
    fn drop(&mut self) {
        unsafe { (self.api.delete)(self.handle.as_ptr()) };
    }
}

#[cfg(test)]
mod tests;
