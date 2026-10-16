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
    charge_spin: Option<Int>,
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
                // Optional API27 metadata: retain API26 runtime compatibility,
                // but never silently use a default condition on a newer model.
                charge_spin: library
                    .get::<Int>(b"DP_DeepPotGetDimChgSpin\0")
                    .ok()
                    .map(|s| *s),
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

/// Model-specific inputs exposed by the official runtime, independent of the
/// common EFV Potential contract. Parameter units come from the training data.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelDetails {
    pub frame_parameter_dim: usize,
    pub atom_parameter_dim: usize,
}

/// One frame's explicit parameters. No broadcasting, inferred units, default
/// values or silent zero filling is performed. Atom rows follow input order.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct EvalParameters {
    #[serde(default)]
    pub fparam: Vec<f64>,
    #[serde(default)]
    pub aparam: Vec<Vec<f64>>,
}

/// Optional runtime decomposition, in input atom order. Atomic virials follow
/// the same row-major convention and units as total virial, without dividing by volume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomicEvaluation {
    pub energies: Vec<f64>,
    pub virials: Vec<[[f64; 3]; 3]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedEvaluation {
    pub evaluation: Evaluation,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub atomic: Option<AtomicEvaluation>,
}

#[derive(Default)]
struct Buffers {
    coords: Vec<f64>,
    cells: Vec<f64>,
    fparam: Vec<f64>,
    aparam: Vec<f64>,
    energies: Vec<f64>,
    forces: Vec<f64>,
    virials: Vec<f64>,
    atomic_energies: Vec<f64>,
    atomic_virials: Vec<f64>,
}

// The C wrapper multiplies lengths as signed C ints. Check the individual
// products, not merely the dimensions, before allocation or pointer handoff.
fn c_product(dimensions: &[usize]) -> Result<usize> {
    dimensions.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d)
            .filter(|&v| v <= c_int::MAX as usize)
            .ok_or_else(|| {
                Error("batch buffer dimensions exceed the C API signed-int limit".into())
            })
    })
}

fn nan_buffer(buffer: &mut Vec<f64>, len: usize) {
    buffer.resize(len, f64::NAN);
    buffer.fill(f64::NAN);
}

fn input_pointer(values: &[f64]) -> *const f64 {
    if values.is_empty() {
        ptr::null()
    } else {
        values.as_ptr()
    }
}

fn matrix(values: &[f64]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| values[3 * i + j]))
}

/// The model is loaded once, released once and deliberately not Send or Sync.
pub struct DeepMd {
    api: Api,
    handle: NonNull<c_void>,
    metadata: ModelMetadata,
    details: ModelDetails,
    cuda: Option<cuda::CudaGuard>,
    cuda_kernel_launches: u64,
    buffers: Buffers,
    _single_thread: PhantomData<Rc<()>>,
}

impl DeepMd {
    pub fn details(&self) -> &ModelDetails {
        &self.details
    }
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
            details: ModelDetails::default(),
            cuda: None,
            cuda_kernel_launches: 0,
            buffers: Buffers::default(),
        };
        let h = model.handle.as_ptr();
        model.api.check(h, "load model")?;
        let ntypes = unsafe { (model.api.ntypes)(h) };
        let nspin = unsafe { (model.api.nspin)(h) };
        let fparam = unsafe { (model.api.fparam)(h) };
        let aparam = unsafe { (model.api.aparam)(h) };
        let charge_spin = model
            .api
            .charge_spin
            .map(|query| unsafe { query(h) })
            .unwrap_or(0);
        model.metadata.cutoff = unsafe { (model.api.cutoff)(h) };
        model.api.check(h, "read model metadata")?;
        if charge_spin != 0 {
            return Err(Error(format!("unsupported model: charge/spin input dimension={charge_spin}; the Compute2 adapter does not supply charge/spin conditions")));
        }
        if nspin != 0 || fparam < 0 || aparam < 0 {
            return Err(Error(format!("unsupported model metadata: spin types={nspin}, frame parameters={fparam}, atom parameters={aparam}; spin must be zero and parameter dimensions nonnegative")));
        }
        model.details = ModelDetails {
            frame_parameter_dim: fparam as usize,
            atom_parameter_dim: aparam as usize,
        };
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

impl DeepMd {
    /// Evaluate a homogeneous group with exactly one combined C API call.
    /// All frames must share atom count, ordered species and periodicity.
    /// Parameters are explicit per frame, with no broadcasting or defaults.
    /// Optional atomic outputs share the same call. Scratch arrays are retained.
    pub fn evaluate(
        &mut self,
        frames: &[Frame],
        parameters: Option<&[EvalParameters]>,
        atomic: bool,
    ) -> Result<Vec<DetailedEvaluation>> {
        let first = frames
            .first()
            .ok_or_else(|| Error("at least one frame is required".into()))?;
        let types = first.type_ids(&self.metadata)?;
        let nf = frames.len();
        let n = first.positions.len();
        let df = self.details.frame_parameter_dim;
        let da = self.details.atom_parameter_dim;
        let nx = c_product(&[nf, n, 3])?;
        let nv = c_product(&[nf, 9])?;
        // Check the largest C-wrapper product even when atom outputs are omitted.
        let nav = c_product(&[nf, n, 9])?;
        let ne = c_product(&[nf, n])?;
        let nfp = c_product(&[nf, df])?;
        let nap = c_product(&[nf, n, da])?;
        let values = [
            nx,
            nx,
            nv,
            nv,
            nf,
            nfp,
            nap,
            if atomic { ne } else { 0 },
            if atomic { nav } else { 0 },
        ]
        .iter()
        .try_fold(0usize, |a, b| a.checked_add(*b))
        .ok_or_else(|| Error("batch allocation size overflow".into()))?;
        if values > 32 * 1024 * 1024 {
            return Err(Error(
                "batch scratch arrays exceed 256 MiB; reduce batch_size".into(),
            ));
        }
        if let Some(params) = parameters {
            if params.len() != nf {
                return Err(Error(format!(
                    "parameters must contain one entry per frame: expected {nf}, got {}",
                    params.len()
                )));
            }
        } else if df != 0 || da != 0 {
            return Err(Error(format!("model requires explicit parameters: fparam dimension={df}, aparam dimension={da}; use DeepMd::evaluate or the standalone evaluator parameters field")));
        }
        // Validate the entire batch before packing or entering the runtime.
        for (i, frame) in frames.iter().enumerate() {
            frame.validate()?;
            if frame.units != first.units
                || frame.species != first.species
                || frame.positions.len() != n
                || frame.cell.is_some() != first.cell.is_some()
            {
                return Err(Error(format!("frame {i}: a C API batch requires identical atom count, ordered species, units and periodicity; use batch_size=1 for heterogeneous inputs")));
            }
            if let Some(params) = parameters {
                let p = &params[i];
                if p.fparam.len() != df || !p.fparam.iter().all(|x| x.is_finite()) {
                    return Err(Error(format!(
                        "frame {i}: fparam requires exactly {df} finite values"
                    )));
                }
                let rows = if da == 0 { 0 } else { n };
                if p.aparam.len() != rows
                    || p.aparam
                        .iter()
                        .any(|row| row.len() != da || !row.iter().all(|x| x.is_finite()))
                {
                    return Err(Error(format!("frame {i}: aparam requires {rows} atom rows of {da} finite values in input order")));
                }
            }
        }
        let b = &mut self.buffers;
        b.coords.clear();
        b.cells.clear();
        b.fparam.clear();
        b.aparam.clear();
        for (i, frame) in frames.iter().enumerate() {
            b.coords.extend(frame.positions.iter().flatten().copied());
            if let Some(cell) = &frame.cell {
                b.cells.extend(cell.iter().flatten().copied());
            }
            if let Some(params) = parameters {
                b.fparam.extend_from_slice(&params[i].fparam);
                b.aparam.extend(params[i].aparam.iter().flatten().copied());
            }
        }
        nan_buffer(&mut b.energies, nf);
        nan_buffer(&mut b.forces, nx);
        nan_buffer(&mut b.virials, nv);
        nan_buffer(&mut b.atomic_energies, if atomic { ne } else { 0 });
        nan_buffer(&mut b.atomic_virials, if atomic { nav } else { 0 });
        let h = self.handle.as_ptr();
        self.api.check(h, "before evaluation")?;
        if let Some(cuda) = &mut self.cuda {
            cuda.begin()?;
        }
        unsafe {
            (self.api.compute)(
                h,
                nf as c_int,
                n as c_int,
                b.coords.as_ptr(),
                types.as_ptr(),
                input_pointer(&b.cells),
                input_pointer(&b.fparam),
                input_pointer(&b.aparam),
                b.energies.as_mut_ptr(),
                b.forces.as_mut_ptr(),
                b.virials.as_mut_ptr(),
                if atomic {
                    b.atomic_energies.as_mut_ptr()
                } else {
                    ptr::null_mut()
                },
                if atomic {
                    b.atomic_virials.as_mut_ptr()
                } else {
                    ptr::null_mut()
                },
            );
        }
        let runtime_result = self.api.check(h, "evaluate model");
        // Always finish verification/synchronization, including on runtime error.
        let gpu_result = self.cuda.as_mut().map(|cuda| cuda.finish()).transpose();
        runtime_result?;
        let launches = gpu_result?;
        if !b
            .atomic_energies
            .iter()
            .chain(&b.atomic_virials)
            .all(|x| x.is_finite())
        {
            return Err(Error(
                "atomic outputs contain missing or nonfinite values".into(),
            ));
        }
        let mut outputs = Vec::with_capacity(nf);
        for i in 0..nf {
            let evaluation = Evaluation {
                energy: b.energies[i],
                forces: b.forces[i * n * 3..(i + 1) * n * 3]
                    .chunks_exact(3)
                    .map(|v| [v[0], v[1], v[2]])
                    .collect(),
                virial: matrix(&b.virials[i * 9..(i + 1) * 9]),
                units: Units::EvAngstrom,
            };
            evaluation.validate(n)?;
            let atomic = atomic.then(|| AtomicEvaluation {
                energies: b.atomic_energies[i * n..(i + 1) * n].to_vec(),
                virials: b.atomic_virials[i * n * 9..(i + 1) * n * 9]
                    .chunks_exact(9)
                    .map(matrix)
                    .collect(),
            });
            outputs.push(DetailedEvaluation { evaluation, atomic });
        }
        if let Some(launches) = launches {
            self.cuda_kernel_launches = self.cuda_kernel_launches.saturating_add(launches);
            self.metadata.backend = "deepmd-c-api-cuda-kernels-verified".into();
        }
        Ok(outputs)
    }
}

impl Potential for DeepMd {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn eval(&mut self, frame: &Frame) -> Result<Evaluation> {
        Ok(self
            .evaluate(std::slice::from_ref(frame), None, false)?
            .remove(0)
            .evaluation)
    }
}

impl Drop for DeepMd {
    fn drop(&mut self) {
        unsafe { (self.api.delete)(self.handle.as_ptr()) };
    }
}

#[cfg(test)]
mod tests;
