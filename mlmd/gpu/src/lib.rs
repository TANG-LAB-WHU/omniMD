//! Device runtime shared by omniMD GPU paths (plan W5, ADR-2).
//!
//! One [`Device`] owns a CUDA context, its default stream, a cuBLAS handle and a
//! [`KernelCache`]. CUDA libraries are loaded at run time, so this crate builds
//! and its host-only tests run on machines without a driver; [`Device::new`]
//! then returns an error instead of panicking. Allocations go through the
//! stream-ordered driver allocator (`cuMemAllocAsync`), which already pools.
mod dense;
mod fused;
mod kernels;
mod nlist;
mod table;
mod train;
mod train_set;

pub use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr, DevicePtrMut,
    DeviceSlice, DriverError, LaunchConfig, PushKernelArg,
};
pub use dense::{Activation, DenseLayer, DenseNetwork, DenseTape};
pub use fused::{GradientNetwork, LayerParameters, ScalarNetwork};
pub use kernels::{CacheStats, KernelCache};
pub use nlist::{Cell, NeighborList, NeighborSearch, NeighborStats, COINCIDENT, TRUNCATED};
pub use table::{quintic_coefficients, ScalarTable, TableSegments};
pub use train::{TrainLayer, TrainNetwork, TrainTape};
pub use train_set::{SetTape, TrainSet};

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{CudaContext, DeviceRepr, ValidAsZeroBits};
use omnimd_ml_contract::{Error, Result};
use std::fmt::Debug;
use std::sync::Arc;

pub(crate) fn cuda_error<E: Debug>(operation: &'static str) -> impl FnOnce(E) -> Error {
    move |error| Error(format!("CUDA {operation} failed: {error:?}"))
}

/// Element types with device kernels and cuBLAS GEMM.
pub trait Scalar: DeviceRepr + ValidAsZeroBits + Copy + Default + Debug + 'static {
    /// Suffix of the `extern "C"` kernel instantiations (`f32` or `f64`).
    const SUFFIX: &'static str;
    const ONE: Self;
    const ZERO: Self;
    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;
    /// The first `count` values of a buffer of 64-bit words, as they lie in
    /// memory.
    fn from_words(words: &[u64], count: usize) -> Vec<Self>;
    #[doc(hidden)]
    unsafe fn gemm<A: DevicePtr<Self>, B: DevicePtr<Self>, C: DevicePtrMut<Self>>(
        blas: &CudaBlas,
        cfg: GemmConfig<Self>,
        a: &A,
        b: &B,
        c: &mut C,
    ) -> std::result::Result<(), cudarc::cublas::result::CublasError>;
    #[doc(hidden)]
    unsafe fn gemm_strided_batched(
        blas: &CudaBlas,
        cfg: StridedBatchedConfig<Self>,
        a: &CudaSlice<Self>,
        b: &CudaSlice<Self>,
        c: &mut CudaSlice<Self>,
    ) -> std::result::Result<(), cudarc::cublas::result::CublasError>;
}

macro_rules! scalar {
    ($t:ty, $suffix:literal) => {
        impl Scalar for $t {
            const SUFFIX: &'static str = $suffix;
            const ONE: Self = 1.0;
            const ZERO: Self = 0.0;
            fn from_f64(value: f64) -> Self {
                value as $t
            }
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn from_words(words: &[u64], count: usize) -> Vec<Self> {
                let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_ne_bytes()).collect();
                bytes
                    .chunks_exact(std::mem::size_of::<$t>())
                    .take(count)
                    .map(|chunk| {
                        let mut raw = [0; std::mem::size_of::<$t>()];
                        raw.copy_from_slice(chunk);
                        <$t>::from_ne_bytes(raw)
                    })
                    .collect()
            }
            unsafe fn gemm<A: DevicePtr<Self>, B: DevicePtr<Self>, C: DevicePtrMut<Self>>(
                blas: &CudaBlas,
                cfg: GemmConfig<Self>,
                a: &A,
                b: &B,
                c: &mut C,
            ) -> std::result::Result<(), cudarc::cublas::result::CublasError> {
                Gemm::<$t>::gemm(blas, cfg, a, b, c)
            }
            unsafe fn gemm_strided_batched(
                blas: &CudaBlas,
                cfg: StridedBatchedConfig<Self>,
                a: &CudaSlice<Self>,
                b: &CudaSlice<Self>,
                c: &mut CudaSlice<Self>,
            ) -> std::result::Result<(), cudarc::cublas::result::CublasError> {
                Gemm::<$t>::gemm_strided_batched(blas, cfg, a, b, c)
            }
        }
    };
}
scalar!(f32, "f32");
scalar!(f64, "f64");

/// Shape of one row-major product `c[m×n] = a[m×k] · b[k×n]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatmulShape {
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

impl MatmulShape {
    fn dims(self) -> Result<(i32, i32, i32)> {
        let fit = |v: usize| i32::try_from(v).ok().filter(|&v| v > 0);
        match (fit(self.m), fit(self.k), fit(self.n)) {
            (Some(m), Some(k), Some(n)) => Ok((m, k, n)),
            _ => Err(Error(format!(
                "matmul dimensions must be in 1..=i32::MAX, got {self:?}"
            ))),
        }
    }

    /// Row-major operands are the transposes of column-major ones, so cuBLAS
    /// computes cᵀ[n×m] = op(b)ᵀ[n×k] · aᵀ[k×m]. With `transpose_b`, `b` is
    /// stored as `[n][k]` and cuBLAS transposes it back.
    fn config<T: Scalar>(self, accumulate: bool, transpose_b: bool) -> Result<GemmConfig<T>> {
        use cudarc::cublas::sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T};
        let (m, k, n) = self.dims()?;
        Ok(GemmConfig {
            transa: if transpose_b {
                CUBLAS_OP_T
            } else {
                CUBLAS_OP_N
            },
            transb: CUBLAS_OP_N,
            m: n,
            n: m,
            k,
            alpha: T::ONE,
            lda: if transpose_b { k } else { n },
            ldb: k,
            beta: if accumulate { T::ONE } else { T::ZERO },
            ldc: n,
        })
    }
}

fn check_len(name: &str, actual: usize, needed: usize) -> Result<()> {
    if actual < needed {
        return Err(Error(format!(
            "{name} holds {actual} elements, needs {needed}"
        )));
    }
    Ok(())
}

pub struct Device {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    kernels: KernelCache,
    name: String,
    capability: (i32, i32),
}

impl Device {
    /// Opens GPU `ordinal`. Fails (never panics) when the driver, NVRTC or
    /// cuBLAS library cannot be loaded or no such device exists.
    pub fn new(ordinal: usize) -> Result<Self> {
        let missing = unsafe {
            [
                ("CUDA driver", cudarc::driver::sys::is_culib_present()),
                ("NVRTC", cudarc::nvrtc::sys::is_culib_present()),
                ("cuBLAS", cudarc::cublas::sys::is_culib_present()),
            ]
        };
        if let Some((library, _)) = missing.iter().find(|(_, present)| !present) {
            return Err(Error(format!(
                "{library} library not found; install the NVIDIA driver and CUDA 12 toolkit and put its bin directory on the library path"
            )));
        }
        let context = CudaContext::new(ordinal).map_err(cuda_error("context creation"))?;
        // cudarc can attach events to every buffer to order its uses across
        // streams, at the price of extra driver calls around every launch
        // once a second stream exists. No buffer of this crate crosses
        // streams: a handle allocates, uses and frees its buffers on its own
        // stream (see `sibling`).
        // SAFETY: the conditions of `disable_event_tracking` are that
        // invariant.
        unsafe { context.disable_event_tracking() };
        if context.has_async_alloc() {
            // Keep freed pool memory mapped: with the default release
            // threshold of 0 every synchronization hands it back to the OS
            // and the next evaluation pays to map it again.
            use cudarc::driver::sys;
            let mut pool = std::ptr::null_mut();
            let mut threshold = u64::MAX;
            unsafe {
                sys::cuDeviceGetDefaultMemPool(&mut pool, context.cu_device())
                    .result()
                    .map_err(cuda_error("memory pool query"))?;
                sys::cuMemPoolSetAttribute(
                    pool,
                    sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    (&mut threshold as *mut u64).cast(),
                )
                .result()
                .map_err(cuda_error("memory pool configuration"))?;
            }
        }
        let stream = context.default_stream();
        let blas = CudaBlas::new(stream.clone()).map_err(cuda_error("cuBLAS handle creation"))?;
        let capability = context
            .compute_capability()
            .map_err(cuda_error("compute capability query"))?;
        let name = context.name().map_err(cuda_error("device name query"))?;
        let kernels = KernelCache::new(context.clone(), capability);
        Ok(Self {
            context,
            stream,
            blas,
            kernels,
            name,
            capability,
        })
    }

    /// A second handle on the same GPU context with a stream, a cuBLAS
    /// handle and a kernel cache of its own. Work issued through it is
    /// ordered only against itself, so two threads that each drive one handle
    /// overlap on the device. Buffers must stay with the handle that made
    /// them: nothing orders a buffer's use on one stream against its use or
    /// release on another.
    pub fn sibling(&self) -> Result<Self> {
        let stream = self
            .context
            .new_stream()
            .map_err(cuda_error("stream creation"))?;
        let blas = CudaBlas::new(stream.clone()).map_err(cuda_error("cuBLAS handle creation"))?;
        Ok(Self {
            context: self.context.clone(),
            stream,
            blas,
            kernels: KernelCache::new(self.context.clone(), self.capability),
            name: self.name.clone(),
            capability: self.capability,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn compute_capability(&self) -> (i32, i32) {
        self.capability
    }

    /// Free and total device memory in bytes.
    pub fn memory_info(&self) -> Result<(usize, usize)> {
        self.context
            .mem_get_info()
            .map_err(cuda_error("memory query"))
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn kernels(&self) -> &KernelCache {
        &self.kernels
    }

    pub fn upload<T: DeviceRepr>(&self, host: &[T]) -> Result<CudaSlice<T>> {
        self.stream
            .clone_htod(host)
            .map_err(cuda_error("host-to-device copy"))
    }

    pub fn download<T: DeviceRepr>(&self, device: &CudaSlice<T>) -> Result<Vec<T>> {
        self.stream
            .clone_dtoh(device)
            .map_err(cuda_error("device-to-host copy"))
    }

    pub fn zeros<T: DeviceRepr + ValidAsZeroBits>(&self, len: usize) -> Result<CudaSlice<T>> {
        self.stream
            .alloc_zeros(len)
            .map_err(cuda_error("allocation"))
    }

    /// Allocation without the zero fill, for buffers a kernel overwrites
    /// completely before anything reads them.
    ///
    /// # Safety
    /// Every element must be written on the device before it is read.
    pub unsafe fn alloc_uninit<T: DeviceRepr>(&self, len: usize) -> Result<CudaSlice<T>> {
        self.stream.alloc(len).map_err(cuda_error("allocation"))
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(cuda_error("stream synchronize"))
    }

    /// Row-major `c = a·b`, or `c += a·b` when `accumulate`. Full-precision
    /// cuBLAS (no TF32) so FP32 results stay comparable to the CPU reference.
    /// Operands may be whole buffers or sub-range views.
    pub fn matmul<T: Scalar>(
        &self,
        a: &impl DevicePtr<T>,
        b: &impl DevicePtr<T>,
        c: &mut impl DevicePtrMut<T>,
        shape: MatmulShape,
        accumulate: bool,
    ) -> Result<()> {
        self.product(a, b, c, shape, accumulate, false)
    }

    /// Row-major `c = a·bᵀ` (or `+=`), with `b` stored as `[n][k]`; the
    /// reverse pass of a layer whose weights are `[k][n]`.
    pub fn matmul_bt<T: Scalar>(
        &self,
        a: &impl DevicePtr<T>,
        b: &impl DevicePtr<T>,
        c: &mut impl DevicePtrMut<T>,
        shape: MatmulShape,
        accumulate: bool,
    ) -> Result<()> {
        self.product(a, b, c, shape, accumulate, true)
    }

    /// Row-major `c[k][n] = aᵀ·b` (or `+=`) for `a[m][k]` and `b[m][n]`: the
    /// weight gradient of a layer from its inputs and its output adjoints.
    pub fn matmul_at<T: Scalar>(
        &self,
        a: &impl DevicePtr<T>,
        b: &impl DevicePtr<T>,
        c: &mut impl DevicePtrMut<T>,
        shape: MatmulShape,
        accumulate: bool,
    ) -> Result<()> {
        use cudarc::cublas::sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T};
        let (m, k, n) = shape.dims()?;
        check_len("matmul a", a.len(), shape.m * shape.k)?;
        check_len("matmul b", b.len(), shape.m * shape.n)?;
        check_len("matmul c", c.len(), shape.k * shape.n)?;
        // Column-major view: cᵀ[n×k] = bᵀ[n×m] · (aᵀ[k×m])ᵀ.
        let cfg = GemmConfig {
            transa: CUBLAS_OP_N,
            transb: CUBLAS_OP_T,
            m: n,
            n: k,
            k: m,
            alpha: T::ONE,
            lda: n,
            ldb: k,
            beta: if accumulate { T::ONE } else { T::ZERO },
            ldc: n,
        };
        unsafe { T::gemm(&self.blas, cfg, b, a, c) }.map_err(cuda_error("cuBLAS GEMM"))
    }

    fn product<T: Scalar>(
        &self,
        a: &impl DevicePtr<T>,
        b: &impl DevicePtr<T>,
        c: &mut impl DevicePtrMut<T>,
        shape: MatmulShape,
        accumulate: bool,
        transpose_b: bool,
    ) -> Result<()> {
        let cfg = shape.config::<T>(accumulate, transpose_b)?;
        check_len("matmul a", a.len(), shape.m * shape.k)?;
        check_len("matmul b", b.len(), shape.k * shape.n)?;
        check_len("matmul c", c.len(), shape.m * shape.n)?;
        // Operand order is swapped for the column-major call; see `config`.
        unsafe { T::gemm(&self.blas, cfg, b, a, c) }.map_err(cuda_error("cuBLAS GEMM"))
    }

    /// `batch` independent row-major products laid out back to back, e.g. one
    /// embedding layer per (center type, neighbor type) pair.
    pub fn matmul_batched<T: Scalar>(
        &self,
        a: &CudaSlice<T>,
        b: &CudaSlice<T>,
        c: &mut CudaSlice<T>,
        shape: MatmulShape,
        batch: usize,
    ) -> Result<()> {
        let gemm = shape.config::<T>(false, false)?;
        let batch_size = i32::try_from(batch)
            .ok()
            .filter(|&b| b > 0)
            .ok_or_else(|| Error(format!("batch must be in 1..=i32::MAX, got {batch}")))?;
        let (sa, sb, sc) = (shape.m * shape.k, shape.k * shape.n, shape.m * shape.n);
        check_len("batched matmul a", a.len(), sa * batch)?;
        check_len("batched matmul b", b.len(), sb * batch)?;
        check_len("batched matmul c", c.len(), sc * batch)?;
        let cfg = StridedBatchedConfig {
            gemm,
            batch_size,
            stride_a: sb as i64,
            stride_b: sa as i64,
            stride_c: sc as i64,
        };
        unsafe { T::gemm_strided_batched(&self.blas, cfg, b, a, c) }
            .map_err(cuda_error("cuBLAS strided batched GEMM"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_shape_maps_to_swapped_column_major_call() {
        let cfg = MatmulShape { m: 7, k: 3, n: 5 }
            .config::<f64>(true, false)
            .unwrap();
        assert_eq!((cfg.m, cfg.n, cfg.k), (5, 7, 3));
        assert_eq!((cfg.lda, cfg.ldb, cfg.ldc), (5, 3, 5));
        assert_eq!((cfg.alpha, cfg.beta), (1.0, 1.0));
        let cfg = MatmulShape { m: 7, k: 3, n: 5 }
            .config::<f64>(false, true)
            .unwrap();
        assert_eq!((cfg.lda, cfg.ldb, cfg.ldc), (3, 3, 5));
        assert_eq!(
            MatmulShape { m: 7, k: 3, n: 5 }
                .config::<f32>(false, false)
                .unwrap()
                .beta,
            0.0
        );
        for bad in [(0, 1, 1), (1, 0, 1), (1, 1, i32::MAX as usize + 1)] {
            let shape = MatmulShape {
                m: bad.0,
                k: bad.1,
                n: bad.2,
            };
            assert!(shape.config::<f64>(false, false).is_err(), "{shape:?}");
        }
    }
}
