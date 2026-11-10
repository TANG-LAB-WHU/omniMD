//! A whole scalar-input network in one kernel (plan I-3, embedding nets).
//!
//! The se_e2_a embedding net maps one number per neighbor, `s(r)`, to a row of
//! `G` and needs `dG/ds` for the forces. Layer-by-layer GEMMs write every
//! intermediate activation and tangent to device memory and read it back,
//! which is what limits that path. Here one warp carries one row through all
//! layers: activations and tangents live in shared memory, weights are read
//! coalesced across the lanes (lane = output neuron), and only the last
//! layer touches global memory. Same layer semantics as [`crate::DenseLayer`].
use crate::dense::validate;
use crate::{check_len, cuda_error, Activation, Device, Scalar};
use cudarc::driver::{
    CudaFunction, CudaSlice, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
};
use omnimd_ml_contract::{Error, Result};

const SOURCE: &str = r#"
template <typename T>
__device__ void activate(int act, T z, T* v, T* s) {
    if (act == 1) {
        *v = tanh(z);
        *s = T(1) - *v * *v;
    } else if (act == 2 || act == 3) {
        T e = exp(z >= T(0) ? -z : z);
        T sigmoid = z >= T(0) ? T(1) / (T(1) + e) : e / (T(1) + e);
        if (act == 2) {
            *v = sigmoid;
            *s = sigmoid * (T(1) - sigmoid);
        } else {
            *v = (z > T(0) ? z : T(0)) + log1p(e);
            *s = sigmoid;
        }
    } else {
        *v = z;
        *s = T(1);
    }
}

// table: 7 ints per layer {in, out, activation, residual, weights, bias, timestep}
// with the last three as offsets into params; weights are [in][out].
//
// One warp per row. Each lane owns up to C output neurons at a time
// (n, n+32, ...) and keeps their sums in registers while the loop over the
// inputs broadcasts one activation and its tangent from shared memory, so
// every weight load (coalesced across the lanes) feeds two multiply-adds.
template <typename T, int C>
__device__ void scalar_network(const T* x, long long rows, const int* table, int layers,
        const T* params, int width, T* out, T* dout) {
    extern __shared__ double shared[];
    int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    long long row = (long long)blockIdx.x * warps + warp;
    if (row >= rows) return;
    T* y = (T*)shared + (long long)warp * 4 * width;
    T* dy = y + width;
    T* next = dy + width;
    T* dnext = next + width;
    if (lane == 0) { y[0] = x[row]; dy[0] = T(1); }
    __syncwarp();
    for (int l = 0; l < layers; ++l) {
        const int* e = table + 7 * l;
        int in = e[0], on = e[1], act = e[2], residual = e[3];
        const T* w = params + e[4];
        const T* bias = params + e[5];
        const T* dt = params + e[6];
        bool last = l + 1 == layers;
        for (int n0 = lane; n0 < on; n0 += 32 * C) {
            T z[C], d[C];
            for (int c = 0; c < C; ++c) { z[c] = T(0); d[c] = T(0); }
            const T* wk = w;
            for (int k = 0; k < in; ++k, wk += on) {
                T yk = y[k], dk = dy[k];
                for (int c = 0; c < C; ++c) {
                    int n = n0 + 32 * c;
                    if (n < on) { T v = wk[n]; z[c] += yk * v; d[c] += dk * v; }
                }
            }
            for (int c = 0; c < C; ++c) {
                int n = n0 + 32 * c;
                if (n >= on) break;
                T v, s;
                activate(act, z[c] + bias[n], &v, &s);
                v *= dt[n];
                T dv = s * dt[n] * d[c];
                if (residual) { v += y[n % in]; dv += dy[n % in]; }
                if (last) {
                    out[row * on + n] = v;
                    dout[row * on + n] = dv;
                } else {
                    next[n] = v;
                    dnext[n] = dv;
                }
            }
        }
        __syncwarp();
        T* t = y; y = next; next = t;
        t = dy; dy = dnext; dnext = t;
    }
}

// Float32 is limited by loads, so four neurons per lane; float64 is limited
// by the double-precision units and measures fastest with one.
#define INSTANTIATE(T, S, C)                                                                  \
extern "C" __global__ void scalar_network_##S(const T* x, long long rows, const int* table,   \
        int layers, const T* params, int width, T* out, T* dout) {                            \
    scalar_network<T, C>(x, rows, table, layers, params, width, out, dout);                   \
}
INSTANTIATE(float, f32, 4)
INSTANTIATE(double, f64, 1)
"#;

/// Shared memory one block may use without opting in to more.
const SHARED_LIMIT: usize = 48 * 1024;
const MAX_WARPS: usize = 8;

/// Host-side parameters of one dense layer; `weights` is row-major
/// `[input][output]`.
#[derive(Clone, Debug)]
pub struct LayerParameters<T> {
    pub input: usize,
    pub output: usize,
    pub weights: Vec<T>,
    pub bias: Vec<T>,
    pub timestep: Option<Vec<T>>,
    pub activation: Activation,
    pub residual: bool,
}

/// A network with one input per row, evaluated with its derivative.
pub struct ScalarNetwork<T: Scalar> {
    output: usize,
    layers: i32,
    width: usize,
    warps: usize,
    table: CudaSlice<i32>,
    params: CudaSlice<T>,
    function: CudaFunction,
}

impl<T: Scalar> ScalarNetwork<T> {
    pub fn new(device: &Device, layers: &[LayerParameters<T>]) -> Result<Self> {
        let Some(first) = layers.first() else {
            return Err(Error("a scalar network needs at least one layer".into()));
        };
        if first.input != 1 {
            return Err(Error(format!(
                "a scalar network takes one input per row, got {}",
                first.input
            )));
        }
        let mut table = Vec::with_capacity(7 * layers.len());
        let mut params: Vec<T> = Vec::new();
        let mut input = 1;
        let mut width = 1;
        for layer in layers {
            validate(
                layer.input,
                layer.output,
                layer.weights.len(),
                layer.bias.len(),
                layer.timestep.as_ref().map(Vec::len),
                layer.residual,
            )?;
            if layer.input != input {
                return Err(Error(format!(
                    "layer input {} does not match previous output {input}",
                    layer.input
                )));
            }
            let weights = params.len();
            params.extend_from_slice(&layer.weights);
            let bias = params.len();
            params.extend_from_slice(&layer.bias);
            let timestep = params.len();
            match &layer.timestep {
                Some(t) => params.extend_from_slice(t),
                None => params.extend(std::iter::repeat_n(T::ONE, layer.output)),
            }
            if i32::try_from(params.len()).is_err() {
                return Err(Error("scalar network has too many parameters".into()));
            }
            table.extend([
                layer.input as i32,
                layer.output as i32,
                layer.activation.code(),
                i32::from(layer.residual),
                weights as i32,
                bias as i32,
                timestep as i32,
            ]);
            input = layer.output;
            width = width.max(layer.output);
        }
        let warp_bytes = 4 * width * std::mem::size_of::<T>();
        if warp_bytes > SHARED_LIMIT {
            return Err(Error(format!(
                "scalar network width {width} needs {warp_bytes} bytes of shared memory per row; the limit is {SHARED_LIMIT}"
            )));
        }
        Ok(Self {
            output: input,
            layers: layers.len() as i32,
            width,
            warps: (SHARED_LIMIT / warp_bytes).clamp(1, MAX_WARPS),
            table: device.upload(&table)?,
            params: device.upload(&params)?,
            function: device
                .kernels()
                .function(SOURCE, &format!("scalar_network_{}", T::SUFFIX))?,
        })
    }

    pub fn output(&self) -> usize {
        self.output
    }

    /// `out[rows][output]` and its derivative with respect to `x[rows]`.
    pub fn forward_tangent_into(
        &self,
        device: &Device,
        x: &impl DevicePtr<T>,
        rows: usize,
        out: &mut impl DevicePtrMut<T>,
        dout: &mut impl DevicePtrMut<T>,
    ) -> Result<()> {
        check_len("scalar network input", x.len(), rows)?;
        check_len("scalar network output", out.len(), rows * self.output)?;
        check_len("scalar network tangent", dout.len(), rows * self.output)?;
        let blocks = rows.div_ceil(self.warps);
        if rows == 0 || blocks > u32::MAX as usize {
            return Err(Error(format!(
                "scalar network over {rows} rows is out of range"
            )));
        }
        let stream = device.stream();
        let (px, _g0) = x.device_ptr(stream);
        let (pout, _g1) = out.device_ptr_mut(stream);
        let (pdout, _g2) = dout.device_ptr_mut(stream);
        let (rows_i, width_i) = (rows as i64, self.width as i32);
        let mut launch = stream.launch_builder(&self.function);
        launch
            .arg(&px)
            .arg(&rows_i)
            .arg(&self.table)
            .arg(&self.layers)
            .arg(&self.params)
            .arg(&width_i)
            .arg(&pout)
            .arg(&pdout);
        let config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (32 * self.warps as u32, 1, 1),
            shared_mem_bytes: (self.warps * 4 * self.width * std::mem::size_of::<T>()) as u32,
        };
        unsafe { launch.launch(config) }
            .map(drop)
            .map_err(cuda_error("scalar network launch"))
    }
}
