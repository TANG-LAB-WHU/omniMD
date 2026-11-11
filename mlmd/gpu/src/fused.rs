//! Whole networks in one kernel (plan I-3).
//!
//! Layer-by-layer GEMMs write every intermediate activation to device memory
//! and need a dozen launches per network, which limits large batches by
//! bandwidth and small ones by launch latency. The two kernels here keep the
//! intermediates in shared memory and take one launch for any number of
//! same-shaped networks: each row carries the index of the network it uses.
//!
//! - [`ScalarNetwork`]: one input per row, value and derivative of every
//!   output (the se_e2_a embedding nets, one per type pair).
//! - [`GradientNetwork`]: one output per row, value and gradient with respect
//!   to the inputs (the fitting nets, one per center type).
//!
//! Same layer semantics as [`crate::DenseLayer`].
use crate::dense::validate;
use crate::{check_len, cuda_error, Activation, Device, Scalar};
use cudarc::driver::{
    CudaFunction, CudaSlice, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
};
use omnimd_ml_contract::{Error, Result};

const ACTIVATE: &str = r#"
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
"#;

const SCALAR_SOURCE: &str = r#"
// table: 7 ints per layer {in, out, activation, residual, weights, bias, timestep}
// with the last three as offsets into one network's parameter block; weights
// are [in][out]. Network k's block starts at params + k * net_size.
//
// One warp per row. Each lane owns up to C output neurons at a time
// (n, n+32, ...) and keeps their sums in registers while the loop over the
// inputs broadcasts one activation and its tangent from shared memory, so
// every weight load (coalesced across the lanes) feeds two multiply-adds.
// A row with a negative id is skipped: nothing is written for it.
template <typename T, int C>
__device__ void scalar_network(const T* x, const int* ids, long long rows, const int* table,
        int layers, const T* params, int net_size, int width, T* out, T* dout) {
    extern __shared__ double shared[];
    int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    long long row = (long long)blockIdx.x * warps + warp;
    if (row >= rows || (ids && ids[row] < 0)) return;
    const T* block = params + (ids ? (long long)ids[row] * net_size : 0);
    T* y = (T*)shared + (long long)warp * 4 * width;
    T* dy = y + width;
    T* next = dy + width;
    T* dnext = next + width;
    if (lane == 0) { y[0] = x[row]; dy[0] = T(1); }
    __syncwarp();
    for (int l = 0; l < layers; ++l) {
        const int* e = table + 7 * l;
        int in = e[0], on = e[1], act = e[2], residual = e[3];
        const T* w = block + e[4];
        const T* bias = block + e[5];
        const T* dt = block + e[6];
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
extern "C" __global__ void scalar_network_##S(const T* x, const int* ids, long long rows,     \
        const int* table, int layers, const T* params, int net_size, int width, T* out,       \
        T* dout) {                                                                            \
    scalar_network<T, C>(x, ids, rows, table, layers, params, net_size, width, out, dout);    \
}
INSTANTIATE(float, f32, 4)
INSTANTIATE(double, f64, 1)
"#;

const GRADIENT_SOURCE: &str = r#"
template <typename T>
__device__ T warp_sum(T v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

// sum over k < count of a[k] * w[k * stride], in U independent partial sums.
// With few rows the kernel is bound by the latency of this chain in float32
// (U = 8); float64 is bound by the double-precision units and measures
// fastest with a single sum.
template <typename T, int U>
__device__ T strided_dot(const T* a, const T* w, int count, int stride) {
    T acc[U];
    for (int u = 0; u < U; ++u) acc[u] = T(0);
    int k = 0;
    for (; k + U <= count; k += U, w += U * (long long)stride)
        for (int u = 0; u < U; ++u) acc[u] += a[k + u] * w[u * (long long)stride];
    for (; k < count; ++k, w += stride) acc[0] += a[k] * *w;
    T total = acc[0];
    for (int u = 1; u < U; ++u) total += acc[u];
    return total;
}

// table: 10 ints per layer {in, out, activation, residual, weights, transposed
// weights, bias, timestep, y, slope}: the middle four are offsets into one
// network's parameter block (weights [in][out], transposed [out][in]), the
// last two offsets into shared memory of that layer's outputs and slopes
// (hidden layers only). Shared memory: the input row, every hidden layer's
// outputs and slopes, two adjoint buffers of the widest hidden layer, one
// partial sum per warp.
//
// One block per row. Forward: thread n owns output neuron n, the loop over the
// inputs broadcasts one activation from shared memory and reads one weight
// row coalesced across the block. Backward: thread k owns input k and reads
// the transposed weights the same way, so the last pass writes the gradient
// row coalesced.
template <typename T, int U>
__device__ void gradient_network(const T* x, const int* ids, long long rows, const int* table,
        int layers, const T* params, int net_size, int adjoint, T* value, T* gradient) {
    extern __shared__ double shared[];
    long long row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x, threads = blockDim.x;
    const T* block = params + (ids ? (long long)ids[row] * net_size : 0);
    T* sh = (T*)shared;
    int in0 = table[0];
    for (int k = tid; k < in0; k += threads) sh[k] = x[row * in0 + k];
    __syncthreads();
    const T* current = sh;
    for (int l = 0; l + 1 < layers; ++l) {
        const int* e = table + 10 * l;
        int in = e[0], on = e[1], act = e[2], residual = e[3];
        const T* w = block + e[4];
        const T* bias = block + e[6];
        const T* dt = block + e[7];
        T* y = sh + e[8];
        T* slope = sh + e[9];
        for (int n = tid; n < on; n += threads) {
            T z = bias[n] + strided_dot<T, U>(current, w + n, in, on);
            T v, s;
            activate(act, z, &v, &s);
            v *= dt[n];
            if (residual) v += current[n % in];
            y[n] = v;
            slope[n] = s * dt[n];
        }
        __syncthreads();
        current = y;
    }
    // Output layer: one value per row, reduced over the block.
    const int* last = table + 10 * (layers - 1);
    int width = last[0];
    const T* w_out = block + last[4];
    T* a = sh + adjoint;
    T* b = a + table[10 * layers];
    T* partial = b + table[10 * layers];
    T* seed = partial + (threads >> 5);
    T sum = T(0);
    for (int k = tid; k < width; k += threads) sum += current[k] * w_out[k];
    sum = warp_sum(sum);
    if ((tid & 31) == 0) partial[tid >> 5] = sum;
    __syncthreads();
    if (tid == 0) {
        T z = (block + last[6])[0];
        for (int wi = 0; wi < (threads >> 5); ++wi) z += partial[wi];
        T v, s;
        activate(last[2], z, &v, &s);
        T scale = (block + last[7])[0];
        value[row] = v * scale;
        *seed = s * scale;
    }
    __syncthreads();
    T g = *seed;
    if (layers == 1) {
        for (int k = tid; k < width; k += threads) gradient[row * in0 + k] = g * w_out[k];
        return;
    }
    for (int k = tid; k < width; k += threads) a[k] = g * w_out[k];
    __syncthreads();
    for (int l = layers - 2; l >= 0; --l) {
        const int* e = table + 10 * l;
        int in = e[0], on = e[1], residual = e[3];
        const T* wt = block + e[5];
        T* slope = sh + e[9];
        // The slopes are not needed again: turn them into the adjoint of the
        // layer's pre-activations.
        for (int n = tid; n < on; n += threads) slope[n] *= a[n];
        __syncthreads();
        T* target = l == 0 ? gradient + row * in0 : b;
        for (int k = tid; k < in; k += threads) {
            T acc = strided_dot<T, U>((const T*)slope, wt + k, on, in);
            if (residual)
                for (int n = k; n < on; n += in) acc += a[n];
            target[k] = acc;
        }
        __syncthreads();
        T* t = a; a = b; b = t;
    }
}

#define INSTANTIATE(T, S, U)                                                                  \
extern "C" __global__ void gradient_network_##S(const T* x, const int* ids, long long rows,   \
        const int* table, int layers, const T* params, int net_size, int adjoint, T* value,   \
        T* gradient) {                                                                        \
    gradient_network<T, U>(x, ids, rows, table, layers, params, net_size, adjoint, value,     \
                           gradient);                                                         \
}
INSTANTIATE(float, f32, 8)
INSTANTIATE(double, f64, 1)
"#;

/// Shared memory one block may use without opting in to more.
const SHARED_LIMIT: usize = 48 * 1024;
const MAX_WARPS: usize = 8;
/// Threads per row of [`GradientNetwork`].
const GRADIENT_BLOCK: usize = 256;

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

/// Checks that `networks` is a nonempty set of chains with identical layer
/// shapes, activations and residual flags, starting from `input` inputs when
/// given. Returns the shared layer list.
fn common_shape<T: Scalar>(
    networks: &[Vec<LayerParameters<T>>],
    input: Option<usize>,
) -> Result<&[LayerParameters<T>]> {
    let Some(first) = networks.first().filter(|n| !n.is_empty()) else {
        return Err(Error(
            "a fused network set needs at least one network with one layer".into(),
        ));
    };
    if input.is_some_and(|expected| first[0].input != expected) {
        return Err(Error(format!(
            "network takes {} inputs per row, expected {}",
            first[0].input,
            input.unwrap()
        )));
    }
    let mut previous = first[0].input;
    for layer in first {
        if layer.input != previous {
            return Err(Error(format!(
                "layer input {} does not match previous output {previous}",
                layer.input
            )));
        }
        previous = layer.output;
    }
    for network in networks {
        if network.len() != first.len() {
            return Err(Error("networks of a set must have the same depth".into()));
        }
        for (layer, reference) in network.iter().zip(first) {
            validate(
                layer.input,
                layer.output,
                layer.weights.len(),
                layer.bias.len(),
                layer.timestep.as_ref().map(Vec::len),
                layer.residual,
            )?;
            if (layer.input, layer.output, layer.activation, layer.residual)
                != (
                    reference.input,
                    reference.output,
                    reference.activation,
                    reference.residual,
                )
            {
                return Err(Error(
                    "networks of a set must have identical layer shapes, activations and residuals"
                        .into(),
                ));
            }
        }
    }
    Ok(first)
}

fn timestep_or_ones<T: Scalar>(layer: &LayerParameters<T>) -> Vec<T> {
    layer
        .timestep
        .clone()
        .unwrap_or_else(|| vec![T::ONE; layer.output])
}

fn check_ids(ids: Option<usize>, rows: usize, networks: usize) -> Result<()> {
    match ids {
        Some(len) => check_len("network ids", len, rows),
        None if networks == 1 => Ok(()),
        None => Err(Error(format!(
            "a set of {networks} networks needs one network id per row"
        ))),
    }
}

/// Same-shaped networks with one input per row, evaluated with their
/// derivative.
pub struct ScalarNetwork<T: Scalar> {
    networks: usize,
    output: usize,
    layers: i32,
    net_size: i32,
    width: usize,
    warps: usize,
    table: CudaSlice<i32>,
    params: CudaSlice<T>,
    function: CudaFunction,
}

impl<T: Scalar> ScalarNetwork<T> {
    /// A single network.
    pub fn new(device: &Device, layers: &[LayerParameters<T>]) -> Result<Self> {
        Self::new_set(device, &[layers.to_vec()])
    }

    /// Networks with identical layer shapes; rows select one by index.
    pub fn new_set(device: &Device, networks: &[Vec<LayerParameters<T>>]) -> Result<Self> {
        let shape = common_shape(networks, Some(1))?;
        let mut table = Vec::with_capacity(7 * shape.len());
        let mut offset = 0usize;
        let mut width = 1;
        for layer in shape {
            let weights = offset;
            let bias = weights + layer.input * layer.output;
            let timestep = bias + layer.output;
            offset = timestep + layer.output;
            table.extend([
                layer.input as i32,
                layer.output as i32,
                layer.activation.code(),
                i32::from(layer.residual),
                weights as i32,
                bias as i32,
                timestep as i32,
            ]);
            width = width.max(layer.output);
        }
        if i32::try_from(offset * networks.len()).is_err() {
            return Err(Error("scalar network set has too many parameters".into()));
        }
        let mut params: Vec<T> = Vec::with_capacity(offset * networks.len());
        for network in networks {
            for layer in network {
                params.extend_from_slice(&layer.weights);
                params.extend_from_slice(&layer.bias);
                params.extend(timestep_or_ones(layer));
            }
        }
        let warp_bytes = 4 * width * std::mem::size_of::<T>();
        if warp_bytes > SHARED_LIMIT {
            return Err(Error(format!(
                "scalar network width {width} needs {warp_bytes} bytes of shared memory per row; the limit is {SHARED_LIMIT}"
            )));
        }
        Ok(Self {
            networks: networks.len(),
            output: shape.last().unwrap().output,
            layers: shape.len() as i32,
            net_size: offset as i32,
            width,
            warps: (SHARED_LIMIT / warp_bytes).clamp(1, MAX_WARPS),
            table: device.upload(&table)?,
            params: device.upload(&params)?,
            function: device.kernels().function(
                &format!("{ACTIVATE}{SCALAR_SOURCE}"),
                &format!("scalar_network_{}", T::SUFFIX),
            )?,
        })
    }

    pub fn output(&self) -> usize {
        self.output
    }

    pub fn networks(&self) -> usize {
        self.networks
    }

    /// `out[rows][output]` and its derivative with respect to `x[rows]`, for
    /// a single network.
    pub fn forward_tangent_into(
        &self,
        device: &Device,
        x: &impl DevicePtr<T>,
        rows: usize,
        out: &mut impl DevicePtrMut<T>,
        dout: &mut impl DevicePtrMut<T>,
    ) -> Result<()> {
        self.launch(device, x, None::<&CudaSlice<i32>>, rows, out, dout)
    }

    /// Like [`Self::forward_tangent_into`], with row `r` evaluated by network
    /// `ids[r]` (which must be below [`Self::networks`]). A row with a
    /// negative id is skipped: its outputs are left as they were.
    pub fn forward_tangent_by_id(
        &self,
        device: &Device,
        x: &impl DevicePtr<T>,
        ids: &impl DevicePtr<i32>,
        rows: usize,
        out: &mut impl DevicePtrMut<T>,
        dout: &mut impl DevicePtrMut<T>,
    ) -> Result<()> {
        self.launch(device, x, Some(ids), rows, out, dout)
    }

    fn launch(
        &self,
        device: &Device,
        x: &impl DevicePtr<T>,
        ids: Option<&impl DevicePtr<i32>>,
        rows: usize,
        out: &mut impl DevicePtrMut<T>,
        dout: &mut impl DevicePtrMut<T>,
    ) -> Result<()> {
        check_len("scalar network input", x.len(), rows)?;
        check_len("scalar network output", out.len(), rows * self.output)?;
        check_len("scalar network tangent", dout.len(), rows * self.output)?;
        check_ids(ids.map(|i| i.len()), rows, self.networks)?;
        let blocks = rows.div_ceil(self.warps);
        if rows == 0 || blocks > u32::MAX as usize {
            return Err(Error(format!(
                "scalar network over {rows} rows is out of range"
            )));
        }
        let stream = device.stream();
        let (px, _g0) = x.device_ptr(stream);
        let ids = ids.map(|i| i.device_ptr(stream));
        let pids = ids.as_ref().map_or(0, |(pointer, _)| *pointer);
        let (pout, _g1) = out.device_ptr_mut(stream);
        let (pdout, _g2) = dout.device_ptr_mut(stream);
        let (rows_i, width_i) = (rows as i64, self.width as i32);
        let mut launch = stream.launch_builder(&self.function);
        launch
            .arg(&px)
            .arg(&pids)
            .arg(&rows_i)
            .arg(&self.table)
            .arg(&self.layers)
            .arg(&self.params)
            .arg(&self.net_size)
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

/// Same-shaped networks with one output per row, evaluated with the gradient
/// of that output with respect to the row's inputs.
pub struct GradientNetwork<T: Scalar> {
    networks: usize,
    input: usize,
    layers: i32,
    net_size: i32,
    adjoint: i32,
    shared_bytes: usize,
    table: CudaSlice<i32>,
    params: CudaSlice<T>,
    function: CudaFunction,
}

impl<T: Scalar> GradientNetwork<T> {
    /// Networks with identical layer shapes ending in one output; rows select
    /// one by index. Fails when a row's intermediates do not fit in shared
    /// memory (then the layer-by-layer path is the one to use).
    pub fn new_set(device: &Device, networks: &[Vec<LayerParameters<T>>]) -> Result<Self> {
        let shape = common_shape(networks, None)?;
        if shape.last().unwrap().output != 1 {
            return Err(Error(format!(
                "a gradient network ends in one output, got {}",
                shape.last().unwrap().output
            )));
        }
        let input = shape[0].input;
        let hidden = &shape[..shape.len() - 1];
        let widest = hidden.iter().map(|l| l.output).max().unwrap_or(0);
        // Shared memory in elements: input row, outputs and slopes of each
        // hidden layer, two adjoint buffers, per-warp partial sums, the seed.
        let mut shared = input;
        let mut offset = 0usize;
        let mut table = Vec::with_capacity(10 * shape.len() + 1);
        for (index, layer) in shape.iter().enumerate() {
            let weights = offset;
            let transposed = weights + layer.input * layer.output;
            let bias = transposed + layer.input * layer.output;
            let timestep = bias + layer.output;
            offset = timestep + layer.output;
            let (y, slope) = if index < hidden.len() {
                let y = shared;
                shared += 2 * layer.output;
                (y, y + layer.output)
            } else {
                (0, 0)
            };
            table.extend([
                layer.input as i32,
                layer.output as i32,
                layer.activation.code(),
                i32::from(layer.residual),
                weights as i32,
                transposed as i32,
                bias as i32,
                timestep as i32,
                y as i32,
                slope as i32,
            ]);
        }
        table.push(widest as i32);
        let adjoint = shared;
        shared += 2 * widest + GRADIENT_BLOCK / 32 + 1;
        let shared_bytes = (shared * std::mem::size_of::<T>()).next_multiple_of(8);
        if shared_bytes > SHARED_LIMIT {
            return Err(Error(format!(
                "gradient network needs {shared_bytes} bytes of shared memory per row; the limit is {SHARED_LIMIT}"
            )));
        }
        if i32::try_from(offset * networks.len()).is_err() {
            return Err(Error("gradient network set has too many parameters".into()));
        }
        let mut params: Vec<T> = Vec::with_capacity(offset * networks.len());
        for network in networks {
            for layer in network {
                params.extend_from_slice(&layer.weights);
                for n in 0..layer.output {
                    params.extend((0..layer.input).map(|k| layer.weights[k * layer.output + n]));
                }
                params.extend_from_slice(&layer.bias);
                params.extend(timestep_or_ones(layer));
            }
        }
        Ok(Self {
            networks: networks.len(),
            input,
            layers: shape.len() as i32,
            net_size: offset as i32,
            adjoint: adjoint as i32,
            shared_bytes,
            table: device.upload(&table)?,
            params: device.upload(&params)?,
            function: device.kernels().function(
                &format!("{ACTIVATE}{GRADIENT_SOURCE}"),
                &format!("gradient_network_{}", T::SUFFIX),
            )?,
        })
    }

    pub fn input(&self) -> usize {
        self.input
    }

    pub fn networks(&self) -> usize {
        self.networks
    }

    /// `value[rows]` and `gradient[rows][input]` for `x[rows][input]`; row `r`
    /// uses network `ids[r]` (`None` only for a single network).
    pub fn forward_gradient_into(
        &self,
        device: &Device,
        x: &impl DevicePtr<T>,
        ids: Option<&impl DevicePtr<i32>>,
        rows: usize,
        value: &mut impl DevicePtrMut<T>,
        gradient: &mut impl DevicePtrMut<T>,
    ) -> Result<()> {
        check_len("gradient network input", x.len(), rows * self.input)?;
        check_len("gradient network value", value.len(), rows)?;
        check_len(
            "gradient network gradient",
            gradient.len(),
            rows * self.input,
        )?;
        check_ids(ids.map(|i| i.len()), rows, self.networks)?;
        if rows == 0 || rows > u32::MAX as usize {
            return Err(Error(format!(
                "gradient network over {rows} rows is out of range"
            )));
        }
        let stream = device.stream();
        let (px, _g0) = x.device_ptr(stream);
        let ids = ids.map(|i| i.device_ptr(stream));
        let pids = ids.as_ref().map_or(0, |(pointer, _)| *pointer);
        let (pvalue, _g1) = value.device_ptr_mut(stream);
        let (pgradient, _g2) = gradient.device_ptr_mut(stream);
        let rows_i = rows as i64;
        let mut launch = stream.launch_builder(&self.function);
        launch
            .arg(&px)
            .arg(&pids)
            .arg(&rows_i)
            .arg(&self.table)
            .arg(&self.layers)
            .arg(&self.params)
            .arg(&self.net_size)
            .arg(&self.adjoint)
            .arg(&pvalue)
            .arg(&pgradient);
        let config = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (GRADIENT_BLOCK as u32, 1, 1),
            shared_mem_bytes: self.shared_bytes as u32,
        };
        unsafe { launch.launch(config) }
            .map(drop)
            .map_err(cuda_error("gradient network launch"))
    }
}
