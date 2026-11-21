//! Fitting nets of a set, layer by layer, every net in each launch.
//!
//! [`crate::DenseNetwork`] runs one net at a time: for the fitting nets of a
//! model that is about twenty launches per center type (products, bias and
//! activation, slopes, residuals, a copy of the energies), and a committee of
//! models multiplies them. Each costs a few microseconds whatever its size,
//! so with thousands of rows the launches cost more than the arithmetic.
//! [`GroupedNetwork`] gives each layer one grouped cuBLAS product for all
//! nets (`cublas<t>gemmGroupedBatched`: a row group per net, of its own
//! length) and one launch for what follows the product, and folds the
//! output layer, which has one output, into the last hidden layer's launch:
//! with three hidden layers, six products and five other launches for any
//! number of nets.
//!
//! Same layer semantics and the same groups as
//! [`crate::GradientNetwork::forward_gradient_grouped`]; the results agree
//! to rounding (the products and the sum of the output layer add in other
//! orders).
//!
//! A grouped product takes a device array of operand addresses per net. The
//! work buffers stay with the set between calls, so the addresses change
//! only with the input, the output or a larger batch, and the arrays are
//! copied to the device only then.
use crate::fused::{common_shape, ACTIVATE};
use crate::{check_len, cuda_error, Activation, Device, LayerParameters, Scalar};
use cudarc::cublas::sys::{cublasOperation_t, cublasStatus_t};
use cudarc::driver::{
    CudaFunction, CudaSlice, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
};
use omnimd_ml_contract::{Error, Result};

/// The first cuBLAS with grouped products (CUDA 12.5).
pub(crate) const GROUPED_CUBLAS: i32 = 120_500;

/// The most row groups one call of [`GroupedNetwork`] takes: its kernels
/// put the groups in the second dimension of the grid.
pub const GROUPED_MAX_GROUPS: usize = 65_535;

const SOURCE: &str = r#"
#define FULL_MASK 0xffffffffu

// Per group g: groups[3g] its first row, groups[3g+1] its rows,
// groups[3g+2] its net. blockIdx.y is the group.

// y holds x·W of a hidden layer on entry, its output on return.
template <typename T>
__device__ void grouped_epilogue(T* y, T* slope, const T* x, const long long* groups,
                                 const T* bias, const T* dt, int in, int out, int act,
                                 int residual) {
    const long long* group = groups + 3 * blockIdx.y;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= group[1] * out) return;
    long long net = group[2];
    int j = (int)(idx % out);
    long long r = group[0] + idx / out;
    long long at = r * out + j;
    T v, s;
    activate(act, y[at] + bias[net * out + j], &v, &s);
    T scale = dt ? dt[net * out + j] : T(1);
    v *= scale;
    s *= scale;
    if (residual) v += x[r * in + j % in];
    y[at] = v;
    slope[at] = s;
}

// The last hidden layer and the output layer (one output, linear), a warp a
// row. z holds x·W on entry and the adjoint of the layer's activation input
// on return, w_out ⊙ slope; g, when given, receives w_out, the adjoint of
// the layer's output, for its residual.
template <typename T>
__device__ void grouped_head(T* z, T* g, const T* x, const long long* groups, const T* bias,
                             const T* dt, const T* w_out, const T* b_out, T* energy, int in,
                             int width, int act, int residual) {
    const long long* group = groups + 3 * blockIdx.y;
    int lane = threadIdx.x & 31;
    long long local = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (local >= group[1]) return;
    long long net = group[2];
    long long r = group[0] + local;
    const T* w = w_out + net * width;
    T sum = T(0);
    for (int j = lane; j < width; j += 32) {
        long long at = r * width + j;
        T v, s;
        activate(act, z[at] + bias[net * width + j], &v, &s);
        T scale = dt ? dt[net * width + j] : T(1);
        v *= scale;
        s *= scale;
        if (residual) v += x[r * in + j % in];
        sum += v * w[j];
        z[at] = w[j] * s;
        if (g) g[at] = w[j];
    }
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_xor_sync(FULL_MASK, sum, offset);
    if (lane == 0) energy[r] = sum + b_out[net];
}

template <typename T>
__device__ void grouped_scale(T* h, const T* g, const T* slope, long long n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) h[idx] = g[idx] * slope[idx];
}

#define INSTANTIATE(T, S)                                                                        \
extern "C" __global__ void grouped_epilogue_##S(T* y, T* slope, const T* x,                      \
        const long long* groups, const T* bias, const T* dt, int in, int out, int act,           \
        int residual) {                                                                          \
    grouped_epilogue<T>(y, slope, x, groups, bias, dt, in, out, act, residual);                  \
}                                                                                                \
extern "C" __global__ void grouped_head_##S(T* z, T* g, const T* x, const long long* groups,     \
        const T* bias, const T* dt, const T* w_out, const T* b_out, T* energy, int in,           \
        int width, int act, int residual) {                                                      \
    grouped_head<T>(z, g, x, groups, bias, dt, w_out, b_out, energy, in, width, act, residual);  \
}                                                                                                \
extern "C" __global__ void grouped_scale_##S(T* h, const T* g, const T* slope, long long n) {    \
    grouped_scale<T>(h, g, slope, n);                                                            \
}
INSTANTIATE(float, f32)
INSTANTIATE(double, f64)
"#;

const BLOCK: u32 = 256;
const HEAD_ROWS: u32 = BLOCK / 32;

/// One hidden layer of every net: weights `[net][input][output]`, biases and
/// timesteps `[net][output]`.
struct Layer<T: Scalar> {
    input: usize,
    output: usize,
    activation: Activation,
    residual: bool,
    weights: CudaSlice<T>,
    bias: CudaSlice<T>,
    timestep: Option<CudaSlice<T>>,
}

/// Buffers of one call, kept for the next: `rows` rows of each.
struct Work<T: Scalar> {
    rows: usize,
    /// Outputs and slopes of the hidden layers but the last.
    outputs: Vec<CudaSlice<T>>,
    slopes: Vec<CudaSlice<T>>,
    /// The last hidden layer's product, then its adjoint.
    last: CudaSlice<T>,
    /// The adjoint of a hidden layer's output: one buffer serves all
    /// layers, as the products read `last` and `scaled` only.
    adjoint: CudaSlice<T>,
    /// The adjoint of a hidden layer's activation input.
    scaled: CudaSlice<T>,
}

pub struct GroupedNetwork<T: Scalar> {
    networks: usize,
    input: usize,
    hidden: Vec<Layer<T>>,
    /// Output layer: weights `[net][width]`, bias `[net]`.
    out_weights: CudaSlice<T>,
    out_bias: CudaSlice<T>,
    epilogue: CudaFunction,
    head: CudaFunction,
    scale: CudaFunction,
    work: Option<Work<T>>,
    /// Operand addresses of the products and the group table, as last
    /// copied to `arrays`.
    addresses: Vec<u64>,
    arrays: Option<CudaSlice<u64>>,
}

/// One grouped product `c = op(a) · b (+ c)` per group, column-major as
/// cuBLAS sees row-major operands: `m` outputs per row, `k` inputs.
struct Product {
    transpose_a: bool,
    m: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    accumulate: bool,
}

impl<T: Scalar> GroupedNetwork<T> {
    /// The nets of one shape, each ending in a single linear output without
    /// timestep or residual, with residual hidden layers that keep their
    /// width and a first layer without one. Fails for any other shape and
    /// when the cuBLAS library has no grouped products (before 12.5).
    pub fn new_set(device: &Device, networks: &[Vec<LayerParameters<T>>]) -> Result<Self> {
        let shape = common_shape(networks, None)?;
        let (last, hidden) = shape.split_last().unwrap();
        if hidden.is_empty()
            || last.output != 1
            || last.activation != Activation::Linear
            || last.residual
            || networks
                .iter()
                .any(|n| n.last().unwrap().timestep.is_some())
        {
            return Err(Error(
                "grouped nets need hidden layers and one linear output without timestep or residual"
                    .into(),
            ));
        }
        if hidden[0].residual || hidden.iter().any(|l| l.residual && l.input != l.output) {
            return Err(Error(
                "grouped nets take residual hidden layers of unchanged width after the first"
                    .into(),
            ));
        }
        let version = device.blas_version()?;
        if version < GROUPED_CUBLAS {
            return Err(Error(format!(
                "cuBLAS {version} has no grouped products (from {GROUPED_CUBLAS})"
            )));
        }
        let gather = |f: &dyn Fn(&LayerParameters<T>) -> &[T], at: usize| -> Vec<T> {
            networks.iter().flat_map(|n| f(&n[at]).to_vec()).collect()
        };
        let mut layers = Vec::with_capacity(hidden.len());
        for (at, layer) in hidden.iter().enumerate() {
            let timesteps = networks.iter().any(|n| n[at].timestep.is_some());
            let timestep = match timesteps {
                true => Some(
                    device.upload(
                        &networks
                            .iter()
                            .flat_map(|n| {
                                n[at]
                                    .timestep
                                    .clone()
                                    .unwrap_or_else(|| vec![T::ONE; layer.output])
                            })
                            .collect::<Vec<_>>(),
                    )?,
                ),
                false => None,
            };
            layers.push(Layer {
                input: layer.input,
                output: layer.output,
                activation: layer.activation,
                residual: layer.residual,
                weights: device.upload(&gather(&|l| &l.weights, at))?,
                bias: device.upload(&gather(&|l| &l.bias, at))?,
                timestep,
            });
        }
        let at = hidden.len();
        let source = format!("{ACTIVATE}{SOURCE}");
        let function = |name: &str| {
            device
                .kernels()
                .function(&source, &format!("{name}_{}", T::SUFFIX))
        };
        Ok(Self {
            networks: networks.len(),
            input: hidden[0].input,
            hidden: layers,
            out_weights: device.upload(&gather(&|l| &l.weights, at))?,
            out_bias: device.upload(&gather(&|l| &l.bias, at))?,
            epilogue: function("grouped_epilogue")?,
            head: function("grouped_head")?,
            scale: function("grouped_scale")?,
            work: None,
            addresses: Vec::new(),
            arrays: None,
        })
    }

    pub fn input(&self) -> usize {
        self.input
    }

    fn work(&mut self, device: &Device, rows: usize) -> Result<()> {
        if self.work.as_ref().is_some_and(|w| w.rows >= rows) {
            return Ok(());
        }
        let (last, rest) = self.hidden.split_last().unwrap();
        let widest = self
            .hidden
            .iter()
            .map(|l| l.input.max(l.output))
            .max()
            .unwrap();
        // SAFETY: every element a call reads is written earlier in that call.
        let buffer = |width: usize| unsafe { device.alloc_uninit::<T>(rows * width) };
        self.work = Some(Work {
            rows,
            outputs: rest
                .iter()
                .map(|l| buffer(l.output))
                .collect::<Result<_>>()?,
            slopes: rest
                .iter()
                .map(|l| buffer(l.output))
                .collect::<Result<_>>()?,
            last: buffer(last.output)?,
            adjoint: buffer(widest)?,
            scaled: buffer(widest)?,
        });
        Ok(())
    }

    /// Value and input gradient of every row, the rows in `groups` of
    /// `(net, first row, rows)`, consecutive and nonempty: as
    /// [`crate::GradientNetwork::forward_gradient_grouped`].
    pub fn forward_gradient_grouped(
        &mut self,
        device: &Device,
        x: &impl DevicePtr<T>,
        groups: &[(usize, usize, usize)],
        rows: usize,
        value: &mut impl DevicePtrMut<T>,
        gradient: &mut impl DevicePtrMut<T>,
    ) -> Result<()> {
        if groups.len() > GROUPED_MAX_GROUPS {
            return Err(Error(format!(
                "{} row groups, more than the {GROUPED_MAX_GROUPS} a grouped call takes",
                groups.len()
            )));
        }
        let mut next = 0;
        for &(network, first, count) in groups {
            if network >= self.networks || first != next || count == 0 {
                return Err(Error(format!(
                    "row groups must be consecutive nonempty runs of networks below {}; got ({network}, {first}, {count}) at row {next}",
                    self.networks
                )));
            }
            next += count;
        }
        if next != rows || rows == 0 || rows > i32::MAX as usize {
            return Err(Error(format!(
                "row groups cover {next} rows, expected {rows} (at least one)"
            )));
        }
        check_len("grouped network input", x.len(), rows * self.input)?;
        check_len("grouped network value", value.len(), rows)?;
        check_len(
            "grouped network gradient",
            gradient.len(),
            rows * self.input,
        )?;
        self.work(device, rows)?;
        let stream = device.stream().clone();
        let size = std::mem::size_of::<T>() as u64;
        let (px, _gx) = x.device_ptr(&stream);
        let (pvalue, _gv) = value.device_ptr_mut(&stream);
        let (pgradient, _gg) = gradient.device_ptr_mut(&stream);
        let work = self.work.as_ref().unwrap();
        let address = |buffer: &CudaSlice<T>| buffer.device_ptr(&stream).0;
        let outputs: Vec<u64> = work.outputs.iter().map(address).collect();
        let slopes: Vec<u64> = work.slopes.iter().map(address).collect();
        let last = address(&work.last);
        let adjoint = address(&work.adjoint);
        let scaled = address(&work.scaled);
        let depth = self.hidden.len();
        let input_of = |l: usize| if l == 0 { px } else { outputs[l - 1] };
        // The products in order, each with its weights and the bases of its
        // two row operands (input, output) and their widths.
        let mut products: Vec<(Product, u64, u64, u64, u64, u64)> = Vec::with_capacity(2 * depth);
        for (l, layer) in self.hidden.iter().enumerate() {
            let out = if l + 1 < depth { outputs[l] } else { last };
            products.push((
                Product {
                    transpose_a: false,
                    m: layer.output,
                    k: layer.input,
                    lda: layer.output,
                    ldb: layer.input,
                    ldc: layer.output,
                    accumulate: false,
                },
                address(&layer.weights),
                input_of(l),
                layer.input as u64,
                out,
                layer.output as u64,
            ));
        }
        // Backward: `h` the adjoint of the activation input of layer l (the
        // last layer's in `last`, then `scaled`); the adjoint of the input,
        // the output of the layer below, goes to `adjoint`, added to what is
        // there for a residual layer (the adjoint of its output, written by
        // the layer above or for the last one by the output layer).
        let mut targets = Vec::with_capacity(depth);
        for l in (0..depth).rev() {
            let layer = &self.hidden[l];
            let h = if l + 1 == depth { last } else { scaled };
            let (target, accumulate) = match l {
                0 => (pgradient, false),
                _ => (adjoint, layer.residual),
            };
            targets.push(target);
            products.push((
                Product {
                    transpose_a: true,
                    m: layer.input,
                    k: layer.output,
                    lda: layer.output,
                    ldb: layer.output,
                    ldc: layer.input,
                    accumulate,
                },
                address(&layer.weights),
                h,
                layer.output as u64,
                target,
                layer.input as u64,
            ));
        }
        // Operand addresses, then the group table.
        let count = groups.len();
        let mut addresses = Vec::with_capacity(3 * count * (products.len() + 1));
        for (product, weights, input, input_width, output, output_width) in &products {
            let per_net = (product.m * product.k) as u64 * size;
            addresses.extend(
                groups
                    .iter()
                    .map(|&(net, _, _)| weights + net as u64 * per_net),
            );
            addresses.extend(
                groups
                    .iter()
                    .map(|&(_, first, _)| input + first as u64 * input_width * size),
            );
            addresses.extend(
                groups
                    .iter()
                    .map(|&(_, first, _)| output + first as u64 * output_width * size),
            );
        }
        let table = addresses.len();
        for &(net, first, count) in groups {
            addresses.extend([first as u64, count as u64, net as u64]);
        }
        if self
            .arrays
            .as_ref()
            .is_none_or(|a| a.len() != addresses.len())
        {
            // SAFETY: written in full below.
            self.arrays = Some(unsafe { device.alloc_uninit::<u64>(addresses.len())? });
            self.addresses.clear();
        }
        let arrays = self.arrays.as_mut().unwrap();
        if self.addresses != addresses {
            stream
                .memcpy_htod(&addresses, arrays)
                .map_err(cuda_error("grouped product addresses"))?;
            self.addresses = addresses;
        }
        let (parrays, _ga) = arrays.device_ptr(&stream);
        let ptable = parrays + 8 * table as u64;
        let lengths: Vec<i32> = groups.iter().map(|&(_, _, n)| n as i32).collect();
        let widest_group = *lengths.iter().max().unwrap() as usize;
        let product = |at: usize| -> Result<()> {
            let (p, ..) = &products[at];
            let base = parrays + 8 * (3 * count * at) as u64;
            device.grouped_matmul::<T>(
                p.transpose_a,
                p.m,
                p.k,
                &lengths,
                (p.lda, p.ldb, p.ldc),
                p.accumulate,
                [base, base + 8 * count as u64, base + 16 * count as u64],
            )
        };
        let groups_y = count as u32;
        // Forward.
        for (l, layer) in self.hidden.iter().enumerate() {
            product(l)?;
            let (in_i, out_i) = (layer.input as i32, layer.output as i32);
            let (act, residual) = (layer.activation.code(), i32::from(layer.residual));
            let (pbias, _gb) = layer.bias.device_ptr(&stream);
            let pdt = layer
                .timestep
                .as_ref()
                .map_or(0, |t| t.device_ptr(&stream).0);
            let pinput = input_of(l);
            if l + 1 < depth {
                let mut launch = stream.launch_builder(&self.epilogue);
                launch
                    .arg(&outputs[l])
                    .arg(&slopes[l])
                    .arg(&pinput)
                    .arg(&ptable)
                    .arg(&pbias)
                    .arg(&pdt)
                    .arg(&in_i)
                    .arg(&out_i)
                    .arg(&act)
                    .arg(&residual);
                let blocks = (widest_group * layer.output).div_ceil(BLOCK as usize);
                let config = LaunchConfig {
                    grid_dim: (blocks as u32, groups_y, 1),
                    block_dim: (BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { launch.launch(config) }
                    .map(drop)
                    .map_err(cuda_error("grouped epilogue launch"))?;
            } else {
                let g = if layer.residual { adjoint } else { 0 };
                let (pw, _gw) = self.out_weights.device_ptr(&stream);
                let (pb, _gob) = self.out_bias.device_ptr(&stream);
                let mut launch = stream.launch_builder(&self.head);
                launch
                    .arg(&last)
                    .arg(&g)
                    .arg(&pinput)
                    .arg(&ptable)
                    .arg(&pbias)
                    .arg(&pdt)
                    .arg(&pw)
                    .arg(&pb)
                    .arg(&pvalue)
                    .arg(&in_i)
                    .arg(&out_i)
                    .arg(&act)
                    .arg(&residual);
                let config = LaunchConfig {
                    grid_dim: (
                        widest_group.div_ceil(HEAD_ROWS as usize) as u32,
                        groups_y,
                        1,
                    ),
                    block_dim: (BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { launch.launch(config) }
                    .map(drop)
                    .map_err(cuda_error("grouped output layer launch"))?;
            }
        }
        // Backward: a product per layer, the slopes of the layer below
        // between them.
        for (step, l) in (0..depth).rev().enumerate() {
            product(depth + step)?;
            if l > 0 {
                let n = (rows * self.hidden[l].input) as i64;
                let mut launch = stream.launch_builder(&self.scale);
                launch
                    .arg(&scaled)
                    .arg(&targets[step])
                    .arg(&slopes[l - 1])
                    .arg(&n);
                let config = LaunchConfig {
                    grid_dim: ((n as usize).div_ceil(BLOCK as usize) as u32, 1, 1),
                    block_dim: (BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { launch.launch(config) }
                    .map(drop)
                    .map_err(cuda_error("grouped slope launch"))?;
            }
        }
        Ok(())
    }
}

/// The status of a cuBLAS call as a result.
pub(crate) fn blas_status(status: cublasStatus_t, what: &str) -> Result<()> {
    match status {
        cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
        other => Err(Error(format!("cuBLAS {what} failed: {other:?}"))),
    }
}

pub(crate) fn operation(transpose: bool) -> cublasOperation_t {
    match transpose {
        true => cublasOperation_t::CUBLAS_OP_T,
        false => cublasOperation_t::CUBLAS_OP_N,
    }
}
