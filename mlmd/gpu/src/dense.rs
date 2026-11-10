//! Dense layers with the native network semantics (`native/src/network.rs`):
//! `y = act(x·W + b)·dt`, plus `x[j mod input]` when residual, and
//! `slope = act'(x·W + b)·dt`. The products run in cuBLAS; bias, activation,
//! timestep and residual are fused epilogue kernels.
//!
//! Three passes: `forward` (optionally recording slopes), `forward_tangent`
//! (value and directional derivative, used for the scalar-input embedding net)
//! and `backward` (input adjoint from output adjoint and recorded slopes).
use crate::{check_len, cuda_error, Device, MatmulShape, Scalar};
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

// y holds x·W on entry; dy (tangent mode) holds dx·W.
template <typename T>
__device__ void epilogue(T* y, T* slope, T* dy, const T* x, const T* dx, const T* bias,
                         const T* dt, long long rows, int in, int out, int act, int residual) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * out) return;
    int j = (int)(idx % out);
    long long r = idx / out;
    T v, s;
    activate(act, y[idx] + bias[j], &v, &s);
    T scale = dt ? dt[j] : T(1);
    v *= scale;
    s *= scale;
    long long source = r * in + j % in;
    if (residual) v += x[source];
    y[idx] = v;
    if (slope) slope[idx] = s;
    if (dy) dy[idx] = s * dy[idx] + (residual ? dx[source] : T(0));
}

template <typename T>
__device__ void scale_by_slope(T* out, const T* adjoint, const T* slope, long long n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = adjoint[idx] * slope[idx];
}

// dx[r][i] += sum over j = i (mod in) of dy[r][j]: adjoint of the residual.
template <typename T>
__device__ void residual_adjoint(T* dx, const T* dy, long long rows, int in, int out) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * in) return;
    int i = (int)(idx % in);
    long long r = idx / in;
    T sum = T(0);
    for (int j = i; j < out; j += in) sum += dy[r * out + j];
    dx[idx] += sum;
}

#define INSTANTIATE(T, S)                                                                     \
extern "C" __global__ void dense_epilogue_##S(T* y, T* slope, T* dy, const T* x, const T* dx, \
        const T* bias, const T* dt, long long rows, int in, int out, int act, int residual) {   \
    epilogue<T>(y, slope, dy, x, dx, bias, dt, rows, in, out, act, residual);                 \
}                                                                                             \
extern "C" __global__ void dense_scale_##S(T* out, const T* adjoint, const T* slope,          \
        long long n) {                                                                        \
    scale_by_slope<T>(out, adjoint, slope, n);                                                \
}                                                                                             \
extern "C" __global__ void dense_residual_adjoint_##S(T* dx, const T* dy, long long rows,     \
        int in, int out) {                                                                    \
    residual_adjoint<T>(dx, dy, rows, in, out);                                               \
}
INSTANTIATE(float, f32)
INSTANTIATE(double, f64)
"#;

const BLOCK: u32 = 256;

fn grid(total: usize) -> Result<LaunchConfig> {
    let blocks = total.div_ceil(BLOCK as usize);
    if total == 0 || blocks > u32::MAX as usize {
        return Err(Error(format!(
            "kernel over {total} elements is out of range"
        )));
    }
    Ok(LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Linear,
    Tanh,
    Sigmoid,
    Softplus,
}

impl Activation {
    pub(crate) fn code(self) -> i32 {
        match self {
            Self::Linear => 0,
            Self::Tanh => 1,
            Self::Sigmoid => 2,
            Self::Softplus => 3,
        }
    }
}

/// Host-side shape rules, checked before anything touches the device.
pub(crate) fn validate(
    input: usize,
    output: usize,
    weights: usize,
    bias: usize,
    timestep: Option<usize>,
    residual: bool,
) -> Result<()> {
    if input == 0 || output == 0 || i32::try_from(input.max(output)).is_err() {
        return Err(Error(format!(
            "dense layer widths must be in 1..=i32::MAX, got {input}->{output}"
        )));
    }
    if weights != input * output || bias != output || timestep.is_some_and(|t| t != output) {
        return Err(Error(format!(
            "dense layer {input}->{output} needs {} weights, {output} biases and {output} timesteps; got {weights}, {bias}, {timestep:?}",
            input * output
        )));
    }
    if residual && output != input && output != 2 * input {
        return Err(Error(format!(
            "residual dense layer must keep or double its width, got {input}->{output}"
        )));
    }
    Ok(())
}

struct Kernels {
    epilogue: CudaFunction,
    scale: CudaFunction,
    residual_adjoint: CudaFunction,
}

pub struct DenseLayer<T: Scalar> {
    input: usize,
    output: usize,
    activation: Activation,
    residual: bool,
    weights: CudaSlice<T>,
    bias: CudaSlice<T>,
    timestep: Option<CudaSlice<T>>,
    kernels: Kernels,
}

impl<T: Scalar> DenseLayer<T> {
    /// `weights` is row-major `[input][output]`, like `Layer::weights`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        input: usize,
        output: usize,
        weights: &[T],
        bias: &[T],
        timestep: Option<&[T]>,
        activation: Activation,
        residual: bool,
    ) -> Result<Self> {
        validate(
            input,
            output,
            weights.len(),
            bias.len(),
            timestep.map(<[T]>::len),
            residual,
        )?;
        let function = |name: &str| {
            device
                .kernels()
                .function(SOURCE, &format!("{name}_{}", T::SUFFIX))
        };
        Ok(Self {
            input,
            output,
            activation,
            residual,
            weights: device.upload(weights)?,
            bias: device.upload(bias)?,
            timestep: timestep.map(|t| device.upload(t)).transpose()?,
            kernels: Kernels {
                epilogue: function("dense_epilogue")?,
                scale: function("dense_scale")?,
                residual_adjoint: function("dense_residual_adjoint")?,
            },
        })
    }

    pub fn input(&self) -> usize {
        self.input
    }

    pub fn output(&self) -> usize {
        self.output
    }

    fn shape(&self, rows: usize) -> MatmulShape {
        MatmulShape {
            m: rows,
            k: self.input,
            n: self.output,
        }
    }

    /// `rows` independent inputs `x[rows][input]` -> `y[rows][output]`.
    /// `slopes`, when given, receives `act'·dt` for the reverse pass.
    pub fn forward(
        &self,
        device: &Device,
        x: &impl DevicePtr<T>,
        rows: usize,
        y: &mut impl DevicePtrMut<T>,
        slopes: Option<&mut CudaSlice<T>>,
    ) -> Result<()> {
        device.matmul(x, &self.weights, y, self.shape(rows), false)?;
        let total = rows * self.output;
        let stream = device.stream();
        let (px, _gx) = x.device_ptr(stream);
        let (py, _gy) = y.device_ptr_mut(stream);
        let (pslope, _gs) = match slopes {
            Some(s) => {
                check_len("dense slopes", s.len(), total)?;
                let (p, g) = s.device_ptr_mut(stream);
                (p, Some(g))
            }
            None => (0, None),
        };
        self.epilogue(device, rows, [py, pslope, 0, px, 0])
    }

    /// Value and tangent: `dy` is the derivative of `y` along input direction `dx`.
    pub fn forward_tangent(
        &self,
        device: &Device,
        x: &impl DevicePtr<T>,
        dx: &impl DevicePtr<T>,
        rows: usize,
        y: &mut impl DevicePtrMut<T>,
        dy: &mut impl DevicePtrMut<T>,
    ) -> Result<()> {
        device.matmul(x, &self.weights, y, self.shape(rows), false)?;
        device.matmul(dx, &self.weights, dy, self.shape(rows), false)?;
        let stream = device.stream();
        let (px, _gx) = x.device_ptr(stream);
        let (pdx, _gdx) = dx.device_ptr(stream);
        let (py, _gy) = y.device_ptr_mut(stream);
        let (pdy, _gdy) = dy.device_ptr_mut(stream);
        self.epilogue(device, rows, [py, 0, pdy, px, pdx])
    }

    /// Bias, activation, timestep and residual on `y = x·W` (and its tangent).
    /// Pointers are `[y, slope, dy, x, dx]`, 0 when absent; callers check
    /// lengths (cuBLAS already checked y, dy, x and dx) and hold the borrows.
    fn epilogue(&self, device: &Device, rows: usize, pointers: [u64; 5]) -> Result<()> {
        let config = grid(rows * self.output)?;
        let stream = device.stream();
        let (pdt, _gdt) = match &self.timestep {
            Some(t) => {
                let (p, g) = t.device_ptr(stream);
                (p, Some(g))
            }
            None => (0, None),
        };
        let (pbias, _gbias) = self.bias.device_ptr(stream);
        let [py, pslope, pdy, px, pdx] = pointers;
        let (rows, input, output) = (rows as i64, self.input as i32, self.output as i32);
        let (act, residual) = (self.activation.code(), i32::from(self.residual));
        let mut launch = stream.launch_builder(&self.kernels.epilogue);
        launch
            .arg(&py)
            .arg(&pslope)
            .arg(&pdy)
            .arg(&px)
            .arg(&pdx)
            .arg(&pbias)
            .arg(&pdt)
            .arg(&rows)
            .arg(&input)
            .arg(&output)
            .arg(&act)
            .arg(&residual);
        unsafe { launch.launch(config) }
            .map(drop)
            .map_err(cuda_error("dense epilogue launch"))
    }
    /// Input adjoint `dx[rows][input]` from output adjoint `dy` and the slopes
    /// recorded by `forward`. `dy` is not modified.
    pub fn backward(
        &self,
        device: &Device,
        dy: &impl DevicePtr<T>,
        slopes: &CudaSlice<T>,
        rows: usize,
        dx: &mut impl DevicePtrMut<T>,
    ) -> Result<()> {
        let total = rows * self.output;
        check_len("dense output adjoint", dy.len(), total)?;
        check_len("dense slopes", slopes.len(), total)?;
        let stream = device.stream();
        // SAFETY: every element is written by the scale kernel below.
        let mut scaled = unsafe { device.alloc_uninit::<T>(total)? };
        {
            let (pout, _g0) = scaled.device_ptr_mut(stream);
            let (pdy, _g1) = dy.device_ptr(stream);
            let (ps, _g2) = slopes.device_ptr(stream);
            let n = total as i64;
            let mut launch = stream.launch_builder(&self.kernels.scale);
            launch.arg(&pout).arg(&pdy).arg(&ps).arg(&n);
            unsafe { launch.launch(grid(total)?) }
                .map(drop)
                .map_err(cuda_error("dense slope scaling"))?;
        }
        // dx = (dy ⊙ slope) · Wᵀ, with W stored [input][output] = [n][k].
        let shape = MatmulShape {
            m: rows,
            k: self.output,
            n: self.input,
        };
        device.matmul_bt(&scaled, &self.weights, dx, shape, false)?;
        if self.residual {
            let (pdx, _g0) = dx.device_ptr_mut(stream);
            let (pdy, _g1) = dy.device_ptr(stream);
            let (r, i, o) = (rows as i64, self.input as i32, self.output as i32);
            let mut launch = stream.launch_builder(&self.kernels.residual_adjoint);
            launch.arg(&pdx).arg(&pdy).arg(&r).arg(&i).arg(&o);
            unsafe { launch.launch(grid(rows * self.input)?) }
                .map(drop)
                .map_err(cuda_error("dense residual adjoint"))?;
        }
        Ok(())
    }
}

/// What `DenseNetwork::forward_record` keeps for `backward`: per-layer slopes
/// and the final output.
pub struct DenseTape<T: Scalar> {
    rows: usize,
    slopes: Vec<CudaSlice<T>>,
    pub output: CudaSlice<T>,
}

/// A chain of dense layers, e.g. an embedding net [1->25->50->100].
pub struct DenseNetwork<T: Scalar> {
    layers: Vec<DenseLayer<T>>,
}

impl<T: Scalar> DenseNetwork<T> {
    pub fn new(layers: Vec<DenseLayer<T>>) -> Result<Self> {
        if layers.is_empty() {
            return Err(Error("dense network needs at least one layer".into()));
        }
        if let Some(pair) = layers.windows(2).find(|p| p[0].output != p[1].input) {
            return Err(Error(format!(
                "dense network width mismatch: {} outputs feed {} inputs",
                pair[0].output, pair[1].input
            )));
        }
        Ok(Self { layers })
    }

    pub fn input(&self) -> usize {
        self.layers[0].input
    }

    pub fn output(&self) -> usize {
        self.layers.last().unwrap().output
    }

    /// Inference forward pass; intermediates come from the stream-ordered pool.
    pub fn forward(
        &self,
        device: &Device,
        x: &impl DevicePtr<T>,
        rows: usize,
    ) -> Result<CudaSlice<T>> {
        // SAFETY: fully overwritten by the GEMM (beta = 0) and epilogue.
        let mut out = unsafe { device.alloc_uninit(rows * self.output())? };
        self.forward_into(device, x, rows, &mut out)?;
        Ok(out)
    }

    /// Forward pass writing the final layer straight into `out`.
    pub fn forward_into(
        &self,
        device: &Device,
        x: &impl DevicePtr<T>,
        rows: usize,
        out: &mut impl DevicePtrMut<T>,
    ) -> Result<()> {
        check_len("network input", x.len(), rows * self.input())?;
        let (last, hidden) = self.layers.split_last().unwrap();
        let mut current: Option<CudaSlice<T>> = None;
        for layer in hidden {
            // SAFETY: fully overwritten by the GEMM (beta = 0) and epilogue.
            let mut y = unsafe { device.alloc_uninit(rows * layer.output)? };
            match &current {
                Some(c) => layer.forward(device, c, rows, &mut y, None)?,
                None => layer.forward(device, x, rows, &mut y, None)?,
            }
            current = Some(y);
        }
        match &current {
            Some(c) => last.forward(device, c, rows, out, None),
            None => last.forward(device, x, rows, out, None),
        }
    }

    /// Value and tangent along input direction `dx`, written to `out`/`dout`.
    pub fn forward_tangent_into(
        &self,
        device: &Device,
        x: &impl DevicePtr<T>,
        dx: &impl DevicePtr<T>,
        rows: usize,
        out: &mut impl DevicePtrMut<T>,
        dout: &mut impl DevicePtrMut<T>,
    ) -> Result<()> {
        check_len("network input", x.len(), rows * self.input())?;
        let (last, hidden) = self.layers.split_last().unwrap();
        let mut current: Option<(CudaSlice<T>, CudaSlice<T>)> = None;
        for layer in hidden {
            // SAFETY: fully overwritten by the GEMM (beta = 0) and epilogue.
            let mut y = unsafe { device.alloc_uninit(rows * layer.output)? };
            // SAFETY: fully overwritten by the GEMM (beta = 0) and epilogue.
            let mut dy = unsafe { device.alloc_uninit(rows * layer.output)? };
            match &current {
                Some((c, dc)) => layer.forward_tangent(device, c, dc, rows, &mut y, &mut dy)?,
                None => layer.forward_tangent(device, x, dx, rows, &mut y, &mut dy)?,
            }
            current = Some((y, dy));
        }
        match &current {
            Some((c, dc)) => last.forward_tangent(device, c, dc, rows, out, dout),
            None => last.forward_tangent(device, x, dx, rows, out, dout),
        }
    }

    /// Forward pass that records slopes for `backward`.
    pub fn forward_record(
        &self,
        device: &Device,
        x: &impl DevicePtr<T>,
        rows: usize,
    ) -> Result<DenseTape<T>> {
        check_len("network input", x.len(), rows * self.input())?;
        let mut slopes = Vec::with_capacity(self.layers.len());
        let mut current: Option<CudaSlice<T>> = None;
        for layer in &self.layers {
            // SAFETY: fully overwritten by the GEMM (beta = 0) and epilogue.
            let mut y = unsafe { device.alloc_uninit(rows * layer.output)? };
            // SAFETY: fully overwritten by the GEMM (beta = 0) and epilogue.
            let mut slope = unsafe { device.alloc_uninit(rows * layer.output)? };
            match &current {
                Some(c) => layer.forward(device, c, rows, &mut y, Some(&mut slope))?,
                None => layer.forward(device, x, rows, &mut y, Some(&mut slope))?,
            }
            slopes.push(slope);
            current = Some(y);
        }
        Ok(DenseTape {
            rows,
            slopes,
            output: current.unwrap(),
        })
    }

    /// Input adjoint `dx[rows][input]` for output adjoint `dout[rows][output]`.
    pub fn backward(
        &self,
        device: &Device,
        tape: &DenseTape<T>,
        dout: &impl DevicePtr<T>,
        dx: &mut impl DevicePtrMut<T>,
    ) -> Result<()> {
        let rows = tape.rows;
        let (first, rest) = self.layers.split_first().unwrap();
        let mut adjoint: Option<CudaSlice<T>> = None;
        for (layer, slope) in rest.iter().zip(&tape.slopes[1..]).rev() {
            // SAFETY: fully overwritten by the GEMM (beta = 0) and epilogue.
            let mut previous = unsafe { device.alloc_uninit(rows * layer.input)? };
            match &adjoint {
                Some(a) => layer.backward(device, a, slope, rows, &mut previous)?,
                None => layer.backward(device, dout, slope, rows, &mut previous)?,
            }
            adjoint = Some(previous);
        }
        match &adjoint {
            Some(a) => first.backward(device, a, &tape.slopes[0], rows, dx),
            None => first.backward(device, dout, &tape.slopes[0], rows, dx),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::validate;

    #[test]
    fn shape_rules() {
        assert!(validate(3, 4, 12, 4, None, false).is_ok());
        assert!(validate(3, 3, 9, 3, Some(3), true).is_ok());
        assert!(validate(3, 6, 18, 6, None, true).is_ok());
        assert!(
            validate(3, 5, 15, 5, None, true).is_err(),
            "residual needs keep/double"
        );
        assert!(validate(3, 4, 11, 4, None, false).is_err(), "weights");
        assert!(validate(3, 4, 12, 3, None, false).is_err(), "bias");
        assert!(validate(3, 4, 12, 4, Some(3), false).is_err(), "timestep");
        assert!(validate(0, 4, 0, 4, None, false).is_err(), "empty input");
    }
}
