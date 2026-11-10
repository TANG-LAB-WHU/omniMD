//! Dense networks for training (plan I-8): a forward pass that carries a
//! tangent, and a reverse pass through both.
//!
//! A force-matching loss depends on derivatives of the energy, so its
//! parameter gradient needs second-order information. Here that is
//! reverse-over-forward differentiation: for a scalar `Φ` that depends on a
//! network's outputs `y` and on their directional derivatives `ẏ` along an
//! input direction `ẋ`, [`TrainNetwork::backward`] takes `∂Φ/∂y` and `∂Φ/∂ẏ`
//! and accumulates `∂Φ/∂θ` for every weight, bias and timestep, and
//! optionally returns `∂Φ/∂x` and `∂Φ/∂ẋ`.
//!
//! Layer semantics are those of [`crate::DenseLayer`]:
//! `y = φ(xW + b)·dt + x[j mod in]` (the last term when residual), and so
//! `ẏ = φ'(xW + b)·(ẋW)·dt + ẋ[j mod in]`.
//!
//! Parameters of every layer live in one flat device vector, so an optimizer
//! updates them in a single kernel; a layer only knows its offsets.
use crate::{check_len, cuda_error, Activation, Device, MatmulShape, Scalar};
use cudarc::driver::{
    CudaFunction, CudaSlice, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
};
use omnimd_ml_contract::{Error, Result};

pub(crate) const ACTIVATE2: &str = r#"
// Value, first and second derivative of the activation.
template <typename T>
__device__ void activate2(int act, T z, T* v, T* d1, T* d2) {
    if (act == 1) {
        *v = tanh(z);
        *d1 = T(1) - *v * *v;
        *d2 = T(-2) * *v * *d1;
    } else if (act == 2 || act == 3) {
        T e = exp(z >= T(0) ? -z : z);
        T sigmoid = z >= T(0) ? T(1) / (T(1) + e) : e / (T(1) + e);
        T slope = sigmoid * (T(1) - sigmoid);
        if (act == 2) {
            *v = sigmoid;
            *d1 = slope;
            *d2 = slope * (T(1) - T(2) * sigmoid);
        } else {
            *v = (z > T(0) ? z : T(0)) + log1p(e);
            *d1 = sigmoid;
            *d2 = slope;
        }
    } else {
        *v = z;
        *d1 = T(1);
        *d2 = T(0);
    }
}
"#;

const SOURCE: &str = r#"
// On entry y = x·W and zdot = ẋ·W. Writes y, ẏ and the records the reverse
// pass needs: φ', φ'' and, for a trainable timestep, φ.
template <typename T>
__device__ void train_forward(T* y, T* ydot, const T* zdot, T* p1, T* p2, T* phi, const T* x,
        const T* xdot, const T* bias, const T* dt, long long rows, int in, int out, int act,
        int residual) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * out) return;
    int j = (int)(idx % out);
    long long source = (idx / out) * in + j % in;
    T v, d1, d2;
    activate2(act, y[idx] + bias[j], &v, &d1, &d2);
    T scale = dt ? dt[j] : T(1);
    p1[idx] = d1;
    p2[idx] = d2;
    if (phi) phi[idx] = v;
    y[idx] = v * scale + (residual ? x[source] : T(0));
    ydot[idx] = d1 * scale * zdot[idx] + (residual ? xdot[source] : T(0));
}

// Adjoints of the pre-activation z = xW + b and of its tangent ż = ẋW from
// the adjoints of y and ẏ; with a timestep, also its per-row gradient terms.
template <typename T>
__device__ void train_backward(const T* ybar, const T* ydotbar, const T* p1, const T* p2,
        const T* phi, const T* zdot, const T* dt, T* zbar, T* zdbar, T* dtacc, long long rows,
        int out) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * out) return;
    int j = (int)(idx % out);
    T scale = dt ? dt[j] : T(1);
    zbar[idx] = (ybar[idx] * p1[idx] + ydotbar[idx] * p2[idx] * zdot[idx]) * scale;
    zdbar[idx] = ydotbar[idx] * p1[idx] * scale;
    if (dtacc) dtacc[idx] = ybar[idx] * phi[idx] + ydotbar[idx] * p1[idx] * zdot[idx];
}

// xbar[r][i] += sum over j = i (mod in) of ybar[r][j]: adjoint of the residual.
template <typename T>
__device__ void train_residual(T* xbar, const T* ybar, long long rows, int in, int out) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * in) return;
    int i = (int)(idx % in);
    long long r = idx / in;
    T sum = T(0);
    for (int j = i; j < out; j += in) sum += ybar[r * out + j];
    xbar[idx] += sum;
}

#define INSTANTIATE(T, S)                                                                     \
extern "C" __global__ void train_forward_##S(T* y, T* ydot, const T* zdot, T* p1, T* p2,      \
        T* phi, const T* x, const T* xdot, const T* bias, const T* dt, long long rows, int in,  \
        int out, int act, int residual) {                                                     \
    train_forward<T>(y, ydot, zdot, p1, p2, phi, x, xdot, bias, dt, rows, in, out, act,       \
                     residual);                                                               \
}                                                                                             \
extern "C" __global__ void train_backward_##S(const T* ybar, const T* ydotbar, const T* p1,   \
        const T* p2, const T* phi, const T* zdot, const T* dt, T* zbar, T* zdbar, T* dtacc,   \
        long long rows, int out) {                                                            \
    train_backward<T>(ybar, ydotbar, p1, p2, phi, zdot, dt, zbar, zdbar, dtacc, rows, out);   \
}                                                                                             \
extern "C" __global__ void train_residual_##S(T* xbar, const T* ybar, long long rows, int in, \
        int out) {                                                                            \
    train_residual<T>(xbar, ybar, rows, in, out);                                             \
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

/// One dense layer of a [`TrainNetwork`]: its shape and where its parameters
/// sit in the flat parameter vector (`weights` is row-major `[input][output]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainLayer {
    pub input: usize,
    pub output: usize,
    pub activation: Activation,
    pub residual: bool,
    pub weights: usize,
    pub bias: usize,
    pub timestep: Option<usize>,
}

struct LayerTape<T: Scalar> {
    y: CudaSlice<T>,
    ydot: CudaSlice<T>,
    zdot: CudaSlice<T>,
    p1: CudaSlice<T>,
    p2: CudaSlice<T>,
    phi: Option<CudaSlice<T>>,
}

/// What [`TrainNetwork::forward`] recorded for [`TrainNetwork::backward`].
pub struct TrainTape<T: Scalar> {
    rows: usize,
    layers: Vec<LayerTape<T>>,
}

impl<T: Scalar> TrainTape<T> {
    /// The network outputs `[rows][output]`.
    pub fn output(&self) -> &CudaSlice<T> {
        &self.layers.last().unwrap().y
    }

    /// Their tangents along the input direction given to `forward`.
    pub fn tangent(&self) -> &CudaSlice<T> {
        &self.layers.last().unwrap().ydot
    }
}

struct Kernels {
    forward: CudaFunction,
    backward: CudaFunction,
    residual: CudaFunction,
}

/// A chain of dense layers over a shared flat parameter vector.
pub struct TrainNetwork<T: Scalar> {
    layers: Vec<TrainLayer>,
    kernels: Kernels,
    _scalar: std::marker::PhantomData<T>,
}

impl<T: Scalar> TrainNetwork<T> {
    pub fn new(device: &Device, layers: Vec<TrainLayer>) -> Result<Self> {
        if layers.is_empty() {
            return Err(Error("a training network needs at least one layer".into()));
        }
        for pair in layers.windows(2) {
            if pair[0].output != pair[1].input {
                return Err(Error(format!(
                    "layer output {} does not match next input {}",
                    pair[0].output, pair[1].input
                )));
            }
        }
        for layer in &layers {
            if layer.input == 0
                || layer.output == 0
                || i32::try_from(layer.input.max(layer.output)).is_err()
                || (layer.residual
                    && layer.output != layer.input
                    && layer.output != 2 * layer.input)
            {
                return Err(Error(format!(
                    "unsupported training layer shape {}->{} (residual: {})",
                    layer.input, layer.output, layer.residual
                )));
            }
        }
        let source = format!("{ACTIVATE2}{SOURCE}");
        let function = |name: &str| {
            device
                .kernels()
                .function(&source, &format!("{name}_{}", T::SUFFIX))
        };
        Ok(Self {
            layers,
            kernels: Kernels {
                forward: function("train_forward")?,
                backward: function("train_backward")?,
                residual: function("train_residual")?,
            },
            _scalar: std::marker::PhantomData,
        })
    }

    pub fn layers(&self) -> &[TrainLayer] {
        &self.layers
    }

    pub fn input(&self) -> usize {
        self.layers[0].input
    }

    pub fn output(&self) -> usize {
        self.layers.last().unwrap().output
    }

    /// Outputs and their tangents for inputs `x[rows][input]` and input
    /// directions `xdot`, recording what `backward` needs.
    pub fn forward(
        &self,
        device: &Device,
        theta: &CudaSlice<T>,
        x: &impl DevicePtr<T>,
        xdot: &impl DevicePtr<T>,
        rows: usize,
    ) -> Result<TrainTape<T>> {
        check_len("training input", x.len(), rows * self.input())?;
        check_len("training input tangent", xdot.len(), rows * self.input())?;
        let stream = device.stream();
        let mut tape: Vec<LayerTape<T>> = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let total = rows * layer.output;
            let shape = MatmulShape {
                m: rows,
                k: layer.input,
                n: layer.output,
            };
            let weights = theta.slice(layer.weights..layer.weights + layer.input * layer.output);
            let bias = theta.slice(layer.bias..layer.bias + layer.output);
            let timestep = layer.timestep.map(|t| theta.slice(t..t + layer.output));
            // SAFETY: every element of these is written below before it is read.
            let (mut y, mut zdot, mut ydot, mut p1, mut p2) = unsafe {
                (
                    device.alloc_uninit::<T>(total)?,
                    device.alloc_uninit::<T>(total)?,
                    device.alloc_uninit::<T>(total)?,
                    device.alloc_uninit::<T>(total)?,
                    device.alloc_uninit::<T>(total)?,
                )
            };
            let mut phi = match layer.timestep {
                Some(_) => Some(unsafe { device.alloc_uninit::<T>(total)? }),
                None => None,
            };
            match tape.last() {
                Some(previous) => {
                    device.matmul(&previous.y, &weights, &mut y, shape, false)?;
                    device.matmul(&previous.ydot, &weights, &mut zdot, shape, false)?;
                }
                None => {
                    device.matmul(x, &weights, &mut y, shape, false)?;
                    device.matmul(xdot, &weights, &mut zdot, shape, false)?;
                }
            }
            {
                // Inputs of this layer: the previous outputs, or the caller's.
                let inputs = match tape.last() {
                    Some(previous) => (
                        previous.y.device_ptr(stream),
                        previous.ydot.device_ptr(stream),
                    ),
                    None => (x.device_ptr(stream), xdot.device_ptr(stream)),
                };
                let (px, pxdot) = (inputs.0 .0, inputs.1 .0);
                let (pbias, _bias) = bias.device_ptr(stream);
                let dt = timestep.as_ref().map(|t| t.device_ptr(stream));
                let pdt = dt.as_ref().map_or(0, |(pointer, _)| *pointer);
                let recorded = phi.as_mut().map(|p| p.device_ptr_mut(stream));
                let pphi = recorded.as_ref().map_or(0, |(pointer, _)| *pointer);
                let (rows_i, in_i, out_i) = (rows as i64, layer.input as i32, layer.output as i32);
                let (act, residual) = (layer.activation.code(), i32::from(layer.residual));
                let mut launch = stream.launch_builder(&self.kernels.forward);
                launch
                    .arg(&mut y)
                    .arg(&mut ydot)
                    .arg(&zdot)
                    .arg(&mut p1)
                    .arg(&mut p2)
                    .arg(&pphi)
                    .arg(&px)
                    .arg(&pxdot)
                    .arg(&pbias)
                    .arg(&pdt)
                    .arg(&rows_i)
                    .arg(&in_i)
                    .arg(&out_i)
                    .arg(&act)
                    .arg(&residual);
                unsafe { launch.launch(grid(total)?) }
                    .map(drop)
                    .map_err(cuda_error("training forward launch"))?;
            }
            tape.push(LayerTape {
                y,
                ydot,
                zdot,
                p1,
                p2,
                phi,
            });
        }
        Ok(TrainTape { rows, layers: tape })
    }

    /// Reverse pass for a scalar `Φ` with `ybar = ∂Φ/∂y` and
    /// `ydotbar = ∂Φ/∂ẏ` (`[rows][output]`): adds `∂Φ/∂θ` into `gradient`
    /// (same layout as `theta`). `x` and `xdot` are the inputs given to
    /// `forward`; `ones` holds at least `rows` ones.
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        device: &Device,
        theta: &CudaSlice<T>,
        gradient: &mut CudaSlice<T>,
        tape: &TrainTape<T>,
        x: &impl DevicePtr<T>,
        xdot: &impl DevicePtr<T>,
        ybar: &impl DevicePtr<T>,
        ydotbar: &impl DevicePtr<T>,
        ones: &CudaSlice<T>,
    ) -> Result<()> {
        let input: Option<(&mut CudaSlice<T>, &mut CudaSlice<T>)> = None;
        self.reverse(
            device, theta, gradient, true, tape, x, xdot, ybar, ydotbar, ones, input,
        )
    }

    /// Like [`Self::backward`], and also writes `∂Φ/∂x` into `xbar` and
    /// `∂Φ/∂ẋ` into `xdotbar` (`[rows][input]`). With `parameters` false the
    /// parameter gradient is skipped and `gradient` is left untouched.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_to_input(
        &self,
        device: &Device,
        theta: &CudaSlice<T>,
        gradient: &mut CudaSlice<T>,
        parameters: bool,
        tape: &TrainTape<T>,
        x: &impl DevicePtr<T>,
        xdot: &impl DevicePtr<T>,
        ybar: &impl DevicePtr<T>,
        ydotbar: &impl DevicePtr<T>,
        ones: &CudaSlice<T>,
        xbar: &mut impl DevicePtrMut<T>,
        xdotbar: &mut impl DevicePtrMut<T>,
    ) -> Result<()> {
        self.reverse(
            device,
            theta,
            gradient,
            parameters,
            tape,
            x,
            xdot,
            ybar,
            ydotbar,
            ones,
            Some((xbar, xdotbar)),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn reverse<A: DevicePtrMut<T>, B: DevicePtrMut<T>>(
        &self,
        device: &Device,
        theta: &CudaSlice<T>,
        gradient: &mut CudaSlice<T>,
        parameters: bool,
        tape: &TrainTape<T>,
        x: &impl DevicePtr<T>,
        xdot: &impl DevicePtr<T>,
        ybar: &impl DevicePtr<T>,
        ydotbar: &impl DevicePtr<T>,
        ones: &CudaSlice<T>,
        mut input: Option<(&mut A, &mut B)>,
    ) -> Result<()> {
        let rows = tape.rows;
        check_len("training output adjoint", ybar.len(), rows * self.output())?;
        check_len(
            "training tangent adjoint",
            ydotbar.len(),
            rows * self.output(),
        )?;
        check_len("training gradient", gradient.len(), theta.len())?;
        if ones.len() < rows {
            return Err(Error(format!(
                "training reverse pass needs {rows} ones, got {}",
                ones.len()
            )));
        }
        let ones = ones.slice(0..rows);
        let stream = device.stream();
        // Adjoints of the current layer's outputs; the caller's for the last.
        let mut adjoints: Option<(CudaSlice<T>, CudaSlice<T>)> = None;
        for (index, layer) in self.layers.iter().enumerate().rev() {
            let record = &tape.layers[index];
            let total = rows * layer.output;
            let weights = theta.slice(layer.weights..layer.weights + layer.input * layer.output);
            let timestep = layer.timestep.map(|t| theta.slice(t..t + layer.output));
            // SAFETY: fully written by the kernel below.
            let (mut zbar, mut zdbar) = unsafe {
                (
                    device.alloc_uninit::<T>(total)?,
                    device.alloc_uninit::<T>(total)?,
                )
            };
            let mut dtacc = match (layer.timestep, parameters) {
                (Some(_), true) => Some(unsafe { device.alloc_uninit::<T>(total)? }),
                _ => None,
            };
            {
                let pointers = match &adjoints {
                    Some((a, b)) => (a.device_ptr(stream), b.device_ptr(stream)),
                    None => (ybar.device_ptr(stream), ydotbar.device_ptr(stream)),
                };
                let (pybar, pydotbar) = (pointers.0 .0, pointers.1 .0);
                let phi = record.phi.as_ref().map(|p| p.device_ptr(stream));
                let pphi = phi.as_ref().map_or(0, |(pointer, _)| *pointer);
                let dt = timestep.as_ref().map(|t| t.device_ptr(stream));
                let pdt = dt.as_ref().map_or(0, |(pointer, _)| *pointer);
                let acc = dtacc.as_mut().map(|p| p.device_ptr_mut(stream));
                let pacc = acc.as_ref().map_or(0, |(pointer, _)| *pointer);
                let (rows_i, out_i) = (rows as i64, layer.output as i32);
                let mut launch = stream.launch_builder(&self.kernels.backward);
                launch
                    .arg(&pybar)
                    .arg(&pydotbar)
                    .arg(&record.p1)
                    .arg(&record.p2)
                    .arg(&pphi)
                    .arg(&record.zdot)
                    .arg(&pdt)
                    .arg(&mut zbar)
                    .arg(&mut zdbar)
                    .arg(&pacc)
                    .arg(&rows_i)
                    .arg(&out_i);
                unsafe { launch.launch(grid(total)?) }
                    .map(drop)
                    .map_err(cuda_error("training backward launch"))?;
            }
            if parameters {
                let shape = MatmulShape {
                    m: rows,
                    k: layer.input,
                    n: layer.output,
                };
                {
                    let mut g = gradient
                        .slice_mut(layer.weights..layer.weights + layer.input * layer.output);
                    match index {
                        0 => {
                            device.matmul_at(x, &zbar, &mut g, shape, true)?;
                            device.matmul_at(xdot, &zdbar, &mut g, shape, true)?;
                        }
                        _ => {
                            let previous = &tape.layers[index - 1];
                            device.matmul_at(&previous.y, &zbar, &mut g, shape, true)?;
                            device.matmul_at(&previous.ydot, &zdbar, &mut g, shape, true)?;
                        }
                    }
                }
                let column = MatmulShape {
                    m: rows,
                    k: 1,
                    n: layer.output,
                };
                {
                    let mut g = gradient.slice_mut(layer.bias..layer.bias + layer.output);
                    device.matmul_at(&ones, &zbar, &mut g, column, true)?;
                }
                if let (Some(offset), Some(acc)) = (layer.timestep, &dtacc) {
                    let mut g = gradient.slice_mut(offset..offset + layer.output);
                    device.matmul_at(&ones, acc, &mut g, column, true)?;
                }
            }
            if index == 0 && input.is_none() {
                break;
            }
            let back = MatmulShape {
                m: rows,
                k: layer.output,
                n: layer.input,
            };
            let residual = |target: u64, source: u64| -> Result<()> {
                let (rows_i, in_i, out_i) = (rows as i64, layer.input as i32, layer.output as i32);
                let mut launch = stream.launch_builder(&self.kernels.residual);
                launch
                    .arg(&target)
                    .arg(&source)
                    .arg(&rows_i)
                    .arg(&in_i)
                    .arg(&out_i);
                unsafe { launch.launch(grid(rows * layer.input)?) }
                    .map(drop)
                    .map_err(cuda_error("training residual launch"))
            };
            let sources = match &adjoints {
                Some((a, b)) => (a.device_ptr(stream).0, b.device_ptr(stream).0),
                None => (ybar.device_ptr(stream).0, ydotbar.device_ptr(stream).0),
            };
            if index == 0 {
                let (xbar, xdbar) = input.as_mut().unwrap();
                check_len("training input adjoint", xbar.len(), rows * layer.input)?;
                check_len(
                    "training input tangent adjoint",
                    xdbar.len(),
                    rows * layer.input,
                )?;
                device.matmul_bt(&zbar, &weights, &mut **xbar, back, false)?;
                device.matmul_bt(&zdbar, &weights, &mut **xdbar, back, false)?;
                if layer.residual {
                    residual(xbar.device_ptr_mut(stream).0, sources.0)?;
                    residual(xdbar.device_ptr_mut(stream).0, sources.1)?;
                }
            } else {
                // SAFETY: fully written by the products below.
                let (mut xbar, mut xdbar) = unsafe {
                    (
                        device.alloc_uninit::<T>(rows * layer.input)?,
                        device.alloc_uninit::<T>(rows * layer.input)?,
                    )
                };
                device.matmul_bt(&zbar, &weights, &mut xbar, back, false)?;
                device.matmul_bt(&zdbar, &weights, &mut xdbar, back, false)?;
                if layer.residual {
                    residual(xbar.device_ptr_mut(stream).0, sources.0)?;
                    residual(xdbar.device_ptr_mut(stream).0, sources.1)?;
                }
                adjoints = Some((xbar, xdbar));
            }
        }
        Ok(())
    }
}
