//! Dense layers with the native network semantics (`native/src/network.rs`):
//! `y = act(x·W + b)·dt`, plus `x[j mod input]` when residual, and
//! `slope = act'(x·W + b)·dt`. The product runs in cuBLAS; bias, activation,
//! timestep and residual are one fused epilogue kernel.
use crate::{check_len, cuda_error, Device, MatmulShape, Scalar};
use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use omnimd_ml_contract::{Error, Result};

const EPILOGUE: &str = r#"
template <typename T>
__device__ void dense_epilogue(T* y, T* slope, const T* x, const T* bias, const T* dt,
                               long long rows, int in, int out, int act, int residual) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * out) return;
    int j = (int)(idx % out);
    long long r = idx / out;
    T z = y[idx] + bias[j];
    T v, s;
    if (act == 1) {
        v = tanh(z);
        s = T(1) - v * v;
    } else if (act == 2 || act == 3) {
        T e = exp(z >= T(0) ? -z : z);
        T sigmoid = z >= T(0) ? T(1) / (T(1) + e) : e / (T(1) + e);
        if (act == 2) {
            v = sigmoid;
            s = sigmoid * (T(1) - sigmoid);
        } else {
            v = (z > T(0) ? z : T(0)) + log1p(e);
            s = sigmoid;
        }
    } else {
        v = z;
        s = T(1);
    }
    T scale = dt ? dt[j] : T(1);
    v *= scale;
    s *= scale;
    if (residual) v += x[r * in + j % in];
    y[idx] = v;
    if (slope) slope[idx] = s;
}

extern "C" __global__ void dense_epilogue_f32(float* y, float* slope, const float* x,
        const float* bias, const float* dt, long long rows, int in, int out, int act, int residual) {
    dense_epilogue<float>(y, slope, x, bias, dt, rows, in, out, act, residual);
}

extern "C" __global__ void dense_epilogue_f64(double* y, double* slope, const double* x,
        const double* bias, const double* dt, long long rows, int in, int out, int act, int residual) {
    dense_epilogue<double>(y, slope, x, bias, dt, rows, in, out, act, residual);
}
"#;

const BLOCK: u32 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Linear,
    Tanh,
    Sigmoid,
    Softplus,
}

impl Activation {
    fn code(self) -> i32 {
        match self {
            Self::Linear => 0,
            Self::Tanh => 1,
            Self::Sigmoid => 2,
            Self::Softplus => 3,
        }
    }
}

/// Host-side shape rules, checked before anything touches the device.
fn validate(
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

pub struct DenseLayer<T: Scalar> {
    input: usize,
    output: usize,
    activation: Activation,
    residual: bool,
    weights: CudaSlice<T>,
    bias: CudaSlice<T>,
    timestep: Option<CudaSlice<T>>,
    epilogue: CudaFunction,
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
        let name = format!("dense_epilogue_{}", T::SUFFIX);
        Ok(Self {
            input,
            output,
            activation,
            residual,
            weights: device.upload(weights)?,
            bias: device.upload(bias)?,
            timestep: timestep.map(|t| device.upload(t)).transpose()?,
            epilogue: device.kernels().function(EPILOGUE, &name)?,
        })
    }

    pub fn input(&self) -> usize {
        self.input
    }

    pub fn output(&self) -> usize {
        self.output
    }

    /// `rows` independent inputs `x[rows][input]` -> `y[rows][output]`.
    /// `slopes`, when given, receives `act'·dt` for the reverse pass.
    pub fn forward(
        &self,
        device: &Device,
        x: &CudaSlice<T>,
        rows: usize,
        y: &mut CudaSlice<T>,
        slopes: Option<&mut CudaSlice<T>>,
    ) -> Result<()> {
        let total = rows
            .checked_mul(self.output)
            .filter(|&t| t > 0 && t.div_ceil(BLOCK as usize) <= u32::MAX as usize)
            .ok_or_else(|| Error(format!("dense forward over {rows} rows is out of range")))?;
        if let Some(slopes) = &slopes {
            check_len("dense slopes", slopes.len(), total)?;
        }
        let shape = MatmulShape {
            m: rows,
            k: self.input,
            n: self.output,
        };
        device.matmul(x, &self.weights, y, shape, false)?;
        let null = 0u64;
        let (rows, input, output) = (rows as i64, self.input as i32, self.output as i32);
        let (act, residual) = (self.activation.code(), i32::from(self.residual));
        let mut launch = device.stream().launch_builder(&self.epilogue);
        launch.arg(y);
        match slopes {
            Some(slopes) => launch.arg(slopes),
            None => launch.arg(&null),
        };
        launch.arg(x).arg(&self.bias);
        match &self.timestep {
            Some(timestep) => launch.arg(timestep),
            None => launch.arg(&null),
        };
        launch
            .arg(&rows)
            .arg(&input)
            .arg(&output)
            .arg(&act)
            .arg(&residual);
        let config = LaunchConfig {
            grid_dim: (total.div_ceil(BLOCK as usize) as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { launch.launch(config) }
            .map(drop)
            .map_err(cuda_error("dense epilogue launch"))
    }
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
    pub fn forward(&self, device: &Device, x: &CudaSlice<T>, rows: usize) -> Result<CudaSlice<T>> {
        check_len("network input", x.len(), rows * self.input())?;
        let mut current: Option<CudaSlice<T>> = None;
        for layer in &self.layers {
            let mut y = device.zeros(rows * layer.output)?;
            layer.forward(device, current.as_ref().unwrap_or(x), rows, &mut y, None)?;
            current = Some(y);
        }
        Ok(current.unwrap())
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
