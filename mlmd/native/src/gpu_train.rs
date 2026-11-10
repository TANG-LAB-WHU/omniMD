//! se_e2_a training on the omnimd-gpu runtime (plan I-8): the loss of
//! DeePMD-kit's `ener` loss for one frame and its gradient with respect to
//! every network parameter, plus an Adam update, all on the device.
//!
//! Loss for a frame of `N` atoms, with prefactors `p_e`, `p_f`, `p_v`:
//!
//! ```text
//! L = p_e (E - E*)² / N + p_f mean((F - F*)²) + p_v mean((V - V*)²) / N
//! ```
//!
//! Forces and virial are derivatives of the energy, so `∂L/∂θ` needs
//! second-order information. With `a = ∂L/∂E`, `c = ∂L/∂F` and `w = ∂L/∂V`
//! held fixed, `∂L/∂θ = ∂Φ/∂θ` for `Φ = a·E − Ė`, where `Ė` is the
//! directional derivative of `E` when every neighbor vector `d = r_j − r_i`
//! moves along `c_j − c_i + wᵀd`. One forward pass carries that tangent
//! through environment matrix, embedding, descriptor and fitting; one
//! reverse pass through both gives the gradient (reverse-over-forward, see
//! [`omnimd_gpu::TrainNetwork`]).
//!
//! Networks run in `P`; geometry, the `B` accumulation, forces, virial and
//! the loss are float64. A frame is processed in one piece (no chunking), so
//! it must fit in device memory with its recorded intermediates.
use crate::gpu_model::{copy_error, launch_error, SOURCE as MODEL_SOURCE};
use crate::{Activation, Layer, Model, Network, Residual};
use omnimd_gpu::{
    Activation as DeviceActivation, Cell, CudaFunction, CudaSlice, Device, LaunchConfig,
    NeighborList, PushKernelArg, Scalar, TrainLayer, TrainNetwork, TrainTape,
};
use omnimd_ml_contract::{Error, Evaluation, Frame, ModelMetadata, Result, Units};
use std::sync::Arc;

const SOURCE: &str = r#"
__device__ void train_switch(double r, double rmin, double rmax, double* q, double* dq) {
    if (r < rmin) { *q = 1.0; *dq = 0.0; return; }
    double t = (r - rmin) / (rmax - rmin), z = 1.0 - t;
    *q = z * z * z * (1.0 + 3.0 * t + 6.0 * t * t);
    *dq = -30.0 * t * t * z * z / (rmax - rmin);
}

__device__ long long train_row(int count, int i, int t, int local, const int* sel, const int* offset) {
    return (long long)count * offset[t] + (long long)i * sel[t] + local;
}

// Tangent of the normalized environment rows when every neighbor vector d
// moves along c[j] - c[i] + w^T d. One thread per slot; empty slots get 0.
extern "C" __global__ void se_e2_a_env_tangent(const int* index, const double* delta,
        const int* atype, const double* dstd, const double* c, const double* w, int natoms,
        int slots, double rmin, double rmax, double* envdot) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)natoms * slots) return;
    int i = (int)(idx / slots), j = index[idx];
    double out[4] = {0.0, 0.0, 0.0, 0.0};
    if (j >= 0) {
        const double* d = delta + 3 * idx;
        double dd[3];
        for (int b = 0; b < 3; ++b)
            dd[b] = c[3 * (long long)j + b] - c[3 * (long long)i + b]
                  + w[b] * d[0] + w[3 + b] * d[1] + w[6 + b] * d[2];
        double rr = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]), q, dq;
        train_switch(rr, rmin, rmax, &q, &dq);
        double u[3] = {d[0] / rr, d[1] / rr, d[2] / rr};
        double sw = q / rr, dsw = dq / rr - sw / rr;
        double along = u[0] * dd[0] + u[1] * dd[1] + u[2] * dd[2];
        const double* std_c = dstd + 4 * atype[i];
        out[0] = dsw * along / std_c[0];
        for (int a = 0; a < 3; ++a)
            out[a + 1] = (dsw * u[a] * along + sw / rr * (dd[a] - u[a] * along)) / std_c[a + 1];
    }
    for (int a = 0; a < 4; ++a) envdot[4 * idx + a] = out[a];
}

// One block per atom: tangents of B and of D = B·B[:axis]^T. The embedding
// tangent of a row is dg (its derivative along the radial input) times the
// tangent of that input, envdot[..][0].
template <typename P>
__device__ void descriptor_tangent(const double* env, const double* envdot, const P* g,
        const P* dg, const double* b_in, const int* slot_type, const int* slot_local,
        const int* sel, const int* offset, int count, int slots, int width, int axis,
        double* bdot_out, P* ddot_out) {
    extern __shared__ double shared[];
    double* b = shared;
    double* bd = b + 4 * width;
    int i = blockIdx.x;
    for (int m = threadIdx.x; m < width; m += blockDim.x) {
        double acc[4] = {0.0, 0.0, 0.0, 0.0};
        for (int s = 0; s < slots; ++s) {
            long long row = train_row(count, i, slot_type[s], slot_local[s], sel, offset) * width + m;
            double gm = (double)g[row], gpm = (double)dg[row];
            const double* r = env + ((long long)i * slots + s) * 4;
            const double* rd = envdot + ((long long)i * slots + s) * 4;
            for (int a = 0; a < 4; ++a) acc[a] += gpm * rd[0] * r[a] + gm * rd[a];
        }
        for (int a = 0; a < 4; ++a) {
            bd[4 * m + a] = acc[a] / slots;
            bdot_out[((long long)i * width + m) * 4 + a] = acc[a] / slots;
            b[4 * m + a] = b_in[((long long)i * width + m) * 4 + a];
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < width * axis; e += blockDim.x) {
        int m = e / axis, k = e % axis;
        double v = 0.0;
        for (int a = 0; a < 4; ++a) v += bd[4 * m + a] * b[4 * k + a] + b[4 * m + a] * bd[4 * k + a];
        ddot_out[(long long)i * width * axis + e] = (P)v;
    }
}

// One block per atom: adjoints of B and of its tangent from the adjoints of
// D and of its tangent, then of every embedding row and of its derivative
// along the radial input. Empty slots take part: their rows hold the
// normalized zero, which is not zero.
template <typename P>
__device__ void descriptor_reverse(const double* env, const double* envdot, const double* b_in,
        const double* bdot_in, const P* dbar_in, const P* ddbar_in, const int* slot_type,
        const int* slot_local, const int* sel, const int* offset, int count, int slots, int width,
        int axis, P* gbar, P* gpbar) {
    extern __shared__ double shared[];
    double* b = shared;
    double* bd = b + 4 * width;
    double* bb = bd + 4 * width;
    double* bdb = bb + 4 * width;
    int i = blockIdx.x;
    for (int e = threadIdx.x; e < 4 * width; e += blockDim.x) {
        b[e] = b_in[(long long)i * width * 4 + e];
        bd[e] = bdot_in[(long long)i * width * 4 + e];
    }
    __syncthreads();
    const P* dbar = dbar_in + (long long)i * width * axis;
    const P* ddbar = ddbar_in + (long long)i * width * axis;
    for (int m = threadIdx.x; m < width; m += blockDim.x) {
        double value[4] = {0.0, 0.0, 0.0, 0.0}, tangent[4] = {0.0, 0.0, 0.0, 0.0};
        for (int k = 0; k < axis; ++k) {
            double w1 = (double)dbar[m * axis + k], w2 = (double)ddbar[m * axis + k];
            for (int a = 0; a < 4; ++a) {
                value[a] += w1 * b[4 * k + a] + w2 * bd[4 * k + a];
                tangent[a] += w2 * b[4 * k + a];
            }
        }
        if (m < axis)
            for (int n = 0; n < width; ++n) {
                double w1 = (double)dbar[n * axis + m], w2 = (double)ddbar[n * axis + m];
                for (int a = 0; a < 4; ++a) {
                    value[a] += w1 * b[4 * n + a] + w2 * bd[4 * n + a];
                    tangent[a] += w2 * b[4 * n + a];
                }
            }
        for (int a = 0; a < 4; ++a) {
            bb[4 * m + a] = value[a];
            bdb[4 * m + a] = tangent[a];
        }
    }
    __syncthreads();
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (int s = warp; s < slots; s += warps) {
        long long row = train_row(count, i, slot_type[s], slot_local[s], sel, offset) * width;
        const double* r = env + 4 * ((long long)i * slots + s);
        const double* rd = envdot + 4 * ((long long)i * slots + s);
        for (int m = lane; m < width; m += 32) {
            double v1 = 0.0, v2 = 0.0;
            for (int a = 0; a < 4; ++a) {
                v1 += bb[4 * m + a] * r[a] + bdb[4 * m + a] * rd[a];
                v2 += bdb[4 * m + a] * r[a];
            }
            gbar[row + m] = (P)(v1 / slots);
            gpbar[row + m] = (P)(rd[0] * v2 / slots);
        }
    }
}

// Adam with bias correction, as torch.optim.Adam.
template <typename P>
__device__ void adam(P* theta, const P* grad, P* m, P* v, long long n, P lr, P beta1, P beta2,
        P eps, P correction1, P correction2) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    P g = grad[idx];
    P mi = beta1 * m[idx] + (P(1) - beta1) * g;
    P vi = beta2 * v[idx] + (P(1) - beta2) * g * g;
    m[idx] = mi;
    v[idx] = vi;
    theta[idx] -= lr * (mi / correction1) / (sqrt(vi / correction2) + eps);
}

#define INSTANTIATE(P, S)                                                                         \
extern "C" __global__ void se_e2_a_descriptor_tangent_##S(const double* env,                      \
        const double* envdot, const P* g, const P* dg, const double* b_in, const int* slot_type,  \
        const int* slot_local, const int* sel, const int* offset, int count, int slots,           \
        int width, int axis, double* bdot_out, P* ddot_out) {                                     \
    descriptor_tangent<P>(env, envdot, g, dg, b_in, slot_type, slot_local, sel, offset, count,    \
                          slots, width, axis, bdot_out, ddot_out);                                \
}                                                                                                 \
extern "C" __global__ void se_e2_a_descriptor_reverse_##S(const double* env,                      \
        const double* envdot, const double* b_in, const double* bdot_in, const P* dbar,           \
        const P* ddbar, const int* slot_type, const int* slot_local, const int* sel,              \
        const int* offset, int count, int slots, int width, int axis, P* gbar, P* gpbar) {        \
    descriptor_reverse<P>(env, envdot, b_in, bdot_in, dbar, ddbar, slot_type, slot_local, sel,    \
                          offset, count, slots, width, axis, gbar, gpbar);                        \
}                                                                                                 \
extern "C" __global__ void adam_##S(P* theta, const P* grad, P* m, P* v, long long n, P lr,       \
        P beta1, P beta2, P eps, P correction1, P correction2) {                                  \
    adam<P>(theta, grad, m, v, n, lr, beta1, beta2, eps, correction1, correction2);               \
}
INSTANTIATE(float, f32)
INSTANTIATE(double, f64)
"#;

const BLOCK: u32 = 128;
const SHARED_LIMIT: usize = 48 * 1024;

/// Loss prefactors of the current step (DeePMD's `pref_e`, `pref_f`, `pref_v`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LossWeights {
    pub energy: f64,
    pub force: f64,
    pub virial: f64,
}

/// Labels of one frame; a missing label drops its loss term.
#[derive(Clone, Copy, Debug)]
pub struct FrameLabels<'a> {
    pub energy: Option<f64>,
    pub forces: Option<&'a [[f64; 3]]>,
    pub virial: Option<[[f64; 3]; 3]>,
}

/// The loss of one frame and its parts.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
pub struct LossTerms {
    pub loss: f64,
    pub natoms: usize,
    /// `(E − E*)²`, `mean((F − F*)²)` and `mean((V − V*)²)`; zero without
    /// the label.
    pub l2_energy: f64,
    pub l2_force: f64,
    pub l2_virial: f64,
}

impl LossTerms {
    /// Energy RMSE per atom, as DeePMD's `rmse_e`.
    pub fn rmse_energy_per_atom(&self) -> f64 {
        self.l2_energy.sqrt() / self.natoms as f64
    }

    pub fn rmse_force(&self) -> f64 {
        self.l2_force.sqrt()
    }

    /// Virial RMSE per atom, as DeePMD's `rmse_v`.
    pub fn rmse_virial_per_atom(&self) -> f64 {
        self.l2_virial.sqrt() / self.natoms as f64
    }
}

struct Kernels {
    env: CudaFunction,
    descriptor: CudaFunction,
    force: CudaFunction,
    env_tangent: CudaFunction,
    descriptor_tangent: CudaFunction,
    descriptor_reverse: CudaFunction,
    adam: CudaFunction,
}

/// Adam's step count and moment estimates on the host, for checkpoints.
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizerState {
    /// Completed updates.
    pub step: u64,
    /// First and second moment estimates, in the layout of
    /// [`GpuTrainer::parameters`].
    pub m: Vec<f64>,
    pub v: Vec<f64>,
}

/// Adam moments and step count.
struct AdamState<P: Scalar> {
    m: CudaSlice<P>,
    v: CudaSlice<P>,
    step: u64,
}

pub struct GpuTrainer<P: Scalar> {
    device: Arc<Device>,
    metadata: ModelMetadata,
    /// Structure, normalization and bias; parameters are refreshed on export.
    model: Model,
    slots: usize,
    width: usize,
    axis: usize,
    theta: CudaSlice<P>,
    gradient: CudaSlice<P>,
    adam: Option<AdamState<P>>,
    /// Net `center type · ntypes + neighbor type`.
    embedding: Vec<TrainNetwork<P>>,
    fitting: Vec<TrainNetwork<P>>,
    davg: CudaSlice<f64>,
    dstd: CudaSlice<f64>,
    slot_type: CudaSlice<i32>,
    slot_local: CudaSlice<i32>,
    sel_d: CudaSlice<i32>,
    offset_d: CudaSlice<i32>,
    kernels: Kernels,
}

fn device_activation(activation: Activation) -> DeviceActivation {
    match activation {
        Activation::Linear => DeviceActivation::Linear,
        Activation::Tanh => DeviceActivation::Tanh,
        Activation::Sigmoid => DeviceActivation::Sigmoid,
        Activation::Softplus => DeviceActivation::Softplus,
    }
}

/// Appends a network's parameters to `theta` and returns its layout.
fn pack(network: &Network, mut input: usize, theta: &mut Vec<f64>) -> Vec<TrainLayer> {
    network
        .layers
        .iter()
        .map(|layer| {
            let output = layer.bias.len();
            let weights = theta.len();
            theta.extend(layer.weights.iter().flatten());
            let bias = theta.len();
            theta.extend(&layer.bias);
            let timestep = layer.timestep.as_ref().map(|t| {
                let offset = theta.len();
                theta.extend(t);
                offset
            });
            let slots = TrainLayer {
                input,
                output,
                activation: device_activation(layer.activation),
                residual: !matches!(layer.residual, Residual::None),
                weights,
                bias,
                timestep,
            };
            input = output;
            slots
        })
        .collect()
}

/// Writes a network's parameters from `theta` back into its layers.
fn unpack(layers: &mut [Layer], layout: &[TrainLayer], theta: &[f64]) {
    for (layer, slots) in layers.iter_mut().zip(layout) {
        for (i, row) in layer.weights.iter_mut().enumerate() {
            row.copy_from_slice(&theta[slots.weights + i * slots.output..][..slots.output]);
        }
        layer
            .bias
            .copy_from_slice(&theta[slots.bias..][..slots.output]);
        if let (Some(timestep), Some(offset)) = (&mut layer.timestep, slots.timestep) {
            timestep.copy_from_slice(&theta[offset..][..slots.output]);
        }
    }
}

impl<P: Scalar> GpuTrainer<P> {
    pub fn new(model: &Model, device: Arc<Device>) -> Result<Self> {
        let metadata = model.validate()?;
        let d = &model.descriptor;
        let ntypes = model.type_map.len();
        let width = d.embedding[0][0].output_size();
        let slots: usize = d.sel.iter().sum();
        if (16 * width + 12 + 4 * slots) * 8 + 4 * width * std::mem::size_of::<P>() > SHARED_LIMIT {
            return Err(Error(format!(
                "GPU se_e2_a training does not fit shared memory with embedding width {width} and {slots} neighbor slots"
            )));
        }
        let mut theta = Vec::new();
        let mut embedding = Vec::with_capacity(ntypes * ntypes);
        for network in d.embedding.iter().flatten() {
            embedding.push(TrainNetwork::new(&device, pack(network, 1, &mut theta))?);
        }
        let mut fitting = Vec::with_capacity(ntypes);
        for network in &model.fitting {
            let layers = pack(network, width * d.axis_neuron, &mut theta);
            if layers.last().map(|l| l.output) != Some(1) {
                return Err(Error("fitting networks must end in one output".into()));
            }
            fitting.push(TrainNetwork::new(&device, layers)?);
        }
        let davg: Vec<f64> = d.normalization.iter().flat_map(|n| n.mean).collect();
        let dstd: Vec<f64> = d.normalization.iter().flat_map(|n| n.std).collect();
        let (mut slot_type, mut slot_local, mut offsets) = (Vec::new(), Vec::new(), Vec::new());
        for (t, &count) in d.sel.iter().enumerate() {
            offsets.push(slot_type.len() as i32);
            slot_type.extend(std::iter::repeat_n(t as i32, count));
            slot_local.extend(0..count as i32);
        }
        let sel_i: Vec<i32> = d.sel.iter().map(|&s| s as i32).collect();
        let model_kernel = |name: &str| {
            device
                .kernels()
                .function(MODEL_SOURCE, &format!("{name}_{}", P::SUFFIX))
        };
        let train_kernel = |name: &str| device.kernels().function(SOURCE, name);
        let kernels = Kernels {
            env: model_kernel("se_e2_a_env")?,
            descriptor: model_kernel("se_e2_a_descriptor")?,
            force: model_kernel("se_e2_a_backward")?,
            env_tangent: train_kernel("se_e2_a_env_tangent")?,
            descriptor_tangent: train_kernel(&format!("se_e2_a_descriptor_tangent_{}", P::SUFFIX))?,
            descriptor_reverse: train_kernel(&format!("se_e2_a_descriptor_reverse_{}", P::SUFFIX))?,
            adam: train_kernel(&format!("adam_{}", P::SUFFIX))?,
        };
        let cast: Vec<P> = theta.iter().map(|&v| P::from_f64(v)).collect();
        Ok(Self {
            metadata,
            model: model.clone(),
            slots,
            width,
            axis: d.axis_neuron,
            gradient: device.zeros::<P>(cast.len())?,
            theta: device.upload(&cast)?,
            adam: None,
            embedding,
            fitting,
            davg: device.upload(&davg)?,
            dstd: device.upload(&dstd)?,
            slot_type: device.upload(&slot_type)?,
            slot_local: device.upload(&slot_local)?,
            sel_d: device.upload(&sel_i)?,
            offset_d: device.upload(&offsets)?,
            kernels,
            device,
        })
    }

    /// Number of trainable parameters (weights, biases and timesteps).
    pub fn parameter_count(&self) -> usize {
        self.theta.len()
    }

    /// The parameters, embedding nets first (center type major), then the
    /// fitting nets; per layer weights `[input][output]`, bias, timestep.
    pub fn parameters(&self) -> Result<Vec<f64>> {
        Ok(self
            .device
            .download(&self.theta)?
            .into_iter()
            .map(P::to_f64)
            .collect())
    }

    pub fn set_parameters(&mut self, parameters: &[f64]) -> Result<()> {
        if parameters.len() != self.theta.len() || parameters.iter().any(|v| !v.is_finite()) {
            return Err(Error(format!(
                "expected {} finite parameters, got {}",
                self.theta.len(),
                parameters.len()
            )));
        }
        let cast: Vec<P> = parameters.iter().map(|&v| P::from_f64(v)).collect();
        self.theta = self.device.upload(&cast)?;
        Ok(())
    }

    /// The model with the current parameters.
    pub fn model(&self) -> Result<Model> {
        let theta = self.parameters()?;
        let mut model = self.model.clone();
        let networks = model
            .descriptor
            .embedding
            .iter_mut()
            .flatten()
            .zip(&self.embedding)
            .chain(model.fitting.iter_mut().zip(&self.fitting));
        for (network, layout) in networks {
            unpack(&mut network.layers, layout.layers(), &theta);
        }
        model.validate()?;
        Ok(model)
    }

    pub fn zero_gradient(&mut self) -> Result<()> {
        self.gradient = self.device.zeros::<P>(self.theta.len())?;
        Ok(())
    }

    /// The accumulated loss gradient, in the layout of [`Self::parameters`].
    pub fn gradient(&self) -> Result<Vec<f64>> {
        Ok(self
            .device
            .download(&self.gradient)?
            .into_iter()
            .map(P::to_f64)
            .collect())
    }

    /// One Adam update from the accumulated gradient (`torch.optim.Adam`
    /// with betas 0.9 and 0.999, epsilon 1e-8, no weight decay).
    pub fn adam_step(&mut self, learning_rate: f64) -> Result<()> {
        if !(learning_rate.is_finite() && learning_rate >= 0.0) {
            return Err(Error(format!(
                "learning rate must be finite and nonnegative, got {learning_rate}"
            )));
        }
        let n = self.theta.len();
        if self.adam.is_none() {
            self.adam = Some(AdamState {
                m: self.device.zeros::<P>(n)?,
                v: self.device.zeros::<P>(n)?,
                step: 0,
            });
        }
        let state = self.adam.as_mut().unwrap();
        state.step += 1;
        let (beta1, beta2) = (0.9_f64, 0.999_f64);
        let scalars = [
            learning_rate,
            beta1,
            beta2,
            1e-8,
            1.0 - beta1.powf(state.step as f64),
            1.0 - beta2.powf(state.step as f64),
        ]
        .map(P::from_f64);
        let n_i = n as i64;
        let stream = self.device.stream();
        let mut launch = stream.launch_builder(&self.kernels.adam);
        launch
            .arg(&mut self.theta)
            .arg(&self.gradient)
            .arg(&mut state.m)
            .arg(&mut state.v)
            .arg(&n_i);
        for scalar in &scalars {
            launch.arg(scalar);
        }
        let config = LaunchConfig {
            grid_dim: (n.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { launch.launch(config) }
            .map(drop)
            .map_err(launch_error("Adam"))
    }

    /// Completed Adam updates.
    pub fn optimizer_steps(&self) -> u64 {
        self.adam.as_ref().map_or(0, |state| state.step)
    }

    /// The optimizer state (all zeros before the first update).
    pub fn optimizer_state(&self) -> Result<OptimizerState> {
        let widen = |values: Vec<P>| values.into_iter().map(P::to_f64).collect();
        Ok(match &self.adam {
            Some(state) => OptimizerState {
                step: state.step,
                m: widen(self.device.download(&state.m)?),
                v: widen(self.device.download(&state.v)?),
            },
            None => OptimizerState {
                step: 0,
                m: vec![0.0; self.theta.len()],
                v: vec![0.0; self.theta.len()],
            },
        })
    }

    /// Continues from a saved optimizer state.
    pub fn set_optimizer_state(&mut self, state: &OptimizerState) -> Result<()> {
        let n = self.theta.len();
        if state.m.len() != n
            || state.v.len() != n
            || state.m.iter().any(|x| !x.is_finite())
            || state.v.iter().any(|x| !(x.is_finite() && *x >= 0.0))
        {
            return Err(Error(format!(
                "optimizer state needs {n} finite first moments and nonnegative second moments"
            )));
        }
        let narrow = |values: &[f64]| values.iter().map(|&x| P::from_f64(x)).collect::<Vec<P>>();
        self.adam = Some(AdamState {
            m: self.device.upload(&narrow(&state.m))?,
            v: self.device.upload(&narrow(&state.v))?,
            step: state.step,
        });
        Ok(())
    }

    /// Evaluates one frame, adds `scale` times the gradient of its loss to
    /// the accumulated gradient, and returns the loss terms (unscaled) and
    /// the prediction. `scale` is 1 for single-frame batches and `1/B` for a
    /// batch of `B` frames.
    pub fn accumulate(
        &mut self,
        frame: &Frame,
        labels: &FrameLabels,
        weights: &LossWeights,
        scale: f64,
    ) -> Result<(LossTerms, Evaluation)> {
        frame.validate()?;
        let device = Arc::clone(&self.device);
        let device = &*device;
        let stream = device.stream();
        let types = frame.type_ids(&self.metadata)?;
        let natoms = types.len();
        if labels.forces.is_some_and(|f| f.len() != natoms) {
            return Err(Error("force labels do not match the atom count".into()));
        }
        let d = &self.model.descriptor;
        let (ntypes, slots, width, axis) = (d.sel.len(), self.slots, self.width, self.axis);
        let features = width * axis;

        // Atoms sorted by type, so every network sees contiguous rows.
        let mut order: Vec<usize> = (0..natoms).collect();
        order.sort_by_key(|&i| types[i]);
        let sorted_types: Vec<i32> = order.iter().map(|&i| types[i]).collect();
        let positions: Vec<[f64; 3]> = order.iter().map(|&i| frame.positions[i]).collect();
        let mut block = vec![0usize; ntypes + 1];
        for &t in &sorted_types {
            block[t as usize + 1] += 1;
        }
        for t in 0..ntypes {
            block[t + 1] += block[t];
        }
        let cell = frame.cell.map_or(Cell::Open, Cell::Periodic);
        let atype = device.upload(&sorted_types)?;
        let list = NeighborList::build(device, &positions, &sorted_types, &cell, d.rcut, &d.sel)?;

        let rows_total = natoms * slots;
        let (n_i, slots_i, w_i, axis_i) = (natoms as i32, slots as i32, width as i32, axis as i32);
        let (ntypes_i, zero_i) = (ntypes as i32, 0i32);
        let per_slot = LaunchConfig {
            grid_dim: (rows_total.div_ceil(BLOCK as usize) as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let per_atom = |shared: usize| LaunchConfig {
            grid_dim: (natoms as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: shared as u32,
        };
        // SAFETY (all `alloc_uninit` below): each buffer is fully written by
        // the kernel or copy that follows before anything reads it.
        let mut env = unsafe { device.alloc_uninit::<f64>(rows_total * 4)? };
        let mut radial = unsafe { device.alloc_uninit::<P>(rows_total)? };
        let mut network = unsafe { device.alloc_uninit::<i32>(rows_total)? };
        {
            let mut launch = stream.launch_builder(&self.kernels.env);
            launch
                .arg(&list.index)
                .arg(&list.delta)
                .arg(&atype)
                .arg(&self.davg)
                .arg(&self.dstd)
                .arg(&self.slot_type)
                .arg(&self.slot_local)
                .arg(&self.sel_d)
                .arg(&self.offset_d)
                .arg(&zero_i)
                .arg(&n_i)
                .arg(&slots_i)
                .arg(&ntypes_i)
                .arg(&d.rcut_smth)
                .arg(&d.rcut)
                .arg(&mut env)
                .arg(&mut radial)
                .arg(&mut network);
            unsafe { launch.launch(per_slot) }
                .map(drop)
                .map_err(launch_error("environment"))?;
        }

        // Embedding with the unit tangent: G and dG/d(radial input), one net
        // per (center type, neighbor type) block of rows.
        let ones = device.upload(&vec![P::ONE; rows_total.max(natoms)])?;
        let mut g = unsafe { device.alloc_uninit::<P>(rows_total * width)? };
        let mut dg = unsafe { device.alloc_uninit::<P>(rows_total * width)? };
        // (net, first row, rows) of every nonempty block.
        let mut blocks = Vec::new();
        let mut start = 0;
        for (tj, &sel_j) in d.sel.iter().enumerate() {
            for ti in 0..ntypes {
                let rows = (block[ti + 1] - block[ti]) * sel_j;
                if rows > 0 {
                    blocks.push((ti * ntypes + tj, start + block[ti] * sel_j, rows));
                }
            }
            start += natoms * sel_j;
        }
        let mut embedding_tapes: Vec<TrainTape<P>> = Vec::with_capacity(blocks.len());
        for &(net, first, rows) in &blocks {
            let tape = self.embedding[net].forward(
                device,
                &self.theta,
                &radial.slice(first..first + rows),
                &ones.slice(0..rows),
                rows,
            )?;
            stream
                .memcpy_dtod(
                    tape.output(),
                    &mut g.slice_mut(first * width..(first + rows) * width),
                )
                .map_err(copy_error)?;
            stream
                .memcpy_dtod(
                    tape.tangent(),
                    &mut dg.slice_mut(first * width..(first + rows) * width),
                )
                .map_err(copy_error)?;
            embedding_tapes.push(tape);
        }

        let mut b = unsafe { device.alloc_uninit::<f64>(natoms * width * 4)? };
        let mut descriptor = unsafe { device.alloc_uninit::<P>(natoms * features)? };
        {
            let mut launch = stream.launch_builder(&self.kernels.descriptor);
            launch
                .arg(&env)
                .arg(&g)
                .arg(&self.slot_type)
                .arg(&self.slot_local)
                .arg(&self.sel_d)
                .arg(&self.offset_d)
                .arg(&n_i)
                .arg(&slots_i)
                .arg(&w_i)
                .arg(&axis_i)
                .arg(&mut b)
                .arg(&mut descriptor);
            unsafe { launch.launch(per_atom(4 * width * 8)) }
                .map(drop)
                .map_err(launch_error("descriptor"))?;
        }

        // First fitting pass: atomic energies and dE/dD for forces and virial.
        let type_ranges: Vec<(usize, usize, usize)> = (0..ntypes)
            .filter(|&t| block[t + 1] > block[t])
            .map(|t| (t, block[t], block[t + 1]))
            .collect();
        let zeros = device.zeros::<P>(natoms * features)?;
        let mut d_descriptor = unsafe { device.alloc_uninit::<P>(natoms * features)? };
        let mut scratch = unsafe { device.alloc_uninit::<P>(natoms * features)? };
        let mut atomic = unsafe { device.alloc_uninit::<P>(natoms)? };
        for &(t, lo, hi) in &type_ranges {
            let rows = hi - lo;
            let x = descriptor.slice(lo * features..hi * features);
            let xdot = zeros.slice(lo * features..hi * features);
            let tape = self.fitting[t].forward(device, &self.theta, &x, &xdot, rows)?;
            self.fitting[t].backward_to_input(
                device,
                &self.theta,
                &mut self.gradient,
                false,
                &tape,
                &x,
                &xdot,
                &ones.slice(0..rows),
                &zeros.slice(0..rows),
                &ones,
                &mut d_descriptor.slice_mut(lo * features..hi * features),
                &mut scratch.slice_mut(lo * features..hi * features),
            )?;
            stream
                .memcpy_dtod(tape.output(), &mut atomic.slice_mut(lo..hi))
                .map_err(copy_error)?;
        }
        let mut force = device.zeros::<f64>(natoms * 3)?;
        let mut virial = unsafe { device.alloc_uninit::<f64>(natoms * 9)? };
        {
            let mut launch = stream.launch_builder(&self.kernels.force);
            launch
                .arg(&env)
                .arg(&g)
                .arg(&dg)
                .arg(&b)
                .arg(&d_descriptor)
                .arg(&list.index)
                .arg(&list.delta)
                .arg(&atype)
                .arg(&self.dstd)
                .arg(&self.slot_type)
                .arg(&self.slot_local)
                .arg(&self.sel_d)
                .arg(&self.offset_d)
                .arg(&zero_i)
                .arg(&n_i)
                .arg(&slots_i)
                .arg(&w_i)
                .arg(&axis_i)
                .arg(&d.rcut_smth)
                .arg(&d.rcut)
                .arg(&mut force)
                .arg(&mut virial);
            let shared = (4 * width + 4 * slots + 12) * 8 + 4 * width * std::mem::size_of::<P>();
            unsafe { launch.launch(per_atom(shared)) }
                .map(drop)
                .map_err(launch_error("force"))?;
        }
        list.check(device)?;

        // Prediction, loss and the seeds of the gradient pass (host, float64).
        let atomic_host = device.download(&atomic)?;
        let mut energy = 0.0;
        for t in 0..ntypes {
            let range = &atomic_host[block[t]..block[t + 1]];
            energy += range.iter().map(|&e| e.to_f64()).sum::<f64>()
                + self.model.atom_energy_bias[t] * range.len() as f64;
        }
        let sorted_forces = device.download(&force)?;
        let mut total_virial = [[0.0; 3]; 3];
        for row in device.download(&virial)?.chunks_exact(9) {
            for (e, v) in row.iter().enumerate() {
                total_virial[e / 3][e % 3] += v;
            }
        }
        let n = natoms as f64;
        let mut terms = LossTerms {
            natoms,
            ..LossTerms::default()
        };
        let mut energy_seed = 0.0;
        if let Some(label) = labels.energy {
            terms.l2_energy = (energy - label).powi(2);
            energy_seed = scale * 2.0 * weights.energy * (energy - label) / n;
        }
        // c in sorted atom order.
        let mut force_seed = vec![0.0; natoms * 3];
        if let Some(label) = labels.forces {
            let factor = scale * 2.0 * weights.force / (3.0 * n);
            for (k, &atom) in order.iter().enumerate() {
                for a in 0..3 {
                    let difference = sorted_forces[3 * k + a] - label[atom][a];
                    terms.l2_force += difference * difference / (3.0 * n);
                    force_seed[3 * k + a] = factor * difference;
                }
            }
        }
        let mut virial_seed = [0.0; 9];
        if let Some(label) = labels.virial {
            let factor = scale * 2.0 * weights.virial / (9.0 * n);
            for e in 0..9 {
                let difference = total_virial[e / 3][e % 3] - label[e / 3][e % 3];
                terms.l2_virial += difference * difference / 9.0;
                virial_seed[e] = factor * difference;
            }
        }
        terms.loss = weights.energy * terms.l2_energy / n
            + weights.force * terms.l2_force
            + weights.virial * terms.l2_virial / n;
        let mut forces = vec![[0.0; 3]; natoms];
        for (k, &atom) in order.iter().enumerate() {
            forces[atom] = [
                sorted_forces[3 * k],
                sorted_forces[3 * k + 1],
                sorted_forces[3 * k + 2],
            ];
        }
        let evaluation = Evaluation {
            energy,
            forces,
            virial: total_virial,
            units: Units::EvAngstrom,
        };
        evaluation.validate(natoms)?;

        // Tangent pass: environment rows, B and D along the seed direction.
        let c = device.upload(&force_seed)?;
        let w = device.upload(&virial_seed)?;
        let mut envdot = unsafe { device.alloc_uninit::<f64>(rows_total * 4)? };
        {
            let mut launch = stream.launch_builder(&self.kernels.env_tangent);
            launch
                .arg(&list.index)
                .arg(&list.delta)
                .arg(&atype)
                .arg(&self.dstd)
                .arg(&c)
                .arg(&w)
                .arg(&n_i)
                .arg(&slots_i)
                .arg(&d.rcut_smth)
                .arg(&d.rcut)
                .arg(&mut envdot);
            unsafe { launch.launch(per_slot) }
                .map(drop)
                .map_err(launch_error("environment tangent"))?;
        }
        let mut bdot = unsafe { device.alloc_uninit::<f64>(natoms * width * 4)? };
        let mut ddot = unsafe { device.alloc_uninit::<P>(natoms * features)? };
        {
            let mut launch = stream.launch_builder(&self.kernels.descriptor_tangent);
            launch
                .arg(&env)
                .arg(&envdot)
                .arg(&g)
                .arg(&dg)
                .arg(&b)
                .arg(&self.slot_type)
                .arg(&self.slot_local)
                .arg(&self.sel_d)
                .arg(&self.offset_d)
                .arg(&n_i)
                .arg(&slots_i)
                .arg(&w_i)
                .arg(&axis_i)
                .arg(&mut bdot)
                .arg(&mut ddot);
            unsafe { launch.launch(per_atom(8 * width * 8)) }
                .map(drop)
                .map_err(launch_error("descriptor tangent"))?;
        }

        // Fitting with the tangent, then its reverse pass with seeds
        // dΦ/dE = a and dΦ/dĖ = −1.
        let energy_bar = device.upload(&vec![P::from_f64(energy_seed); natoms])?;
        let tangent_bar = device.upload(&vec![P::from_f64(-1.0); natoms])?;
        let mut dbar = unsafe { device.alloc_uninit::<P>(natoms * features)? };
        let mut ddbar = unsafe { device.alloc_uninit::<P>(natoms * features)? };
        for &(t, lo, hi) in &type_ranges {
            let rows = hi - lo;
            let x = descriptor.slice(lo * features..hi * features);
            let xdot = ddot.slice(lo * features..hi * features);
            let tape = self.fitting[t].forward(device, &self.theta, &x, &xdot, rows)?;
            self.fitting[t].backward_to_input(
                device,
                &self.theta,
                &mut self.gradient,
                true,
                &tape,
                &x,
                &xdot,
                &energy_bar.slice(0..rows),
                &tangent_bar.slice(0..rows),
                &ones,
                &mut dbar.slice_mut(lo * features..hi * features),
                &mut ddbar.slice_mut(lo * features..hi * features),
            )?;
        }

        // Back through the descriptor to every embedding row, then through
        // the embedding nets.
        let mut gbar = unsafe { device.alloc_uninit::<P>(rows_total * width)? };
        let mut gpbar = unsafe { device.alloc_uninit::<P>(rows_total * width)? };
        {
            let mut launch = stream.launch_builder(&self.kernels.descriptor_reverse);
            launch
                .arg(&env)
                .arg(&envdot)
                .arg(&b)
                .arg(&bdot)
                .arg(&dbar)
                .arg(&ddbar)
                .arg(&self.slot_type)
                .arg(&self.slot_local)
                .arg(&self.sel_d)
                .arg(&self.offset_d)
                .arg(&n_i)
                .arg(&slots_i)
                .arg(&w_i)
                .arg(&axis_i)
                .arg(&mut gbar)
                .arg(&mut gpbar);
            unsafe { launch.launch(per_atom(16 * width * 8)) }
                .map(drop)
                .map_err(launch_error("descriptor reverse"))?;
        }
        for (&(net, first, rows), tape) in blocks.iter().zip(&embedding_tapes) {
            self.embedding[net].backward(
                device,
                &self.theta,
                &mut self.gradient,
                tape,
                &radial.slice(first..first + rows),
                &ones.slice(0..rows),
                &gbar.slice(first * width..(first + rows) * width),
                &gpbar.slice(first * width..(first + rows) * width),
                &ones,
            )?;
        }
        Ok((terms, evaluation))
    }
}
