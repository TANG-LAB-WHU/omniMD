//! se_e2_a inference on the omnimd-gpu runtime (plan W6–W7).
//!
//! Pipeline per frame, all on the device after one upload:
//! cell-list neighbors (DeePMD truncation to `sel`) → environment matrix →
//! embedding nets per (center type, neighbor type), fused with their tangents →
//! descriptor `D = B·B[:, :axis]ᵀ` → fitting nets per center type with a
//! recorded reverse pass → forces and virial from the chain rule back to the
//! neighbor deltas. Atoms are sorted by type so every network sees
//! contiguous rows; results are returned in the caller's atom order.
//!
//! `P` is the network precision (ADR-3). Geometry, the environment matrix,
//! the `B` accumulation, forces and virial are always float64; with `P = f32`
//! the embedding and fitting nets (almost all of the arithmetic) run in
//! float32, like a DeePMD float32 model but with float64 reductions.
//!
//! Unlike the CPU `NativePotential`, which rejects frames with more
//! neighbors than `sel`, this path keeps the nearest `sel` like DeePMD does;
//! [`GpuPotential::last_truncated`] reports when that happened.
use crate::{Activation, Model, Network, Residual};
use omnimd_gpu::{
    Activation as DeviceActivation, Cell, CudaFunction, CudaSlice, DenseLayer, DenseNetwork,
    Device, LaunchConfig, LayerParameters, NeighborList, PushKernelArg, Scalar, ScalarNetwork,
};
use omnimd_ml_contract::{Error, Evaluation, Frame, ModelMetadata, Potential, Result, Units};
use std::sync::Arc;
use std::time::Instant;

const SOURCE: &str = r#"
__device__ void switch_fn(double r, double rmin, double rmax, double* q, double* dq) {
    if (r < rmin) { *q = 1.0; *dq = 0.0; return; }
    double t = (r - rmin) / (rmax - rmin), z = 1.0 - t;
    *q = z * z * z * (1.0 + 3.0 * t + 6.0 * t * t);
    *dq = -30.0 * t * t * z * z / (rmax - rmin);
}

__device__ long long embed_row(int count, int i, int t, int local, const int* sel, const int* offset) {
    return (long long)count * offset[t] + (long long)i * sel[t] + local;
}

template <typename P>
__device__ void env_impl(const int* index, const double* delta, const int* atype, const double* davg,
        const double* dstd, const int* slot_type, const int* slot_local, const int* sel,
        const int* offset, int first, int count, int slots, double rmin, double rmax, double* env,
        P* radial) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)count * slots) return;
    int li = (int)(idx / slots), s = (int)(idx % slots), i = first + li, c = atype[i];
    long long slot = (long long)i * slots + s;
    double r4[4] = {0.0, 0.0, 0.0, 0.0};
    if (index[slot] >= 0) {
        const double* d = delta + 3 * slot;
        double r = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]), q, dq;
        switch_fn(r, rmin, rmax, &q, &dq);
        double sw = q / r;
        r4[0] = sw;
        for (int a = 0; a < 3; ++a) r4[a + 1] = sw * (d[a] / r);
    }
    for (int a = 0; a < 4; ++a) {
        r4[a] = (r4[a] - davg[4 * c + a]) / dstd[4 * c + a];
        env[4 * idx + a] = r4[a];
    }
    radial[embed_row(count, li, slot_type[s], slot_local[s], sel, offset)] = (P)r4[0];
}

// One block per atom: B[m][a] = sum_s G[s][m] env[s][a] / slots (float64),
// D = B·B[:axis]ᵀ in the network precision.
template <typename P>
__device__ void descriptor_impl(const double* env, const P* g, const int* slot_type,
        const int* slot_local, const int* sel, const int* offset, int count, int slots,
        int width, int axis, double* b_out, P* d_out) {
    extern __shared__ double b[];
    int i = blockIdx.x;
    for (int m = threadIdx.x; m < width; m += blockDim.x) {
        double acc[4] = {0.0, 0.0, 0.0, 0.0};
        for (int s = 0; s < slots; ++s) {
            double gm = (double)g[embed_row(count, i, slot_type[s], slot_local[s], sel, offset) * width + m];
            const double* r = env + ((long long)i * slots + s) * 4;
            for (int a = 0; a < 4; ++a) acc[a] += gm * r[a];
        }
        for (int a = 0; a < 4; ++a) {
            b[4 * m + a] = acc[a] / slots;
            b_out[((long long)i * width + m) * 4 + a] = acc[a] / slots;
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < width * axis; e += blockDim.x) {
        int m = e / axis, k = e % axis;
        double v = 0.0;
        for (int a = 0; a < 4; ++a) v += b[4 * m + a] * b[4 * k + a];
        d_out[(long long)i * width * axis + e] = (P)v;
    }
}

template <typename T>
__device__ T warp_sum(T v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

// One block per atom, three phases:
//  1. dB = dE/dB from the descriptor adjoint (float64), stored as P in a
//     [4][width] layout so the per-slot loop reads shared memory without
//     bank conflicts;
//  2. one warp per neighbor slot (lanes over the embedding width, so G rows
//     are read coalesced) contracts dB with G and dG/dr in the network
//     precision into dE/d(env row), kept in shared memory as float64;
//  3. one thread per slot applies the float64 environment Jacobian; only the
//     neighbor's force needs a global atomic, the center force and the
//     virial are reduced over the block first.
template <typename P>
__device__ void backward_impl(const double* env, const P* g, const P* dg, const double* b_in,
        const P* dd_in, const int* index, const double* delta, const int* atype,
        const double* dstd, const int* slot_type, const int* slot_local, const int* sel,
        const int* offset, int first, int count, int slots, int width, int axis, double rmin,
        double rmax, double* force, double* virial) {
    extern __shared__ double shared[];
    double* b = shared;                  // [width][4]
    double* dr_all = b + 4 * width;      // [slots][4]
    double* red = dr_all + 4 * slots;    // [12]: center force, virial
    P* db = (P*)(red + 12);              // [4][width]
    int li = blockIdx.x, i = first + li;
    for (int e = threadIdx.x; e < 4 * width; e += blockDim.x) b[e] = b_in[(long long)li * width * 4 + e];
    for (int e = threadIdx.x; e < 12; e += blockDim.x) red[e] = 0.0;
    __syncthreads();
    const P* dd = dd_in + (long long)li * width * axis;
    for (int m = threadIdx.x; m < width; m += blockDim.x) {
        double acc[4] = {0.0, 0.0, 0.0, 0.0};
        for (int k = 0; k < axis; ++k) {
            double w = (double)dd[m * axis + k];
            for (int a = 0; a < 4; ++a) acc[a] += w * b[4 * k + a];
        }
        if (m < axis)
            for (int n = 0; n < width; ++n) {
                double w = (double)dd[n * axis + m];
                for (int a = 0; a < 4; ++a) acc[a] += w * b[4 * n + a];
            }
        for (int a = 0; a < 4; ++a) db[a * width + m] = (P)acc[a];
    }
    __syncthreads();
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (int s = warp; s < slots; s += warps) {
        long long slot = (long long)i * slots + s;
        if (index[slot] < 0) continue;
        long long row = embed_row(count, li, slot_type[s], slot_local[s], sel, offset) * width;
        const double* r = env + 4 * ((long long)li * slots + s);
        P rp[4] = {(P)r[0], (P)r[1], (P)r[2], (P)r[3]};
        P dr[4] = {0, 0, 0, 0}, tangent = 0;
        for (int m = lane; m < width; m += 32) {
            P gm = g[row + m], projection = 0;
            for (int a = 0; a < 4; ++a) {
                P w = db[a * width + m];
                dr[a] += w * gm;
                projection += w * rp[a];
            }
            tangent += projection * dg[row + m];
        }
        for (int a = 0; a < 4; ++a) dr[a] = warp_sum(dr[a]);
        tangent = warp_sum(tangent);
        if (lane == 0) {
            for (int a = 0; a < 4; ++a) dr_all[4 * s + a] = (double)dr[a] / slots;
            dr_all[4 * s] += (double)tangent / slots;
        }
    }
    __syncthreads();
    const double* std_c = dstd + 4 * atype[i];
    double fi[3] = {0.0, 0.0, 0.0}, vir[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int s = threadIdx.x; s < slots; s += blockDim.x) {
        long long slot = (long long)i * slots + s;
        int j = index[slot];
        if (j < 0) continue;
        const double* dr = dr_all + 4 * s;
        const double* d = delta + 3 * slot;
        double rr = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]), q, dq;
        switch_fn(rr, rmin, rmax, &q, &dq);
        double u[3] = {d[0] / rr, d[1] / rr, d[2] / rr};
        double sw = q / rr, dsw = dq / rr - sw / rr;
        for (int c = 0; c < 3; ++c) {
            double grad = dr[0] / std_c[0] * dsw * u[c];
            for (int a = 0; a < 3; ++a) {
                double jac = dsw * u[a] * u[c] + sw / rr * ((a == c ? 1.0 : 0.0) - u[a] * u[c]);
                grad += dr[a + 1] / std_c[a + 1] * jac;
            }
            atomicAdd(&force[3 * (long long)j + c], -grad);
            fi[c] += grad;
            for (int a = 0; a < 3; ++a) vir[3 * a + c] -= d[a] * grad;
        }
    }
    for (int c = 0; c < 3; ++c) fi[c] = warp_sum(fi[c]);
    for (int e = 0; e < 9; ++e) vir[e] = warp_sum(vir[e]);
    if (lane == 0) {
        for (int c = 0; c < 3; ++c) atomicAdd(&red[c], fi[c]);
        for (int e = 0; e < 9; ++e) atomicAdd(&red[3 + e], vir[e]);
    }
    __syncthreads();
    for (int c = threadIdx.x; c < 3; c += blockDim.x) atomicAdd(&force[3 * i + c], red[c]);
    for (int e = threadIdx.x; e < 9; e += blockDim.x) virial[(long long)i * 9 + e] = red[3 + e];
}

#define INSTANTIATE(P, S)                                                                         \
extern "C" __global__ void se_e2_a_env_##S(const int* index, const double* delta,                 \
        const int* atype, const double* davg, const double* dstd, const int* slot_type,           \
        const int* slot_local, const int* sel, const int* offset, int first, int count,           \
        int slots, double rmin, double rmax, double* env, P* radial) {                            \
    env_impl<P>(index, delta, atype, davg, dstd, slot_type, slot_local, sel, offset, first,       \
                count, slots, rmin, rmax, env, radial);                                           \
}                                                                                                 \
extern "C" __global__ void se_e2_a_descriptor_##S(const double* env, const P* g,                  \
        const int* slot_type, const int* slot_local, const int* sel, const int* offset,           \
        int count, int slots, int width, int axis, double* b_out, P* d_out) {                     \
    descriptor_impl<P>(env, g, slot_type, slot_local, sel, offset, count, slots, width, axis,     \
                       b_out, d_out);                                                             \
}                                                                                                 \
extern "C" __global__ void se_e2_a_backward_##S(const double* env, const P* g, const P* dg,       \
        const double* b_in, const P* dd_in, const int* index, const double* delta,                \
        const int* atype, const double* dstd, const int* slot_type, const int* slot_local,        \
        const int* sel, const int* offset, int first, int count, int slots, int width, int axis,  \
        double rmin, double rmax, double* force, double* virial) {                                \
    backward_impl<P>(env, g, dg, b_in, dd_in, index, delta, atype, dstd, slot_type, slot_local,   \
                     sel, offset, first, count, slots, width, axis, rmin, rmax, force, virial);   \
}
INSTANTIATE(float, f32)
INSTANTIATE(double, f64)
"#;

const BLOCK: u32 = 128;
/// Static shared-memory limit per block without opting in to more.
const SHARED_LIMIT: usize = 48 * 1024;
/// Fraction of the free device memory one chunk of per-atom work may use.
const MEMORY_FRACTION: f64 = 0.5;

struct Kernels {
    env: CudaFunction,
    descriptor: CudaFunction,
    backward: CudaFunction,
}

/// Wall-clock milliseconds per stage of the last evaluation (summed over
/// chunks), measured with a device synchronization after each stage, so only
/// when profiling is on.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct StageTimes {
    pub host_prepare: f64,
    pub neighbors: f64,
    pub environment: f64,
    pub embedding: f64,
    pub descriptor: f64,
    pub fitting: f64,
    pub forces: f64,
    pub download: f64,
    pub chunks: usize,
}

pub struct GpuPotential<P: Scalar> {
    device: Arc<Device>,
    metadata: ModelMetadata,
    rcut: f64,
    rcut_smth: f64,
    sel: Vec<usize>,
    slots: usize,
    width: usize,
    axis: usize,
    bias: Vec<f64>,
    /// `[center type][neighbor type]`.
    embedding: Vec<Vec<ScalarNetwork<P>>>,
    fitting: Vec<DenseNetwork<P>>,
    davg: CudaSlice<f64>,
    dstd: CudaSlice<f64>,
    slot_type: CudaSlice<i32>,
    slot_local: CudaSlice<i32>,
    sel_d: CudaSlice<i32>,
    offset_d: CudaSlice<i32>,
    kernels: Kernels,
    /// Dynamic shared memory of the force kernel, bytes.
    backward_shared: usize,
    chunk_atoms: usize,
    last_truncated: bool,
    profile: Option<StageTimes>,
}

fn layer_parameters<P: Scalar>(network: &Network, mut input: usize) -> Vec<LayerParameters<P>> {
    let cast = |values: &[f64]| values.iter().map(|&v| P::from_f64(v)).collect::<Vec<P>>();
    network
        .layers
        .iter()
        .map(|layer| {
            let parameters = LayerParameters {
                input,
                output: layer.bias.len(),
                weights: cast(&layer.weights.concat()),
                bias: cast(&layer.bias),
                timestep: layer.timestep.as_deref().map(cast),
                activation: match layer.activation {
                    Activation::Linear => DeviceActivation::Linear,
                    Activation::Tanh => DeviceActivation::Tanh,
                    Activation::Sigmoid => DeviceActivation::Sigmoid,
                    Activation::Softplus => DeviceActivation::Softplus,
                },
                residual: !matches!(layer.residual, Residual::None),
            };
            input = parameters.output;
            parameters
        })
        .collect()
}

fn dense_network<P: Scalar>(
    device: &Device,
    network: &Network,
    input: usize,
) -> Result<DenseNetwork<P>> {
    let layers = layer_parameters::<P>(network, input)
        .iter()
        .map(|l| {
            DenseLayer::new(
                device,
                l.input,
                l.output,
                &l.weights,
                &l.bias,
                l.timestep.as_deref(),
                l.activation,
                l.residual,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    DenseNetwork::new(layers)
}

fn launch_error(what: &'static str) -> impl FnOnce(omnimd_gpu::DriverError) -> Error {
    move |e| Error(format!("CUDA {what} launch failed: {e:?}"))
}

fn copy_error(e: omnimd_gpu::DriverError) -> Error {
    Error(format!("CUDA copy failed: {e:?}"))
}

/// Stage clock: a no-op unless profiling, so production runs never sync.
struct Clock<'a> {
    device: &'a Device,
    times: Option<StageTimes>,
    last: Instant,
}

impl Clock<'_> {
    fn lap(&mut self, stage: impl FnOnce(&mut StageTimes) -> &mut f64) -> Result<()> {
        if let Some(times) = &mut self.times {
            self.device.synchronize()?;
            let now = Instant::now();
            *stage(times) += (now - self.last).as_secs_f64() * 1e3;
            self.last = now;
        }
        Ok(())
    }
}

/// Device bytes per center atom in one chunk: environment, radial input,
/// embedding values and tangents, B, descriptor and its adjoint, fitting tapes.
fn bytes_per_atom<P>(slots: usize, width: usize, axis: usize, fitting_width: usize) -> usize {
    let p = std::mem::size_of::<P>();
    slots * (4 * 8 + p + 2 * width * p)
        + width * 4 * 8
        + 2 * width * axis * p
        + 4 * fitting_width * p
}

impl<P: Scalar> GpuPotential<P> {
    pub fn new(model: &Model, device: Arc<Device>) -> Result<Self> {
        let metadata = model.validate()?;
        let d = &model.descriptor;
        let ntypes = model.type_map.len();
        let width = d.embedding[0][0].output_size();
        let slots: usize = d.sel.iter().sum();
        let backward_shared =
            (4 * width + 4 * slots + 12) * 8 + 4 * width * std::mem::size_of::<P>();
        if backward_shared > SHARED_LIMIT {
            return Err(Error(format!(
                "GPU se_e2_a needs {backward_shared} bytes of shared memory per atom                  (embedding width {width}, {slots} neighbor slots); the limit is {SHARED_LIMIT}"
            )));
        }
        let embedding = d
            .embedding
            .iter()
            .map(|row| {
                row.iter()
                    .map(|n| ScalarNetwork::new(&device, &layer_parameters(n, 1)))
                    .collect()
            })
            .collect::<Result<Vec<Vec<_>>>>()?;
        let fitting = model
            .fitting
            .iter()
            .map(|n| dense_network(&device, n, width * d.axis_neuron))
            .collect::<Result<Vec<_>>>()?;
        let davg: Vec<f64> = d.normalization.iter().flat_map(|n| n.mean).collect();
        let dstd: Vec<f64> = d.normalization.iter().flat_map(|n| n.std).collect();
        let mut slot_type = Vec::new();
        let mut slot_local = Vec::new();
        let mut offsets = Vec::with_capacity(ntypes);
        for (t, &count) in d.sel.iter().enumerate() {
            offsets.push(slot_type.len() as i32);
            slot_type.extend(std::iter::repeat_n(t as i32, count));
            slot_local.extend(0..count as i32);
        }
        let sel_i: Vec<i32> = d.sel.iter().map(|&s| s as i32).collect();
        let function = |name: &str| {
            device
                .kernels()
                .function(SOURCE, &format!("{name}_{}", P::SUFFIX))
        };
        let kernels = Kernels {
            env: function("se_e2_a_env")?,
            descriptor: function("se_e2_a_descriptor")?,
            backward: function("se_e2_a_backward")?,
        };
        let fitting_width = model
            .fitting
            .iter()
            .flat_map(|n| n.layers.iter().map(|l| l.bias.len()))
            .sum::<usize>();
        let (free, _) = device.memory_info()?;
        let per_atom = bytes_per_atom::<P>(slot_type.len(), width, d.axis_neuron, fitting_width);
        let chunk_atoms = ((free as f64 * MEMORY_FRACTION) as usize / per_atom.max(1)).max(1);
        Ok(Self {
            metadata,
            rcut: d.rcut,
            rcut_smth: d.rcut_smth,
            sel: d.sel.clone(),
            slots: slot_type.len(),
            width,
            axis: d.axis_neuron,
            bias: model.atom_energy_bias.clone(),
            embedding,
            fitting,
            davg: device.upload(&davg)?,
            dstd: device.upload(&dstd)?,
            slot_type: device.upload(&slot_type)?,
            slot_local: device.upload(&slot_local)?,
            sel_d: device.upload(&sel_i)?,
            offset_d: device.upload(&offsets)?,
            kernels,
            backward_shared,
            chunk_atoms,
            last_truncated: false,
            profile: None,
            device,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Whether the last frame had more neighbors than `sel` for some center.
    pub fn last_truncated(&self) -> bool {
        self.last_truncated
    }

    /// Center atoms evaluated per chunk; per-atom device memory (embedding
    /// values and tangents dominate) scales with this, not with the frame.
    pub fn chunk_atoms(&self) -> usize {
        self.chunk_atoms
    }

    /// Override the chunk size derived from free device memory at load time.
    pub fn set_chunk_atoms(&mut self, atoms: usize) {
        self.chunk_atoms = atoms.max(1);
    }

    /// Turn per-stage timing on (adds a device sync per stage) or off.
    pub fn set_profiling(&mut self, on: bool) {
        self.profile = on.then(StageTimes::default);
    }

    /// Stage times of the last evaluation, when profiling.
    pub fn stage_times(&self) -> Option<&StageTimes> {
        self.profile.as_ref()
    }

    fn evaluate(&mut self, frame: &Frame) -> Result<Evaluation> {
        let device = Arc::clone(&self.device);
        let mut clock = Clock {
            device: &device,
            times: self.profile.take().map(|_| StageTimes::default()),
            last: Instant::now(),
        };
        let result = self.run(&device, frame, &mut clock);
        self.profile = clock.times;
        result
    }

    fn run(&mut self, device: &Device, frame: &Frame, clock: &mut Clock) -> Result<Evaluation> {
        frame.validate()?;
        let types = frame.type_ids(&self.metadata)?;
        let natoms = types.len();
        let stream = device.stream();

        // Sort atoms by type so each (center type) block is contiguous.
        let mut order: Vec<usize> = (0..natoms).collect();
        order.sort_by_key(|&i| types[i]);
        let sorted_types: Vec<i32> = order.iter().map(|&i| types[i]).collect();
        let positions: Vec<[f64; 3]> = order.iter().map(|&i| frame.positions[i]).collect();
        let ntypes = self.sel.len();
        let mut block = vec![0usize; ntypes + 1];
        for &t in &sorted_types {
            block[t as usize + 1] += 1;
        }
        for t in 0..ntypes {
            block[t + 1] += block[t];
        }
        let cell = frame.cell.map_or(Cell::Open, Cell::Periodic);
        let atype = device.upload(&sorted_types)?;
        clock.lap(|t| &mut t.host_prepare)?;
        let list = NeighborList::build(
            device,
            &positions,
            &sorted_types,
            &cell,
            self.rcut,
            &self.sel,
        )?;
        clock.lap(|t| &mut t.neighbors)?;

        let (slots, width, axis) = (self.slots, self.width, self.axis);
        let features = width * axis;
        let chunk = self.chunk_atoms.min(natoms).max(1);
        // Per-chunk work buffers, reused by every chunk. Each element a chunk
        // reads was written earlier in the same chunk.
        let mut env = unsafe { device.alloc_uninit::<f64>(chunk * slots * 4)? };
        let mut radial = unsafe { device.alloc_uninit::<P>(chunk * slots)? };
        let mut g = unsafe { device.alloc_uninit::<P>(chunk * slots * width)? };
        let mut dg = unsafe { device.alloc_uninit::<P>(chunk * slots * width)? };
        let mut b = unsafe { device.alloc_uninit::<f64>(chunk * width * 4)? };
        let mut descriptor = unsafe { device.alloc_uninit::<P>(chunk * features)? };
        let mut d_descriptor = unsafe { device.alloc_uninit::<P>(chunk * features)? };
        let ones = device.upload(&vec![P::ONE; chunk])?;
        // Per-frame results; every atom's energy and virial row is written once.
        let mut atomic = unsafe { device.alloc_uninit::<P>(natoms)? };
        let mut force = device.zeros::<f64>(natoms * 3)?;
        let mut virial = unsafe { device.alloc_uninit::<f64>(natoms * 9)? };
        let (slots_i, w_i, axis_i) = (slots as i32, width as i32, axis as i32);

        for first in (0..natoms).step_by(chunk) {
            let end = (first + chunk).min(natoms);
            let count = end - first;
            let (first_i, count_i) = (first as i32, count as i32);
            // Center-type ranges of this chunk, in chunk-local atoms.
            let ranges: Vec<(usize, usize, usize)> = (0..ntypes)
                .filter_map(|t| {
                    let (lo, hi) = (block[t].max(first), block[t + 1].min(end));
                    (hi > lo).then_some((t, lo - first, hi - first))
                })
                .collect();
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
                    .arg(&first_i)
                    .arg(&count_i)
                    .arg(&slots_i)
                    .arg(&self.rcut_smth)
                    .arg(&self.rcut)
                    .arg(&mut env)
                    .arg(&mut radial);
                let total = count * slots;
                let config = LaunchConfig {
                    grid_dim: (total.div_ceil(BLOCK as usize) as u32, 1, 1),
                    block_dim: (BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { launch.launch(config) }
                    .map(drop)
                    .map_err(launch_error("environment"))?;
            }
            clock.lap(|t| &mut t.environment)?;

            // Embedding values and d/d(radial) tangents,
            // [neighbor type][local atom][slot][width].
            let mut start = 0;
            for (tj, &sel_j) in self.sel.iter().enumerate() {
                for &(ti, lo, hi) in &ranges {
                    let rows = (hi - lo) * sel_j;
                    if rows == 0 {
                        continue;
                    }
                    let row0 = start + lo * sel_j;
                    let x = radial.slice(row0..row0 + rows);
                    let mut out = g.slice_mut(row0 * width..(row0 + rows) * width);
                    let mut dout = dg.slice_mut(row0 * width..(row0 + rows) * width);
                    self.embedding[ti][tj]
                        .forward_tangent_into(device, &x, rows, &mut out, &mut dout)?;
                }
                start += count * sel_j;
            }
            clock.lap(|t| &mut t.embedding)?;

            {
                let mut launch = stream.launch_builder(&self.kernels.descriptor);
                launch
                    .arg(&env)
                    .arg(&g)
                    .arg(&self.slot_type)
                    .arg(&self.slot_local)
                    .arg(&self.sel_d)
                    .arg(&self.offset_d)
                    .arg(&count_i)
                    .arg(&slots_i)
                    .arg(&w_i)
                    .arg(&axis_i)
                    .arg(&mut b)
                    .arg(&mut descriptor);
                let config = LaunchConfig {
                    grid_dim: (count as u32, 1, 1),
                    block_dim: (BLOCK, 1, 1),
                    shared_mem_bytes: (4 * width * 8) as u32,
                };
                unsafe { launch.launch(config) }
                    .map(drop)
                    .map_err(launch_error("descriptor"))?;
            }
            clock.lap(|t| &mut t.descriptor)?;

            // Fitting per center type: atomic energies (copied out so the tape
            // can go) and dE/dD.
            for &(ti, lo, hi) in &ranges {
                let rows = hi - lo;
                let x = descriptor.slice(lo * features..hi * features);
                let tape = self.fitting[ti].forward_record(device, &x, rows)?;
                let mut dx = d_descriptor.slice_mut(lo * features..hi * features);
                self.fitting[ti].backward(device, &tape, &ones.slice(0..rows), &mut dx)?;
                let mut energies = atomic.slice_mut(first + lo..first + hi);
                stream
                    .memcpy_dtod(&tape.output, &mut energies)
                    .map_err(copy_error)?;
            }
            clock.lap(|t| &mut t.fitting)?;

            {
                let mut launch = stream.launch_builder(&self.kernels.backward);
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
                    .arg(&first_i)
                    .arg(&count_i)
                    .arg(&slots_i)
                    .arg(&w_i)
                    .arg(&axis_i)
                    .arg(&self.rcut_smth)
                    .arg(&self.rcut)
                    .arg(&mut force)
                    .arg(&mut virial);
                let config = LaunchConfig {
                    grid_dim: (count as u32, 1, 1),
                    block_dim: (BLOCK, 1, 1),
                    shared_mem_bytes: self.backward_shared as u32,
                };
                unsafe { launch.launch(config) }
                    .map(drop)
                    .map_err(launch_error("force"))?;
            }
            clock.lap(|t| &mut t.forces)?;
            if let Some(times) = &mut clock.times {
                times.chunks += 1;
            }
        }

        self.last_truncated = list.check(device)?;
        let atomic = device.download(&atomic)?;
        let sorted_forces = device.download(&force)?;
        let per_atom_virial = device.download(&virial)?;
        clock.lap(|t| &mut t.download)?;
        let mut energy = 0.0;
        for t in 0..ntypes {
            let range = &atomic[block[t]..block[t + 1]];
            energy +=
                range.iter().map(|&e| e.to_f64()).sum::<f64>() + self.bias[t] * range.len() as f64;
        }
        let mut forces = vec![[0.0; 3]; natoms];
        for (k, &atom) in order.iter().enumerate() {
            forces[atom] = [
                sorted_forces[3 * k],
                sorted_forces[3 * k + 1],
                sorted_forces[3 * k + 2],
            ];
        }
        let mut total_virial = [[0.0; 3]; 3];
        for row in per_atom_virial.chunks_exact(9) {
            for (e, v) in row.iter().enumerate() {
                total_virial[e / 3][e % 3] += v;
            }
        }
        let result = Evaluation {
            energy,
            forces,
            virial: total_virial,
            units: Units::EvAngstrom,
        };
        result.validate(natoms)?;
        Ok(result)
    }
}

impl<P: Scalar> Potential for GpuPotential<P> {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn eval(&mut self, frame: &Frame) -> Result<Evaluation> {
        self.evaluate(frame)
    }
}
