//! se_e2_a, se_e2_r and se_e3 training on the omnimd-gpu runtime (plan I-8),
//! for a model of one of these descriptors or a hybrid of several: the loss of
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
//! reverse pass through both gives the gradient (reverse-over-forward). All
//! embedding nets run as one [`omnimd_gpu::TrainSet`] and all fitting nets as
//! another, so a step issues about thirty device calls whatever the number
//! of types and networks.
//!
//! Networks run in `P`; geometry, the `B` accumulation, forces, virial and
//! the loss are float64. A frame is processed in one piece (no chunking), so
//! it must fit in device memory with its recorded intermediates.
//!
//! An se_e2_r model takes the same passes with the kernels of its
//! descriptor, a mean of embedding rows: only what lies between the
//! embedding and the fitting nets differs. A model whose embedding nets
//! depend on the neighbor type only (`type_one_side`) has each of them once
//! among the parameters, and every row of that neighbor type goes through
//! it, whatever its center.
//!
//! An se_e3 model has other rows for its embedding nets: the pairs of the
//! neighbors of every atom, n(n + 1)/2 for one with n neighbors, each with
//! the product θ of the two normalized directions as the input of the net of
//! its pair of neighbor types (one net per pair, whichever type comes
//! first). D is the sum of weight·θ·G(θ) over the pairs of an atom, so its
//! tangent is the sum of weight·θ̇·(G + θ·G'), and with `a` and `b` the
//! adjoints of D and of its tangent a row has the adjoints weight·(θ·a + θ̇·b)
//! for G and weight·θ·θ̇·b for G'. An empty slot has no direction and no
//! row. The rows of an atom follow the pairs of neighbor types, so that
//! those of a net follow each other within an atom; their number goes with
//! the square of the neighbors, and a frame must fit the device with all of
//! them (the potential takes such a frame in chunks, the trainer does not).
//!
//! A hybrid has the descriptors of its list side by side: the components of
//! all of them, one after the other, are the input of the fitting nets.
//! Each descriptor has its own embedding nets (a [`omnimd_gpu::TrainSet`]
//! each, since their shapes may differ), cutoffs, slots and statistics, and
//! takes every stage of a step with its own kernels: environment, embedding,
//! descriptor and forces, then tangent and reverse, into and out of its
//! place in the buffers of D. The fitting nets run once for all of them.
//! One neighbor list serves them, made with the largest cutoff and of every
//! type the most slots, as in DeePMD-kit; a descriptor with a shorter cutoff
//! or fewer slots reads its own table of that list (of every type the
//! nearest neighbors in its slots, none at or beyond its cutoff), which is
//! kept with the frame like the list itself.
use crate::gpu_model::{
    descriptor_warps, launch_error, pair_dtheta_shared, pair_force_shared, pair_shared, PAIR_BLOCK,
    PAIR_FORCE_BLOCK, SOURCE as MODEL_SOURCE,
};
use crate::{
    Activation, Descriptor, DescriptorKind, Layer, Model, Network, Residual, RADIAL_RESCALE,
};
use omnimd_gpu::{
    Activation as DeviceActivation, Cell, CudaFunction, CudaSlice, Device, LaunchConfig,
    NeighborList, PushKernelArg, Scalar, SetTape, TrainLayer, TrainSet,
};
use omnimd_ml_contract::{Error, Evaluation, Frame, ModelMetadata, Result, Units};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

const SOURCE: &str = r#"
// As in the model kernels, reciprocals worked out once stand in for
// divisions in double precision (`inv_range` = 1 / (rmax - rmin), `dstd_inv`
// = 1 / dstd, 1 / slots), and distances come from the neighbor search.
__device__ void train_switch(double r, double rmin, double inv_range, double* q, double* dq) {
    if (r < rmin) { *q = 1.0; *dq = 0.0; return; }
    double t = (r - rmin) * inv_range, z = 1.0 - t;
    *q = z * z * z * (1.0 + 3.0 * t + 6.0 * t * t);
    *dq = -30.0 * t * t * z * z * inv_range;
}

// Row of the embedding input and output of neighbor `local` of type `t` of
// atom `i`. `rows` holds per atom and neighbor type the first row of its
// neighbors, their number, and the row all empty slots of that pair of
// types share (the table of `slot_row` in the model kernels); the kernels
// here treat empty slots apart and ask only for neighbors.
__device__ long long train_row(const int* rows, int ntypes, int i, int t, int local) {
    return rows[3 * ((long long)i * ntypes + t)] + local;
}

// Tangent of the normalized environment rows when every neighbor vector d
// moves along c[j] - c[i] + w^T d. One thread per slot; empty slots get 0.
extern "C" __global__ void se_e2_a_env_tangent(const int* index, const double* delta,
        const double* dist, const int* atype, const double* dstd_inv, const double* c,
        const double* w, int natoms, int slots, double rmin, double inv_range, double* envdot) {
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
        double rr = dist[idx], inv = 1.0 / rr, q, dq;
        train_switch(rr, rmin, inv_range, &q, &dq);
        double u[3] = {d[0] * inv, d[1] * inv, d[2] * inv};
        double sw = q * inv, dsw = (dq - sw) * inv, across = sw * inv;
        double along = u[0] * dd[0] + u[1] * dd[1] + u[2] * dd[2];
        const double* inv_std = dstd_inv + 4 * atype[i];
        out[0] = dsw * along * inv_std[0];
        for (int a = 0; a < 3; ++a)
            out[a + 1] = (dsw * u[a] * along + across * (dd[a] - u[a] * along)) * inv_std[a + 1];
    }
    for (int a = 0; a < 4; ++a) envdot[4 * idx + a] = out[a];
}

// The components of an atom lie `stride` values apart in the buffers of D,
// of its tangent and of their adjoints: the number of components of all
// descriptors of the model, with the buffer given from the first component
// of this descriptor on. For a model of one descriptor that is its own
// number of components.
//
// One block per atom: tangents of B and of D = B·B[:axis]^T. The embedding
// tangent of a row is dg (its derivative along the radial input) times the
// tangent of that input, envdot[..][0].
template <typename P>
__device__ void descriptor_tangent(const double* env, const double* envdot, const P* g,
        const P* dg, const double* b_in, const int* slot_type, const int* slot_local,
        const int* rows, int ntypes, int slots, int width, int axis, int stride,
        double* bdot_out, P* ddot_out) {
    extern __shared__ double shared[];
    double* b = shared;
    double* bd = b + 4 * width;
    int i = blockIdx.x;
    const double inv_slots = 1.0 / slots;
    for (int m = threadIdx.x; m < width; m += blockDim.x) {
        double acc[4] = {0.0, 0.0, 0.0, 0.0};
        for (int s = 0; s < slots; ++s) {
            // An empty slot has no tangent: nothing to add.
            if (slot_local[s] >= rows[3 * ((long long)i * ntypes + slot_type[s]) + 1]) continue;
            long long row = train_row(rows, ntypes, i, slot_type[s], slot_local[s]) * width + m;
            double gm = (double)g[row], gpm = (double)dg[row];
            const double* r = env + ((long long)i * slots + s) * 4;
            const double* rd = envdot + ((long long)i * slots + s) * 4;
            for (int a = 0; a < 4; ++a) acc[a] += gpm * rd[0] * r[a] + gm * rd[a];
        }
        for (int a = 0; a < 4; ++a) {
            double v = acc[a] * inv_slots;
            bd[4 * m + a] = v;
            bdot_out[((long long)i * width + m) * 4 + a] = v;
            b[4 * m + a] = b_in[((long long)i * width + m) * 4 + a];
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < width * axis; e += blockDim.x) {
        int m = e / axis, k = e % axis;
        double v = 0.0;
        for (int a = 0; a < 4; ++a) v += bd[4 * m + a] * b[4 * k + a] + b[4 * m + a] * bd[4 * k + a];
        ddot_out[(long long)i * stride + e] = (P)v;
    }
}

// One block per atom: adjoints of B and of its tangent from the adjoints of
// D and of its tangent, then of every embedding row and of its derivative
// along the radial input. Empty slots take part: their rows hold the
// normalized zero, which is not zero. All empty slots of one neighbor type
// have that same row and no tangent, so their adjoints are equal: they are
// added, times their number, to the embedding row those slots share with
// the other atoms of this type. gbar and gpbar start from zero.
template <typename P>
__device__ void descriptor_reverse(const double* env, const double* envdot, const double* b_in,
        const double* bdot_in, const P* dbar_in, const P* ddbar_in, const int* slot_type,
        const int* slot_local, const int* sel, const int* offset, const int* rows, int ntypes,
        int slots, int width, int axis, int stride, P* gbar, P* gpbar) {
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
    const P* dbar = dbar_in + (long long)i * stride;
    const P* ddbar = ddbar_in + (long long)i * stride;
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
    const int* mine = rows + 3 * (long long)i * ntypes;
    const double inv_slots = 1.0 / slots;
    for (int s = warp; s < slots; s += warps) {
        if (slot_local[s] >= mine[3 * slot_type[s] + 1]) continue;
        long long row = train_row(rows, ntypes, i, slot_type[s], slot_local[s]) * width;
        const double* r = env + 4 * ((long long)i * slots + s);
        const double* rd = envdot + 4 * ((long long)i * slots + s);
        for (int m = lane; m < width; m += 32) {
            double v1 = 0.0, v2 = 0.0;
            for (int a = 0; a < 4; ++a) {
                v1 += bb[4 * m + a] * r[a] + bdb[4 * m + a] * rd[a];
                v2 += bdb[4 * m + a] * r[a];
            }
            gbar[row + m] = (P)(v1 * inv_slots);
            gpbar[row + m] = (P)(rd[0] * v2 * inv_slots);
        }
    }
    for (int t = 0; t < ntypes; ++t) {
        int filled = mine[3 * t + 1], empty = sel[t] - filled;
        if (empty <= 0) continue;
        // The first empty slot of this type stands for all of them.
        const double* r = env + 4 * ((long long)i * slots + offset[t] + filled);
        for (int m = threadIdx.x; m < width; m += blockDim.x) {
            double v1 = 0.0;
            for (int a = 0; a < 4; ++a) v1 += bb[4 * m + a] * r[a];
            atomicAdd(&gbar[(long long)mine[3 * t + 2] * width + m], (P)(empty * v1 * inv_slots));
        }
    }
}

// se_e2_r: the tangent of D[m] = scale · sum over the slots of G[slot][m]
// when the radial entry of every neighbor moves by envdot[..][0]. An empty
// slot does not move. One block per atom.
template <typename P>
__device__ void radial_descriptor_tangent(const double* envdot, const P* dg, const int* slot_type,
        const int* slot_local, const int* rows, int ntypes, int slots, int width, double scale,
        int stride, P* ddot_out) {
    int i = blockIdx.x;
    for (int m = threadIdx.x; m < width; m += blockDim.x) {
        double acc = 0.0;
        for (int s = 0; s < slots; ++s) {
            if (slot_local[s] >= rows[3 * ((long long)i * ntypes + slot_type[s]) + 1]) continue;
            long long row = train_row(rows, ntypes, i, slot_type[s], slot_local[s]) * width + m;
            acc += (double)dg[row] * envdot[((long long)i * slots + s) * 4];
        }
        ddot_out[(long long)i * stride + m] = (P)(acc * scale);
    }
}

// se_e2_r: adjoints of every embedding row and of its derivative along the
// radial input, from the adjoints of D and of its tangent. D takes every
// row with the same weight: the adjoint of a neighbor's row is that of D
// times the scale, and the adjoint of its derivative that of the tangent of
// D times the tangent of the neighbor's radial entry. The empty slots of a
// type have no tangent and add the adjoint of D, times their number, to the
// row they share with the other atoms of this type. One block per atom;
// gbar and gpbar start from zero.
template <typename P>
__device__ void radial_descriptor_reverse(const double* envdot, const P* dbar_in,
        const P* ddbar_in, const int* slot_type, const int* slot_local, const int* sel,
        const int* rows, int ntypes, int slots, int width, double scale, int stride, P* gbar,
        P* gpbar) {
    int i = blockIdx.x;
    const P* dbar = dbar_in + (long long)i * stride;
    const P* ddbar = ddbar_in + (long long)i * stride;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    const int* mine = rows + 3 * (long long)i * ntypes;
    for (int s = warp; s < slots; s += warps) {
        if (slot_local[s] >= mine[3 * slot_type[s] + 1]) continue;
        long long row = train_row(rows, ntypes, i, slot_type[s], slot_local[s]) * width;
        double rd = envdot[4 * ((long long)i * slots + s)];
        for (int m = lane; m < width; m += 32) {
            gbar[row + m] = (P)((double)dbar[m] * scale);
            gpbar[row + m] = (P)((double)ddbar[m] * rd * scale);
        }
    }
    for (int t = 0; t < ntypes; ++t) {
        int empty = sel[t] - mine[3 * t + 1];
        if (empty <= 0) continue;
        for (int m = threadIdx.x; m < width; m += blockDim.x)
            atomicAdd(&gbar[(long long)mine[3 * t + 2] * width + m],
                      (P)(empty * (double)dbar[m] * scale));
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

// Adam for float32 networks whose parameters are kept in float64. The step
// is formed in float32 and added in float64: at the end of a schedule the
// learning rate is of the order of the spacing of float32 numbers near the
// parameters, and steps that small would be rounded away.
extern "C" __global__ void adam_master(double* master, float* theta, const float* grad, float* m,
        float* v, long long n, double lr, float beta1, float beta2, float eps, float correction1,
        float correction2) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = grad[idx];
    float mi = beta1 * m[idx] + (1.0f - beta1) * g;
    float vi = beta2 * v[idx] + (1.0f - beta2) * g * g;
    m[idx] = mi;
    v[idx] = vi;
    double value = master[idx] - lr * (double)((mi / correction1) / (sqrtf(vi / correction2) + eps));
    master[idx] = value;
    theta[idx] = (float)value;
}

#define INSTANTIATE(P, S)                                                                         \
extern "C" __global__ void se_e2_a_descriptor_tangent_##S(const double* env,                      \
        const double* envdot, const P* g, const P* dg, const double* b_in, const int* slot_type,  \
        const int* slot_local, const int* rows, int ntypes, int slots, int width, int axis,       \
        int stride, double* bdot_out, P* ddot_out) {                                              \
    descriptor_tangent<P>(env, envdot, g, dg, b_in, slot_type, slot_local, rows, ntypes, slots,   \
                          width, axis, stride, bdot_out, ddot_out);                               \
}                                                                                                 \
extern "C" __global__ void se_e2_a_descriptor_reverse_##S(const double* env,                      \
        const double* envdot, const double* b_in, const double* bdot_in, const P* dbar,           \
        const P* ddbar, const int* slot_type, const int* slot_local, const int* sel,              \
        const int* offset, const int* rows, int ntypes, int slots, int width, int axis,           \
        int stride, P* gbar, P* gpbar) {                                                          \
    descriptor_reverse<P>(env, envdot, b_in, bdot_in, dbar, ddbar, slot_type, slot_local, sel,    \
                          offset, rows, ntypes, slots, width, axis, stride, gbar, gpbar);         \
}                                                                                                 \
extern "C" __global__ void se_e2_r_descriptor_tangent_##S(const double* envdot, const P* dg,      \
        const int* slot_type, const int* slot_local, const int* rows, int ntypes, int slots,      \
        int width, double scale, int stride, P* ddot_out) {                                       \
    radial_descriptor_tangent<P>(envdot, dg, slot_type, slot_local, rows, ntypes, slots, width,   \
                                 scale, stride, ddot_out);                                        \
}                                                                                                 \
extern "C" __global__ void se_e2_r_descriptor_reverse_##S(const double* envdot, const P* dbar,    \
        const P* ddbar, const int* slot_type, const int* slot_local, const int* sel,              \
        const int* rows, int ntypes, int slots, int width, double scale, int stride, P* gbar,     \
        P* gpbar) {                                                                               \
    radial_descriptor_reverse<P>(envdot, dbar, ddbar, slot_type, slot_local, sel, rows, ntypes,   \
                                 slots, width, scale, stride, gbar, gpbar);                       \
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

/// The kernels of one descriptor: what stands between its embedding nets
/// and the fitting nets, forward, along the tangent and in reverse.
struct Kernels {
    /// The inputs of the embedding rows: from the environment rows, or
    /// (se_e3) the pairs of neighbors.
    env: CudaFunction,
    descriptor: CudaFunction,
    force: CudaFunction,
    /// se_e3: dE/dθ of every pair, from which its force kernel goes on.
    pair_force: Option<CudaFunction>,
    env_tangent: CudaFunction,
    /// se_e3: the tangent of θ of every pair.
    pair_tangent: Option<CudaFunction>,
    descriptor_tangent: CudaFunction,
    descriptor_reverse: CudaFunction,
}

/// Wall-clock milliseconds per stage of [`GpuTrainer::accumulate`] and of the
/// optimizer, summed since profiling was turned on. Every stage ends with a
/// device synchronization, so a profiled run is slower than a normal one and
/// the stages add up to more than a normal step takes.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct TrainStageTimes {
    /// Atom ordering and the type upload.
    pub host_prepare: f64,
    pub neighbors: f64,
    pub environment: f64,
    /// Embedding nets with their unit tangent.
    pub embedding: f64,
    pub descriptor: f64,
    /// First fitting pass: atomic energies and their descriptor derivative.
    pub fitting: f64,
    pub forces: f64,
    /// Downloads, the loss on the host and the upload of its seeds.
    pub loss: f64,
    /// Environment and descriptor along the seed direction.
    pub tangent: f64,
    /// Second fitting pass and its reverse pass, with the parameter gradient.
    pub fitting_gradient: f64,
    pub descriptor_reverse: f64,
    /// Reverse pass of the embedding nets, with the parameter gradient.
    pub embedding_gradient: f64,
    /// Gradient reset and Adam update.
    pub optimizer: f64,
    /// Frames accumulated while profiling.
    pub frames: u64,
}

/// Stage clock: a no-op unless profiling, so a normal run never waits for
/// the device between stages.
struct Clock<'a> {
    device: &'a Device,
    times: Option<TrainStageTimes>,
    last: Instant,
}

impl Clock<'_> {
    fn lap(&mut self, stage: impl FnOnce(&mut TrainStageTimes) -> &mut f64) -> Result<()> {
        if let Some(times) = &mut self.times {
            self.device.synchronize()?;
            let now = Instant::now();
            *stage(times) += (now - self.last).as_secs_f64() * 1e3;
            self.last = now;
        }
        Ok(())
    }
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
    /// The descriptors: one for a model of one descriptor, those of the list
    /// of a hybrid in its order.
    parts: Vec<Part<P>>,
    /// Components of the descriptor, for a hybrid those of all its
    /// descriptors one after the other: the input of a fitting net.
    features: usize,
    /// The parameters the networks compute with.
    theta: CudaSlice<P>,
    /// For float32 networks: the parameters in float64, which the optimizer
    /// updates; `theta` is their rounded copy.
    master: Option<CudaSlice<f64>>,
    gradient: CudaSlice<P>,
    adam: Option<AdamState<P>>,
    /// One fitting net per center type: the layouts, and all of them as one
    /// set that evaluates every atom with the net of its type.
    fitting: Vec<Vec<TrainLayer>>,
    fitting_set: TrainSet<P>,
    adam_kernel: CudaFunction,
    adam_master: CudaFunction,
    /// hybrid: writes the table of slots of a descriptor that does not read
    /// the neighbor list as it is.
    narrow: Option<CudaFunction>,
    profile: Option<TrainStageTimes>,
    /// Frames prepared before, by a hash of their content.
    prepared: HashMap<u64, Arc<Prepared>>,
    prepared_bytes: usize,
    prepared_limit: usize,
}

/// One descriptor of the model: its embedding nets among the parameters and
/// what its kernels read. A model has one; a hybrid has one for every
/// descriptor of its list, each with cutoffs, slots, statistics and nets of
/// its own.
struct Part<P: Scalar> {
    /// Which kernels stand between the embedding and the fitting nets.
    kind: DescriptorKind,
    rcut: f64,
    rcut_smth: f64,
    /// 1 / (rcut - rcut_smth), for the switching function.
    inv_range: f64,
    sel: Vec<usize>,
    slots: usize,
    width: usize,
    /// se_e2_a: the axis neurons. None for se_e2_r and se_e3.
    axis: usize,
    /// Components of this descriptor, and the first of them among the
    /// components of all descriptors of the model.
    features: usize,
    first_feature: usize,
    /// The embedding nets depend on the neighbor type only: the parameters
    /// hold one per neighbor type instead of one per pair of types.
    one_side: bool,
    /// Net `center type · ntypes + neighbor type`, or net `neighbor type`
    /// when the nets do not depend on the center: the layouts, and all of
    /// them as one set that evaluates every row with the net it names.
    /// se_e3: net `neighbor type · ntypes + neighbor type`, where the two
    /// orders of a pair of types name the same parameters.
    embedding: Vec<Vec<TrainLayer>>,
    embedding_set: TrainSet<P>,
    davg: CudaSlice<f64>,
    /// Reciprocals of the normalization widths: the kernels multiply.
    dstd_inv: CudaSlice<f64>,
    slot_type: CudaSlice<i32>,
    slot_local: CudaSlice<i32>,
    sel_d: CudaSlice<i32>,
    offset_d: CudaSlice<i32>,
    /// se_e3: 1 / sel of every type (0 for a type without slots), for the
    /// weights of the pairs.
    sel_inv: Option<CudaSlice<f64>>,
    kernels: Kernels,
    /// hybrid: how this descriptor reads the neighbor list of all of them,
    /// when its slots or its cutoff are not those of the list.
    view: Option<View>,
    /// Before a message about this descriptor: its place in the list of a
    /// hybrid, nothing otherwise.
    context: String,
}

/// The slots of the neighbor list by type and the first slot of each type:
/// what the kernel reads that writes a descriptor's own table of the list
/// (`hybrid_slots` of the model kernels).
struct View {
    list_sel: CudaSlice<i32>,
    list_offset: CudaSlice<i32>,
    list_slots: usize,
}

/// What a frame needs before the networks run and what never changes for the
/// same frame: its atoms sorted by type, their neighbor list on the device
/// and, for every descriptor, the layout of the embedding rows. A training
/// set visits the same frames again and again.
struct Prepared {
    /// The frame as given, to recognize it.
    positions: Vec<[f64; 3]>,
    cell: Option<[[f64; 3]; 3]>,
    types: Vec<i32>,
    /// `order[k]` is the atom at sorted place `k`; `block[t]..block[t + 1]`
    /// are the sorted places of type `t`.
    order: Vec<usize>,
    block: Vec<usize>,
    atype: CudaSlice<i32>,
    list: NeighborList,
    /// The frame as each descriptor reads it, in the order of the trainer's.
    parts: Vec<Seen>,
}

/// A frame as one descriptor reads it.
struct Seen {
    /// hybrid: the descriptor's own table of the neighbor list, when it does
    /// not read the list as it is.
    table: Option<SlotTable>,
    /// Where the kernels find the embedding rows of an atom, the rows of
    /// every embedding net as (net, first row, rows), and the number of
    /// rows: none for an se_e3 descriptor in a frame without a neighbor.
    layout: Layout,
    blocks: Vec<(usize, usize, usize)>,
    rows_embedding: usize,
}

/// The embedding rows of a frame as the kernels of a descriptor find them.
enum Layout {
    /// se_e2_a and se_e2_r: the row table of [`embedding_rows`].
    Slots(CudaSlice<i32>),
    /// se_e3: the first row of the pairs of every atom and, after the last,
    /// the number of all rows ([`pair_rows`]).
    Pairs(CudaSlice<i64>),
}

/// The slots of a descriptor for every atom of a frame, like those of a
/// neighbor list: `[atoms][slots]` neighbors (-1 for an empty slot), their
/// vectors and distances.
struct SlotTable {
    index: CudaSlice<i32>,
    delta: CudaSlice<f64>,
    dist: CudaSlice<f64>,
}

impl Seen {
    /// The neighbor slots the kernels of the descriptor read (neighbors,
    /// vectors, distances): its own table, or the list itself.
    fn slots<'a>(
        &'a self,
        list: &'a NeighborList,
    ) -> (&'a CudaSlice<i32>, &'a CudaSlice<f64>, &'a CudaSlice<f64>) {
        match &self.table {
            Some(table) => (&table.index, &table.delta, &table.dist),
            None => (&list.index, &list.delta, &list.dist),
        }
    }

    /// Device memory this holds.
    fn bytes(&self) -> usize {
        let table = self.table.as_ref().map_or(0, |table| {
            4 * table.index.len() + 8 * (table.delta.len() + table.dist.len())
        });
        let layout = match &self.layout {
            Layout::Slots(rows) => 4 * rows.len(),
            Layout::Pairs(start) => 8 * start.len(),
        };
        layout + table
    }
}

impl Prepared {
    /// Device memory this holds.
    fn bytes(&self) -> usize {
        let list = &self.list;
        4 * (self.atype.len() + list.index.len() + list.count.len())
            + 8 * (list.delta.len() + list.dist.len() + list.min_distance.len())
            + self.parts.iter().map(Seen::bytes).sum::<usize>()
    }
}

/// Device memory for prepared frames unless set otherwise: 1 GiB.
const PREPARED_LIMIT: usize = 1 << 30;

fn frame_key(frame: &Frame, types: &[i32]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for position in &frame.positions {
        position.map(f64::to_bits).hash(&mut hasher);
    }
    frame
        .cell
        .map(|cell| cell.map(|row| row.map(f64::to_bits)))
        .hash(&mut hasher);
    types.hash(&mut hasher);
    hasher.finish()
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

/// Neighbors that hold a slot of the list, `[atoms][types]`. Without
/// truncation these are the neighbor counts of the list; a truncated list
/// is counted slot by slot.
fn filled_slots(device: &Device, list: &NeighborList) -> Result<Vec<usize>> {
    let ntypes = list.sel.len();
    let counts = device.download(&list.count)?;
    let truncated = counts.chunks_exact(ntypes).any(|atom| {
        atom.iter()
            .zip(&list.sel)
            .any(|(&count, &sel)| count as usize > sel)
    });
    if !truncated {
        return Ok(counts.into_iter().map(|count| count as usize).collect());
    }
    held_slots(&device.download(&list.index)?, &list.sel)
}

/// Neighbors that hold a slot, `[atoms][types]`, counted slot by slot in
/// `index` (`[atoms][slots]`, -1 for an empty slot, `sel[t]` slots of each
/// type in the order of the types).
fn held_slots(index: &[i32], sel: &[usize]) -> Result<Vec<usize>> {
    let slots: usize = sel.iter().sum();
    let mut filled = Vec::with_capacity(index.len() / slots.max(1) * sel.len());
    for atom in index.chunks_exact(slots) {
        let mut first = 0;
        for &sel in sel {
            let segment = &atom[first..first + sel];
            let held = segment.iter().take_while(|&&j| j >= 0).count();
            if segment[held..].iter().any(|&j| j >= 0) {
                return Err(Error("neighbor slots are not filled from the front".into()));
            }
            filled.push(held);
            first += sel;
        }
    }
    Ok(filled)
}

/// Rows of the embedding nets for a frame whose atoms are sorted by type
/// (`block[t]..block[t + 1]` are those of type `t`) and hold `filled[atom]
/// [type]` neighbors of each type in slots of `sel[type]`.
///
/// Every neighbor gets a row; all empty slots of one net (center type,
/// neighbor type) share one row, because their input is the same. Rows are
/// ordered by neighbor type, then center atom, with the shared row after
/// the neighbors of its net, so the rows of a net are contiguous. Returns
/// the table the kernels read (per atom and neighbor type: first row,
/// neighbors, shared row or -1), the rows of every net as (net, first row,
/// rows), and the number of rows.
///
/// With `one_side` there is one net per neighbor type: the rows are laid
/// out as before, a shared row per center type included (its input differs
/// with the normalization of the center), and all rows of a neighbor type,
/// which follow each other, are that net's.
#[allow(clippy::type_complexity)]
fn embedding_rows(
    filled: &[usize],
    block: &[usize],
    sel: &[usize],
    one_side: bool,
) -> Result<(Vec<i32>, Vec<(usize, usize, usize)>, usize)> {
    let ntypes = sel.len();
    let int =
        |value: usize| i32::try_from(value).map_err(|_| Error("embedding rows exceed i32".into()));
    let mut table = vec![-1i32; filled.len() * 3];
    let (mut nets, mut next) = (Vec::new(), 0usize);
    for (neighbor, &capacity) in sel.iter().enumerate() {
        let of_neighbor = next;
        for center in 0..ntypes {
            let (first, mut shared) = (next, false);
            for atom in block[center]..block[center + 1] {
                let held = filled[atom * ntypes + neighbor];
                if held > capacity {
                    return Err(Error(format!(
                        "atom {atom} holds {held} neighbors of type {neighbor} in {capacity} slots"
                    )));
                }
                let entry = 3 * (atom * ntypes + neighbor);
                (table[entry], table[entry + 1]) = (int(next)?, int(held)?);
                next += held;
                shared |= held < capacity;
            }
            if shared {
                for atom in block[center]..block[center + 1] {
                    table[3 * (atom * ntypes + neighbor) + 2] = int(next)?;
                }
                next += 1;
            }
            if !one_side && next > first {
                nets.push((center * ntypes + neighbor, first, next - first));
            }
        }
        if one_side && next > of_neighbor {
            nets.push((neighbor, of_neighbor, next - of_neighbor));
        }
    }
    int(next)?;
    Ok((table, nets, next))
}

/// Rows of the embedding nets of an se_e3 descriptor for a frame whose atoms
/// hold `filled[atom][type]` neighbors: the pairs of the neighbors of every
/// atom, n(n + 1)/2 for one with n neighbors, atom after atom.
///
/// The rows of an atom go by the types of the two neighbors, t <= u with t
/// first (`pair_row` of the model kernels): the triangle of the pairs within
/// type t, then the pairs of its neighbors with those of every later type.
/// Returns the first row of every atom with the number of all rows after the
/// last, the rows of every net as (net, first row, rows) with net
/// `t · ntypes + u`, and the number of rows.
#[allow(clippy::type_complexity)]
fn pair_rows(
    filled: &[usize],
    ntypes: usize,
) -> Result<(Vec<i64>, Vec<(usize, usize, usize)>, usize)> {
    let mut start = Vec::with_capacity(filled.len() / ntypes.max(1) + 1);
    let mut nets: Vec<(usize, usize, usize)> = Vec::new();
    let mut next = 0usize;
    for atom in filled.chunks_exact(ntypes) {
        start.push(next as i64);
        for (t, &own) in atom.iter().enumerate() {
            for (u, &other) in atom.iter().enumerate().skip(t) {
                let rows = match u == t {
                    true => own * (own + 1) / 2,
                    false => own * other,
                };
                if rows == 0 {
                    continue;
                }
                // With one type the rows of its one net follow from atom to
                // atom: one block.
                match nets.last_mut() {
                    Some(last) if last.0 == t * ntypes + u && last.1 + last.2 == next => {
                        last.2 += rows
                    }
                    _ => nets.push((t * ntypes + u, next, rows)),
                }
                next += rows;
            }
        }
    }
    start.push(next as i64);
    // The gradient kernel of the nets has its rows by 32-bit numbers.
    if next > i32::MAX as usize {
        return Err(Error(format!(
            "{next} pairs of neighbors in one frame exceed the {} rows the embedding nets take",
            i32::MAX
        )));
    }
    Ok((start, nets, next))
}

/// Shared memory of the se_e2_r force kernel per block: dE/d(radial entry)
/// of every slot and the block's sums in double, dE/dD in `P`.
fn radial_force_shared<P>(slots: usize, width: usize) -> usize {
    (slots + 12) * 8 + (width * std::mem::size_of::<P>()).next_multiple_of(8)
}

/// Shared memory of the kernel of the tangent of θ per block: the directions
/// of the neighbors and their tangents in double, the neighbors' slots and
/// the first neighbor of each type.
fn pair_tangent_shared(slots: usize, ntypes: usize) -> usize {
    6 * slots * 8 + (4 * (slots + ntypes + 1)).next_multiple_of(8)
}

/// What the environment kernel gives of one descriptor.
struct Rows<P: Scalar> {
    /// se_e2_a: the normalized environment rows. The others have none.
    env: Option<CudaSlice<f64>>,
    /// The input and the net of every embedding row. se_e3 in a frame
    /// without a pair: one value that nothing reads.
    input: CudaSlice<P>,
    network: CudaSlice<i32>,
    /// se_e3: what every pair counts in the descriptor.
    weight: Option<CudaSlice<P>>,
}

/// What the tangent pass of a descriptor leaves for its reverse pass.
struct Tangents<P: Scalar> {
    /// se_e2_a and se_e2_r: the tangent of the environment rows.
    envdot: Option<CudaSlice<f64>>,
    /// se_e2_a: the tangent of B.
    bdot: Option<CudaSlice<f64>>,
    /// se_e3: the tangent of θ of every pair.
    thetadot: Option<CudaSlice<P>>,
}

fn per_atom(natoms: usize, shared: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (natoms as u32, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: shared as u32,
    }
}

impl<P: Scalar> Part<P> {
    /// The descriptor `d` of a model whose neighbor list has the slots
    /// `list_sel` within `list_rcut`. Its embedding nets go to the end of
    /// `theta`; `first_feature` is the place of its components among those
    /// of the model, and `context` goes before the message of a descriptor
    /// that cannot be trained (its place in the list of a hybrid).
    fn new(
        device: &Device,
        d: &Descriptor,
        list_sel: &[usize],
        list_rcut: f64,
        first_feature: usize,
        theta: &mut Vec<f64>,
        context: &str,
    ) -> Result<Self> {
        let Some(kind) = d.descriptor_kind() else {
            return Err(Error(format!(
                "{context}the GPU trainer supports the se_e2_a, se_e2_r and se_e3 descriptors and hybrids of them, not {}",
                d.kind
            )));
        };
        let width = d.width();
        let slots: usize = d.sel.iter().sum();
        let ntypes = d.sel.len();
        // The most shared memory a kernel of the descriptor takes per block.
        let real = std::mem::size_of::<P>();
        let shared = match kind {
            DescriptorKind::SeE2A => (16 * width + 12 + 4 * slots) * 8 + 4 * width * real,
            DescriptorKind::SeE2R => radial_force_shared::<P>(slots, width),
            // The forces, the tangent of θ, the rows of the pairs, the
            // adjoints of D and of its tangent of an atom, and the sums of
            // one warp of the descriptor kernel.
            DescriptorKind::SeE3 => pair_force_shared(slots, ntypes)
                .max(pair_tangent_shared(slots, ntypes))
                .max(pair_shared(slots, ntypes))
                .max(2 * width * real),
        };
        if shared > SHARED_LIMIT {
            return Err(Error(format!(
                "{context}GPU {} training does not fit shared memory with embedding width {width} and {slots} neighbor slots",
                kind.name()
            )));
        }
        // One net per neighbor type when the nets are shared between the
        // central types, else the net of every pair of types.
        let one_side = d.type_one_side;
        let embedding: Vec<Vec<TrainLayer>> = match kind {
            // se_e3: the net of a pair of neighbor types is among the
            // parameters once, in the order of the pairs t <= u, and both
            // orders of the pair name it.
            DescriptorKind::SeE3 => {
                let mut layouts = vec![Vec::new(); ntypes * ntypes];
                for t in 0..ntypes {
                    for u in t..ntypes {
                        let layout = pack(&d.embedding[t][u], 1, theta);
                        layouts[u * ntypes + t] = layout.clone();
                        layouts[t * ntypes + u] = layout;
                    }
                }
                layouts
            }
            _ => {
                let nets: Vec<&Network> = match one_side {
                    true => d.embedding[0].iter().collect(),
                    false => d.embedding.iter().flatten().collect(),
                };
                nets.into_iter()
                    .map(|network| pack(network, 1, theta))
                    .collect()
            }
        };
        let embedding_set = TrainSet::new(device, embedding.clone())?;
        let davg: Vec<f64> = d.normalization.iter().flat_map(|n| n.mean).collect();
        let dstd_inv: Vec<f64> = d
            .normalization
            .iter()
            .flat_map(|n| n.std.map(|std| 1.0 / std))
            .collect();
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
        let train_kernel = |name: &str| {
            device
                .kernels()
                .function(SOURCE, &format!("{name}_{}", P::SUFFIX))
        };
        // The environment kernels serve se_e2_a and se_e2_r, which reads the
        // radial entry of what they give; the tangent of the environment
        // rows serves all three, se_e3 reading that of the directions.
        let name = kind.name();
        let env_tangent = device.kernels().function(SOURCE, "se_e2_a_env_tangent")?;
        let kernels = match kind {
            // The kernels of the potential for the pairs, D and the forces;
            // its own for the tangents and the adjoints.
            DescriptorKind::SeE3 => Kernels {
                env: model_kernel("se_e3_pairs")?,
                descriptor: model_kernel("se_e3_descriptor")?,
                force: model_kernel("se_e3_backward")?,
                pair_force: Some(model_kernel("se_e3_dtheta")?),
                env_tangent,
                pair_tangent: Some(model_kernel("se_e3_theta_tangent")?),
                descriptor_tangent: model_kernel("se_e3_descriptor_tangent")?,
                descriptor_reverse: model_kernel("se_e3_reverse")?,
            },
            _ => Kernels {
                env: model_kernel("se_e2_a_env_rows")?,
                descriptor: model_kernel(&format!("{name}_descriptor_rows"))?,
                force: model_kernel(&format!("{name}_backward_rows"))?,
                pair_force: None,
                env_tangent,
                pair_tangent: None,
                descriptor_tangent: train_kernel(&format!("{name}_descriptor_tangent"))?,
                descriptor_reverse: train_kernel(&format!("{name}_descriptor_reverse"))?,
            },
        };
        let sel_inv = match kind {
            DescriptorKind::SeE3 => {
                let inverse: Vec<f64> = d
                    .sel
                    .iter()
                    .map(|&s| if s == 0 { 0.0 } else { 1.0 / s as f64 })
                    .collect();
                Some(device.upload(&inverse)?)
            }
            _ => None,
        };
        // A descriptor with the slots and the cutoff of the list reads the
        // list as it is; any other, its own table of it.
        let view = match d.sel == list_sel && d.rcut == list_rcut {
            true => None,
            false => {
                let mut first = 0i32;
                let list_offset: Vec<i32> = list_sel
                    .iter()
                    .map(|&count| {
                        let offset = first;
                        first += count as i32;
                        offset
                    })
                    .collect();
                let list_sel_i: Vec<i32> = list_sel.iter().map(|&s| s as i32).collect();
                Some(View {
                    list_sel: device.upload(&list_sel_i)?,
                    list_offset: device.upload(&list_offset)?,
                    list_slots: list_sel.iter().sum(),
                })
            }
        };
        Ok(Self {
            kind,
            rcut: d.rcut,
            rcut_smth: d.rcut_smth,
            inv_range: 1.0 / (d.rcut - d.rcut_smth),
            sel: d.sel.clone(),
            slots,
            width,
            axis: d.axis_neuron,
            features: d.features(),
            first_feature,
            one_side,
            embedding,
            embedding_set,
            davg: device.upload(&davg)?,
            dstd_inv: device.upload(&dstd_inv)?,
            slot_type: device.upload(&slot_type)?,
            slot_local: device.upload(&slot_local)?,
            sel_d: device.upload(&sel_i)?,
            offset_d: device.upload(&offsets)?,
            sel_inv,
            kernels,
            view,
            context: context.to_string(),
        })
    }

    /// Device bytes an embedding row of this descriptor takes in a step: the
    /// passes of its net, the adjoints of its outputs and of their tangents
    /// and, for a pair of an se_e3 descriptor, θ with its tangent, its
    /// weight, dE/dθ and the net of the pair.
    fn bytes_per_row(&self) -> usize {
        let real = std::mem::size_of::<P>();
        let pair = match self.kind {
            DescriptorKind::SeE3 => 4 * real + 4,
            _ => real + 4,
        };
        self.embedding_set.bytes_per_row() + 2 * self.width * real + pair
    }

    /// se_e2_r: the mean of the embedding rows over the slots, over five.
    fn radial_scale(&self) -> f64 {
        RADIAL_RESCALE / self.slots as f64
    }

    fn per_slot(&self, natoms: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: ((natoms * self.slots).div_ceil(BLOCK as usize) as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// How this descriptor reads a frame: its own table of the neighbor
    /// list, when its slots or its cutoff are not the list's, and the rows
    /// of its embedding nets for atoms sorted by type (`block`).
    fn see(
        &self,
        device: &Device,
        list: &NeighborList,
        block: &[usize],
        narrow: Option<&CudaFunction>,
    ) -> Result<Seen> {
        let natoms = list.natoms;
        let (table, filled) = match &self.view {
            None => (None, filled_slots(device, list)?),
            Some(view) => {
                let function = narrow.ok_or_else(|| {
                    Error("a hybrid trainer without the kernel of its tables".into())
                })?;
                // SAFETY: the kernel below writes every element.
                let mut table = unsafe {
                    SlotTable {
                        index: device.alloc_uninit::<i32>(natoms * self.slots)?,
                        delta: device.alloc_uninit::<f64>(3 * natoms * self.slots)?,
                        dist: device.alloc_uninit::<f64>(natoms * self.slots)?,
                    }
                };
                // The kernel also marks the atoms of which this descriptor
                // leaves neighbors out for lack of slots. Training takes
                // the nearest, as it does when the list itself is short.
                let mut dropped = device.zeros::<i32>(natoms)?;
                let (natoms_i, slots_i) = (natoms as i32, self.slots as i32);
                let (list_slots_i, ntypes_i) = (view.list_slots as i32, self.sel.len() as i32);
                let stream = device.stream();
                let mut launch = stream.launch_builder(function);
                launch
                    .arg(&list.index)
                    .arg(&list.delta)
                    .arg(&list.dist)
                    .arg(&self.slot_type)
                    .arg(&self.slot_local)
                    .arg(&self.sel_d)
                    .arg(&view.list_sel)
                    .arg(&view.list_offset)
                    .arg(&natoms_i)
                    .arg(&slots_i)
                    .arg(&list_slots_i)
                    .arg(&ntypes_i)
                    .arg(&self.rcut)
                    .arg(&mut table.index)
                    .arg(&mut table.delta)
                    .arg(&mut table.dist)
                    .arg(&mut dropped);
                unsafe { launch.launch(self.per_slot(natoms)) }
                    .map(drop)
                    .map_err(launch_error("slot table"))?;
                let filled = held_slots(&device.download(&table.index)?, &self.sel)?;
                (Some(table), filled)
            }
        };
        if self.kind == DescriptorKind::SeE3 {
            // Embedding rows: one per pair of an atom's neighbors.
            let (start, blocks, rows_embedding) = pair_rows(&filled, self.sel.len())?;
            // The rows go with the square of the neighbors, and a step holds
            // all of them at once: say so before the device runs out.
            let needed = rows_embedding.saturating_mul(self.bytes_per_row());
            let (free, _) = device.memory_info()?;
            if needed > free {
                let gib = |bytes: usize| bytes as f64 / (1u64 << 30) as f64;
                return Err(Error(format!(
                    "{}se_e3: the {rows_embedding} pairs of neighbors of this frame of {natoms} atoms need {:.1} GiB of device memory for their embedding rows, and {:.1} GiB are free; the trainer takes a frame in one piece (a shorter cutoff, fewer slots or narrower embedding nets need less)",
                    self.context,
                    gib(needed),
                    gib(free)
                )));
            }
            return Ok(Seen {
                table,
                layout: Layout::Pairs(device.upload(&start)?),
                blocks,
                rows_embedding,
            });
        }
        // Embedding rows: one per neighbor, and one per pair of types for
        // all its empty slots, whose input is the same normalized zero.
        let (row_table, blocks, rows_embedding) =
            embedding_rows(&filled, block, &self.sel, self.one_side)?;
        Ok(Seen {
            table,
            layout: Layout::Slots(device.upload(&row_table)?),
            blocks,
            rows_embedding,
        })
    }

    /// se_e3: θ, the weight and the net of every pair of an atom's
    /// neighbors, the rows of the embedding nets.
    fn pairs(
        &self,
        device: &Device,
        list: &NeighborList,
        seen: &Seen,
        start: &CudaSlice<i64>,
        atype: &CudaSlice<i32>,
    ) -> Result<Rows<P>> {
        let natoms = list.natoms;
        let (index, delta, dist) = seen.slots(list);
        let sel_inv = self
            .sel_inv
            .as_ref()
            .ok_or_else(|| Error("an se_e3 descriptor without the weights of its pairs".into()))?;
        let (zero_i, slots_i, ntypes_i) = (0i32, self.slots as i32, self.sel.len() as i32);
        let rows_i = seen.rows_embedding as i64;
        // A frame without a pair has one value that nothing reads: no
        // buffer of no length is asked of the driver.
        let rows = seen.rows_embedding.max(1);
        // SAFETY: the kernel writes every row; every row is some atom's.
        let mut input = unsafe { device.alloc_uninit::<P>(rows)? };
        let mut weight = unsafe { device.alloc_uninit::<P>(rows)? };
        let mut network = unsafe { device.alloc_uninit::<i32>(rows)? };
        let stream = device.stream();
        let mut launch = stream.launch_builder(&self.kernels.env);
        launch
            .arg(index)
            .arg(delta)
            .arg(dist)
            .arg(atype)
            .arg(&self.dstd_inv)
            .arg(&self.slot_type)
            .arg(&self.slot_local)
            .arg(&self.sel_d)
            .arg(sel_inv)
            .arg(&zero_i)
            .arg(&slots_i)
            .arg(&ntypes_i)
            .arg(&self.rcut_smth)
            .arg(&self.inv_range)
            .arg(start)
            .arg(&rows_i)
            .arg(&mut input)
            .arg(&mut weight)
            .arg(&mut network);
        let shared = pair_shared(self.slots, self.sel.len());
        unsafe { launch.launch(per_atom(natoms, shared)) }
            .map(drop)
            .map_err(launch_error("pairs"))?;
        Ok(Rows {
            env: None,
            input,
            network,
            weight: Some(weight),
        })
    }

    /// The normalized environment rows of every atom (se_e2_a) and, for the
    /// embedding nets, the radial entry and the net of every row; for se_e3
    /// the rows of the pairs ([`Self::pairs`]).
    fn environment(
        &self,
        device: &Device,
        list: &NeighborList,
        seen: &Seen,
        atype: &CudaSlice<i32>,
    ) -> Result<Rows<P>> {
        let row_table = match &seen.layout {
            Layout::Slots(row_table) => row_table,
            Layout::Pairs(start) => return self.pairs(device, list, seen, start, atype),
        };
        let natoms = list.natoms;
        let (index, delta, dist) = seen.slots(list);
        let (zero_i, n_i, slots_i) = (0i32, natoms as i32, self.slots as i32);
        let (ntypes_i, one_side_i) = (self.sel.len() as i32, i32::from(self.one_side));
        // What a kernel gets for a buffer it is not to touch.
        let no_buffer = 0u64;
        // SAFETY: the kernel writes every element of these buffers.
        let mut env = match self.kind {
            DescriptorKind::SeE2A => {
                Some(unsafe { device.alloc_uninit::<f64>(natoms * self.slots * 4)? })
            }
            _ => None,
        };
        let mut input = unsafe { device.alloc_uninit::<P>(seen.rows_embedding)? };
        let mut network = unsafe { device.alloc_uninit::<i32>(seen.rows_embedding)? };
        let stream = device.stream();
        let mut launch = stream.launch_builder(&self.kernels.env);
        launch
            .arg(index)
            .arg(delta)
            .arg(dist)
            .arg(atype)
            .arg(&self.davg)
            .arg(&self.dstd_inv)
            .arg(&self.slot_type)
            .arg(&self.slot_local)
            .arg(&self.sel_d)
            .arg(&self.offset_d)
            .arg(&zero_i)
            .arg(&n_i)
            .arg(&slots_i)
            .arg(&ntypes_i)
            .arg(&self.rcut_smth)
            .arg(&self.inv_range);
        match env.as_mut() {
            Some(env) => launch.arg(env),
            None => launch.arg(&no_buffer),
        };
        launch
            .arg(&mut input)
            .arg(&mut network)
            .arg(row_table)
            .arg(&one_side_i);
        unsafe { launch.launch(self.per_slot(natoms)) }
            .map(drop)
            .map_err(launch_error("environment"))?;
        Ok(Rows {
            env,
            input,
            network,
            weight: None,
        })
    }

    /// The row table of an se_e2_a or se_e2_r descriptor.
    fn row_table<'a>(&self, seen: &'a Seen) -> Result<&'a CudaSlice<i32>> {
        match &seen.layout {
            Layout::Slots(row_table) => Ok(row_table),
            Layout::Pairs(_) => Err(Error(format!(
                "an {} descriptor with the rows of pairs",
                self.kind.name()
            ))),
        }
    }

    /// The rows of the pairs of an se_e3 descriptor: the first of every
    /// atom, their number, and what every pair counts.
    fn pair_layout<'a>(
        &self,
        seen: &'a Seen,
        rows: &'a Rows<P>,
    ) -> Result<(&'a CudaSlice<i64>, i64, &'a CudaSlice<P>)> {
        match (&seen.layout, &rows.weight) {
            (Layout::Pairs(start), Some(weight)) => Ok((start, seen.rows_embedding as i64, weight)),
            _ => Err(Error(
                "an se_e3 descriptor without the rows of its pairs".into(),
            )),
        }
    }

    /// Blocks of the se_e3 kernels that sum over the rows of an atom: the
    /// warps that share the rows, with their sums in shared memory.
    fn per_atom_of_pairs(&self, natoms: usize) -> LaunchConfig {
        let warps = descriptor_warps::<P>(self.width);
        LaunchConfig {
            grid_dim: (natoms as u32, 1, 1),
            block_dim: (32 * warps as u32, 1, 1),
            shared_mem_bytes: (warps * self.width * std::mem::size_of::<P>()).next_multiple_of(8)
                as u32,
        }
    }

    /// The components of this descriptor of every atom from the embedding
    /// rows `g`, into their places in `descriptor` (`stride` components per
    /// atom). se_e2_a also gives B.
    #[allow(clippy::too_many_arguments)]
    fn descriptor(
        &self,
        device: &Device,
        seen: &Seen,
        natoms: usize,
        rows: &Rows<P>,
        g: &CudaSlice<P>,
        descriptor: &mut CudaSlice<P>,
        stride: usize,
    ) -> Result<Option<CudaSlice<f64>>> {
        let (n_i, slots_i, w_i) = (natoms as i32, self.slots as i32, self.width as i32);
        let (ntypes_i, stride_i) = (self.sel.len() as i32, stride as i32);
        let mut out = descriptor.slice_mut(self.first_feature..);
        let stream = device.stream();
        let mut launch = stream.launch_builder(&self.kernels.descriptor);
        if self.kind == DescriptorKind::SeE3 {
            let (start, rows_i, weight) = self.pair_layout(seen, rows)?;
            let zero_i = 0i32;
            launch
                .arg(g)
                .arg(&rows.input)
                .arg(weight)
                .arg(start)
                .arg(&rows_i)
                .arg(&zero_i)
                .arg(&n_i)
                .arg(&w_i)
                .arg(&mut out)
                .arg(&stride_i);
            unsafe { launch.launch(self.per_atom_of_pairs(natoms)) }
                .map(drop)
                .map_err(launch_error("descriptor"))?;
            return Ok(None);
        }
        let row_table = self.row_table(seen)?;
        let Some(env) = rows.env.as_ref() else {
            let scale = self.radial_scale();
            launch
                .arg(g)
                .arg(&self.slot_type)
                .arg(&self.slot_local)
                .arg(&self.sel_d)
                .arg(&self.offset_d)
                .arg(&n_i)
                .arg(&slots_i)
                .arg(&w_i)
                .arg(&scale)
                .arg(&mut out)
                .arg(row_table)
                .arg(&ntypes_i)
                .arg(&stride_i);
            unsafe { launch.launch(per_atom(natoms, 0)) }
                .map(drop)
                .map_err(launch_error("descriptor"))?;
            return Ok(None);
        };
        let axis_i = self.axis as i32;
        // SAFETY: the kernel writes every element.
        let mut b = unsafe { device.alloc_uninit::<f64>(natoms * self.width * 4)? };
        launch
            .arg(env)
            .arg(g)
            .arg(&self.slot_type)
            .arg(&self.slot_local)
            .arg(&self.sel_d)
            .arg(&self.offset_d)
            .arg(&n_i)
            .arg(&slots_i)
            .arg(&w_i)
            .arg(&axis_i)
            .arg(&mut b)
            .arg(&mut out)
            .arg(row_table)
            .arg(&ntypes_i)
            .arg(&stride_i);
        unsafe { launch.launch(per_atom(natoms, 4 * self.width * 8)) }
            .map(drop)
            .map_err(launch_error("descriptor"))?;
        Ok(Some(b))
    }

    /// Adds to `force` what this descriptor gives every atom, from dE/dD
    /// (`d_descriptor`, `stride` components per atom), and its share of the
    /// virial of every atom: written when `add` is false, added to what is
    /// there otherwise.
    #[allow(clippy::too_many_arguments)]
    fn forces(
        &self,
        device: &Device,
        list: &NeighborList,
        seen: &Seen,
        atype: &CudaSlice<i32>,
        rows: &Rows<P>,
        b: Option<&CudaSlice<f64>>,
        g: &CudaSlice<P>,
        dg: &CudaSlice<P>,
        d_descriptor: &CudaSlice<P>,
        stride: usize,
        add: bool,
        force: &mut CudaSlice<f64>,
        virial: &mut CudaSlice<f64>,
    ) -> Result<()> {
        let natoms = list.natoms;
        let (index, delta, dist) = seen.slots(list);
        let (zero_i, n_i, slots_i) = (0i32, natoms as i32, self.slots as i32);
        let (w_i, ntypes_i) = (self.width as i32, self.sel.len() as i32);
        let (stride_i, add_i) = (stride as i32, i32::from(add));
        let (axis_i, scale) = (self.axis as i32, self.radial_scale());
        let dd = d_descriptor.slice(self.first_feature..);
        let stream = device.stream();
        let mut launch = stream.launch_builder(&self.kernels.force);
        if self.kind == DescriptorKind::SeE3 {
            let (start, rows_i, weight) = self.pair_layout(seen, rows)?;
            let function = self.kernels.pair_force.as_ref().ok_or_else(|| {
                Error("an se_e3 descriptor without the kernel of its pairs".into())
            })?;
            // dE/dθ of every pair, then the forces from it.
            // SAFETY: the first kernel writes every row, and every row is
            // some atom's.
            let mut dtheta = unsafe { device.alloc_uninit::<P>(seen.rows_embedding.max(1))? };
            {
                let mut before = stream.launch_builder(function);
                before
                    .arg(g)
                    .arg(dg)
                    .arg(&rows.input)
                    .arg(weight)
                    .arg(&dd)
                    .arg(start)
                    .arg(&rows_i)
                    .arg(&zero_i)
                    .arg(&n_i)
                    .arg(&w_i)
                    .arg(&stride_i)
                    .arg(&mut dtheta);
                let config = LaunchConfig {
                    grid_dim: (natoms as u32, 1, 1),
                    block_dim: (PAIR_BLOCK, 1, 1),
                    shared_mem_bytes: pair_dtheta_shared::<P>(self.width) as u32,
                };
                unsafe { before.launch(config) }
                    .map(drop)
                    .map_err(launch_error("pair force"))?;
            }
            launch
                .arg(&dtheta)
                .arg(index)
                .arg(delta)
                .arg(dist)
                .arg(atype)
                .arg(&self.dstd_inv)
                .arg(&self.slot_type)
                .arg(&self.slot_local)
                .arg(&self.sel_d)
                .arg(start)
                .arg(&rows_i)
                .arg(&zero_i)
                .arg(&slots_i)
                .arg(&ntypes_i)
                .arg(&self.rcut_smth)
                .arg(&self.inv_range)
                .arg(force)
                .arg(virial)
                .arg(&n_i)
                .arg(&add_i);
            let config = LaunchConfig {
                grid_dim: (natoms as u32, 1, 1),
                block_dim: (PAIR_FORCE_BLOCK, 1, 1),
                shared_mem_bytes: pair_force_shared(self.slots, self.sel.len()) as u32,
            };
            return unsafe { launch.launch(config) }
                .map(drop)
                .map_err(launch_error("force"));
        }
        let row_table = self.row_table(seen)?;
        let shared = match (&rows.env, b) {
            (Some(env), Some(b)) => {
                launch.arg(env).arg(g).arg(dg).arg(b);
                (4 * self.width + 4 * self.slots + 12) * 8
                    + 4 * self.width * std::mem::size_of::<P>()
            }
            _ => {
                launch.arg(dg);
                radial_force_shared::<P>(self.slots, self.width)
            }
        };
        launch
            .arg(&dd)
            .arg(index)
            .arg(delta)
            .arg(dist)
            .arg(atype)
            .arg(&self.dstd_inv)
            .arg(&self.slot_type)
            .arg(&self.slot_local)
            .arg(&self.sel_d)
            .arg(&self.offset_d)
            .arg(&zero_i)
            .arg(&n_i)
            .arg(&slots_i)
            .arg(&w_i);
        // The axis neurons of se_e2_a, the scale of se_e2_r.
        match self.kind {
            DescriptorKind::SeE2A => launch.arg(&axis_i),
            _ => launch.arg(&scale),
        };
        launch
            .arg(&self.rcut_smth)
            .arg(&self.inv_range)
            .arg(force)
            .arg(virial)
            .arg(row_table)
            .arg(&ntypes_i)
            .arg(&stride_i)
            .arg(&add_i);
        unsafe { launch.launch(per_atom(natoms, shared)) }
            .map(drop)
            .map_err(launch_error("force"))
    }

    /// The tangent of this descriptor when every neighbor vector `d` moves
    /// along `c[j] - c[i] + wᵀd`: that of the environment rows, which is
    /// returned with that of B (se_e2_a) or gives that of θ of every pair
    /// (se_e3), and that of D, into its place in `ddot` (`stride`
    /// components per atom).
    #[allow(clippy::too_many_arguments)]
    fn tangent(
        &self,
        device: &Device,
        list: &NeighborList,
        seen: &Seen,
        atype: &CudaSlice<i32>,
        c: &CudaSlice<f64>,
        w: &CudaSlice<f64>,
        rows: &Rows<P>,
        b: Option<&CudaSlice<f64>>,
        g: &CudaSlice<P>,
        dg: &CudaSlice<P>,
        ddot: &mut CudaSlice<P>,
        stride: usize,
    ) -> Result<Tangents<P>> {
        let natoms = list.natoms;
        let (index, delta, dist) = seen.slots(list);
        let (n_i, slots_i, w_i) = (natoms as i32, self.slots as i32, self.width as i32);
        let (ntypes_i, stride_i) = (self.sel.len() as i32, stride as i32);
        let stream = device.stream();
        // SAFETY: the kernels write every element of these buffers.
        let mut envdot = unsafe { device.alloc_uninit::<f64>(natoms * self.slots * 4)? };
        {
            let mut launch = stream.launch_builder(&self.kernels.env_tangent);
            launch
                .arg(index)
                .arg(delta)
                .arg(dist)
                .arg(atype)
                .arg(&self.dstd_inv)
                .arg(c)
                .arg(w)
                .arg(&n_i)
                .arg(&slots_i)
                .arg(&self.rcut_smth)
                .arg(&self.inv_range)
                .arg(&mut envdot);
            unsafe { launch.launch(self.per_slot(natoms)) }
                .map(drop)
                .map_err(launch_error("environment tangent"))?;
        }
        let mut out = ddot.slice_mut(self.first_feature..);
        if self.kind == DescriptorKind::SeE3 {
            let (start, rows_i, weight) = self.pair_layout(seen, rows)?;
            let function = self.kernels.pair_tangent.as_ref().ok_or_else(|| {
                Error("an se_e3 descriptor without the kernel of its tangent".into())
            })?;
            // The tangent of θ of every pair, from that of the directions.
            // SAFETY: the kernel writes every row; every row is some atom's.
            let mut thetadot = unsafe { device.alloc_uninit::<P>(seen.rows_embedding.max(1))? };
            {
                let mut launch = stream.launch_builder(function);
                launch
                    .arg(index)
                    .arg(delta)
                    .arg(dist)
                    .arg(atype)
                    .arg(&self.dstd_inv)
                    .arg(&envdot)
                    .arg(&self.slot_type)
                    .arg(&self.slot_local)
                    .arg(&self.sel_d)
                    .arg(&slots_i)
                    .arg(&ntypes_i)
                    .arg(&self.rcut_smth)
                    .arg(&self.inv_range)
                    .arg(start)
                    .arg(&mut thetadot);
                let shared = pair_tangent_shared(self.slots, self.sel.len());
                unsafe { launch.launch(per_atom(natoms, shared)) }
                    .map(drop)
                    .map_err(launch_error("pair tangent"))?;
            }
            let mut launch = stream.launch_builder(&self.kernels.descriptor_tangent);
            launch
                .arg(g)
                .arg(dg)
                .arg(&rows.input)
                .arg(&thetadot)
                .arg(weight)
                .arg(start)
                .arg(&rows_i)
                .arg(&n_i)
                .arg(&w_i)
                .arg(&mut out)
                .arg(&stride_i);
            unsafe { launch.launch(self.per_atom_of_pairs(natoms)) }
                .map(drop)
                .map_err(launch_error("descriptor tangent"))?;
            return Ok(Tangents {
                envdot: None,
                bdot: None,
                thetadot: Some(thetadot),
            });
        }
        let row_table = self.row_table(seen)?;
        let mut launch = stream.launch_builder(&self.kernels.descriptor_tangent);
        let (Some(env), Some(b)) = (&rows.env, b) else {
            let scale = self.radial_scale();
            launch
                .arg(&envdot)
                .arg(dg)
                .arg(&self.slot_type)
                .arg(&self.slot_local)
                .arg(row_table)
                .arg(&ntypes_i)
                .arg(&slots_i)
                .arg(&w_i)
                .arg(&scale)
                .arg(&stride_i)
                .arg(&mut out);
            unsafe { launch.launch(per_atom(natoms, 0)) }
                .map(drop)
                .map_err(launch_error("descriptor tangent"))?;
            return Ok(Tangents {
                envdot: Some(envdot),
                bdot: None,
                thetadot: None,
            });
        };
        let axis_i = self.axis as i32;
        let mut bdot = unsafe { device.alloc_uninit::<f64>(natoms * self.width * 4)? };
        launch
            .arg(env)
            .arg(&envdot)
            .arg(g)
            .arg(dg)
            .arg(b)
            .arg(&self.slot_type)
            .arg(&self.slot_local)
            .arg(row_table)
            .arg(&ntypes_i)
            .arg(&slots_i)
            .arg(&w_i)
            .arg(&axis_i)
            .arg(&stride_i)
            .arg(&mut bdot)
            .arg(&mut out);
        unsafe { launch.launch(per_atom(natoms, 8 * self.width * 8)) }
            .map(drop)
            .map_err(launch_error("descriptor tangent"))?;
        Ok(Tangents {
            envdot: Some(envdot),
            bdot: Some(bdot),
            thetadot: None,
        })
    }

    /// Back through this descriptor: the adjoints of every embedding row
    /// and of its derivative along its input, from the adjoints of D and of
    /// its tangent (`dbar`, `ddbar`: `stride` components per atom). The
    /// frame has embedding rows.
    #[allow(clippy::too_many_arguments)]
    fn reverse(
        &self,
        device: &Device,
        seen: &Seen,
        natoms: usize,
        rows: &Rows<P>,
        tangents: &Tangents<P>,
        b: Option<&CudaSlice<f64>>,
        dbar: &CudaSlice<P>,
        ddbar: &CudaSlice<P>,
        stride: usize,
    ) -> Result<(CudaSlice<P>, CudaSlice<P>)> {
        let (slots_i, w_i) = (self.slots as i32, self.width as i32);
        let (ntypes_i, stride_i) = (self.sel.len() as i32, stride as i32);
        let (dbar, ddbar) = (
            dbar.slice(self.first_feature..),
            ddbar.slice(self.first_feature..),
        );
        let stream = device.stream();
        let mut launch = stream.launch_builder(&self.kernels.descriptor_reverse);
        if self.kind == DescriptorKind::SeE3 {
            let (start, _, weight) = self.pair_layout(seen, rows)?;
            let thetadot = tangents.thetadot.as_ref().ok_or_else(|| {
                Error("an se_e3 descriptor without the tangent of its pairs".into())
            })?;
            // SAFETY: the kernel writes every row, and every row is some
            // atom's.
            let mut gbar = unsafe { device.alloc_uninit::<P>(seen.rows_embedding * self.width)? };
            let mut gpbar = unsafe { device.alloc_uninit::<P>(seen.rows_embedding * self.width)? };
            launch
                .arg(&rows.input)
                .arg(thetadot)
                .arg(weight)
                .arg(&dbar)
                .arg(&ddbar)
                .arg(start)
                .arg(&w_i)
                .arg(&stride_i)
                .arg(&mut gbar)
                .arg(&mut gpbar);
            let config = LaunchConfig {
                grid_dim: (natoms as u32, 1, 1),
                block_dim: (PAIR_BLOCK, 1, 1),
                shared_mem_bytes: (2 * self.width * std::mem::size_of::<P>()).next_multiple_of(8)
                    as u32,
            };
            unsafe { launch.launch(config) }
                .map(drop)
                .map_err(launch_error("descriptor reverse"))?;
            return Ok((gbar, gpbar));
        }
        let row_table = self.row_table(seen)?;
        let envdot = tangents.envdot.as_ref().ok_or_else(|| {
            Error("a descriptor without the tangent of its environment rows".into())
        })?;
        let bdot = tangents.bdot.as_ref();
        // From zero: the rows of the empty slots collect sums.
        let mut gbar = device.zeros::<P>(seen.rows_embedding * self.width)?;
        let mut gpbar = device.zeros::<P>(seen.rows_embedding * self.width)?;
        let (axis_i, scale) = (self.axis as i32, self.radial_scale());
        let shared = match (&rows.env, b, bdot) {
            (Some(env), Some(b), Some(bdot)) => {
                launch
                    .arg(env)
                    .arg(envdot)
                    .arg(b)
                    .arg(bdot)
                    .arg(&dbar)
                    .arg(&ddbar)
                    .arg(&self.slot_type)
                    .arg(&self.slot_local)
                    .arg(&self.sel_d)
                    .arg(&self.offset_d)
                    .arg(row_table)
                    .arg(&ntypes_i)
                    .arg(&slots_i)
                    .arg(&w_i)
                    .arg(&axis_i);
                16 * self.width * 8
            }
            _ => {
                launch
                    .arg(envdot)
                    .arg(&dbar)
                    .arg(&ddbar)
                    .arg(&self.slot_type)
                    .arg(&self.slot_local)
                    .arg(&self.sel_d)
                    .arg(row_table)
                    .arg(&ntypes_i)
                    .arg(&slots_i)
                    .arg(&w_i)
                    .arg(&scale);
                0
            }
        };
        launch.arg(&stride_i).arg(&mut gbar).arg(&mut gpbar);
        unsafe { launch.launch(per_atom(natoms, shared)) }
            .map(drop)
            .map_err(launch_error("descriptor reverse"))?;
        Ok((gbar, gpbar))
    }
}

impl<P: Scalar> GpuTrainer<P> {
    pub fn new(model: &Model, device: Arc<Device>) -> Result<Self> {
        let metadata = model.validate()?;
        let d = &model.descriptor;
        // The parameters: the embedding nets of every descriptor in the
        // order of the descriptors, then the fitting nets.
        let mut theta = Vec::new();
        let mut parts = Vec::with_capacity(d.parts().len());
        let mut features = 0;
        for (number, part) in d.parts().iter().enumerate() {
            let context = match d.is_hybrid() {
                true => format!("descriptor.list[{number}]: "),
                false => String::new(),
            };
            let part = Part::new(
                &device, part, &d.sel, d.rcut, features, &mut theta, &context,
            )?;
            features += part.features;
            parts.push(part);
        }
        let mut fitting = Vec::with_capacity(model.fitting.len());
        for network in &model.fitting {
            let layers = pack(network, features, &mut theta);
            if layers.last().map(|l| l.output) != Some(1) {
                return Err(Error("fitting networks must end in one output".into()));
            }
            fitting.push(layers);
        }
        let fitting_set = TrainSet::new(&device, fitting.clone())?;
        let narrow = match parts.iter().any(|part| part.view.is_some()) {
            true => Some(device.kernels().function(MODEL_SOURCE, "hybrid_slots")?),
            false => None,
        };
        let adam_kernel = device
            .kernels()
            .function(SOURCE, &format!("adam_{}", P::SUFFIX))?;
        let adam_master = device.kernels().function(SOURCE, "adam_master")?;
        let cast: Vec<P> = theta.iter().map(|&v| P::from_f64(v)).collect();
        Ok(Self {
            metadata,
            model: model.clone(),
            parts,
            features,
            gradient: device.zeros::<P>(cast.len())?,
            theta: device.upload(&cast)?,
            master: match P::SUFFIX {
                "f32" => Some(device.upload(&theta)?),
                _ => None,
            },
            adam: None,
            fitting,
            fitting_set,
            adam_kernel,
            adam_master,
            narrow,
            profile: None,
            prepared: HashMap::new(),
            prepared_bytes: 0,
            prepared_limit: PREPARED_LIMIT,
            device,
        })
    }

    /// Device memory, in bytes, that may hold the neighbor lists of frames
    /// seen before (1 GiB unless set; 0 prepares every frame afresh). Frames
    /// beyond it are prepared at every visit. Frames kept so far stay.
    pub fn set_frame_cache(&mut self, bytes: usize) {
        self.prepared_limit = bytes;
    }

    /// Frames whose neighbor lists are kept.
    pub fn cached_frames(&self) -> usize {
        self.prepared.len()
    }

    /// Sorted atoms, neighbor list and embedding rows of `frame`: kept from
    /// an earlier visit when the frame is the same in every number.
    fn prepare(
        &mut self,
        frame: &Frame,
        types: &[i32],
        clock: &mut Clock,
    ) -> Result<Arc<Prepared>> {
        let key = frame_key(frame, types);
        if let Some(kept) = self.prepared.get(&key) {
            if kept.positions == frame.positions && kept.cell == frame.cell && kept.types == types {
                return Ok(Arc::clone(kept));
            }
        }
        let device = &*self.device;
        let d = &self.model.descriptor;
        let (natoms, ntypes) = (types.len(), d.sel.len());
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
        clock.lap(|t| &mut t.host_prepare)?;
        // One list for all descriptors: of a hybrid, with the largest of
        // their cutoffs and of every type the most slots any of them has.
        let list = NeighborList::build(device, &positions, &sorted_types, &cell, d.rcut, &d.sel)?;
        list.check(device)?;
        let mut parts = Vec::with_capacity(self.parts.len());
        for part in &self.parts {
            parts.push(part.see(device, &list, &block, self.narrow.as_ref())?);
        }
        let prepared = Arc::new(Prepared {
            positions: frame.positions.clone(),
            cell: frame.cell,
            types: types.to_vec(),
            order,
            block,
            atype,
            list,
            parts,
        });
        clock.lap(|t| &mut t.neighbors)?;
        let bytes = prepared.bytes();
        if self.prepared_bytes + bytes <= self.prepared_limit && !self.prepared.contains_key(&key) {
            self.prepared_bytes += bytes;
            self.prepared.insert(key, Arc::clone(&prepared));
        }
        Ok(prepared)
    }

    /// Turn per-stage timing on (a device synchronization after every stage,
    /// times start from zero) or off.
    pub fn set_profiling(&mut self, on: bool) {
        self.profile = on.then(TrainStageTimes::default);
    }

    /// Stage times summed since profiling was turned on.
    pub fn stage_times(&self) -> Option<&TrainStageTimes> {
        self.profile.as_ref()
    }

    /// Adds the time since `start` to the optimizer stage, when profiling.
    fn optimizer_lap(&mut self, start: Instant) -> Result<()> {
        if let Some(times) = &mut self.profile {
            self.device.synchronize()?;
            times.optimizer += start.elapsed().as_secs_f64() * 1e3;
        }
        Ok(())
    }

    /// Number of trainable parameters (weights, biases and timesteps).
    pub fn parameter_count(&self) -> usize {
        self.theta.len()
    }

    /// The parameters, embedding nets first (center type major; with
    /// `type_one_side` one per neighbor type; of se_e3 one per pair of
    /// neighbor types `t <= u`, `t` major; of a hybrid those of its
    /// descriptors in the order of its list), then the fitting nets; per
    /// layer weights `[input][output]`, bias, timestep.
    ///
    /// Float32 networks compute with these rounded to float32; the values
    /// here are the float64 ones the optimizer keeps.
    pub fn parameters(&self) -> Result<Vec<f64>> {
        if let Some(master) = &self.master {
            return self.device.download(master);
        }
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
        if self.master.is_some() {
            self.master = Some(self.device.upload(parameters)?);
        }
        Ok(())
    }

    /// The model with the current parameters.
    pub fn model(&self) -> Result<Model> {
        let theta = self.parameters()?;
        let mut model = self.model.clone();
        for (part, descriptor) in self.parts.iter().zip(model.descriptor.parts_mut()) {
            // Its nets in the order of the parameters; a net shared between
            // the central types goes to every row.
            let nets = part.embedding.len();
            let embedding = descriptor.embedding.iter_mut().flatten();
            for (place, network) in embedding.enumerate() {
                unpack(&mut network.layers, &part.embedding[place % nets], &theta);
            }
        }
        for (network, layout) in model.fitting.iter_mut().zip(&self.fitting) {
            unpack(&mut network.layers, layout, &theta);
        }
        model.validate()?;
        Ok(model)
    }

    pub fn zero_gradient(&mut self) -> Result<()> {
        let start = Instant::now();
        self.gradient = self.device.zeros::<P>(self.theta.len())?;
        self.optimizer_lap(start)
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
        let start = Instant::now();
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
        // Float32 networks: the step is added to the float64 parameters, of
        // which `theta` is the rounded copy.
        let mixed = self.master.is_some();
        let mut launch = match &mut self.master {
            Some(master) => {
                let mut launch = stream.launch_builder(&self.adam_master);
                launch.arg(master);
                launch
            }
            None => stream.launch_builder(&self.adam_kernel),
        };
        launch
            .arg(&mut self.theta)
            .arg(&self.gradient)
            .arg(&mut state.m)
            .arg(&mut state.v)
            .arg(&n_i);
        if mixed {
            launch.arg(&learning_rate);
        } else {
            launch.arg(&scalars[0]);
        }
        for scalar in &scalars[1..] {
            launch.arg(scalar);
        }
        let config = LaunchConfig {
            grid_dim: (n.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { launch.launch(config) }
            .map(drop)
            .map_err(launch_error("Adam"))?;
        self.optimizer_lap(start)
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
        let device = Arc::clone(&self.device);
        let mut clock = Clock {
            device: &device,
            times: self.profile.take(),
            last: Instant::now(),
        };
        let result = self.accumulate_frame(frame, labels, weights, scale, &mut clock);
        self.profile = clock.times;
        result
    }

    fn accumulate_frame(
        &mut self,
        frame: &Frame,
        labels: &FrameLabels,
        weights: &LossWeights,
        scale: f64,
        clock: &mut Clock,
    ) -> Result<(LossTerms, Evaluation)> {
        frame.validate()?;
        let device = Arc::clone(&self.device);
        let device = &*device;
        let types = frame.type_ids(&self.metadata)?;
        let natoms = types.len();
        if labels.forces.is_some_and(|f| f.len() != natoms) {
            return Err(Error("force labels do not match the atom count".into()));
        }
        let ntypes = self.model.descriptor.sel.len();
        // The components of all descriptors of an atom follow each other in
        // its row of D, of the tangent of D and of their adjoints.
        let features = self.features;

        let prepared = self.prepare(frame, &types, clock)?;
        let (order, block, list) = (&prepared.order, &prepared.block, &prepared.list);
        let atype = &prepared.atype;
        // Every descriptor with what it reads of the frame.
        let parts = self.parts.iter().zip(&prepared.parts);

        let mut rows = Vec::with_capacity(self.parts.len());
        for (part, seen) in parts.clone() {
            rows.push(part.environment(device, list, seen, atype)?);
        }
        clock.lap(|t| &mut t.environment)?;

        // Embedding with the unit tangent: G and dG/d(input); every row
        // names its net. A descriptor without a row (se_e3 where no atom has
        // a neighbor within its cutoff) has no pass.
        let mut tapes: Vec<Option<SetTape<P>>> = Vec::with_capacity(self.parts.len());
        for ((part, seen), rows) in parts.clone().zip(&rows) {
            tapes.push(match seen.rows_embedding {
                0 => None,
                count => Some(part.embedding_set.forward(
                    device,
                    &self.theta,
                    &rows.input,
                    &rows.network,
                    count,
                )?),
            });
        }
        // G and dG/d(input) of the rows of every descriptor; for one without
        // a row, a value that nothing reads.
        let blank = device.zeros::<P>(1)?;
        let embedded: Vec<(&CudaSlice<P>, &CudaSlice<P>)> = tapes
            .iter()
            .map(|tape| match tape {
                Some(tape) => (tape.output(), tape.tangent()),
                None => (&blank, &blank),
            })
            .collect();
        clock.lap(|t| &mut t.embedding)?;

        // SAFETY (all `alloc_uninit` below): each buffer is fully written by
        // the kernels or the copy that follow before anything reads it; of
        // the buffers of D and of what goes with it, every descriptor
        // writes its own components of every atom.
        let mut descriptor = unsafe { device.alloc_uninit::<P>(natoms * features)? };
        // B of every se_e2_a descriptor.
        let mut b = Vec::with_capacity(self.parts.len());
        for (((part, seen), rows), (g, _)) in parts.clone().zip(&rows).zip(&embedded) {
            b.push(part.descriptor(device, seen, natoms, rows, g, &mut descriptor, features)?);
        }
        clock.lap(|t| &mut t.descriptor)?;

        // First fitting pass: atomic energies and dE/dD for forces and virial.
        // (type, first atom, atoms) of every type present.
        let type_ranges: Vec<(usize, usize, usize)> = (0..ntypes)
            .filter(|&t| block[t + 1] > block[t])
            .map(|t| (t, block[t], block[t + 1] - block[t]))
            .collect();
        let zeros = device.zeros::<P>(natoms * features)?;
        // The seed of the first pass: dE/d(atomic energy) = 1.
        let ones = device.upload(&vec![P::ONE; natoms])?;
        let mut d_descriptor = unsafe { device.alloc_uninit::<P>(natoms * features)? };
        let mut scratch = unsafe { device.alloc_uninit::<P>(natoms * features)? };
        // Every atom names the fitting net of its type.
        let first_pass = self.fitting_set.forward_with_tangent(
            device,
            &self.theta,
            &descriptor,
            &zeros,
            atype,
            natoms,
        )?;
        self.fitting_set.backward_to_input(
            device,
            &self.theta,
            &mut self.gradient,
            false,
            &first_pass,
            atype,
            &ones,
            &zeros.slice(0..natoms),
            &type_ranges,
            &mut d_descriptor,
            &mut scratch,
        )?;
        let atomic = first_pass.output();
        clock.lap(|t| &mut t.fitting)?;
        // Every descriptor adds its forces; the virial of an atom is
        // written by the first and added to by the others.
        let mut force = device.zeros::<f64>(natoms * 3)?;
        let mut virial = unsafe { device.alloc_uninit::<f64>(natoms * 9)? };
        for (number, (((part, seen), rows), (g, dg))) in
            parts.clone().zip(&rows).zip(&embedded).enumerate()
        {
            part.forces(
                device,
                list,
                seen,
                atype,
                rows,
                b[number].as_ref(),
                g,
                dg,
                &d_descriptor,
                features,
                number > 0,
                &mut force,
                &mut virial,
            )?;
        }
        clock.lap(|t| &mut t.forces)?;

        // Prediction, loss and the seeds of the gradient pass (host, float64).
        let atomic_host = device.download(atomic)?;
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
        clock.lap(|t| &mut t.loss)?;
        let mut ddot = unsafe { device.alloc_uninit::<P>(natoms * features)? };
        // What the tangent pass of every descriptor leaves for its reverse
        // pass: the tangents of its environment rows and of B, or of θ.
        let mut tangents = Vec::with_capacity(self.parts.len());
        for (number, (((part, seen), rows), (g, dg))) in
            parts.clone().zip(&rows).zip(&embedded).enumerate()
        {
            tangents.push(part.tangent(
                device,
                list,
                seen,
                atype,
                &c,
                &w,
                rows,
                b[number].as_ref(),
                g,
                dg,
                &mut ddot,
                features,
            )?);
        }
        clock.lap(|t| &mut t.tangent)?;

        // Fitting with the tangent, then its reverse pass with seeds
        // dΦ/dE = a and dΦ/dĖ = −1.
        let energy_bar = device.upload(&vec![P::from_f64(energy_seed); natoms])?;
        let tangent_bar = device.upload(&vec![P::from_f64(-1.0); natoms])?;
        let mut dbar = unsafe { device.alloc_uninit::<P>(natoms * features)? };
        let mut ddbar = unsafe { device.alloc_uninit::<P>(natoms * features)? };
        let second_pass = self.fitting_set.forward_with_tangent(
            device,
            &self.theta,
            &descriptor,
            &ddot,
            atype,
            natoms,
        )?;
        self.fitting_set.backward_to_input(
            device,
            &self.theta,
            &mut self.gradient,
            true,
            &second_pass,
            atype,
            &energy_bar,
            &tangent_bar,
            &type_ranges,
            &mut dbar,
            &mut ddbar,
        )?;
        clock.lap(|t| &mut t.fitting_gradient)?;

        // Back through every descriptor to its embedding rows, then through
        // its embedding nets. One without a row has neither.
        let mut adjoints = Vec::with_capacity(self.parts.len());
        for (number, ((part, seen), rows)) in parts.clone().zip(&rows).enumerate() {
            adjoints.push(match &tapes[number] {
                None => None,
                Some(_) => Some(part.reverse(
                    device,
                    seen,
                    natoms,
                    rows,
                    &tangents[number],
                    b[number].as_ref(),
                    &dbar,
                    &ddbar,
                    features,
                )?),
            });
        }
        clock.lap(|t| &mut t.descriptor_reverse)?;
        for (number, ((part, seen), rows)) in parts.zip(&rows).enumerate() {
            let (Some(tape), Some((gbar, gpbar))) = (&tapes[number], &adjoints[number]) else {
                continue;
            };
            part.embedding_set.backward(
                device,
                &self.theta,
                &mut self.gradient,
                tape,
                &rows.network,
                gbar,
                gpbar,
                &seen.blocks,
            )?;
        }
        clock.lap(|t| &mut t.embedding_gradient)?;
        if let Some(times) = &mut clock.times {
            times.frames += 1;
        }
        Ok((terms, evaluation))
    }
}

#[cfg(test)]
mod tests {
    use super::embedding_rows;

    #[test]
    fn embedding_rows_follow_the_nets_and_share_empty_slots() {
        // Atoms 0 and 1 of type 0, atom 2 of type 1; two and three slots.
        // Neighbors held, per atom and neighbor type:
        let filled = [2, 1, 1, 3, 0, 0];
        let (table, nets, rows) = embedding_rows(&filled, &[0, 2, 3], &[2, 3], false).unwrap();
        // Neighbor type 0: net (0, 0) has the rows 0..3 of its neighbors and
        // the shared row 3, net (1, 0) only the shared row 4. Neighbor type
        // 1: net (0, 1) has 5..9 and the shared row 9, net (1, 1) row 10.
        assert_eq!(rows, 11);
        assert_eq!(nets, vec![(0, 0, 4), (2, 4, 1), (1, 5, 5), (3, 10, 1)]);
        // Per atom and neighbor type: first row, neighbors, shared row.
        assert_eq!(
            table,
            vec![0, 2, 3, 5, 1, 9, 2, 1, 3, 6, 3, 9, 4, 0, 4, 10, 0, 10]
        );
        // One net per neighbor type: the same rows, and those of a neighbor
        // type, shared rows included, are one net's.
        let (shared, nets, total) = embedding_rows(&filled, &[0, 2, 3], &[2, 3], true).unwrap();
        assert_eq!((shared, total), (table.clone(), 11));
        assert_eq!(nets, vec![(0, 0, 5), (1, 5, 6)]);
        // Full slots share no row, and a type without atoms has no net.
        let (table, nets, rows) = embedding_rows(&[1, 1], &[0, 1, 1], &[1, 1], false).unwrap();
        assert_eq!((table, rows), (vec![0, 1, -1, 1, 1, -1], 2));
        assert_eq!(nets, vec![(0, 0, 1), (1, 1, 1)]);
        // More neighbors than slots cannot be laid out.
        assert!(embedding_rows(&[3, 0], &[0, 1, 1], &[2, 1], false).is_err());
        let (_, nets, _) = embedding_rows(&[1, 1], &[0, 1, 1], &[1, 1], true).unwrap();
        assert_eq!(nets, vec![(0, 0, 1), (1, 1, 1)]);
    }
}
