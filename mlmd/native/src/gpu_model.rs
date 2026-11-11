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
//! neighbors than `sel`, this path selects neighbors like DeePMD-kit 3 (the
//! nearest `sum(sel)` of any type, then at most `sel[t]` per type);
//! [`GpuPotential::last_truncated`] reports when that dropped neighbors.
use crate::{Activation, Model, Network, Residual};
use omnimd_gpu::{
    Activation as DeviceActivation, Cell, CudaFunction, CudaSlice, CudaViewMut, DenseLayer,
    DenseNetwork, Device, GradientNetwork, LaunchConfig, LayerParameters, NeighborList,
    NeighborSearch, PushKernelArg, Scalar, ScalarNetwork, ScalarTable, TableSegments,
};
use omnimd_ml_contract::{
    Committee, Error, Evaluation, Frame, ModelMetadata, Potential, Result, Units,
};
use std::sync::Arc;
use std::time::Instant;

pub(crate) const SOURCE: &str = r#"
// A division or a square root in double precision costs the device as much
// as some sixty multiplications, so the kernels multiply by reciprocals that
// are worked out once (`inv_range` = 1 / (rmax - rmin), `dstd_inv` = 1 / dstd,
// one 1 / r per neighbor) and read the neighbor distances from the neighbor
// search instead of taking the root again.
__device__ void switch_fn(double r, double rmin, double inv_range, double* q, double* dq) {
    if (r < rmin) { *q = 1.0; *dq = 0.0; return; }
    double t = (r - rmin) * inv_range, z = 1.0 - t;
    *q = z * z * z * (1.0 + 3.0 * t + 6.0 * t * t);
    *dq = -30.0 * t * t * z * z * inv_range;
}

__device__ long long embed_row(int count, int i, int t, int local, const int* sel, const int* offset) {
    return (long long)count * offset[t] + (long long)i * sel[t] + local;
}

// Row of the embedding input and output of slot `local` of neighbor type `t`
// of atom `i`. Without `rows` every slot has a row of its own. With it (per
// atom and neighbor type three ints: first row, neighbors, shared row) only
// the neighbors do, and all empty slots of one pair of types share a row:
// their input, the normalized zero, is the same.
__device__ long long slot_row(const int* rows, int ntypes, int count, int i, int t, int local,
        const int* sel, const int* offset) {
    if (!rows) return embed_row(count, i, t, local, sel, offset);
    const int* entry = rows + 3 * ((long long)i * ntypes + t);
    return local < entry[1] ? entry[0] + local : entry[2];
}

// Several models of one architecture (a committee) go through these kernels
// together: the second dimension of the launch grid is the model. Model `m`
// reads the same neighbor lists, has its statistics and networks at center
// type `m * ntypes + type`, and its work rows and results after those of the
// models before it. A grid one model deep is the single model.
//
// One thread per slot. Without `rows` an empty slot only marks its row as
// one to skip (network id -1): the embedding kernels leave such rows alone
// and the descriptor adds what the empty slots of a type contribute in one
// term. With `rows` the empty slots write the normalized zero to the row
// they share.
template <typename P>
__device__ void env_impl(const int* index, const double* delta, const double* dist,
        const int* atype, const double* davg, const double* dstd_inv, const int* slot_type,
        const int* slot_local, const int* sel, const int* offset, int first, int count, int slots,
        int ntypes, double rmin, double inv_range, double* env, P* radial, int* network,
        const int* rows) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)count * slots) return;
    int model = blockIdx.y;
    long long base = (long long)model * count * slots;
    int li = (int)(idx / slots), s = (int)(idx % slots), i = first + li;
    int c = model * ntypes + atype[i];
    long long slot = (long long)i * slots + s;
    long long row = base
        + slot_row(rows, ntypes, count, li, slot_type[s], slot_local[s], sel, offset);
    bool neighbor = index[slot] >= 0;
    if (!rows && !neighbor) {
        network[row] = -1;
        return;
    }
    double r4[4] = {0.0, 0.0, 0.0, 0.0};
    if (neighbor) {
        const double* d = delta + 3 * slot;
        double r = dist[slot], inv = 1.0 / r, q, dq;
        switch_fn(r, rmin, inv_range, &q, &dq);
        double sw = q * inv;
        r4[0] = sw;
        for (int a = 0; a < 3; ++a) r4[a + 1] = sw * (d[a] * inv);
    }
    for (int a = 0; a < 4; ++a) {
        r4[a] = (r4[a] - davg[4 * c + a]) * dstd_inv[4 * c + a];
        env[4 * (base + idx) + a] = r4[a];
    }
    // Every empty slot of a shared row writes the same two values there.
    radial[row] = (P)r4[0];
    // Embedding net of this row: (model, center type, neighbor type).
    network[row] = c * ntypes + slot_type[s];
}

// One block per atom: B[m][a] = sum_s G[s][m] env[s][a] / slots (float64),
// D = B·B[:axis]ᵀ in the network precision. Without `rows` only the
// neighbors are summed slot by slot (they fill the first slots of their
// type); the empty slots of a type all hold the same row, the embedding
// `g_empty` of the normalized zero, and enter as one term.
template <typename P>
__device__ void descriptor_impl(const double* env, const P* g, const int* slot_type,
        const int* slot_local, const int* sel, const int* offset, int count, int slots,
        int width, int axis, double* b_out, P* d_out, const int* rows, int ntypes,
        const int* index, const int* atype, const P* g_empty, const double* davg,
        const double* dstd_inv, int first) {
    extern __shared__ double b[];
    int i = blockIdx.x, model = blockIdx.y;
    // This model's rows of the embedding and of the environment, and its
    // place among the centers.
    g += (long long)model * count * slots * width;
    env += (long long)model * count * slots * 4;
    long long center = (long long)model * count + i;
    const double inv_slots = 1.0 / slots;
    for (int m = threadIdx.x; m < width; m += blockDim.x) {
        double acc[4] = {0.0, 0.0, 0.0, 0.0};
        if (rows) {
            for (int s = 0; s < slots; ++s) {
                double gm = (double)g[slot_row(rows, ntypes, count, i, slot_type[s], slot_local[s], sel, offset) * width + m];
                const double* r = env + ((long long)i * slots + s) * 4;
                for (int a = 0; a < 4; ++a) acc[a] += gm * r[a];
            }
        } else {
            int c = model * ntypes + atype[first + i];
            const int* mine = index + (long long)(first + i) * slots;
            for (int t = 0; t < ntypes; ++t) {
                int filled = 0;
                for (; filled < sel[t] && mine[offset[t] + filled] >= 0; ++filled) {
                    double gm = (double)g[embed_row(count, i, t, filled, sel, offset) * width + m];
                    const double* r = env + ((long long)i * slots + offset[t] + filled) * 4;
                    for (int a = 0; a < 4; ++a) acc[a] += gm * r[a];
                }
                if (filled < sel[t]) {
                    double gm = (sel[t] - filled) * (double)g_empty[((long long)c * ntypes + t) * width + m];
                    for (int a = 0; a < 4; ++a)
                        acc[a] += gm * ((0.0 - davg[4 * c + a]) * dstd_inv[4 * c + a]);
                }
            }
        }
        for (int a = 0; a < 4; ++a) {
            double v = acc[a] * inv_slots;
            b[4 * m + a] = v;
            b_out[(center * width + m) * 4 + a] = v;
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < width * axis; e += blockDim.x) {
        int m = e / axis, k = e % axis;
        double v = 0.0;
        for (int a = 0; a < 4; ++a) v += b[4 * m + a] * b[4 * k + a];
        d_out[center * width * axis + e] = (P)v;
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
        const P* dd_in, const int* index, const double* delta, const double* dist,
        const int* atype, const double* dstd_inv, const int* slot_type, const int* slot_local,
        const int* sel, const int* offset, int first, int count, int slots, int width, int axis,
        double rmin, double inv_range, double* force, double* virial, const int* rows,
        int ntypes, int natoms) {
    extern __shared__ double shared[];
    double* b = shared;                  // [width][4]
    double* dr_all = b + 4 * width;      // [slots][4]
    double* red = dr_all + 4 * slots;    // [12]: center force, virial
    P* db = (P*)(red + 12);              // [4][width]
    int li = blockIdx.x, i = first + li, model = blockIdx.y;
    // This model's work rows and its place among the centers; its forces
    // and virials follow those of the models before it.
    long long before = (long long)model * count * slots;
    env += before * 4;
    g += before * width;
    dg += before * width;
    long long center = (long long)model * count + li;
    force += 3 * (long long)model * natoms;
    virial += 9 * (long long)model * natoms;
    for (int e = threadIdx.x; e < 4 * width; e += blockDim.x) b[e] = b_in[center * width * 4 + e];
    for (int e = threadIdx.x; e < 12; e += blockDim.x) red[e] = 0.0;
    __syncthreads();
    const P* dd = dd_in + center * width * axis;
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
    const double inv_slots = 1.0 / slots;
    for (int s = warp; s < slots; s += warps) {
        long long slot = (long long)i * slots + s;
        if (index[slot] < 0) continue;
        long long row = slot_row(rows, ntypes, count, li, slot_type[s], slot_local[s], sel, offset) * width;
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
            for (int a = 0; a < 4; ++a) dr_all[4 * s + a] = (double)dr[a] * inv_slots;
            dr_all[4 * s] += (double)tangent * inv_slots;
        }
    }
    __syncthreads();
    const double* inv_std = dstd_inv + 4 * (model * ntypes + atype[i]);
    double fi[3] = {0.0, 0.0, 0.0}, vir[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int s = threadIdx.x; s < slots; s += blockDim.x) {
        long long slot = (long long)i * slots + s;
        int j = index[slot];
        if (j < 0) continue;
        const double* d = delta + 3 * slot;
        double rr = dist[slot], inv = 1.0 / rr, q, dq;
        switch_fn(rr, rmin, inv_range, &q, &dq);
        double u[3] = {d[0] * inv, d[1] * inv, d[2] * inv};
        // The row is (sw, sw u) with sw = q / r, so its derivative along c is
        // dsw u_c for the first entry and dsw u_a u_c + (sw / r)(delta_ac -
        // u_a u_c) for the others. Contracted with the adjoint row e over
        // the normalization, that is u_c [e_0 dsw + (dsw - sw / r)(e · u)]
        // + (sw / r) e_c.
        double sw = q * inv, dsw = (dq - sw) * inv, across = sw * inv;
        double adj[4];
        for (int a = 0; a < 4; ++a) adj[a] = dr_all[4 * s + a] * inv_std[a];
        double along = adj[0] * dsw + (dsw - across) * (adj[1] * u[0] + adj[2] * u[1] + adj[3] * u[2]);
        for (int c = 0; c < 3; ++c) {
            double grad = u[c] * along + across * adj[c + 1];
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

// The `_rows` kernels take the row table of `slot_row` (and the number of
// types it is laid out by) and are launched one model deep; the others skip
// empty slots and take the number of atoms a model's forces span.
#define INSTANTIATE(P, S)                                                                         \
extern "C" __global__ void se_e2_a_env_##S(const int* index, const double* delta,                 \
        const double* dist, const int* atype, const double* davg, const double* dstd_inv,         \
        const int* slot_type, const int* slot_local, const int* sel, const int* offset,           \
        int first, int count, int slots, int ntypes, double rmin, double inv_range, double* env,  \
        P* radial, int* network) {                                                                \
    env_impl<P>(index, delta, dist, atype, davg, dstd_inv, slot_type, slot_local, sel, offset,    \
                first, count, slots, ntypes, rmin, inv_range, env, radial, network, nullptr);     \
}                                                                                                 \
extern "C" __global__ void se_e2_a_env_rows_##S(const int* index, const double* delta,            \
        const double* dist, const int* atype, const double* davg, const double* dstd_inv,         \
        const int* slot_type, const int* slot_local, const int* sel, const int* offset,           \
        int first, int count, int slots, int ntypes, double rmin, double inv_range, double* env,  \
        P* radial, int* network, const int* rows) {                                               \
    env_impl<P>(index, delta, dist, atype, davg, dstd_inv, slot_type, slot_local, sel, offset,    \
                first, count, slots, ntypes, rmin, inv_range, env, radial, network, rows);        \
}                                                                                                 \
extern "C" __global__ void se_e2_a_descriptor_##S(const double* env, const P* g,                  \
        const int* slot_type, const int* slot_local, const int* sel, const int* offset,           \
        int count, int slots, int width, int axis, double* b_out, P* d_out, int ntypes,           \
        const int* index, const int* atype, const P* g_empty, const double* davg,                 \
        const double* dstd_inv, int first) {                                                      \
    descriptor_impl<P>(env, g, slot_type, slot_local, sel, offset, count, slots, width, axis,     \
                       b_out, d_out, nullptr, ntypes, index, atype, g_empty, davg, dstd_inv,      \
                       first);                                                                    \
}                                                                                                 \
extern "C" __global__ void se_e2_a_descriptor_rows_##S(const double* env, const P* g,             \
        const int* slot_type, const int* slot_local, const int* sel, const int* offset,           \
        int count, int slots, int width, int axis, double* b_out, P* d_out, const int* rows,      \
        int ntypes) {                                                                             \
    descriptor_impl<P>(env, g, slot_type, slot_local, sel, offset, count, slots, width, axis,     \
                       b_out, d_out, rows, ntypes, nullptr, nullptr, nullptr, nullptr, nullptr,   \
                       0);                                                                        \
}                                                                                                 \
extern "C" __global__ void se_e2_a_backward_##S(const double* env, const P* g, const P* dg,       \
        const double* b_in, const P* dd_in, const int* index, const double* delta,                \
        const double* dist, const int* atype, const double* dstd_inv, const int* slot_type,       \
        const int* slot_local, const int* sel, const int* offset, int first, int count,           \
        int slots, int width, int axis, double rmin, double inv_range, double* force,             \
        double* virial, int ntypes, int natoms) {                                                 \
    backward_impl<P>(env, g, dg, b_in, dd_in, index, delta, dist, atype, dstd_inv, slot_type,     \
                     slot_local, sel, offset, first, count, slots, width, axis, rmin, inv_range,  \
                     force, virial, nullptr, ntypes, natoms);                                     \
}                                                                                                 \
extern "C" __global__ void se_e2_a_backward_rows_##S(const double* env, const P* g,               \
        const P* dg, const double* b_in, const P* dd_in, const int* index, const double* delta,   \
        const double* dist, const int* atype, const double* dstd_inv, const int* slot_type,       \
        const int* slot_local, const int* sel, const int* offset, int first, int count,           \
        int slots, int width, int axis, double rmin, double inv_range, double* force,             \
        double* virial, const int* rows, int ntypes) {                                            \
    backward_impl<P>(env, g, dg, b_in, dd_in, index, delta, dist, atype, dstd_inv, slot_type,     \
                     slot_local, sel, offset, first, count, slots, width, axis, rmin, inv_range,  \
                     force, virial, rows, ntypes, 0);                                             \
}
INSTANTIATE(float, f32)
INSTANTIATE(double, f64)
"#;

const BLOCK: u32 = 128;
/// Static shared-memory limit per block without opting in to more.
const SHARED_LIMIT: usize = 48 * 1024;
/// The embedding tables continue below `min_distance` down to
/// `min_distance / TAIL_REACH` with `TAIL_STEP_FACTOR` times the step
/// (DeePMD's defaults: extrapolate 5, stride ratio 10).
const TAIL_REACH: f64 = 5.0;
const TAIL_STEP_FACTOR: f64 = 10.0;
/// Float32 chunks of at most this many atoms per center type run the fitting
/// nets in the fused kernel instead of layer-by-layer cuBLAS, whose fixed cost
/// (a dozen launches per type) dominates below that. Float64 always uses the
/// fused kernel: both paths are bound by the double-precision units there and
/// the fused one has no fixed cost. Measured on an RTX 4090 with
/// [240, 240, 240] fitting nets on 1600 descriptor components (fitting stage,
/// 2 types: 192 atoms 0.20 vs 0.32 ms, 1536 atoms 0.89 vs 0.30 ms).
const FUSED_FITTING_ATOMS_PER_TYPE_F32: usize = 256;
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

impl StageTimes {
    /// Adds the times of another evaluation.
    pub fn add(&mut self, other: &Self) {
        self.host_prepare += other.host_prepare;
        self.neighbors += other.neighbors;
        self.environment += other.environment;
        self.embedding += other.embedding;
        self.descriptor += other.descriptor;
        self.fitting += other.fitting;
        self.forces += other.forces;
        self.download += other.download;
        self.chunks += other.chunks;
    }

    /// The mean over `evaluations` added times.
    pub fn mean(&self, evaluations: usize) -> Self {
        let n = evaluations.max(1) as f64;
        Self {
            host_prepare: self.host_prepare / n,
            neighbors: self.neighbors / n,
            environment: self.environment / n,
            embedding: self.embedding / n,
            descriptor: self.descriptor / n,
            fitting: self.fitting / n,
            forces: self.forces / n,
            download: self.download / n,
            chunks: self.chunks / evaluations.max(1),
        }
    }
}

pub struct GpuPotential<P: Scalar> {
    device: Arc<Device>,
    metadata: ModelMetadata,
    /// Models evaluated together (a committee); one as a rule. The tables
    /// below that go by center type hold model `m` at center type `m *
    /// ntypes + type`.
    models: usize,
    rcut: f64,
    rcut_smth: f64,
    sel: Vec<usize>,
    slots: usize,
    width: usize,
    axis: usize,
    /// Energy bias per center type.
    bias: Vec<f64>,
    /// Every embedding net in one set; net `center type · ntypes + neighbor
    /// type` (the ids the environment kernel writes).
    embedding: ScalarNetwork<P>,
    /// Host copies of the embedding nets and the radial (mean, std) per
    /// center type, for building tables.
    embedding_host: Vec<Vec<Network>>,
    radial_normalization: Vec<(f64, f64)>,
    /// Tabulated embedding nets in the same order, when compressed.
    tables: Option<ScalarTable<P>>,
    table_fallbacks: usize,
    /// Layer-by-layer fitting nets per center type (cuBLAS): best for many
    /// atoms.
    fitting: Vec<DenseNetwork<P>>,
    /// All fitting nets in one kernel, forward and backward: best for few
    /// atoms, where the launches of the layer-by-layer path dominate.
    fused_fitting: Option<GradientNetwork<P>>,
    /// Chunks of at most this many atoms use `fused_fitting`.
    fused_fitting_max_atoms: usize,
    davg: CudaSlice<f64>,
    /// Reciprocals of the normalization widths: the kernels multiply.
    dstd_inv: CudaSlice<f64>,
    /// 1 / (rcut - rcut_smth), for the switching function.
    inv_range: f64,
    /// The embedding of the input every empty slot has (the normalized
    /// zero) for each pair of center type and neighbor type,
    /// `[ntypes²][width]`. Empty slots are not evaluated one by one: the
    /// descriptor adds this row times their number. It comes from the
    /// networks also when the neighbors go through tables.
    empty_g: CudaSlice<P>,
    slot_type: CudaSlice<i32>,
    slot_local: CudaSlice<i32>,
    sel_d: CudaSlice<i32>,
    offset_d: CudaSlice<i32>,
    kernels: Kernels,
    /// Dynamic shared memory of the force kernel, bytes.
    backward_shared: usize,
    chunk_atoms: usize,
    /// Neighbor search with its device buffers, kept between evaluations.
    search: NeighborSearch,
    last_truncated: bool,
    profile: Option<StageTimes>,
}

/// Part of a buffer of 64-bit words seen as `len` values of `S`.
///
/// # Safety
/// `S` must be plain numbers no more aligned than a 64-bit word.
unsafe fn view_as<'a, S>(
    words: &'a mut CudaViewMut<'_, u64>,
    len: usize,
) -> Result<CudaViewMut<'a, S>> {
    words
        .transmute_mut(len)
        .ok_or_else(|| Error("result buffer is smaller than its layout".into()))
}

/// The `index`th int of a buffer of 64-bit words, as it lies in memory.
fn int_at(words: &[u64], index: usize) -> i32 {
    let bytes = words[index / 2].to_ne_bytes();
    let mut raw = [0; 4];
    raw.copy_from_slice(&bytes[4 * (index % 2)..][..4]);
    i32::from_ne_bytes(raw)
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

pub(crate) fn launch_error(what: &'static str) -> impl FnOnce(omnimd_gpu::DriverError) -> Error {
    move |e| Error(format!("CUDA {what} launch failed: {e:?}"))
}

pub(crate) fn copy_error(e: omnimd_gpu::DriverError) -> Error {
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
    slots * (4 * 8 + p + 4 + 2 * width * p)
        + width * 4 * 8
        + 2 * width * axis * p
        + 4 * fitting_width * p
}

impl<P: Scalar> GpuPotential<P> {
    pub fn new(model: &Model, device: Arc<Device>) -> Result<Self> {
        Self::committee(&[model], device)
    }

    /// Several models as one potential that evaluates them together (see
    /// [`Self::evaluate_frames`]): a committee, for model deviation. The
    /// models must share the type map, the descriptor's cutoffs, `sel` and
    /// `axis_neuron` and the shapes of their networks; weights, statistics
    /// and energy biases are each model's own. As a [`Potential`] the
    /// committee is its first model.
    pub fn committee(models: &[&Model], device: Arc<Device>) -> Result<Self> {
        let Some(&first) = models.first() else {
            return Err(Error("a committee needs at least one model".into()));
        };
        let mut metadata = first.validate()?;
        metadata.backend = "native-gpu-se-e2-a".into();
        // Network precision; geometry, forces and virial are float64 either way.
        metadata.precision = if std::mem::size_of::<P>() == 4 {
            "float32".into()
        } else {
            "float64".into()
        };
        let d = &first.descriptor;
        let ntypes = first.type_map.len();
        let width = d.embedding[0][0].output_size();
        let slots: usize = d.sel.iter().sum();
        for (index, model) in models.iter().enumerate().skip(1) {
            model.validate()?;
            let other = &model.descriptor;
            if model.type_map != first.type_map
                || other.rcut != d.rcut
                || other.rcut_smth != d.rcut_smth
                || other.sel != d.sel
                || other.axis_neuron != d.axis_neuron
                || other.embedding[0][0].output_size() != width
            {
                return Err(Error(format!(
                    "committee member {index} differs from the first in its type map, cutoffs, sel, axis_neuron or embedding width; such models are evaluated one by one"
                )));
            }
        }
        let same_shape = |e: Error| match models.len() {
            1 => e,
            _ => Error(format!(
                "the members of a committee need networks of one shape: {e}"
            )),
        };
        let backward_shared =
            (4 * width + 4 * slots + 12) * 8 + 4 * width * std::mem::size_of::<P>();
        if backward_shared > SHARED_LIMIT {
            return Err(Error(format!(
                "GPU se_e2_a needs {backward_shared} bytes of shared memory per atom                  (embedding width {width}, {slots} neighbor slots); the limit is {SHARED_LIMIT}"
            )));
        }
        // Model `m` has its nets and statistics at center type `m * ntypes +
        // type`, as the kernels look them up.
        let embedding_host: Vec<Vec<Network>> = models
            .iter()
            .flat_map(|model| model.descriptor.embedding.iter().cloned())
            .collect();
        let embedding_layers: Vec<_> = embedding_host
            .iter()
            .flatten()
            .map(|n| layer_parameters::<P>(n, 1))
            .collect();
        let embedding = ScalarNetwork::new_set(&device, &embedding_layers).map_err(same_shape)?;
        let features = width * d.axis_neuron;
        let fitting_host: Vec<&Network> = models.iter().flat_map(|model| &model.fitting).collect();
        let fitting_layers: Vec<_> = fitting_host
            .iter()
            .map(|n| layer_parameters::<P>(n, features))
            .collect();
        if models.len() > 1 {
            // One shape for all, whether or not the fused kernel takes it.
            let shape = |layers: &[LayerParameters<P>]| {
                let of = |l: &LayerParameters<P>| (l.input, l.output, l.activation, l.residual);
                layers.iter().map(of).collect::<Vec<_>>()
            };
            // Each model's net of a type against the first model's.
            let differ = fitting_layers
                .iter()
                .enumerate()
                .any(|(net, layers)| shape(layers) != shape(&fitting_layers[net % ntypes]));
            if differ {
                return Err(same_shape(Error("the fitting nets differ".into())));
            }
        }
        // Not every fitting net fits in shared memory; those keep cuBLAS only.
        let fused_fitting = GradientNetwork::new_set(&device, &fitting_layers).ok();
        let fitting = fitting_host
            .iter()
            .map(|n| dense_network(&device, n, features))
            .collect::<Result<Vec<_>>>()?;
        let normalization = || models.iter().flat_map(|model| &model.descriptor.normalization);
        let davg: Vec<f64> = normalization().flat_map(|n| n.mean).collect();
        let dstd_inv: Vec<f64> = normalization()
            .flat_map(|n| n.std.map(|std| 1.0 / std))
            .collect();
        // As the environment kernel normalizes an empty slot.
        let centers = models.len() * ntypes;
        let empty_input: Vec<P> = (0..centers)
            .flat_map(|c| {
                let zero = P::from_f64((0.0 - davg[4 * c]) * dstd_inv[4 * c]);
                std::iter::repeat_n(zero, ntypes)
            })
            .collect();
        let empty_ids: Vec<i32> = (0..(centers * ntypes) as i32).collect();
        let empty_input = device.upload(&empty_input)?;
        let empty_ids = device.upload(&empty_ids)?;
        let mut empty_g = device.zeros::<P>(centers * ntypes * width)?;
        {
            let mut tangent = device.zeros::<P>(centers * ntypes * width)?;
            embedding.forward_tangent_by_id(
                &device,
                &empty_input,
                &empty_ids,
                centers * ntypes,
                &mut empty_g,
                &mut tangent,
            )?;
        }
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
        let fitting_width = first
            .fitting
            .iter()
            .flat_map(|n| n.layers.iter().map(|l| l.bias.len()))
            .sum::<usize>();
        let (free, _) = device.memory_info()?;
        let per_atom = bytes_per_atom::<P>(slot_type.len(), width, d.axis_neuron, fitting_width);
        let chunk_atoms = ((free as f64 * MEMORY_FRACTION) as usize / per_atom.max(1)).max(1);
        Ok(Self {
            metadata,
            models: models.len(),
            rcut: d.rcut,
            rcut_smth: d.rcut_smth,
            sel: d.sel.clone(),
            slots: slot_type.len(),
            width,
            axis: d.axis_neuron,
            bias: models
                .iter()
                .flat_map(|model| model.atom_energy_bias.iter().copied())
                .collect(),
            embedding,
            embedding_host,
            radial_normalization: normalization().map(|n| (n.mean[0], n.std[0])).collect(),
            tables: None,
            table_fallbacks: 0,
            fitting,
            fused_fitting,
            fused_fitting_max_atoms: if std::mem::size_of::<P>() == 4 {
                FUSED_FITTING_ATOMS_PER_TYPE_F32 * ntypes
            } else {
                usize::MAX
            },
            davg: device.upload(&davg)?,
            dstd_inv: device.upload(&dstd_inv)?,
            inv_range: 1.0 / (d.rcut - d.rcut_smth),
            empty_g,
            slot_type: device.upload(&slot_type)?,
            slot_local: device.upload(&slot_local)?,
            sel_d: device.upload(&sel_i)?,
            offset_d: device.upload(&offsets)?,
            kernels,
            backward_shared,
            chunk_atoms,
            search: NeighborSearch::new(d.rcut, &d.sel)?,
            last_truncated: false,
            profile: None,
            device,
        })
    }

    /// Models this potential evaluates together: one unless it was made by
    /// [`Self::committee`].
    pub fn models(&self) -> usize {
        self.models
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
    /// The members of a committee share it: each chunk holds this many
    /// atoms divided by the members evaluated.
    pub fn chunk_atoms(&self) -> usize {
        self.chunk_atoms
    }

    /// Override the chunk size derived from free device memory at load time.
    pub fn set_chunk_atoms(&mut self, atoms: usize) {
        self.chunk_atoms = atoms.max(1);
    }

    /// Model compression (DeePMD `dp compress`): replaces every embedding
    /// net by a piecewise quintic table with knots every `step` in the
    /// normalized radial input, covering neighbor distances from
    /// `min_distance` out to the cutoff. With `step = 0.01` the table agrees
    /// with the network to rounding level. Like DeePMD's extrapolation
    /// region, the tables continue with ten times the step down to a fifth
    /// of `min_distance`. A frame with a pair closer than that is evaluated
    /// with the exact networks instead (see [`Self::table_fallbacks`]), so
    /// results never depend on the range. Empty neighbor slots do not go
    /// through the tables: their row is the networks' own.
    pub fn compress(&mut self, min_distance: f64, step: f64) -> Result<()> {
        if !(min_distance.is_finite() && 0.0 < min_distance && min_distance < self.rcut) {
            return Err(Error(format!(
                "compression needs 0 < min_distance < rcut, got {min_distance}"
            )));
        }
        if !(step.is_finite() && 1e-4 <= step && step <= 1.0) {
            return Err(Error(format!(
                "table step must be in [1e-4, 1], got {step}"
            )));
        }
        // s(r) = switch(r)/r, the unnormalized radial input at distance r.
        let radial = |r: f64| {
            let t = ((r - self.rcut_smth) / (self.rcut - self.rcut_smth)).max(0.0);
            (t * t * t * (-6.0 * t * t + 15.0 * t - 10.0) + 1.0) / r
        };
        let width = self.width;
        // Coefficients of `count` intervals of width `h` starting at `start`.
        let segment = |network: &Network, start: f64, h: f64, count: usize| -> Result<Vec<f64>> {
            let (mut values, mut first, mut second) = (Vec::new(), Vec::new(), Vec::new());
            for knot in 0..=count {
                let [v, d, dd] = network.scalar_jet(start + knot as f64 * h)?;
                values.extend(v);
                first.extend(d);
                second.extend(dd);
            }
            omnimd_gpu::quintic_coefficients(h, width, &values, &first, &second)
        };
        let mut tables = Vec::new();
        for (row, &(mean, std)) in self.embedding_host.iter().zip(&self.radial_normalization) {
            // Empty slots and the cutoff sit at s = 0, the lower end.
            let lower = -mean / std;
            let intervals = (((radial(min_distance) - mean) / std - lower) / step)
                .ceil()
                .max(1.0) as usize;
            let upper = lower + intervals as f64 * step;
            let tail_step = TAIL_STEP_FACTOR * step;
            let tail = (((radial(min_distance / TAIL_REACH) - mean) / std - upper) / tail_step)
                .ceil()
                .max(1.0) as usize;
            for network in row {
                tables.push(TableSegments {
                    lower,
                    step,
                    coefficients: segment(network, lower, step, intervals)?,
                    tail_step,
                    tail_coefficients: segment(network, upper, tail_step, tail)?,
                });
            }
        }
        self.tables = Some(ScalarTable::new_set(&self.device, width, &tables)?);
        Ok(())
    }

    pub fn is_compressed(&self) -> bool {
        self.tables.is_some()
    }

    /// Chunks of at most this many atoms evaluate the fitting nets in one
    /// fused kernel; larger chunks use layer-by-layer cuBLAS. Zero disables
    /// the fused kernel.
    pub fn set_fused_fitting_max_atoms(&mut self, atoms: usize) {
        self.fused_fitting_max_atoms = atoms;
    }

    /// Turns the tiled first layer of the fused fitting kernel on or off
    /// (see `GradientNetwork::forward_gradient_grouped`) and returns whether
    /// it is on afterwards. The default follows the size of that layer and
    /// the precision; both settings give the same result to rounding.
    pub fn set_tiled_fitting(&mut self, on: bool) -> bool {
        self.fused_fitting
            .as_mut()
            .is_some_and(|fused| fused.set_tiled(on))
    }

    /// Evaluations that left the table range and used the exact networks.
    pub fn table_fallbacks(&self) -> usize {
        self.table_fallbacks
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
        self.evaluate_frames(&[frame], 1)?
            .pop()
            .and_then(|mut members| members.pop())
            .ok_or_else(|| Error("evaluation returned no result".into()))
    }

    /// Evaluates every frame with the first `members` models in one pass
    /// over the device: `[frame][member]`, each as that model would give it
    /// for that frame alone (to rounding: sums over atoms may run in another
    /// order). The frames need not be alike in atoms or cell. All their
    /// atoms are one list on the device, so a batch costs the launches of a
    /// single frame, and each frame's neighbors are searched once for all
    /// members.
    ///
    /// An error of any frame fails the batch and names the frame. When a
    /// frame of a batch leaves the range of the embedding tables, the frames
    /// are evaluated again one at a time, so that only those outside the
    /// tables use the exact networks (for all members).
    pub fn evaluate_frames(
        &mut self,
        frames: &[&Frame],
        members: usize,
    ) -> Result<Vec<Vec<Evaluation>>> {
        if members == 0 || members > self.models {
            return Err(Error(format!(
                "this potential evaluates 1..={} models together, not {members}",
                self.models
            )));
        }
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let device = Arc::clone(&self.device);
        let mut clock = Clock {
            device: &device,
            times: self.profile.take().map(|_| StageTimes::default()),
            last: Instant::now(),
        };
        let mut result = self.run(&device, frames, members, &mut clock, self.tables.is_some());
        if matches!(result, Ok(None)) {
            if frames.len() > 1 {
                self.profile = clock.times;
                let mut all = Vec::with_capacity(frames.len());
                for (index, frame) in frames.iter().enumerate() {
                    let one = self
                        .evaluate_frames(std::slice::from_ref(frame), members)
                        .map_err(|e| Error(format!("frame {index} of the batch: {e}")))?;
                    all.extend(one);
                }
                return Ok(all);
            }
            // Some input was outside the tables: use the exact networks.
            self.table_fallbacks += 1;
            result = self.run(&device, frames, members, &mut clock, false);
        }
        self.profile = clock.times;
        result?.ok_or_else(|| Error("exact evaluation reported a table range".into()))
    }

    /// `None` when `compressed` and an embedding input left the table range.
    fn run(
        &mut self,
        device: &Device,
        frames: &[&Frame],
        members: usize,
        clock: &mut Clock,
        compressed: bool,
    ) -> Result<Option<Vec<Vec<Evaluation>>>> {
        let ntypes = self.sel.len();
        let batch = frames.len() > 1;
        let in_batch = |index: usize, e: Error| match batch {
            true => Error(format!("frame {index} of the batch: {e}")),
            false => e,
        };
        let mut frame_types = Vec::with_capacity(frames.len());
        // First atom of each center type among all atoms of the batch.
        let mut block = vec![0usize; ntypes + 1];
        for (index, frame) in frames.iter().enumerate() {
            let types = frame
                .type_ids(&self.metadata)
                .map_err(|e| in_batch(index, e))?;
            for &t in &types {
                block[t as usize + 1] += 1;
            }
            frame_types.push(types);
        }
        for t in 0..ntypes {
            block[t + 1] += block[t];
        }
        let natoms = block[ntypes];
        if i32::try_from(natoms.saturating_mul(members)).is_err() {
            return Err(Error(format!(
                "a batch of {natoms} atoms for {members} models is too large for one pass"
            )));
        }
        // All atoms in one list, sorted by type so that each center type is
        // a contiguous block; within a type the frames follow each other and
        // the atoms of a frame keep their order.
        let mut cursor = block.clone();
        let mut source = vec![(0u32, 0u32); natoms];
        let mut positions = vec![[0.0; 3]; natoms];
        let mut sorted_types = vec![0i32; natoms];
        let mut system = vec![0i32; if batch { natoms } else { 0 }];
        for (index, (frame, types)) in frames.iter().zip(&frame_types).enumerate() {
            for (atom, &t) in types.iter().enumerate() {
                let k = cursor[t as usize];
                cursor[t as usize] += 1;
                source[k] = (index as u32, atom as u32);
                positions[k] = frame.positions[atom];
                sorted_types[k] = t;
                if batch {
                    system[k] = index as i32;
                }
            }
        }
        let cells: Vec<Cell> = frames
            .iter()
            .map(|frame| frame.cell.map_or(Cell::Open, Cell::Periodic))
            .collect();
        let stream = device.stream();
        clock.lap(|t| &mut t.host_prepare)?;

        // Everything that goes back to the host, in one buffer for a single
        // copy: the forces of each model [members][natoms][3] and the
        // table-range flag (cleared together), the virial of each center
        // [members][natoms][9], the neighbor status of each atom (ints) and
        // the atomic energies (`P`, chunk by chunk, see below).
        let centers = members * natoms;
        let cleared_words = 3 * centers + 1;
        let status_words = natoms.div_ceil(2);
        let energy_words = (centers * std::mem::size_of::<P>()).div_ceil(8);
        let mut results = unsafe {
            device.alloc_uninit::<u64>(cleared_words + 9 * centers + status_words + energy_words)?
        };
        let (mut cleared, mut rest) = results.split_at_mut(cleared_words);
        stream.memset_zeros(&mut cleared).map_err(copy_error)?;
        let (mut force_words, mut flag_words) = cleared.split_at_mut(3 * centers);
        let (mut virial_words, mut rest) = rest.split_at_mut(9 * centers);
        let (mut status_part, mut energy_part) = rest.split_at_mut(status_words);
        // SAFETY: float64, float32 and int32 are plain numbers that fit the
        // alignment of the words they are laid over.
        let mut force = unsafe { view_as::<f64>(&mut force_words, 3 * centers)? };
        let mut outside_table = unsafe { view_as::<i32>(&mut flag_words, 1)? };
        let mut virial = unsafe { view_as::<f64>(&mut virial_words, 9 * centers)? };
        let mut status = unsafe { view_as::<i32>(&mut status_part, natoms)? };
        let mut atomic = unsafe { view_as::<P>(&mut energy_part, centers)? };

        let list: &NeighborList = self.search.run_systems_mirrored(
            device,
            &positions,
            &sorted_types,
            &system,
            &cells,
            &mut status,
        )?;
        let atype = &list.types;
        clock.lap(|t| &mut t.neighbors)?;

        let (slots, width, axis) = (self.slots, self.width, self.axis);
        let features = width * axis;
        // Atoms per chunk: every model works on a chunk at once.
        let chunk = (self.chunk_atoms / members).clamp(1, natoms);
        let wide = members * chunk;
        // Per-chunk work buffers, reused by every chunk. Each element a chunk
        // reads was written earlier in the same chunk.
        let mut env = unsafe { device.alloc_uninit::<f64>(wide * slots * 4)? };
        let mut radial = unsafe { device.alloc_uninit::<P>(wide * slots)? };
        let mut network = unsafe { device.alloc_uninit::<i32>(wide * slots)? };
        let mut g = unsafe { device.alloc_uninit::<P>(wide * slots * width)? };
        let mut dg = unsafe { device.alloc_uninit::<P>(wide * slots * width)? };
        let mut b = unsafe { device.alloc_uninit::<f64>(wide * width * 4)? };
        let mut descriptor = unsafe { device.alloc_uninit::<P>(wide * features)? };
        let mut d_descriptor = unsafe { device.alloc_uninit::<P>(wide * features)? };
        // Adjoint seed of the layer-by-layer fitting path, made on first use.
        let mut ones: Option<CudaSlice<P>> = None;
        let (slots_i, w_i, axis_i) = (slots as i32, width as i32, axis as i32);
        let (ntypes_i, natoms_i) = (ntypes as i32, natoms as i32);
        let deep = members as u32;
        let tables = self.tables.as_ref().filter(|_| compressed);

        for first in (0..natoms).step_by(chunk) {
            let end = (first + chunk).min(natoms);
            let count = end - first;
            let (first_i, count_i) = (first as i32, count as i32);
            // Center-type ranges of this chunk, in chunk-local atoms.
            let ranges: Vec<(usize, usize, usize)> = (0..ntypes)
                .filter_map(|t| {
                    let (lo, hi) = (block[t].max(first), block[t + 1].min(end));
                    // Lazy: `hi - first` underflows for a type that ends before the chunk.
                    (hi > lo).then(|| (t, lo - first, hi - first))
                })
                .collect();
            {
                let mut launch = stream.launch_builder(&self.kernels.env);
                launch
                    .arg(&list.index)
                    .arg(&list.delta)
                    .arg(&list.dist)
                    .arg(atype)
                    .arg(&self.davg)
                    .arg(&self.dstd_inv)
                    .arg(&self.slot_type)
                    .arg(&self.slot_local)
                    .arg(&self.sel_d)
                    .arg(&self.offset_d)
                    .arg(&first_i)
                    .arg(&count_i)
                    .arg(&slots_i)
                    .arg(&ntypes_i)
                    .arg(&self.rcut_smth)
                    .arg(&self.inv_range)
                    .arg(&mut env)
                    .arg(&mut radial)
                    .arg(&mut network);
                let total = count * slots;
                let config = LaunchConfig {
                    grid_dim: (total.div_ceil(BLOCK as usize) as u32, deep, 1),
                    block_dim: (BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { launch.launch(config) }
                    .map(drop)
                    .map_err(launch_error("environment"))?;
            }
            clock.lap(|t| &mut t.environment)?;

            // Embedding values and d/d(radial) tangents for every row of the
            // chunk in one launch, [model][neighbor type][local atom][slot][width].
            {
                let rows = members * count * slots;
                let x = radial.slice(0..rows);
                let ids = network.slice(0..rows);
                let mut out = g.slice_mut(0..rows * width);
                let mut dout = dg.slice_mut(0..rows * width);
                match tables {
                    Some(tables) => tables.forward_tangent_by_id(
                        device,
                        &x,
                        &ids,
                        rows,
                        &mut out,
                        &mut dout,
                        &mut outside_table,
                    )?,
                    None => self
                        .embedding
                        .forward_tangent_by_id(device, &x, &ids, rows, &mut out, &mut dout)?,
                }
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
                    .arg(&mut descriptor)
                    .arg(&ntypes_i)
                    .arg(&list.index)
                    .arg(atype)
                    .arg(&self.empty_g)
                    .arg(&self.davg)
                    .arg(&self.dstd_inv)
                    .arg(&first_i);
                let config = LaunchConfig {
                    grid_dim: (count as u32, deep, 1),
                    block_dim: (BLOCK, 1, 1),
                    shared_mem_bytes: (4 * width * 8) as u32,
                };
                unsafe { launch.launch(config) }
                    .map(drop)
                    .map_err(launch_error("descriptor"))?;
            }
            clock.lap(|t| &mut t.descriptor)?;

            // Fitting: atomic energies and dE/dD, the centers of a chunk as
            // [model][local atom]. Few atoms: every model and center type
            // in one fused kernel. Many atoms: cuBLAS per model and center
            // type, with the energies copied out so the tape can go.
            let limit = self.fused_fitting_max_atoms;
            let fused = self.fused_fitting.as_mut().filter(|_| count <= limit);
            // The atoms are sorted by type: runs of one fitting net each.
            let runs = (0..members).flat_map(|model| {
                ranges
                    .iter()
                    .map(move |&(t, lo, hi)| (model * ntypes + t, model * count + lo, hi - lo))
            });
            let mut energies = atomic.slice_mut(members * first..members * end);
            if let Some(fused) = fused {
                let x = descriptor.slice(0..members * count * features);
                let groups: Vec<(usize, usize, usize)> = runs.collect();
                let mut dx = d_descriptor.slice_mut(0..members * count * features);
                fused.forward_gradient_grouped(
                    device,
                    &x,
                    &groups,
                    members * count,
                    &mut energies,
                    &mut dx,
                )?;
            } else {
                let ones = match &ones {
                    Some(ones) => ones,
                    None => ones.insert(device.upload(&vec![P::ONE; chunk])?),
                };
                for (net, lo, rows) in runs {
                    let hi = lo + rows;
                    let x = descriptor.slice(lo * features..hi * features);
                    let tape = self.fitting[net].forward_record(device, &x, rows)?;
                    let mut dx = d_descriptor.slice_mut(lo * features..hi * features);
                    self.fitting[net].backward(device, &tape, &ones.slice(0..rows), &mut dx)?;
                    stream
                        .memcpy_dtod(&tape.output, &mut energies.slice_mut(lo..hi))
                        .map_err(copy_error)?;
                }
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
                    .arg(&list.dist)
                    .arg(atype)
                    .arg(&self.dstd_inv)
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
                    .arg(&self.inv_range)
                    .arg(&mut force)
                    .arg(&mut virial)
                    .arg(&ntypes_i)
                    .arg(&natoms_i);
                let config = LaunchConfig {
                    grid_dim: (count as u32, deep, 1),
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

        let words = device.download(&results)?;
        clock.lap(|t| &mut t.download)?;
        let (cleared, rest) = words.split_at(cleared_words);
        let (virial_words, rest) = rest.split_at(9 * centers);
        let (status_part, energy_part) = rest.split_at(status_words);
        // The neighbor status of each frame: its atoms' bits together.
        let mut status = vec![0; frames.len()];
        for (k, &(frame, _)) in source.iter().enumerate() {
            status[frame as usize] |= int_at(status_part, k);
        }
        self.last_truncated = false;
        for (index, &bits) in status.iter().enumerate() {
            self.last_truncated |= NeighborList::status(bits).map_err(|e| in_batch(index, e))?;
        }
        if tables.is_some() && int_at(&cleared[3 * centers..], 0) != 0 {
            return Ok(None);
        }
        let atomic = P::from_words(energy_part, centers);
        // The energy of an atom for a model: the energies of a chunk are
        // [model][local atom], chunk after chunk.
        let energy_at = |model: usize, k: usize| {
            let first = k - k % chunk;
            let count = chunk.min(natoms - first);
            atomic[members * first + model * count + (k - first)].to_f64()
        };
        let mut all: Vec<Vec<Evaluation>> = frames
            .iter()
            .map(|frame| {
                (0..members)
                    .map(|_| Evaluation {
                        energy: 0.0,
                        forces: vec![[0.0; 3]; frame.positions.len()],
                        virial: [[0.0; 3]; 3],
                        units: Units::EvAngstrom,
                    })
                    .collect()
            })
            .collect();
        // Sums of one type at a time, then its bias, as for a single frame.
        let mut sums = vec![0.0; frames.len() * members];
        let mut counts = vec![0usize; frames.len()];
        for t in 0..ntypes {
            sums.fill(0.0);
            counts.fill(0);
            for (k, &(frame, _)) in source.iter().enumerate().take(block[t + 1]).skip(block[t]) {
                let frame = frame as usize;
                counts[frame] += 1;
                for model in 0..members {
                    sums[frame * members + model] += energy_at(model, k);
                }
            }
            for (frame, evaluations) in all.iter_mut().enumerate() {
                for (model, evaluation) in evaluations.iter_mut().enumerate() {
                    evaluation.energy += sums[frame * members + model]
                        + self.bias[model * ntypes + t] * counts[frame] as f64;
                }
            }
        }
        for (k, &(frame, atom)) in source.iter().enumerate() {
            for (model, evaluation) in all[frame as usize].iter_mut().enumerate() {
                let center = model * natoms + k;
                evaluation.forces[atom as usize] =
                    std::array::from_fn(|c| f64::from_bits(cleared[3 * center + c]));
                for (e, v) in virial_words[9 * center..9 * center + 9].iter().enumerate() {
                    evaluation.virial[e / 3][e % 3] += f64::from_bits(*v);
                }
            }
        }
        for (index, (frame, evaluations)) in frames.iter().zip(&all).enumerate() {
            for evaluation in evaluations {
                evaluation
                    .validate(frame.positions.len())
                    .map_err(|e| in_batch(index, e))?;
            }
        }
        Ok(Some(all))
    }
}

/// Runtime choices for [`open`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuOptions {
    /// Run the networks in float32; geometry, forces and virial stay float64.
    pub float32: bool,
    /// Tabulate the embedding nets: (shortest neighbor distance in Å the
    /// tables cover, table step). See [`GpuPotential::compress`].
    pub compress: Option<(f64, f64)>,
}

/// Opens `model` on CUDA device `ordinal` behind the [`Potential`] trait, for
/// callers that do not care about the network precision type.
pub fn open(model: &Model, ordinal: usize, options: GpuOptions) -> Result<Box<dyn Potential>> {
    fn build<P: Scalar>(
        model: &Model,
        device: Arc<Device>,
        options: GpuOptions,
    ) -> Result<Box<dyn Potential>> {
        let mut potential = GpuPotential::<P>::new(model, device)?;
        if let Some((min_distance, step)) = options.compress {
            potential.compress(min_distance, step)?;
        }
        Ok(Box::new(potential))
    }
    let device = Arc::new(Device::new(ordinal)?);
    if options.float32 {
        build::<f32>(model, device, options)
    } else {
        build::<f64>(model, device, options)
    }
}

/// Whether `models` can be one committee on the device (see
/// [`GpuPotential::committee`]): the reason when they cannot. Checked on the
/// host, before any device is opened.
pub fn committee_mismatch(models: &[&Model]) -> Option<String> {
    let Some(&first) = models.first() else {
        return Some("no models".into());
    };
    let shape = |network: &Network| -> Vec<(usize, usize, u8, u8)> {
        network
            .layers
            .iter()
            .map(|layer| {
                (
                    layer.weights.len(),
                    layer.bias.len(),
                    layer.activation as u8,
                    layer.residual as u8,
                )
            })
            .collect()
    };
    let d = &first.descriptor;
    for (index, model) in models.iter().enumerate().skip(1) {
        let other = &model.descriptor;
        let what = if model.type_map != first.type_map {
            "type map"
        } else if other.rcut != d.rcut || other.rcut_smth != d.rcut_smth {
            "cutoffs"
        } else if other.sel != d.sel {
            "sel"
        } else if other.axis_neuron != d.axis_neuron {
            "axis_neuron"
        } else if other.embedding.len() != d.embedding.len()
            || other
                .embedding
                .iter()
                .flatten()
                .map(shape)
                .ne(d.embedding.iter().flatten().map(shape))
        {
            "embedding nets"
        } else if model.fitting.len() != first.fitting.len()
            || model
                .fitting
                .iter()
                .map(shape)
                .ne(first.fitting.iter().map(shape))
        {
            "fitting nets"
        } else {
            continue;
        };
        return Some(format!(
            "model {index} differs from the first in its {what}"
        ));
    }
    None
}

/// Opens `models` on CUDA device `ordinal` as one committee that evaluates
/// them together (see [`GpuPotential::committee`]).
pub fn open_committee(
    models: &[&Model],
    ordinal: usize,
    options: GpuOptions,
) -> Result<Box<dyn Committee>> {
    fn build<P: Scalar>(
        models: &[&Model],
        device: Arc<Device>,
        options: GpuOptions,
    ) -> Result<Box<dyn Committee>> {
        let mut potential = GpuPotential::<P>::committee(models, device)?;
        if let Some((min_distance, step)) = options.compress {
            potential.compress(min_distance, step)?;
        }
        Ok(Box::new(potential))
    }
    let device = Arc::new(Device::new(ordinal)?);
    if options.float32 {
        build::<f32>(models, device, options)
    } else {
        build::<f64>(models, device, options)
    }
}

impl<P: Scalar> Potential for GpuPotential<P> {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn eval(&mut self, frame: &Frame) -> Result<Evaluation> {
        self.evaluate(frame)
    }

    fn eval_batch(&mut self, frames: &[&Frame]) -> Result<Vec<Evaluation>> {
        let all = self.evaluate_frames(frames, 1)?;
        Ok(all.into_iter().flatten().collect())
    }
}

impl<P: Scalar> Committee for GpuPotential<P> {
    fn members(&self) -> usize {
        self.models
    }

    fn eval_members(&mut self, frames: &[&Frame]) -> Result<Vec<Vec<Evaluation>>> {
        self.evaluate_frames(frames, self.models)
    }
}
