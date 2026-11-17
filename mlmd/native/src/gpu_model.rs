//! se_e2_a, se_e2_r, se_e3 and hybrid inference on the omnimd-gpu runtime
//! (plan W6–W7).
//!
//! Pipeline per frame, all on the device after one upload:
//! cell-list neighbors (DeePMD truncation to `sel`) → environment matrix →
//! embedding nets per (center type, neighbor type), fused with their tangents →
//! descriptor `D = B·B[:, :axis]ᵀ` → fitting nets per center type with a
//! recorded reverse pass → forces and virial from the chain rule back to the
//! neighbor deltas. Atoms are sorted by type so every network sees
//! contiguous rows; results are returned in the caller's atom order.
//!
//! An se_e2_r model goes the same way with a descriptor of its own: a fifth
//! of the mean of the embedding rows, which needs the radial entry of the
//! environment only (see [`crate::DescriptorKind`]).
//!
//! An se_e3 model has other rows for its embedding nets: not the neighbor
//! slots of an atom but the pairs of its neighbors, whose number goes with
//! the square of the neighbors an atom has. They are laid out from the
//! numbers of neighbors (one small download per evaluation), and a chunk
//! of atoms ends where its rows would be too many; the embedding nets
//! themselves are the same kernel.
//!
//! A hybrid model has several of these descriptors, whose components one
//! after the other are the input of the fitting nets. They read one neighbor
//! list, made with the largest of their cutoffs and the most slots of every
//! type: a descriptor with fewer slots or a shorter cutoff has a table of
//! the list in its own layout (the nearest of each type, none at or beyond
//! its cutoff), and its kernels run on that as they run on the list of a
//! model of one descriptor. Every chunk of atoms goes through the rows, the
//! embedding nets and the components of each descriptor, then through the
//! fitting nets once, then back through each descriptor to the forces.
//!
//! `P` is the network precision (ADR-3). Geometry is always float64: the
//! neighbor vectors, the environment rows as they are worked out, and their
//! Jacobian with the sums that give forces and virial. What lies between is
//! in `P`: the embedding and fitting nets, the environment rows as they are
//! stored, `B`, the descriptor and their adjoints. With `P = f32` no
//! arithmetic in double precision is left between the environment matrix
//! and its adjoint.
//!
//! Unlike the CPU `NativePotential`, which rejects frames with more
//! neighbors than `sel`, this path selects neighbors like DeePMD-kit 3 (the
//! nearest `sum(sel)` of any type, then at most `sel[t]` per type);
//! [`GpuPotential::last_truncated`] reports when that dropped neighbors.
use crate::{Activation, Descriptor, DescriptorKind, Model, Network, Residual, RADIAL_RESCALE};
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
// `P` is the precision of the networks, `E` that of what lies between them:
// the environment rows as they are stored, B and the sums that make it, and
// the adjoint of B. The potential runs with E = P, so that a float32 model
// does no arithmetic in double precision between its geometry and its
// forces (on this card double precision has a sixty-fourth of the rate, and
// the descriptor kernel was bound by it); the trainer keeps E = double.
// Geometry is double in either case: the rows are worked out in double
// before they are stored, and their Jacobian for the forces is double.
//
// One thread per slot. Without `rows` an empty slot only marks its row as
// one to skip (network id -1): the embedding kernels leave such rows alone
// and the descriptor adds what the empty slots of a type contribute in one
// term. With `rows` the empty slots write the normalized zero to the row
// they share. `one_side` names the embedding net of a row by its neighbor
// type alone: the nets of a trainer whose model has one per neighbor type.
template <typename P, typename E>
__device__ void env_impl(const int* index, const double* delta, const double* dist,
        const int* atype, const double* davg, const double* dstd_inv, const int* slot_type,
        const int* slot_local, const int* sel, const int* offset, int first, int count, int slots,
        int ntypes, double rmin, double inv_range, E* env, P* radial, int* network,
        const int* rows, int one_side) {
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
    // (A descriptor that takes the radial entry only has no use for the
    // rows: it passes no `env`.)
    for (int a = 0; a < 4; ++a) {
        r4[a] = (r4[a] - davg[4 * c + a]) * dstd_inv[4 * c + a];
        if (env) env[4 * (base + idx) + a] = (E)r4[a];
    }
    // Every empty slot of a shared row writes the same two values there.
    radial[row] = (P)r4[0];
    // Embedding net of this row: (model, center type, neighbor type), or the
    // neighbor type when the nets are the same for every center.
    network[row] = one_side ? slot_type[s] : c * ntypes + slot_type[s];
}

// How many neighbors of each type an atom has, counted by all threads of its
// block for use after the next __syncthreads. `mine` are the atom's slots.
// The neighbors of a type fill its first slots: the last of them knows their
// number, and an empty first slot that there are none. A type without slots
// has none either. One thread writes for each type.
__device__ void count_neighbors(const int* mine, const int* slot_type, const int* slot_local,
        const int* sel, int slots, int ntypes, int* filled) {
    for (int t = threadIdx.x; t < ntypes; t += blockDim.x)
        if (sel[t] == 0) filled[t] = 0;
    for (int s = threadIdx.x; s < slots; s += blockDim.x) {
        int t = slot_type[s], local = slot_local[s];
        bool here = mine[s] >= 0;
        if (here && (local + 1 == sel[t] || mine[s + 1] < 0)) filled[t] = local + 1;
        if (!here && local == 0) filled[t] = 0;
    }
}

// One block per atom: B[m][a] = sum_s G[s][m] env[s][a] / slots in `E`,
// D = B·B[:axis]ᵀ stored in the network precision. Without `rows` only the
// neighbors are summed slot by slot (they fill the first slots of their
// type), the sum of each neighbor type apart so that a sum in single
// precision stays short; the empty slots of a type all hold the same row,
// the embedding `g_empty` of the normalized zero `empty_row` (per center
// type, as the environment kernel normalizes it), and enter as one term.
// The atom's environment rows and its number of neighbors of each type go
// to shared memory first: every thread needs them for every neighbor, and
// from global memory those loads were what bound the sums in single
// precision.
//
// The components of a center go to `d_out` every `stride` values: its own
// number for a model of this descriptor, and for a hybrid the number of all
// its descriptors, with `d_out` at the first component of this one. The
// force kernels read dE/dD the same way.
template <typename P, typename E>
__device__ void descriptor_impl(const E* env, const P* g, const int* slot_type,
        const int* slot_local, const int* sel, const int* offset, int count, int slots,
        int width, int axis, E* b_out, P* d_out, const int* rows, int ntypes,
        const int* index, const int* atype, const P* g_empty, const E* empty_row, int first,
        int stride) {
    extern __shared__ double b_shared[];
    E* b = (E*)b_shared;                    // [width][4]
    E* near = b + 4 * width;                // without `rows`: this atom's rows, [slots][4]
    int* filled = (int*)(near + 4 * slots); // and its neighbors of each type, [ntypes]
    int i = blockIdx.x, model = blockIdx.y;
    // This model's rows of the embedding and of the environment, and its
    // place among the centers.
    g += (long long)model * count * slots * width;
    env += (long long)model * count * slots * 4;
    long long center = (long long)model * count + i;
    const E inv_slots = (E)(1.0 / slots);
    if (!rows) {
        const E* stored = env + (long long)i * slots * 4;
        for (int e = threadIdx.x; e < 4 * slots; e += blockDim.x) near[e] = stored[e];
        count_neighbors(index + (long long)(first + i) * slots, slot_type, slot_local, sel,
                        slots, ntypes, filled);
        __syncthreads();
    }
    for (int m = threadIdx.x; m < width; m += blockDim.x) {
        E acc[4] = {0, 0, 0, 0};
        if (rows) {
            for (int s = 0; s < slots; ++s) {
                E gm = (E)g[slot_row(rows, ntypes, count, i, slot_type[s], slot_local[s], sel, offset) * width + m];
                const E* r = env + ((long long)i * slots + s) * 4;
                for (int a = 0; a < 4; ++a) acc[a] += gm * r[a];
            }
        } else {
            int c = model * ntypes + atype[first + i];
            for (int t = 0; t < ntypes; ++t) {
                E part[4] = {0, 0, 0, 0};
                int neighbors = filled[t];
                const P* mine = g + embed_row(count, i, t, 0, sel, offset) * width + m;
                const E* r = near + 4 * offset[t];
                for (int k = 0; k < neighbors; ++k) {
                    E gm = (E)mine[(long long)k * width];
                    for (int a = 0; a < 4; ++a) part[a] += gm * r[4 * k + a];
                }
                if (neighbors < sel[t]) {
                    E gm = (E)(sel[t] - neighbors) * (E)g_empty[((long long)c * ntypes + t) * width + m];
                    for (int a = 0; a < 4; ++a) part[a] += gm * empty_row[4 * c + a];
                }
                for (int a = 0; a < 4; ++a) acc[a] += part[a];
            }
        }
        for (int a = 0; a < 4; ++a) {
            E v = acc[a] * inv_slots;
            b[4 * m + a] = v;
            b_out[(center * width + m) * 4 + a] = v;
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < width * axis; e += blockDim.x) {
        int m = e / axis, k = e % axis;
        E v = 0;
        for (int a = 0; a < 4; ++a) v += b[4 * m + a] * b[4 * k + a];
        d_out[center * stride + e] = (P)v;
    }
}

template <typename T>
__device__ T warp_sum(T v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

// The end of a force kernel. Each thread holds what its slots gave to the
// force on the center `i` (`fi`) and to the virial of the block (`vir`);
// they are summed over the block in `red` ([12], cleared before) and only
// then go to global memory: the force with an atomic, since other blocks add
// to it for their neighbors, the virial as the block's own. With `add` the
// virial is added to what is there: the kernels of the descriptors of a
// hybrid run one after the other, and all but the first add.
__device__ void block_sums(double* fi, double* vir, double* red, double* force, double* virial,
        int i, int add) {
    for (int c = 0; c < 3; ++c) fi[c] = warp_sum(fi[c]);
    for (int e = 0; e < 9; ++e) vir[e] = warp_sum(vir[e]);
    if ((threadIdx.x & 31) == 0) {
        for (int c = 0; c < 3; ++c) atomicAdd(&red[c], fi[c]);
        for (int e = 0; e < 9; ++e) atomicAdd(&red[3 + e], vir[e]);
    }
    __syncthreads();
    for (int c = threadIdx.x; c < 3; c += blockDim.x) atomicAdd(&force[3 * i + c], red[c]);
    for (int e = threadIdx.x; e < 9; e += blockDim.x) {
        double* out = virial + (long long)i * 9 + e;
        *out = add ? *out + red[3 + e] : red[3 + e];
    }
}

// One block per atom, three phases:
//  1. dB = dE/dB from the descriptor adjoint (in `E`), stored as P in a
//     [4][width] layout so the per-slot loop reads shared memory without
//     bank conflicts;
//  2. one warp per neighbor slot (lanes over the embedding width, so G rows
//     are read coalesced) contracts dB with G and dG/dr in the network
//     precision into dE/d(env row), kept in shared memory as float64;
//  3. one thread per slot applies the float64 environment Jacobian; only the
//     neighbor's force needs a global atomic, the center force and the
//     virial are reduced over the block first.
template <typename P, typename E>
__device__ void backward_impl(const E* env, const P* g, const P* dg, const E* b_in,
        const P* dd_in, const int* index, const double* delta, const double* dist,
        const int* atype, const double* dstd_inv, const int* slot_type, const int* slot_local,
        const int* sel, const int* offset, int first, int count, int slots, int width, int axis,
        double rmin, double inv_range, double* force, double* virial, const int* rows,
        int ntypes, int natoms, int stride, int add) {
    extern __shared__ double shared[];
    double* dr_all = shared;             // [slots][4]
    double* red = dr_all + 4 * slots;    // [12]: center force, virial
    E* b = (E*)(red + 12);               // [width][4]
    P* db = (P*)(b + 4 * width);         // [4][width]
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
    const P* dd = dd_in + center * stride;
    for (int m = threadIdx.x; m < width; m += blockDim.x) {
        E acc[4] = {0, 0, 0, 0};
        for (int k = 0; k < axis; ++k) {
            E w = (E)dd[m * axis + k];
            for (int a = 0; a < 4; ++a) acc[a] += w * b[4 * k + a];
        }
        if (m < axis) {
            E more[4] = {0, 0, 0, 0};
            for (int n = 0; n < width; ++n) {
                E w = (E)dd[n * axis + m];
                for (int a = 0; a < 4; ++a) more[a] += w * b[4 * n + a];
            }
            for (int a = 0; a < 4; ++a) acc[a] += more[a];
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
        const E* r = env + 4 * ((long long)li * slots + s);
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
    block_sums(fi, vir, red, force, virial, i, add);
}

// se_e2_r: the descriptor is a mean of embedding rows, D[m] = scale · sum
// over the slots of G[slot][m] with scale = 1 / (5 slots): DeePMD-kit
// averages the rows of each neighbor type, weights the types by their slots
// and divides by five. One block per atom. The neighbors are summed slot by
// slot, the sum of each type apart, and the empty slots of a type enter as
// their number times the row they share, as in the se_e2_a kernel. The
// environment rows are not needed: the embedding has taken their radial
// entry already. With `rows` (the trainer) every slot has its row in that
// table, the empty ones of a pair of types the one they share, and all are
// summed alike. The sums are formed in `E`, as those of se_e2_a.
template <typename P, typename E>
__device__ void radial_descriptor_impl(const P* g, const int* slot_type, const int* slot_local,
        const int* sel, const int* offset, int count, int slots, int width, double scale,
        P* d_out, int ntypes, const int* index, const int* atype, const P* g_empty, int first,
        const int* rows, int stride) {
    extern __shared__ int radial_filled[];  // this atom's neighbors of each type, [ntypes]
    int i = blockIdx.x, model = blockIdx.y;
    g += (long long)model * count * slots * width;
    long long center = (long long)model * count + i;
    if (!rows) {
        count_neighbors(index + (long long)(first + i) * slots, slot_type, slot_local, sel,
                        slots, ntypes, radial_filled);
        __syncthreads();
    }
    for (int m = threadIdx.x; m < width; m += blockDim.x) {
        E acc = 0;
        if (rows) {
            for (int s = 0; s < slots; ++s)
                acc += (E)g[slot_row(rows, ntypes, count, i, slot_type[s], slot_local[s], sel, offset) * width + m];
        } else {
            int c = model * ntypes + atype[first + i];
            for (int t = 0; t < ntypes; ++t) {
                E part = 0;
                int neighbors = radial_filled[t];
                const P* mine = g + embed_row(count, i, t, 0, sel, offset) * width + m;
                for (int k = 0; k < neighbors; ++k) part += (E)mine[(long long)k * width];
                if (neighbors < sel[t])
                    part += (E)(sel[t] - neighbors) * (E)g_empty[((long long)c * ntypes + t) * width + m];
                acc += part;
            }
        }
        d_out[center * stride + m] = (P)(acc * (E)scale);
    }
}

// The forces of an se_e2_r model, one block per atom, two phases:
//  1. one warp per neighbor slot (lanes over the embedding width) contracts
//     dE/dD with the slot's dG/ds in the network precision into
//     dE/d(radial entry), kept in shared memory as float64;
//  2. one thread per slot applies the float64 Jacobian of that entry, as the
//     last phase of the se_e2_a kernel does for a row that has nothing but
//     its radial entry: the entry sw = q / r changes along the neighbor
//     vector only.
template <typename P>
__device__ void radial_backward_impl(const P* dg, const P* dd_in, const int* index,
        const double* delta, const double* dist, const int* atype, const double* dstd_inv,
        const int* slot_type, const int* slot_local, const int* sel, const int* offset,
        int first, int count, int slots, int width, double scale, double rmin, double inv_range,
        double* force, double* virial, int ntypes, int natoms, const int* rows, int stride,
        int add) {
    extern __shared__ double radial_shared[];
    double* ds_all = radial_shared;      // [slots]: dE/d(radial entry)
    double* red = ds_all + slots;        // [12]: center force, virial
    P* dd = (P*)(red + 12);              // [width]: dE/dD of this atom
    int li = blockIdx.x, i = first + li, model = blockIdx.y;
    dg += (long long)model * count * slots * width;
    long long center = (long long)model * count + li;
    force += 3 * (long long)model * natoms;
    virial += 9 * (long long)model * natoms;
    for (int m = threadIdx.x; m < width; m += blockDim.x) dd[m] = dd_in[center * stride + m];
    for (int e = threadIdx.x; e < 12; e += blockDim.x) red[e] = 0.0;
    __syncthreads();
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (int s = warp; s < slots; s += warps) {
        long long slot = (long long)i * slots + s;
        if (index[slot] < 0) continue;
        long long row = slot_row(rows, ntypes, count, li, slot_type[s], slot_local[s], sel, offset) * width;
        P sum = 0;
        for (int m = lane; m < width; m += 32) sum += dd[m] * dg[row + m];
        sum = warp_sum(sum);
        if (lane == 0) ds_all[s] = (double)sum * scale;
    }
    __syncthreads();
    const double inv_std = dstd_inv[4 * (model * ntypes + atype[i])];
    double fi[3] = {0.0, 0.0, 0.0}, vir[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int s = threadIdx.x; s < slots; s += blockDim.x) {
        long long slot = (long long)i * slots + s;
        int j = index[slot];
        if (j < 0) continue;
        const double* d = delta + 3 * slot;
        double rr = dist[slot], inv = 1.0 / rr, q, dq;
        switch_fn(rr, rmin, inv_range, &q, &dq);
        double sw = q * inv, dsw = (dq - sw) * inv;
        double along = ds_all[s] * inv_std * dsw;
        for (int c = 0; c < 3; ++c) {
            double grad = d[c] * inv * along;
            atomicAdd(&force[3 * (long long)j + c], -grad);
            fi[c] += grad;
            for (int a = 0; a < 3; ++a) vir[3 * a + c] -= d[a] * grad;
        }
    }
    block_sums(fi, vir, red, force, virial, i, add);
}

// se_e3, DeePMD-kit's three-body descriptor. With a = sw·d / (r·std) the
// direction entries of the normalized environment row of a neighbor, every
// pair of an atom's neighbors gives θ = a_j·a_k to the embedding net of its
// pair of neighbor types, and D[m] is the sum over the pairs of
// weight·θ·G[m](θ). DeePMD-kit sums the pairs of types t <= u over all slots
// of both and divides by sel[t]·sel[u]: an empty slot has no direction and
// adds nothing, and within one type two neighbors come in both orders. So
// every pair of neighbors is one row of the embedding nets here, with the
// weight 2 / sel[t]² for two neighbors of one type and 1 / (sel[t]·sel[u])
// otherwise (a neighbor with itself, or neighbors of two types).
//
// The neighbors of an atom are numbered in the order of their slots, 0 to
// n - 1, so that their types do not decrease, and the pair (j <= k) is row
// j·n - j(j - 1)/2 + (k - j) of the atom's n(n + 1)/2 rows. `row_start` has
// the first row of every atom of the frame as if all its atoms were laid
// out one after the other; a chunk has its rows from that of its first atom
// on, model after model (`chunk_rows` each).

// Neighbors held by every atom, one thread per atom: the host lays out the
// rows from them.
extern "C" __global__ void se_e3_count(const int* index, int natoms, int slots, int* held) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;
    const int* mine = index + i * slots;
    int n = 0;
    for (int s = 0; s < slots; ++s) n += mine[s] >= 0;
    held[i] = n;
}

// The neighbors of the block's atom in shared memory, there for all threads
// when this returns: `list[k]` the slot of neighbor k, `a[3k..3k + 3]` its
// direction entries in double precision, `first_of[t]` the number of the
// first neighbor of type t and `first_of[ntypes]` the number of neighbors.
// `mine`, `delta` and `dist` are the atom's own slots.
__device__ void three_body_neighbors(const int* mine, const double* delta, const double* dist,
        double inv_std, const int* slot_type, const int* slot_local, const int* sel, int slots,
        int ntypes, double rmin, double inv_range, double* a, int* list, int* first_of) {
    count_neighbors(mine, slot_type, slot_local, sel, slots, ntypes, first_of);
    __syncthreads();
    if (threadIdx.x == 0) {
        int before = 0;
        for (int t = 0; t < ntypes; ++t) {
            int n = first_of[t];
            first_of[t] = before;
            before += n;
        }
        first_of[ntypes] = before;
    }
    __syncthreads();
    for (int s = threadIdx.x; s < slots; s += blockDim.x) {
        if (mine[s] < 0) continue;
        const double* d = delta + 3 * s;
        double r = dist[s], inv = 1.0 / r, q, dq;
        switch_fn(r, rmin, inv_range, &q, &dq);
        double scale = q * inv * inv * inv_std;
        int k = first_of[slot_type[s]] + slot_local[s];
        list[k] = s;
        for (int c = 0; c < 3; ++c) a[3 * k + c] = d[c] * scale;
    }
    __syncthreads();
}

// One block per atom writes the rows of its pairs: the input θ of the
// embedding net (worked out in double precision), the net of the pair of
// types (of this model) and the weight of the pair in the descriptor.
// Thread j has the pairs (j, k >= j).
template <typename P>
__device__ void three_body_pairs_impl(const int* index, const double* delta, const double* dist,
        const int* atype, const double* dstd_inv, const int* slot_type, const int* slot_local,
        const int* sel, const double* sel_inv, int first, int slots, int ntypes, double rmin,
        double inv_range, const long long* row_start, long long chunk_rows, P* theta, P* weight,
        int* network) {
    extern __shared__ double pairs_shared[];
    double* a = pairs_shared;            // [slots][3]
    int* list = (int*)(a + 3 * slots);   // [slots]
    int* first_of = list + slots;        // [ntypes + 1]
    int i = first + blockIdx.x, model = blockIdx.y;
    long long at = (long long)i * slots;
    int c = model * ntypes + atype[i];
    three_body_neighbors(index + at, delta + 3 * at, dist + at, dstd_inv[4 * c + 1], slot_type,
                         slot_local, sel, slots, ntypes, rmin, inv_range, a, list, first_of);
    int n = first_of[ntypes];
    long long base = (long long)model * chunk_rows + (row_start[i] - row_start[first]);
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        int tj = slot_type[list[j]];
        long long row = base + (long long)j * n - (long long)j * (j - 1) / 2;
        for (int k = j; k < n; ++k, ++row) {
            int tk = slot_type[list[k]];
            double th = a[3 * j] * a[3 * k] + a[3 * j + 1] * a[3 * k + 1]
                      + a[3 * j + 2] * a[3 * k + 2];
            double orders = (k != j && tk == tj) ? 2.0 : 1.0;
            theta[row] = (P)th;
            weight[row] = (P)(orders * sel_inv[tj] * sel_inv[tk]);
            network[row] = (model * ntypes + tj) * ntypes + tk;
        }
    }
}

// One block per atom: D[m] = sum over the atom's rows of weight·θ·G[row][m].
// The rows are streams of device memory, and a thread waits for each of its
// loads. So the warps of the block share the rows (warp w has the rows w,
// w + warps, ...) and the lanes of a warp the outputs, C of them a lane at a
// time (m, m + 32, ...) in registers: a row is read once, by one warp. A
// warp adds its rows thirty-two at a time, so that a sum in single precision
// stays short, and leaves its sums in shared memory, where the first threads
// add those of all warps.
template <typename P, int C>
__device__ void three_body_descriptor_impl(const P* g, const P* theta, const P* weight,
        const long long* row_start, long long chunk_rows, int first, int count, int width,
        P* d_out, int stride) {
    extern __shared__ double descriptor_shared[];
    P* sums = (P*)descriptor_shared;     // [warps][width]
    int li = blockIdx.x, model = blockIdx.y;
    long long base = (long long)model * chunk_rows + (row_start[first + li] - row_start[first]);
    long long rows = row_start[first + li + 1] - row_start[first + li];
    long long center = (long long)model * count + li;
    theta += base;
    weight += base;
    g += base * width;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (int m0 = lane; m0 < width; m0 += 32 * C) {
        P acc[C], part[C];
        for (int c = 0; c < C; ++c) { acc[c] = 0; part[c] = 0; }
        int held = 0;
        for (long long p = warp; p < rows; p += warps) {
            P scaled = weight[p] * theta[p];
            const P* row = g + p * width + m0;
            for (int c = 0; c < C; ++c)
                if (m0 + 32 * c < width) part[c] += scaled * row[32 * c];
            if (++held == 32) {
                for (int c = 0; c < C; ++c) { acc[c] += part[c]; part[c] = 0; }
                held = 0;
            }
        }
        for (int c = 0; c < C; ++c)
            if (m0 + 32 * c < width) sums[warp * width + m0 + 32 * c] = acc[c] + part[c];
    }
    __syncthreads();
    for (int m = threadIdx.x; m < width; m += blockDim.x) {
        P total = 0;
        for (int w = 0; w < warps; ++w) total += sums[w * width + m];
        d_out[center * stride + m] = total;
    }
}

// The forces of an se_e3 model, one block per atom, two phases:
//  1. one warp per row (lanes over the embedding width, so that the rows of
//     G and dG/dθ are read coalesced) contracts dE/dD with G + θ·dG/dθ in
//     the network precision and leaves dE/dθ of the pair, weight included,
//     at the row's place in `dtheta`. The warps of the block share the rows,
//     as in the descriptor kernel. (A thread per row, each summing over the
//     width by itself, was no faster for a few hundred atoms and up to
//     twice slower for thousands: its loads are scattered.)
//  2. one thread per neighbor j adds dE/dθ·a_k over its partners k, in
//     float64: θ of a pair moves by a_k with a_j, and that of j with itself
//     by twice a_j. The row of the pair (j, k) is that of the smaller number
//     first. The same thread then applies the float64 Jacobian of the
//     direction entries of j, as the last phase of the se_e2_a kernel does
//     for a row without its radial entry.
template <typename P>
__device__ void three_body_backward_impl(const P* g, const P* dg, const P* theta,
        const P* weight, P* dtheta, const P* dd_in, const int* index, const double* delta,
        const double* dist, const int* atype, const double* dstd_inv, const int* slot_type,
        const int* slot_local, const int* sel, const long long* row_start, long long chunk_rows,
        int first, int count, int slots, int width, int ntypes, double rmin, double inv_range,
        double* force, double* virial, int natoms, int stride, int add) {
    extern __shared__ double back_shared[];
    double* a = back_shared;             // [slots][3]: directions of the neighbors
    double* red = a + 3 * slots;         // [12]: center force, virial
    P* dd = (P*)(red + 12);              // [width]: dE/dD of this atom, on 8 bytes' bounds
    int* list = (int*)(red + 12 + (width * sizeof(P) + 7) / 8);  // [slots]
    int* first_of = list + slots;        // [ntypes + 1]
    int li = blockIdx.x, i = first + li, model = blockIdx.y;
    long long at = (long long)i * slots;
    long long center = (long long)model * count + li;
    const double inv_std = dstd_inv[4 * (model * ntypes + atype[i]) + 1];
    force += 3 * (long long)model * natoms;
    virial += 9 * (long long)model * natoms;
    for (int m = threadIdx.x; m < width; m += blockDim.x) dd[m] = dd_in[center * stride + m];
    for (int e = threadIdx.x; e < 12; e += blockDim.x) red[e] = 0.0;
    three_body_neighbors(index + at, delta + 3 * at, dist + at, inv_std, slot_type, slot_local,
                         sel, slots, ntypes, rmin, inv_range, a, list, first_of);
    int n = first_of[ntypes];
    long long base = (long long)model * chunk_rows + (row_start[i] - row_start[first]);
    long long rows = row_start[i + 1] - row_start[i];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (long long p = warp; p < rows; p += warps) {
        long long row = base + p;
        const P* gr = g + row * width;
        const P* dgr = dg + row * width;
        P th = theta[row], sum = 0;
        for (int m = lane; m < width; m += 32) sum += dd[m] * (gr[m] + th * dgr[m]);
        sum = warp_sum(sum);
        if (lane == 0) dtheta[row] = weight[row] * sum;
    }
    __syncthreads();
    double fi[3] = {0.0, 0.0, 0.0}, vir[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        // dE/da_j. The pairs (j, k >= j) follow each other from `own` on.
        long long own = base + (long long)j * n - (long long)j * (j - 1) / 2;
        double da[3] = {0.0, 0.0, 0.0};
        for (int k = 0; k < n; ++k) {
            long long row = k >= j ? own + (k - j)
                                   : base + (long long)k * n - (long long)k * (k - 1) / 2 + (j - k);
            double w = (double)dtheta[row];
            if (k == j) w *= 2.0;
            for (int c = 0; c < 3; ++c) da[c] += w * a[3 * k + c];
        }
        long long slot = at + list[j];
        int neighbor = index[slot];
        const double* d = delta + 3 * slot;
        double rr = dist[slot], inv = 1.0 / rr, q, dq;
        switch_fn(rr, rmin, inv_range, &q, &dq);
        double u[3] = {d[0] * inv, d[1] * inv, d[2] * inv};
        // The direction entries are sw u with sw = q / r: along c they change
        // by dsw u_a u_c + (sw / r)(delta_ac - u_a u_c). Contracted with the
        // adjoint e over the normalization: u_c (dsw - sw / r)(e · u)
        // + (sw / r) e_c.
        double sw = q * inv, dsw = (dq - sw) * inv, across = sw * inv;
        double adj[3] = {da[0] * inv_std, da[1] * inv_std, da[2] * inv_std};
        double along = (dsw - across) * (adj[0] * u[0] + adj[1] * u[1] + adj[2] * u[2]);
        for (int c = 0; c < 3; ++c) {
            double grad = u[c] * along + across * adj[c];
            atomicAdd(&force[3 * (long long)neighbor + c], -grad);
            fi[c] += grad;
            for (int e = 0; e < 3; ++e) vir[3 * e + c] -= d[e] * grad;
        }
    }
    block_sums(fi, vir, red, force, virial, i, add);
}

// The descriptors of a hybrid read one neighbor list, made with the largest
// of their cutoffs and for every type the most slots any of them has. A
// descriptor whose own slots or cutoff are smaller has a table of the list
// in its own layout: of every type the first `sel[t]` slots of the list (the
// neighbors of a type are nearest first there), and no neighbor at or
// beyond its own cutoff, which is an empty slot to it (DeePMD-kit's switch
// is zero from the cutoff on). So the kernels of the descriptor read the
// table as those of a model read their list, and the neighbors of a type
// still fill its first slots. One thread per slot of the table. The thread
// of an atom's first slot also looks whether the descriptor leaves out a
// neighbor within its cutoff, for lack of slots: the first slot of the list
// past its own of a type tells, and the atom's status gets the bit that the
// search sets for a list that had to drop neighbors.
extern "C" __global__ void hybrid_slots(const int* index, const double* delta,
        const double* dist, const int* slot_type, const int* slot_local, const int* sel,
        const int* list_sel, const int* list_offset, int natoms, int slots, int list_slots,
        int ntypes, double rcut, int* index_out, double* delta_out, double* dist_out,
        int* status) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)natoms * slots) return;
    long long i = idx / slots;
    int s = (int)(idx % slots);
    long long from = i * list_slots + list_offset[slot_type[s]] + slot_local[s];
    bool here = index[from] >= 0 && dist[from] < rcut;
    index_out[idx] = here ? index[from] : -1;
    dist_out[idx] = here ? dist[from] : __longlong_as_double(0x7ff0000000000000LL);
    for (int a = 0; a < 3; ++a) delta_out[3 * idx + a] = here ? delta[3 * from + a] : 0.0;
    if (s != 0) return;
    for (int t = 0; t < ntypes; ++t) {
        if (sel[t] >= list_sel[t]) continue;
        long long next = i * list_slots + list_offset[t] + sel[t];
        if (index[next] >= 0 && dist[next] < rcut) {
            status[i] |= 2;  // the search's TRUNCATED
            return;
        }
    }
}

// The `_rows` kernels take the row table of `slot_row` (and the number of
// types it is laid out by), are launched one model deep and keep the
// environment and B in double precision (the trainer's); the others skip
// empty slots, take the number of atoms a model's forces span and keep
// them in the precision of the networks, and take the stride of the
// descriptor in the buffers of all descriptors of the model and whether
// their virial is added to the one that is there.
#define INSTANTIATE(P, S)                                                                         \
extern "C" __global__ void se_e2_a_env_##S(const int* index, const double* delta,                 \
        const double* dist, const int* atype, const double* davg, const double* dstd_inv,         \
        const int* slot_type, const int* slot_local, const int* sel, const int* offset,           \
        int first, int count, int slots, int ntypes, double rmin, double inv_range, P* env,       \
        P* radial, int* network) {                                                                \
    env_impl<P, P>(index, delta, dist, atype, davg, dstd_inv, slot_type, slot_local, sel,         \
                   offset, first, count, slots, ntypes, rmin, inv_range, env, radial, network,    \
                   nullptr, 0);                                                                   \
}                                                                                                 \
extern "C" __global__ void se_e2_a_env_rows_##S(const int* index, const double* delta,            \
        const double* dist, const int* atype, const double* davg, const double* dstd_inv,         \
        const int* slot_type, const int* slot_local, const int* sel, const int* offset,           \
        int first, int count, int slots, int ntypes, double rmin, double inv_range, double* env,  \
        P* radial, int* network, const int* rows, int one_side) {                                 \
    env_impl<P, double>(index, delta, dist, atype, davg, dstd_inv, slot_type, slot_local, sel,    \
                        offset, first, count, slots, ntypes, rmin, inv_range, env, radial,        \
                        network, rows, one_side);                                                 \
}                                                                                                 \
extern "C" __global__ void se_e2_a_descriptor_##S(const P* env, const P* g,                       \
        const int* slot_type, const int* slot_local, const int* sel, const int* offset,           \
        int count, int slots, int width, int axis, P* b_out, P* d_out, int ntypes,                \
        const int* index, const int* atype, const P* g_empty, const P* empty_row, int first,      \
        int stride) {                                                                             \
    descriptor_impl<P, P>(env, g, slot_type, slot_local, sel, offset, count, slots, width, axis,  \
                          b_out, d_out, nullptr, ntypes, index, atype, g_empty, empty_row,        \
                          first, stride);                                                         \
}                                                                                                 \
extern "C" __global__ void se_e2_a_descriptor_rows_##S(const double* env, const P* g,             \
        const int* slot_type, const int* slot_local, const int* sel, const int* offset,           \
        int count, int slots, int width, int axis, double* b_out, P* d_out, const int* rows,      \
        int ntypes) {                                                                             \
    descriptor_impl<P, double>(env, g, slot_type, slot_local, sel, offset, count, slots, width,   \
                               axis, b_out, d_out, rows, ntypes, nullptr, nullptr, nullptr,       \
                               nullptr, 0, width * axis);                                         \
}                                                                                                 \
extern "C" __global__ void se_e2_a_backward_##S(const P* env, const P* g, const P* dg,            \
        const P* b_in, const P* dd_in, const int* index, const double* delta,                     \
        const double* dist, const int* atype, const double* dstd_inv, const int* slot_type,       \
        const int* slot_local, const int* sel, const int* offset, int first, int count,           \
        int slots, int width, int axis, double rmin, double inv_range, double* force,             \
        double* virial, int ntypes, int natoms, int stride, int add) {                            \
    backward_impl<P, P>(env, g, dg, b_in, dd_in, index, delta, dist, atype, dstd_inv, slot_type,  \
                        slot_local, sel, offset, first, count, slots, width, axis, rmin,          \
                        inv_range, force, virial, nullptr, ntypes, natoms, stride, add);          \
}                                                                                                 \
extern "C" __global__ void se_e2_a_backward_rows_##S(const double* env, const P* g,               \
        const P* dg, const double* b_in, const P* dd_in, const int* index, const double* delta,   \
        const double* dist, const int* atype, const double* dstd_inv, const int* slot_type,       \
        const int* slot_local, const int* sel, const int* offset, int first, int count,           \
        int slots, int width, int axis, double rmin, double inv_range, double* force,             \
        double* virial, const int* rows, int ntypes) {                                            \
    backward_impl<P, double>(env, g, dg, b_in, dd_in, index, delta, dist, atype, dstd_inv,        \
                             slot_type, slot_local, sel, offset, first, count, slots, width,      \
                             axis, rmin, inv_range, force, virial, rows, ntypes, 0,               \
                             width * axis, 0);                                                    \
}                                                                                                 \
extern "C" __global__ void se_e2_r_descriptor_##S(const P* g, const int* slot_type,               \
        const int* slot_local, const int* sel, const int* offset, int count, int slots,           \
        int width, double scale, P* d_out, int ntypes, const int* index, const int* atype,        \
        const P* g_empty, int first, int stride) {                                                \
    radial_descriptor_impl<P, P>(g, slot_type, slot_local, sel, offset, count, slots, width,      \
                                 scale, d_out, ntypes, index, atype, g_empty, first, nullptr,     \
                                 stride);                                                         \
}                                                                                                 \
extern "C" __global__ void se_e2_r_descriptor_rows_##S(const P* g, const int* slot_type,          \
        const int* slot_local, const int* sel, const int* offset, int count, int slots,           \
        int width, double scale, P* d_out, const int* rows, int ntypes) {                         \
    radial_descriptor_impl<P, double>(g, slot_type, slot_local, sel, offset, count, slots,        \
                                      width, scale, d_out, ntypes, nullptr, nullptr, nullptr, 0,  \
                                      rows, width);                                               \
}                                                                                                 \
extern "C" __global__ void se_e2_r_backward_##S(const P* dg, const P* dd_in, const int* index,    \
        const double* delta, const double* dist, const int* atype, const double* dstd_inv,        \
        const int* slot_type, const int* slot_local, const int* sel, const int* offset,           \
        int first, int count, int slots, int width, double scale, double rmin,                    \
        double inv_range, double* force, double* virial, int ntypes, int natoms, int stride,      \
        int add) {                                                                                \
    radial_backward_impl<P>(dg, dd_in, index, delta, dist, atype, dstd_inv, slot_type,            \
                            slot_local, sel, offset, first, count, slots, width, scale, rmin,     \
                            inv_range, force, virial, ntypes, natoms, nullptr, stride, add);      \
}                                                                                                 \
extern "C" __global__ void se_e2_r_backward_rows_##S(const P* dg, const P* dd_in,                 \
        const int* index, const double* delta, const double* dist, const int* atype,              \
        const double* dstd_inv, const int* slot_type, const int* slot_local, const int* sel,      \
        const int* offset, int first, int count, int slots, int width, double scale,              \
        double rmin, double inv_range, double* force, double* virial, const int* rows,            \
        int ntypes) {                                                                             \
    radial_backward_impl<P>(dg, dd_in, index, delta, dist, atype, dstd_inv, slot_type,            \
                            slot_local, sel, offset, first, count, slots, width, scale, rmin,     \
                            inv_range, force, virial, ntypes, 0, rows, width, 0);                 \
}                                                                                                 \
extern "C" __global__ void se_e3_pairs_##S(const int* index, const double* delta,                 \
        const double* dist, const int* atype, const double* dstd_inv, const int* slot_type,       \
        const int* slot_local, const int* sel, const double* sel_inv, int first, int slots,       \
        int ntypes, double rmin, double inv_range, const long long* row_start,                    \
        long long chunk_rows, P* theta, P* weight, int* network) {                                \
    three_body_pairs_impl<P>(index, delta, dist, atype, dstd_inv, slot_type, slot_local, sel,     \
                             sel_inv, first, slots, ntypes, rmin, inv_range, row_start,           \
                             chunk_rows, theta, weight, network);                                 \
}                                                                                                 \
extern "C" __global__ void se_e3_descriptor_##S(const P* g, const P* theta, const P* weight,      \
        const long long* row_start, long long chunk_rows, int first, int count, int width,        \
        P* d_out, int stride) {                                                                   \
    three_body_descriptor_impl<P, 4>(g, theta, weight, row_start, chunk_rows, first, count,       \
                                     width, d_out, stride);                                       \
}                                                                                                 \
extern "C" __global__ void se_e3_backward_##S(const P* g, const P* dg, const P* theta,            \
        const P* weight, P* dtheta, const P* dd_in, const int* index, const double* delta,        \
        const double* dist, const int* atype, const double* dstd_inv, const int* slot_type,       \
        const int* slot_local, const int* sel, const long long* row_start,                        \
        long long chunk_rows, int first, int count, int slots, int width, int ntypes,             \
        double rmin, double inv_range, double* force, double* virial, int natoms, int stride,     \
        int add) {                                                                                \
    three_body_backward_impl<P>(g, dg, theta, weight, dtheta, dd_in, index, delta, dist, atype,   \
                                dstd_inv, slot_type, slot_local, sel, row_start, chunk_rows,      \
                                first, count, slots, width, ntypes, rmin, inv_range, force,       \
                                virial, natoms, stride, add);                                     \
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
/// Float32 chunks of at most this many atoms per center type and model run
/// the fitting nets in the fused kernels instead of layer-by-layer cuBLAS,
/// whose fixed cost (some twenty-five launches per type, 0.68 ms for five
/// types) dominates below that. Float64 always uses the fused kernel: both
/// paths are bound by the double-precision units there and the fused one has
/// no fixed cost. Measured on an RTX 4090 with [240, 240, 240] fitting nets
/// on 1600 descriptor components, whole passes over batches of a frame in
/// float32: with five types the two paths meet near 1,800 to 2,000 atoms
/// (88-atom frames: 1.28 against 1.37 ms at 1,408 atoms, 1.96 against 1.72
/// ms at 2,112; 169-atom frames: equal at 2,028), with two types near 680
/// (192-atom frames: 0.60 against 0.63 ms at 576 atoms, 0.74 against 0.71 ms
/// at 768).
const FUSED_FITTING_ATOMS_PER_TYPE_F32: usize = 340;
/// Fraction of the free device memory one chunk of per-atom work may use.
const MEMORY_FRACTION: f64 = 0.5;

struct Kernels {
    env: CudaFunction,
    descriptor: CudaFunction,
    backward: CudaFunction,
    /// se_e3: the neighbors every atom holds, and the rows of their pairs.
    count: Option<CudaFunction>,
    pairs: Option<CudaFunction>,
}

/// The rows of the embedding nets of an se_e3 descriptor for a frame: the
/// pairs of the neighbors of each atom, n(n + 1)/2 for an atom with n
/// neighbors.
struct PairRows {
    /// The first row of every atom, as if all atoms were laid out one after
    /// the other, and the number of all rows after the last.
    start: CudaSlice<i64>,
    host: Vec<i64>,
}

impl PairRows {
    /// Rows of the atoms `first..end`.
    fn rows(&self, first: usize, end: usize) -> usize {
        (self.host[end] - self.host[first]) as usize
    }
}

/// The first row of every atom that holds `held[atom]` neighbors, and after
/// the last the number of all rows.
fn pair_starts(held: &[i32]) -> Vec<i64> {
    let mut start = Vec::with_capacity(held.len() + 1);
    let mut next = 0i64;
    for &n in held {
        start.push(next);
        next += i64::from(n) * (i64::from(n) + 1) / 2;
    }
    start.push(next);
    start
}

/// `atoms` atoms in chunks of at most `limit` atoms that have at most `rows`
/// rows in each of the layouts `starts` (one of [`pair_starts`] for every
/// se_e3 descriptor of the model; none for a model without). An atom is
/// never split: one with more rows than that is a chunk of its own.
fn pair_chunks(starts: &[&[i64]], atoms: usize, limit: usize, rows: usize) -> Vec<(usize, usize)> {
    let fits = |first: usize, end: usize| {
        starts
            .iter()
            .all(|start| (start[end + 1] - start[first]) as u64 <= rows as u64)
    };
    let mut chunks = Vec::new();
    let mut first = 0;
    while first < atoms {
        let mut end = first + 1;
        while end < atoms && end - first < limit.max(1) && fits(first, end) {
            end += 1;
        }
        chunks.push((first, end));
        first = end;
    }
    chunks
}

/// Device bytes of one row of an se_e3 chunk: the embedding values and
/// tangents, θ of the pair, its weight and dE/dθ, and its net.
fn bytes_per_pair<P>(width: usize) -> usize {
    let p = std::mem::size_of::<P>();
    2 * width * p + 3 * p + 4
}

/// Threads per block of the se_e3 force kernel: thirty-two warps, each with
/// a row of the atom at a time. A warp waits for the rows it reads, so a
/// few hundred atoms go faster the more warps share an atom's rows (192
/// atoms in float32: 0.43 ms with thirty-two warps, 0.67 with eight);
/// thousands of atoms are bound by reading the rows either way.
const PAIR_BLOCK: u32 = 1024;

/// Warps per block of the se_e3 descriptor kernel, which share the rows of
/// an atom: as many as have their sums (`width` each) in shared memory, at
/// most eight.
fn descriptor_warps<P>(width: usize) -> usize {
    (SHARED_LIMIT / (width * std::mem::size_of::<P>()).max(1)).clamp(1, 8)
}

/// Shared memory of the se_e3 pair kernel per block: the directions of the
/// atom's neighbors in double, their slots and the first of each type.
fn pair_shared(slots: usize, ntypes: usize) -> usize {
    3 * slots * 8 + (4 * (slots + ntypes + 1)).next_multiple_of(8)
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
    /// of a descriptor that go by center type hold model `m` at center type
    /// `m * ntypes + type`.
    models: usize,
    /// The slots of the neighbor list by type: those of the model's
    /// descriptor, which for a hybrid are the most of any of its
    /// descriptors.
    sel: Vec<usize>,
    /// The descriptors of the models on the device: one, or those of a
    /// hybrid in the order of their components.
    parts: Vec<Part<P>>,
    /// Components of the descriptor: the input of a fitting net.
    features: usize,
    /// Energy bias per center type.
    bias: Vec<f64>,
    table_fallbacks: usize,
    /// Layer-by-layer fitting nets per center type (cuBLAS): best for many
    /// atoms.
    fitting: Vec<DenseNetwork<P>>,
    /// All fitting nets in one kernel, forward and backward: best for few
    /// atoms, where the launches of the layer-by-layer path dominate.
    fused_fitting: Option<GradientNetwork<P>>,
    /// Chunks of at most this many atoms use `fused_fitting`.
    fused_fitting_max_atoms: usize,
    /// hybrid: writes the table of slots of a descriptor that does not read
    /// the neighbor list as it is.
    narrow: Option<CudaFunction>,
    chunk_atoms: usize,
    /// se_e3: rows of the embedding nets (pairs of neighbors) per chunk.
    chunk_pairs: usize,
    /// Neighbor search with its device buffers, kept between evaluations.
    search: NeighborSearch,
    last_truncated: bool,
    profile: Option<StageTimes>,
}

/// One descriptor of the models on the device: its embedding nets and what
/// its kernels read. A model has one; a hybrid has one for every descriptor
/// of its list, each with cutoffs, slots, statistics and nets of its own.
struct Part<P: Scalar> {
    /// Which kernels the embedding rows go to.
    kind: DescriptorKind,
    rcut: f64,
    rcut_smth: f64,
    /// 1 / (rcut - rcut_smth), for the switching function.
    inv_range: f64,
    slots: usize,
    width: usize,
    /// se_e2_a: the axis neurons. None for the others.
    axis: usize,
    /// Components of this descriptor, and the first of them among the
    /// components of all descriptors of the model.
    features: usize,
    first_feature: usize,
    /// Every embedding net in one set; net `center type · ntypes + neighbor
    /// type` (the ids the environment kernel writes).
    embedding: ScalarNetwork<P>,
    /// Host copies of the embedding nets and the radial (mean, std) per
    /// center type, for building tables.
    embedding_host: Vec<Vec<Network>>,
    radial_normalization: Vec<(f64, f64)>,
    /// Tabulated embedding nets in the same order, when compressed.
    tables: Option<ScalarTable<P>>,
    davg: CudaSlice<f64>,
    /// Reciprocals of the normalization widths: the kernels multiply.
    dstd_inv: CudaSlice<f64>,
    /// The environment row of an empty slot, the normalized zero, per center
    /// type: `[center types][4]`.
    empty_row: CudaSlice<P>,
    /// The embedding of the input every empty slot has (the normalized
    /// zero) for each pair of center type and neighbor type,
    /// `[ntypes²][width]`. Empty slots are not evaluated one by one: the
    /// descriptor adds this row times their number. It comes from the
    /// networks also when the neighbors go through tables.
    empty_g: CudaSlice<P>,
    slot_type: CudaSlice<i32>,
    slot_local: CudaSlice<i32>,
    sel_d: CudaSlice<i32>,
    /// 1 / sel per neighbor type (zero for a type without slots): what a
    /// pair of an se_e3 atom's neighbors counts is a product of two.
    sel_inv: CudaSlice<f64>,
    offset_d: CudaSlice<i32>,
    kernels: Kernels,
    /// Dynamic shared memory of the force kernel, bytes.
    backward_shared: usize,
    /// hybrid: this descriptor's own table of the neighbor list, when its
    /// slots or its cutoff are not those of the list.
    view: Option<View>,
}

/// How a descriptor of a hybrid reads the neighbor list of all of them: of
/// every type the first of the list's slots, as many as it has itself, and
/// no neighbor at or beyond its own cutoff (the `hybrid_slots` kernel).
struct View {
    /// The slots of the list by type, and the first slot of each type.
    list_sel: CudaSlice<i32>,
    list_offset: CudaSlice<i32>,
    list_slots: usize,
    /// The table of the last search, kept between evaluations.
    table: Option<SlotTable>,
}

/// The slots of a descriptor for every atom of a search, like those of a
/// neighbor list: `[atoms][slots]` neighbors (-1 for an empty slot), their
/// vectors and distances.
struct SlotTable {
    /// Atoms the buffers hold.
    atoms: usize,
    index: CudaSlice<i32>,
    delta: CudaSlice<f64>,
    dist: CudaSlice<f64>,
}

/// The work buffers of a descriptor for the chunks of an evaluation. Each
/// element a chunk reads was written earlier in the same chunk.
struct Work<P: Scalar> {
    /// se_e2_a: the environment rows and B. The others have neither.
    env: Option<CudaSlice<P>>,
    b: Option<CudaSlice<P>>,
    /// The input of the embedding nets (the radial entry of a slot, or θ of
    /// a pair of neighbors), the net of each row, and the values and
    /// tangents of the nets.
    input: CudaSlice<P>,
    network: CudaSlice<i32>,
    g: CudaSlice<P>,
    dg: CudaSlice<P>,
    /// se_e3: what each pair counts in the descriptor and dE/dθ of it, and
    /// where the rows of each atom are.
    of_pairs: Option<(CudaSlice<P>, CudaSlice<P>)>,
    pairs: Option<PairRows>,
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

/// `f` of every item, in the order of the items, on as many threads as the
/// machine has cores (no more than items).
fn in_parallel<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism()
        .map_or(1, |cores| cores.get())
        .min(items.len());
    if threads <= 1 {
        return items.iter().map(f).collect();
    }
    let share = items.len().div_ceil(threads);
    let f = &f;
    std::thread::scope(|scope| {
        let workers: Vec<_> = items
            .chunks(share)
            .map(|part| scope.spawn(move || part.iter().map(f).collect::<Vec<R>>()))
            .collect();
        workers
            .into_iter()
            .flat_map(|worker| match worker.join() {
                Ok(results) => results,
                // A panic of `f`: pass it on as it was.
                Err(panic) => std::panic::resume_unwind(panic),
            })
            .collect()
    })
}

/// Whether two descriptors (not hybrids) are laid out alike on the device:
/// the kind, the cutoffs, the slots, the axis neurons and the width of the
/// embedding.
fn same_layout(a: &Descriptor, b: &Descriptor) -> bool {
    let width = |d: &Descriptor| {
        d.embedding
            .first()
            .and_then(|row| row.first())
            .map(Network::output_size)
    };
    a.kind == b.kind
        && a.rcut == b.rcut
        && a.rcut_smth == b.rcut_smth
        && a.sel == b.sel
        && a.axis_neuron == b.axis_neuron
        && width(a) == width(b)
}

impl<P: Scalar> Part<P> {
    /// The descriptor `descriptors[m]` of model `m` on the device (all laid
    /// out like the first). `list_sel` and `list_rcut` are the slots and the
    /// cutoff of the neighbor list it reads, `first_feature` is the place of
    /// its components among those of the model, and `same_shape` words the
    /// error of nets that cannot be one set.
    fn new(
        device: &Device,
        descriptors: &[&Descriptor],
        ntypes: usize,
        list_sel: &[usize],
        list_rcut: f64,
        first_feature: usize,
        same_shape: &dyn Fn(Error) -> Error,
    ) -> Result<Self> {
        let d = descriptors[0];
        let kind = d
            .descriptor_kind()
            .ok_or_else(|| Error(format!("no kernels for the descriptor {:?}", d.kind)))?;
        let width = d.width();
        let slots: usize = d.sel.iter().sum();
        // Of the force kernel: dE/d(env row) and the block's sums in double,
        // B and its adjoint in the precision of the networks; for se_e2_r
        // dE/d(radial entry) and the sums in double, dE/dD in that precision.
        let backward_shared = match kind {
            DescriptorKind::SeE2A => (4 * slots + 12) * 8 + 8 * width * std::mem::size_of::<P>(),
            DescriptorKind::SeE2R => {
                (slots + 12) * 8 + (width * std::mem::size_of::<P>()).next_multiple_of(8)
            }
            // se_e3: the directions of the neighbors and the block's sums in
            // double, dE/dD in the precision of the networks, the neighbors'
            // slots and the first of each type.
            DescriptorKind::SeE3 => {
                (3 * slots + 12) * 8
                    + (width * std::mem::size_of::<P>()).next_multiple_of(8)
                    + (4 * (slots + ntypes + 1)).next_multiple_of(8)
            }
        };
        if backward_shared > SHARED_LIMIT {
            return Err(Error(format!(
                "GPU {} needs {backward_shared} bytes of shared memory per atom (embedding width {width}, {slots} neighbor slots); the limit is {SHARED_LIMIT}",
                kind.name()
            )));
        }
        // Model `m` has its nets and statistics at center type `m * ntypes +
        // type`, as the kernels look them up.
        let embedding_host: Vec<Vec<Network>> = descriptors
            .iter()
            .flat_map(|d| d.embedding.iter().cloned())
            .collect();
        let embedding_layers: Vec<_> = embedding_host
            .iter()
            .flatten()
            .map(|n| layer_parameters::<P>(n, 1))
            .collect();
        let embedding = ScalarNetwork::new_set(device, &embedding_layers).map_err(same_shape)?;
        let normalization = || descriptors.iter().flat_map(|d| &d.normalization);
        let davg: Vec<f64> = normalization().flat_map(|n| n.mean).collect();
        let dstd_inv: Vec<f64> = normalization()
            .flat_map(|n| n.std.map(|std| 1.0 / std))
            .collect();
        // As the environment kernel normalizes an empty slot: its whole row
        // for the descriptor, the radial entry for the embedding nets.
        let centers = descriptors.len() * ntypes;
        let empty_row: Vec<P> = davg
            .iter()
            .zip(&dstd_inv)
            .map(|(mean, inv)| P::from_f64((0.0 - mean) * inv))
            .collect();
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
                device,
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
        let sel_inv: Vec<f64> = d
            .sel
            .iter()
            .map(|&s| if s == 0 { 0.0 } else { 1.0 / s as f64 })
            .collect();
        let function = |name: &str| {
            device
                .kernels()
                .function(SOURCE, &format!("{name}_{}", P::SUFFIX))
        };
        let three_body = kind == DescriptorKind::SeE3;
        // The environment kernel serves se_e2_a and se_e2_r; se_e3 has its
        // rows from a kernel of its own.
        let kernels = Kernels {
            env: function("se_e2_a_env")?,
            descriptor: function(&format!("{}_descriptor", kind.name()))?,
            backward: function(&format!("{}_backward", kind.name()))?,
            count: match three_body {
                true => Some(device.kernels().function(SOURCE, "se_e3_count")?),
                false => None,
            },
            pairs: match three_body {
                true => Some(function("se_e3_pairs")?),
                false => None,
            },
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
                    table: None,
                })
            }
        };
        Ok(Self {
            kind,
            rcut: d.rcut,
            rcut_smth: d.rcut_smth,
            inv_range: 1.0 / (d.rcut - d.rcut_smth),
            slots,
            width,
            axis: d.axis_neuron,
            features: d.features(),
            first_feature,
            embedding,
            embedding_host,
            radial_normalization: normalization().map(|n| (n.mean[0], n.std[0])).collect(),
            tables: None,
            davg: device.upload(&davg)?,
            dstd_inv: device.upload(&dstd_inv)?,
            empty_row: device.upload(&empty_row)?,
            empty_g,
            slot_type: device.upload(&slot_type)?,
            slot_local: device.upload(&slot_local)?,
            sel_d: device.upload(&sel_i)?,
            sel_inv: device.upload(&sel_inv)?,
            offset_d: device.upload(&offsets)?,
            kernels,
            backward_shared,
            view,
        })
    }

    /// Device bytes per center atom in one chunk for what goes by slot: the
    /// radial input and the net of each row, the embedding values and
    /// tangents, and for se_e2_a the environment rows and B. The embedding
    /// rows of se_e3 do not go by atom: see [`bytes_per_pair`].
    fn bytes_per_atom(&self) -> usize {
        let p = std::mem::size_of::<P>();
        let rows = self.slots * (p + 4 + 2 * self.width * p);
        match self.kind {
            DescriptorKind::SeE2A => rows + self.slots * 4 * p + self.width * 4 * p,
            DescriptorKind::SeE2R => rows,
            DescriptorKind::SeE3 => 0,
        }
    }

    /// The neighbor slots the kernels of this descriptor read (neighbors,
    /// vectors, distances): the list itself, or its own table of it, which
    /// [`Self::read_list`] wrote for this search.
    fn slots_of<'a>(
        &'a self,
        list: &'a NeighborList,
    ) -> Result<(&'a CudaSlice<i32>, &'a CudaSlice<f64>, &'a CudaSlice<f64>)> {
        match &self.view {
            None => Ok((&list.index, &list.delta, &list.dist)),
            Some(view) => match &view.table {
                Some(table) if table.atoms >= list.natoms => {
                    Ok((&table.index, &table.delta, &table.dist))
                }
                _ => Err(Error(
                    "a descriptor of a hybrid without its table of the neighbor list".into(),
                )),
            },
        }
    }

    /// hybrid: writes this descriptor's table of the neighbor list, if it
    /// has one, and adds to `status` (`[atoms]`, the bits of the search)
    /// that neighbors were dropped where the descriptor has fewer slots
    /// than neighbors within its cutoff.
    fn read_list(
        &mut self,
        device: &Device,
        list: &NeighborList,
        narrow: Option<&CudaFunction>,
        status: &mut CudaViewMut<i32>,
    ) -> Result<()> {
        let Some(view) = &mut self.view else {
            return Ok(());
        };
        let function = narrow
            .ok_or_else(|| Error("a hybrid potential without the kernel of its tables".into()))?;
        let (natoms, slots) = (list.natoms, self.slots);
        if view.table.as_ref().is_none_or(|table| table.atoms < natoms) {
            // SAFETY: the kernel below writes every element of these atoms
            // before anything reads them.
            view.table = Some(unsafe {
                SlotTable {
                    atoms: natoms,
                    index: device.alloc_uninit::<i32>(natoms * slots)?,
                    delta: device.alloc_uninit::<f64>(3 * natoms * slots)?,
                    dist: device.alloc_uninit::<f64>(natoms * slots)?,
                }
            });
        }
        let Some(table) = &mut view.table else {
            return Err(Error("the table of a descriptor was not made".into()));
        };
        let (natoms_i, slots_i) = (natoms as i32, slots as i32);
        let (list_slots_i, ntypes_i) = (view.list_slots as i32, list.sel.len() as i32);
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
            .arg(status);
        let config = LaunchConfig {
            grid_dim: ((natoms * slots).div_ceil(BLOCK as usize) as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { launch.launch(config) }
            .map(drop)
            .map_err(launch_error("slot table"))
    }

    /// The embedding nets of this descriptor as tables: see
    /// [`GpuPotential::compress`].
    fn tabulate(&self, device: &Device, min_distance: f64, step: f64) -> Result<ScalarTable<P>> {
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
        // Every net with the range of its center type: (net, lower end,
        // intervals, tail intervals).
        let mut nets = Vec::new();
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
            nets.extend(row.iter().map(|network| (network, lower, intervals, tail)));
        }
        // A table is a few thousand evaluations of its net with two
        // derivatives, and the nets do not depend on each other: all cores.
        let tables = in_parallel(&nets, |&(network, lower, intervals, tail)| {
            let tail_step = TAIL_STEP_FACTOR * step;
            let upper = lower + intervals as f64 * step;
            Ok(TableSegments {
                lower,
                step,
                coefficients: segment(network, lower, step, intervals)?,
                tail_step,
                tail_coefficients: segment(network, upper, tail_step, tail)?,
            })
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
        ScalarTable::new_set(device, width, &tables)
    }
}

impl<P: Scalar> GpuPotential<P> {
    pub fn new(model: &Model, device: Arc<Device>) -> Result<Self> {
        Self::committee(&[model], device)
    }

    /// Several models as one potential that evaluates them together (see
    /// [`Self::evaluate_frames`]): a committee, for model deviation. The
    /// models must share the type map, the descriptor's cutoffs, `sel` and
    /// `axis_neuron` (of every descriptor of a hybrid) and the shapes of
    /// their networks; weights, statistics and energy biases are each
    /// model's own. As a [`Potential`] the committee is its first model.
    pub fn committee(models: &[&Model], device: Arc<Device>) -> Result<Self> {
        let Some(&first) = models.first() else {
            return Err(Error("a committee needs at least one model".into()));
        };
        let mut metadata = first.validate()?;
        let d = &first.descriptor;
        metadata.backend = match d.descriptor_kind() {
            Some(DescriptorKind::SeE2A) => "native-gpu-se-e2-a".into(),
            Some(DescriptorKind::SeE2R) => "native-gpu-se-e2-r".into(),
            Some(DescriptorKind::SeE3) => "native-gpu-se-e3".into(),
            None => "native-gpu-hybrid".into(),
        };
        // Network precision; geometry, forces and virial are float64 either way.
        metadata.precision = if std::mem::size_of::<P>() == 4 {
            "float32".into()
        } else {
            "float64".into()
        };
        let ntypes = first.type_map.len();
        for (index, model) in models.iter().enumerate().skip(1) {
            model.validate()?;
            let other = &model.descriptor;
            let alike = other.kind == d.kind
                && model.type_map == first.type_map
                && other.parts().len() == d.parts().len()
                && other
                    .parts()
                    .iter()
                    .zip(d.parts())
                    .all(|(a, b)| same_layout(a, b));
            if !alike {
                return Err(Error(format!(
                    "committee member {index} differs from the first in its descriptor, type map, cutoffs, sel, axis_neuron or embedding width; such models are evaluated one by one"
                )));
            }
        }
        let same_shape = |e: Error| match models.len() {
            1 => e,
            _ => Error(format!(
                "the members of a committee need networks of one shape: {e}"
            )),
        };
        // The descriptors of the models, each with the same one of every
        // member; their components follow each other.
        let mut parts = Vec::with_capacity(d.parts().len());
        let mut features = 0;
        for index in 0..d.parts().len() {
            let descriptors: Vec<&Descriptor> = models
                .iter()
                .map(|model| &model.descriptor.parts()[index])
                .collect();
            let part = Part::<P>::new(
                &device,
                &descriptors,
                ntypes,
                &d.sel,
                d.rcut,
                features,
                &same_shape,
            )?;
            features += part.features;
            parts.push(part);
        }
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
        let narrow = match parts.iter().any(|part| part.view.is_some()) {
            true => Some(device.kernels().function(SOURCE, "hybrid_slots")?),
            false => None,
        };
        let fitting_width = first
            .fitting
            .iter()
            .flat_map(|n| n.layers.iter().map(|l| l.bias.len()))
            .sum::<usize>();
        let (free, _) = device.memory_info()?;
        // Device bytes per center atom in one chunk: what the descriptors
        // keep by slot, the descriptor and its adjoint, the fitting tapes.
        let p = std::mem::size_of::<P>();
        let per_atom = parts.iter().map(Part::bytes_per_atom).sum::<usize>()
            + 2 * features * p
            + 4 * fitting_width * p;
        let budget = (free as f64 * MEMORY_FRACTION) as usize;
        // se_e3: half for what goes by atom, half for the rows of the pairs,
        // shared between the se_e3 descriptors of the model.
        let three_body = parts
            .iter()
            .filter(|part| part.kind == DescriptorKind::SeE3)
            .count();
        let (by_atom, by_pair) = match three_body {
            0 => (budget, 0),
            _ => (budget / 2, budget / 2),
        };
        let chunk_atoms = (by_atom / per_atom.max(1)).max(1);
        let chunk_pairs = parts
            .iter()
            .filter(|part| part.kind == DescriptorKind::SeE3)
            .map(|part| by_pair / three_body / bytes_per_pair::<P>(part.width))
            .min()
            .unwrap_or(1)
            .max(1);
        Ok(Self {
            metadata,
            models: models.len(),
            sel: d.sel.clone(),
            parts,
            features,
            bias: models
                .iter()
                .flat_map(|model| model.atom_energy_bias.iter().copied())
                .collect(),
            table_fallbacks: 0,
            fitting,
            fused_fitting,
            fused_fitting_max_atoms: if std::mem::size_of::<P>() == 4 {
                FUSED_FITTING_ATOMS_PER_TYPE_F32 * ntypes
            } else {
                usize::MAX
            },
            narrow,
            chunk_atoms,
            chunk_pairs,
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

    /// Whether the last frame had more neighbors than `sel` for some center:
    /// in the neighbor list, or for a descriptor of a hybrid within its own
    /// cutoff.
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

    /// se_e3: the rows of the embedding nets (pairs of an atom's neighbors,
    /// n(n + 1)/2 for n neighbors) evaluated per chunk. Their values and
    /// tangents are what takes the device memory of such a model, so a
    /// chunk of atoms ends where its rows would pass this number, before
    /// [`Self::chunk_atoms`] if need be; an atom is never split. The
    /// members of a committee share it like the atoms. A hybrid with
    /// several se_e3 descriptors has this many rows of each.
    pub fn chunk_pairs(&self) -> usize {
        self.chunk_pairs
    }

    /// Override the number of rows per chunk derived from free device
    /// memory at load time.
    pub fn set_chunk_pairs(&mut self, pairs: usize) {
        self.chunk_pairs = pairs.max(1);
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
    ///
    /// The nets of se_e3 are not tabulated. A hybrid gets tables for its
    /// se_e2_a and se_e2_r descriptors, each over the range of its own
    /// cutoff and statistics, and keeps the networks of its se_e3
    /// descriptors; a model with nothing to tabulate is refused.
    pub fn compress(&mut self, min_distance: f64, step: f64) -> Result<()> {
        let tabulated = |part: &Part<P>| part.kind != DescriptorKind::SeE3;
        if !self.parts.iter().any(tabulated) {
            return Err(Error(
                "the embedding nets of se_e3 are not tabulated: evaluate the model without compression"
                    .into(),
            ));
        }
        // The shortest cutoff of the descriptors that get tables.
        let rcut = self
            .parts
            .iter()
            .filter(|part| tabulated(part))
            .map(|part| part.rcut)
            .fold(f64::INFINITY, f64::min);
        if !(min_distance.is_finite() && 0.0 < min_distance && min_distance < rcut) {
            return Err(Error(format!(
                "compression needs 0 < min_distance < rcut, got {min_distance}"
            )));
        }
        if !(step.is_finite() && 1e-4 <= step && step <= 1.0) {
            return Err(Error(format!(
                "table step must be in [1e-4, 1], got {step}"
            )));
        }
        // All tables first: an error leaves the potential as it was.
        let mut tables = Vec::with_capacity(self.parts.len());
        for part in &self.parts {
            tables.push(match tabulated(part) {
                true => Some(part.tabulate(&self.device, min_distance, step)?),
                false => None,
            });
        }
        for (part, table) in self.parts.iter_mut().zip(tables) {
            part.tables = table;
        }
        Ok(())
    }

    /// Whether embedding nets are tabulated (see [`Self::compress`]).
    pub fn is_compressed(&self) -> bool {
        self.parts.iter().any(|part| part.tables.is_some())
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
        let mut result = self.run(&device, frames, members, &mut clock, self.is_compressed());
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
        // hybrid: the tables of the descriptors that do not read the list
        // as it is. They add to the status what the search cannot know.
        for part in &mut self.parts {
            part.read_list(device, list, self.narrow.as_ref(), &mut status)?;
        }
        clock.lap(|t| &mut t.neighbors)?;

        let features = self.features;
        // What a kernel gets for a buffer it is not to touch.
        let no_buffer = 0u64;
        // Atoms per chunk: every model works on a chunk at once.
        let chunk = (self.chunk_atoms / members).clamp(1, natoms);
        // se_e3: the rows of the embedding nets are the pairs of each atom's
        // neighbors, laid out from the neighbors every atom holds. Those of
        // all such descriptors come back in one download.
        let three_body: Vec<usize> = (0..self.parts.len())
            .filter(|&index| self.parts[index].kind == DescriptorKind::SeE3)
            .collect();
        let mut pair_rows: Vec<Option<PairRows>> = self.parts.iter().map(|_| None).collect();
        if !three_body.is_empty() {
            // SAFETY: the kernels write every element before the download.
            let mut held = unsafe { device.alloc_uninit::<i32>(three_body.len() * natoms)? };
            let natoms_i = natoms as i32;
            for (k, &index) in three_body.iter().enumerate() {
                let part = &self.parts[index];
                let (slots_index, _, _) = part.slots_of(list)?;
                let function = part.kernels.count.as_ref().ok_or_else(|| {
                    Error("an se_e3 descriptor without its neighbor count kernel".into())
                })?;
                let slots_i = part.slots as i32;
                let mut mine = held.slice_mut(k * natoms..(k + 1) * natoms);
                let mut launch = stream.launch_builder(function);
                launch
                    .arg(slots_index)
                    .arg(&natoms_i)
                    .arg(&slots_i)
                    .arg(&mut mine);
                let config = LaunchConfig {
                    grid_dim: (natoms.div_ceil(BLOCK as usize) as u32, 1, 1),
                    block_dim: (BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { launch.launch(config) }
                    .map(drop)
                    .map_err(launch_error("neighbor count"))?;
            }
            let held = device.download(&held)?;
            for (k, &index) in three_body.iter().enumerate() {
                let host = pair_starts(&held[k * natoms..(k + 1) * natoms]);
                pair_rows[index] = Some(PairRows {
                    start: device.upload(&host)?,
                    host,
                });
            }
        }
        // A chunk ends where its atoms are too many, or its rows in one of
        // the se_e3 descriptors.
        let chunks = {
            let starts: Vec<&[i64]> = pair_rows
                .iter()
                .flatten()
                .map(|rows| rows.host.as_slice())
                .collect();
            let rows = (self.chunk_pairs / members).max(1);
            pair_chunks(&starts, natoms, chunk, rows)
        };
        // The centers of the largest chunk for all models.
        let most = chunks
            .iter()
            .map(|&(first, end)| end - first)
            .max()
            .unwrap_or(1);
        let wide = members * most;
        // Per-chunk work buffers of every descriptor, reused by every chunk.
        let mut works: Vec<Work<P>> = Vec::with_capacity(self.parts.len());
        for (part, pairs) in self.parts.iter().zip(pair_rows) {
            // The rows of the embedding nets of a chunk: a row per slot, or
            // per pair.
            let rows = match &pairs {
                Some(pairs) => chunks
                    .iter()
                    .map(|&(first, end)| pairs.rows(first, end))
                    .max()
                    .unwrap_or(0)
                    .max(1),
                None => most * part.slots,
            };
            let all_rows = members * rows;
            let angular = part.kind == DescriptorKind::SeE2A;
            // SAFETY: each element a chunk reads was written earlier in the
            // same chunk.
            works.push(unsafe {
                Work {
                    env: match angular {
                        true => Some(device.alloc_uninit::<P>(wide * part.slots * 4)?),
                        false => None,
                    },
                    b: match angular {
                        true => Some(device.alloc_uninit::<P>(wide * part.width * 4)?),
                        false => None,
                    },
                    input: device.alloc_uninit::<P>(all_rows)?,
                    network: device.alloc_uninit::<i32>(all_rows)?,
                    g: device.alloc_uninit::<P>(all_rows * part.width)?,
                    dg: device.alloc_uninit::<P>(all_rows * part.width)?,
                    of_pairs: match &pairs {
                        Some(_) => Some((
                            device.alloc_uninit::<P>(all_rows)?,
                            device.alloc_uninit::<P>(all_rows)?,
                        )),
                        None => None,
                    },
                    pairs,
                }
            });
        }
        // The components of all descriptors of a center one after the
        // other, and dE/d(them): each descriptor has its own from
        // `first_feature` on, every `features` values.
        let mut descriptor = unsafe { device.alloc_uninit::<P>(wide * features)? };
        let mut d_descriptor = unsafe { device.alloc_uninit::<P>(wide * features)? };
        // Adjoint seed of the layer-by-layer fitting path, made on first use.
        let mut ones: Option<CudaSlice<P>> = None;
        let (ntypes_i, natoms_i, stride_i) = (ntypes as i32, natoms as i32, features as i32);
        let deep = members as u32;
        let mut tabulated = false;

        for &(first, end) in &chunks {
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

            // Every descriptor: its rows, the embedding nets of the rows,
            // and its components of the chunk's centers.
            for (part, work) in self.parts.iter().zip(&mut works) {
                let (index, delta, dist) = part.slots_of(list)?;
                let (slots, width) = (part.slots, part.width);
                let (slots_i, w_i, axis_i) = (slots as i32, width as i32, part.axis as i32);
                // se_e2_r: the mean of the embedding rows over the slots, over five.
                let radial_scale = RADIAL_RESCALE / slots as f64;
                // Rows of the embedding nets of this chunk for one model.
                let chunk_rows = match &work.pairs {
                    Some(pairs) => pairs.rows(first, end),
                    None => count * slots,
                };
                let chunk_rows_i = chunk_rows as i64;
                if let (Some(pairs), Some((weight, _))) = (&work.pairs, &mut work.of_pairs) {
                    let function = part.kernels.pairs.as_ref().ok_or_else(|| {
                        Error("an se_e3 descriptor without its pair kernel".into())
                    })?;
                    let mut launch = stream.launch_builder(function);
                    launch
                        .arg(index)
                        .arg(delta)
                        .arg(dist)
                        .arg(atype)
                        .arg(&part.dstd_inv)
                        .arg(&part.slot_type)
                        .arg(&part.slot_local)
                        .arg(&part.sel_d)
                        .arg(&part.sel_inv)
                        .arg(&first_i)
                        .arg(&slots_i)
                        .arg(&ntypes_i)
                        .arg(&part.rcut_smth)
                        .arg(&part.inv_range)
                        .arg(&pairs.start)
                        .arg(&chunk_rows_i)
                        .arg(&mut work.input)
                        .arg(weight)
                        .arg(&mut work.network);
                    let config = LaunchConfig {
                        grid_dim: (count as u32, deep, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: pair_shared(slots, ntypes) as u32,
                    };
                    unsafe { launch.launch(config) }
                        .map(drop)
                        .map_err(launch_error("pairs"))?;
                } else {
                    let mut launch = stream.launch_builder(&part.kernels.env);
                    launch
                        .arg(index)
                        .arg(delta)
                        .arg(dist)
                        .arg(atype)
                        .arg(&part.davg)
                        .arg(&part.dstd_inv)
                        .arg(&part.slot_type)
                        .arg(&part.slot_local)
                        .arg(&part.sel_d)
                        .arg(&part.offset_d)
                        .arg(&first_i)
                        .arg(&count_i)
                        .arg(&slots_i)
                        .arg(&ntypes_i)
                        .arg(&part.rcut_smth)
                        .arg(&part.inv_range);
                    match work.env.as_mut() {
                        Some(env) => launch.arg(env),
                        None => launch.arg(&no_buffer),
                    };
                    launch.arg(&mut work.input).arg(&mut work.network);
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

                // Embedding values and d/d(radial) tangents for every row of
                // the chunk in one launch, [model][neighbor type][local
                // atom][slot][width] (se_e3: [model][local atom][pair][width],
                // and atoms without neighbors have no row).
                if chunk_rows > 0 {
                    let rows = members * chunk_rows;
                    let x = work.input.slice(0..rows);
                    let ids = work.network.slice(0..rows);
                    let mut out = work.g.slice_mut(0..rows * width);
                    let mut dout = work.dg.slice_mut(0..rows * width);
                    match part.tables.as_ref().filter(|_| compressed) {
                        Some(tables) => {
                            tabulated = true;
                            tables.forward_tangent_by_id(
                                device,
                                &x,
                                &ids,
                                rows,
                                &mut out,
                                &mut dout,
                                &mut outside_table,
                            )?
                        }
                        None => part
                            .embedding
                            .forward_tangent_by_id(device, &x, &ids, rows, &mut out, &mut dout)?,
                    }
                }
                clock.lap(|t| &mut t.embedding)?;

                // The components of this descriptor, at their place among
                // those of all.
                let mut d_out = descriptor.slice_mut(part.first_feature..);
                let mut launch = stream.launch_builder(&part.kernels.descriptor);
                let config = match part.kind {
                    DescriptorKind::SeE3 => {
                        let (Some(pairs), Some((weight, _))) = (&work.pairs, &work.of_pairs) else {
                            return Err(Error("an se_e3 descriptor without its rows".into()));
                        };
                        launch
                            .arg(&work.g)
                            .arg(&work.input)
                            .arg(weight)
                            .arg(&pairs.start)
                            .arg(&chunk_rows_i)
                            .arg(&first_i)
                            .arg(&count_i)
                            .arg(&w_i)
                            .arg(&mut d_out)
                            .arg(&stride_i);
                        let warps = descriptor_warps::<P>(width);
                        LaunchConfig {
                            grid_dim: (count as u32, deep, 1),
                            block_dim: (32 * warps as u32, 1, 1),
                            // The sums of every warp.
                            shared_mem_bytes: (warps * width * std::mem::size_of::<P>())
                                .next_multiple_of(8)
                                as u32,
                        }
                    }
                    DescriptorKind::SeE2A => {
                        let (Some(env), Some(b)) = (&work.env, &mut work.b) else {
                            return Err(Error("an se_e2_a descriptor without its rows".into()));
                        };
                        launch
                            .arg(env)
                            .arg(&work.g)
                            .arg(&part.slot_type)
                            .arg(&part.slot_local)
                            .arg(&part.sel_d)
                            .arg(&part.offset_d)
                            .arg(&count_i)
                            .arg(&slots_i)
                            .arg(&w_i)
                            .arg(&axis_i)
                            .arg(b)
                            .arg(&mut d_out)
                            .arg(&ntypes_i)
                            .arg(index)
                            .arg(atype)
                            .arg(&part.empty_g)
                            .arg(&part.empty_row)
                            .arg(&first_i)
                            .arg(&stride_i);
                        LaunchConfig {
                            grid_dim: (count as u32, deep, 1),
                            block_dim: (BLOCK, 1, 1),
                            // B, the atom's environment rows, its neighbors per type.
                            shared_mem_bytes: (4 * (width + slots) * std::mem::size_of::<P>()
                                + 4 * ntypes)
                                .next_multiple_of(8)
                                as u32,
                        }
                    }
                    DescriptorKind::SeE2R => {
                        launch
                            .arg(&work.g)
                            .arg(&part.slot_type)
                            .arg(&part.slot_local)
                            .arg(&part.sel_d)
                            .arg(&part.offset_d)
                            .arg(&count_i)
                            .arg(&slots_i)
                            .arg(&w_i)
                            .arg(&radial_scale)
                            .arg(&mut d_out)
                            .arg(&ntypes_i)
                            .arg(index)
                            .arg(atype)
                            .arg(&part.empty_g)
                            .arg(&first_i)
                            .arg(&stride_i);
                        LaunchConfig {
                            grid_dim: (count as u32, deep, 1),
                            block_dim: (BLOCK, 1, 1),
                            // The atom's neighbors per type.
                            shared_mem_bytes: (4 * ntypes).next_multiple_of(8) as u32,
                        }
                    }
                };
                unsafe { launch.launch(config) }
                    .map(drop)
                    .map_err(launch_error("descriptor"))?;
                clock.lap(|t| &mut t.descriptor)?;
            }

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

            // Forces and virial: every descriptor from its own part of
            // dE/dD. The first writes the virial of the chunk's centers,
            // the others add to it.
            for (number, (part, work)) in self.parts.iter().zip(&mut works).enumerate() {
                let (index, delta, dist) = part.slots_of(list)?;
                let (slots, width) = (part.slots, part.width);
                let (slots_i, w_i, axis_i) = (slots as i32, width as i32, part.axis as i32);
                let radial_scale = RADIAL_RESCALE / slots as f64;
                let add_i = i32::from(number > 0);
                let chunk_rows = match &work.pairs {
                    Some(pairs) => pairs.rows(first, end),
                    None => count * slots,
                };
                let chunk_rows_i = chunk_rows as i64;
                let dd = d_descriptor.slice(part.first_feature..);
                let mut launch = stream.launch_builder(&part.kernels.backward);
                let block_dim = match part.kind {
                    DescriptorKind::SeE3 => {
                        let (Some(pairs), Some((weight, dtheta))) =
                            (&work.pairs, &mut work.of_pairs)
                        else {
                            return Err(Error("an se_e3 descriptor without its rows".into()));
                        };
                        launch
                            .arg(&work.g)
                            .arg(&work.dg)
                            .arg(&work.input)
                            .arg(&*weight)
                            .arg(dtheta)
                            .arg(&dd)
                            .arg(index)
                            .arg(delta)
                            .arg(dist)
                            .arg(atype)
                            .arg(&part.dstd_inv)
                            .arg(&part.slot_type)
                            .arg(&part.slot_local)
                            .arg(&part.sel_d)
                            .arg(&pairs.start)
                            .arg(&chunk_rows_i)
                            .arg(&first_i)
                            .arg(&count_i)
                            .arg(&slots_i)
                            .arg(&w_i)
                            .arg(&ntypes_i)
                            .arg(&part.rcut_smth)
                            .arg(&part.inv_range)
                            .arg(&mut force)
                            .arg(&mut virial)
                            .arg(&natoms_i)
                            .arg(&stride_i)
                            .arg(&add_i);
                        PAIR_BLOCK
                    }
                    DescriptorKind::SeE2A | DescriptorKind::SeE2R => {
                        if let (Some(env), Some(b)) = (&work.env, &work.b) {
                            launch.arg(env).arg(&work.g).arg(&work.dg).arg(b);
                        } else {
                            launch.arg(&work.dg);
                        }
                        launch
                            .arg(&dd)
                            .arg(index)
                            .arg(delta)
                            .arg(dist)
                            .arg(atype)
                            .arg(&part.dstd_inv)
                            .arg(&part.slot_type)
                            .arg(&part.slot_local)
                            .arg(&part.sel_d)
                            .arg(&part.offset_d)
                            .arg(&first_i)
                            .arg(&count_i)
                            .arg(&slots_i)
                            .arg(&w_i);
                        // The axis neurons of se_e2_a, the scale of se_e2_r.
                        match part.kind == DescriptorKind::SeE2R {
                            false => launch.arg(&axis_i),
                            true => launch.arg(&radial_scale),
                        };
                        launch
                            .arg(&part.rcut_smth)
                            .arg(&part.inv_range)
                            .arg(&mut force)
                            .arg(&mut virial)
                            .arg(&ntypes_i)
                            .arg(&natoms_i)
                            .arg(&stride_i)
                            .arg(&add_i);
                        BLOCK
                    }
                };
                let config = LaunchConfig {
                    grid_dim: (count as u32, deep, 1),
                    block_dim: (block_dim, 1, 1),
                    shared_mem_bytes: part.backward_shared as u32,
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
        if tabulated && int_at(&cleared[3 * centers..], 0) != 0 {
            return Ok(None);
        }
        let atomic = P::from_words(energy_part, centers);
        // The energy of an atom for a model: the energies of a chunk are
        // [model][local atom], chunk after chunk.
        let energy_at = |model: usize, k: usize| {
            let (first, end) = chunks[chunks.partition_point(|&(first, _)| first <= k) - 1];
            atomic[members * first + model * (end - first) + (k - first)].to_f64()
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
    /// Run the networks, and what lies between the stored environment
    /// matrix and its adjoint, in float32; the geometry and the sums that
    /// give forces and virial stay float64.
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
        // The descriptors of a hybrid one by one, or the one of the model.
        let parts = || other.parts().iter().zip(d.parts());
        let what = if other.kind != d.kind
            || other.parts().len() != d.parts().len()
            || parts().any(|(a, b)| a.kind != b.kind)
        {
            "descriptor"
        } else if model.type_map != first.type_map {
            "type map"
        } else if parts().any(|(a, b)| a.rcut != b.rcut || a.rcut_smth != b.rcut_smth) {
            "cutoffs"
        } else if parts().any(|(a, b)| a.sel != b.sel) {
            "sel"
        } else if parts().any(|(a, b)| a.axis_neuron != b.axis_neuron) {
            "axis_neuron"
        } else if parts().any(|(a, b)| {
            a.embedding.len() != b.embedding.len()
                || a.embedding.iter().flatten().map(shape).ne(b
                    .embedding
                    .iter()
                    .flatten()
                    .map(shape))
        }) {
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

#[cfg(test)]
mod tests {
    use super::{in_parallel, pair_chunks, pair_starts};

    /// The rows of one se_e3 descriptor and the chunks they allow.
    fn pair_layout(held: &[i32], atoms: usize, rows: usize) -> (Vec<i64>, Vec<(usize, usize)>) {
        let start = pair_starts(held);
        let chunks = pair_chunks(&[&start], held.len(), atoms, rows);
        (start, chunks)
    }

    #[test]
    fn pairs_of_neighbors_are_laid_out_atom_by_atom_in_chunks() {
        // Two, no, three and one neighbors: three, no, six and one pair.
        let held = [2, 0, 3, 1];
        let (start, chunks) = pair_layout(&held, 2, 7);
        assert_eq!(start, [0, 3, 3, 9, 10]);
        // Two atoms a chunk, and the last two have seven rows together.
        assert_eq!(chunks, [(0, 2), (2, 4)]);
        // At most five rows: the third atom's six are a chunk of their own,
        // since an atom is not split.
        assert_eq!(pair_layout(&held, 4, 5).1, [(0, 2), (2, 3), (3, 4)]);
        // Rows enough, atoms one at a time; and all in one chunk.
        assert_eq!(
            pair_layout(&held, 1, 100).1,
            [(0, 1), (1, 2), (2, 3), (3, 4)]
        );
        assert_eq!(pair_layout(&held, 100, 100).1, [(0, 4)]);
        // Atoms without neighbors have no row and still their chunks.
        assert_eq!(
            pair_layout(&[0, 0, 0], 2, 1),
            (vec![0; 4], vec![(0, 2), (2, 3)])
        );
        assert_eq!(pair_layout(&[], 2, 1), (vec![0], vec![]));
        // Many neighbors: the rows do not fit thirty-two bits.
        let (start, _) = pair_layout(&[65535, 65535], 2, 1);
        assert_eq!(start[2], 2 * 65535 * 65536 / 2);
    }

    #[test]
    fn chunks_hold_the_rows_of_every_descriptor_and_plain_ones_only_atoms() {
        // No se_e3 descriptor: the atoms in chunks of the limit.
        assert_eq!(pair_chunks(&[], 5, 2, 1), [(0, 2), (2, 4), (4, 5)]);
        assert_eq!(pair_chunks(&[], 0, 2, 1), []);
        // A limit of no atoms is one atom.
        assert_eq!(pair_chunks(&[], 2, 0, 1), [(0, 1), (1, 2)]);
        // Two se_e3 descriptors of a hybrid: a chunk ends where the rows of
        // either would be too many. The first has 3, 0, 6, 1 rows, the
        // second 1, 10, 1, 1.
        let (a, b) = (pair_starts(&[2, 0, 3, 1]), pair_starts(&[1, 4, 1, 1]));
        assert_eq!(pair_chunks(&[&a], 4, 4, 9), [(0, 3), (3, 4)]);
        assert_eq!(pair_chunks(&[&b], 4, 4, 9), [(0, 1), (1, 2), (2, 4)]);
        assert_eq!(pair_chunks(&[&a, &b], 4, 4, 9), [(0, 1), (1, 2), (2, 4)]);
        assert_eq!(pair_chunks(&[&a, &b], 4, 4, 12), [(0, 3), (3, 4)]);
        assert_eq!(pair_chunks(&[&b, &a], 4, 4, 12), [(0, 3), (3, 4)]);
        assert_eq!(pair_chunks(&[&a, &b], 4, 4, 13), [(0, 4)]);
        assert_eq!(pair_chunks(&[&a, &b], 4, 2, 13), [(0, 2), (2, 4)]);
    }

    #[test]
    fn parallel_results_keep_the_order_of_their_items() {
        for count in [0usize, 1, 2, 7, 64, 1000] {
            let items: Vec<usize> = (0..count).collect();
            let squares = in_parallel(&items, |&item| item * item);
            assert_eq!(squares, items.iter().map(|i| i * i).collect::<Vec<_>>());
        }
        // Errors come back in place, like any other result.
        let odd = in_parallel(&[1, 2, 3], |&item| {
            (item % 2 == 1).then_some(item).ok_or(item)
        });
        assert_eq!(odd, [Ok(1), Err(2), Ok(3)]);
    }
}
