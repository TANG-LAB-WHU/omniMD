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
//! themselves are the same kernel. With tables for these nets (in pieces of
//! uneven length: see [`GpuPotential::compress`]) no rows of them are kept:
//! the descriptor and force kernels read the tables themselves, and a frame
//! of few atoms has its rows laid out on the device, without that download
//! (see [`GpuPotential::layout_atoms`]).
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
use crate::adaptive_table::{NetTable, TableOptions, TableSpan};
use crate::{Activation, Descriptor, DescriptorKind, Model, Network, Residual, RADIAL_RESCALE};
use omnimd_gpu::{
    Activation as DeviceActivation, Cell, CellSegments, CellTable, CudaFunction, CudaSlice,
    CudaViewMut, DenseLayer, DenseNetwork, Device, GradientNetwork, GroupedNetwork, LaunchConfig,
    LayerParameters, Mark, NeighborList, NeighborSearch, PushKernelArg, Scalar, ScalarNetwork,
    CELL_LOOKUP, GROUPED_MAX_GROUPS,
};
use omnimd_ml_contract::{
    Committee, Error, Evaluation, Frame, ModelMetadata, Potential, Result, Units,
};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

/// The kernels of the potential, which the trainer shares: the lookup of
/// the tables in pieces of uneven length, then [`KERNELS`].
pub(crate) fn source() -> &'static str {
    static SOURCE: OnceLock<String> = OnceLock::new();
    SOURCE.get_or_init(|| format!("{CELL_LOOKUP}{KERNELS}"))
}

const KERNELS: &str = r#"
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

// The embedding values of an se_e2_a descriptor with tables, for its
// descriptor and force kernels to look up themselves instead of reading
// what a kernel of the tables stored for every row: the input and the table
// of every row as the environment kernel wrote them (`x`, `network`; -1 for
// an empty slot), and the tables (see `cell_piece`). A row outside its table
// sets `flag`, if there is one.
template <typename P>
struct RowTables {
    const P* x;
    const int* network;
    const P* coef;
    int bits;
    const int* shape;
    const P* range;
    const int* base;
    const P* scale;
    int* flag;
};

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
// With `unshifted` the input of a neighbor's row is its radial entry over
// the width alone, s/std, without the mean taken off: the place of the
// neighbor in the tables of the nets, which count from the entry of a
// neighbor at the cutoff. (The environment rows are normalized as always.)
template <typename P, typename E>
__device__ void env_impl(const int* index, const double* delta, const double* dist,
        const int* atype, const double* davg, const double* dstd_inv, const int* slot_type,
        const int* slot_local, const int* sel, const int* offset, int first, int count, int slots,
        int ntypes, double rmin, double inv_range, E* env, P* radial, int* network,
        const int* rows, int one_side, int unshifted) {
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
    double place = r4[0] * dstd_inv[4 * c];
    // (A descriptor that takes the radial entry only has no use for the
    // rows: it passes no `env`.)
    for (int a = 0; a < 4; ++a) {
        r4[a] = (r4[a] - davg[4 * c + a]) * dstd_inv[4 * c + a];
        if (env) env[4 * (base + idx) + a] = (E)r4[a];
    }
    // Every empty slot of a shared row writes the same two values there.
    radial[row] = (P)(unshifted ? place : r4[0]);
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
//
// With `TABLES` the embedding values are not read from `g` but looked up in
// the tables (see `RowTables`): the piece of every neighbor's row and its
// place there are found once, by a thread per slot, and kept in shared
// memory; every output then evaluates its quintic there as the kernel of
// the tables would have, and gets the number that kernel would have
// stored. A row outside its table sets the flag.
template <typename P, typename E, bool TABLES = false>
__device__ void descriptor_impl(const E* env, const P* g, const int* slot_type,
        const int* slot_local, const int* sel, const int* offset, int count, int slots,
        int width, int axis, E* b_out, P* d_out, const int* rows, int ntypes,
        const int* index, const int* atype, const P* g_empty, const E* empty_row, int first,
        int stride, RowTables<P> tables = RowTables<P>()) {
    extern __shared__ double b_shared[];
    E* b = (E*)b_shared;                    // [width][4]
    E* near = b + 4 * width;                // without `rows`: this atom's rows, [slots][4]
    P* place = (P*)(near + 4 * slots);      // with tables: the place of each slot's row, [slots]
    int* filled = (int*)(place + (TABLES ? slots : 0)); // its neighbors of each type, [ntypes]
    int* piece = filled + ntypes;           // with tables: the piece of each slot's row, [slots]
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
        if (TABLES) {
            // An empty slot has no row: its net is -1, and nothing reads
            // its piece.
            long long before = (long long)model * count * slots;
            for (int s = threadIdx.x; s < slots; s += blockDim.x) {
                long long row = before + embed_row(count, i, slot_type[s], slot_local[s], sel, offset);
                int net = tables.network[row];
                P u = 0, du;
                piece[s] = net < 0 ? 0
                    : cell_piece(tables.x[row], net, tables.bits, tables.shape, tables.range,
                                 tables.base, tables.scale, &u, &du, tables.flag);
                place[s] = u;
            }
        }
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
                    E gm;
                    if (TABLES) {
                        int s = offset[t] + k;
                        gm = (E)cell_value(tables.coef + ((long long)piece[s] * width + m) * 6,
                                           place[s]);
                    } else {
                        gm = (E)mine[(long long)k * width];
                    }
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
// they are summed over the block in `red` ([12]) and only then go to global
// memory: the force with an atomic, since other blocks add to it for their
// neighbors, the virial as the block's own. With `add` the virial is added
// to what is there: the kernels of the descriptors of a hybrid run one after
// the other, and all but the first add.
//
// The twelve sums are in double precision, where an instruction takes a
// warp as long with one lane at work as with all, and the reduction of a
// warp's lanes is five of them with half the lanes at work, then a quarter,
// and so on. With `all` ([12][threads] of shared memory) the threads
// therefore put down what they hold, and a warp for each sum adds up the
// warps lane by lane and reduces its lanes once: twelve reductions a block
// instead of twelve a warp. Without it every warp reduces its own twelve and
// adds them to `red` (cleared before) with atomics.
//
// Either way a barrier of the block comes before the first reduction. The
// threads leave the loop over their slots at different times when an atom
// has more neighbors than the block has threads, and without the barrier
// the sums of such a block came out wrong in the force kernel of se_e3: the
// force on an atom with 257 to 300 neighbors was 4e-9 to 1.5e-8 eV/A off, in
// every run alike, and the forces of the frame added up to that much instead
// of to nothing (what the block gave its neighbors was right). Found
// 2026-11-19 and not explained: a barrier of the warp in that place set it
// right for one model and not for another, a barrier of the block between
// the reductions and the atomics did not, and the two-body kernels summed
// rightly without any in the three cases that were tried.
__device__ void block_sums(double* fi, double* vir, double* red, double* all, double* force,
        double* virial, int i, int add) {
    int threads = blockDim.x, lane = threadIdx.x & 31;
    if (all) {
        for (int c = 0; c < 3; ++c) all[c * threads + threadIdx.x] = fi[c];
        for (int e = 0; e < 9; ++e) all[(3 + e) * threads + threadIdx.x] = vir[e];
        __syncthreads();
        int warps = threads >> 5;
        for (int v = threadIdx.x >> 5; v < 12; v += warps) {
            double sum = 0.0;
            for (int w = 0; w < warps; ++w) sum += all[v * threads + 32 * w + lane];
            sum = warp_sum(sum);
            if (lane == 0) red[v] = sum;
        }
    } else {
        __syncthreads();
        for (int c = 0; c < 3; ++c) fi[c] = warp_sum(fi[c]);
        for (int e = 0; e < 9; ++e) vir[e] = warp_sum(vir[e]);
        if (lane == 0) {
            for (int c = 0; c < 3; ++c) atomicAdd(&red[c], fi[c]);
            for (int e = 0; e < 9; ++e) atomicAdd(&red[3 + e], vir[e]);
        }
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
//     precision into dE/d(env row), which goes to shared memory as it is,
//     five sums a slot: a warp has one slot here, and what its first lane
//     did in double precision took every other warp of the block as long;
//  3. one thread per slot turns the five into float64 and applies the
//     float64 environment Jacobian; only the neighbor's force needs a global
//     atomic, the center force and the virial are reduced over the block
//     first (`block_sums`, with room for it when bit 1 of `add` is set).
// With `TABLES` phase 2 looks G and dG/dr up in the tables (see `RowTables`)
// instead of reading them: the piece of every neighbor's row, its place
// there and du/dx are found with B, by a thread per slot, and the lanes of a
// warp evaluate the quintics of their outputs and their derivatives as the
// kernel of the tables would have.
template <typename P, typename E, bool TABLES = false>
__device__ void backward_impl(const E* env, const P* g, const P* dg, const E* b_in,
        const P* dd_in, const int* index, const double* delta, const double* dist,
        const int* atype, const double* dstd_inv, const int* slot_type, const int* slot_local,
        const int* sel, const int* offset, int first, int count, int slots, int width, int axis,
        double rmin, double inv_range, double* force, double* virial, const int* rows,
        int ntypes, int natoms, int stride, int add, RowTables<P> tables = RowTables<P>()) {
    extern __shared__ double shared[];
    P* dr_all = (P*)shared;              // [slots][5]: dE/d(env row), the tangent term
    double* red = shared + (5 * slots * sizeof(P) + 7) / 8;  // [12]: center force, virial
    E* b = (E*)(red + 12);               // [width][4]
    P* db = (P*)(b + 4 * width);         // [4][width]
    // With tables: the place of each slot's row and du/dx there, [slots]
    // each, and its piece, [slots].
    P* place = db + 4 * width;
    P* slope = place + slots;
    int* piece = (int*)(slope + slots);
    char* after = (char*)place + (TABLES ? (2 * slots * sizeof(P) + 4 * slots + 7) / 8 * 8 : 0);
    // [12][threads], what every thread adds to the block's sums.
    double* all = (add & 2) ? (double*)after : nullptr;
    add &= 1;
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
    if (TABLES) {
        for (int s = threadIdx.x; s < slots; s += blockDim.x) {
            long long row = before + embed_row(count, li, slot_type[s], slot_local[s], sel, offset);
            int net = tables.network[row];
            P u = 0, du = 0;
            piece[s] = net < 0 ? 0
                : cell_piece(tables.x[row], net, tables.bits, tables.shape, tables.range,
                             tables.base, tables.scale, &u, &du, tables.flag);
            place[s] = u;
            slope[s] = du;
        }
    }
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
        if (TABLES) {
            const P* of_piece = tables.coef + (long long)piece[s] * width * 6;
            P u = place[s], du = slope[s];
            for (int m = lane; m < width; m += 32) {
                const P* c = of_piece + m * 6;
                P gm = cell_value(c, u), projection = 0;
                for (int a = 0; a < 4; ++a) {
                    P w = db[a * width + m];
                    dr[a] += w * gm;
                    projection += w * rp[a];
                }
                tangent += projection * cell_tangent(c, u, du);
            }
        } else {
            for (int m = lane; m < width; m += 32) {
                P gm = g[row + m], projection = 0;
                for (int a = 0; a < 4; ++a) {
                    P w = db[a * width + m];
                    dr[a] += w * gm;
                    projection += w * rp[a];
                }
                tangent += projection * dg[row + m];
            }
        }
        for (int a = 0; a < 4; ++a) dr[a] = warp_sum(dr[a]);
        tangent = warp_sum(tangent);
        if (lane == 0) {
            P* mine = dr_all + 5 * s;
            for (int a = 0; a < 4; ++a) mine[a] = dr[a];
            mine[4] = tangent;
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
        const P* mine = dr_all + 5 * s;
        double adj[4];
        for (int a = 0; a < 4; ++a) adj[a] = (double)mine[a] * inv_slots;
        adj[0] += (double)mine[4] * inv_slots;
        for (int a = 0; a < 4; ++a) adj[a] *= inv_std[a];
        double along = adj[0] * dsw + (dsw - across) * (adj[1] * u[0] + adj[2] * u[1] + adj[3] * u[2]);
        for (int c = 0; c < 3; ++c) {
            double grad = u[c] * along + across * adj[c + 1];
            atomicAdd(&force[3 * (long long)j + c], -grad);
            fi[c] += grad;
            for (int a = 0; a < 3; ++a) vir[3 * a + c] -= d[a] * grad;
        }
    }
    block_sums(fi, vir, red, all, force, virial, i, add);
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
    P* ds_all = (P*)radial_shared;       // [slots]: dE/d(radial entry) before its scale
    double* red = radial_shared + (slots * sizeof(P) + 7) / 8;  // [12]: center force, virial
    P* dd = (P*)(red + 12);              // [width]: dE/dD of this atom
    // [12][threads], what every thread adds to the block's sums.
    double* all = (add & 2) ? red + 12 + (width * sizeof(P) + 7) / 8 : nullptr;
    add &= 1;
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
        // As it is: the thread that has the slot below turns it into double.
        if (lane == 0) ds_all[s] = sum;
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
        double along = (double)ds_all[s] * scale * inv_std * dsw;
        for (int c = 0; c < 3; ++c) {
            double grad = d[c] * inv * along;
            atomicAdd(&force[3 * (long long)j + c], -grad);
            fi[c] += grad;
            for (int a = 0; a < 3; ++a) vir[3 * a + c] -= d[a] * grad;
        }
    }
    block_sums(fi, vir, red, all, force, virial, i, add);
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
// n - 1, so that their types do not decrease, and the pairs (j <= k) are the
// atom's n(n + 1)/2 rows. They go by the types of the two neighbors (t <= u,
// t first) and within a pair of types by j, then k: the rows of one
// embedding net follow each other, which the trainer needs for the gradient
// of that net (see pair_row). `row_start` has the first row of every atom of
// the frame as if all its atoms were laid out one after the other; a chunk
// has its rows from that of its first atom on, model after model
// (`chunk_rows` each).

// The row of the pair of neighbors j <= k, of types tj <= tk, among the rows
// of their atom. `first_of[t]` is the number of the first neighbor of type
// t and `first_of[ntypes]` the number of neighbors n. The f neighbors before
// those of type tj have the f·n - f(f - 1)/2 rows before (neighbor j' has
// n - j' of them). Then come the pairs within type tj, row after row of a
// triangle, then for every later type u the pairs with its neighbors, by j
// and by the neighbor of type u. (The rows of one atom are counted in 32
// bits: an atom has far fewer than the 46,000 neighbors that would take.)
__device__ int pair_row(const int* first_of, int ntypes, int tj, int tk, int j, int k) {
    int n = first_of[ntypes], f = first_of[tj], own = first_of[tj + 1] - f, lj = j - f;
    int row = f * n - f * (f - 1) / 2;
    if (tk == tj) return row + lj * own - lj * (lj - 1) / 2 + (k - j);
    return row + own * (own + 1) / 2 + own * (first_of[tk] - first_of[tj + 1])
         + lj * (first_of[tk + 1] - first_of[tk]) + (k - first_of[tk]);
}

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

// The same layout made on the device, for a frame whose rows the host need
// not know: `row_start[natoms + 1]`, the first row of every atom and after
// the last the number of all rows, as the host makes it from the neighbors
// the atoms hold. One block with a thread for every atom (and threads
// without one, up to a whole warp): each counts the neighbors of its atom,
// n(n + 1)/2 rows, and a scan over the block's shared memory gives every
// atom the rows of those before it and its own. `sums`: a number per atom.
// The neighbors of a type fill its first slots (as for `count_neighbors`),
// so their number is the place of the first empty slot of the type, which
// halving finds in a few loads where counting takes one for every slot
// (thirteen against 138 for the slots of the water model). A block has one
// multiprocessor, so what its threads load is kept small; the timings did
// not tell the two apart. Every thread goes through every barrier, with an
// atom or without.
extern "C" __global__ void se_e3_layout(const int* index, const int* sel, int natoms, int slots,
        int ntypes, long long* row_start) {
    extern __shared__ long long sums[];
    int t = threadIdx.x;
    bool atom = t < natoms;
    if (atom) {
        const int* mine = index + (long long)t * slots;
        int n = 0;
        for (int u = 0; u < ntypes; ++u) {
            // The first empty slot of the type is one of lo..hi.
            int lo = 0, hi = sel[u];
            while (lo < hi) {
                int middle = (lo + hi) / 2;
                if (mine[middle] >= 0) lo = middle + 1; else hi = middle;
            }
            n += lo;
            mine += sel[u];
        }
        sums[t] = (long long)n * (n + 1) / 2;
    }
    __syncthreads();
    // After the round of `step`, sums[t] holds the rows of the 2·step atoms
    // up to t (or of all up to t, if they are fewer).
    for (int step = 1; step < natoms; step *= 2) {
        long long before = atom && t >= step ? sums[t - step] : 0;
        __syncthreads();
        if (atom) sums[t] += before;
        __syncthreads();
    }
    if (atom) row_start[t + 1] = sums[t];
    if (t == 0) row_start[0] = 0;
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
// Thread j has the pairs (j, k >= j): for every type from that of j on, a
// run of rows.
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
        for (int tk = tj; tk < ntypes; ++tk) {
            int from = tk == tj ? j : first_of[tk], to = first_of[tk + 1];
            if (from >= to) continue;
            long long row = base + pair_row(first_of, ntypes, tj, tk, j, from);
            double once = sel_inv[tj] * sel_inv[tk];
            int net = (model * ntypes + tj) * ntypes + tk;
            for (int k = from; k < to; ++k, ++row) {
                double th = a[3 * j] * a[3 * k] + a[3 * j + 1] * a[3 * k + 1]
                          + a[3 * j + 2] * a[3 * k + 2];
                double orders = (k != j && tk == tj) ? 2.0 : 1.0;
                theta[row] = (P)th;
                weight[row] = (P)(orders * once);
                network[row] = net;
            }
        }
    }
}

// The trainer's: the tangent of θ of every pair when the direction entries
// of the neighbors move by those of `envdot` (the tangent of the normalized
// environment rows, four entries a slot, of which the last three are read).
// θ = a_j·a_k moves by ȧ_j·a_k + a_j·ȧ_k. One block per atom, thread j with
// the pairs (j, k >= j) as above; one model, all atoms of the frame.
template <typename P>
__device__ void three_body_tangent_impl(const int* index, const double* delta, const double* dist,
        const int* atype, const double* dstd_inv, const double* envdot, const int* slot_type,
        const int* slot_local, const int* sel, int slots, int ntypes, double rmin,
        double inv_range, const long long* row_start, P* thetadot) {
    extern __shared__ double tangent_shared[];
    double* a = tangent_shared;          // [slots][3]
    double* ad = a + 3 * slots;          // [slots][3]: their tangents
    int* list = (int*)(ad + 3 * slots);  // [slots]
    int* first_of = list + slots;        // [ntypes + 1]
    int i = blockIdx.x;
    long long at = (long long)i * slots;
    three_body_neighbors(index + at, delta + 3 * at, dist + at, dstd_inv[4 * atype[i] + 1],
                         slot_type, slot_local, sel, slots, ntypes, rmin, inv_range, a, list,
                         first_of);
    int n = first_of[ntypes];
    for (int k = threadIdx.x; k < n; k += blockDim.x)
        for (int c = 0; c < 3; ++c) ad[3 * k + c] = envdot[4 * (at + list[k]) + 1 + c];
    __syncthreads();
    long long base = row_start[i];
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        int tj = slot_type[list[j]];
        for (int tk = tj; tk < ntypes; ++tk) {
            int from = tk == tj ? j : first_of[tk], to = first_of[tk + 1];
            if (from >= to) continue;
            long long row = base + pair_row(first_of, ntypes, tj, tk, j, from);
            for (int k = from; k < to; ++k, ++row)
                thetadot[row] = (P)(ad[3 * j] * a[3 * k] + ad[3 * j + 1] * a[3 * k + 1]
                                  + ad[3 * j + 2] * a[3 * k + 2] + a[3 * j] * ad[3 * k]
                                  + a[3 * j + 1] * ad[3 * k + 1] + a[3 * j + 2] * ad[3 * k + 2]);
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
//
// With TANGENT (the trainer's) the sum is the tangent of D when θ of every
// row moves by `thetadot`: weight·θ̇·(G[row][m] + θ·dG/dθ[row][m]).
template <typename P, int C, bool TANGENT>
__device__ void three_body_descriptor_impl(const P* g, const P* dg, const P* theta,
        const P* thetadot, const P* weight, const long long* row_start, long long chunk_rows,
        int first, int count, int width, P* d_out, int stride) {
    extern __shared__ double descriptor_shared[];
    P* sums = (P*)descriptor_shared;     // [warps][width]
    int li = blockIdx.x, model = blockIdx.y;
    long long base = (long long)model * chunk_rows + (row_start[first + li] - row_start[first]);
    long long rows = row_start[first + li + 1] - row_start[first + li];
    long long center = (long long)model * count + li;
    theta += base;
    weight += base;
    g += base * width;
    if (TANGENT) {
        thetadot += base;
        dg += base * width;
    }
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (int m0 = lane; m0 < width; m0 += 32 * C) {
        P acc[C], part[C];
        for (int c = 0; c < C; ++c) { acc[c] = 0; part[c] = 0; }
        int held = 0;
        for (long long p = warp; p < rows; p += warps) {
            P th = theta[p];
            P scaled = weight[p] * (TANGENT ? thetadot[p] : th);
            const P* row = g + p * width + m0;
            const P* drow = dg + p * width + m0;
            for (int c = 0; c < C; ++c)
                if (m0 + 32 * c < width)
                    part[c] += scaled * (TANGENT ? row[32 * c] + th * drow[32 * c] : row[32 * c]);
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

// The forces of an se_e3 model come from two kernels, each with one block
// per atom. They were one kernel of two phases once; its second phase left
// no room among the sixty-four registers a thread has in a block of a
// thousand, and it has no use for so many threads.
//
// The first: dE/dθ of every pair. One warp per row (lanes over the embedding
// width, so that the rows of G and dG/dθ are read coalesced) contracts dE/dD
// with G + θ·dG/dθ in the network precision and leaves dE/dθ of the pair,
// weight included, at the row's place in `dtheta`. The warps of the block
// share the rows, as in the descriptor kernel. (A thread per row, each
// summing over the width by itself, was no faster for a few hundred atoms
// and up to twice slower for thousands: its loads are scattered.)
template <typename P>
__device__ void three_body_dtheta_impl(const P* g, const P* dg, const P* theta,
        const P* weight, const P* dd_in, const long long* row_start, long long chunk_rows,
        int first, int count, int width, int stride, P* dtheta) {
    extern __shared__ double dtheta_shared[];
    P* dd = (P*)dtheta_shared;           // [width]: dE/dD of this atom
    int li = blockIdx.x, i = first + li, model = blockIdx.y;
    long long center = (long long)model * count + li;
    for (int m = threadIdx.x; m < width; m += blockDim.x) dd[m] = dd_in[center * stride + m];
    __syncthreads();
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
}

// With tables for the embedding nets (see the lookup above) the rows of G
// and dG/dθ are not stored at all: the kernels below work them out from the
// piece of θ where the stored rows would be read. A piece has six numbers
// for every output where a stored row has one or two, but they come from
// tables of a few megabytes that many rows share, where the stored rows are
// hundreds of megabytes that are written once and read twice.
//
// Who goes through the outputs of a row was measured both ways (192 atoms of
// water with 3,600 rows an atom; 224 atoms of a HAP frame with 310):
// - A warp per row, its lanes over the outputs, reads the six numbers of
//   every output of a piece together. In float32 that is the faster way to
//   the components for few rows (0.028 against 0.050 ms for the HAP frame)
//   and as fast for many.
// - A thread per row, alone over the outputs, has every lane of a warp at
//   work in every instruction. In float64 the time goes with the number of
//   instructions in double precision that the warps issue, whatever the
//   number of lanes that take part in one, and a warp per row issues two
//   rounds of them for forty outputs with three in four lanes idle in the
//   second: for the water the components take 0.42 against 1.2 ms and the
//   force stage, of which dE/dθ is the first half, 0.49 against 2.6. dE/dθ
//   is faster with a thread per row in float32 as well (0.11 against 0.26 ms
//   of the force stage for the water) and within two percent of the whole
//   evaluation for the HAP frames.
// So dE/dθ has a thread per row, and the components a warp per row in
// float32 and a thread per row in float64.
//
// The components with a warp per row: D[m] = sum over the atom's rows of
// weight·θ·G[m](θ), warps and lanes as in the kernel of the stored rows. A θ
// outside its table sets the flag (the potential then goes through the frame
// with the nets).
template <typename P, int C>
__device__ void three_body_table_descriptor_impl(const P* theta, const P* weight,
        const int* network, const long long* row_start, long long chunk_rows, int first,
        int count, int width, P* d_out, int stride, const P* coef, int bits, const int* shape,
        const P* range, const int* base, const P* scale, int* flag) {
    extern __shared__ double table_descriptor_shared[];
    P* sums = (P*)table_descriptor_shared;   // [warps][width]
    int li = blockIdx.x, model = blockIdx.y;
    long long start = (long long)model * chunk_rows + (row_start[first + li] - row_start[first]);
    long long rows = row_start[first + li + 1] - row_start[first + li];
    long long center = (long long)model * count + li;
    theta += start;
    weight += start;
    network += start;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (int m0 = lane; m0 < width; m0 += 32 * C) {
        P acc[C], part[C];
        for (int c = 0; c < C; ++c) { acc[c] = 0; part[c] = 0; }
        int held = 0;
        for (long long p = warp; p < rows; p += warps) {
            P th = theta[p], u, du;
            int piece = cell_piece(th, network[p], bits, shape, range, base, scale, &u, &du, flag);
            P scaled = weight[p] * th;
            const P* k = coef + ((long long)piece * width + m0) * 6;
            for (int c = 0; c < C; ++c, k += 192)
                if (m0 + 32 * c < width)
                    part[c] += scaled
                        * (k[0] + u * (k[1] + u * (k[2] + u * (k[3] + u * (k[4] + u * k[5])))));
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

// The components with a thread per row: every thread adds its rows to sums
// of its own in shared memory, and the first threads add those of all.
template <typename P>
__device__ void three_body_table_descriptor_rows_impl(const P* theta, const P* weight,
        const int* network, const long long* row_start, long long chunk_rows, int first,
        int count, int width, P* d_out, int stride, const P* coef, int bits, const int* shape,
        const P* range, const int* base, const P* scale, int* flag) {
    extern __shared__ double table_rows_shared[];
    P* mine = (P*)table_rows_shared + (long long)threadIdx.x * width;   // [threads][width]
    int li = blockIdx.x, model = blockIdx.y;
    long long start = (long long)model * chunk_rows + (row_start[first + li] - row_start[first]);
    long long rows = row_start[first + li + 1] - row_start[first + li];
    long long center = (long long)model * count + li;
    theta += start;
    weight += start;
    network += start;
    for (int m = 0; m < width; ++m) mine[m] = 0;
    for (long long p = threadIdx.x; p < rows; p += blockDim.x) {
        P th = theta[p], u, du;
        int piece = cell_piece(th, network[p], bits, shape, range, base, scale, &u, &du, flag);
        P scaled = weight[p] * th;
        const P* k = coef + (long long)piece * width * 6;
        for (int m = 0; m < width; ++m, k += 6)
            mine[m] += scaled
                * (k[0] + u * (k[1] + u * (k[2] + u * (k[3] + u * (k[4] + u * k[5])))));
    }
    __syncthreads();
    const P* sums = (const P*)table_rows_shared;
    int threads = blockDim.x;
    for (int m = threadIdx.x; m < width; m += blockDim.x) {
        P total = 0;
        for (int t = 0; t < threads; ++t) total += sums[(long long)t * width + m];
        d_out[center * stride + m] = total;
    }
}

// dE/dθ of every pair, weight included: dE/dD contracted with G + θ·dG/dθ.
// Both are quintics in the place u of θ within its piece, so their sum is
// one: the term of u^k counts u^k + θ·(du/dθ)·k·u^(k-1), six numbers of the
// row that every output shares. A thread per row.
template <typename P>
__device__ void three_body_table_dtheta_impl(const P* theta, const P* weight, const int* network,
        const P* dd_in, const long long* row_start, long long chunk_rows, int first, int count,
        int width, int stride, P* dtheta, const P* coef, int bits, const int* shape,
        const P* range, const int* base, const P* scale) {
    extern __shared__ double table_dtheta_shared[];
    P* dd = (P*)table_dtheta_shared;         // [width]: dE/dD of this atom
    int li = blockIdx.x, i = first + li, model = blockIdx.y;
    long long center = (long long)model * count + li;
    for (int m = threadIdx.x; m < width; m += blockDim.x) dd[m] = dd_in[center * stride + m];
    __syncthreads();
    long long start = (long long)model * chunk_rows + (row_start[i] - row_start[first]);
    long long rows = row_start[i + 1] - row_start[i];
    for (long long p = threadIdx.x; p < rows; p += blockDim.x) {
        long long row = start + p;
        P th = theta[row], u, du;
        int piece = cell_piece(th, network[row], bits, shape, range, base, scale, &u, &du,
                               (int*)nullptr);
        P along = th * du, u2 = u * u, u3 = u2 * u, u4 = u3 * u;
        P b1 = u + along, b2 = u2 + P(2) * along * u, b3 = u3 + P(3) * along * u2;
        P b4 = u4 + P(4) * along * u3, b5 = u4 * u + P(5) * along * u4;
        const P* c = coef + (long long)piece * width * 6;
        P sum = 0;
        for (int m = 0; m < width; ++m, c += 6)
            sum += dd[m] * (c[0] + c[1] * b1 + c[2] * b2 + c[3] * b3 + c[4] * b4 + c[5] * b5);
        dtheta[row] = weight[row] * sum;
    }
}

// The second: one thread per neighbor j adds dE/dθ·a_k over its partners k,
// in float64: θ of a pair moves by a_k with a_j, and that of j with itself
// by twice a_j. The row of a pair is that of the smaller number first, so
// the partners go type by type with a row that is stepped, not worked out
// again: those of an earlier type have one row each among the rows of the
// two types, as many apart as there are neighbors of the type of j; of the
// type of j those before it have their rows in a triangle, one fewer apart
// with each; all from j on follow each other. The same thread then applies
// the float64 Jacobian of the direction entries of j, as the last phase of
// the se_e2_a kernel does for a row without its radial entry.
template <typename P>
__device__ void three_body_backward_impl(const P* dtheta, const int* index, const double* delta,
        const double* dist, const int* atype, const double* dstd_inv, const int* slot_type,
        const int* slot_local, const int* sel, const long long* row_start, long long chunk_rows,
        int first, int slots, int ntypes, double rmin, double inv_range, double* force,
        double* virial, int natoms, int add) {
    extern __shared__ double back_shared[];
    double* a = back_shared;             // [slots][3]: directions of the neighbors
    double* red = a + 3 * slots;         // [12]: center force, virial
    int* list = (int*)(red + 12);        // [slots]
    int* first_of = list + slots;        // [ntypes + 1]
    // [12][threads], what every thread adds to the block's sums.
    double* all = (add & 2) ? red + 12 + (slots + ntypes + 2) / 2 : nullptr;
    add &= 1;
    int i = first + blockIdx.x, model = blockIdx.y;
    long long at = (long long)i * slots;
    const double inv_std = dstd_inv[4 * (model * ntypes + atype[i]) + 1];
    force += 3 * (long long)model * natoms;
    virial += 9 * (long long)model * natoms;
    for (int e = threadIdx.x; e < 12; e += blockDim.x) red[e] = 0.0;
    three_body_neighbors(index + at, delta + 3 * at, dist + at, inv_std, slot_type, slot_local,
                         sel, slots, ntypes, rmin, inv_range, a, list, first_of);
    int n = first_of[ntypes];
    // dE/dθ of the atom's pairs, from its first row on.
    dtheta += (long long)model * chunk_rows + (row_start[i] - row_start[first]);
    double fi[3] = {0.0, 0.0, 0.0}, vir[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        int tj = slot_type[list[j]];
        double da[3] = {0.0, 0.0, 0.0};
        for (int tk = 0; tk < ntypes; ++tk) {
            int from = first_of[tk], to = first_of[tk + 1];
            if (from >= to) continue;
            int k = from;
            if (tk <= tj) {
                // Before j: the pair is (k, j).
                int until = tk < tj ? to : j;
                int row = pair_row(first_of, ntypes, tk, tj, from, j);
                // To the row of the next partner: the neighbors of the type
                // of j, or within one type one fewer with every partner.
                int step = tk < tj ? first_of[tj + 1] - first_of[tj] : to - from;
                for (; k < until; ++k) {
                    double w = (double)dtheta[row];
                    for (int c = 0; c < 3; ++c) da[c] += w * a[3 * k + c];
                    if (tk == tj) --step;
                    row += step;
                }
            }
            if (k >= to) continue;
            // From j on: the pair is (j, k), and j with itself counts twice.
            int row = pair_row(first_of, ntypes, tj, tk, j, k);
            for (; k < to; ++k, ++row) {
                double w = (double)dtheta[row];
                if (k == j) w *= 2.0;
                for (int c = 0; c < 3; ++c) da[c] += w * a[3 * k + c];
            }
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
    block_sums(fi, vir, red, all, force, virial, i, add);
}

// The trainer's: the adjoints of every embedding row of an se_e3 descriptor
// and of its derivative along θ, from the adjoints of D (`dbar_in`) and of
// its tangent (`ddbar_in`), both `stride` values an atom. With
// D[m] = sum of weight·θ·G[m] and its tangent sum of weight·θ̇·(G[m] + θ·G'[m])
// over the rows of an atom, a row has weight·(θ·dbar[m] + θ̇·ddbar[m]) for
// G[m] and weight·θ̇·θ·ddbar[m] for G'[m]. One block per atom of the frame:
// its warps share the rows and the lanes of a warp the outputs, so that a
// row is written coalesced. Every row is some atom's: nothing is left
// unwritten.
template <typename P>
__device__ void three_body_reverse_impl(const P* theta, const P* thetadot, const P* weight,
        const P* dbar_in, const P* ddbar_in, const long long* row_start, int width, int stride,
        P* gbar, P* gpbar) {
    extern __shared__ double reverse_shared[];
    P* dbar = (P*)reverse_shared;        // [width]
    P* ddbar = dbar + width;             // [width]
    int i = blockIdx.x;
    for (int m = threadIdx.x; m < width; m += blockDim.x) {
        dbar[m] = dbar_in[(long long)i * stride + m];
        ddbar[m] = ddbar_in[(long long)i * stride + m];
    }
    __syncthreads();
    long long base = row_start[i], rows = row_start[i + 1] - base;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (long long p = warp; p < rows; p += warps) {
        long long row = base + p;
        P value = weight[row] * theta[row], tangent = weight[row] * thetadot[row];
        P second = tangent * theta[row];
        P* out = gbar + row * width;
        P* dout = gpbar + row * width;
        for (int m = lane; m < width; m += 32) {
            out[m] = value * dbar[m] + tangent * ddbar[m];
            dout[m] = second * ddbar[m];
        }
    }
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
// them in the precision of the networks. All take the stride of the
// descriptor in the buffers of all descriptors of the model, and the force
// kernels whether their virial is added to the one that is there. The
// se_e3 kernels serve the potential and the trainer alike; those of the
// tangent of θ, of the tangent of D and of the adjoints of the rows are the
// trainer's, one model deep and over all atoms of a frame.
#define INSTANTIATE(P, S)                                                                      \
extern "C" __global__ void se_e2_a_env_##S(const int* index, const double* delta,                 \
        const double* dist, const int* atype, const double* davg, const double* dstd_inv,         \
        const int* slot_type, const int* slot_local, const int* sel, const int* offset,           \
        int first, int count, int slots, int ntypes, double rmin, double inv_range, P* env,       \
        P* radial, int* network, int unshifted) {                                                 \
    env_impl<P, P>(index, delta, dist, atype, davg, dstd_inv, slot_type, slot_local, sel,         \
                   offset, first, count, slots, ntypes, rmin, inv_range, env, radial, network,    \
                   nullptr, 0, unshifted);                                                        \
}                                                                                                 \
extern "C" __global__ void se_e2_a_env_rows_##S(const int* index, const double* delta,            \
        const double* dist, const int* atype, const double* davg, const double* dstd_inv,         \
        const int* slot_type, const int* slot_local, const int* sel, const int* offset,           \
        int first, int count, int slots, int ntypes, double rmin, double inv_range, double* env,  \
        P* radial, int* network, const int* rows, int one_side) {                                 \
    env_impl<P, double>(index, delta, dist, atype, davg, dstd_inv, slot_type, slot_local, sel,    \
                        offset, first, count, slots, ntypes, rmin, inv_range, env, radial,        \
                        network, rows, one_side, 0);                                              \
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
extern "C" __global__ void se_e2_a_table_descriptor_##S(const P* env, const P* x,                 \
        const int* network, const P* coef, int bits, const int* shape, const P* range,            \
        const int* base, const P* scale, int* flag, const int* slot_type,                         \
        const int* slot_local, const int* sel, const int* offset, int count, int slots,           \
        int width, int axis, P* b_out, P* d_out, int ntypes, const int* index,                    \
        const int* atype, const P* g_empty, const P* empty_row, int first, int stride) {          \
    RowTables<P> tables = {x, network, coef, bits, shape, range, base, scale, flag};              \
    descriptor_impl<P, P, true>(env, nullptr, slot_type, slot_local, sel, offset, count, slots,   \
                                width, axis, b_out, d_out, nullptr, ntypes, index, atype,         \
                                g_empty, empty_row, first, stride, tables);                       \
}                                                                                                 \
extern "C" __global__ void se_e2_a_descriptor_rows_##S(const double* env, const P* g,             \
        const int* slot_type, const int* slot_local, const int* sel, const int* offset,           \
        int count, int slots, int width, int axis, double* b_out, P* d_out, const int* rows,      \
        int ntypes, int stride) {                                                                 \
    descriptor_impl<P, double>(env, g, slot_type, slot_local, sel, offset, count, slots, width,   \
                               axis, b_out, d_out, rows, ntypes, nullptr, nullptr, nullptr,       \
                               nullptr, 0, stride);                                               \
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
extern "C" __global__ void se_e2_a_table_backward_##S(const P* env, const P* x,                   \
        const int* network, const P* coef, int bits, const int* shape, const P* range,            \
        const int* base, const P* scale, const P* b_in, const P* dd_in, const int* index,         \
        const double* delta, const double* dist, const int* atype, const double* dstd_inv,        \
        const int* slot_type, const int* slot_local, const int* sel, const int* offset,           \
        int first, int count, int slots, int width, int axis, double rmin, double inv_range,      \
        double* force, double* virial, int ntypes, int natoms, int stride, int add) {             \
    RowTables<P> tables = {x, network, coef, bits, shape, range, base, scale, nullptr};           \
    backward_impl<P, P, true>(env, nullptr, nullptr, b_in, dd_in, index, delta, dist, atype,      \
                              dstd_inv, slot_type, slot_local, sel, offset, first, count, slots,  \
                              width, axis, rmin, inv_range, force, virial, nullptr, ntypes,       \
                              natoms, stride, add, tables);                                       \
}                                                                                                 \
extern "C" __global__ void se_e2_a_backward_rows_##S(const double* env, const P* g,               \
        const P* dg, const double* b_in, const P* dd_in, const int* index, const double* delta,   \
        const double* dist, const int* atype, const double* dstd_inv, const int* slot_type,       \
        const int* slot_local, const int* sel, const int* offset, int first, int count,           \
        int slots, int width, int axis, double rmin, double inv_range, double* force,             \
        double* virial, const int* rows, int ntypes, int stride, int add) {                       \
    backward_impl<P, double>(env, g, dg, b_in, dd_in, index, delta, dist, atype, dstd_inv,        \
                             slot_type, slot_local, sel, offset, first, count, slots, width,      \
                             axis, rmin, inv_range, force, virial, rows, ntypes, 0, stride,       \
                             add);                                                                \
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
        int width, double scale, P* d_out, const int* rows, int ntypes, int stride) {             \
    radial_descriptor_impl<P, double>(g, slot_type, slot_local, sel, offset, count, slots,        \
                                      width, scale, d_out, ntypes, nullptr, nullptr, nullptr, 0,  \
                                      rows, stride);                                              \
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
        int ntypes, int stride, int add) {                                                        \
    radial_backward_impl<P>(dg, dd_in, index, delta, dist, atype, dstd_inv, slot_type,            \
                            slot_local, sel, offset, first, count, slots, width, scale, rmin,     \
                            inv_range, force, virial, ntypes, 0, rows, stride, add);              \
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
    three_body_descriptor_impl<P, 4, false>(g, nullptr, theta, nullptr, weight, row_start,        \
                                            chunk_rows, first, count, width, d_out, stride);      \
}                                                                                                 \
extern "C" __global__ void se_e3_theta_tangent_##S(const int* index, const double* delta,         \
        const double* dist, const int* atype, const double* dstd_inv, const double* envdot,       \
        const int* slot_type, const int* slot_local, const int* sel, int slots, int ntypes,       \
        double rmin, double inv_range, const long long* row_start, P* thetadot) {                 \
    three_body_tangent_impl<P>(index, delta, dist, atype, dstd_inv, envdot, slot_type,            \
                               slot_local, sel, slots, ntypes, rmin, inv_range, row_start,        \
                               thetadot);                                                         \
}                                                                                                 \
extern "C" __global__ void se_e3_descriptor_tangent_##S(const P* g, const P* dg, const P* theta,  \
        const P* thetadot, const P* weight, const long long* row_start, long long chunk_rows,     \
        int count, int width, P* ddot_out, int stride) {                                          \
    three_body_descriptor_impl<P, 4, true>(g, dg, theta, thetadot, weight, row_start,             \
                                           chunk_rows, 0, count, width, ddot_out, stride);        \
}                                                                                                 \
extern "C" __global__ void se_e3_reverse_##S(const P* theta, const P* thetadot,                   \
        const P* weight, const P* dbar, const P* ddbar, const long long* row_start, int width,    \
        int stride, P* gbar, P* gpbar) {                                                          \
    three_body_reverse_impl<P>(theta, thetadot, weight, dbar, ddbar, row_start, width, stride,    \
                               gbar, gpbar);                                                      \
}                                                                                                 \
extern "C" __global__ void se_e3_dtheta_##S(const P* g, const P* dg, const P* theta,              \
        const P* weight, const P* dd_in, const long long* row_start, long long chunk_rows,        \
        int first, int count, int width, int stride, P* dtheta) {                                 \
    three_body_dtheta_impl<P>(g, dg, theta, weight, dd_in, row_start, chunk_rows, first, count,   \
                              width, stride, dtheta);                                             \
}                                                                                                 \
extern "C" __global__ void se_e3_table_descriptor_##S(const P* theta, const P* weight,            \
        const int* network, const long long* row_start, long long chunk_rows, int first,          \
        int count, int width, P* d_out, int stride, const P* coef, int bits, const int* shape,    \
        const P* range, const int* base, const P* scale, int* flag) {                             \
    three_body_table_descriptor_impl<P, 2>(theta, weight, network, row_start, chunk_rows, first,  \
                                           count, width, d_out, stride, coef, bits, shape,        \
                                           range, base, scale, flag);                             \
}                                                                                                 \
extern "C" __global__ void se_e3_table_descriptor_rows_##S(const P* theta, const P* weight,       \
        const int* network, const long long* row_start, long long chunk_rows, int first,          \
        int count, int width, P* d_out, int stride, const P* coef, int bits, const int* shape,    \
        const P* range, const int* base, const P* scale, int* flag) {                             \
    three_body_table_descriptor_rows_impl<P>(theta, weight, network, row_start, chunk_rows,       \
                                             first, count, width, d_out, stride, coef, bits,      \
                                             shape, range, base, scale, flag);                    \
}                                                                                                 \
extern "C" __global__ void se_e3_table_dtheta_##S(const P* theta, const P* weight,                \
        const int* network, const P* dd_in, const long long* row_start, long long chunk_rows,     \
        int first, int count, int width, int stride, P* dtheta, const P* coef, int bits,          \
        const int* shape, const P* range, const int* base, const P* scale) {                      \
    three_body_table_dtheta_impl<P>(theta, weight, network, dd_in, row_start, chunk_rows, first,  \
                                    count, width, stride, dtheta, coef, bits, shape, range,       \
                                    base, scale);                                                 \
}                                                                                                 \
extern "C" __global__ void se_e3_backward_##S(const P* dtheta, const int* index,                  \
        const double* delta, const double* dist, const int* atype, const double* dstd_inv,        \
        const int* slot_type, const int* slot_local, const int* sel,                              \
        const long long* row_start, long long chunk_rows, int first, int slots, int ntypes,       \
        double rmin, double inv_range, double* force, double* virial, int natoms, int add) {      \
    three_body_backward_impl<P>(dtheta, index, delta, dist, atype, dstd_inv, slot_type,           \
                                slot_local, sel, row_start, chunk_rows, first, slots, ntypes,     \
                                rmin, inv_range, force, virial, natoms, add);                     \
}
INSTANTIATE(float, f32)
INSTANTIATE(double, f64)
"#;

const BLOCK: u32 = 128;
/// Threads of a wide block of the force kernel of a two-body descriptor
/// (see [`FORCE_WAYS`]).
const WIDE_BLOCK: u32 = 256;
/// Where the driver does not say what the device keeps of a force kernel at
/// once, float32 passes of at most this many centers (the atoms of a chunk
/// times the models) give the force kernel of an se_e2_a or se_e2_r
/// descriptor blocks of [`WIDE_BLOCK`] threads instead of [`BLOCK`]; where
/// it says, [`force_way`] chooses.
///
/// That kernel is a block per center, and one of its steps takes the
/// neighbor slots of the center a warp each, the warps of the block in
/// turns: with eight warps a center with 212 slots is through in 27 turns
/// instead of 53. Whether a pass is sooner done for it goes by the number of
/// its blocks. Measured on an RTX 4090 with the HAP models and tables, whole
/// evaluations with the wider blocks against the narrower, the sums of a
/// block formed as they are since the afternoon of 2026-11-19
/// (`bench/results/hap-eval-force-blocks-rtx4090-2026-11-19-r2.json`): 4% to
/// 6% shorter at 54 to 108 centers, 1.2% to 5% shorter at 162 to 352, and
/// 1.6% to 3.2% longer at 528, 896 and 12,288 (within 1.2% at 1,536). While
/// every warp of a block reduced its own sums (sixty additions in double
/// precision a warp, so twice as many for a block of eight warps as for one
/// of four), the gain ended at 224 centers.
///
/// With the kernels of se_e2_a looking their tables up (float32 since
/// 2026-11-21, `bench/results/inline-tables-force-blocks-rtx4090-2026-11-21.json`)
/// the narrower blocks took 0.5% to 4.3% less with the HAP model from 264 to
/// 352 centers, and the wider ones with the sums across the warps 3.1% less
/// with the water model at 384, past both limits. The blocks the device
/// keeps at once serve both (see [`force_way`]).
pub const WIDE_FORCE_CENTERS_F32: usize = 352;
/// The same for float64 passes. There the wider blocks took 1.6% to 4.4% off
/// an evaluation with se_e2_r up to 224 centers, less than 1% from 264 to
/// 352, and added 1.7% at 528. They left one with se_e2_a within 1.7% at
/// every size but one: a committee of 352 centers, 4% longer.
pub const WIDE_FORCE_CENTERS_F64: usize = 224;
/// Where the driver does not say what the device keeps of a force kernel at
/// once, passes of at most this many centers give the force kernel of an
/// se_e2_a descriptor room in shared memory for the sums of a block across
/// its warps (twelve doubles a thread, see `block_sums` in the kernels), and
/// that of se_e2_r has it in a pass of any size; where it says,
/// [`force_way`] chooses. The force kernels of se_e3 have it in a pass of
/// any size either way.
///
/// Measured on an RTX 4090 with the HAP models and tables, whole evaluations
/// with the sums across the warps against every warp reducing its own
/// (`bench/results/hap-eval-block-sums-rtx4090-2026-11-19.json`): with
/// se_e2_a 1.8% to 3.7% shorter in float32 up to 352 centers and 0.2% to
/// 1.4% in float64, and 2.8% to 3.7% longer at 528; with se_e2_r up to 4.7%
/// shorter and never longer from 54 to 528 centers (from 792 to 968, where
/// the narrow blocks with the sums need a second wave and those without do
/// not, 0.1% to 2.1% longer in float32 and between 0.3% shorter and 1.7%
/// longer in float64: `bench/results/force-ways-rtx4090-2026-11-22.json`);
/// with se_e3 up to 3.9% shorter, and longer only in float64 at 88 centers
/// (1.4% and 3.1% in two cases).
pub const BLOCK_SUMS_CENTERS: usize = 352;
/// Static shared-memory limit per block without opting in to more.
const SHARED_LIMIT: usize = 48 * 1024;

/// A way to launch the force kernel of an se_e2_a or se_e2_r descriptor:
/// the threads of a block, and whether a block forms its sums across its
/// warps in shared memory (see `block_sums` in the kernels).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ForceBlocks {
    pub threads: u32,
    pub sums: bool,
}

/// The four ways of [`ForceBlocks`] in the order a pass prefers them: the
/// wide blocks with the sums across the warps, the narrow ones with them,
/// the wide ones without, the narrow ones without (see [`force_way`]).
pub const FORCE_WAYS: [ForceBlocks; 4] = [
    ForceBlocks {
        threads: WIDE_BLOCK,
        sums: true,
    },
    ForceBlocks {
        threads: BLOCK,
        sums: true,
    },
    ForceBlocks {
        threads: WIDE_BLOCK,
        sums: false,
    },
    ForceBlocks {
        threads: BLOCK,
        sums: false,
    },
];

/// The blocks of a force kernel with `own` bytes of shared memory besides
/// the sums of a block that one multiprocessor keeps at once, in each way
/// of [`FORCE_WAYS`]: zero for a way past [`SHARED_LIMIT`], None when the
/// driver does not say.
fn resident_force_blocks(function: &CudaFunction, own: usize) -> Option<[u32; 4]> {
    let mut blocks = [0; 4];
    for (way, resident) in FORCE_WAYS.iter().zip(&mut blocks) {
        let shared = own
            + match way.sums {
                true => block_sums_shared(way.threads),
                false => 0,
            };
        if shared <= SHARED_LIMIT {
            *resident = function
                .occupancy_max_active_blocks_per_multiprocessor(way.threads, shared, None)
                .ok()?;
        }
    }
    Some(blocks)
}

/// The way of the force kernel of an se_e2_a or se_e2_r descriptor in a
/// pass of `blocks` blocks (its centers times the models): the first of
/// [`FORCE_WAYS`] that the overrides leave (`wide` and `sums`, where set)
/// and whose blocks the device keeps all at once. `capacity` holds what it
/// keeps of each way, in their order: the blocks one multiprocessor keeps
/// times the multiprocessors, zero for a way without room for its sums.
/// Past every way that is left, the pass takes the narrow blocks (the wide
/// ones where `wide` says so), with the sums across the warps where
/// `sums_past` (or `sums`) says so and they have room.
///
/// A pass whose blocks the device keeps all at once is done with its
/// slowest block; one that needs a second wave of them waits for it. The
/// wide blocks (a center's slots in half the turns) and the sums across the
/// warps (no warp reducing its own) shorten a block but take more of a
/// multiprocessor, so in float32 they gain while every block of the pass
/// fits (the wide blocks without the sums only up to about 224 blocks: from
/// 264 they came out between 0.8% shorter and 2.8% longer, though the
/// device still kept them all), and a way that keeps them all mostly beats
/// a faster one that does not. In float64 the two widths with the sums
/// came out within 1.5% of each other while both fit, and the wide ones
/// stay first. Measured on an RTX 4090 against the limits of centers that
/// chose before, in two runs
/// (`bench/results/force-ways-rtx4090-2026-11-22.json`: the HAP and water
/// models, 88 to 1,536 centers, both precisions, every way forced as
/// well): where the two choose differently an evaluation was 0.4% to 4.1%
/// shorter in float32 and between 1.3% shorter and 0.4% longer in float64,
/// and the way chosen here was the fastest of the four forced or within
/// 1.5% of it. Past every way the narrow blocks keep their sums for
/// se_e2_r (`sums_past`) and go without for se_e2_a, which suits the water
/// model and not quite the HAP one (`bench/results/SUMMARY.md`).
pub fn force_way(
    blocks: usize,
    capacity: [usize; 4],
    wide: Option<bool>,
    sums: Option<bool>,
    sums_past: bool,
) -> ForceBlocks {
    let left = |way: &ForceBlocks| {
        wide.is_none_or(|wide| wide == (way.threads == WIDE_BLOCK))
            && sums.is_none_or(|sums| sums == way.sums)
    };
    if let Some((way, _)) = FORCE_WAYS
        .iter()
        .zip(capacity)
        .find(|&(way, at_once)| left(way) && blocks <= at_once)
    {
        return *way;
    }
    let threads = match wide {
        Some(true) => WIDE_BLOCK,
        _ => BLOCK,
    };
    let room = FORCE_WAYS
        .iter()
        .zip(capacity)
        .any(|(way, at_once)| way.threads == threads && way.sums && at_once > 0);
    ForceBlocks {
        threads,
        sums: sums.unwrap_or(sums_past) && room,
    }
}

/// The embedding tables reach from `min_distance` down to
/// `min_distance / TAIL_REACH` (DeePMD's default for the part of its tables
/// below the shortest distance of the training data: extrapolate 5).
const TAIL_REACH: f64 = 5.0;
/// The embedding tables (see [`GpuPotential::compress`]): how close a table
/// is to its net at the step 0.01, as a fraction of the largest value of the
/// net, and what the precision of the networks holds of that at best (a
/// float32 rounds at 6e-8, the checks of a table in float64 at 1e-16).
const TABLE_TOLERANCE: f64 = 1e-13;
const TABLE_FLOOR_F32: f64 = 1.5e-8;
const TABLE_FLOOR_F64: f64 = 1e-14;
/// The first bits of an input after its exponent that tell a cell of its
/// table, and the bits of the input that are left for the place within a
/// piece: a piece is no narrower than its far end over two to that power.
const TABLE_KEY_BITS: u32 = 6;
const TABLE_RESOLUTION_F32: u32 = 20;
const TABLE_RESOLUTION_F64: u32 = 28;
/// Without grouped products (cuBLAS before 12.5, nets they do not take, or
/// fewer than `GROUPED_FITTING_MIN_NETS` nets in a chunk), float32 chunks of
/// at most this many atoms per center type and model run the fitting nets in
/// the fused kernels instead of layer-by-layer cuBLAS a net at a time, whose
/// fixed cost (some twenty-five launches per type, 0.68 ms for five types)
/// dominates below that. Measured on an RTX 4090 with
/// [240, 240, 240] fitting nets on 1600 descriptor components, whole passes
/// over batches of a frame in float32: with five types the two paths meet
/// near 1,800 to 2,000 atoms (88-atom frames: 1.28 against 1.37 ms at 1,408
/// atoms, 1.96 against 1.72 ms at 2,112; 169-atom frames: equal at 2,028),
/// with two types near 680 (192-atom frames: 0.60 against 0.63 ms at 576
/// atoms, 0.74 against 0.71 ms at 768).
const FUSED_FITTING_ATOMS_PER_TYPE_F32: usize = 340;
/// Float32 chunks with at least `GROUPED_FITTING_MIN_NETS` fitting nets and
/// at most this many rows of them (atoms times models) run them in the fused
/// kernels; larger ones in grouped products, all nets at once
/// (`GroupedNetwork`), which cost some 0.2 ms more than the fused kernel for
/// one row and grow more slowly. Float64 always uses
/// the fused kernel: both paths are bound by the double-precision units
/// there and the fused one has no fixed cost. Measured on an RTX 4090 with
/// [240, 240, 240] fitting nets on 1600 descriptor components, whole passes
/// over batches in float32 with tables (fused against grouped): HAP frames
/// of 88 atoms 480 against 498 us at 704 rows and 591 against 558 at 880;
/// of 169 atoms 469 against 491 at 676 and 692 against 649 at 1,014; four
/// models on one such frame 490 against 529 at 676 rows and on two 879
/// against 751; water frames of 192 atoms 516 against 518 at 768.
const FUSED_FITTING_ROWS_F32: usize = 800;
/// The grouped products take the place of cuBLAS a net at a time for chunks
/// with at least this many fitting nets (center types times models), or
/// with at least `GROUPED_FITTING_ROWS_FEW_NETS` rows. Their fixed cost does
/// not grow with the nets; that of a net at a time grows by some 0.13 to
/// 0.14 ms a net. Measured as above (grouped against a net at a time):
/// four nets (HAP frames of 88 atoms) 498 against 770 us at 704 rows, 558
/// against 861 at 880; two nets (water frames of 192 atoms) 524 against
/// 500 at 768 rows, 896 against 876 at 1,536, 1,043 against 1,042 at 1,920,
/// 1,179 against 1,205 at 2,304 and 2,015 against 2,103 at 4,608. One net
/// was not measured.
const GROUPED_FITTING_MIN_NETS: usize = 3;
/// See `GROUPED_FITTING_MIN_NETS`.
const GROUPED_FITTING_ROWS_FEW_NETS: usize = 1_920;

/// How the fitting nets of a chunk are evaluated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FittingPath {
    /// Every net in the fused kernels (`GradientNetwork`).
    Fused,
    /// Every net in grouped cuBLAS products (`GroupedNetwork`).
    Grouped,
    /// cuBLAS a net at a time (`DenseNetwork`).
    NetByNet,
}

/// What chooses the fitting path of a chunk, besides its size.
#[derive(Clone, Copy, Debug)]
struct FittingRule {
    /// Float64 networks.
    double: bool,
    /// The center types of a model.
    ntypes: usize,
    /// Whether the fused kernels take these nets.
    fused: bool,
    /// Whether the grouped products take them.
    grouped: bool,
    /// What `GpuPotential::set_fused_fitting_max_rows` set, if called.
    fused_max_rows: Option<usize>,
    /// What `GpuPotential::set_grouped_fitting` set, if called.
    grouped_on: Option<bool>,
}

impl FittingRule {
    /// The path for a chunk with `rows` rows of the fitting nets (its
    /// `atoms` atoms times the models) in `nets` nets (the center types in
    /// it times the models).
    fn path(&self, rows: usize, atoms: usize, nets: usize) -> FittingPath {
        let many_nets = nets >= GROUPED_FITTING_MIN_NETS;
        let grouped = self.grouped
            && nets <= GROUPED_MAX_GROUPS
            && self
                .grouped_on
                .unwrap_or(many_nets || rows >= GROUPED_FITTING_ROWS_FEW_NETS);
        let fused = self.fused
            && match self.fused_max_rows {
                Some(limit) => rows <= limit,
                None if self.double => true,
                None if grouped && many_nets => rows <= FUSED_FITTING_ROWS_F32,
                None => atoms <= FUSED_FITTING_ATOMS_PER_TYPE_F32 * self.ntypes,
            };
        match (fused, grouped) {
            (true, _) => FittingPath::Fused,
            (false, true) => FittingPath::Grouped,
            (false, false) => FittingPath::NetByNet,
        }
    }
}

/// Fraction of the free device memory one chunk of per-atom work may use.
const MEMORY_FRACTION: f64 = 0.5;

struct Kernels {
    env: CudaFunction,
    descriptor: CudaFunction,
    backward: CudaFunction,
    /// se_e3: the neighbors every atom holds, the rows of their pairs, and
    /// dE/dθ of every pair, from which the force kernel goes on.
    count: Option<CudaFunction>,
    /// se_e3: the layout of the rows made on the device (see
    /// [`GpuPotential::layout_atoms`]).
    layout: Option<CudaFunction>,
    pairs: Option<CudaFunction>,
    dtheta: Option<CudaFunction>,
    /// se_e3 with tables: the components and dE/dθ straight from the tables
    /// (the components with a warp per row, or a thread per row: see
    /// [`table_descriptor_threads`]).
    table_descriptor: Option<CudaFunction>,
    table_descriptor_rows: Option<CudaFunction>,
    table_dtheta: Option<CudaFunction>,
    /// se_e2_a with tables: the descriptor and force kernels that look the
    /// embedding values up themselves (see [`GpuPotential::inline_tables`]).
    inline_descriptor: Option<CudaFunction>,
    inline_backward: Option<CudaFunction>,
}

/// The rows of the embedding nets of an se_e3 descriptor for a frame: the
/// pairs of the neighbors of each atom, n(n + 1)/2 for an atom with n
/// neighbors.
struct PairRows {
    /// The first row of every atom, as if all atoms were laid out one after
    /// the other, and the number of all rows after the last.
    start: CudaSlice<i64>,
    known: Laid,
}

/// What the host knows of the rows of a frame.
enum Laid {
    /// It made the layout: the first row of every atom, as on the device.
    OnHost(Vec<i64>),
    /// The device made it, and the host has not seen it: an atom has at
    /// most this many rows (every one of its slots a neighbor).
    OnDevice { most: usize },
}

impl PairRows {
    /// Rows of the atoms `first..end`; for a layout made on the device the
    /// most they can be. That is enough for the two things the host needs
    /// them for with tables: buffers that hold the rows, and how far the
    /// rows of one model are from those of the next.
    fn rows(&self, first: usize, end: usize) -> usize {
        match &self.known {
            Laid::OnHost(start) => (start[end] - start[first]) as usize,
            Laid::OnDevice { most } => (end - first) * most,
        }
    }
}

/// The most rows an atom of an se_e3 descriptor with `slots` neighbor slots
/// can have: the pairs of as many neighbors.
fn most_pairs(slots: usize) -> usize {
    slots * (slots + 1) / 2
}

/// The most atoms of a frame whose se_e3 rows are laid out on the device:
/// the threads of one block, which has one for every atom.
pub const LAYOUT_ATOMS: usize = 1024;

/// The first row of every atom of an se_e3 descriptor, laid out by the
/// kernel `se_e3_layout`: `index` is the table of neighbor slots the
/// descriptor reads, `[atoms][slots]`, and `sel` its slots by type;
/// `shape` is (atoms, slots, types). The launch is asynchronous like any
/// other: nothing comes back to the host.
fn lay_out_rows(
    device: &Device,
    function: &CudaFunction,
    index: &CudaSlice<i32>,
    sel: &CudaSlice<i32>,
    shape: (usize, usize, usize),
) -> Result<CudaSlice<i64>> {
    let (natoms, slots, ntypes) = shape;
    if natoms == 0 || natoms > LAYOUT_ATOMS || index.len() < natoms * slots || sel.len() < ntypes {
        return Err(Error(format!(
            "the rows of {natoms} atoms with {slots} slots are not laid out on the device"
        )));
    }
    let (natoms_i, slots_i, ntypes_i) = (natoms as i32, slots as i32, ntypes as i32);
    // SAFETY: the kernel writes every element.
    let mut start = unsafe { device.alloc_uninit::<i64>(natoms + 1)? };
    let stream = device.stream();
    let mut launch = stream.launch_builder(function);
    launch
        .arg(index)
        .arg(sel)
        .arg(&natoms_i)
        .arg(&slots_i)
        .arg(&ntypes_i)
        .arg(&mut start);
    // One block: a thread for every atom, in whole warps, and a number for
    // every atom in its shared memory.
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (natoms.next_multiple_of(32) as u32, 1, 1),
        shared_mem_bytes: (natoms * std::mem::size_of::<i64>()) as u32,
    };
    unsafe { launch.launch(config) }
        .map(drop)
        .map_err(launch_error("row layout"))?;
    Ok(start)
}

/// What the device makes of a table of neighbor slots when it lays out the
/// rows of an se_e3 descriptor (see [`GpuPotential::layout_atoms`]), brought
/// back for a look: the first row of every atom, and after the last the
/// number of all rows. `index` is `[atoms][slots]`, -1 for an empty slot,
/// with `sel[t]` slots of type `t` one type after the other and the
/// neighbors of a type in its first slots, as a neighbor search leaves
/// them. The potential keeps the layout on the device; this is for the
/// tests of the kernel.
pub fn device_row_starts(device: &Device, index: &[i32], sel: &[usize]) -> Result<Vec<i64>> {
    let slots: usize = sel.iter().sum();
    if slots == 0 || index.len() % slots != 0 {
        return Err(Error(format!(
            "{} neighbor slots are not rows of {slots}",
            index.len()
        )));
    }
    let natoms = index.len() / slots;
    let sel_i: Vec<i32> = sel.iter().map(|&s| s as i32).collect();
    let function = device.kernels().function(source(), "se_e3_layout")?;
    let start = lay_out_rows(
        device,
        &function,
        &device.upload(index)?,
        &device.upload(&sel_i)?,
        (natoms, slots, sel.len()),
    )?;
    device.download(&start)
}

/// The first row of every atom that holds `held[atom]` neighbors, and after
/// the last the number of all rows.
pub(crate) fn pair_starts(held: &[i32]) -> Vec<i64> {
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
/// tangents, θ of the pair, its weight and dE/dθ, and its net. With a width
/// of zero, a row whose embedding values are not kept (they come from
/// tables where they are used).
fn bytes_per_pair<P>(width: usize) -> usize {
    let p = std::mem::size_of::<P>();
    2 * width * p + 3 * p + 4
}

/// Threads per block of the se_e3 kernel of dE/dθ: thirty-two warps, each
/// with a row of the atom at a time. A warp waits for the rows it reads, so
/// a few hundred atoms go faster the more warps share an atom's rows (192
/// atoms in float32: 0.43 ms with thirty-two warps, 0.67 with eight);
/// thousands of atoms are bound by reading the rows either way. The
/// trainer's kernel of the adjoints of the rows runs the same way. A block
/// of this size leaves a thread 64 registers: these kernels take 40 to 48
/// (`ptxas -v` on the cached PTX tells).
pub(crate) const PAIR_BLOCK: u32 = 1024;

/// The most warps of a block of the se_e3 descriptor kernel that reads the
/// tables, with a warp per row and with a thread per row. (A block of
/// thirty-two warps leaves a thread 64 registers: the kernel with a warp per
/// row takes 38, in float64 39.)
const TABLE_DESCRIPTOR_WARPS: usize = 32;
const TABLE_ROW_WARPS: usize = 8;

/// Threads per block of the se_e3 kernel of dE/dθ that reads the tables, a
/// thread per row: for 310 rows an atom two percent of an evaluation faster
/// than a thousand, for 3,600 the same.
const PAIR_TABLE_BLOCK: u32 = 256;

/// Threads per block of the se_e3 force kernel, which has a thread for
/// every neighbor of the atom (one with more neighbors gives its threads
/// several).
pub(crate) const PAIR_FORCE_BLOCK: u32 = 256;

/// Shared memory of the se_e3 kernel of dE/dθ per block: dE/dD of the atom
/// in the precision of the networks.
pub(crate) fn pair_dtheta_shared<P>(width: usize) -> usize {
    (width * std::mem::size_of::<P>()).next_multiple_of(8)
}

/// Shared memory of the se_e3 force kernel per block: the directions of the
/// neighbors and the block's sums in double, the neighbors' slots and the
/// first neighbor of each type.
pub(crate) fn pair_force_shared(slots: usize, ntypes: usize) -> usize {
    (3 * slots + 12) * 8 + (4 * (slots + ntypes + 1)).next_multiple_of(8)
}

/// Shared memory of the se_e2_a force kernel per block: five sums a slot in
/// the precision of the networks, the block's sums in double, B in `b_size`
/// bytes a number (those of the networks for the potential, double for the
/// trainer) and its adjoint in the precision of the networks.
pub(crate) fn angular_force_shared<P>(slots: usize, width: usize, b_size: usize) -> usize {
    let size = std::mem::size_of::<P>();
    (5 * slots * size).next_multiple_of(8) + 12 * 8 + 4 * width * (b_size + size)
}

/// Shared memory the se_e2_a force kernel that looks the embedding values
/// up in the tables needs on top of [`angular_force_shared`]: the place of
/// every slot's row in its piece and du/dx there in the precision of the
/// networks, and the piece.
fn angular_table_shared<P>(slots: usize) -> usize {
    (2 * slots * std::mem::size_of::<P>() + 4 * slots).next_multiple_of(8)
}

/// Shared memory of the se_e2_a descriptor kernel of the potential per
/// block: B and the atom's environment rows in the precision of the
/// networks, its neighbors of each type, and when it looks the embedding
/// values up in the tables the place and the piece of every slot's row.
fn angular_descriptor_shared<P>(slots: usize, width: usize, ntypes: usize, tables: bool) -> usize {
    let size = std::mem::size_of::<P>();
    let looked_up = match tables {
        true => slots * (size + 4),
        false => 0,
    };
    (4 * (width + slots) * size + looked_up + 4 * ntypes).next_multiple_of(8)
}

/// Shared memory of the se_e2_r force kernel per block: dE/d(radial entry)
/// of every slot and dE/dD in the precision of the networks, the block's
/// sums in double.
pub(crate) fn radial_force_shared<P>(slots: usize, width: usize) -> usize {
    let size = std::mem::size_of::<P>();
    (slots * size).next_multiple_of(8) + 12 * 8 + (width * size).next_multiple_of(8)
}

/// Shared memory a force kernel needs on top of its own to form the sums of
/// a block of `threads` across its warps: twelve doubles a thread (see
/// `block_sums` in the kernels).
pub(crate) fn block_sums_shared(threads: u32) -> usize {
    12 * 8 * threads as usize
}

/// Warps per block of the se_e3 descriptor kernel, which share the rows of
/// an atom: as many as have their sums (`width` each) in shared memory, at
/// most eight.
pub(crate) fn descriptor_warps<P>(width: usize) -> usize {
    (SHARED_LIMIT / (width * std::mem::size_of::<P>()).max(1)).clamp(1, 8)
}

/// How a block of the se_e3 descriptor kernel that reads the tables goes
/// through the rows of its atom: whether with a thread per row, the threads
/// of a block, and the bytes of its sums in shared memory.
///
/// Float32 has a warp per row, as many warps as have their sums in shared
/// memory, thirty-two at most (0.14 against 0.23 ms with eight for 192 atoms
/// of 3,600 rows each). Float64 has a thread per row, as many warps of them
/// as have a sum for every thread in shared memory, eight at most; a net too
/// wide for one warp of such sums has a warp per row there too. Why the two
/// differ is with the kernels.
fn table_descriptor_threads<P>(width: usize) -> (bool, u32, usize) {
    let sums = (width * std::mem::size_of::<P>()).max(1);
    let by_thread = SHARED_LIMIT / (32 * sums);
    if std::mem::size_of::<P>() == 8 && by_thread > 0 {
        let warps = by_thread.min(TABLE_ROW_WARPS);
        (true, 32 * warps as u32, 32 * warps * sums)
    } else {
        let warps = (SHARED_LIMIT / sums).clamp(1, TABLE_DESCRIPTOR_WARPS);
        (false, 32 * warps as u32, warps * sums)
    }
}

/// Shared memory of the se_e3 pair kernel per block: the directions of the
/// atom's neighbors in double, their slots and the first of each type.
pub(crate) fn pair_shared(slots: usize, ntypes: usize) -> usize {
    3 * slots * 8 + (4 * (slots + ntypes + 1)).next_multiple_of(8)
}

/// Milliseconds for each stage of an evaluation that has device calls
/// (summed over chunks).
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Stages {
    pub neighbors: f64,
    pub environment: f64,
    pub embedding: f64,
    pub descriptor: f64,
    pub fitting: f64,
    pub forces: f64,
    pub download: f64,
}

impl Stages {
    const ALL: [Stage; 7] = [
        |s| &mut s.neighbors,
        |s| &mut s.environment,
        |s| &mut s.embedding,
        |s| &mut s.descriptor,
        |s| &mut s.fitting,
        |s| &mut s.forces,
        |s| &mut s.download,
    ];

    /// The stages together.
    pub fn sum(&self) -> f64 {
        self.neighbors
            + self.environment
            + self.embedding
            + self.descriptor
            + self.fitting
            + self.forces
            + self.download
    }

    fn add(&mut self, other: &Self) {
        let mut other = other.clone();
        for stage in Self::ALL {
            *stage(self) += *stage(&mut other);
        }
    }

    fn scale(&mut self, factor: f64) {
        for stage in Self::ALL {
            *stage(self) *= factor;
        }
    }
}

/// One of the numbers of [`Stages`].
type Stage = fn(&mut Stages) -> &mut f64;

/// Milliseconds of the last evaluation, taken only when profiling is on.
///
/// An evaluation is one chain of device calls that the host issues without
/// waiting and one copy back that waits for all of it, so the host's clock
/// alone says nothing of a stage: the host is done issuing long before the
/// device is done. The stages are therefore timed twice. `issue` is the
/// host's time to issue the calls of a stage and the mark after them (for
/// the first stage with the mark where the device calls begin, for
/// `download` with its wait for the device). `device` is the device's time
/// between marks in its stream, one after the calls of each stage
/// ([`omnimd_gpu::Mark`]). Nothing is waited for in between, so the
/// evaluation that is timed runs as it does otherwise but for the marks,
/// each some 5 microseconds of the host and 1.5 of the device.
///
/// For the shortest passes the marks tell in the times. An evaluation of 88
/// atoms in float32 took 180 microseconds with them where it takes 151, the
/// host issuing for twice as long, and its stages came out up to 5
/// microseconds longer than with marks at the two ends of one stage alone
/// (the fitting nets 67 against 62). For passes of 200 microseconds and more
/// the two ways agreed within a microsecond a stage
/// (`bench/results/SUMMARY.md`, 2026-11-19).
///
/// A stage's `device` time is the device's work on the stage and whatever
/// time it had none of it to do yet. The device can only have waited
/// between getting to the stage (the running sum of `device` up to the
/// stage before) and the host's last call of the stage (the running sum of
/// `issue` through the stage): where the first is the later, the time is
/// all the device's own, and otherwise their difference is the most that it
/// waited. (The sums of a pass of several chunks do not tell: the chunks
/// come one after the other, each with its own stages.) The device waits at
/// the start: the time of the first stage runs from the first device call
/// of the pass, and the device has nothing to do while the host packs the
/// input of the neighbor search, so `device.neighbors` holds that packing.
///
/// Both sets cover the same stretch, from the first device call to the
/// results on the host; `host_prepare` and `host_finish` are the host's work
/// before and after it, and `wall` is all of it. That is, when the device
/// gets to the first mark as the host issues it: now and then it gets there
/// late, and `device` is then short of `issue` by the delay. Of 200
/// evaluations of 12 atoms in float32 with tables, one each in a process of
/// its own, 8 had `device` below half of `issue`, down to a quarter
/// (2026-11-21).
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct StageTimes {
    pub wall: f64,
    pub host_prepare: f64,
    pub issue: Stages,
    pub device: Stages,
    pub host_finish: f64,
    pub chunks: usize,
}

impl StageTimes {
    /// Adds the times of another evaluation.
    pub fn add(&mut self, other: &Self) {
        self.wall += other.wall;
        self.host_prepare += other.host_prepare;
        self.issue.add(&other.issue);
        self.device.add(&other.device);
        self.host_finish += other.host_finish;
        self.chunks += other.chunks;
    }

    /// The mean over `evaluations` added times.
    pub fn mean(&self, evaluations: usize) -> Self {
        let n = 1.0 / evaluations.max(1) as f64;
        let mut mean = self.clone();
        mean.wall *= n;
        mean.host_prepare *= n;
        mean.issue.scale(n);
        mean.device.scale(n);
        mean.host_finish *= n;
        mean.chunks = self.chunks / evaluations.max(1);
        mean
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
    /// All fitting nets in grouped cuBLAS products, layer by layer: best
    /// for many rows. None when the library or the nets do not allow it.
    grouped_fitting: Option<GroupedNetwork<P>>,
    /// Whether `grouped_fitting` takes the place of cuBLAS a net at a time:
    /// always, never, or None by the rule of `GROUPED_FITTING_MIN_NETS`.
    grouped_fitting_on: Option<bool>,
    /// Chunks of at most this many fitting rows (atoms times models) use
    /// `fused_fitting`; None for the default rule (see
    /// `FUSED_FITTING_ROWS_F32`).
    fused_fitting_max_rows: Option<usize>,
    /// hybrid: writes the table of slots of a descriptor that does not read
    /// the neighbor list as it is.
    narrow: Option<CudaFunction>,
    chunk_atoms: usize,
    /// se_e3: rows of the embedding nets (pairs of neighbors) per chunk.
    chunk_pairs: usize,
    /// se_e3: frames of at most this many atoms can have their rows laid
    /// out on the device, and the passes that had them laid out on the
    /// host instead.
    layout_atoms: usize,
    count_downloads: usize,
    /// Where set, passes of at most this many centers give the force kernel
    /// of a two-body descriptor its wide blocks and the others the narrow
    /// ones (None: [`force_way`] chooses, or [`WIDE_FORCE_CENTERS_F32`] or
    /// `_F64` where the driver does not say), and the threads of a block of
    /// that kernel at its last launch (zero before the first) and whether it
    /// formed the sums of its blocks across their warps then.
    wide_force_centers: Option<usize>,
    force_threads: u32,
    force_sums: bool,
    block_sums_centers: Option<usize>,
    block_sums_shared: bool,
    /// With tables, whether the kernels of se_e2_a look the embedding values
    /// up themselves: always, never, or None by the precision (see
    /// [`Self::inline_tables`]), and whether they did in the last pass.
    inline_tables: Option<bool>,
    tables_looked_up: bool,
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
    /// se_e3: the width the directions of a neighbor are divided by, per
    /// center type (of every model), for the range of θ.
    direction_width: Vec<f64>,
    /// The embedding nets as tables in pieces of uneven length, when
    /// compressed: of se_e2_a and se_e2_r over the radial entry of a slot
    /// (in the order of the nets), of se_e3 over θ.
    tables: Option<CellTable<P>>,
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
    /// se_e2_a and se_e2_r: the blocks of the force kernel one
    /// multiprocessor keeps at once in each way of [`FORCE_WAYS`], for the
    /// kernel that reads stored values and for the one that looks the tables
    /// up (see [`resident_force_blocks`]). None for se_e3, for a kernel the
    /// descriptor does not have, and where the driver did not say.
    force_residence: [Option<[u32; 4]>; 2],
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

/// Stage clock: nothing unless profiling, so that an evaluation issues no
/// marks. It never waits for the device before the results are back.
struct Clock<'a> {
    device: &'a Device,
    times: Option<StageTimes>,
    start: Instant,
    last: Instant,
    /// A mark after the calls of each stage; one without a stage where the
    /// device calls of a pass begin.
    marks: Vec<(Option<Stage>, Mark)>,
}

impl<'a> Clock<'a> {
    fn new(device: &'a Device, profiling: bool) -> Self {
        let now = Instant::now();
        Self {
            device,
            times: profiling.then(StageTimes::default),
            start: now,
            last: now,
            marks: Vec::new(),
        }
    }

    /// The host's milliseconds since it was last asked.
    fn host_ms(&mut self) -> f64 {
        let now = Instant::now();
        let ms = (now - self.last).as_secs_f64() * 1e3;
        self.last = now;
        ms
    }

    /// The device calls of a pass begin: the time up to here was the host's
    /// alone.
    fn begin(&mut self) -> Result<()> {
        if self.times.is_some() {
            let ms = self.host_ms();
            self.marks.push((None, self.device.mark()?));
            if let Some(times) = &mut self.times {
                times.host_prepare += ms;
            }
        }
        Ok(())
    }

    /// The calls of `stage` are issued.
    fn lap(&mut self, stage: Stage) -> Result<()> {
        if self.times.is_some() {
            self.marks.push((Some(stage), self.device.mark()?));
            let ms = self.host_ms();
            if let Some(times) = &mut self.times {
                *stage(&mut times.issue) += ms;
            }
        }
        Ok(())
    }

    /// The results are on the host: the device's time from each mark to the
    /// next goes to the stage of the later one, and the rest of the host's
    /// time was its own.
    fn finish(mut self) -> Result<Option<StageTimes>> {
        let Some(mut times) = self.times.take() else {
            return Ok(None);
        };
        times.host_finish = self.host_ms();
        times.wall = (self.last - self.start).as_secs_f64() * 1e3;
        for pair in self.marks.windows(2) {
            if let Some(stage) = pair[1].0 {
                *stage(&mut times.device) += pair[0].1.ms_until(&pair[1].1)?;
            }
        }
        Ok(Some(times))
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
        // Of the force kernel, without the room for the sums of a block
        // across its warps: a pass has that where it fits.
        let backward_shared = match kind {
            DescriptorKind::SeE2A => {
                angular_force_shared::<P>(slots, width, std::mem::size_of::<P>())
            }
            DescriptorKind::SeE2R => radial_force_shared::<P>(slots, width),
            DescriptorKind::SeE3 => pair_force_shared(slots, ntypes),
        };
        // se_e3 has a kernel before the force kernel, with dE/dD of the atom.
        let before = match kind {
            DescriptorKind::SeE3 => pair_dtheta_shared::<P>(width),
            _ => 0,
        };
        if backward_shared.max(before) > SHARED_LIMIT {
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
                .function(source(), &format!("{name}_{}", P::SUFFIX))
        };
        let three_body = kind == DescriptorKind::SeE3;
        // The environment kernel serves se_e2_a and se_e2_r; se_e3 has its
        // rows from a kernel of its own.
        let kernels = Kernels {
            env: function("se_e2_a_env")?,
            descriptor: function(&format!("{}_descriptor", kind.name()))?,
            backward: function(&format!("{}_backward", kind.name()))?,
            count: match three_body {
                true => Some(device.kernels().function(source(), "se_e3_count")?),
                false => None,
            },
            layout: match three_body {
                true => Some(device.kernels().function(source(), "se_e3_layout")?),
                false => None,
            },
            pairs: match three_body {
                true => Some(function("se_e3_pairs")?),
                false => None,
            },
            dtheta: match three_body {
                true => Some(function("se_e3_dtheta")?),
                false => None,
            },
            table_descriptor: match three_body {
                true => Some(function("se_e3_table_descriptor")?),
                false => None,
            },
            table_descriptor_rows: match three_body {
                true => Some(function("se_e3_table_descriptor_rows")?),
                false => None,
            },
            table_dtheta: match three_body {
                true => Some(function("se_e3_table_dtheta")?),
                false => None,
            },
            inline_descriptor: match kind == DescriptorKind::SeE2A {
                true => Some(function("se_e2_a_table_descriptor")?),
                false => None,
            },
            inline_backward: match kind == DescriptorKind::SeE2A {
                true => Some(function("se_e2_a_table_backward")?),
                false => None,
            },
        };
        let force_residence = match kind {
            DescriptorKind::SeE3 => [None, None],
            DescriptorKind::SeE2A | DescriptorKind::SeE2R => [
                resident_force_blocks(&kernels.backward, backward_shared),
                kernels.inline_backward.as_ref().and_then(|function| {
                    resident_force_blocks(
                        function,
                        backward_shared + angular_table_shared::<P>(slots),
                    )
                }),
            ],
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
            direction_width: normalization().map(|n| n.std[1]).collect(),
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
            force_residence,
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

    /// Whether the descriptor and force kernels of this descriptor look the
    /// embedding values up in its tables themselves: an se_e2_a descriptor
    /// whose tables are in use, when `on` (see
    /// [`GpuPotential::inline_tables`]) and its blocks have room for what the
    /// lookup keeps in shared memory.
    fn looks_up(&self, compressed: bool, on: bool, ntypes: usize) -> bool {
        on && compressed
            && self.tables.is_some()
            && self.kind == DescriptorKind::SeE2A
            && self.backward_shared + angular_table_shared::<P>(self.slots) <= SHARED_LIMIT
            && angular_descriptor_shared::<P>(self.slots, self.width, ntypes, true) <= SHARED_LIMIT
    }

    /// The way of the force kernel of this two-body descriptor in a pass of
    /// `blocks` blocks, launched as the kernel that looks the tables up
    /// (`lookup`) or the other, within what the overrides set (`wide`,
    /// `sums`): by [`force_way`] with what the device and its
    /// `multiprocessors` keep at once; by [`WIDE_FORCE_CENTERS_F32`] (or
    /// `_F64`) and [`BLOCK_SUMS_CENTERS`] where the driver did not say.
    fn force_way(
        &self,
        lookup: bool,
        blocks: usize,
        (wide, sums): (Option<bool>, Option<bool>),
        multiprocessors: usize,
    ) -> ForceBlocks {
        let radial = self.kind == DescriptorKind::SeE2R;
        if let Some(resident) = self.force_residence[usize::from(lookup)] {
            let capacity = resident.map(|blocks| blocks as usize * multiprocessors);
            return force_way(blocks, capacity, wide, sums, radial);
        }
        let limit = match std::mem::size_of::<P>() {
            4 => WIDE_FORCE_CENTERS_F32,
            _ => WIDE_FORCE_CENTERS_F64,
        };
        let threads = match wide.unwrap_or(blocks <= limit) {
            true => WIDE_BLOCK,
            false => BLOCK,
        };
        let own = self.backward_shared
            + match lookup {
                true => angular_table_shared::<P>(self.slots),
                false => 0,
            };
        ForceBlocks {
            threads,
            sums: sums.unwrap_or(radial || blocks <= BLOCK_SUMS_CENTERS)
                && own + block_sums_shared(threads) <= SHARED_LIMIT,
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

    /// The radial entry of a neighbor at the distance `r`, before it is
    /// normalized: s(r) = switch(r)/r, which falls with r.
    fn radial_entry(&self, r: f64) -> f64 {
        let t = ((r - self.rcut_smth) / (self.rcut - self.rcut_smth)).max(0.0);
        (t * t * t * (-6.0 * t * t + 15.0 * t - 10.0) + 1.0) / r
    }

    /// The embedding nets of an se_e2_a or se_e2_r descriptor as tables: see
    /// [`GpuPotential::compress`]. A table counts from the entry of a
    /// neighbor at the cutoff, the lowest its net is given, and has pieces
    /// above that only: the place of a neighbor in it is s(r) over the width
    /// of the radial entry of the central type, which the environment kernel
    /// writes for the rows of a descriptor that runs from its tables.
    fn tabulate(
        &self,
        device: &Device,
        ntypes: usize,
        min_distance: f64,
        step: f64,
    ) -> Result<CellTable<P>> {
        let options = table_options::<P>(step);
        let nearest = self.radial_entry(min_distance / TAIL_REACH);
        let models = self.embedding_host.len() / ntypes.max(1);
        // Every net with the origin and the reach of its central type.
        let mut nets = Vec::new();
        for (center, (row, &(mean, std))) in self
            .embedding_host
            .iter()
            .zip(&self.radial_normalization)
            .enumerate()
        {
            let reach = nearest / std;
            let span = TableSpan {
                top: table_top(reach)?,
                origin: -mean / std,
                below: false,
            };
            let of = (center / ntypes, center % ntypes);
            nets.extend(
                row.iter()
                    .enumerate()
                    .map(|(neighbor, network)| (network, span, reach, of, neighbor)),
            );
        }
        // A table is some thousands of evaluations of its net with two
        // derivatives, and the nets do not depend on each other: all cores.
        let segments = in_parallel(&nets, |&(network, span, reach, (model, center), neighbor)| {
            let table = NetTable::build(network, span, options).map_err(|e| {
                Error(format!(
                    "the embedding net of {}{} for a center of type {center} and neighbors of type {neighbor}: {e}",
                    self.kind.name(),
                    of_model(model, models)
                ))
            })?;
            Ok(CellSegments {
                top: table.top,
                reach: [reach, 0.0],
                lowest: table.lowest,
                cells: table.cells,
                coefficients: table.coefficients,
            })
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
        let of: Vec<usize> = (0..segments.len()).collect();
        CellTable::new_set(device, self.width, options.key_bits, &segments, &of)
    }

    /// The embedding nets of an se_e3 descriptor as tables over θ: see
    /// [`GpuPotential::compress`]. The nets go by the pair of neighbor
    /// types, in either order one net, which is tabulated once.
    fn tabulate_pairs(
        &self,
        device: &Device,
        ntypes: usize,
        min_distance: f64,
        step: f64,
    ) -> Result<CellTable<P>> {
        let options = table_options::<P>(step);
        // The longest direction entry a neighbor has at the distance the
        // tables reach to.
        let nearest = self.radial_entry(min_distance / TAIL_REACH);
        // Every net once, with the range of θ = a_j·a_k its model has: the
        // directions are divided by the width of the center type, and a net
        // serves the centers of every type.
        let models = self.embedding_host.len() / ntypes.max(1);
        let mut nets = Vec::new();
        let mut of = vec![0; models * ntypes * ntypes];
        for model in 0..models {
            let narrowest = self.direction_width[model * ntypes..(model + 1) * ntypes]
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let reach = (nearest / narrowest).powi(2);
            let span = TableSpan {
                top: table_top(reach)?,
                origin: 0.0,
                below: true,
            };
            for tj in 0..ntypes {
                for tk in tj..ntypes {
                    let index = nets.len();
                    of[(model * ntypes + tj) * ntypes + tk] = index;
                    of[(model * ntypes + tk) * ntypes + tj] = index;
                    let network = &self.embedding_host[model * ntypes + tj][tk];
                    nets.push((network, span, reach, model, tj, tk));
                }
            }
        }
        let segments = in_parallel(&nets, |&(network, span, reach, model, tj, tk)| {
            let table = NetTable::build(network, span, options).map_err(|e| {
                Error(format!(
                    "the embedding net of se_e3{} for neighbors of types {tj} and {tk}: {e}",
                    of_model(model, models)
                ))
            })?;
            Ok(CellSegments {
                top: table.top,
                reach: [reach; 2],
                lowest: table.lowest,
                cells: table.cells,
                coefficients: table.coefficients,
            })
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
        CellTable::new_set(device, self.width, options.key_bits, &segments, &of)
    }
}

/// Which model of several a net belongs to, for the message that refuses
/// its table: nothing for a potential of one model.
fn of_model(model: usize, models: usize) -> String {
    match models {
        0 | 1 => String::new(),
        _ => format!(" of model {model}"),
    }
}

/// How the tables of a descriptor are made for the step asked for (see
/// [`GpuPotential::compress`]), in the precision of the networks.
fn table_options<P>(step: f64) -> TableOptions {
    let single = std::mem::size_of::<P>() == 4;
    TableOptions {
        tolerance: (TABLE_TOLERANCE * (step / 0.01).powi(6)).max(match single {
            true => TABLE_FLOOR_F32,
            false => TABLE_FLOOR_F64,
        }),
        key_bits: TABLE_KEY_BITS,
        resolution: match single {
            true => TABLE_RESOLUTION_F32,
            false => TABLE_RESOLUTION_F64,
        },
    }
}

/// The power of two from which on no input of a table reaches: its pieces
/// go that far.
fn table_top(reach: f64) -> Result<i32> {
    let top = reach.log2().ceil();
    match top.is_finite() && (-60.0..=100.0).contains(&top) {
        true => Ok(top as i32),
        false => Err(Error(format!(
            "the inputs of the embedding nets reach {reach:e} at the distance the tables are asked to cover: no table has that range"
        ))),
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
        let grouped_fitting = GroupedNetwork::new_set(&device, &fitting_layers).ok();
        let fitting = fitting_host
            .iter()
            .map(|n| dense_network(&device, n, features))
            .collect::<Result<Vec<_>>>()?;
        let narrow = match parts.iter().any(|part| part.view.is_some()) {
            true => Some(device.kernels().function(source(), "hybrid_slots")?),
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
            grouped_fitting,
            grouped_fitting_on: None,
            fused_fitting_max_rows: None,
            narrow,
            chunk_atoms,
            chunk_pairs,
            layout_atoms: LAYOUT_ATOMS,
            count_downloads: 0,
            wide_force_centers: None,
            force_threads: 0,
            force_sums: false,
            block_sums_centers: None,
            block_sums_shared: false,
            inline_tables: None,
            tables_looked_up: false,
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

    /// se_e3: the most atoms of a pass whose rows are laid out on the
    /// device. The host lays the rows out from the neighbors every atom
    /// holds, which it has to bring back from the device in the middle of
    /// an evaluation: 17 to 30 microseconds where it was measured, for the
    /// wait and the calls, which a frame of a hundred atoms feels. It
    /// needs the rows for three things: to launch the embedding nets over
    /// them, to end a chunk where they would be too many, and to size the
    /// buffers that hold them. With tables nothing is launched over the
    /// rows. So a pass that reads the tables of every se_e3 descriptor, has
    /// no more atoms than this and is one chunk even if every neighbor slot
    /// of every atom is full gets its layout from a kernel instead, and
    /// nothing comes back from the device before the results. Its buffers
    /// are made for the rows of full slots, which is some ten times the
    /// rows a frame of the HAP systems has. [`Self::count_downloads`]
    /// counts the other passes.
    pub fn layout_atoms(&self) -> usize {
        self.layout_atoms
    }

    /// Override [`Self::layout_atoms`]: at most [`LAYOUT_ATOMS`], the
    /// threads of the block that makes the layout; zero has every layout
    /// made on the host.
    pub fn set_layout_atoms(&mut self, atoms: usize) {
        self.layout_atoms = atoms.min(LAYOUT_ATOMS);
    }

    /// Passes over the device that brought the neighbor counts of the se_e3
    /// descriptors back to lay their rows out on the host (see
    /// [`Self::layout_atoms`]). None for a model without such a descriptor.
    pub fn count_downloads(&self) -> usize {
        self.count_downloads
    }

    /// The most centers of a pass (the atoms of a chunk times the models)
    /// whose force kernels of se_e2_a and se_e2_r take the wide blocks, when
    /// one number was set; the others take the narrow ones. Where the driver
    /// says what the device keeps at once, the sums across the warps are
    /// still chosen by it among the ways of that width, so the number can
    /// change them too (see [`force_way`]). `None` is how a potential
    /// starts: the width goes with the sums by the blocks the device keeps
    /// at once (see [`FORCE_WAYS`]), or by [`WIDE_FORCE_CENTERS_F32`] or
    /// `_F64` where the driver does not say. The forces are the same to
    /// rounding either way: the block of a center sums what its neighbors
    /// give in another order.
    pub fn wide_force_centers(&self) -> Option<usize> {
        self.wide_force_centers
    }

    /// Override [`Self::wide_force_centers`]; `Some(0)` keeps every pass to
    /// the narrow blocks, `None` restores the default.
    pub fn set_wide_force_centers(&mut self, centers: Option<usize>) {
        self.wide_force_centers = centers;
    }

    /// Threads of a block of the force kernel of a two-body descriptor at
    /// its last launch: which of the two sizes the last pass (its last
    /// chunk) took. Zero before the first launch and for a model without
    /// such a descriptor.
    pub fn force_threads(&self) -> u32 {
        self.force_threads
    }

    /// Whether the force kernel of a two-body descriptor formed the sums of
    /// its blocks across their warps at its last launch: with
    /// [`Self::force_threads`], the way of [`FORCE_WAYS`] it took. False
    /// before the first launch and for a model without such a descriptor.
    /// [`Self::block_sums_shared`] tells the same of the force kernel
    /// launched last, which in a hybrid may be that of se_e3.
    pub fn force_sums(&self) -> bool {
        self.force_sums
    }

    /// The most centers of a pass (the atoms of a chunk times the models)
    /// whose force kernels form the sums of a block across its warps in
    /// shared memory, when one number was set for every descriptor. Where
    /// the driver says what the device keeps at once, the width of the
    /// two-body kernels is still chosen by it among the ways left, so the
    /// number can change the width too (see [`force_way`]). `None` is how a
    /// potential starts: the kernels of se_e2_a and se_e2_r by the blocks
    /// the device keeps at once, together with the width of the blocks (see
    /// [`FORCE_WAYS`]; where the driver does not say, se_e2_a up to
    /// [`BLOCK_SUMS_CENTERS`] and se_e2_r in any pass), the kernel of se_e3
    /// in any pass. A kernel whose block has no room for the sums within the
    /// limit of shared memory goes without at any number. The forces and the virial
    /// are the same to rounding either way: the sums of a block are formed
    /// in another order.
    pub fn block_sums_centers(&self) -> Option<usize> {
        self.block_sums_centers
    }

    /// Override [`Self::block_sums_centers`] for every descriptor; zero has
    /// every warp reduce its own sums, as all did before 2026-11-19.
    pub fn set_block_sums_centers(&mut self, centers: Option<usize>) {
        self.block_sums_centers = centers;
    }

    /// Drop what the driver said the device keeps of the force kernels at
    /// once, as if it had not said: the passes then go by
    /// [`WIDE_FORCE_CENTERS_F32`] (or `_F64`) and [`BLOCK_SUMS_CENTERS`].
    /// For tests.
    #[doc(hidden)]
    pub fn forget_force_residence(&mut self) {
        for part in &mut self.parts {
            part.force_residence = [None, None];
        }
    }

    /// Whether the force kernel launched last formed the sums of its blocks
    /// across their warps in shared memory. False before the first launch.
    pub fn block_sums_shared(&self) -> bool {
        self.block_sums_shared
    }

    /// The blocks of the force kernel of every descriptor (in the order of
    /// a hybrid's list) that the device keeps at once in each way of
    /// [`FORCE_WAYS`]: what one multiprocessor keeps times the
    /// multiprocessors, zero for a way without room in shared memory. For
    /// the kernel a pass launches, which with tables in use (`compressed`)
    /// is the one that looks them up where [`Self::looks_up_tables`] says
    /// so. None for se_e3 and where the driver did not say.
    pub fn force_capacity(&self, compressed: bool) -> Vec<Option<[usize; 4]>> {
        let (on, ntypes) = (self.inline_tables(), self.sel.len());
        let multiprocessors = self.device.multiprocessors() as usize;
        self.parts
            .iter()
            .map(|part| {
                let kernel = usize::from(part.looks_up(compressed, on, ntypes));
                part.force_residence[kernel]
                    .map(|blocks| blocks.map(|b| b as usize * multiprocessors))
            })
            .collect()
    }

    /// Model compression (DeePMD `dp compress`): replaces every embedding
    /// net by a piecewise quintic table.
    ///
    /// A trained net turns within a hundredth of a unit of its input in
    /// some places and not at all in most, so a table has pieces of uneven
    /// length and no step of its own: a piece is halved until its quintic
    /// is within a tolerance of the net. `step` names that tolerance by the
    /// table of one step it stands for: 1e-13 of the largest value of the
    /// net at the step 0.01, which is where a table with a knot every 0.01
    /// follows a smooth net, and (step / 0.01)⁶ times that at another
    /// step, which is how the error of such a table goes with its step.
    /// The slope is within a hundred times that fraction of the largest
    /// slope. A float32 potential has 1.5e-8 at best, below what it rounds
    /// at. A net that turns faster than the precision of its input can
    /// follow is refused, with the place.
    ///
    /// The nets of se_e2_a and se_e2_r take the normalized radial entry of
    /// a neighbor. Their tables count from the entry of a neighbor at the
    /// cutoff and cover neighbors down to a fifth of `min_distance`. Empty
    /// neighbor slots do not go through the tables: their row is the
    /// networks' own. The nets of se_e3 take θ, the product of the
    /// normalized directions of two neighbors, which spans thousands of
    /// units where the radial entry spans tens; their tables cover θ of
    /// either sign for neighbors down to the same distance. With them the
    /// descriptor keeps no embedding rows: its kernels read the tables
    /// where they would read the rows, and a chunk holds as many times more
    /// pairs as a pair takes less memory.
    ///
    /// A table has its tolerance all the way: below `min_distance` it is no
    /// coarser (DeePMD's tables take ten times the step there). A frame
    /// with an input beyond a table (for se_e2_a and se_e2_r, a pair closer
    /// than a fifth of `min_distance`; for se_e3, a θ larger than such a
    /// pair has around a center of the type with the narrowest directions)
    /// is evaluated with the exact networks instead (see
    /// [`Self::table_fallbacks`]), so results never depend on the range.
    ///
    /// Every descriptor of a hybrid gets tables, each over the range of its
    /// own cutoff and statistics. When the nets of one cannot be tabulated
    /// the potential is left as it was, with the nets of all of them: a
    /// hybrid has tables for every descriptor or for none.
    pub fn compress(&mut self, min_distance: f64, step: f64) -> Result<()> {
        let ntypes = self.sel.len();
        // The shortest cutoff of the descriptors.
        let rcut = self
            .parts
            .iter()
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
            tables.push(match part.kind {
                DescriptorKind::SeE3 => {
                    part.tabulate_pairs(&self.device, ntypes, min_distance, step)?
                }
                _ => part.tabulate(&self.device, ntypes, min_distance, step)?,
            });
        }
        for (part, table) in self.parts.iter_mut().zip(tables) {
            part.tables = Some(table);
        }
        Ok(())
    }

    /// Whether embedding nets are tabulated (see [`Self::compress`]).
    pub fn is_compressed(&self) -> bool {
        self.parts.iter().any(|part| part.tables.is_some())
    }

    /// Device bytes of the embedding tables, and the pieces they have
    /// between them (their number follows from the nets and the tolerance,
    /// not from a step).
    pub fn table_size(&self) -> (usize, usize) {
        let tables = || self.parts.iter().filter_map(|part| part.tables.as_ref());
        (
            tables().map(CellTable::bytes).sum(),
            tables().map(CellTable::pieces).sum(),
        )
    }

    /// Chunks of at most this many fitting rows (atoms times models)
    /// evaluate the fitting nets in one fused kernel; larger chunks use
    /// layer-by-layer cuBLAS. Zero disables the fused kernel, `usize::MAX`
    /// takes it always. The default depends on the precision and on whether
    /// the grouped products are available (see `set_grouped_fitting`).
    pub fn set_fused_fitting_max_rows(&mut self, rows: usize) {
        self.fused_fitting_max_rows = Some(rows);
    }

    /// Has the layer-by-layer path take the grouped products always (on) or
    /// never (off: cuBLAS a net at a time) instead of by the default rule,
    /// and returns whether they are taken: they need cuBLAS 12.5 and fitting
    /// nets that end in one linear output, and take chunks of at most
    /// `GROUPED_MAX_GROUPS` nets. Both give the same results to rounding.
    pub fn set_grouped_fitting(&mut self, on: bool) -> bool {
        self.grouped_fitting_on = Some(on);
        on && self.grouped_fitting.is_some()
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

    /// Whether, with tables, the descriptor and force kernels of se_e2_a
    /// look the embedding values up in them themselves instead of reading
    /// what a kernel of the tables stored for every row of a chunk: by
    /// default in float32 and not in float64. Either gives the same
    /// numbers, since both evaluate the quintics alike. A descriptor whose
    /// blocks have no room in shared memory for what the lookup keeps reads
    /// the stored values at any setting.
    pub fn inline_tables(&self) -> bool {
        self.inline_tables.unwrap_or(std::mem::size_of::<P>() == 4)
    }

    /// Whether a pass with the tables has the kernels of an se_e2_a
    /// descriptor look the embedding values up in them: with tables, by the
    /// setting of [`Self::inline_tables`] and where the blocks have room.
    pub fn looks_up_tables(&self) -> bool {
        let (on, ntypes) = (self.inline_tables(), self.sel.len());
        self.parts
            .iter()
            .any(|part| part.looks_up(true, on, ntypes))
    }

    /// Override [`Self::inline_tables`]; `None` restores the default of the
    /// precision.
    pub fn set_inline_tables(&mut self, on: Option<bool>) {
        self.inline_tables = on;
    }

    /// Whether the kernels of an se_e2_a descriptor looked the embedding
    /// values up in its tables in the last pass over the device (see
    /// [`Self::inline_tables`]): false before the first and after a pass
    /// with the exact networks, as when a frame left the tables.
    pub fn tables_looked_up(&self) -> bool {
        self.tables_looked_up
    }

    /// Turn per-stage timing on or off. On, an evaluation issues a mark in
    /// the device's stream after each stage and reads their times when the
    /// results are back; it waits for nothing it would not wait for anyway.
    pub fn set_profiling(&mut self, on: bool) {
        self.profile = on.then(StageTimes::default);
    }

    /// Stage times of the last evaluation, when profiling (see
    /// [`StageTimes`]). Those of an evaluation that failed are empty.
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
        let profiling = self.profile.is_some();
        let mut clock = Clock::new(&device, profiling);
        let mut result = self.run(&device, frames, members, &mut clock, self.is_compressed());
        if matches!(result, Ok(None)) {
            if frames.len() > 1 {
                // Each frame is an evaluation with times of its own: those
                // of the last are kept.
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
        self.profile = match &result {
            Ok(_) => clock.finish()?,
            Err(_) => profiling.then(StageTimes::default),
        };
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
        let multiprocessors = device.multiprocessors() as usize;
        // se_e2_a with tables: whether its kernels look the embedding values
        // up themselves (see `inline_tables`).
        let inline_on = self.inline_tables();
        self.tables_looked_up = compressed && self.looks_up_tables();
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
        clock.begin()?;

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
        // neighbors, laid out from the neighbors every atom holds.
        let three_body: Vec<usize> = (0..self.parts.len())
            .filter(|&index| self.parts[index].kind == DescriptorKind::SeE3)
            .collect();
        // The most rows of a chunk in each of these descriptors. With
        // tables a row keeps no embedding values, and as many times more
        // rows fit a chunk.
        let most_rows = {
            let fewer = self
                .parts
                .iter()
                .filter(|part| part.kind == DescriptorKind::SeE3)
                .map(|part| match part.tables.as_ref().filter(|_| compressed) {
                    Some(_) => bytes_per_pair::<P>(part.width) / bytes_per_pair::<P>(0),
                    None => 1,
                })
                .min()
                .unwrap_or(1);
            (self.chunk_pairs / members).max(1).saturating_mul(fewer)
        };
        // The layout is made on the device when the host has no need to see
        // it (see `layout_atoms`): every se_e3 descriptor reads its tables,
        // and the frame is one chunk whatever its atoms hold.
        let on_device = !three_body.is_empty()
            && compressed
            && natoms <= self.layout_atoms.min(chunk)
            && three_body.iter().all(|&index| {
                let part = &self.parts[index];
                part.tables.is_some() && natoms.saturating_mul(most_pairs(part.slots)) <= most_rows
            });
        let mut pair_rows: Vec<Option<PairRows>> = self.parts.iter().map(|_| None).collect();
        if on_device {
            for &index in &three_body {
                let part = &self.parts[index];
                let (slots_index, _, _) = part.slots_of(list)?;
                let function =
                    part.kernels.layout.as_ref().ok_or_else(|| {
                        Error("an se_e3 descriptor without its layout kernel".into())
                    })?;
                let shape = (natoms, part.slots, ntypes);
                pair_rows[index] = Some(PairRows {
                    start: lay_out_rows(device, function, slots_index, &part.sel_d, shape)?,
                    known: Laid::OnDevice {
                        most: most_pairs(part.slots),
                    },
                });
            }
        } else if !three_body.is_empty() {
            // The neighbors every atom holds, of all these descriptors in
            // one download, and the layouts the host makes of them.
            self.count_downloads += 1;
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
                    known: Laid::OnHost(host),
                });
            }
        }
        // A chunk ends where its atoms are too many, or its rows in one of
        // the se_e3 descriptors. (A layout made on the device is of a frame
        // that is one chunk.)
        let chunks = {
            let starts: Vec<&[i64]> = pair_rows
                .iter()
                .flatten()
                .filter_map(|rows| match &rows.known {
                    Laid::OnHost(start) => Some(start.as_slice()),
                    Laid::OnDevice { .. } => None,
                })
                .collect();
            match on_device {
                true => vec![(0, natoms)],
                false => pair_chunks(&starts, natoms, chunk, most_rows),
            }
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
            // se_e3 from its tables, and se_e2_a where its kernels look the
            // values up there: no embedding rows are kept. (The chunks stay
            // as large as the exact networks take them, which an evaluation
            // that leaves the tables falls back to.)
            let from_tables = compressed && part.tables.is_some();
            let kept = match (from_tables && part.kind == DescriptorKind::SeE3)
                || part.looks_up(compressed, inline_on, ntypes)
            {
                true => 1,
                false => all_rows * part.width,
            };
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
                    g: device.alloc_uninit::<P>(kept)?,
                    dg: device.alloc_uninit::<P>(kept)?,
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
                    // From its tables, the descriptor has the place of a
                    // neighbor in them for the input of its rows.
                    let unshifted_i = i32::from(compressed && part.tables.is_some());
                    launch
                        .arg(&mut work.input)
                        .arg(&mut work.network)
                        .arg(&unshifted_i);
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
                // The tables of se_e3 are read by its descriptor and force
                // kernels themselves, and those of se_e2_a where its kernels
                // look the values up (`looks_up`): nothing is evaluated here
                // for them.
                let pair_tables = part
                    .tables
                    .as_ref()
                    .filter(|_| compressed && part.kind == DescriptorKind::SeE3);
                let row_tables = part
                    .tables
                    .as_ref()
                    .filter(|_| part.looks_up(compressed, inline_on, ntypes));
                if chunk_rows > 0 && pair_tables.is_none() && row_tables.is_none() {
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
                // (a thread per row, threads of a block, bytes of its sums)
                let by_table = pair_tables.map(|_| table_descriptor_threads::<P>(width));
                let function = match by_table {
                    Some((by_thread, _, _)) => match by_thread {
                        true => &part.kernels.table_descriptor_rows,
                        false => &part.kernels.table_descriptor,
                    }
                    .as_ref()
                    .ok_or_else(|| {
                        Error("an se_e3 descriptor without the kernel of its tables".into())
                    })?,
                    None if row_tables.is_some() => {
                        part.kernels.inline_descriptor.as_ref().ok_or_else(|| {
                            Error("an se_e2_a descriptor without the kernel of its tables".into())
                        })?
                    }
                    None => &part.kernels.descriptor,
                };
                let mut launch = stream.launch_builder(function);
                let buffers = pair_tables.map(CellTable::buffers);
                let lookup = row_tables.map(CellTable::buffers);
                let config = match part.kind {
                    DescriptorKind::SeE3 => {
                        let (Some(pairs), Some((weight, _))) = (&work.pairs, &work.of_pairs) else {
                            return Err(Error("an se_e3 descriptor without its rows".into()));
                        };
                        if let Some(buffers) = &buffers {
                            tabulated = true;
                            launch
                                .arg(&work.input)
                                .arg(weight)
                                .arg(&work.network)
                                .arg(&pairs.start)
                                .arg(&chunk_rows_i)
                                .arg(&first_i)
                                .arg(&count_i)
                                .arg(&w_i)
                                .arg(&mut d_out)
                                .arg(&stride_i)
                                .arg(buffers.coefficients)
                                .arg(&buffers.key_bits)
                                .arg(buffers.shape)
                                .arg(buffers.range)
                                .arg(buffers.base)
                                .arg(buffers.scale)
                                .arg(&mut outside_table);
                        } else {
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
                        }
                        // The sums of every warp, or with tables as
                        // the kernel of the tables keeps them.
                        let (threads, shared) = match by_table {
                            Some((_, threads, shared)) => (threads, shared),
                            None => {
                                let warps = descriptor_warps::<P>(width);
                                (32 * warps as u32, warps * width * std::mem::size_of::<P>())
                            }
                        };
                        LaunchConfig {
                            grid_dim: (count as u32, deep, 1),
                            block_dim: (threads, 1, 1),
                            shared_mem_bytes: shared.next_multiple_of(8) as u32,
                        }
                    }
                    DescriptorKind::SeE2A => {
                        let (Some(env), Some(b)) = (&work.env, &mut work.b) else {
                            return Err(Error("an se_e2_a descriptor without its rows".into()));
                        };
                        launch.arg(env);
                        // The values of the rows, or what looks them up.
                        match &lookup {
                            Some(tables) => {
                                tabulated = true;
                                launch
                                    .arg(&work.input)
                                    .arg(&work.network)
                                    .arg(tables.coefficients)
                                    .arg(&tables.key_bits)
                                    .arg(tables.shape)
                                    .arg(tables.range)
                                    .arg(tables.base)
                                    .arg(tables.scale)
                                    .arg(&mut outside_table)
                            }
                            None => launch.arg(&work.g),
                        };
                        launch
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
                            shared_mem_bytes: angular_descriptor_shared::<P>(
                                slots,
                                width,
                                ntypes,
                                lookup.is_some(),
                            ) as u32,
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
            // in one fused kernel. Many atoms: grouped cuBLAS products for
            // all of them, or cuBLAS per model and center type (few nets,
            // or no grouped products), with the energies copied out so the
            // tape can go.
            let rows = members * count;
            let rule = FittingRule {
                double: std::mem::size_of::<P>() == 8,
                ntypes,
                fused: self.fused_fitting.is_some(),
                grouped: self.grouped_fitting.is_some(),
                fused_max_rows: self.fused_fitting_max_rows,
                grouped_on: self.grouped_fitting_on,
            };
            let path = rule.path(rows, count, members * ranges.len());
            // The atoms are sorted by type: runs of one fitting net each.
            let runs = (0..members).flat_map(|model| {
                ranges
                    .iter()
                    .map(move |&(t, lo, hi)| (model * ntypes + t, model * count + lo, hi - lo))
            });
            let mut energies = atomic.slice_mut(members * first..members * end);
            match (
                path,
                self.fused_fitting.as_mut(),
                self.grouped_fitting.as_mut(),
            ) {
                (FittingPath::Fused, Some(fused), _) => {
                    let x = descriptor.slice(0..rows * features);
                    let groups: Vec<(usize, usize, usize)> = runs.collect();
                    let mut dx = d_descriptor.slice_mut(0..rows * features);
                    fused.forward_gradient_grouped(
                        device,
                        &x,
                        &groups,
                        rows,
                        &mut energies,
                        &mut dx,
                    )?;
                }
                (FittingPath::Grouped, _, Some(grouped)) => {
                    let x = descriptor.slice(0..rows * features);
                    let groups: Vec<(usize, usize, usize)> = runs.collect();
                    let mut dx = d_descriptor.slice_mut(0..rows * features);
                    grouped.forward_gradient_grouped(
                        device,
                        &x,
                        &groups,
                        rows,
                        &mut energies,
                        &mut dx,
                    )?;
                }
                _ => {
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
                let add = i32::from(number > 0);
                let chunk_rows = match &work.pairs {
                    Some(pairs) => pairs.rows(first, end),
                    None => count * slots,
                };
                let chunk_rows_i = chunk_rows as i64;
                let dd = d_descriptor.slice(part.first_feature..);
                // se_e2_a where its kernels look the embedding values up in
                // the tables (`looks_up`): what the lookup keeps in shared
                // memory is added to the kernel's own.
                let lookup = part
                    .tables
                    .as_ref()
                    .filter(|_| part.looks_up(compressed, inline_on, ntypes))
                    .map(CellTable::buffers);
                let function = match lookup {
                    Some(_) => part.kernels.inline_backward.as_ref().ok_or_else(|| {
                        Error("an se_e2_a descriptor without the force kernel of its tables".into())
                    })?,
                    None => &part.kernels.backward,
                };
                let own_shared = part.backward_shared
                    + match lookup {
                        Some(_) => angular_table_shared::<P>(slots),
                        None => 0,
                    };
                let mut launch = stream.launch_builder(function);
                // The sums across the warps of the two-body kernels go with
                // the width of their blocks.
                let mut two_body_sums = None;
                let block_dim = match part.kind {
                    DescriptorKind::SeE3 => {
                        let (Some(pairs), Some((weight, dtheta))) =
                            (&work.pairs, &mut work.of_pairs)
                        else {
                            return Err(Error("an se_e3 descriptor without its rows".into()));
                        };
                        // dE/dθ of every pair first, by row: from the
                        // stored rows of the nets, or from their tables.
                        let pair_tables = part.tables.as_ref().filter(|_| compressed);
                        let function = match pair_tables {
                            Some(_) => &part.kernels.table_dtheta,
                            None => &part.kernels.dtheta,
                        }
                        .as_ref()
                        .ok_or_else(|| {
                            Error("an se_e3 descriptor without the kernel of its pairs".into())
                        })?;
                        let buffers = pair_tables.map(CellTable::buffers);
                        let mut before = stream.launch_builder(function);
                        if let Some(buffers) = &buffers {
                            before
                                .arg(&work.input)
                                .arg(&*weight)
                                .arg(&work.network)
                                .arg(&dd)
                                .arg(&pairs.start)
                                .arg(&chunk_rows_i)
                                .arg(&first_i)
                                .arg(&count_i)
                                .arg(&w_i)
                                .arg(&stride_i)
                                .arg(&mut *dtheta)
                                .arg(buffers.coefficients)
                                .arg(&buffers.key_bits)
                                .arg(buffers.shape)
                                .arg(buffers.range)
                                .arg(buffers.base)
                                .arg(buffers.scale);
                        } else {
                            before
                                .arg(&work.g)
                                .arg(&work.dg)
                                .arg(&work.input)
                                .arg(&*weight)
                                .arg(&dd)
                                .arg(&pairs.start)
                                .arg(&chunk_rows_i)
                                .arg(&first_i)
                                .arg(&count_i)
                                .arg(&w_i)
                                .arg(&stride_i)
                                .arg(&mut *dtheta);
                        }
                        let config = LaunchConfig {
                            grid_dim: (count as u32, deep, 1),
                            block_dim: (
                                match pair_tables {
                                    Some(_) => PAIR_TABLE_BLOCK,
                                    None => PAIR_BLOCK,
                                },
                                1,
                                1,
                            ),
                            shared_mem_bytes: pair_dtheta_shared::<P>(width) as u32,
                        };
                        unsafe { before.launch(config) }
                            .map(drop)
                            .map_err(launch_error("pair force"))?;
                        launch
                            .arg(&*dtheta)
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
                            .arg(&slots_i)
                            .arg(&ntypes_i)
                            .arg(&part.rcut_smth)
                            .arg(&part.inv_range)
                            .arg(&mut force)
                            .arg(&mut virial)
                            .arg(&natoms_i);
                        PAIR_FORCE_BLOCK
                    }
                    DescriptorKind::SeE2A | DescriptorKind::SeE2R => {
                        match (&work.env, &work.b, &lookup) {
                            (Some(env), Some(b), Some(tables)) => launch
                                .arg(env)
                                .arg(&work.input)
                                .arg(&work.network)
                                .arg(tables.coefficients)
                                .arg(&tables.key_bits)
                                .arg(tables.shape)
                                .arg(tables.range)
                                .arg(tables.base)
                                .arg(tables.scale)
                                .arg(b),
                            (Some(env), Some(b), None) => {
                                launch.arg(env).arg(&work.g).arg(&work.dg).arg(b)
                            }
                            _ => launch.arg(&work.dg),
                        };
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
                            .arg(&stride_i);
                        // A block per center and model, as wide and with the
                        // sums across its warps as the device keeps all of
                        // them at once (see `force_way`).
                        let blocks = members * count;
                        let set = (
                            self.wide_force_centers.map(|limit| blocks <= limit),
                            self.block_sums_centers.map(|limit| blocks <= limit),
                        );
                        let way = part.force_way(lookup.is_some(), blocks, set, multiprocessors);
                        self.force_threads = way.threads;
                        self.force_sums = way.sums;
                        two_body_sums = Some(way.sums);
                        way.threads
                    }
                };
                // The last argument of every force kernel: bit 0 has it add
                // to the virial, bit 1 says that its blocks have room for
                // their sums across the warps.
                let with_sums = own_shared + block_sums_shared(block_dim);
                self.block_sums_shared = match two_body_sums {
                    Some(sums) => sums,
                    None => {
                        self.block_sums_centers
                            .is_none_or(|limit| members * count <= limit)
                            && with_sums <= SHARED_LIMIT
                    }
                };
                let (shared, add_i) = match self.block_sums_shared {
                    true => (with_sums, add | 2),
                    false => (own_shared, add),
                };
                launch.arg(&add_i);
                let config = LaunchConfig {
                    grid_dim: (count as u32, deep, 1),
                    block_dim: (block_dim, 1, 1),
                    shared_mem_bytes: shared as u32,
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
    use super::{
        angular_descriptor_shared, angular_table_shared, force_way, in_parallel, pair_chunks,
        pair_starts, FittingPath, FittingRule, BLOCK, FORCE_WAYS, GROUPED_MAX_GROUPS, WIDE_BLOCK,
    };
    use FittingPath::{Fused, Grouped, NetByNet};

    /// Float32 models of five center types (those of HAP), with every path.
    fn five_types() -> FittingRule {
        FittingRule {
            double: false,
            ntypes: 5,
            fused: true,
            grouped: true,
            fused_max_rows: None,
            grouped_on: None,
        }
    }

    #[test]
    fn the_fitting_path_follows_the_rows_and_the_nets() {
        // Rows, atoms, nets.
        let float32 = five_types();
        // Three nets or more: the fused kernels up to 800 rows, the grouped
        // products above (one model on frames of 88 atoms has four nets,
        // four models on frames of 169 atoms twenty).
        assert_eq!(float32.path(704, 704, 4), Fused);
        assert_eq!(float32.path(800, 800, 4), Fused);
        assert_eq!(float32.path(801, 801, 4), Grouped);
        assert_eq!(float32.path(676, 169, 20), Fused);
        assert_eq!(float32.path(1_352, 338, 20), Grouped);
        assert_eq!(float32.path(10, 10, 3), Fused);
        assert_eq!(float32.path(900, 900, 3), Grouped);
        // Fewer: the fused kernels up to 340 atoms a center type of the
        // model, then a net at a time, and the grouped products from 1,920
        // rows (water: both types of two).
        let water = FittingRule {
            ntypes: 2,
            ..float32
        };
        assert_eq!(water.path(680, 680, 2), Fused);
        assert_eq!(water.path(768, 768, 2), NetByNet);
        assert_eq!(water.path(1_536, 1_536, 2), NetByNet);
        assert_eq!(water.path(1_919, 1_919, 2), NetByNet);
        assert_eq!(water.path(1_920, 1_920, 2), Grouped);
        // Two types of five (lime): the fused kernels up to 1,700 atoms.
        assert_eq!(float32.path(1_700, 1_700, 2), Fused);
        assert_eq!(float32.path(1_728, 1_728, 2), NetByNet);
        assert_eq!(float32.path(3_456, 3_456, 2), Grouped);
        assert_eq!(float32.path(1_701, 1_701, 1), NetByNet);
        assert_eq!(float32.path(1_920, 1_920, 1), Grouped);
        // Float64: the fused kernels at any size.
        let float64 = FittingRule {
            double: true,
            ..float32
        };
        assert_eq!(float64.path(100_000, 25_000, 20), Fused);
        assert_eq!(float64.path(100_000, 100_000, 2), Fused);
    }

    #[test]
    fn the_fitting_path_takes_what_is_set_and_what_there_is() {
        let float32 = five_types();
        // A limit of rows for the fused kernels in place of the rule.
        let limit = FittingRule {
            fused_max_rows: Some(1_000),
            ..float32
        };
        assert_eq!(limit.path(1_000, 1_000, 4), Fused);
        assert_eq!(limit.path(1_001, 1_001, 4), Grouped);
        assert_eq!(limit.path(1_001, 1_001, 2), NetByNet);
        let always = FittingRule {
            fused_max_rows: Some(usize::MAX),
            ..float32
        };
        assert_eq!(always.path(1 << 20, 1 << 20, 20), Fused);
        // Float64 without the fused kernels: as float32 with them past
        // their limit.
        let float64 = FittingRule {
            double: true,
            fused_max_rows: Some(0),
            ..float32
        };
        assert_eq!(float64.path(676, 169, 20), Grouped);
        assert_eq!(float64.path(500, 500, 2), NetByNet);
        assert_eq!(float64.path(1_920, 1_920, 2), Grouped);
        // No grouped products, by choice or for want of them: the rule of
        // before.
        for before in [
            FittingRule {
                grouped_on: Some(false),
                ..float32
            },
            FittingRule {
                grouped: false,
                ..float32
            },
        ] {
            assert_eq!(before.path(880, 880, 4), Fused);
            assert_eq!(before.path(1_352, 338, 20), Fused);
            assert_eq!(before.path(10_816, 2_704, 20), NetByNet);
            assert_eq!(before.path(3_456, 3_456, 2), NetByNet);
        }
        // Grouped products always: in place of every net at a time, after
        // the fused kernels by the rule; without them a net at a time.
        let on = FittingRule {
            ntypes: 2,
            grouped_on: Some(true),
            ..float32
        };
        assert_eq!(on.path(680, 680, 2), Fused);
        assert_eq!(on.path(768, 768, 2), Grouped);
        assert_eq!(
            FittingRule {
                grouped: false,
                ..on
            }
            .path(768, 768, 2),
            NetByNet
        );
        // No fused kernels.
        let unfused = FittingRule {
            fused: false,
            ..float32
        };
        assert_eq!(unfused.path(10, 10, 4), Grouped);
        assert_eq!(unfused.path(10, 10, 2), NetByNet);
        // More nets than one grouped call takes.
        let layered = FittingRule {
            fused_max_rows: Some(0),
            grouped_on: Some(true),
            ..float32
        };
        assert_eq!(layered.path(1 << 20, 16, GROUPED_MAX_GROUPS), Grouped);
        assert_eq!(layered.path(1 << 20, 16, GROUPED_MAX_GROUPS + 1), NetByNet);
    }

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

    /// What the se_e2_a kernels that look the tables up keep in shared
    /// memory, worked out by hand: the force kernel a place and a slope in
    /// the precision of the nets and a piece (four bytes) for every slot, up
    /// to the next eight bytes, where the sums of the block follow; the
    /// descriptor kernel B and the rows (four numbers each for every output
    /// and slot), the neighbors of each type (four bytes), and with the
    /// lookups a place and a piece for every slot.
    #[test]
    fn shared_memory_of_the_lookups() {
        // An odd number of slots ends off the eight-byte grid.
        assert_eq!(angular_table_shared::<f32>(121), 2 * 484 + 484 + 4);
        assert_eq!(angular_table_shared::<f64>(121), 2 * 968 + 484 + 4);
        // The slots of the HAP models.
        assert_eq!(angular_table_shared::<f32>(212), 2 * 848 + 848);
        assert_eq!(angular_table_shared::<f64>(212), 2 * 1696 + 848);
        assert_eq!(
            angular_descriptor_shared::<f32>(212, 100, 5, false),
            4 * 312 * 4 + 20 + 4
        );
        assert_eq!(
            angular_descriptor_shared::<f32>(212, 100, 5, true),
            4 * 312 * 4 + 212 * 8 + 20 + 4
        );
        assert_eq!(
            angular_descriptor_shared::<f64>(121, 16, 2, true),
            4 * 137 * 8 + 121 * 12 + 8 + 4
        );
    }

    #[test]
    fn force_ways_by_the_blocks_kept_at_once() {
        let [wide_sums, narrow_sums, wide, narrow] = FORCE_WAYS;
        assert_eq!((wide_sums.threads, wide_sums.sums), (WIDE_BLOCK, true));
        assert_eq!((narrow.threads, narrow.sums), (BLOCK, false));
        // What an RTX 4090 keeps of the HAP se_e2_a model in float32 with
        // lookups, and of the HAP se_e2_r model.
        let hap = [256, 512, 384, 896];
        let radial = [384, 768, 512, 1024];
        for (blocks, expected) in [
            (1, wide_sums),
            (88, wide_sums),
            (256, wide_sums),
            (257, narrow_sums),
            (512, narrow_sums),
            (513, narrow),
            (896, narrow),
        ] {
            for sums_past in [false, true] {
                let way = force_way(blocks, hap, None, None, sums_past);
                assert_eq!(way, expected, "{blocks} blocks");
            }
        }
        // Past one wave of every way: the narrow blocks, with the sums
        // where the descriptor keeps them past it.
        assert_eq!(force_way(897, hap, None, None, false), narrow);
        assert_eq!(force_way(897, hap, None, None, true), narrow_sums);
        assert_eq!(force_way(384, radial, None, None, true), wide_sums);
        assert_eq!(force_way(768, radial, None, None, true), narrow_sums);
        assert_eq!(force_way(769, radial, None, None, true), narrow);
        assert_eq!(force_way(1025, radial, None, None, true), narrow_sums);
        // No room for the sums: the wide blocks without them come first,
        // and the sums stay off past every way.
        let roomless = [0, 0, 384, 896];
        assert_eq!(force_way(384, roomless, None, None, true), wide);
        assert_eq!(force_way(385, roomless, None, None, true), narrow);
        assert_eq!(force_way(897, roomless, None, None, true), narrow);
        assert_eq!(force_way(897, roomless, None, Some(true), true), narrow);
        // Room for the sums of the narrow blocks only.
        let narrow_room = [0, 512, 384, 896];
        assert_eq!(force_way(88, narrow_room, None, None, false), narrow_sums);
        assert_eq!(force_way(600, narrow_room, Some(true), None, true), wide);
        // A width set by the override.
        assert_eq!(force_way(88, hap, Some(false), None, false), narrow_sums);
        assert_eq!(force_way(600, hap, Some(false), None, false), narrow);
        assert_eq!(force_way(1000, hap, Some(false), None, true), narrow_sums);
        assert_eq!(force_way(88, hap, Some(true), None, false), wide_sums);
        assert_eq!(force_way(300, hap, Some(true), None, false), wide);
        assert_eq!(force_way(400, hap, Some(true), None, false), wide);
        assert_eq!(force_way(400, hap, Some(true), None, true), wide_sums);
        // The sums set by the override.
        assert_eq!(force_way(88, hap, None, Some(false), true), wide);
        assert_eq!(force_way(400, hap, None, Some(false), true), narrow);
        assert_eq!(force_way(300, hap, None, Some(true), false), narrow_sums);
        assert_eq!(force_way(600, hap, None, Some(true), false), narrow_sums);
        // Past every way left, the sums set off stay off even where the
        // descriptor keeps them past it.
        assert_eq!(force_way(1025, radial, None, Some(false), true), narrow);
        assert_eq!(force_way(1025, radial, Some(true), Some(false), true), wide);
        assert_eq!(force_way(1025, radial, Some(true), None, true), wide_sums);
        // Both set: that way at any size.
        assert_eq!(
            force_way(5000, hap, Some(true), Some(true), false),
            wide_sums
        );
        assert_eq!(force_way(1, hap, Some(false), Some(false), true), narrow);
    }
}
