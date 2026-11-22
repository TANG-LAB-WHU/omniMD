//! Cell-list neighbor search with DeePMD semantics (plan I-1).
//!
//! Atoms are binned on the host (counting sort over a grid whose cells are at
//! least `rcut` wide); a GPU block of a few warps per center then scans the
//! stencil of cells, including periodic images, into shared memory and
//! selects like DeePMD-kit 3 (`build_neighbor_list`): the `sum(sel)` nearest
//! neighbors of any type, then at most `sel[type]` of each type, ordered by
//! (distance, atom, delta). Centers with more neighbors than the shared
//! buffer holds fall back to one thread with insertion into the slots, which
//! gives the same result. Neighbors beyond `sel` are dropped like DeePMD's
//! `format_nlist`, and the full per-type counts and the minimum distance are
//! kept for `dp neighbor-stat`.
//!
//! Small periodic boxes are handled by widening the stencil: offsets map to
//! distinct (cell, image) pairs, so every image within the cutoff is visited
//! once, however many images that takes.
use crate::{cuda_error, Device};
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
};
use omnimd_ml_contract::{Error, Result};
use std::sync::Arc;

const SOURCE: &str = r#"
#define FULL_MASK 0xffffffffu
// Status bits of a center.
#define COINCIDENT 1
#define TRUNCATED 2

__device__ bool before(double r, int j, const double* d, double r2, int j2, const double* d2) {
    if (r != r2) return r < r2;
    if (j != j2) return j < j2;
    for (int a = 0; a < 3; ++a) if (d[a] != d2[a]) return d[a] < d2[a];
    return false;
}

__device__ int floor_div(int a, int n) { return a >= 0 ? a / n : -((-a + n - 1) / n); }

// Stencil offset -> (cell, periodic image q); false when outside an open box.
__device__ bool stencil_cell(const int* ci, const int* o, const int* n, int periodic, int* cell, int* q) {
    int c[3];
    for (int a = 0; a < 3; ++a) {
        int raw = ci[a] + o[a];
        if (periodic) {
            q[a] = floor_div(raw, n[a]);
            c[a] = raw - q[a] * n[a];
        } else {
            q[a] = 0;
            c[a] = raw;
            if (raw < 0 || raw >= n[a]) return false;
        }
    }
    *cell = (c[0] * n[1] + c[1]) * n[2] + c[2];
    return true;
}

// The next stencil offset along an axis of `n` cells: the next cell, or in a
// periodic box the first cell of the next image.
__device__ void next_offset(int* c, int* q, int n, int periodic) {
    if (++*c == n && periodic) {
        *c = 0;
        ++*q;
    }
}

__device__ void image_delta(const double* pos, const double* lattice, const double* pi, int j,
        const int* q, double* d) {
    for (int a = 0; a < 3; ++a) {
        double shift = q[0] * lattice[a] + q[1] * lattice[3 + a] + q[2] * lattice[6 + a];
        d[a] = pos[3 * (long long)j + a] + shift - pi[a];
    }
}

__device__ int pack_image(const int* q) {
    return (q[0] + 512) | ((q[1] + 512) << 10) | ((q[2] + 512) << 20);
}

__device__ void unpack_image(int packed, int* q) {
    for (int a = 0; a < 3; ++a) q[a] = ((packed >> (10 * a)) & 1023) - 512;
}

// One thread does a whole center, in two scans over the stencil. The first
// keeps the `slots` nearest neighbors of any type, sorted, in the output
// arrays; its last entry is the selection threshold. The second starts over
// and inserts the neighbors at or inside the threshold into the sorted
// per-type segments, at most sel[t] each. Used when a center has more
// neighbors than the shared buffer of its block holds.
__device__ void serial_center(int i,
        const double* pos, const int* types, const int* cell_of, const int* cell_start,
        const int* cell_atoms, const double* lattice, int ntypes, const int* n,
        const int* reach, int periodic, double rcut, const int* sel, const int* offset, int slots,
        int* index, double* delta, double* dist, int* count, double* min_dist, int* flags,
        int* mirror) {
    long long base = (long long)i * slots;
    int status = 0;
    const double infinity = __longlong_as_double(0x7ff0000000000000LL);
    for (int t = 0; t < ntypes; ++t) count[(long long)i * ntypes + t] = 0;
    double best = infinity;
    const double* pi = pos + 3 * (long long)i;
    int ci[3] = {cell_of[3 * i], cell_of[3 * i + 1], cell_of[3 * i + 2]};
    int total = 0, limit_j = 0;
    double limit_r = infinity, limit_d[3] = {0.0, 0.0, 0.0};
    for (int pass = 0; pass < 2; ++pass) {
        for (int s = 0; s < slots; ++s) {
            index[base + s] = -1;
            dist[base + s] = infinity;
            for (int a = 0; a < 3; ++a) delta[(base + s) * 3 + a] = 0.0;
        }
        for (int ox = -reach[0]; ox <= reach[0]; ++ox)
        for (int oy = -reach[1]; oy <= reach[1]; ++oy)
        for (int oz = -reach[2]; oz <= reach[2]; ++oz) {
            int o[3] = {ox, oy, oz}, q[3], cell;
            if (!stencil_cell(ci, o, n, periodic, &cell, q)) continue;
            for (int k = cell_start[cell]; k < cell_start[cell + 1]; ++k) {
                int j = cell_atoms[k];
                if (j == i && q[0] == 0 && q[1] == 0 && q[2] == 0) continue;
                double d[3];
                image_delta(pos, lattice, pi, j, q, d);
                double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if (!(r2 < rcut * rcut)) continue;
                double r = sqrt(r2);
                if (r <= 1e-12) { status |= COINCIDENT; continue; }
                int t = types[j];
                // Segment to insert into: all slots, then the slots of type t.
                long long first = base;
                int capacity = slots, filled;
                if (pass == 0) {
                    count[(long long)i * ntypes + t] += 1;
                    if (r < best) best = r;
                    filled = min(total, slots);
                    ++total;
                } else {
                    if (sel[t] == 0 || before(limit_r, limit_j, limit_d, r, j, d)) continue;
                    first = base + offset[t];
                    capacity = sel[t];
                    filled = 0;
                    while (filled < capacity && index[first + filled] >= 0) ++filled;
                }
                int p = filled;
                if (filled == capacity) {
                    // Full segment: the new neighbor must beat the last entry.
                    long long last = first + capacity - 1;
                    if (!before(r, j, d, dist[last], index[last], delta + 3 * last)) continue;
                    p = capacity - 1;
                }
                while (p > 0) {
                    long long prev = first + p - 1;
                    if (!before(r, j, d, dist[prev], index[prev], delta + 3 * prev)) break;
                    dist[prev + 1] = dist[prev];
                    index[prev + 1] = index[prev];
                    for (int a = 0; a < 3; ++a) delta[3 * (prev + 1) + a] = delta[3 * prev + a];
                    --p;
                }
                long long at = first + p;
                dist[at] = r;
                index[at] = j;
                for (int a = 0; a < 3; ++a) delta[3 * at + a] = d[a];
            }
        }
        if (pass == 0 && total > slots) {
            long long last = base + slots - 1;
            limit_r = dist[last];
            limit_j = index[last];
            for (int a = 0; a < 3; ++a) limit_d[a] = delta[3 * last + a];
        }
    }
    min_dist[i] = best;
    for (int t = 0; t < ntypes; ++t)
        if (count[(long long)i * ntypes + t] > sel[t]) status |= TRUNCATED;
    flags[i] = status;
    if (mirror) mirror[i] = status;
}

// Adds to `overall` and `rank` the candidates at exactly the distance of
// candidate `e` that come before it: those of a lower atom index, and images
// of the same atom with a smaller delta.
__device__ __noinline__ void break_ties(int e, int total, const long long* key, const int* cj,
        const int* cimg, const int* ctype, const double* pos, const double* lattice,
        const double* pi, const double* de, int* overall, int* rank) {
    const double* cr = (const double*)key;
    for (int f = 0; f < total; ++f) {
        if (f == e || key[f] != key[e]) continue;
        double df[3];
        int qf[3];
        unpack_image(cimg[f], qf);
        image_delta(pos, lattice, pi, cj[f], qf, df);
        if (before(cr[f], cj[f], df, cr[e], cj[e], de)) {
            *overall += 1;
            if (ctype[f] == ctype[e]) *rank += 1;
        }
    }
}

// One block of a few warps per center, in three steps. The warps first scan
// the stencil's cells in single precision, each warp a share of its columns
// (the cells along the last axis at one pair of offsets along the first
// two), and append every atom within the cutoff plus a margin (which covers
// the rounding of that arithmetic, so no neighbor is lost) to the block's
// shared buffer: a ballot and popc place a batch of 32 atoms, and one atomic
// add on the count of candidates gives the batch its place. The few
// candidates are then measured in double precision, which decides exactly as
// a scan in double precision would. Last, each thread ranks candidates by
// (distance, atom, delta), overall and within their type, and writes those
// among the `slots` nearest overall and the sel[t] nearest of their type
// straight to their slots. A slot follows from the ranks alone, so the order
// in which the warps filled the buffer does not matter: every count of warps
// gives the same list, bit for bit. Same result as `serial_center`, which
// takes over when the buffer would overflow.
//
// Double precision is scarce on the device (about 1/64 of the single
// precision rate), so it is kept to the candidates, and distances are ranked
// by their bit patterns: positive doubles order like integers.
//
// A warp's share of the scan is its work from end to end, so a center waits
// for every operation of the longest share. Hence the several warps: few
// centers leave most of the device idle, and the blocks of many still fit
// it side by side. (On an RTX 4090, launched back to back, the kernel took
// 19.7 microseconds for 88 atoms to 6 A with one warp per center and takes
// 12.8 with four; 114 and 108 for 5,632 atoms. See `WARPS_PER_CENTER`.)
// A division by the cells of an axis takes a warp as long as fourteen
// multiply-adds (`arithmetic_costs` in `tests/costs.rs`), so the stencil is
// also walked without one for each of its cells: along an axis the cell and
// the image of the first offset come from one division and those of the
// others by steps (`next_offset`). Along the last axis the cells of one
// image lie one after the other in the grid, and so do their atoms, so they
// are scanned as one run of atoms: fewer bounds to read and fuller batches
// for the lanes. (On an RTX 4090 a search of 88 atoms to 6 A, with its
// packing and a wait for the device, took 37 microseconds with a division
// for every axis of every cell and 30 without, one warp per center; one of
// 5,632 atoms took 313 and 280: `neighbor_search_costs` in
// `tests/costs.rs`.)
//
// The atoms may belong to several systems that do not see each other (a
// batch of frames): every system has a cell and a grid of its own, and a
// center scans the cells of its system only. Atom indices are positions in
// the one list of all atoms throughout.
//
// `packed` is the per-search input in one buffer. As doubles: the lattice
// [9] of every system, then the positions [natoms][3]. Then in four-byte
// entries: a header of HEADER entries per system (as floats the lattice [9]
// and the square of the cutoff plus the scan margin; as ints the cells per
// axis [3], the stencil reach [3], whether it is periodic, its first cell
// and whether its image offsets fit `pack_image`); as floats the positions
// in cell order, so the scan reads them in sequence; then as ints the cell
// of each atom [natoms][3], the atoms in cell order [natoms] (those of a
// system together), the system of each atom [natoms] and the first entry of
// each cell [cells + 1] (the cells of a system together, a system's last
// cell ending where the next system starts). `tables` is [sel | first slot
// of each type]. Every center writes all of its outputs, including its
// status bits to `flags` and, when given, `mirror`.
#define HEADER 20
extern "C" __global__ void neighbor_search(
        const double* packed, const int* types, const int* tables, int natoms, int ntypes,
        int systems, double rcut, int slots, int capacity,
        int* index, double* delta, double* dist, int* count, double* min_dist, int* flags,
        int* mirror) {
    extern __shared__ double smem[];
    int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int tid = threadIdx.x, threads = blockDim.x;
    int i = blockIdx.x;
    if (i >= natoms) return;
    const double* pos = packed + 9 * (long long)systems;
    const int* headers = (const int*)(pos + 3 * (long long)natoms);
    const float* pos_f = (const float*)(headers + HEADER * (long long)systems);
    const int* cell_of = (const int*)(pos_f + 3 * (long long)natoms);
    const int* cell_atoms = cell_of + 3 * (long long)natoms;
    const int* system_of = cell_atoms + natoms;
    int system = system_of[i];
    const double* lattice = packed + 9 * (long long)system;
    const int* header = headers + HEADER * (long long)system;
    const float* lattice_f = (const float*)header;
    float wide2 = lattice_f[9];
    int n[3] = {header[10], header[11], header[12]};
    int reach[3] = {header[13], header[14], header[15]};
    int sx = reach[0], sy = reach[1], sz = reach[2], periodic = header[16];
    // The cells of this system.
    const int* cell_start = system_of + natoms + header[17];
    // Image offsets too large to pack: nothing is buffered and the center
    // goes to the serial path.
    int room = header[18] ? capacity : 0;
    const int* sel = tables;
    const int* offset = tables + ntypes;
    // Per block: r[capacity] and the nearest distance each warp found
    // [warps] as doubles, then as ints j, packed image and type [capacity],
    // the per-type counts [ntypes] and the candidates found so far.
    double* cr = smem;
    double* nearest = cr + capacity;
    int* cj = (int*)(nearest + warps);
    int* cimg = cj + capacity;
    int* ctype = cimg + capacity;
    int* cnt = ctype + capacity;
    int* found = cnt + ntypes;
    for (int t = tid; t < ntypes; t += threads) cnt[t] = 0;
    if (tid == 0) *found = 0;
    __syncthreads();
    const double* pi = pos + 3 * (long long)i;
    float pi_f[3] = {(float)pi[0], (float)pi[1], (float)pi[2]};
    int ci[3] = {cell_of[3 * i], cell_of[3 * i + 1], cell_of[3 * i + 2]};
    // The cell and the image of the first offset along each axis.
    int c0[3], q0[3];
    for (int a = 0; a < 3; ++a) {
        int raw = ci[a] - reach[a];
        q0[a] = periodic ? floor_div(raw, n[a]) : 0;
        c0[a] = raw - q0[a] * n[a];
    }
    // The warps take the columns of the stencil in turn.
    int column = 0;
    for (int ox = -sx, cx = c0[0], qx = q0[0]; ox <= sx; ++ox, next_offset(&cx, &qx, n[0], periodic))
    for (int oy = -sy, cy = c0[1], qy = q0[1]; oy <= sy; ++oy, next_offset(&cy, &qy, n[1], periodic)) {
        if (column++ % warps != warp) continue;
        // Outside an open box.
        if (cx < 0 || cx >= n[0] || cy < 0 || cy >= n[1]) continue;
        // Along the last axis: `left` offsets to go from cell `cz` of image
        // `qz`, a run of `cells` of them at a time.
        int cz = c0[2], qz = q0[2];
        for (int left = 2 * sz + 1, cells = 1; left > 0; left -= cells, cz += cells) {
            if (cz == n[2] && periodic) {
                cz = 0;
                ++qz;
            }
            cells = 1;
            if (cz < 0 || cz >= n[2]) continue;
            cells = min(left, n[2] - cz);
            int q[3] = {qx, qy, qz}, cell = (cx * n[1] + cy) * n[2] + cz;
            bool self_image = q[0] == 0 && q[1] == 0 && q[2] == 0;
            int image = pack_image(q);
            float shift[3];
            for (int a = 0; a < 3; ++a)
                shift[a] = q[0] * lattice_f[a] + q[1] * lattice_f[3 + a] + q[2] * lattice_f[6 + a]
                           - pi_f[a];
            int end = cell_start[cell + cells];
            for (int k0 = cell_start[cell]; k0 < end; k0 += 32) {
                int k = k0 + lane, j = -1;
                bool near = false;
                if (k < end) {
                    j = cell_atoms[k];
                    const float* pj = pos_f + 3 * (long long)k;
                    float dx = pj[0] + shift[0], dy = pj[1] + shift[1], dz = pj[2] + shift[2];
                    // Negated, so a sum that is not a number goes on to the exact pass.
                    near = !(j == i && self_image) && !(dx * dx + dy * dy + dz * dz >= wide2);
                }
                unsigned mask = __ballot_sync(FULL_MASK, near);
                int start = 0;
                if (lane == 0 && mask) start = atomicAdd(found, __popc(mask));
                start = __shfl_sync(FULL_MASK, start, 0);
                if (near) {
                    int at = start + __popc(mask & ((1u << lane) - 1u));
                    if (at < room) {
                        cj[at] = j;
                        cimg[at] = image;
                    }
                }
            }
        }
    }
    __syncthreads();
    int total = *found;
    if (total > room) {
        if (tid == 0)
            serial_center(i, pos, types, cell_of, cell_start, cell_atoms, lattice, ntypes, n, reach,
                          periodic, rcut, sel, offset, slots, index, delta, dist, count, min_dist,
                          flags, mirror);
        return;
    }
    // The candidates in double precision. Those outside the cutoff stay in
    // the buffer with an infinite distance and no type.
    const double infinity = __longlong_as_double(0x7ff0000000000000LL);
    double best = infinity;
    bool touching = false;
    for (int e = tid; e < total; e += threads) {
        int j = cj[e], q[3], t = -1;
        double d[3], r = infinity;
        unpack_image(cimg[e], q);
        image_delta(pos, lattice, pi, j, q, d);
        double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if (r2 < rcut * rcut) {
            double root = sqrt(r2);
            if (root <= 1e-12) touching = true;
            else {
                r = root;
                t = types[j];
                best = fmin(best, r);
                atomicAdd(&cnt[t], 1);
            }
        }
        cr[e] = r;
        ctype[e] = t;
    }
    for (int offset_ = 16; offset_ > 0; offset_ >>= 1)
        best = fmin(best, __shfl_down_sync(FULL_MASK, best, offset_));
    if (lane == 0) nearest[warp] = best;
    // A barrier: the counts, distances and types are all in.
    bool coincident = __syncthreads_or(touching);
    long long base = (long long)i * slots;
    for (int s = tid; s < slots; s += threads) {
        index[base + s] = -1;
        dist[base + s] = infinity;
        for (int a = 0; a < 3; ++a) delta[(base + s) * 3 + a] = 0.0;
    }
    bool over = false;
    for (int t = tid; t < ntypes; t += threads) {
        count[(long long)i * ntypes + t] = cnt[t];
        over |= cnt[t] > sel[t];
    }
    // A barrier too: every slot is emptied before any is filled.
    bool truncated = __syncthreads_or(over);
    if (tid == 0) {
        for (int w = 1; w < warps; ++w) best = fmin(best, nearest[w]);
        int status = (coincident ? COINCIDENT : 0) | (truncated ? TRUNCATED : 0);
        min_dist[i] = best;
        flags[i] = status;
        if (mirror) mirror[i] = status;
    }
    // Distances as integers: for positive doubles the order is the same.
    // Candidates at exactly the distance of `e` are only counted here; the
    // rare ties are decided afterwards, in a function of their own, so that
    // their double precision arithmetic stays out of this loop (inlined, the
    // compiler runs it for every pair and discards the result).
    const long long* key = (const long long*)cr;
    for (int e = tid; e < total; e += threads) {
        int t = ctype[e];
        if (t < 0 || sel[t] == 0) continue;
        long long ke = key[e];
        int rank = 0, overall = 0, ties = 0;
        for (int f = 0; f < total; ++f) {
            long long kf = key[f];
            int nearer = kf < ke;
            ties += kf == ke;
            overall += nearer;
            rank += nearer & (ctype[f] == t);
        }
        double re = cr[e], de[3];
        int je = cj[e], qe[3];
        unpack_image(cimg[e], qe);
        image_delta(pos, lattice, pi, je, qe, de);
        // `e` itself is one of the ties.
        if (ties > 1)
            break_ties(e, total, key, cj, cimg, ctype, pos, lattice, pi, de, &overall, &rank);
        if (overall < slots && rank < sel[t]) {
            long long at = base + offset[t] + rank;
            index[at] = je;
            dist[at] = re;
            for (int a = 0; a < 3; ++a) delta[3 * at + a] = de[a];
        }
    }
}
"#;

/// Simulation cell: rows are lattice vectors (Å), as in `Frame::cell`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Cell {
    Open,
    Periodic([[f64; 3]; 3]),
}

/// `dp neighbor-stat` numbers for one frame.
#[derive(Clone, Debug, PartialEq)]
pub struct NeighborStats {
    /// Largest per-type neighbor count within `rcut` over all centers.
    pub max_count: Vec<usize>,
    /// Smallest neighbor distance (infinite when no pair is within `rcut`).
    pub min_distance: f64,
}

/// Fixed-capacity neighbor slots, `sel[t]` per neighbor type in type order.
pub struct NeighborList {
    pub natoms: usize,
    pub sel: Vec<usize>,
    pub slots: usize,
    /// `[natoms]` atom types, as given to the search.
    pub types: CudaSlice<i32>,
    /// `[natoms][slots]`, -1 for an empty slot.
    pub index: CudaSlice<i32>,
    /// `[natoms][slots][3]`, image-shifted `r_j - r_i`; zero in empty slots.
    pub delta: CudaSlice<f64>,
    /// `[natoms][slots]`, the length of `delta`; infinite in empty slots.
    /// Later kernels read it instead of taking the square root again.
    pub dist: CudaSlice<f64>,
    /// `[natoms][ntypes]` neighbors within `rcut` before truncation to `sel`.
    pub count: CudaSlice<i32>,
    /// `[natoms]` nearest-neighbor distance.
    pub min_distance: CudaSlice<f64>,
    /// `[natoms]` status bits ([`COINCIDENT`], [`TRUNCATED`]), read by `check`
    /// so a search never syncs.
    flags: CudaSlice<i32>,
}

/// Status bit of a center with another atom within 1e-12 Å.
pub const COINCIDENT: i32 = 1;
/// Status bit of a center with more neighbors of some type than `sel`.
pub const TRUNCATED: i32 = 2;

/// A neighbor search that keeps its device buffers between calls. Searching
/// a system of the same size again costs one upload and one launch, against
/// fifteen device calls for a search from nothing.
pub struct NeighborSearch {
    rcut: f64,
    sel: Vec<usize>,
    slots: usize,
    /// Host copy of the packed kernel input, reused between calls, and the
    /// memory of putting it together.
    words: Vec<u64>,
    scratch: Scratch,
    /// Warps of the block of a center.
    warps: usize,
    state: Option<State>,
}

/// Device side of a search, for one stream, one atom count and one number of
/// systems.
struct State {
    stream: Arc<CudaStream>,
    /// The search kernel, looked up once: the lookup hashes the source.
    function: CudaFunction,
    /// `[sel | first slot of each type]`.
    tables: CudaSlice<i32>,
    /// Host copy of `list.types`.
    types: Vec<i32>,
    systems: usize,
    packed: CudaSlice<u64>,
    list: NeighborList,
}

impl State {
    fn new(
        device: &Device,
        sel: &[usize],
        slots: usize,
        types: &[i32],
        systems: usize,
    ) -> Result<Self> {
        let natoms = types.len();
        // Room for the input on the largest grids these atoms can get.
        let words = packed_words(natoms, cells_bound(natoms, systems), systems);
        let mut tables: Vec<i32> = sel.iter().map(|&s| s as i32).collect();
        let mut first = 0;
        for &s in sel {
            tables.push(first as i32);
            first += s;
        }
        // Every center writes all of its outputs before anything reads them.
        let list = unsafe {
            NeighborList {
                natoms,
                sel: sel.to_vec(),
                slots,
                types: device.upload(types)?,
                index: device.alloc_uninit(natoms * slots)?,
                delta: device.alloc_uninit(natoms * slots * 3)?,
                dist: device.alloc_uninit(natoms * slots)?,
                count: device.alloc_uninit(natoms * sel.len())?,
                min_distance: device.alloc_uninit(natoms)?,
                flags: device.alloc_uninit(natoms)?,
            }
        };
        Ok(Self {
            stream: device.stream().clone(),
            function: device.kernels().function(SOURCE, "neighbor_search")?,
            tables: device.upload(&tables)?,
            types: types.to_vec(),
            systems,
            packed: unsafe { device.alloc_uninit(words)? },
            list,
        })
    }
}

/// Four-byte entries of a system's header in the kernel input.
const HEADER: usize = 20;

/// Host memory of the packing, kept between searches.
#[derive(Default)]
struct Scratch {
    /// The atoms of every system in their order: those of system `f` are
    /// `order[first[f]..first[f + 1]]`.
    order: Vec<u32>,
    first: Vec<usize>,
    /// The grid of every system.
    shapes: Vec<Shape>,
    /// The cell of each atom of the system being packed, in `order`.
    flat: Vec<u32>,
    /// Atoms a cell, then the cursors of the counting sort.
    counts: Vec<usize>,
}

/// Bins the atoms of every system and writes the kernel's input (see
/// `neighbor_search`) to `words`. `system` names the system of each atom
/// and is empty when there is one. Returns the number of cells of all grids.
fn pack(
    words: &mut Vec<u64>,
    scratch: &mut Scratch,
    positions: &[[f64; 3]],
    system: &[i32],
    cells: &[Cell],
    rcut: f64,
) -> Result<usize> {
    let (total, length) = plan(scratch, positions, system, cells, rcut)?;
    // `fill` writes every word: no zero fill unless the length changes.
    if words.len() != length {
        words.clear();
        words.resize(length, 0);
    }
    fill(words, scratch, positions, cells, rcut, total);
    Ok(total)
}

/// Groups the atoms by system and lays out the grid of every system in
/// `scratch`. Returns the number of cells of all grids and the words of the
/// kernel input.
fn plan(
    scratch: &mut Scratch,
    positions: &[[f64; 3]],
    system: &[i32],
    cells: &[Cell],
    rcut: f64,
) -> Result<(usize, usize)> {
    let (natoms, systems) = (positions.len(), cells.len());
    let Scratch {
        order,
        first,
        shapes,
        counts,
        ..
    } = scratch;
    first.clear();
    first.resize(systems + 1, 0);
    order.clear();
    if system.is_empty() {
        first[1..].fill(natoms);
        order.extend(0..natoms as u32);
    } else {
        for &of in system {
            first[of as usize + 1] += 1;
        }
        for f in 0..systems {
            first[f + 1] += first[f];
        }
        order.resize(natoms, 0);
        counts.clear();
        counts.extend_from_slice(first);
        for (atom, &of) in system.iter().enumerate() {
            order[counts[of as usize]] = atom as u32;
            counts[of as usize] += 1;
        }
    }
    shapes.clear();
    let mut total = 0;
    for (f, cell) in cells.iter().enumerate() {
        let mine = &order[first[f]..first[f + 1]];
        if mine.is_empty() {
            return Err(Error(format!(
                "system {f} of the neighbor search has no atoms"
            )));
        }
        let shape = Shape::new(positions, mine, cell, rcut)?;
        total += shape.cells.iter().product::<usize>();
        shapes.push(shape);
    }
    Ok((total, packed_words(natoms, total, systems)))
}

/// Writes all of the kernel input that `plan` laid out to `words`, its
/// length: the lattices and the positions as doubles, then the four-byte
/// entries, two to a word as they lie in memory. `total` is the number of
/// cells of all grids.
fn fill(
    words: &mut [u64],
    scratch: &mut Scratch,
    positions: &[[f64; 3]],
    cells: &[Cell],
    rcut: f64,
    total: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        // SAFETY: the processor has SSE4.1.
        unsafe { fill_sse41(words, scratch, positions, cells, rcut, total) };
        return;
    }
    fill_any(words, scratch, positions, cells, rcut, total);
}

/// `fill` for a processor with SSE4.1, whose rounding instruction takes the
/// floor of a fractional coordinate where the x86-64 baseline calls the C
/// library: the same words, in 8% to 15% less time on a performance core of
/// an i9-14900K and 15% to 24% on an efficiency core (88 to 12,288 atoms).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn fill_sse41(
    words: &mut [u64],
    scratch: &mut Scratch,
    positions: &[[f64; 3]],
    cells: &[Cell],
    rcut: f64,
    total: usize,
) {
    fill_any(words, scratch, positions, cells, rcut, total)
}

#[inline(always)]
fn fill_any(
    words: &mut [u64],
    scratch: &mut Scratch,
    positions: &[[f64; 3]],
    cells: &[Cell],
    rcut: f64,
    total: usize,
) {
    let (natoms, systems) = (positions.len(), cells.len());
    debug_assert_eq!(words.len(), packed_words(natoms, total, systems));
    let Scratch {
        order,
        first,
        shapes,
        flat,
        counts,
    } = scratch;
    let at_positions = 9 * systems;
    let (doubles, rest) = words.split_at_mut(at_positions + 3 * natoms);
    // SAFETY: u32 has no invalid bit patterns and an alignment that divides
    // that of u64, and `rest` holds twice as many of them as it holds words.
    let halves: &mut [u32] =
        unsafe { std::slice::from_raw_parts_mut(rest.as_mut_ptr().cast(), 2 * rest.len()) };
    let at_floats = HEADER * systems;
    let at_cell_of = at_floats + 3 * natoms;
    let at_atoms = at_cell_of + 3 * natoms;
    let at_system = at_atoms + natoms;
    let at_starts = at_system + natoms;
    flat.resize(natoms, 0);
    let mut before = 0;
    for (f, (cell, shape)) in cells.iter().zip(shapes.iter()).enumerate() {
        let mine = &order[first[f]..first[f + 1]];
        let grid = shape.cells;
        let count: usize = grid.iter().product();
        let mut lattice = [0.0; 9];
        let periodic = match cell {
            Cell::Periodic(rows) => {
                for (entry, value) in lattice.iter_mut().zip(rows.iter().flatten()) {
                    *entry = *value;
                }
                true
            }
            Cell::Open => false,
        };
        for (entry, value) in doubles[9 * f..9 * f + 9].iter_mut().zip(lattice) {
            *entry = value.to_bits();
        }
        // One pass places the atoms, writes what goes in atom order, and
        // counts the atoms of each cell.
        counts.clear();
        counts.resize(count + 1, 0);
        let mut magnitude = 0.0f64;
        for (slot, &atom) in flat.iter_mut().zip(mine) {
            let atom = atom as usize;
            let (c, position) = shape.place(&positions[atom]);
            for axis in 0..3 {
                doubles[at_positions + 3 * atom + axis] = position[axis].to_bits();
                halves[at_cell_of + 3 * atom + axis] = c[axis] as u32;
            }
            if !periodic {
                magnitude = position.iter().fold(magnitude, |m, v| m.max(v.abs()));
            }
            let index = (c[0] * grid[1] + c[1]) * grid[2] + c[2];
            *slot = index as u32;
            counts[index + 1] += 1;
        }
        let header = &mut halves[HEADER * f..HEADER * (f + 1)];
        for (entry, value) in header.iter_mut().zip(lattice) {
            *entry = (value as f32).to_bits();
        }
        if periodic {
            magnitude = image_magnitude(&lattice, &shape.reach);
        }
        header[9] = scan_radius2(rcut, magnitude).to_bits();
        for axis in 0..3 {
            header[10 + axis] = grid[axis] as u32;
            header[13 + axis] = shape.reach[axis] as u32;
        }
        header[16] = u32::from(periodic);
        header[17] = before as u32;
        // Packed images need |q| < 512; past that the centers go serial.
        header[18] = u32::from(shape.reach.iter().all(|&r| r < 500));
        header[19] = 0;
        // Counting sort of the system's atoms by cell. The first entry of
        // each cell is a place in the list of all atoms in cell order.
        for k in 0..count {
            counts[k + 1] += counts[k];
        }
        let base = first[f];
        for (entry, &k) in halves[at_starts + before..at_starts + before + count]
            .iter_mut()
            .zip(&counts[..count])
        {
            *entry = (base + k) as u32;
        }
        for (&index, &atom) in flat.iter().zip(mine) {
            let index = index as usize;
            let k = base + counts[index];
            counts[index] += 1;
            let atom = atom as usize;
            for axis in 0..3 {
                let position = f64::from_bits(doubles[at_positions + 3 * atom + axis]);
                halves[at_floats + 3 * k + axis] = (position as f32).to_bits();
            }
            halves[at_atoms + k] = atom as u32;
            halves[at_system + atom] = f as u32;
        }
        before += count;
    }
    halves[at_starts + total] = natoms as u32;
    // The half of the last word past the entries, when there is one.
    if let Some(pad) = halves.get_mut(at_starts + total + 1) {
        *pad = 0;
    }
}

/// Square of the radius of the kernel's single-precision scan: the cutoff
/// plus what that arithmetic can misplace a distance by, so the scan keeps
/// every neighbor (and a few atoms just outside, which the pass in double
/// precision drops).
///
/// With coordinates and image shifts of magnitude at most `m`
/// (`magnitude`), rounding positions and lattice to single precision and
/// the sums that form a separation move each of its components by less
/// than 14 units of roundoff of `m` (7 epsilon), the distance by less than
/// 13 epsilon `m`, and the squared sum adds under one epsilon of the
/// cutoff. 32 epsilon of `m + rcut` leaves more than a factor two.
fn scan_radius2(rcut: f64, magnitude: f64) -> f32 {
    let wide = rcut + 32.0 * f64::from(f32::EPSILON) * (magnitude + rcut);
    let square = (wide * wide) as f32;
    // One step up, so rounding the square cannot shrink it.
    if square.is_finite() {
        f32::from_bits(square.to_bits() + 1)
    } else {
        f32::INFINITY
    }
}

/// `m` of `scan_radius2` for a periodic system: the largest coordinate of
/// an image shift its stencil reaches (positions wrapped into the cell are
/// smaller). An open system's is its largest coordinate.
fn image_magnitude(lattice: &[f64; 9], reach: &[usize; 3]) -> f64 {
    (0..3)
        .map(|a| {
            (0..3)
                .map(|b| (reach[b] + 1) as f64 * lattice[3 * b + a].abs())
                .sum::<f64>()
        })
        .fold(0.0, f64::max)
}

fn inverse(m: &[[f64; 3]; 3]) -> Result<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let inv = [
        [
            m[1][1] * m[2][2] - m[1][2] * m[2][1],
            m[0][2] * m[2][1] - m[0][1] * m[2][2],
            m[0][1] * m[1][2] - m[0][2] * m[1][1],
        ],
        [
            m[1][2] * m[2][0] - m[1][0] * m[2][2],
            m[0][0] * m[2][2] - m[0][2] * m[2][0],
            m[0][2] * m[1][0] - m[0][0] * m[1][2],
        ],
        [
            m[1][0] * m[2][1] - m[1][1] * m[2][0],
            m[0][1] * m[2][0] - m[0][0] * m[2][1],
            m[0][0] * m[1][1] - m[0][1] * m[1][0],
        ],
    ]
    .map(|row| row.map(|v| v / det));
    if !det.is_finite() || det.abs() < 1e-12 || inv.iter().flatten().any(|v| !v.is_finite()) {
        return Err(Error("cell is singular or nonfinite".into()));
    }
    Ok(inv)
}

/// Grid size per axis (cells at least `rcut` wide, at most 512 per axis) and
/// the stencil half-width that reaches every neighbor within `rcut`.
fn grid_axis(width: f64, rcut: f64) -> (usize, usize) {
    let cells = ((width / rcut).floor() as usize).clamp(1, 512);
    (cells, stencil_reach(width, rcut, cells))
}

/// Cells to look at on each side so that everything within `rcut` is seen,
/// on an axis of `width` cut into `cells`.
fn stencil_reach(width: f64, rcut: f64, cells: usize) -> usize {
    ((rcut * cells as f64 / width * (1.0 + 1e-12)).ceil() as usize).max(1)
}

/// The most cells a grid for `atoms` atoms may have.
fn cell_limit(atoms: usize) -> usize {
    (4 * atoms).max(64)
}

/// The most cells the grids of `systems` systems with `atoms` atoms between
/// them may have: `cell_limit` of each.
fn cells_bound(atoms: usize, systems: usize) -> usize {
    4 * atoms + 64 * systems
}

/// Words of the packed kernel input for `atoms` atoms in `systems` systems
/// whose grids have `cells` cells in all: doubles, then four-byte entries
/// two to a word.
fn packed_words(atoms: usize, cells: usize, systems: usize) -> usize {
    let doubles = 9 * systems + 3 * atoms;
    let halves = HEADER * systems + 8 * atoms + cells + 1;
    doubles + halves.div_ceil(2)
}

/// Fewer, wider cells when the grid of `grid_axis` has many more cells than
/// there are atoms (a molecule in a large box): at most `cell_limit`. Empty
/// cells cost host time and upload size for nothing, and the limit bounds
/// the size of the kernel input.
fn coarsen(
    cells: &mut [usize; 3],
    reach: &mut [usize; 3],
    widths: &[f64; 3],
    rcut: f64,
    atoms: usize,
) {
    let limit = cell_limit(atoms);
    loop {
        let total: usize = cells.iter().product();
        // Axes that still have cells to give up share the reduction.
        let divisible = cells.iter().filter(|&&c| c > 1).count();
        if total <= limit || divisible == 0 {
            break;
        }
        let shrink = (limit as f64 / total as f64).powf(1.0 / divisible as f64);
        for count in cells.iter_mut().filter(|c| **c > 1) {
            *count = ((*count as f64 * shrink) as usize).max(1);
        }
    }
    for axis in 0..3 {
        reach[axis] = stencil_reach(widths[axis], rcut, cells[axis]);
    }
}

/// The grid of one system: cells per axis, how far the stencil reaches, and
/// where an atom goes.
struct Shape {
    cells: [usize; 3],
    reach: [usize; 3],
    place: Place,
}

enum Place {
    /// Fractional coordinates by the inverse lattice, wrapped into the cell.
    Periodic {
        inverse: [[f64; 3]; 3],
        lattice: [[f64; 3]; 3],
    },
    /// Coordinates in the box around the atoms, `width` at least the cutoff.
    Open { lo: [f64; 3], width: [f64; 3] },
}

impl Shape {
    /// The grid of the atoms `mine` of `positions` in `cell`.
    fn new(positions: &[[f64; 3]], mine: &[u32], cell: &Cell, rcut: f64) -> Result<Self> {
        let mut cells = [1; 3];
        let mut reach = [1; 3];
        let mut widths = [0.0; 3];
        let place = match cell {
            Cell::Periodic(lattice) => {
                let inverse = inverse(lattice)?;
                for axis in 0..3 {
                    // Distance between lattice planes = 1 / |reciprocal vector|.
                    let norm = (0..3).map(|a| inverse[a][axis].powi(2)).sum::<f64>().sqrt();
                    widths[axis] = 1.0 / norm;
                    (cells[axis], reach[axis]) = grid_axis(widths[axis], rcut);
                }
                Place::Periodic {
                    inverse,
                    lattice: *lattice,
                }
            }
            Cell::Open => {
                let lo: [f64; 3] = std::array::from_fn(|a| {
                    mine.iter()
                        .map(|&atom| positions[atom as usize][a])
                        .fold(f64::INFINITY, f64::min)
                });
                let hi: [f64; 3] = std::array::from_fn(|a| {
                    mine.iter()
                        .map(|&atom| positions[atom as usize][a])
                        .fold(f64::NEG_INFINITY, f64::max)
                });
                for axis in 0..3 {
                    widths[axis] = (hi[axis] - lo[axis]).max(rcut);
                    (cells[axis], reach[axis]) = grid_axis(widths[axis], rcut);
                }
                Place::Open { lo, width: widths }
            }
        };
        coarsen(&mut cells, &mut reach, &widths, rcut, mine.len());
        Ok(Self {
            cells,
            reach,
            place,
        })
    }

    /// The cell of an atom at `p`, and its position: wrapped into the cell
    /// when the system is periodic, as it is when open.
    #[inline(always)]
    fn place(&self, p: &[f64; 3]) -> ([usize; 3], [f64; 3]) {
        let (fraction, position): ([f64; 3], [f64; 3]) = match &self.place {
            Place::Periodic { inverse, lattice } => {
                let mut f: [f64; 3] =
                    std::array::from_fn(|b| (0..3).map(|a| p[a] * inverse[a][b]).sum());
                for v in &mut f {
                    *v -= v.floor();
                    if *v >= 1.0 {
                        *v = 0.0;
                    }
                }
                let wrapped = std::array::from_fn(|a| (0..3).map(|b| f[b] * lattice[b][a]).sum());
                (f, wrapped)
            }
            Place::Open { lo, width } => (
                std::array::from_fn(|a| ((p[a] - lo[a]) / width[a]).min(1.0 - f64::EPSILON)),
                *p,
            ),
        };
        let cell = std::array::from_fn(|a| {
            ((fraction[a] * self.cells[a] as f64) as usize).min(self.cells[a] - 1)
        });
        (cell, position)
    }
}

/// Shared memory one block may use without opting in to more.
const SHARED_LIMIT: usize = 48 * 1024;

/// Doubles of shared memory of a block of `warps` warps for a buffer of
/// `capacity` neighbors.
fn block_doubles(capacity: usize, ntypes: usize, warps: usize) -> usize {
    capacity + warps + (3 * capacity + ntypes + 2) / 2
}

/// The largest buffer whose block fits [`SHARED_LIMIT`].
fn max_capacity(ntypes: usize, warps: usize) -> usize {
    (2 * (SHARED_LIMIT / 8) - 2 * warps - ntypes - 2) / 5
}

/// Warps that work on one center by default. Timed on an RTX 4090 with one,
/// two, four and eight, in evaluations of the HAP model (float32 with
/// tables; 88 to 5,632 atoms) and of a water model (float64; 192 and 1,536):
/// four were the fastest or within the scatter of the fastest throughout.
/// Two kept up only for thousands of atoms; eight were within the scatter
/// of four up to 224 atoms, slower from 704 on, and cost the kernel a third
/// more time for 5,632. Against one warp, an evaluation of 88 to 704 atoms
/// is 2 to 7 microseconds shorter. One of 12,288 water atoms (of 60,000) was
/// longer in three runs, by 26 to 164, which the scatter of the runs (41 to
/// 138) leaves unsettled (`bench/results/SUMMARY.md`).
const WARPS_PER_CENTER: usize = 4;

/// Most warps per center. Within a multiprocessor's 65,536 registers up to
/// 128 a thread (the kernel takes 80).
const MAX_WARPS_PER_CENTER: usize = 16;

impl NeighborSearch {
    pub fn new(rcut: f64, sel: &[usize]) -> Result<Self> {
        if !(rcut.is_finite() && rcut > 0.0) {
            return Err(Error(format!(
                "rcut must be finite and positive, got {rcut}"
            )));
        }
        let slots: usize = sel.iter().sum();
        if slots == 0 || slots > 65536 {
            return Err(Error(format!("sum(sel) must be in 1..=65536, got {slots}")));
        }
        Ok(Self {
            rcut,
            sel: sel.to_vec(),
            slots,
            words: Vec::new(),
            scratch: Scratch::default(),
            warps: WARPS_PER_CENTER,
            state: None,
        })
    }

    /// The warps that work on one center, 1 to `MAX_WARPS_PER_CENTER` (16).
    /// Every count gives the same list. For tests and probes.
    #[doc(hidden)]
    pub fn set_warps_per_center(&mut self, warps: usize) {
        self.warps = warps.clamp(1, MAX_WARPS_PER_CENTER);
    }

    /// Neighbors of every atom. The list is overwritten by the next search.
    pub fn run(
        &mut self,
        device: &Device,
        positions: &[[f64; 3]],
        types: &[i32],
        cell: &Cell,
    ) -> Result<&NeighborList> {
        let mirror = None::<&mut CudaSlice<i32>>;
        let cells = std::slice::from_ref(cell);
        self.search(device, positions, types, &[], cells, None, mirror)
    }

    /// `run` that also writes the status bits of each center to `mirror`
    /// (`[natoms]`), for a caller that fetches them along with its own
    /// results and folds them with [`NeighborList::status`].
    pub fn run_mirrored(
        &mut self,
        device: &Device,
        positions: &[[f64; 3]],
        types: &[i32],
        cell: &Cell,
        mirror: &mut impl DevicePtrMut<i32>,
    ) -> Result<&NeighborList> {
        let cells = std::slice::from_ref(cell);
        self.search(device, positions, types, &[], cells, None, Some(mirror))
    }

    /// `run` for several systems that do not see each other, such as the
    /// frames of a batch, in one launch: atom `i` belongs to system
    /// `system[i]`, whose cell is `cells[system[i]]`, and gets the neighbors
    /// it would get if its system were searched alone. Neighbor indices are
    /// positions in `positions`, the one list of all atoms, which may be in
    /// any order; atoms of one system rank among themselves by that order.
    /// Every system needs an atom. `system` may be left empty when there is
    /// one system.
    pub fn run_systems(
        &mut self,
        device: &Device,
        positions: &[[f64; 3]],
        types: &[i32],
        system: &[i32],
        cells: &[Cell],
    ) -> Result<&NeighborList> {
        let mirror = None::<&mut CudaSlice<i32>>;
        self.search(device, positions, types, system, cells, None, mirror)
    }

    /// [`Self::run_systems`] with the status mirror of
    /// [`Self::run_mirrored`].
    pub fn run_systems_mirrored(
        &mut self,
        device: &Device,
        positions: &[[f64; 3]],
        types: &[i32],
        system: &[i32],
        cells: &[Cell],
        mirror: &mut impl DevicePtrMut<i32>,
    ) -> Result<&NeighborList> {
        self.search(device, positions, types, system, cells, None, Some(mirror))
    }

    /// `run` with an explicit per-center shared buffer size; centers with
    /// more neighbors take the serial path. For tests of that path.
    #[doc(hidden)]
    pub fn run_with_capacity(
        &mut self,
        device: &Device,
        positions: &[[f64; 3]],
        types: &[i32],
        cell: &Cell,
        capacity: Option<usize>,
    ) -> Result<&NeighborList> {
        let mirror = None::<&mut CudaSlice<i32>>;
        let cells = std::slice::from_ref(cell);
        self.search(device, positions, types, &[], cells, capacity, mirror)
    }

    /// [`Self::run_systems`] with the buffer size of
    /// [`Self::run_with_capacity`].
    #[doc(hidden)]
    pub fn run_systems_with_capacity(
        &mut self,
        device: &Device,
        positions: &[[f64; 3]],
        types: &[i32],
        system: &[i32],
        cells: &[Cell],
        capacity: Option<usize>,
    ) -> Result<&NeighborList> {
        let mirror = None::<&mut CudaSlice<i32>>;
        self.search(device, positions, types, system, cells, capacity, mirror)
    }

    /// The list of the last search.
    pub fn list(&self) -> Option<&NeighborList> {
        self.state.as_ref().map(|state| &state.list)
    }

    /// Gives the list of the last search away.
    pub fn into_list(self) -> Option<NeighborList> {
        self.state.map(|state| state.list)
    }

    #[allow(clippy::too_many_arguments)]
    fn search<M: DevicePtrMut<i32>>(
        &mut self,
        device: &Device,
        positions: &[[f64; 3]],
        types: &[i32],
        system: &[i32],
        cells: &[Cell],
        capacity: Option<usize>,
        mut mirror: Option<&mut M>,
    ) -> Result<&NeighborList> {
        let natoms = positions.len();
        let ntypes = self.sel.len();
        let systems = cells.len();
        let (rcut, slots) = (self.rcut, self.slots);
        if natoms == 0 || types.len() != natoms || i32::try_from(natoms).is_err() {
            return Err(Error(format!(
                "neighbor search needs 1..=i32::MAX atoms with one type each; got {natoms} positions and {} types",
                types.len()
            )));
        }
        if ntypes == 0 || types.iter().any(|&t| t < 0 || t as usize >= ntypes) {
            return Err(Error(format!("atom types must be in 0..{ntypes}")));
        }
        if positions.iter().flatten().any(|v| !v.is_finite()) {
            return Err(Error("nonfinite atom position".into()));
        }
        // One system needs no list of who belongs to it.
        let named = !(system.is_empty() && systems == 1);
        if systems == 0
            || systems > natoms
            || (named && system.len() != natoms)
            || system.iter().any(|&s| s < 0 || s as usize >= systems)
        {
            return Err(Error(format!(
                "neighbor search over {systems} systems needs one system below that for each of the {natoms} atoms; got {}",
                system.len()
            )));
        }
        if mirror.as_ref().is_some_and(|m| m.len() != natoms) {
            return Err(Error(format!(
                "the status mirror needs one entry for each of the {natoms} atoms"
            )));
        }
        pack(
            &mut self.words,
            &mut self.scratch,
            positions,
            system,
            cells,
            rcut,
        )?;

        let stream = device.stream();
        let reusable = self.state.as_ref().is_some_and(|s| {
            Arc::ptr_eq(&s.stream, stream) && s.list.natoms == natoms && s.systems == systems
        });
        if !reusable {
            // Frees the old buffers before the new ones are allocated.
            self.state = None;
            self.state = Some(State::new(device, &self.sel, slots, types, systems)?);
        }
        let Some(state) = self.state.as_mut() else {
            return Err(Error("neighbor search lost its device state".into()));
        };
        if state.types != types {
            stream
                .memcpy_htod(types, &mut state.list.types)
                .map_err(cuda_error("host-to-device copy"))?;
            state.types = types.to_vec();
        }
        if state.packed.len() < self.words.len() {
            return Err(Error(format!(
                "neighbor search input of {} words exceeds its bound of {}",
                self.words.len(),
                state.packed.len()
            )));
        }
        stream
            .memcpy_htod(&self.words, &mut state.packed)
            .map_err(cuda_error("host-to-device copy"))?;

        let scalars = [natoms as i32, ntypes as i32, systems as i32];
        let slots_i = slots as i32;
        // One center per block, of `self.warps` warps: blocks spread over the
        // multiprocessors, each with its own few double precision units.
        // (Measured with a warp per center: 4 centers per block cost 15% more
        // time for 88 atoms and the same for thousands.)
        let warps = self.warps;
        let capacity = capacity
            .unwrap_or(slots + slots / 2 + 32)
            .min(max_capacity(ntypes, warps));
        let block_bytes = block_doubles(capacity, ntypes, warps) * 8;
        let capacity_i = capacity as i32;
        {
            let list = &mut state.list;
            let inputs = [
                state.packed.device_ptr(stream),
                list.types.device_ptr(stream),
                state.tables.device_ptr(stream),
            ];
            let outputs = [
                list.index.device_ptr_mut(stream),
                list.delta.device_ptr_mut(stream),
                list.dist.device_ptr_mut(stream),
                list.count.device_ptr_mut(stream),
                list.min_distance.device_ptr_mut(stream),
                list.flags.device_ptr_mut(stream),
            ];
            // A null pointer when the caller wants no copy of the status.
            let mirror = mirror.as_mut().map(|m| m.device_ptr_mut(stream));
            let mirror_pointer = mirror.as_ref().map_or(0, |(pointer, _)| *pointer);
            let mut launch = stream.launch_builder(&state.function);
            for (p, _) in &inputs {
                launch.arg(p);
            }
            for s in &scalars {
                launch.arg(s);
            }
            launch.arg(&rcut);
            launch.arg(&slots_i);
            launch.arg(&capacity_i);
            for (p, _) in &outputs {
                launch.arg(p);
            }
            launch.arg(&mirror_pointer);
            let config = LaunchConfig {
                grid_dim: (natoms as u32, 1, 1),
                block_dim: (32 * warps as u32, 1, 1),
                shared_mem_bytes: block_bytes as u32,
            };
            unsafe { launch.launch(config) }
                .map(drop)
                .map_err(cuda_error("neighbor search launch"))?;
        }
        Ok(&state.list)
    }
}

impl NeighborList {
    pub fn build(
        device: &Device,
        positions: &[[f64; 3]],
        types: &[i32],
        cell: &Cell,
        rcut: f64,
        sel: &[usize],
    ) -> Result<Self> {
        Self::build_with_capacity(device, positions, types, cell, rcut, sel, None)
    }

    /// `build` with an explicit per-center shared buffer size; centers with
    /// more neighbors take the serial path. For tests of that path.
    #[doc(hidden)]
    pub fn build_with_capacity(
        device: &Device,
        positions: &[[f64; 3]],
        types: &[i32],
        cell: &Cell,
        rcut: f64,
        sel: &[usize],
        capacity: Option<usize>,
    ) -> Result<Self> {
        let mut search = NeighborSearch::new(rcut, sel)?;
        search.run_with_capacity(device, positions, types, cell, capacity)?;
        search
            .into_list()
            .ok_or_else(|| Error("neighbor search kept no list".into()))
    }

    /// Reads status bits; the OR over all centers says the same as each on
    /// its own. Errors on coincident atoms; otherwise returns whether some
    /// center had more neighbors of a type than `sel` (the farthest were
    /// dropped).
    pub fn status(bits: i32) -> Result<bool> {
        if bits & COINCIDENT != 0 {
            return Err(Error(
                "coincident or near-coincident atoms (distance <= 1e-12 angstrom)".into(),
            ));
        }
        Ok(bits & TRUNCATED != 0)
    }

    /// Waits for the search and reads its status with [`Self::status`].
    pub fn check(&self, device: &Device) -> Result<bool> {
        let flags = device.download(&self.flags)?;
        Self::status(flags.iter().fold(0, |all, flag| all | flag))
    }

    pub fn stats(&self, device: &Device) -> Result<NeighborStats> {
        let ntypes = self.sel.len();
        let counts = device.download(&self.count)?;
        let max_count = (0..ntypes)
            .map(|t| {
                counts
                    .iter()
                    .skip(t)
                    .step_by(ntypes)
                    .map(|&c| c as usize)
                    .max()
                    .unwrap_or(0)
            })
            .collect();
        let min_distance = device
            .download(&self.min_distance)?
            .into_iter()
            .fold(f64::INFINITY, f64::min);
        Ok(NeighborStats {
            max_count,
            min_distance,
        })
    }

    /// Same as `check`.
    pub fn truncated(&self, device: &Device) -> Result<bool> {
        self.check(device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stencil_reaches_cutoff_on_small_and_large_axes() {
        // 12.43 Å box, 6 Å cutoff: 2 cells, one neighbor cell each way.
        assert_eq!(grid_axis(12.43, 6.0), (2, 1));
        // Box narrower than the cutoff: one cell, stencil over two images.
        assert_eq!(grid_axis(4.0, 6.0), (1, 2));
        assert_eq!(grid_axis(49.7, 6.0), (8, 1));
        assert_eq!(grid_axis(1e6, 1.0).0, 512);
    }

    #[test]
    fn a_sparse_box_gets_few_cells() {
        // Three atoms in a 400 Å box: 80 cells per axis by the cutoff alone.
        let cell = Cell::Periodic([[400.0, 0.0, 0.0], [0.0, 400.0, 0.0], [0.0, 0.0, 400.0]]);
        let atoms = [[1.0, 2.0, 3.0], [399.0, 200.0, 5.0], [150.0, 150.0, 150.0]];
        let grid = Shape::new(&atoms, &[0, 1, 2], &cell, 5.0).unwrap();
        let total: usize = grid.cells.iter().product();
        assert!((27..=64).contains(&total), "{:?}", grid.cells);
        assert!(grid.cells.iter().all(|&c| c == grid.cells[0]));
        assert_eq!(grid.reach, [1, 1, 1]);
        // The packed input of any grid fits the bound it is allocated by.
        let (mut words, mut scratch) = (Vec::new(), Scratch::default());
        let packed = pack(&mut words, &mut scratch, &atoms, &[], &[cell], 5.0).unwrap();
        assert_eq!(packed, total);
        assert_eq!(words.len(), packed_words(3, total, 1));
        assert!(words.len() <= packed_words(3, cells_bound(3, 1), 1));
        assert!(cell_limit(3) <= cells_bound(3, 1));
        // A dense box keeps cells of the cutoff.
        let dense: Vec<[f64; 3]> = (0..4000)
            .map(|i| {
                [
                    i as f64 % 20.0,
                    (i / 20) as f64 % 20.0,
                    (i / 400) as f64 * 2.0,
                ]
            })
            .collect();
        let cell = Cell::Periodic([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]]);
        let all: Vec<u32> = (0..4000).collect();
        assert_eq!(
            Shape::new(&dense, &all, &cell, 5.0).unwrap().cells,
            [4, 4, 4]
        );
        // Wider cells never shorten the stencil below what the cutoff needs.
        // An axis with one cell has none to give up; its stencil stays as
        // wide as the cutoff needs.
        let (mut cells, mut reach) = ([1, 80, 80], [3, 1, 1]);
        coarsen(&mut cells, &mut reach, &[2.0, 400.0, 400.0], 5.0, 3);
        assert!(cells[0] == 1 && cells[1] == cells[2] && (6..=8).contains(&cells[1]));
        assert_eq!(reach, [3, 1, 1]);
        // Two cells on an axis can only become one.
        let (mut cells, mut reach) = ([2, 2, 500], [1, 1, 1]);
        coarsen(&mut cells, &mut reach, &[12.0, 12.0, 3000.0], 5.0, 1);
        assert!(
            cells.iter().product::<usize>() <= 64 && cells[2] >= 16,
            "{cells:?}"
        );
    }

    #[test]
    fn systems_are_packed_side_by_side() {
        // A periodic box and an open pair, their atoms interleaved.
        let cells = [
            Cell::Periodic([[12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 24.0]]),
            Cell::Open,
        ];
        let positions = [
            [1.0, 1.0, 1.0],
            [100.0, 0.0, 0.0],
            [7.0, 7.0, 13.0],
            [101.0, 0.0, 0.0],
            [1.0, 1.0, 47.0],
        ];
        let system = [0, 1, 0, 1, 0];
        let (mut words, mut scratch) = (Vec::new(), Scratch::default());
        let total = pack(&mut words, &mut scratch, &positions, &system, &cells, 6.0).unwrap();
        // 2 x 2 x 4 cells of the cutoff in the box, one around the pair.
        assert_eq!(total, 17);
        assert_eq!(words.len(), packed_words(5, total, 2));
        assert!(words.len() <= packed_words(5, cells_bound(5, 2), 2));
        let halves = entries(&words, 9 * 2 + 3 * 5);
        let header = |f: usize| &halves[HEADER * f..HEADER * (f + 1)];
        assert_eq!(header(0)[10..13], [2, 2, 4]);
        assert_eq!(f32::from_bits(header(0)[8]), 24.0);
        assert_eq!((header(0)[16], header(1)[16]), (1, 0));
        assert_eq!((header(0)[17], header(1)[17]), (0, 16));
        assert_eq!((header(0)[18], header(1)[18]), (1, 1));
        // The margin of each system follows its own coordinates.
        assert!(f32::from_bits(header(1)[9]) > f32::from_bits(header(0)[9]));
        // Atoms in cell order: those of the box, then the pair.
        let at_atoms = HEADER * 2 + 6 * 5;
        let mut boxed = halves[at_atoms..at_atoms + 3].to_vec();
        boxed.sort_unstable();
        assert_eq!(boxed, [0, 2, 4]);
        assert_eq!(halves[at_atoms + 3..at_atoms + 5], [1, 3]);
        assert_eq!(halves[at_atoms + 5..at_atoms + 10], [0, 1, 0, 1, 0]);
        // Cell starts run on from one system into the next.
        let start = &halves[at_atoms + 10..];
        assert_eq!(start.len(), total + 1);
        assert!(start.windows(2).all(|pair| pair[0] <= pair[1]));
        assert_eq!((start[0], start[16], start[17]), (0, 3, 5));
        // Positions stay at their atoms; those of the box are wrapped into it.
        let position = |atom: usize, axis: usize| f64::from_bits(words[18 + 3 * atom + axis]);
        assert_eq!(position(1, 0), 100.0);
        assert!((position(4, 2) - 23.0).abs() < 1e-12);
        assert_eq!(f64::from_bits(words[8]), 24.0);
        assert_eq!(f64::from_bits(words[9]), 0.0);
        // The float copy of an atom lies where the cell order puts it.
        let place = halves[at_atoms..at_atoms + 5]
            .iter()
            .position(|&a| a == 3)
            .unwrap();
        assert_eq!(f32::from_bits(halves[HEADER * 2 + 3 * place]), 101.0);
        // Every system needs an atom.
        assert!(pack(&mut words, &mut scratch, &positions, &[0; 5], &cells, 6.0).is_err());
    }

    #[test]
    fn the_scan_radius_covers_single_precision_rounding() {
        // The radius of the header of a system of one atom.
        let scan = |cell: Cell, atom: [f64; 3]| {
            let (mut words, mut scratch) = (Vec::new(), Scratch::default());
            pack(&mut words, &mut scratch, &[atom], &[], &[cell], 5.0).unwrap();
            f64::from(f32::from_bits(entries(&words, 9 + 3)[9])).sqrt()
        };
        let cell = Cell::Periodic([[400.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]]);
        let wide = scan(cell, [1.0, 1.0, 1.0]);
        // Coordinates and shifts reach 800 Å here: 32 epsilon of that.
        let margin = 32.0 * f64::from(f32::EPSILON) * (800.0 + 5.0);
        assert!(wide > 5.0 + margin && wide < 5.0 + 1.01 * margin, "{wide}");
        // In an open cell the coordinates themselves set the scale.
        let wide = scan(Cell::Open, [3.0e5, -1.0e5, 0.0]);
        assert!(wide > 5.0 + 1.1 && wide < 5.0 + 1.2, "{wide}");
        // Out of the range of floats, everything passes the scan.
        assert_eq!(scan_radius2(1e30, 3.0e5), f32::INFINITY);
    }

    #[test]
    fn the_largest_buffer_fits_the_shared_memory_of_a_block() {
        let fits =
            |capacity, ntypes, warps| block_doubles(capacity, ntypes, warps) * 8 <= SHARED_LIMIT;
        for warps in 1..=MAX_WARPS_PER_CENTER {
            for ntypes in 1..=40 {
                let capacity = max_capacity(ntypes, warps);
                assert!(
                    fits(capacity, ntypes, warps),
                    "{ntypes} types, {warps} warps"
                );
                // At most one neighbor short of the most that fit.
                assert!(
                    !fits(capacity + 2, ntypes, warps),
                    "{ntypes} types, {warps} warps"
                );
            }
        }
        // The block holds what the default capacity asks for at the selection
        // of the HAP models.
        assert!(max_capacity(5, WARPS_PER_CENTER) >= 212 + 212 / 2 + 32);
    }

    #[test]
    fn periodic_binning_wraps_into_the_cell() {
        let cell = Cell::Periodic([[10.0, 0.0, 0.0], [2.0, 10.0, 0.0], [0.0, 0.0, 10.0]]);
        let atoms = [[-1.0, 5.0, 25.0], [9.999, 0.0, 0.0]];
        let grid = Shape::new(&atoms, &[0, 1], &cell, 3.0).unwrap();
        let placed = atoms.map(|p| grid.place(&p));
        for (c, p) in &placed {
            assert!(c.iter().zip(&grid.cells).all(|(&c, &n)| c < n));
            assert!(p.iter().all(|v| v.is_finite()));
        }
        assert!((placed[0].1[2] - 5.0).abs() < 1e-12);
        assert!(inverse(&[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]]).is_err());
    }

    /// The four-byte entries of a packed input with `doubles` doubles, two
    /// to a word as they lie in memory.
    fn entries(words: &[u64], doubles: usize) -> Vec<u32> {
        words[doubles..]
            .iter()
            .flat_map(|word| {
                let bytes = word.to_ne_bytes();
                [0, 4].map(|at| u32::from_ne_bytes([0, 1, 2, 3].map(|b| bytes[at + b])))
            })
            .collect()
    }

    /// Packs one case with the single pass, on both of its paths and over
    /// stale words and scratch, and with the packing it replaced: the same
    /// words, or the same error. The number of four-byte entries, if packed.
    fn pack_both(
        scratch: &mut Scratch,
        positions: &[[f64; 3]],
        system: &[i32],
        cells: &[Cell],
        rcut: f64,
    ) -> Option<usize> {
        let (mut expected, mut halves) = (Vec::new(), Vec::new());
        let old = reference::pack(&mut expected, &mut halves, positions, system, cells, rcut);
        let new = plan(scratch, positions, system, cells, rcut);
        let (total, length) = match (old, new) {
            (Ok(old), Ok((total, length))) => {
                assert_eq!((old, length), (total, expected.len()));
                (total, length)
            }
            (Err(old), Err(new)) => {
                assert_eq!(old, new);
                return None;
            }
            (old, new) => panic!("reference {old:?}, single pass {new:?}"),
        };
        let differs = |words: &[u64]| words.iter().zip(&expected).position(|(a, b)| a != b);
        let mut words = vec![u64::MAX; length];
        fill_any(&mut words, scratch, positions, cells, rcut, total);
        assert_eq!(differs(&words), None, "portable path, {} words", length);
        words.fill(0x5a5a_5a5a_5a5a_5a5a);
        fill(&mut words, scratch, positions, cells, rcut, total);
        assert_eq!(differs(&words), None, "dispatched path, {} words", length);
        words.fill(u64::MAX);
        assert_eq!(
            pack(&mut words, scratch, positions, system, cells, rcut),
            Ok(total)
        );
        assert_eq!(differs(&words), None, "pack, {} words", length);
        Some(halves.len())
    }

    #[test]
    fn the_single_pass_packs_the_words_of_the_reference() {
        let mut scratch = Scratch::default();
        let mut parities = [0; 2];
        let mut count = |entries: Option<usize>| {
            if let Some(entries) = entries {
                parities[entries % 2] += 1;
            }
        };
        let skewed = Cell::Periodic([[10.0, 0.0, 0.0], [3.0, 9.0, 0.0], [-2.0, 1.5, 11.0]]);
        // On faces and corners, outside the cell, signed zeros, subnormals.
        let edges = [
            [0.0, 0.0, 0.0],
            [-0.0, -0.0, -0.0],
            [1e-300, -1e-300, 5e-324],
            [-5e-324, 0.0, -1e-300],
            [10.0, 0.0, 0.0],
            [13.0, 9.0, 0.0],
            [11.0, 10.5, 11.0],
            [-25.0, 40.0, -33.0],
            [
                9.999_999_999_999_998,
                8.999_999_999_999_998,
                10.999_999_999_999_998,
            ],
            [5.0, 4.5, 5.5],
            [1e5, -3e4, 2e5],
        ];
        for n in 1..=edges.len() {
            count(pack_both(&mut scratch, &edges[..n], &[], &[skewed], 4.0));
            count(pack_both(
                &mut scratch,
                &edges[..n],
                &[],
                &[Cell::Open],
                4.0,
            ));
        }
        // An axis far thinner than the cutoff: a reach past the packed images.
        let thin = Cell::Periodic([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 0.01]]);
        count(pack_both(&mut scratch, &edges, &[], &[thin], 6.0));
        // A sparse box, coarsened; atoms that coincide in an open cell.
        let sparse = Cell::Periodic([[400.0, 0.0, 0.0], [0.0, 400.0, 0.0], [0.0, 0.0, 400.0]]);
        count(pack_both(&mut scratch, &edges, &[], &[sparse], 5.0));
        count(pack_both(
            &mut scratch,
            &[[1.0, 2.0, 3.0]; 4],
            &[],
            &[Cell::Open],
            5.0,
        ));
        // Three systems, their atoms interleaved.
        let small = Cell::Periodic([[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 30.0]]);
        let cells = [skewed, Cell::Open, small];
        let system: Vec<i32> = (0..edges.len() as i32).map(|i| i % 3).collect();
        count(pack_both(&mut scratch, &edges, &system, &cells, 4.0));
        // A system without atoms and a singular cell fail alike.
        assert_eq!(pack_both(&mut scratch, &edges, &[0; 11], &cells, 4.0), None);
        let flat = Cell::Periodic([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
        assert_eq!(pack_both(&mut scratch, &edges, &[], &[flat], 4.0), None);
        // Random systems: skewed cells of 3 to 40 Å, open ones, atoms in and
        // around them at several scales.
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut uniform = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..400 {
            let systems = 1 + (uniform() * 3.0) as usize;
            let natoms = systems + (uniform() * 60.0) as usize;
            let scale = [1.0, 1.0, 10.0, 1e4][(uniform() * 4.0) as usize];
            let cells: Vec<Cell> = (0..systems)
                .map(|_| {
                    if uniform() < 0.25 {
                        return Cell::Open;
                    }
                    let length: [f64; 3] = std::array::from_fn(|_| 3.0 + 37.0 * uniform());
                    Cell::Periodic(std::array::from_fn(|a| {
                        std::array::from_fn(|b| {
                            if a == b {
                                length[a]
                            } else {
                                (uniform() - 0.5) * 0.6 * length[a]
                            }
                        })
                    }))
                })
                .collect();
            let positions: Vec<[f64; 3]> = (0..natoms)
                .map(|_| std::array::from_fn(|_| (uniform() * 50.0 - 15.0) * scale))
                .collect();
            // Every system gets an atom; the rest go anywhere.
            let system: Vec<i32> = (0..natoms)
                .map(|i| {
                    if i < systems {
                        i as i32
                    } else {
                        (uniform() * systems as f64) as i32
                    }
                })
                .collect();
            let rcut = 1.5 + 6.5 * uniform();
            count(pack_both(&mut scratch, &positions, &system, &cells, rcut));
        }
        // Inputs whose entries fill the last word and some that leave half.
        assert!(parities.iter().all(|&n| n > 20), "{parities:?}");
    }
}

#[cfg(test)]
mod reference;
