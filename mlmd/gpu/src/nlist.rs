//! Cell-list neighbor search with DeePMD semantics (plan I-1).
//!
//! Atoms are binned on the host (counting sort over a grid whose cells are at
//! least `rcut` wide); one GPU warp per center then scans the stencil of
//! cells, including periodic images, into shared memory and selects like
//! DeePMD-kit 3 (`build_neighbor_list`): the `sum(sel)` nearest neighbors of
//! any type, then at most `sel[type]` of each type, ordered by (distance,
//! atom, delta). Centers with more neighbors than the shared buffer holds
//! fall back to one thread with insertion into the slots, which gives the
//! same result. Neighbors
//! beyond `sel` are dropped like DeePMD's `format_nlist`, and the full per-type
//! counts and the minimum distance are kept for `dp neighbor-stat`.
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
// neighbors than the shared buffer of the warp path holds.
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

// One warp per center, in three steps. The lanes first scan each stencil
// cell together in single precision and append every atom within the cutoff
// plus a margin (which covers the rounding of that arithmetic, so no
// neighbor is lost) to a per-warp shared buffer (ballot + popc). The few
// candidates are then measured in double precision, which decides exactly as
// a scan in double precision would. Last, each lane ranks its neighbors by
// (distance, atom, delta), overall and within their type, and writes those
// among the `slots` nearest overall and the sel[t] nearest of their type
// straight to their slots. Same result as `serial_center`, which takes over
// when the buffer would overflow.
//
// Double precision is scarce on the device (about 1/64 of the single
// precision rate), so it is kept to the candidates, and distances are ranked
// by their bit patterns: positive doubles order like integers.
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
    int i = blockIdx.x * warps + warp;
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
    // Per warp: r[capacity] doubles, then j, packed image, type [capacity]
    // ints, then the per-type counts.
    int per_warp = capacity + (3 * capacity + ntypes + 1) / 2;
    double* cr = smem + (long long)warp * per_warp;
    int* cj = (int*)(cr + capacity);
    int* cimg = cj + capacity;
    int* ctype = cimg + capacity;
    int* cnt = ctype + capacity;
    for (int t = lane; t < ntypes; t += 32) cnt[t] = 0;
    __syncwarp();
    const double* pi = pos + 3 * (long long)i;
    float pi_f[3] = {(float)pi[0], (float)pi[1], (float)pi[2]};
    int ci[3] = {cell_of[3 * i], cell_of[3 * i + 1], cell_of[3 * i + 2]};
    int total = 0;
    for (int ox = -sx; ox <= sx; ++ox)
    for (int oy = -sy; oy <= sy; ++oy)
    for (int oz = -sz; oz <= sz; ++oz) {
        int o[3] = {ox, oy, oz}, q[3], cell;
        if (!stencil_cell(ci, o, n, periodic, &cell, q)) continue;
        bool self_image = q[0] == 0 && q[1] == 0 && q[2] == 0;
        int image = pack_image(q);
        float shift[3];
        for (int a = 0; a < 3; ++a)
            shift[a] = q[0] * lattice_f[a] + q[1] * lattice_f[3 + a] + q[2] * lattice_f[6 + a]
                       - pi_f[a];
        int end = cell_start[cell + 1];
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
            if (near) {
                int at = total + __popc(mask & ((1u << lane) - 1u));
                if (at < room) {
                    cj[at] = j;
                    cimg[at] = image;
                }
            }
            total += __popc(mask);
        }
    }
    __syncwarp();
    if (total > room) {
        if (lane == 0)
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
    for (int e = lane; e < total; e += 32) {
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
    bool coincident = __any_sync(FULL_MASK, touching);
    __syncwarp();
    long long base = (long long)i * slots;
    for (int s = lane; s < slots; s += 32) {
        index[base + s] = -1;
        dist[base + s] = infinity;
        for (int a = 0; a < 3; ++a) delta[(base + s) * 3 + a] = 0.0;
    }
    bool over = false;
    for (int t = lane; t < ntypes; t += 32) {
        count[(long long)i * ntypes + t] = cnt[t];
        over |= cnt[t] > sel[t];
    }
    bool truncated = __any_sync(FULL_MASK, over);
    if (lane == 0) {
        int status = (coincident ? COINCIDENT : 0) | (truncated ? TRUNCATED : 0);
        min_dist[i] = best;
        flags[i] = status;
        if (mirror) mirror[i] = status;
    }
    __syncwarp();
    // Distances as integers: for positive doubles the order is the same.
    // Candidates at exactly the distance of `e` are only counted here; the
    // rare ties are decided afterwards, in a function of their own, so that
    // their double precision arithmetic stays out of this loop (inlined, the
    // compiler runs it for every pair and discards the result).
    const long long* key = (const long long*)cr;
    for (int e = lane; e < total; e += 32) {
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

/// Simulation cell: rows are lattice vectors (脜), as in `Frame::cell`.
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

/// Status bit of a center with another atom within 1e-12 脜.
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
    /// Host copy of the packed kernel input, reused between calls, and its
    /// four-byte entries while they are put together.
    words: Vec<u64>,
    halves: Vec<u32>,
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

/// Bins the atoms of every system and writes the kernel's input (see
/// `neighbor_search`) to `words`, with `halves` as scratch for its four-byte
/// entries. `system` names the system of each atom and is empty when there
/// is one. Returns the number of cells of all grids.
fn pack(
    words: &mut Vec<u64>,
    halves: &mut Vec<u32>,
    positions: &[[f64; 3]],
    system: &[i32],
    cells: &[Cell],
    rcut: f64,
) -> Result<usize> {
    let (natoms, systems) = (positions.len(), cells.len());
    // The atoms of system `f`, in their order: `first[f]..first[f + 1]` of
    // `atoms`.
    let mut first = vec![0usize; systems + 1];
    let mut atoms = vec![0i32; natoms];
    if system.is_empty() {
        first[1..].fill(natoms);
        for (entry, atom) in atoms.iter_mut().zip(0..) {
            *entry = atom;
        }
    } else {
        for &of in system {
            first[of as usize + 1] += 1;
        }
        for f in 0..systems {
            first[f + 1] += first[f];
        }
        let mut cursor = first.clone();
        for (atom, &of) in system.iter().enumerate() {
            atoms[cursor[of as usize]] = atom as i32;
            cursor[of as usize] += 1;
        }
    }
    let at_positions = 9 * systems;
    words.clear();
    words.resize(at_positions + 3 * natoms, 0);
    let at_floats = HEADER * systems;
    let at_cell_of = at_floats + 3 * natoms;
    let at_atoms = at_cell_of + 3 * natoms;
    let at_system = at_atoms + natoms;
    halves.clear();
    halves.resize(at_system + natoms, 0);
    let mut local = Vec::new();
    let mut total_cells = 0usize;
    for (f, cell) in cells.iter().enumerate() {
        let mine = &atoms[first[f]..first[f + 1]];
        if mine.is_empty() {
            return Err(Error(format!(
                "system {f} of the neighbor search has no atoms"
            )));
        }
        local.clear();
        local.extend(mine.iter().map(|&atom| positions[atom as usize]));
        let grid = bin(&local, cell, rcut)?;
        let count = grid.cells.iter().product::<usize>();
        let flat = |c: &[i32; 3]| {
            (c[0] as usize * grid.cells[1] + c[1] as usize) * grid.cells[2] + c[2] as usize
        };
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
        for (entry, value) in words[9 * f..9 * f + 9].iter_mut().zip(lattice) {
            *entry = value.to_bits();
        }
        let header = &mut halves[HEADER * f..HEADER * (f + 1)];
        for (entry, value) in header.iter_mut().zip(lattice) {
            *entry = (value as f32).to_bits();
        }
        header[9] = scan_radius2(rcut, &lattice, &grid, periodic).to_bits();
        for axis in 0..3 {
            header[10 + axis] = grid.cells[axis] as u32;
            header[13 + axis] = grid.reach[axis] as u32;
        }
        header[16] = u32::from(periodic);
        header[17] = total_cells as u32;
        // Packed images need |q| < 512; past that the centers go serial.
        header[18] = u32::from(grid.reach.iter().all(|&r| r < 500));
        // Counting sort of the system's atoms by cell. The first entry of
        // each cell is a place in the list of all atoms in cell order.
        let mut cursor = vec![0usize; count + 1];
        for c in &grid.cell_of {
            cursor[flat(c) + 1] += 1;
        }
        for k in 0..count {
            cursor[k + 1] += cursor[k];
        }
        halves.extend(cursor[..count].iter().map(|&k| (first[f] + k) as u32));
        for ((&atom, c), position) in mine.iter().zip(&grid.cell_of).zip(&grid.positions) {
            let atom = atom as usize;
            let k = first[f] + cursor[flat(c)];
            cursor[flat(c)] += 1;
            for axis in 0..3 {
                words[at_positions + 3 * atom + axis] = position[axis].to_bits();
                halves[at_floats + 3 * k + axis] = (position[axis] as f32).to_bits();
                halves[at_cell_of + 3 * atom + axis] = c[axis] as u32;
            }
            halves[at_atoms + k] = atom as u32;
            halves[at_system + atom] = f as u32;
        }
        total_cells += count;
    }
    halves.push(natoms as u32);
    // Two to a word, as they lie in memory.
    words.extend(halves.chunks(2).map(|pair| {
        let mut bytes = [0; 8];
        bytes[..4].copy_from_slice(&pair[0].to_ne_bytes());
        if let Some(high) = pair.get(1) {
            bytes[4..].copy_from_slice(&high.to_ne_bytes());
        }
        u64::from_ne_bytes(bytes)
    }));
    Ok(total_cells)
}

/// Square of the radius of the kernel's single-precision scan: the cutoff
/// plus what that arithmetic can misplace a distance by, so the scan keeps
/// every neighbor (and a few atoms just outside, which the pass in double
/// precision drops).
///
/// With coordinates and image shifts of magnitude at most `m`, rounding
/// positions and lattice to single precision and the sums that form a
/// separation move each of its components by less than 14 units of
/// roundoff of `m` (7 epsilon), the distance by less than 13 epsilon `m`,
/// and the squared sum adds under one epsilon of the cutoff. 32 epsilon of
/// `m + rcut` leaves more than a factor two.
fn scan_radius2(rcut: f64, lattice: &[f64; 9], grid: &Grid, periodic: bool) -> f32 {
    let magnitude = if periodic {
        (0..3)
            .map(|a| {
                (0..3)
                    .map(|b| (grid.reach[b] + 1) as f64 * lattice[3 * b + a].abs())
                    .sum::<f64>()
            })
            .fold(0.0, f64::max)
    } else {
        grid.positions
            .iter()
            .flatten()
            .fold(0.0, |m: f64, v| m.max(v.abs()))
    };
    let wide = rcut + 32.0 * f64::from(f32::EPSILON) * (magnitude + rcut);
    let square = (wide * wide) as f32;
    // One step up, so rounding the square cannot shrink it.
    if square.is_finite() {
        f32::from_bits(square.to_bits() + 1)
    } else {
        f32::INFINITY
    }
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

struct Grid {
    cells: [usize; 3],
    reach: [usize; 3],
    cell_of: Vec<[i32; 3]>,
    positions: Vec<[f64; 3]>,
}

fn bin(positions: &[[f64; 3]], cell: &Cell, rcut: f64) -> Result<Grid> {
    let mut cells = [1; 3];
    let mut reach = [1; 3];
    let mut widths = [0.0; 3];
    let mut fractions = Vec::with_capacity(positions.len());
    let mut wrapped = Vec::with_capacity(positions.len());
    match cell {
        Cell::Periodic(lattice) => {
            let inv = inverse(lattice)?;
            for axis in 0..3 {
                // Distance between lattice planes = 1 / |reciprocal vector|.
                let norm = (0..3).map(|a| inv[a][axis].powi(2)).sum::<f64>().sqrt();
                widths[axis] = 1.0 / norm;
                (cells[axis], reach[axis]) = grid_axis(widths[axis], rcut);
            }
            coarsen(&mut cells, &mut reach, &widths, rcut, positions.len());
            for p in positions {
                let mut f: [f64; 3] =
                    std::array::from_fn(|b| (0..3).map(|a| p[a] * inv[a][b]).sum());
                for v in &mut f {
                    *v -= v.floor();
                    if *v >= 1.0 {
                        *v = 0.0;
                    }
                }
                wrapped.push(std::array::from_fn(|a| {
                    (0..3).map(|b| f[b] * lattice[b][a]).sum()
                }));
                fractions.push(f);
            }
        }
        Cell::Open => {
            let lo: [f64; 3] = std::array::from_fn(|a| {
                positions.iter().map(|p| p[a]).fold(f64::INFINITY, f64::min)
            });
            let hi: [f64; 3] = std::array::from_fn(|a| {
                positions
                    .iter()
                    .map(|p| p[a])
                    .fold(f64::NEG_INFINITY, f64::max)
            });
            for axis in 0..3 {
                widths[axis] = (hi[axis] - lo[axis]).max(rcut);
                (cells[axis], reach[axis]) = grid_axis(widths[axis], rcut);
            }
            coarsen(&mut cells, &mut reach, &widths, rcut, positions.len());
            for p in positions {
                fractions.push(std::array::from_fn(|a| {
                    ((p[a] - lo[a]) / (hi[a] - lo[a]).max(rcut)).min(1.0 - f64::EPSILON)
                }));
                wrapped.push(*p);
            }
        }
    }
    let cell_of = fractions
        .iter()
        .map(|f| {
            std::array::from_fn(|a| ((f[a] * cells[a] as f64) as usize).min(cells[a] - 1) as i32)
        })
        .collect();
    Ok(Grid {
        cells,
        reach,
        cell_of,
        positions: wrapped,
    })
}

/// Shared memory one block may use without opting in to more.
const SHARED_LIMIT: usize = 48 * 1024;

/// Doubles of shared memory per warp for a buffer of `capacity` neighbors.
fn warp_doubles(capacity: usize, ntypes: usize) -> usize {
    capacity + (3 * capacity + ntypes + 1) / 2
}

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
            halves: Vec::new(),
            state: None,
        })
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
            &mut self.halves,
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
        let max_capacity = (2 * (SHARED_LIMIT / 8) - ntypes - 1) / 5;
        let capacity = capacity.unwrap_or(slots + slots / 2 + 32).min(max_capacity);
        let warp_bytes = warp_doubles(capacity, ntypes) * 8;
        // One center per block: blocks spread over the multiprocessors, each
        // with its own few double precision units. (Measured: 4 centers per
        // block cost 15% more time for 88 atoms and the same for thousands.)
        let warps = 1;
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
                grid_dim: (natoms.div_ceil(warps) as u32, 1, 1),
                block_dim: (32 * warps as u32, 1, 1),
                shared_mem_bytes: (warps * warp_bytes) as u32,
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
        // 12.43 脜 box, 6 脜 cutoff: 2 cells, one neighbor cell each way.
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
        let grid = bin(&atoms, &cell, 5.0).unwrap();
        let total: usize = grid.cells.iter().product();
        assert!((27..=64).contains(&total), "{:?}", grid.cells);
        assert!(grid.cells.iter().all(|&c| c == grid.cells[0]));
        assert_eq!(grid.reach, [1, 1, 1]);
        // The packed input of any grid fits the bound it is allocated by.
        let (mut words, mut halves) = (Vec::new(), Vec::new());
        let packed = pack(&mut words, &mut halves, &atoms, &[], &[cell], 5.0).unwrap();
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
        assert_eq!(bin(&dense, &cell, 5.0).unwrap().cells, [4, 4, 4]);
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
        let (mut words, mut halves) = (Vec::new(), Vec::new());
        let total = pack(&mut words, &mut halves, &positions, &system, &cells, 6.0).unwrap();
        // 2 x 2 x 4 cells of the cutoff in the box, one around the pair.
        assert_eq!(total, 17);
        assert_eq!(words.len(), packed_words(5, total, 2));
        assert!(words.len() <= packed_words(5, cells_bound(5, 2), 2));
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
        let place = halves[at_atoms..at_atoms + 5].iter().position(|&a| a == 3).unwrap();
        assert_eq!(f32::from_bits(halves[HEADER * 2 + 3 * place]), 101.0);
        // Every system needs an atom.
        assert!(pack(&mut words, &mut halves, &positions, &[0; 5], &cells, 6.0).is_err());
    }

    #[test]
    fn the_scan_radius_covers_single_precision_rounding() {
        let lattice = [400.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 30.0];
        let cell = Cell::Periodic([[400.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]]);
        let grid = bin(&[[1.0, 1.0, 1.0]], &cell, 5.0).unwrap();
        let wide = f64::from(scan_radius2(5.0, &lattice, &grid, true)).sqrt();
        // Coordinates and shifts reach 800 Å here: 32 epsilon of that.
        let margin = 32.0 * f64::from(f32::EPSILON) * (800.0 + 5.0);
        assert!(wide > 5.0 + margin && wide < 5.0 + 1.01 * margin, "{wide}");
        // In an open cell the coordinates themselves set the scale.
        let far = bin(&[[3.0e5, -1.0e5, 0.0]], &Cell::Open, 5.0).unwrap();
        let wide = f64::from(scan_radius2(5.0, &[0.0; 9], &far, false)).sqrt();
        assert!(wide > 5.0 + 1.1 && wide < 5.0 + 1.2, "{wide}");
        // Out of the range of floats, everything passes the scan.
        assert_eq!(scan_radius2(1e30, &[0.0; 9], &far, false), f32::INFINITY);
    }

    #[test]
    fn periodic_binning_wraps_into_the_cell() {
        let cell = Cell::Periodic([[10.0, 0.0, 0.0], [2.0, 10.0, 0.0], [0.0, 0.0, 10.0]]);
        let grid = bin(&[[-1.0, 5.0, 25.0], [9.999, 0.0, 0.0]], &cell, 3.0).unwrap();
        for (p, c) in grid.positions.iter().zip(&grid.cell_of) {
            assert!(c
                .iter()
                .zip(&grid.cells)
                .all(|(&c, &n)| c >= 0 && (c as usize) < n));
            assert!(p.iter().all(|v| v.is_finite()));
        }
        assert!((grid.positions[0][2] - 5.0).abs() < 1e-12);
        assert!(inverse(&[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]]).is_err());
    }
}
