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
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
use omnimd_ml_contract::{Error, Result};

const SOURCE: &str = r#"
#define FULL_MASK 0xffffffffu

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
        int* index, double* delta, double* dist, int* count, double* min_dist, int* flags) {
    long long base = (long long)i * slots;
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
                if (r <= 1e-12) { atomicExch(&flags[0], 1); continue; }
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
        if (count[(long long)i * ntypes + t] > sel[t]) atomicExch(&flags[1], 1);
}

// One warp per center. The lanes scan each stencil cell together and append
// accepted neighbors to a per-warp shared buffer (ballot + popc), then each
// lane ranks its entries by (distance, atom, delta), overall and within
// their type, and writes those among the `slots` nearest overall and the
// sel[t] nearest of their type straight to their slots. Same result as
// `serial_center`, which takes over when the buffer would overflow.
extern "C" __global__ void neighbor_search(
        const double* pos, const int* types, const int* cell_of, const int* cell_start,
        const int* cell_atoms, const double* lattice, int natoms, int ntypes,
        int nx, int ny, int nz, int sx, int sy, int sz, int periodic, double rcut,
        const int* sel, const int* offset, int slots, int capacity,
        int* index, double* delta, double* dist, int* count, double* min_dist, int* flags) {
    extern __shared__ double smem[];
    int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int i = blockIdx.x * warps + warp;
    if (i >= natoms) return;
    int n[3] = {nx, ny, nz}, reach[3] = {sx, sy, sz};
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
    int ci[3] = {cell_of[3 * i], cell_of[3 * i + 1], cell_of[3 * i + 2]};
    double best = __longlong_as_double(0x7ff0000000000000LL);
    int total = 0;
    for (int ox = -sx; ox <= sx; ++ox)
    for (int oy = -sy; oy <= sy; ++oy)
    for (int oz = -sz; oz <= sz; ++oz) {
        int o[3] = {ox, oy, oz}, q[3], cell;
        if (!stencil_cell(ci, o, n, periodic, &cell, q)) continue;
        bool self_image = q[0] == 0 && q[1] == 0 && q[2] == 0;
        int end = cell_start[cell + 1];
        for (int k0 = cell_start[cell]; k0 < end; k0 += 32) {
            int k = k0 + lane, j = -1, t = 0;
            double r = 0.0;
            bool accept = false;
            if (k < end) {
                j = cell_atoms[k];
                if (!(j == i && self_image)) {
                    double d[3];
                    image_delta(pos, lattice, pi, j, q, d);
                    double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if (r2 < rcut * rcut) {
                        r = sqrt(r2);
                        if (r <= 1e-12) atomicExch(&flags[0], 1);
                        else { accept = true; t = types[j]; best = fmin(best, r); }
                    }
                }
            }
            unsigned mask = __ballot_sync(FULL_MASK, accept);
            if (accept) {
                int at = total + __popc(mask & ((1u << lane) - 1u));
                atomicAdd(&cnt[t], 1);
                if (at < capacity) {
                    cr[at] = r;
                    cj[at] = j;
                    cimg[at] = pack_image(q);
                    ctype[at] = t;
                }
            }
            total += __popc(mask);
        }
    }
    for (int offset_ = 16; offset_ > 0; offset_ >>= 1)
        best = fmin(best, __shfl_down_sync(FULL_MASK, best, offset_));
    __syncwarp();
    if (total > capacity) {
        if (lane == 0)
            serial_center(i, pos, types, cell_of, cell_start, cell_atoms, lattice, ntypes, n, reach,
                          periodic, rcut, sel, offset, slots, index, delta, dist, count, min_dist,
                          flags);
        return;
    }
    long long base = (long long)i * slots;
    for (int s = lane; s < slots; s += 32) {
        index[base + s] = -1;
        dist[base + s] = __longlong_as_double(0x7ff0000000000000LL);
        for (int a = 0; a < 3; ++a) delta[(base + s) * 3 + a] = 0.0;
    }
    for (int t = lane; t < ntypes; t += 32) {
        count[(long long)i * ntypes + t] = cnt[t];
        if (cnt[t] > sel[t]) atomicExch(&flags[1], 1);
    }
    if (lane == 0) min_dist[i] = best;
    __syncwarp();
    for (int e = lane; e < total; e += 32) {
        int t = ctype[e];
        if (sel[t] == 0) continue;
        double re = cr[e], de[3];
        int je = cj[e], qe[3], rank = 0, overall = 0;
        unpack_image(cimg[e], qe);
        image_delta(pos, lattice, pi, je, qe, de);
        for (int f = 0; f < total; ++f) {
            if (f == e) continue;
            double rf = cr[f];
            bool nearer = rf < re;
            if (rf == re) {
                double df[3];
                int qf[3];
                unpack_image(cimg[f], qf);
                image_delta(pos, lattice, pi, cj[f], qf, df);
                nearer = before(rf, cj[f], df, re, je, de);
            }
            if (nearer) {
                ++overall;
                if (ctype[f] == t) ++rank;
            }
        }
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
    /// `[natoms][slots]`, -1 for an empty slot.
    pub index: CudaSlice<i32>,
    /// `[natoms][slots][3]`, image-shifted `r_j - r_i`; zero in empty slots.
    pub delta: CudaSlice<f64>,
    /// `[natoms][ntypes]` neighbors within `rcut` before truncation to `sel`.
    pub count: CudaSlice<i32>,
    /// `[natoms]` nearest-neighbor distance.
    pub min_distance: CudaSlice<f64>,
    /// `[coincident atoms, truncated]`, read by `check` so `build` never syncs.
    flags: CudaSlice<i32>,
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
    let reach = (rcut * cells as f64 / width * (1.0 + 1e-12)).ceil() as usize;
    (cells, reach.max(1))
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
    let mut fractions = Vec::with_capacity(positions.len());
    let mut wrapped = Vec::with_capacity(positions.len());
    match cell {
        Cell::Periodic(lattice) => {
            let inv = inverse(lattice)?;
            for axis in 0..3 {
                // Distance between lattice planes = 1 / |reciprocal vector|.
                let norm = (0..3).map(|a| inv[a][axis].powi(2)).sum::<f64>().sqrt();
                (cells[axis], reach[axis]) = grid_axis(1.0 / norm, rcut);
            }
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
                let width = (hi[axis] - lo[axis]).max(rcut);
                (cells[axis], reach[axis]) = grid_axis(width, rcut);
            }
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
        let natoms = positions.len();
        let ntypes = sel.len();
        if natoms == 0 || types.len() != natoms || i32::try_from(natoms).is_err() {
            return Err(Error(format!(
                "neighbor search needs 1..=i32::MAX atoms with one type each; got {natoms} positions and {} types",
                types.len()
            )));
        }
        if ntypes == 0 || types.iter().any(|&t| t < 0 || t as usize >= ntypes) {
            return Err(Error(format!("atom types must be in 0..{ntypes}")));
        }
        if !(rcut.is_finite() && rcut > 0.0) {
            return Err(Error(format!(
                "rcut must be finite and positive, got {rcut}"
            )));
        }
        if positions.iter().flatten().any(|v| !v.is_finite()) {
            return Err(Error("nonfinite atom position".into()));
        }
        let slots: usize = sel.iter().sum();
        if slots == 0 || slots > 65536 {
            return Err(Error(format!("sum(sel) must be in 1..=65536, got {slots}")));
        }
        let grid = bin(positions, cell, rcut)?;
        let total_cells = grid.cells.iter().product::<usize>();
        let mut start = vec![0i32; total_cells + 1];
        let flat = |c: &[i32; 3]| {
            (c[0] as usize * grid.cells[1] + c[1] as usize) * grid.cells[2] + c[2] as usize
        };
        for c in &grid.cell_of {
            start[flat(c) + 1] += 1;
        }
        for k in 0..total_cells {
            start[k + 1] += start[k];
        }
        let mut cursor = start.clone();
        let mut members = vec![0i32; natoms];
        for (atom, c) in grid.cell_of.iter().enumerate() {
            let k = flat(c);
            members[cursor[k] as usize] = atom as i32;
            cursor[k] += 1;
        }
        let offsets: Vec<i32> = sel
            .iter()
            .scan(0, |acc, &s| {
                let o = *acc;
                *acc += s;
                Some(o as i32)
            })
            .collect();
        let lattice = match cell {
            Cell::Periodic(l) => l.iter().flatten().copied().collect(),
            Cell::Open => vec![0.0; 9],
        };

        let positions_d = device.upload(&grid.positions.concat())?;
        let types_d = device.upload(types)?;
        let cell_of_d = device.upload(&grid.cell_of.concat())?;
        let start_d = device.upload(&start)?;
        let members_d = device.upload(&members)?;
        let lattice_d = device.upload(&lattice)?;
        let sel_i: Vec<i32> = sel.iter().map(|&s| s as i32).collect();
        let sel_d = device.upload(&sel_i)?;
        let offsets_d = device.upload(&offsets)?;
        let mut index = device.zeros::<i32>(natoms * slots)?;
        let mut delta = device.zeros::<f64>(natoms * slots * 3)?;
        let mut dist = device.zeros::<f64>(natoms * slots)?;
        let mut count = device.zeros::<i32>(natoms * ntypes)?;
        let mut min_distance = device.zeros::<f64>(natoms)?;
        let mut flags = device.zeros::<i32>(2)?;

        let function = device.kernels().function(SOURCE, "neighbor_search")?;
        let stream = device.stream();
        let scalars = [
            natoms as i32,
            ntypes as i32,
            grid.cells[0] as i32,
            grid.cells[1] as i32,
            grid.cells[2] as i32,
            grid.reach[0] as i32,
            grid.reach[1] as i32,
            grid.reach[2] as i32,
            i32::from(matches!(cell, Cell::Periodic(_))),
        ];
        let slots_i = slots as i32;
        // Packed images need |q| < 512; past that every center goes serial.
        let packable = grid.reach.iter().all(|&r| r < 500);
        let max_capacity = (2 * (SHARED_LIMIT / 8) - ntypes - 1) / 5;
        let capacity = match (capacity, packable) {
            (_, false) => 0,
            (Some(c), true) => c,
            (None, true) => slots + slots / 2 + 32,
        }
        .min(max_capacity);
        let warp_bytes = warp_doubles(capacity, ntypes) * 8;
        let warps = (SHARED_LIMIT / warp_bytes).clamp(1, 4);
        let capacity_i = capacity as i32;
        {
            let pointers = [
                positions_d.device_ptr(stream),
                types_d.device_ptr(stream),
                cell_of_d.device_ptr(stream),
                start_d.device_ptr(stream),
                members_d.device_ptr(stream),
                lattice_d.device_ptr(stream),
            ];
            let tables = [sel_d.device_ptr(stream), offsets_d.device_ptr(stream)];
            let outputs = [
                index.device_ptr_mut(stream),
                delta.device_ptr_mut(stream),
                dist.device_ptr_mut(stream),
                count.device_ptr_mut(stream),
                min_distance.device_ptr_mut(stream),
                flags.device_ptr_mut(stream),
            ];
            let mut launch = stream.launch_builder(&function);
            for (p, _) in &pointers {
                launch.arg(p);
            }
            for s in &scalars {
                launch.arg(s);
            }
            launch.arg(&rcut);
            for (p, _) in &tables {
                launch.arg(p);
            }
            launch.arg(&slots_i);
            launch.arg(&capacity_i);
            for (p, _) in &outputs {
                launch.arg(p);
            }
            let config = LaunchConfig {
                grid_dim: (natoms.div_ceil(warps) as u32, 1, 1),
                block_dim: (32 * warps as u32, 1, 1),
                shared_mem_bytes: (warps * warp_bytes) as u32,
            };
            unsafe { launch.launch(config) }
                .map(drop)
                .map_err(cuda_error("neighbor search launch"))?;
        }
        Ok(Self {
            natoms,
            sel: sel.to_vec(),
            slots,
            index,
            delta,
            count,
            min_distance,
            flags,
        })
    }

    /// Waits for the search. Errors on coincident atoms; otherwise returns
    /// whether some center had more neighbors of a type than `sel` (the
    /// farthest were dropped).
    pub fn check(&self, device: &Device) -> Result<bool> {
        let flags = device.download(&self.flags)?;
        if flags[0] != 0 {
            return Err(Error(
                "coincident or near-coincident atoms (distance <= 1e-12 angstrom)".into(),
            ));
        }
        Ok(flags[1] != 0)
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
