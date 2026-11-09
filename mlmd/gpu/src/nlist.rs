//! Cell-list neighbor search with DeePMD semantics (plan I-1).
//!
//! Atoms are binned on the host (counting sort over a grid whose cells are at
//! least `rcut` wide); one GPU thread per center then scans the stencil of
//! cells, including periodic images, and keeps for each neighbor type the
//! `sel[type]` nearest neighbors ordered by (distance, atom, delta). Neighbors
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
__device__ bool before(double r, int j, const double* d, double r2, int j2, const double* d2) {
    if (r != r2) return r < r2;
    if (j != j2) return j < j2;
    for (int a = 0; a < 3; ++a) if (d[a] != d2[a]) return d[a] < d2[a];
    return false;
}

__device__ int floor_div(int a, int n) { return a >= 0 ? a / n : -((-a + n - 1) / n); }

extern "C" __global__ void neighbor_search(
        const double* pos, const int* types, const int* cell_of, const int* cell_start,
        const int* cell_atoms, const double* lattice, int natoms, int ntypes,
        int nx, int ny, int nz, int sx, int sy, int sz, int periodic, double rcut,
        const int* sel, const int* offset, int slots,
        int* index, double* delta, double* dist, int* count, double* min_dist, int* flags) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;
    long long base = (long long)i * slots;
    for (int s = 0; s < slots; ++s) {
        index[base + s] = -1;
        dist[base + s] = __longlong_as_double(0x7ff0000000000000LL);
        for (int a = 0; a < 3; ++a) delta[(base + s) * 3 + a] = 0.0;
    }
    for (int t = 0; t < ntypes; ++t) count[(long long)i * ntypes + t] = 0;
    double best = __longlong_as_double(0x7ff0000000000000LL);
    const double* pi = pos + 3 * (long long)i;
    int ci[3] = {cell_of[3 * i], cell_of[3 * i + 1], cell_of[3 * i + 2]};
    int n[3] = {nx, ny, nz};
    for (int ox = -sx; ox <= sx; ++ox)
    for (int oy = -sy; oy <= sy; ++oy)
    for (int oz = -sz; oz <= sz; ++oz) {
        int o[3] = {ox, oy, oz}, c[3], q[3];
        bool skip = false;
        for (int a = 0; a < 3; ++a) {
            int raw = ci[a] + o[a];
            if (periodic) {
                q[a] = floor_div(raw, n[a]);
                c[a] = raw - q[a] * n[a];
            } else {
                q[a] = 0;
                c[a] = raw;
                if (raw < 0 || raw >= n[a]) skip = true;
            }
        }
        if (skip) continue;
        double shift[3];
        for (int a = 0; a < 3; ++a)
            shift[a] = q[0] * lattice[a] + q[1] * lattice[3 + a] + q[2] * lattice[6 + a];
        int cell = (c[0] * ny + c[1]) * nz + c[2];
        for (int k = cell_start[cell]; k < cell_start[cell + 1]; ++k) {
            int j = cell_atoms[k];
            if (j == i && q[0] == 0 && q[1] == 0 && q[2] == 0) continue;
            double d[3];
            for (int a = 0; a < 3; ++a) d[a] = pos[3 * (long long)j + a] + shift[a] - pi[a];
            double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if (!(r2 < rcut * rcut)) continue;
            double r = sqrt(r2);
            if (r <= 1e-12) { atomicExch(&flags[0], 1); continue; }
            int t = types[j];
            count[(long long)i * ntypes + t] += 1;
            if (r < best) best = r;
            if (sel[t] == 0) continue;
            long long first = base + offset[t];
            int filled = min(count[(long long)i * ntypes + t], sel[t]);
            int p = filled - 1;
            // Full segment: the new neighbor must beat the current last entry.
            if (count[(long long)i * ntypes + t] > sel[t]) {
                long long last = first + sel[t] - 1;
                if (!before(r, j, d, dist[last], index[last], delta + 3 * last)) continue;
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
    min_dist[i] = best;
    for (int t = 0; t < ntypes; ++t)
        if (count[(long long)i * ntypes + t] > sel[t]) atomicExch(&flags[1], 1);
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

impl NeighborList {
    pub fn build(
        device: &Device,
        positions: &[[f64; 3]],
        types: &[i32],
        cell: &Cell,
        rcut: f64,
        sel: &[usize],
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
            for (p, _) in &outputs {
                launch.arg(p);
            }
            let config = LaunchConfig {
                grid_dim: (natoms.div_ceil(128) as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
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
