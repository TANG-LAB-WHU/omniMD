//! The packing of the neighbor search input before the single pass of
//! `super::pack` (868614f): a `Grid` binned for each system, the four-byte
//! entries put together in a vector of their own and then paired into
//! words. Kept for the tests, which hold the single pass to its words.
use super::{coarsen, grid_axis, inverse, Cell, HEADER};
use omnimd_ml_contract::{Error, Result};

/// Bins the atoms of every system and writes the kernel's input (see
/// `neighbor_search`) to `words`, with `halves` as scratch for its four-byte
/// entries. `system` names the system of each atom and is empty when there
/// is one. Returns the number of cells of all grids.
pub(super) fn pack(
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
