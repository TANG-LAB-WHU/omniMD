//! Dataset statistics with DeePMD-kit semantics (plan W2).
//!
//! - [`neighbor_stat`]: `dp neighbor-stat` — the smallest neighbor distance
//!   and the largest neighbor count per type within `rcut`.
//! - [`env_mat_stat`]: descriptor input statistics (`davg`, `dstd`), like
//!   DeePMD's `EnvMatStatSe` for se_e2_a: sums over every neighbor slot of
//!   every atom of a center type, empty slots included as zeros.
//! - [`energy_bias`]: per-type energy shift from a minimum-norm least-squares
//!   fit of frame energies to type counts (`compute_stats_from_redu`).
//!
//! Frames are processed on all cores; per-frame sums are merged in frame
//! order, so results do not depend on the thread count.
use crate::least_squares;
use crate::neighbor::Geometry;
use crate::Normalization;
use omnimd_ml_contract::{Error, Frame, Result};
use serde::Serialize;

fn type_ids(frame: &Frame, type_map: &[String]) -> Result<Vec<i32>> {
    frame.validate()?;
    frame
        .species
        .iter()
        .map(|name| {
            type_map
                .iter()
                .position(|t| t == name)
                .map(|i| i as i32)
                .ok_or_else(|| Error(format!("element {name:?} is not in the type map")))
        })
        .collect()
}

/// Runs `per_frame` on every frame using all cores and returns the results in
/// frame order.
fn map_frames<T: Send>(
    frames: &[&Frame],
    per_frame: impl Fn(&Frame) -> Result<T> + Sync,
) -> Result<Vec<T>> {
    let threads = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(frames.len().max(1));
    let chunk = frames.len().div_ceil(threads).max(1);
    let per_frame = &per_frame;
    std::thread::scope(|scope| {
        let workers: Vec<_> = frames
            .chunks(chunk)
            .map(|part| scope.spawn(move || part.iter().map(|f| per_frame(f)).collect::<Vec<_>>()))
            .collect();
        let mut results = Vec::with_capacity(frames.len());
        for worker in workers {
            let part = worker
                .join()
                .map_err(|_| Error("statistics worker panicked".into()))?;
            for item in part {
                results.push(item?);
            }
        }
        Ok(results)
    })
}

/// `dp neighbor-stat` for a set of frames.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NeighborStat {
    /// Smallest distance between two atoms within `rcut` (infinite when no
    /// frame has a pair that close).
    pub min_nbor_dist: f64,
    /// Largest number of neighbors of each type around any atom.
    pub max_nbor_size: Vec<usize>,
}

impl NeighborStat {
    /// DeePMD's `sel: "auto:factor"`: the largest count times `factor`,
    /// truncated and then rounded up to a multiple of 4.
    pub fn auto_sel(&self, factor: f64) -> Vec<usize> {
        self.max_nbor_size
            .iter()
            .map(|&count| ((count as f64 * factor) as usize).div_ceil(4) * 4)
            .collect()
    }
}

pub fn neighbor_stat(frames: &[&Frame], type_map: &[String], rcut: f64) -> Result<NeighborStat> {
    check_cutoff(rcut)?;
    let ntypes = type_map.len();
    let per_frame = map_frames(frames, |frame| {
        let types = type_ids(frame, type_map)?;
        let geometry = Geometry::new(frame, rcut)?;
        let mut nearest = f64::INFINITY;
        let mut largest = vec![0usize; ntypes];
        for center in 0..types.len() {
            let groups = geometry.all_neighbors(frame, &types, center, ntypes)?;
            for (most, group) in largest.iter_mut().zip(&groups) {
                *most = (*most).max(group.len());
                if let Some(first) = group.first() {
                    nearest = nearest.min(first.distance);
                }
            }
        }
        Ok((nearest, largest))
    })?;
    let mut stat = NeighborStat {
        min_nbor_dist: f64::INFINITY,
        max_nbor_size: vec![0; ntypes],
    };
    for (nearest, largest) in per_frame {
        stat.min_nbor_dist = stat.min_nbor_dist.min(nearest);
        for (most, count) in stat.max_nbor_size.iter_mut().zip(largest) {
            *most = (*most).max(count);
        }
    }
    Ok(stat)
}

fn check_cutoff(rcut: f64) -> Result<()> {
    if rcut.is_finite() && rcut > 0.0 {
        Ok(())
    } else {
        Err(Error(format!(
            "rcut must be finite and positive, got {rcut}"
        )))
    }
}

/// Count, sum and sum of squares of one statistic (DeePMD `StatItem`).
#[derive(Clone, Copy, Debug, Default)]
struct Moments {
    number: f64,
    sum: f64,
    squared: f64,
}

impl Moments {
    fn add(&mut self, other: &Moments) {
        self.number += other.number;
        self.sum += other.sum;
        self.squared += other.squared;
    }

    fn mean(&self) -> f64 {
        if self.number == 0.0 {
            0.0
        } else {
            self.sum / self.number
        }
    }

    /// Standard deviation with DeePMD's guards: 0.1 without data, and never
    /// below 0.01.
    fn std(&self) -> f64 {
        if self.number == 0.0 {
            return 0.1;
        }
        let mean = self.sum / self.number;
        let value = (self.squared / self.number - mean * mean).sqrt();
        if value.abs() < 1e-2 {
            1e-2
        } else {
            value
        }
    }
}

/// The smooth switch `s(r)·r`: 1 below `rmin`, 0 at `rmax`.
fn switch(r: f64, rmin: f64, rmax: f64) -> f64 {
    if r < rmin {
        return 1.0;
    }
    let u = (r - rmin) / (rmax - rmin);
    u * u * u * (-6.0 * u * u + 15.0 * u - 10.0) + 1.0
}

/// Descriptor input statistics for se_e2_a: one [`Normalization`] per center
/// type with `mean = [avg(s), 0, 0, 0]` and `std = [std(s), a, a, a]`, where
/// `s = switch(r)/r` and `a` is the pooled standard deviation of the three
/// angular components `s·d/r`. Each atom contributes `sum(sel)` slots, filled
/// by DeePMD-kit 3's selection (the nearest `sum(sel)` neighbors overall,
/// then at most `sel[t]` per type); the rest are zeros.
pub fn env_mat_stat(
    frames: &[&Frame],
    type_map: &[String],
    rcut: f64,
    rcut_smth: f64,
    sel: &[usize],
) -> Result<Vec<Normalization>> {
    check_cutoff(rcut)?;
    if !(rcut_smth.is_finite() && 0.0 <= rcut_smth && rcut_smth < rcut) {
        return Err(Error(format!(
            "rcut_smth must be in [0, rcut), got {rcut_smth}"
        )));
    }
    let ntypes = type_map.len();
    if sel.len() != ntypes {
        return Err(Error(format!(
            "sel needs one entry per type ({ntypes}), got {}",
            sel.len()
        )));
    }
    let slots: usize = sel.iter().sum();
    let per_frame = map_frames(frames, |frame| {
        let types = type_ids(frame, type_map)?;
        let geometry = Geometry::new(frame, rcut)?;
        // [center type] -> (radial, angular)
        let mut moments = vec![(Moments::default(), Moments::default()); ntypes];
        for (center, &kind) in types.iter().enumerate() {
            let (radial, angular) = &mut moments[kind as usize];
            radial.number += slots as f64;
            angular.number += 3.0 * slots as f64;
            for group in geometry.selected_neighbors(frame, &types, center, sel)? {
                for neighbor in &group {
                    let r = neighbor.distance;
                    let s = switch(r, rcut_smth, rcut) / r;
                    radial.sum += s;
                    radial.squared += s * s;
                    for d in neighbor.delta {
                        let a = s * d / r;
                        angular.sum += a;
                        angular.squared += a * a;
                    }
                }
            }
        }
        Ok(moments)
    })?;
    let mut total = vec![(Moments::default(), Moments::default()); ntypes];
    for moments in &per_frame {
        for ((radial, angular), (r, a)) in total.iter_mut().zip(moments) {
            radial.add(r);
            angular.add(a);
        }
    }
    Ok(total
        .iter()
        .map(|(radial, angular)| Normalization {
            mean: [radial.mean(), 0.0, 0.0, 0.0],
            std: [radial.std(), angular.std(), angular.std(), angular.std()],
        })
        .collect())
}

/// Per-type energy bias and the spread of what it leaves unexplained.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EnergyBias {
    /// eV per atom of each type.
    pub bias: Vec<f64>,
    /// Standard deviation over frames of `E - counts·bias` (eV per frame).
    pub std: f64,
    /// Rank of the frames × types count matrix.
    pub rank: usize,
}

/// Fits `E_frame ≈ Σ_t count_t·bias_t` over the frames. When the counts do
/// not determine every type (water: H is always 2·O), the minimum-norm
/// solution is returned, like `numpy.linalg.lstsq` in DeePMD.
pub fn energy_bias(frames: &[&Frame], energies: &[f64], type_map: &[String]) -> Result<EnergyBias> {
    if frames.is_empty() || frames.len() != energies.len() {
        return Err(Error(format!(
            "energy bias needs one energy per frame, got {} frames and {} energies",
            frames.len(),
            energies.len()
        )));
    }
    let mut counts = Vec::with_capacity(frames.len());
    for frame in frames {
        let mut row = vec![0.0; type_map.len()];
        for kind in type_ids(frame, type_map)? {
            row[kind as usize] += 1.0;
        }
        counts.push(row);
    }
    let solution = least_squares::min_norm(&counts, energies, None)?;
    let rest: Vec<f64> = counts
        .iter()
        .zip(energies)
        .map(|(row, e)| e - row.iter().zip(&solution.x).map(|(c, b)| c * b).sum::<f64>())
        .collect();
    let mean = rest.iter().sum::<f64>() / rest.len() as f64;
    let variance = rest.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rest.len() as f64;
    Ok(EnergyBias {
        bias: solution.x,
        std: variance.sqrt(),
        rank: solution.rank,
    })
}
