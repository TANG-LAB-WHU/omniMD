use omnimd_ml_contract::{Error, Frame, Result};

#[derive(Debug, Clone)]
pub(crate) struct Neighbor {
    pub atom: usize,
    pub delta: [f64; 3],
    pub distance: f64,
    #[cfg(feature = "cuda")]
    pub image: [i64; 3],
}

pub(crate) struct Geometry {
    cutoff: f64,
    inverse: Option<[[f64; 3]; 3]>,
}

impl Geometry {
    pub fn new(frame: &Frame, cutoff: f64) -> Result<Self> {
        let inverse = if let Some(cell) = frame.cell {
            let scale = cell.iter().flatten().fold(0.0_f64, |s, v| s.max(v.abs()));
            let h = cell.map(|row| row.map(|v| v / scale));
            let mut adj = [[0.0; 3]; 3];
            adj[0] = [
                h[1][1] * h[2][2] - h[1][2] * h[2][1],
                h[0][2] * h[2][1] - h[0][1] * h[2][2],
                h[0][1] * h[1][2] - h[0][2] * h[1][1],
            ];
            adj[1] = [
                h[1][2] * h[2][0] - h[1][0] * h[2][2],
                h[0][0] * h[2][2] - h[0][2] * h[2][0],
                h[0][2] * h[1][0] - h[0][0] * h[1][2],
            ];
            adj[2] = [
                h[1][0] * h[2][1] - h[1][1] * h[2][0],
                h[0][1] * h[2][0] - h[0][0] * h[2][1],
                h[0][0] * h[1][1] - h[0][1] * h[1][0],
            ];
            let det = (0..3).map(|a| h[0][a] * adj[a][0]).sum::<f64>();
            let inv = adj.map(|row| row.map(|v| v / det / scale));
            if inv.iter().flatten().any(|v| !v.is_finite()) {
                return Err(Error("nonfinite inverse cell".into()));
            }
            Some(inv)
        } else {
            None
        };
        Ok(Self { cutoff, inverse })
    }

    /// Conservative arithmetic scale for a cached image list. This includes
    /// cancellation in unwrapped coordinates and skew-cell image translations.
    #[cfg(feature = "cuda")]
    pub fn cache_roundoff_scale(&self, frame: &Frame) -> f64 {
        // An extra angstrom covers the 0.4 A skin, allowed motion and margin.
        let delta_bound = 2.0
            * frame
                .positions
                .iter()
                .flatten()
                .fold(0.0_f64, |m, x| m.max(x.abs()))
            + 1.0;
        let mut scale = 1.0 + delta_bound + self.cutoff;
        if let (Some(inv), Some(cell)) = (self.inverse, frame.cell) {
            for b in 0..3 {
                let shift_bound = delta_bound * (0..3).map(|a| inv[a][b].abs()).sum::<f64>()
                    + (self.cutoff + 1.0) * inv[0][b].hypot(inv[1][b]).hypot(inv[2][b])
                    + 3.0;
                scale += shift_bound * cell[b].iter().fold(0.0_f64, |m, x| m.max(x.abs()));
            }
        }
        scale
    }

    pub fn neighbors(
        &self,
        frame: &Frame,
        types: &[i32],
        center: usize,
        sel: &[usize],
    ) -> Result<Vec<Vec<Neighbor>>> {
        let mut groups = vec![Vec::new(); sel.len()];
        let mut candidates = 0usize;
        for (atom, position) in frame.positions.iter().enumerate() {
            let delta = std::array::from_fn(|a| position[a] - frame.positions[center][a]);
            if delta.iter().any(|v| !v.is_finite()) {
                return Err(Error("coordinate difference overflow".into()));
            }
            let mut ranges = [(0i64, 0i64); 3];
            if let Some(inv) = self.inverse {
                for b in 0..3 {
                    let frac = (0..3).map(|a| delta[a] * inv[a][b]).sum::<f64>();
                    let bound = self.cutoff * inv[0][b].hypot(inv[1][b]).hypot(inv[2][b]);
                    // A sphere in Cartesian space implies this interval on each
                    // reciprocal coordinate. The extra integer protects endpoints.
                    let lo = (-frac - bound).floor() - 1.0;
                    let hi = (-frac + bound).ceil() + 1.0;
                    if !lo.is_finite() || !hi.is_finite() || lo.abs().max(hi.abs()) > 4.0e15 {
                        return Err(Error("periodic image range is not representable".into()));
                    }
                    ranges[b] = (lo as i64, hi as i64);
                }
            }
            let count = ranges
                .iter()
                .try_fold(1usize, |p, (lo, hi)| {
                    usize::try_from(hi - lo + 1)
                        .ok()
                        .and_then(|n| p.checked_mul(n))
                })
                .ok_or_else(|| Error("periodic image count overflow".into()))?;
            candidates = candidates
                .checked_add(count)
                .ok_or_else(|| Error("neighbor candidate count overflow".into()))?;
            if candidates > 2_000_000 {
                return Err(Error(format!("atom {center}: neighbor search exceeds 2000000 candidates; cell/cutoff too costly")));
            }
            for a in ranges[0].0..=ranges[0].1 {
                for b in ranges[1].0..=ranges[1].1 {
                    for c in ranges[2].0..=ranges[2].1 {
                        if atom == center && a == 0 && b == 0 && c == 0 {
                            continue;
                        }
                        let mut image = delta;
                        if let Some(cell) = frame.cell {
                            for k in 0..3 {
                                image[k] += a as f64 * cell[0][k]
                                    + b as f64 * cell[1][k]
                                    + c as f64 * cell[2][k];
                            }
                        }
                        let distance = image[0].hypot(image[1]).hypot(image[2]);
                        if !distance.is_finite() {
                            return Err(Error("nonfinite image distance".into()));
                        }
                        if distance >= self.cutoff {
                            continue;
                        }
                        if distance <= 1e-12 {
                            return Err(Error(format!("coincident/near-coincident atoms {center} and {atom} (distance <= 1e-12 angstrom)")));
                        }
                        let kind = types[atom] as usize;
                        groups[kind].push(Neighbor {
                            atom,
                            delta: image,
                            distance,
                            #[cfg(feature = "cuda")]
                            image: [a, b, c],
                        });
                        if groups[kind].len() > sel[kind] {
                            return Err(Error(format!("neighbor capacity overflow at atom {center}, type {}: more than sel={}",
                                frame.species[atom], sel[kind])));
                        }
                    }
                }
            }
        }
        for group in &mut groups {
            group.sort_by(|a, b| {
                a.distance
                    .total_cmp(&b.distance)
                    .then(a.atom.cmp(&b.atom))
                    .then_with(|| a.delta[0].total_cmp(&b.delta[0]))
                    .then_with(|| a.delta[1].total_cmp(&b.delta[1]))
                    .then_with(|| a.delta[2].total_cmp(&b.delta[2]))
            });
        }
        Ok(groups)
    }
}
