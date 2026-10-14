//! Conservative Verlet image-membership cache; no cached energies or environments.
use crate::neighbor::{Geometry, Neighbor};
use omnimd_ml_contract::{Error, Frame, Result};
#[derive(Clone)]
struct Candidate {
    atom: usize,
    image: [i64; 3],
}
pub(super) struct NeighborCache {
    reference: Option<Frame>,
    candidates: Option<Vec<Vec<Candidate>>>,
    skin: f64,
    pub rebuilds: usize,
    pub candidate_count: usize,
}
impl NeighborCache {
    pub fn new() -> Self {
        Self {
            reference: None,
            candidates: None,
            skin: 0.4,
            rebuilds: 0,
            candidate_count: 0,
        }
    }
    pub fn update(&mut self, frame: &Frame, types: &[i32], cutoff: f64, sel: &[usize]) -> bool {
        let valid = self.reference.as_ref().is_some_and(|old| {
            old.cell == frame.cell
                && old.species == frame.species
                && old.positions.len() == frame.positions.len()
                && old.positions.iter().zip(&frame.positions).all(|(a, b)| {
                    (a[0] - b[0]).hypot(a[1] - b[1]).hypot(a[2] - b[2]) < self.skin * 0.5
                })
        });
        if valid {
            return false;
        }
        self.rebuilds += 1;
        self.candidates = None;
        self.candidate_count = 0;
        // Enlarged-list resource errors fall back to the original exact search.
        // Thus caching cannot reject a frame the original CUDA geometry accepts.
        if let Ok(candidates) = self.build(frame, types, cutoff, sel) {
            self.candidate_count = candidates.iter().map(Vec::len).sum();
            self.candidates = Some(candidates);
        }
        self.reference = Some(frame.clone());
        true
    }
    fn build(
        &self,
        frame: &Frame,
        types: &[i32],
        cutoff: f64,
        sel: &[usize],
    ) -> Result<Vec<Vec<Candidate>>> {
        // Account for coordinate subtraction and cancellation of lattice shifts,
        // not just the rounded rc+skin endpoint. Large/unrepresentable scales
        // disable caching and keep the original exact-search behavior.
        let base = Geometry::new(frame, cutoff + self.skin)?;
        let margin = 64.0 * f64::EPSILON * base.cache_roundoff_scale(frame);
        if !margin.is_finite() || margin > self.skin * 0.001 {
            return Err(Error(
                "coordinate/image scale too large for neighbor cache".into(),
            ));
        }
        let radius = cutoff + self.skin + margin;
        if !radius.is_finite() {
            return Err(Error("cache radius overflow".into()));
        }
        let geometry = Geometry::new(frame, radius)?;
        let capacities = vec![usize::MAX; sel.len()];
        let mut all = Vec::with_capacity(types.len());
        let mut total = 0;
        for center in 0..types.len() {
            let groups = geometry.neighbors(frame, types, center, &capacities)?;
            let candidates: Vec<_> = groups
                .into_iter()
                .flatten()
                .map(|n| Candidate {
                    atom: n.atom,
                    image: n.image,
                })
                .collect();
            total += candidates.len();
            if total > 1_048_576 {
                return Err(Error("candidate cache exceeds bounded storage".into()));
            }
            all.push(candidates);
        }
        Ok(all)
    }
    pub fn neighbors(
        &self,
        frame: &Frame,
        types: &[i32],
        center: usize,
        sel: &[usize],
        cutoff: f64,
        geometry: &Geometry,
    ) -> Result<Vec<Vec<Neighbor>>> {
        let Some(all) = &self.candidates else {
            return geometry.neighbors(frame, types, center, sel);
        };
        let mut groups = vec![Vec::new(); sel.len()];
        for candidate in &all[center] {
            let atom = candidate.atom;
            let mut delta =
                std::array::from_fn(|a| frame.positions[atom][a] - frame.positions[center][a]);
            if let Some(cell) = frame.cell {
                let [a, b, c] = candidate.image;
                for k in 0..3 {
                    delta[k] +=
                        a as f64 * cell[0][k] + b as f64 * cell[1][k] + c as f64 * cell[2][k];
                }
            }
            let distance = delta[0].hypot(delta[1]).hypot(delta[2]);
            if !distance.is_finite() {
                return Err(Error("nonfinite cached image distance".into()));
            }
            if distance >= cutoff {
                continue;
            }
            if distance <= 1e-12 {
                return Err(Error(format!("coincident/near-coincident atoms {center} and {atom} (distance <= 1e-12 angstrom)")));
            }
            let kind = types[atom] as usize;
            groups[kind].push(Neighbor {
                atom,
                delta,
                distance,
                image: candidate.image,
            });
            if groups[kind].len() > sel[kind] {
                return Err(Error(format!(
                    "neighbor capacity overflow at atom {center}, type {}: more than sel={}",
                    frame.species[atom], sel[kind]
                )));
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
#[cfg(test)]
mod tests {
    use super::*;
    use omnimd_ml_contract::Units;
    fn identical(cache: &mut NeighborCache, frame: &Frame, types: &[i32], sel: &[usize]) -> bool {
        frame.validate().unwrap();
        let changed = cache.update(frame, types, 3.0, sel);
        let fresh = Geometry::new(frame, 3.0).unwrap();
        for center in 0..types.len() {
            let a = cache
                .neighbors(frame, types, center, sel, 3.0, &fresh)
                .unwrap();
            let b = fresh.neighbors(frame, types, center, sel).unwrap();
            for (a, b) in a.iter().zip(b) {
                assert_eq!(a.len(), b.len());
                for (a, b) in a.iter().zip(b) {
                    assert_eq!(a.atom, b.atom);
                    assert_eq!(a.image, b.image);
                    assert_eq!(a.delta, b.delta);
                    assert_eq!(a.distance, b.distance);
                }
            }
        }
        changed
    }
    #[test]
    fn exact_images_survive_cutoff_crossing_rebuild_wrap_cell_and_type_changes() {
        let mut cache = NeighborCache::new();
        let mut f = Frame {
            positions: vec![[0.0; 3], [3.1, 0.0, 0.0]],
            species: vec!["O".into(), "H".into()],
            cell: None,
            units: Units::EvAngstrom,
        };
        assert!(identical(&mut cache, &f, &[0, 1], &[8, 8]));
        f.positions[1][0] = 2.95;
        assert!(!identical(&mut cache, &f, &[0, 1], &[8, 8])); // from skin into cutoff
        f.positions[1][0] = 2.8;
        assert!(identical(&mut cache, &f, &[0, 1], &[8, 8]));
        f.cell = Some([[3.8, 0.0, 0.0], [0.3, 3.9, 0.0], [0.1, 0.2, 4.0]]);
        assert!(identical(&mut cache, &f, &[0, 1], &[8, 8]));
        f.positions[1][1] += 0.01;
        assert!(!identical(&mut cache, &f, &[0, 1], &[8, 8]));
        for k in 0..3 {
            f.positions[1][k] += f.cell.unwrap()[1][k];
        }
        assert!(identical(&mut cache, &f, &[0, 1], &[8, 8]));
        f.cell.as_mut().unwrap()[1][0] += 0.001;
        assert!(identical(&mut cache, &f, &[0, 1], &[8, 8]));
        f.species.swap(0, 1);
        assert!(identical(&mut cache, &f, &[1, 0], &[8, 8]));
        // Near-coincidence and active capacity errors remain strict on reuse.
        let mut f = Frame {
            positions: vec![[0.0; 3], [3.1, 0.0, 0.0]],
            species: vec!["O".into(), "H".into()],
            cell: None,
            units: Units::EvAngstrom,
        };
        let mut cache = NeighborCache::new();
        cache.update(&f, &[0, 1], 3.0, &[0, 0]);
        f.positions[1][0] = 2.95;
        assert!(cache
            .neighbors(
                &f,
                &[0, 1],
                0,
                &[0, 0],
                3.0,
                &Geometry::new(&f, 3.0).unwrap()
            )
            .is_err());
        let self_images = Frame {
            positions: vec![[0.1, 0.2, 0.3]],
            species: vec!["O".into()],
            cell: Some([[2.5, 0.0, 0.0], [0.2, 2.6, 0.0], [0.1, 0.2, 2.7]]),
            units: Units::EvAngstrom,
        };
        identical(&mut NeighborCache::new(), &self_images, &[0], &[32]);
        // Both atoms can move almost half the skin toward one another.
        let mut approaching = Frame {
            positions: vec![[0.0; 3], [3.39, 0.0, 0.0]],
            species: vec!["O".into(), "H".into()],
            cell: None,
            units: Units::EvAngstrom,
        };
        let mut cache = NeighborCache::new();
        assert!(identical(&mut cache, &approaching, &[0, 1], &[8, 8]));
        approaching.positions[0][0] = 0.199;
        approaching.positions[1][0] -= 0.199;
        assert!(!identical(&mut cache, &approaching, &[0, 1], &[8, 8]));
        approaching.positions[0][0] = 0.2;
        assert!(identical(&mut cache, &approaching, &[0, 1], &[8, 8]));
        // A failed conservative rebuild discards a previously valid list.
        for position in &mut approaching.positions {
            position[0] += 1e12;
        }
        assert!(identical(&mut cache, &approaching, &[0, 1], &[8, 8]));
        assert!(cache.candidates.is_none());
        approaching.positions[1][1] += 0.01;
        assert!(!identical(&mut cache, &approaching, &[0, 1], &[8, 8]));
        assert!(cache.candidates.is_none());
    }
}
