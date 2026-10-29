//! Immutable, bounded prepared training geometry. No neural or loss-dependent values.
#[cfg(test)]
use crate::training_data::Dataset;
use crate::{
    neighbor::{Geometry, Neighbor},
    training_data::{DatasetAccess, DatasetFrame},
    Model,
};
use omnimd_ml_contract::{Error, Frame, Potential, Result, Units};
use sha2::{Digest, Sha256};
use std::{mem::size_of, time::Instant};
const LIMIT: usize = 64 * 1024 * 1024;
#[derive(Clone, Debug, serde::Serialize)]
pub struct FixedGeometryStats {
    pub frames: usize,
    pub padded_slots: usize,
    pub retained_payload_bytes: usize,
    pub policy_limit_bytes: usize,
    pub build_ms: f64,
    pub hits: usize,
    pub geometry_sha256: Box<str>,
}
struct Spec {
    type_map: Box<[Box<str>]>,
    units: Units,
    cutoff: u64,
    smooth: u64,
    sel: Box<[usize]>,
    normalization: Box<[[u64; 8]]>,
}
impl Spec {
    fn new(m: &Model) -> Self {
        Self {
            type_map: m
                .type_map
                .iter()
                .map(|s| s.clone().into_boxed_str())
                .collect(),
            units: m.units,
            cutoff: m.descriptor.rcut.to_bits(),
            smooth: m.descriptor.rcut_smth.to_bits(),
            sel: m.descriptor.sel.clone().into_boxed_slice(),
            normalization: m
                .descriptor
                .normalization
                .iter()
                .map(|n| {
                    std::array::from_fn(|i| {
                        if i < 4 {
                            n.mean[i].to_bits()
                        } else {
                            n.std[i - 4].to_bits()
                        }
                    })
                })
                .collect(),
        }
    }
    fn matches(&self, m: &Model) -> bool {
        m.descriptor.kind == "se_e2_a"
            && self.units == m.units
            && self.cutoff == m.descriptor.rcut.to_bits()
            && self.smooth == m.descriptor.rcut_smth.to_bits()
            && *self.sel == m.descriptor.sel
            && self.type_map.len() == m.type_map.len()
            && self
                .type_map
                .iter()
                .zip(&m.type_map)
                .all(|(a, b)| a.as_ref() == b)
            && self.normalization.len() == m.descriptor.normalization.len()
            && self
                .normalization
                .iter()
                .zip(&m.descriptor.normalization)
                .all(|(a, b)| {
                    (0..8).all(|i| {
                        a[i] == if i < 4 {
                            b.mean[i].to_bits()
                        } else {
                            b.std[i - 4].to_bits()
                        }
                    })
                })
    }
}
pub(crate) struct Slot {
    pub species: usize,
    pub neighbor: Option<Neighbor>,
    pub jac: [[f64; 3]; 4],
}
pub(crate) struct PreparedFrame {
    pub types: Box<[i32]>,
    positions: Box<[[u64; 3]]>,
    cell: Option<[[u64; 3]; 3]>,
    pub slots: Box<[Slot]>,
    pub rows: Box<[f64]>,
    pub slots_per_atom: usize,
}
impl PreparedFrame {
    fn new(m: &Model, f: &Frame) -> Result<Self> {
        f.validate()?;
        let metadata = m.validate()?;
        let types = f.type_ids(&metadata)?;
        let d = &m.descriptor;
        let count = d.sel.iter().sum::<usize>();
        let geometry = Geometry::new(f, d.rcut)?;
        let mut slots = Vec::with_capacity(types.len() * count);
        let mut rows = Vec::with_capacity(types.len() * count * 4);
        for center in 0..types.len() {
            let groups = geometry.neighbors(f, &types, center, &d.sel)?;
            for (species, group) in groups.iter().enumerate() {
                for slot in 0..d.sel[species] {
                    let neighbor = group.get(slot);
                    let (r, jac) = crate::environment(
                        neighbor,
                        &d.normalization[types[center] as usize],
                        d.rcut_smth,
                        d.rcut,
                    );
                    if r.iter().chain(jac.iter().flatten()).any(|x| !x.is_finite()) {
                        return Err(Error("nonfinite fixed training environment".into()));
                    }
                    rows.extend(r);
                    slots.push(Slot {
                        species,
                        neighbor: neighbor.cloned(),
                        jac,
                    });
                }
            }
        }
        Ok(Self {
            types: types.into_boxed_slice(),
            positions: f.positions.iter().map(|p| p.map(f64::to_bits)).collect(),
            cell: f.cell.map(|h| h.map(|r| r.map(f64::to_bits))),
            slots: slots.into_boxed_slice(),
            rows: rows.into_boxed_slice(),
            slots_per_atom: count,
        })
    }
    fn matches(&self, f: &Frame, types: &[i32]) -> bool {
        self.types.as_ref() == types
            && self.positions.len() == f.positions.len()
            && self
                .positions
                .iter()
                .zip(&f.positions)
                .all(|(a, b)| *a == b.map(f64::to_bits))
            && self.cell == f.cell.map(|h| h.map(|r| r.map(f64::to_bits)))
    }
    pub fn atom_rows(
        &self,
        center: usize,
    ) -> impl Iterator<Item = (usize, Option<&Neighbor>, [f64; 4], [[f64; 3]; 4])> {
        let begin = center * self.slots_per_atom;
        let end = begin + self.slots_per_atom;
        self.slots[begin..end]
            .iter()
            .zip(self.rows[4 * begin..4 * end].chunks_exact(4))
            .map(|(s, r)| {
                (
                    s.species,
                    s.neighbor.as_ref(),
                    [r[0], r[1], r[2], r[3]],
                    s.jac,
                )
            })
    }
}
struct Entry {
    index: u64,
    frame: PreparedFrame,
}
pub(crate) struct FixedGeometryCache {
    spec: Spec,
    entries: Box<[Entry]>,
    stats: FixedGeometryStats,
    valid: bool,
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| Error("fixed geometry size overflow".into()))
}
fn product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| Error("fixed geometry size overflow".into()))
}
fn enforce(bytes: usize) -> Result<usize> {
    if bytes > LIMIT {
        Err(Error(format!("fixed geometry cache payload {bytes} exceeds {LIMIT} bytes; disable caching explicitly")))
    } else {
        Ok(bytes)
    }
}
pub(crate) fn estimate(m: &Model, d: &dyn DatasetAccess, indices: &[usize]) -> Result<usize> {
    crate::cuda::validate_full_model(m)?;
    if indices.is_empty()
        || indices.len()
            > if d.indexed() {
                crate::indexed_dataset::MAX_FRAMES
            } else {
                1024
            }
    {
        return Err(Error(
            "fixed geometry selected-frame count exceeds dataset access mode limit".into(),
        ));
    }
    let mut total = add(size_of::<FixedGeometryCache>(), 64)?; // SHA256 string payload
    total = add(total, product(indices.len(), size_of::<Entry>())?)?;
    total = add(
        total,
        product(
            m.type_map.len(),
            size_of::<Box<str>>() + size_of::<usize>() + size_of::<[u64; 8]>(),
        )?,
    )?;
    for name in &m.type_map {
        total = add(total, name.len())?;
    }
    let slots = m.descriptor.sel.iter().sum::<usize>();
    for &i in indices {
        let n = d.atoms(i)?;
        if n == 0 || n > crate::full_training::CUDA_MAX_FRAME_ATOMS {
            return Err(Error("fixed geometry requires 1..256 atoms/frame".into()));
        }
        total = add(total, product(n, size_of::<[u64; 3]>() + size_of::<i32>())?)?;
        total = add(
            total,
            product(product(n, slots)?, size_of::<Slot>() + 4 * size_of::<f64>())?,
        )?;
    }
    enforce(total)
}
impl FixedGeometryCache {
    pub fn new(m: &Model, d: &dyn DatasetAccess, indices: &[usize]) -> Result<Self> {
        let start = Instant::now();
        let bytes = estimate(m, d, indices)?;
        let mut ordered = indices.to_vec();
        ordered.sort_by_key(|i| d.index(*i).expect("validated index"));
        if ordered
            .windows(2)
            .any(|w| d.index(w[0]).unwrap() == d.index(w[1]).unwrap())
        {
            return Err(Error("duplicate fixed geometry frame index".into()));
        }
        let spec = Spec::new(m);
        let mut entries = Vec::with_capacity(indices.len());
        let mut digest = Sha256::new();
        let mut count = 0;
        // This audit identity is independent of neural weights, labels and frame IDs.
        digest.update(b"omnimd-fixed-training-geometry-v1/se_e2_a/EvAngstrom/float64");
        digest.update((spec.type_map.len() as u64).to_le_bytes());
        digest.update((indices.len() as u64).to_le_bytes());
        for name in &spec.type_map {
            digest.update((name.len() as u64).to_le_bytes());
            digest.update(name.as_bytes());
        }
        digest.update(spec.cutoff.to_le_bytes());
        digest.update(spec.smooth.to_le_bytes());
        for &s in &spec.sel {
            digest.update((s as u64).to_le_bytes());
        }
        for row in &spec.normalization {
            for &x in row {
                digest.update(x.to_le_bytes());
            }
        }
        for i in ordered {
            let r = d.frame(i)?;
            let frame = PreparedFrame::new(m, &r.frame)?;
            count += frame.slots.len();
            digest.update(r.index.to_le_bytes());
            digest.update((frame.types.len() as u64).to_le_bytes());
            for &t in &frame.types {
                digest.update(t.to_le_bytes());
            }
            for p in &frame.positions {
                for &x in p {
                    digest.update(x.to_le_bytes());
                }
            }
            digest.update([u8::from(frame.cell.is_some())]);
            if let Some(h) = frame.cell {
                for row in h {
                    for x in row {
                        digest.update(x.to_le_bytes());
                    }
                }
            }
            entries.push(Entry {
                index: r.index,
                frame,
            });
        }
        Ok(Self {
            spec,
            entries: entries.into_boxed_slice(),
            valid: true,
            stats: FixedGeometryStats {
                frames: indices.len(),
                padded_slots: count,
                retained_payload_bytes: bytes,
                policy_limit_bytes: LIMIT,
                build_ms: start.elapsed().as_secs_f64() * 1000.,
                hits: 0,
                geometry_sha256: format!("{:x}", digest.finalize()).into_boxed_str(),
            },
        })
    }
    pub fn get(
        &mut self,
        potential: &crate::NativePotential,
        r: &DatasetFrame,
    ) -> Result<&PreparedFrame> {
        if !self.valid {
            return Err(Error("fixed training geometry cache invalidated".into()));
        }
        let m = potential.model();
        let result = (|| {
            r.frame.validate()?;
            let types = r.frame.type_ids(potential.metadata())?;
            let index = self
                .entries
                .binary_search_by_key(&r.index, |e| e.index)
                .map_err(|_| Error("frame absent from fixed training geometry cache".into()))?;
            if !self.spec.matches(m)
                || r.frame.units != self.spec.units
                || !self.entries[index].frame.matches(&r.frame, &types)
            {
                return Err(Error(
                    "fixed training geometry identity changed; recreate trainer/cache".into(),
                ));
            }
            Ok(index)
        })();
        match result {
            Ok(i) => {
                self.stats.hits += 1;
                Ok(&self.entries[i].frame)
            }
            Err(e) => {
                self.valid = false;
                Err(e)
            }
        }
    }
    pub fn stats(&self) -> FixedGeometryStats {
        self.stats.clone()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strict_payload_boundary_and_overflow_without_driver() {
        assert_eq!(enforce(LIMIT).unwrap(), LIMIT);
        assert!(enforce(LIMIT + 1).is_err());
        assert!(product(usize::MAX, 2).is_err());
        assert!(add(usize::MAX, 1).is_err());
    }
}

#[cfg(test)]
mod identity_tests {
    use super::*;
    use crate::{
        training_data::{DatasetHeader, Labels, VirialConvention},
        NativePotential,
    };
    use std::collections::BTreeMap;
    fn data() -> (Model, Dataset) {
        let m: Model = serde_json::from_str(include_str!("../fixtures/full-tanh-v1.json")).unwrap();
        let mut a: Frame = serde_json::from_str(include_str!("../fixtures/frame.jsonl")).unwrap();
        a.cell = None;
        let mut b = a.clone();
        b.cell = Some([[3.8, 0., 0.], [0.3, 3.9, 0.], [0.1, 0.2, 4.]]);
        let c = Frame {
            positions: vec![[0.23, 0.17, 0.11]],
            species: vec!["O".into()],
            cell: Some([[2.5, 0., 0.], [0.3, 2.6, 0.], [0.2, 0.1, 2.7]]),
            units: m.units,
        };
        let frames = [a, b, c]
            .into_iter()
            .enumerate()
            .map(|(i, frame)| DatasetFrame {
                index: i as u64,
                id: format!("case/{i}"),
                frame,
                labels: Labels {
                    energy: None,
                    forces: None,
                    virial: None,
                },
            })
            .collect();
        let d = Dataset {
            header: DatasetHeader {
                schema: "omnimd-dataset-jsonl-v1".into(),
                type_map: m.type_map.clone(),
                units: m.units,
                virial_convention: VirialConvention::MinusDEpsilonRow,
                source: "original cache unit fixture".into(),
                selected_sets: vec![],
            },
            frames,
            source_sha256: BTreeMap::new(),
            dataset_sha256: "unit".into(),
        };
        (m, d)
    }
    fn bits(e: &omnimd_ml_contract::Evaluation) -> Vec<u64> {
        std::iter::once(&e.energy)
            .chain(e.forces.iter().flatten())
            .chain(e.virial.iter().flatten())
            .map(|x| x.to_bits())
            .collect()
    }
    #[test]
    fn prepared_rows_jacobians_and_current_model_efv_are_bitwise_identical() {
        let (mut m, mut d) = data();
        let mut cache = FixedGeometryCache::new(&m, &d, &[0, 1, 2]).unwrap();
        let identity = cache.stats.geometry_sha256.clone();
        for update in 0..2 {
            if update == 1 {
                m.descriptor.embedding[0][0].layers[0].weights[0][0] += 0.2;
                d.frames[0].labels.energy = Some(12.);
                d.frames[0].id = "changed labels and ID".into();
            }
            let mut potential = NativePotential::new(m.clone()).unwrap();
            for record in &d.frames {
                let prepared = cache.get(&potential, record).unwrap();
                let expected = potential.eval(&record.frame).unwrap();
                assert_eq!(
                    bits(&expected),
                    bits(&potential.eval_prepared(prepared).unwrap())
                );
                let geometry = Geometry::new(&record.frame, m.descriptor.rcut).unwrap();
                for center in 0..prepared.types.len() {
                    let groups = geometry
                        .neighbors(&record.frame, &prepared.types, center, &m.descriptor.sel)
                        .unwrap();
                    let mut actual = prepared.atom_rows(center);
                    for (species, group) in groups.iter().enumerate() {
                        for slot in 0..m.descriptor.sel[species] {
                            let (r, j) = crate::environment(
                                group.get(slot),
                                &m.descriptor.normalization[prepared.types[center] as usize],
                                m.descriptor.rcut_smth,
                                m.descriptor.rcut,
                            );
                            let (s, q, cr, cj) = actual.next().unwrap();
                            assert_eq!(s, species);
                            assert_eq!(r.map(f64::to_bits), cr.map(f64::to_bits));
                            assert_eq!(
                                j.map(|x| x.map(f64::to_bits)),
                                cj.map(|x| x.map(f64::to_bits))
                            );
                            assert_eq!(
                                q.map(|v| (v.atom, v.image, v.delta.map(f64::to_bits))),
                                group.get(slot).map(|v| (
                                    v.atom,
                                    v.image,
                                    v.delta.map(f64::to_bits)
                                ))
                            );
                        }
                    }
                    assert!(actual.next().is_none());
                }
            }
        }
        assert_eq!(cache.stats.hits, 6);
        assert_eq!(
            identity,
            FixedGeometryCache::new(&m, &d, &[2, 1, 0])
                .unwrap()
                .stats
                .geometry_sha256
        );
        let actual = size_of::<FixedGeometryCache>()
            + cache.stats.geometry_sha256.len()
            + cache.entries.len() * size_of::<Entry>()
            + cache.spec.type_map.len() * size_of::<Box<str>>()
            + cache.spec.type_map.iter().map(|s| s.len()).sum::<usize>()
            + cache.spec.sel.len() * size_of::<usize>()
            + cache.spec.normalization.len() * size_of::<[u64; 8]>()
            + cache
                .entries
                .iter()
                .map(|e| {
                    e.frame.types.len() * size_of::<i32>()
                        + e.frame.positions.len() * size_of::<[u64; 3]>()
                        + e.frame.slots.len() * size_of::<Slot>()
                        + e.frame.rows.len() * 8
                })
                .sum::<usize>();
        assert_eq!(actual, cache.stats.retained_payload_bytes);
    }
    #[test]
    fn changed_frame_or_geometry_spec_invalidates_without_reuse() {
        let (m, d) = data();
        for case in 0..11 {
            let mut cache = FixedGeometryCache::new(&m, &d, &[0, 1, 2]).unwrap();
            let mut changed = m.clone();
            let mut r = d.frames[1].clone();
            match case {
                0 => r.frame.positions[0][0] += 1e-12,
                1 => r.frame.cell.as_mut().unwrap()[0][1] = -0.0,
                2 => r.frame.cell = None,
                3 => r.frame.species.swap(0, 1),
                4 => {
                    r.frame.positions.pop();
                    r.frame.species.pop();
                }
                5 => changed.descriptor.rcut += 0.1,
                6 => changed.descriptor.rcut_smth -= 0.1,
                7 => changed.descriptor.sel[0] += 1,
                8 => changed.descriptor.normalization[0].mean[0] += 0.001,
                9 => changed.descriptor.normalization[1].std[0] *= 1.01,
                _ => changed.type_map.swap(0, 1),
            }
            let potential = NativePotential::new(changed).unwrap();
            assert!(cache.get(&potential, &r).is_err(), "case {case}");
            assert!(!cache.valid);
            assert!(cache
                .get(&NativePotential::new(m.clone()).unwrap(), &d.frames[1])
                .err()
                .unwrap()
                .0
                .contains("invalidated"));
        }
        let mut cache = FixedGeometryCache::new(&m, &d, &[0]).unwrap();
        assert!(cache
            .get(&NativePotential::new(m.clone()).unwrap(), &d.frames[1])
            .is_err());
    }
    #[test]
    fn selected_frame_budget_and_partial_failure_are_checked_without_driver() {
        let (m, mut d) = data();
        assert!(FixedGeometryCache::new(&m, &d, &[0, 0]).is_err());
        assert!(estimate(&m, &d, &[9]).is_err());
        d.frames[1].frame.positions[1] = d.frames[1].frame.positions[0];
        assert!(FixedGeometryCache::new(&m, &d, &[0, 1]).is_err());
        let mut r = d.frames[0].clone();
        r.frame.positions = (0..64).map(|i| [i as f64 * 4., 0., 0.]).collect();
        r.frame.species = vec!["O".into(); 64];
        d.frames = (0..400)
            .map(|i| {
                let mut r = r.clone();
                r.index = i;
                r
            })
            .collect();
        let indices = (0..400).collect::<Vec<_>>();
        assert!(estimate(&m, &d, &indices)
            .unwrap_err()
            .0
            .contains("exceeds"));
        assert!(FixedGeometryCache::new(&m, &d, &indices).is_err());
    }
}
