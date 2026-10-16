//! Bounded multi-frame reverse execution. Tiles never define optimizer batches.
use super::*;
use crate::training_data::DatasetAccess;
use std::time::Instant;
const HOST_LIMIT: usize = 32 * 1024 * 1024;

#[derive(Clone, Debug, serde::Serialize)]
pub struct TiledResources {
    pub requested: CudaTiling,
    pub effective_max_atoms: usize,
    pub maximum_center_terms: u64,
    pub estimated_scalar_terms_at_capacity: u64,
    pub scalar_work_policy_limit: u64,
    pub host_tile_payload_at_capacity_bytes: usize,
    pub host_tile_payload_policy_limit_bytes: usize,
    pub device_at_capacity: ReverseResources,
}
struct Boundary {
    start: usize,
    end: usize,
    energy: f64,
    objective: f64,
    dataset_index: usize,
}
fn host_bytes(atoms: usize, slots: usize, parameters: usize, frames: usize) -> Result<usize> {
    let per_atom = slots
        .checked_mul(64)
        .and_then(|x| parameters.checked_mul(8).and_then(|p| x.checked_add(p)))
        .and_then(|x| x.checked_add(20))
        .ok_or_else(|| Error("CUDA tile host payload overflow".into()))?;
    atoms
        .checked_mul(per_atom)
        .and_then(|x| {
            frames
                .checked_mul(std::mem::size_of::<Boundary>())
                .and_then(|f| x.checked_add(f))
        })
        .and_then(|x| parameters.checked_mul(16).and_then(|p| x.checked_add(p)))
        .ok_or_else(|| Error("CUDA tile host payload overflow".into()))
}
pub(super) fn plan_resources(
    model: &Model,
    plan: &ReversePlan,
    config: &CudaTiling,
) -> Result<TiledResources> {
    config.validate()?;
    let parameters = plan.parameter_network.len();
    let terms = center_terms(model);
    let per_atom_work = terms
        .checked_mul(parameters as u64)
        .filter(|n| *n > 0)
        .ok_or_else(|| Error("invalid CUDA tile work estimate".into()))?;
    let cap = config
        .max_atoms
        .min(config.max_frames * 64)
        .min((WORK_LIMIT / per_atom_work) as usize);
    for n in (1..=cap).rev() {
        let host = host_bytes(
            n,
            model.descriptor.sel.iter().sum(),
            parameters,
            config.max_frames,
        )?;
        if host > HOST_LIMIT {
            continue;
        }
        if let Ok(device) = plan.tiled_resources(n) {
            return Ok(TiledResources {
                requested: config.clone(),
                effective_max_atoms: n,
                maximum_center_terms: terms,
                estimated_scalar_terms_at_capacity: n as u64 * per_atom_work,
                scalar_work_policy_limit: WORK_LIMIT,
                host_tile_payload_at_capacity_bytes: host,
                host_tile_payload_policy_limit_bytes: HOST_LIMIT,
                device_at_capacity: device,
            });
        }
    }
    Err(Error(
        "CUDA tile cannot fit one atom within work/device/host policy budgets".into(),
    ))
}
struct Tile {
    rows: Vec<f64>,
    tangent: Vec<f64>,
    types: Vec<i32>,
    energy_seeds: Vec<f64>,
    boundaries: Vec<Boundary>,
}
impl Tile {
    fn new(cap: &TiledResources, slots: usize) -> Self {
        let n = cap.effective_max_atoms;
        Self {
            rows: Vec::with_capacity(n * slots * 4),
            tangent: Vec::with_capacity(n * slots * 4),
            types: Vec::with_capacity(n),
            energy_seeds: Vec::with_capacity(n),
            boundaries: Vec::with_capacity(cap.requested.max_frames),
        }
    }
    fn clear(&mut self) {
        self.rows.clear();
        self.tangent.clear();
        self.types.clear();
        self.energy_seeds.clear();
        self.boundaries.clear();
    }
    fn retained_bytes(&self) -> usize {
        (self.rows.capacity() + self.tangent.capacity() + self.energy_seeds.capacity()) * 8
            + self.types.capacity() * 4
            + self.boundaries.capacity() * std::mem::size_of::<Boundary>()
    }
}
impl CudaFullTrainer {
    /// Compute one original sampler batch, with one decoded record live at a time.
    /// On any failure the entire update is discarded and this trainer is poisoned.
    pub fn batch_gradient_with_virial(
        &mut self,
        dataset: &dyn DatasetAccess,
        indices: &[usize],
        loss: &TrainConfig,
        virial: Option<&VirialLoss>,
    ) -> Result<(f64, Vec<f64>)> {
        if self.poisoned {
            return Err(Error(
                "CUDA full trainer stopped after earlier failure; no automatic retry".into(),
            ));
        }
        if self.tiled.is_none() {
            return Err(Error(
                "batch_gradient_with_virial requires the tiled CUDA executor".into(),
            ));
        }
        self.profile = FullGradientProfile::default();
        let result = self.batch_gradient(dataset, indices, loss, virial);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
    fn batch_gradient(
        &mut self,
        dataset: &dyn DatasetAccess,
        indices: &[usize],
        loss: &TrainConfig,
        virial: Option<&VirialLoss>,
    ) -> Result<(f64, Vec<f64>)> {
        validate_loss(loss, virial)?;
        if indices.is_empty() || indices.len() > 1024 {
            return Err(Error(
                "CUDA tiled batch requires 1..1024 selected frames".into(),
            ));
        }
        let cap = self.tiled.as_ref().unwrap().clone();
        let p = self.layout.parameters.len();
        let mut tile = Tile::new(&cap, self.cpu.model().descriptor.sel.iter().sum());
        let mut gradient = vec![0.0; p];
        let mut frame_gradient = vec![0.0; p];
        let mut objective = 0.0;
        for &index in indices {
            let atoms = dataset.atoms(index)?;
            check_work(atoms, p, self.maximum_center_terms)?;
            if atoms > cap.effective_max_atoms {
                return Err(Error(format!(
                    "CUDA tile cannot fit frame {index} with {atoms} atoms"
                )));
            }
            if !tile.boundaries.is_empty()
                && (tile.types.len() + atoms > cap.effective_max_atoms
                    || tile.boundaries.len() == cap.requested.max_frames)
            {
                self.flush_tile(&tile, &mut gradient, &mut frame_gradient, &mut objective)?;
                tile.clear();
            }
            // Cow is released each iteration; never retain an entire decoded batch.
            let record = dataset.frame(index)?;
            if record.frame.positions.len() != atoms {
                return Err(Error("CUDA tile dataset atom count changed".into()));
            }
            self.append_frame(&mut tile, index, &record, loss, virial, indices.len())?;
        }
        self.flush_tile(&tile, &mut gradient, &mut frame_gradient, &mut objective)?;
        if !objective.is_finite() || gradient.iter().any(|v| !v.is_finite()) {
            return Err(Error(
                "nonfinite CUDA tiled batch objective or gradient".into(),
            ));
        }
        Ok((objective, gradient))
    }
    fn append_frame(
        &mut self,
        tile: &mut Tile,
        index: usize,
        record: &DatasetFrame,
        loss: &TrainConfig,
        virial: Option<&VirialLoss>,
        normalization: usize,
    ) -> Result<()> {
        record.frame.validate()?;
        validate_record(record, virial)?;
        let phase = Instant::now();
        let prepared = match &mut self.fixed_geometry {
            Some(cache) => Some(cache.get(&self.cpu, record)?),
            None => None,
        };
        let lookup_ms = phase.elapsed().as_secs_f64() * 1000.;
        let phase = Instant::now();
        let predicted = match prepared {
            Some(frame) => self.cpu.eval_prepared(frame)?,
            None => self.cpu.eval(&record.frame)?,
        };
        let seeds = LossSeeds::new(&predicted, record, loss, virial, normalization)?;
        self.profile.prediction_and_seeds_ms += phase.elapsed().as_secs_f64() * 1000.;
        let phase = Instant::now();
        let d = &self.cpu.model().descriptor;
        let start = tile.types.len();
        let atoms = record.frame.positions.len();
        if let Some(frame) = prepared {
            tile.rows.extend_from_slice(&frame.rows);
            tile.types.extend_from_slice(&frame.types);
            for (i, slot) in frame.slots.iter().enumerate() {
                let center = i / frame.slots_per_atom;
                for a in 0..4 {
                    tile.tangent.push(
                        slot.neighbor
                            .as_ref()
                            .map_or(0.0, |q| seeds.tangent(slot.jac[a], q, center)),
                    );
                }
            }
        } else {
            let types = record.frame.type_ids(self.cpu.metadata())?;
            let geometry = Geometry::new(&record.frame, d.rcut)?;
            for center in 0..atoms {
                let kind = types[center] as usize;
                let groups = geometry.neighbors(&record.frame, &types, center, &d.sel)?;
                for (species, group) in groups.iter().enumerate() {
                    for slot in 0..d.sel[species] {
                        let neighbor = group.get(slot);
                        let (r, jac) = crate::environment(
                            neighbor,
                            &d.normalization[kind],
                            d.rcut_smth,
                            d.rcut,
                        );
                        tile.rows.extend(r);
                        for a in 0..4 {
                            tile.tangent
                                .push(neighbor.map_or(0.0, |q| seeds.tangent(jac[a], q, center)));
                        }
                    }
                }
            }
            tile.types.extend_from_slice(&types);
        }
        tile.energy_seeds
            .extend(std::iter::repeat_n(seeds.energy, atoms));
        tile.boundaries.push(Boundary {
            start,
            end: start + atoms,
            energy: predicted.energy,
            objective: seeds.objective,
            dataset_index: index,
        });
        self.profile.geometry_ms += lookup_ms + phase.elapsed().as_secs_f64() * 1000.;
        Ok(())
    }
    fn flush_tile(
        &mut self,
        tile: &Tile,
        gradient: &mut [f64],
        frame_gradient: &mut [f64],
        objective: &mut f64,
    ) -> Result<()> {
        let n = tile.types.len();
        let p = gradient.len();
        let slots = self.cpu.model().descriptor.sel.iter().sum::<usize>();
        let cap = self.tiled.as_ref().unwrap();
        if n == 0
            || n > cap.effective_max_atoms
            || tile.boundaries.len() > cap.requested.max_frames
            || tile.rows.len() != n * slots * 4
            || tile.tangent.len() != tile.rows.len()
            || tile.energy_seeds.len() != n
        {
            return Err(Error("CUDA tile packing shape/capacity mismatch".into()));
        }
        // Vec capacities and all simultaneously live output/accumulator payloads.
        let host_bytes = tile.retained_bytes() + n * (p + 1) * 8 + (p * 2) * 8;
        if host_bytes > cap.host_tile_payload_at_capacity_bytes || host_bytes > HOST_LIMIT {
            return Err(Error("CUDA tile host payload capacity exceeded".into()));
        }
        let resources = self.reverse.as_ref().unwrap().tiled_resources(n)?;
        let (atomic, energy) = self.runtime.reverse_tile(
            &tile.rows,
            &tile.tangent,
            &tile.types,
            p,
            &tile.energy_seeds,
            &resources,
        )?;
        let device_bytes = self.runtime.resident_buffer_bytes();
        if device_bytes > cap.device_at_capacity.peak_device_buffer_bytes {
            return Err(Error(
                "CUDA tile device resident buffer capacity exceeded".into(),
            ));
        }
        self.profile.frames += tile.boundaries.len();
        self.profile.tiles += 1;
        self.profile.kernel_launches += 4;
        self.profile.synchronizations += 1;
        self.profile.max_frames_per_launch = self
            .profile
            .max_frames_per_launch
            .max(tile.boundaries.len());
        self.profile.max_atoms_per_launch = self.profile.max_atoms_per_launch.max(n);
        self.profile.peak_device_buffer_bytes =
            self.profile.peak_device_buffer_bytes.max(device_bytes);
        self.profile.peak_host_tile_payload_bytes =
            self.profile.peak_host_tile_payload_bytes.max(host_bytes);
        self.profile.upload_ms += self.runtime.last_timing_ms[0];
        self.profile.kernels_and_sync_ms += self.runtime.last_timing_ms[1];
        self.profile.download_ms += self.runtime.last_timing_ms[2];
        let phase = Instant::now();
        for f in &tile.boundaries {
            let primal = (f.start..f.end)
                .map(|i| energy[i] + self.cpu.model().atom_energy_bias[tile.types[i] as usize])
                .sum::<f64>();
            if !primal.is_finite() || (primal - f.energy).abs() > 1e-11 * (1. + f.energy.abs()) {
                return Err(Error(format!(
                    "CUDA tile frame {} primal disagrees with CPU inference",
                    f.dataset_index
                )));
            }
            frame_gradient.fill(0.);
            // Preserve reference reduction grouping exactly, independent of tiles.
            for atom in atomic[f.start * p..f.end * p].chunks_exact(p) {
                for (g, a) in frame_gradient.iter_mut().zip(atom) {
                    *g += a;
                }
            }
            for (g, f) in gradient.iter_mut().zip(frame_gradient.iter()) {
                *g += f;
            }
            *objective += f.objective;
        }
        self.profile.reduction_ms += phase.elapsed().as_secs_f64() * 1000.;
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tile_planner_bounds_and_seed_specialization_without_driver() {
        let model: Model =
            serde_json::from_str(include_str!("../../fixtures/full-tanh-v1.json")).unwrap();
        let (p, _, s) = packed(&model).unwrap();
        let plan = ReversePlan::new(&model, &p, &s).unwrap();
        let r = plan_resources(
            &model,
            &plan,
            &CudaTiling {
                max_frames: 32,
                max_atoms: 1024,
            },
        )
        .unwrap();
        assert_eq!(r.effective_max_atoms, 1024);
        assert!(r.estimated_scalar_terms_at_capacity <= WORK_LIMIT);
        assert!(r.host_tile_payload_at_capacity_bytes <= HOST_LIMIT);
        assert!(r.device_at_capacity.peak_device_buffer_bytes <= 64 * 1024 * 1024);
        assert!(plan.resources(65).is_err());
        assert!(plan.tiled_resources(1025).is_err());
        assert!(host_bytes(usize::MAX, 64, 2048, 32).is_err());
        for (frames, atoms) in [(0, 64), (33, 64), (1, 0), (1, 1025)] {
            assert!(plan_resources(
                &model,
                &plan,
                &CudaTiling {
                    max_frames: frames,
                    max_atoms: atoms
                }
            )
            .is_err());
        }
        let tiled = plan.tiled_source().unwrap();
        assert!(tiled.contains("adj[0]=energy_seed[atom];"));
        assert!(plan.source.contains("adj[0]=energy_seed;"));
        assert!(tiled.contains("const double *energy_seed,"));
    }
}
