//! GPU prediction/seed tile path. Geometry remains host-owned and immutable.
use super::*;
use crate::cuda::transfer;
use crate::training_data::DatasetAccess;
use std::time::Instant;
const HOST_LIMIT: usize = 32 * 1024 * 1024;
#[derive(Clone, Debug, serde::Serialize)]
pub struct DeviceLossResources {
    pub extra_device_bytes: usize,
    pub extra_host_reserved_bytes: usize,
    pub estimated_extra_terms_per_atom: u64,
}
#[derive(Debug, serde::Serialize)]
pub struct DeviceLossDiagnostics {
    pub frame_bounds: Vec<i32>,
    pub frame_predictions: Vec<f64>, // [frame, E + nine row-virial entries]
    pub forces: Vec<f64>,            // [atom,xyz]
    pub frame_seeds: Vec<f64>,       // [frame, energy + nine virial seeds + objective]
    pub force_seeds: Vec<f64>,
    pub directions: Vec<f64>, // normalized environment directional rows
}
pub(crate) struct DeviceTile {
    pub rows: Vec<f64>,
    pub types: Vec<i32>,
    pub owners: Vec<i32>,
    pub geometry: Vec<f64>, // [slot, 12 Jacobian + three full-image displacement]
    pub frame_ids: Vec<i32>,
    pub bounds: Vec<i32>,
    pub force_labels: Vec<f64>,
    pub frame_labels: Vec<f64>,
    pub scales: Vec<f64>,
    pub indices: Vec<usize>,
}
impl DeviceTile {
    fn new(r: &TiledResources, slots: usize) -> Self {
        let n = r.effective_max_atoms;
        let f = r.requested.max_frames;
        Self {
            rows: Vec::with_capacity(n * slots * 4),
            types: Vec::with_capacity(n),
            owners: Vec::with_capacity(n * slots),
            geometry: Vec::with_capacity(n * slots * 15),
            frame_ids: Vec::with_capacity(n),
            bounds: Vec::with_capacity(f * 2),
            force_labels: Vec::with_capacity(n * 3),
            frame_labels: Vec::with_capacity(f * 10),
            scales: Vec::with_capacity(f * 3),
            indices: Vec::with_capacity(f),
        }
    }
    fn clear(&mut self) {
        self.rows.clear();
        self.types.clear();
        self.owners.clear();
        self.geometry.clear();
        self.frame_ids.clear();
        self.bounds.clear();
        self.force_labels.clear();
        self.frame_labels.clear();
        self.scales.clear();
        self.indices.clear();
    }
    fn retained_bytes(&self) -> usize {
        (self.rows.capacity()
            + self.geometry.capacity()
            + self.force_labels.capacity()
            + self.frame_labels.capacity()
            + self.scales.capacity())
            * 8
            + (self.types.capacity()
                + self.owners.capacity()
                + self.frame_ids.capacity()
                + self.bounds.capacity())
                * 4
            + self.indices.capacity() * std::mem::size_of::<usize>()
    }
    fn slot(
        &mut self,
        start: usize,
        neighbor: Option<&crate::neighbor::Neighbor>,
        jac: [[f64; 3]; 4],
    ) {
        self.owners
            .push(neighbor.map_or(-1, |q| (start + q.atom) as i32));
        self.geometry.extend(jac.into_iter().flatten());
        self.geometry.extend(neighbor.map_or([0.; 3], |q| q.delta));
    }
}
pub(crate) struct DeviceLossOutput {
    pub gradient: Vec<f64>,
    pub frame_seeds: Vec<f64>,
    pub diagnostics: Option<DeviceLossDiagnostics>,
}
pub(super) fn plan_resources(
    model: &Model,
    plan: &ReversePlan,
    config: &CudaTiling,
) -> Result<TiledResources> {
    let mut base = tiled::plan_resources(model, plan, config)?;
    let p = plan.parameter_network.len();
    let slots = model.descriptor.sel.iter().sum::<usize>();
    let width = model.descriptor.embedding[0][0].output_size();
    // Additional first-order forward/reverse and bounded per-frame deterministic gather.
    let extra_work =
        2 * center_terms(model) + (config.max_atoms_per_frame * slots * 3 + slots * 27) as u64;
    let per_atom = (p as u64) * center_terms(model) + extra_work;
    let cap = base
        .effective_max_atoms
        .min((WORK_LIMIT / per_atom) as usize);
    for n in (1..=cap).rev() {
        let frames = config.max_frames;
        let extra_device = n * slots * 148
            + n * (76 + 32 * width)
            + frames * 280
            + model.type_map.len() * 8
            + transfer::ALIGNMENT_RESERVE;
        // Includes optional last-tile diagnostic downloads, plus conservative legacy reserves.
        let extra_host = n * slots * 156
            + n * 76
            + frames * 296
            + transfer::input_bytes(n, slots, frames, true)?;
        let host = tiled::host_bytes(n, slots, p, frames)?
            .checked_add(extra_host)
            .ok_or_else(|| Error("GPU loss host payload overflow".into()))?;
        let mut device = plan.tiled_resources(n)?;
        device.peak_device_buffer_bytes = device
            .peak_device_buffer_bytes
            .checked_add(extra_device)
            .ok_or_else(|| Error("GPU loss device payload overflow".into()))?;
        if host > HOST_LIMIT || device.peak_device_buffer_bytes > device.policy_limit_bytes {
            continue;
        }
        base.effective_max_atoms = n;
        base.estimated_scalar_terms_at_capacity = n as u64 * per_atom;
        base.host_tile_payload_at_capacity_bytes = host;
        base.device_at_capacity = device;
        base.device_loss = Some(DeviceLossResources {
            extra_device_bytes: extra_device,
            extra_host_reserved_bytes: extra_host,
            estimated_extra_terms_per_atom: extra_work,
        });
        return Ok(base);
    }
    Err(Error(
        "GPU loss tile cannot fit one atom within unchanged work/memory budgets".into(),
    ))
}
impl CudaFullTrainer {
    /// Opt-in diagnostics retain only the final tile of the most recent successful batch.
    /// They describe pre-update predictions; no CPU verification is hidden in this path.
    pub fn set_device_loss_diagnostics(&mut self, enabled: bool) {
        self.capture_device_loss = enabled;
        self.device_loss_diagnostics = None;
    }
    pub fn device_loss_diagnostics(&self) -> Option<&DeviceLossDiagnostics> {
        self.device_loss_diagnostics.as_ref()
    }
    pub(super) fn device_batch_gradient(
        &mut self,
        dataset: &dyn DatasetAccess,
        indices: &[usize],
        loss: &TrainConfig,
        virial: Option<&VirialLoss>,
    ) -> Result<(f64, Vec<f64>)> {
        self.device_loss_diagnostics = None;
        validate_loss(loss, virial)?;
        if indices.is_empty() || indices.len() > 1024 {
            return Err(Error(
                "GPU loss batch requires 1..1024 selected frames".into(),
            ));
        }
        let cap = self.tiled.as_ref().unwrap().clone();
        let p = self.layout.parameters.len();
        self.preflight_tiled_batch(dataset, indices)?;
        let mut tile = DeviceTile::new(&cap, self.cpu.model().descriptor.sel.iter().sum());
        let mut gradient = vec![0.; p];
        let mut frame_gradient = vec![0.; p];
        let mut objective = 0.;
        for &index in indices {
            let n = dataset.atoms(index)?;
            check_work_bound(
                n,
                p,
                self.maximum_center_terms,
                cap.requested.max_atoms_per_frame,
            )?;
            if n > cap.effective_max_atoms {
                return Err(Error(format!(
                    "GPU loss tile cannot fit frame {index} with {n} atoms"
                )));
            }
            if !tile.indices.is_empty()
                && (tile.types.len() + n > cap.effective_max_atoms
                    || tile.indices.len() == cap.requested.max_frames)
            {
                self.flush_device_tile(
                    &tile,
                    &mut gradient,
                    &mut frame_gradient,
                    &mut objective,
                    false,
                )?;
                tile.clear();
            }
            let record = dataset.frame(index)?;
            if record.frame.positions.len() != n {
                return Err(Error("GPU loss dataset atom count changed".into()));
            }
            self.append_device_frame(&mut tile, index, &record, loss, virial, indices.len())?;
        }
        self.flush_device_tile(
            &tile,
            &mut gradient,
            &mut frame_gradient,
            &mut objective,
            self.capture_device_loss,
        )?;
        if !objective.is_finite() || gradient.iter().any(|v| !v.is_finite()) {
            return Err(Error(
                "nonfinite GPU loss objective or reduced gradient".into(),
            ));
        }
        Ok((objective, gradient))
    }
    fn append_device_frame(
        &mut self,
        tile: &mut DeviceTile,
        index: usize,
        record: &DatasetFrame,
        loss: &TrainConfig,
        virial: Option<&VirialLoss>,
        normalization: usize,
    ) -> Result<()> {
        record.frame.validate()?;
        validate_record(record, virial)?;
        let phase = Instant::now();
        let start = tile.types.len();
        let n = record.frame.positions.len();
        let prepared = match &mut self.fixed_geometry {
            Some(cache) => Some(cache.get(&self.cpu, record)?),
            None => None,
        };
        let d = &self.cpu.model().descriptor;
        if let Some(frame) = prepared {
            tile.rows.extend_from_slice(&frame.rows);
            tile.types.extend_from_slice(&frame.types);
            for slot in frame.slots.iter() {
                tile.slot(start, slot.neighbor.as_ref(), slot.jac);
            }
        } else {
            let types = record.frame.type_ids(self.cpu.metadata())?;
            let geometry = Geometry::new(&record.frame, d.rcut)?;
            for center in 0..n {
                let groups = geometry.neighbors(&record.frame, &types, center, &d.sel)?;
                for (species, group) in groups.iter().enumerate() {
                    for slot in 0..d.sel[species] {
                        let neighbor = group.get(slot);
                        let (r, jac) = crate::environment(
                            neighbor,
                            &d.normalization[types[center] as usize],
                            d.rcut_smth,
                            d.rcut,
                        );
                        tile.rows.extend(r);
                        tile.slot(start, neighbor, jac);
                    }
                }
            }
            tile.types.extend(types);
        }
        tile.frame_ids
            .extend(std::iter::repeat_n(tile.indices.len() as i32, n));
        tile.bounds.extend([start as i32, (start + n) as i32]);
        tile.indices.push(index);
        tile.force_labels.extend(
            record
                .labels
                .forces
                .as_ref()
                .unwrap()
                .iter()
                .flatten()
                .copied(),
        );
        tile.frame_labels.push(record.labels.energy.unwrap());
        tile.frame_labels.extend(
            record
                .labels
                .virial
                .unwrap_or([[0.; 3]; 3])
                .into_iter()
                .flatten(),
        );
        let (es, fs) = crate::training::scales(loss, normalization, n);
        let vs = virial
            .map(|v| v.scale(normalization, n))
            .transpose()?
            .unwrap_or(0.);
        if [es, fs, vs].iter().any(|v| !v.is_finite() || *v < 0.) {
            return Err(Error("invalid GPU loss scale".into()));
        }
        tile.scales.extend([es, fs, vs]);
        self.profile.geometry_ms += phase.elapsed().as_secs_f64() * 1000.;
        Ok(())
    }
    fn flush_device_tile(
        &mut self,
        tile: &DeviceTile,
        gradient: &mut [f64],
        frame_gradient: &mut [f64],
        objective: &mut f64,
        capture: bool,
    ) -> Result<()> {
        let n = tile.types.len();
        let f = tile.indices.len();
        let p = gradient.len();
        let slots = self.cpu.model().descriptor.sel.iter().sum::<usize>();
        let cap = self.tiled.as_ref().unwrap();
        if n == 0
            || n > cap.effective_max_atoms
            || f == 0
            || f > cap.requested.max_frames
            || tile.rows.len() != n * slots * 4
            || tile.geometry.len() != n * slots * 15
            || tile.owners.len() != n * slots
            || tile.frame_ids.len() != n
            || tile.bounds.len() != f * 2
            || tile.force_labels.len() != n * 3
            || tile.frame_labels.len() != f * 10
            || tile.scales.len() != f * 3
        {
            return Err(Error("GPU loss tile shape/capacity mismatch".into()));
        }
        // Check every physical owner against its frame, including self-images.
        for (i, range) in tile.bounds.chunks_exact(2).enumerate() {
            let begin = range[0] as usize;
            let end = range[1] as usize;
            if begin >= end
                || end > n
                || end - begin > cap.requested.max_atoms_per_frame
                || (i == 0 && begin != 0)
                || (i > 0 && range[0] != tile.bounds[2 * i - 1])
                || (i + 1 == f && end != n)
                || tile.frame_ids[begin..end].iter().any(|v| *v != i as i32)
                || tile.owners[begin * slots..end * slots]
                    .iter()
                    .any(|j| *j < -1 || (*j >= 0 && (*j < range[0] || *j >= range[1])))
            {
                return Err(Error(
                    "GPU loss cross-frame owner or boundary mismatch".into(),
                ));
            }
        }
        if tile
            .rows
            .iter()
            .chain(&tile.geometry)
            .any(|v| !v.is_finite())
        {
            return Err(Error("nonfinite GPU loss geometry".into()));
        }
        let diagnostic_bytes = if capture {
            (n * 6 + n * slots * 4 + f * 10) * 8 + f * 2 * 4
        } else {
            0
        };
        let packed_bytes = transfer::input_bytes(n, slots, f, true)?;
        let host_base = tile.retained_bytes() + (n * p + 2 * p + f * 11) * 8 + diagnostic_bytes;
        let available = cap
            .host_tile_payload_at_capacity_bytes
            .min(HOST_LIMIT)
            .checked_sub(host_base)
            .ok_or_else(|| Error("GPU loss host payload exceeded".into()))?;
        let staging_start = Instant::now();
        let staging = self
            .runtime
            .reserve_packed_inputs(packed_bytes, available)?;
        let staging_ms = staging_start.elapsed().as_secs_f64() * 1000.;
        let host = host_base + staging;
        if host > cap.host_tile_payload_at_capacity_bytes || host > HOST_LIMIT {
            return Err(Error("GPU loss host tile payload exceeded".into()));
        }
        let scratch = self.reverse.as_ref().unwrap().tiled_resources(n)?;
        let mut out = self.runtime.device_loss_tile(
            tile,
            p,
            &scratch,
            self.cpu.model().descriptor.embedding[0][0].output_size(),
            capture,
        )?;
        let resident = self.runtime.resident_buffer_bytes();
        if resident > cap.device_at_capacity.peak_device_buffer_bytes {
            return Err(Error("GPU loss resident device payload exceeded".into()));
        }
        self.profile.max_atoms_per_frame = self.profile.max_atoms_per_frame.max(
            tile.bounds
                .chunks_exact(2)
                .map(|b| (b[1] - b[0]) as usize)
                .max()
                .unwrap_or(0),
        );
        self.profile.frames += f;
        self.profile.device_prediction_frames += f;
        self.profile.tiles += 1;
        self.profile.input_transfer_calls += 2;
        self.profile.input_transfer_bytes += packed_bytes + 4;
        self.profile.kernel_launches += 10;
        self.profile.synchronizations += 1;
        self.profile.max_frames_per_launch = self.profile.max_frames_per_launch.max(f);
        self.profile.max_atoms_per_launch = self.profile.max_atoms_per_launch.max(n);
        self.profile.peak_device_buffer_bytes = self.profile.peak_device_buffer_bytes.max(resident);
        self.profile.peak_host_tile_payload_bytes =
            self.profile.peak_host_tile_payload_bytes.max(host);
        self.profile.upload_ms += staging_ms + self.runtime.last_timing_ms[0];
        self.profile.kernels_and_sync_ms += self.runtime.last_timing_ms[1];
        self.profile.download_ms += self.runtime.last_timing_ms[2];
        let phase = Instant::now();
        for (i, bounds) in tile.bounds.chunks_exact(2).enumerate() {
            frame_gradient.fill(0.);
            for atom in out.gradient[bounds[0] as usize * p..bounds[1] as usize * p].chunks_exact(p)
            {
                for (g, v) in frame_gradient.iter_mut().zip(atom) {
                    *g += v;
                }
            }
            for (g, v) in gradient.iter_mut().zip(frame_gradient.iter()) {
                *g += v;
            }
            *objective += out.frame_seeds[i * 11 + 10];
        }
        self.profile.reduction_ms += phase.elapsed().as_secs_f64() * 1000.;
        if let Some(mut diag) = out.diagnostics.take() {
            diag.frame_seeds = out.frame_seeds;
            self.device_loss_diagnostics = Some(diag);
        }
        Ok(())
    }
}
