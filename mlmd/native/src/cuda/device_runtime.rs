//! Dynamic-driver execution for the separate device-loss backend.
use super::*;
use crate::cuda::full::device_loss::{DeviceLossDiagnostics, DeviceLossOutput, DeviceTile};
impl Runtime {
    pub(crate) fn device_loss_tile(
        &mut self,
        t: &DeviceTile,
        p: usize,
        r: &crate::cuda::ReverseResources,
        width: usize,
        capture: bool,
    ) -> Result<DeviceLossOutput> {
        let n = t.types.len();
        let f = t.indices.len();
        let slots = t.owners.len();
        unsafe {
            let _current = Current::enter(self.api, self.context)?;
            let phase = std::time::Instant::now();
            let required = [
                (2, t.rows.len() * 8),
                (3, n * 4),
                (5, t.rows.len() * 8),
                (6, n * p * 8),
                (7, n * 8),
                (12, r.cache_bytes),
                (13, r.descriptor_adjoint_bytes),
                (14, r.slot_gradient_bytes),
                (16, n * 8),
                (17, slots * 4),
                (18, slots * 15 * 8),
                (19, n * 4),
                (20, f * 2 * 4),
                (21, n * 3 * 8),
                (22, f * 10 * 8),
                (23, f * 3 * 8),
                (24, n * 3 * 8),
                (25, f * 10 * 8),
                (26, n * 3 * 8),
                (27, f * 11 * 8),
                (28, slots * 3 * 8),
                (29, n * width * 4 * 8),
            ];
            for (index, bytes) in required {
                if self.buffers[index].1 < bytes && self.buffers[index].0 != 0 {
                    check(
                        (self.api.free)(self.buffers[index].0),
                        "cuMemFree_v2(device loss grow)",
                    )?;
                    self.buffers[index] = (0, 0);
                }
            }
            for (index, bytes) in required {
                self.ensure(index, bytes)?;
            }
            self.upload_current(2, &t.rows)?;
            self.upload_current(3, &t.types)?;
            self.upload_current(17, &t.owners)?;
            self.upload_current(18, &t.geometry)?;
            self.upload_current(19, &t.frame_ids)?;
            self.upload_current(20, &t.bounds)?;
            self.upload_current(21, &t.force_labels)?;
            self.upload_current(22, &t.frame_labels)?;
            self.upload_current(23, &t.scales)?;
            self.upload_current(8, &[0_i32])?;
            let mut address: [u64; 31] = std::array::from_fn(|i| self.buffers[i].0);
            let pointer = |x: &mut u64| x as *mut u64 as *mut c_void;
            let mut atoms = n as i32;
            let mut parameters = p as i32;
            let mut frames = f as i32;
            let mut args = [
                pointer(&mut address[0]),
                pointer(&mut address[1]),
                pointer(&mut address[2]),
                pointer(&mut address[5]),
                pointer(&mut address[3]),
                pointer(&mut address[4]),
                &mut atoms as *mut i32 as *mut c_void,
                &mut parameters as *mut i32 as *mut c_void,
                pointer(&mut address[16]),
                pointer(&mut address[6]),
                pointer(&mut address[7]),
                pointer(&mut address[8]),
                pointer(&mut address[12]),
                pointer(&mut address[13]),
                pointer(&mut address[14]),
                pointer(&mut address[15]),
                pointer(&mut address[17]),
                pointer(&mut address[18]),
                pointer(&mut address[19]),
                pointer(&mut address[20]),
                pointer(&mut address[21]),
                pointer(&mut address[22]),
                pointer(&mut address[23]),
                pointer(&mut address[24]),
                pointer(&mut address[25]),
                pointer(&mut address[26]),
                pointer(&mut address[27]),
                pointer(&mut address[28]),
                pointer(&mut address[29]),
                pointer(&mut address[30]),
                &mut frames as *mut i32 as *mut c_void,
            ];
            self.last_timing_ms[0] = phase.elapsed().as_secs_f64() * 1000.;
            let phase = std::time::Instant::now();
            // One default stream orders full-frame prediction -> detached seeds ->
            // cached directional propagation -> mixed reverse. No mid-tile host sync.
            self.launch_grid(0, slots.div_ceil(64), 64, &mut args)?;
            self.launch_grid(2, n.div_ceil(32), 32, &mut args)?;
            self.launch_grid(3, slots.div_ceil(64), 64, &mut args)?;
            self.launch_grid(4, f, 256, &mut args)?;
            self.launch_grid(5, f.div_ceil(32), 32, &mut args)?;
            self.launch_grid(6, slots.div_ceil(64), 64, &mut args)?;
            self.launch_grid(7, slots.div_ceil(64), 64, &mut args)?;
            self.launch_grid(8, n.div_ceil(32), 32, &mut args)?;
            self.launch_grid(9, slots.div_ceil(64), 64, &mut args[..16])?;
            self.launch_grid(10, (n * p).div_ceil(64), 64, &mut args[..16])?;
            check((self.api.sync)(), "cuCtxSynchronize(device loss)")?;
            self.last_timing_ms[1] = phase.elapsed().as_secs_f64() * 1000.;
            let phase = std::time::Instant::now();
            let mut status = [0_i32];
            Self::download_into(self.api, self.buffers[8].0, &mut status)?;
            if status[0] != 0 {
                return Err(Error(
                    "nonfinite CUDA prediction, loss seed or mixed gradient".into(),
                ));
            }
            let mut gradient = vec![0.; n * p];
            let mut seeds = vec![0.; f * 11];
            Self::download_into(self.api, self.buffers[6].0, &mut gradient)?;
            Self::download_into(self.api, self.buffers[27].0, &mut seeds)?;
            let diagnostics = if capture {
                let mut forces = vec![0.; n * 3];
                let mut predictions = vec![0.; f * 10];
                let mut force_seeds = vec![0.; n * 3];
                let mut directions = vec![0.; t.rows.len()];
                Self::download_into(self.api, self.buffers[24].0, &mut forces)?;
                Self::download_into(self.api, self.buffers[25].0, &mut predictions)?;
                Self::download_into(self.api, self.buffers[26].0, &mut force_seeds)?;
                Self::download_into(self.api, self.buffers[5].0, &mut directions)?;
                Some(DeviceLossDiagnostics {
                    frame_bounds: t.bounds.clone(),
                    frame_predictions: predictions,
                    forces,
                    frame_seeds: Vec::new(),
                    force_seeds,
                    directions,
                })
            } else {
                None
            };
            self.last_timing_ms[2] = phase.elapsed().as_secs_f64() * 1000.;
            Ok(DeviceLossOutput {
                gradient,
                frame_seeds: seeds,
                diagnostics,
            })
        }
    }
}
