//! Optional real CUDA FP64 network/descriptor inference; CPU geometry/reduction.
mod neighbor_cache;
mod runtime;
use crate::{
    neighbor::{Geometry, Neighbor},
    Activation, Model, NativePotential, Residual,
};
use omnimd_ml_contract::{Error, Evaluation, Frame, ModelMetadata, Potential, Result, Units};
pub use runtime::KernelResources;
use runtime::Runtime;

pub struct PackedModel {
    pub source: String,
    metadata: Vec<i32>,
    weights: Vec<f64>,
    slot_types: Vec<i32>,
    width: usize,
    slots: usize,
}
impl PackedModel {
    pub fn new(model: &Model) -> Result<Self> {
        model.validate()?;
        let nt = model.type_map.len();
        let d = &model.descriptor;
        let slots = d.sel.iter().sum::<usize>();
        let width = d.embedding[0][0].output_size();
        if nt > 8 || slots > 256 || width > 64 || width * d.axis_neuron > 128 {
            return Err(Error("CUDA subset requires types<=8, sum(sel)<=256, embedding width<=64, descriptor width<=128".into()));
        }
        let nets: Vec<_> = d.embedding.iter().flatten().chain(&model.fitting).collect();
        let mut metadata = vec![0; nets.len()];
        let mut weights = Vec::new();
        let mut mw = 1;
        let mut ml = 1;
        let mut embedding_mw = 1;
        let mut fitting_mw = 1;
        for (net_index, net) in nets.iter().enumerate() {
            if net.layers.len() > 8 {
                return Err(Error(
                    "CUDA subset supports at most 8 layers per network".into(),
                ));
            }
            if net_index >= nt * nt {
                ml = ml.max(net.layers.len());
            }
            metadata[net_index] = metadata.len() as i32;
            metadata.push(net.layers.len() as i32);
            for layer in &net.layers {
                let ni = layer.weights.len();
                let no = layer.bias.len();
                mw = mw.max(ni).max(no);
                if net_index < nt * nt {
                    embedding_mw = embedding_mw.max(ni).max(no);
                } else {
                    fitting_mw = fitting_mw.max(ni).max(no);
                }
                if mw > 128 {
                    return Err(Error("CUDA subset supports network widths<=128".into()));
                }
                let act = match layer.activation {
                    Activation::Linear => 0,
                    Activation::Tanh => 1,
                    _ => {
                        return Err(Error(
                            "CUDA subset supports only tanh/linear activations".into(),
                        ))
                    }
                };
                let wi = weights.len();
                weights.extend(layer.weights.iter().flatten());
                let bi = weights.len();
                weights.extend(&layer.bias);
                let ti = layer.timestep.as_ref().map_or(-1, |dt| {
                    let start = weights.len() as i32;
                    weights.extend(dt);
                    start
                });
                metadata.extend([
                    ni as i32,
                    no as i32,
                    wi as i32,
                    bi as i32,
                    ti,
                    act,
                    i32::from(!matches!(layer.residual, Residual::None)),
                ]);
                if weights.len() > 2_000_000 {
                    return Err(Error(
                        "CUDA subset supports at most two million weights/biases/timesteps".into(),
                    ));
                }
            }
        }
        let slot_types = d
            .sel
            .iter()
            .enumerate()
            .flat_map(|(i, n)| std::iter::repeat(i as i32).take(*n))
            .collect();
        let source=format!("#define NT {nt}\n#define NSLOTS {slots}\n#define EW {width}\n#define AXIS {}\n#define MW {fitting_mw}\n#define EMW {embedding_mw}\n#define ML {ml}\n{}",d.axis_neuron,include_str!("kernels.cu"));
        Ok(Self {
            source,
            metadata,
            weights,
            slot_types,
            width,
            slots,
        })
    }
    /// Uses installed NVRTC only. No CUDA context, driver call, or GPU execution.
    pub fn compile_only(&self) -> Result<usize> {
        Ok(runtime::compile(&self.source, 75)?.len())
    }
}

/// Host wall-clock segments, including synchronization; not isolated kernel events.
#[derive(Clone, Copy, Default, Debug, serde::Serialize)]
pub struct CudaProfile {
    pub prepare_ms: f64,
    pub neighbor_cache_rebuilt: bool,
    pub neighbor_cache_rebuilds: usize,
    pub neighbor_candidates: usize,
    pub upload_and_allocation_ms: f64,
    pub kernels_and_sync_ms: f64,
    pub download_ms: f64,
    pub reduction_ms: f64,
    pub type_uploads: usize,
    pub upload_bytes: usize,
    pub row_capacity: usize,
    pub record_capacity: usize,
}

/// Deliberately explicit backend selection. A CUDA failure returns an error and
/// poisons this instance; use NativePotential explicitly for CPU fallback.
pub struct CudaPotential {
    cpu: NativePotential,
    metadata: ModelMetadata,
    packed: PackedModel,
    runtime: Runtime,
    poisoned: bool,
    profile: CudaProfile,
    neighbor_cache: neighbor_cache::NeighborCache,
    rows: Vec<f64>,
    records: Vec<(usize, Neighbor, [[f64; 3]; 4])>,
}
impl CudaPotential {
    pub fn new(model: Model, device: i32) -> Result<Self> {
        let packed = PackedModel::new(&model)?;
        let cpu = NativePotential::new(model)?;
        let mut metadata = cpu.metadata().clone();
        metadata.backend = "native-cuda-se-e2-a-v1".into();
        let mut runtime = Runtime::new(&packed.source, device)?;
        runtime.upload(0, &packed.metadata)?;
        runtime.upload(1, &packed.weights)?;
        runtime.upload(4, &packed.slot_types)?;
        Ok(Self {
            cpu,
            metadata,
            packed,
            runtime,
            poisoned: false,
            profile: CudaProfile::default(),
            neighbor_cache: neighbor_cache::NeighborCache::new(),
            rows: Vec::new(),
            records: Vec::new(),
        })
    }
    pub fn device_info(&self) -> &str {
        &self.runtime.device_info
    }
    pub fn kernel_resources(&self) -> &[KernelResources] {
        &self.runtime.kernel_resources
    }
    pub fn last_profile(&self) -> CudaProfile {
        self.profile
    }
    fn evaluate(&mut self, frame: &Frame) -> Result<Evaluation> {
        let phase = std::time::Instant::now();
        frame.validate()?;
        let types = frame.type_ids(&self.metadata)?;
        let n = types.len();
        if n > 2048 {
            return Err(Error(
                "CUDA subset supports at most 2048 atoms per frame".into(),
            ));
        }
        let d = &self.cpu.model.descriptor;
        let geometry = Geometry::new(frame, d.rcut)?;
        self.profile.neighbor_cache_rebuilt =
            self.neighbor_cache.update(frame, &types, d.rcut, &d.sel);
        self.profile.neighbor_cache_rebuilds = self.neighbor_cache.rebuilds;
        self.profile.neighbor_candidates = self.neighbor_cache.candidate_count;
        let s = self.packed.slots;
        self.rows.clear();
        self.rows.reserve(n * s * 4);
        self.records.clear();
        for center in 0..n {
            let groups = self
                .neighbor_cache
                .neighbors(frame, &types, center, &d.sel, d.rcut, &geometry)?;
            for (species, group) in groups.iter().enumerate() {
                for slot in 0..d.sel[species] {
                    let neighbor = group.get(slot);
                    let (r, jac) = crate::environment(
                        neighbor,
                        &d.normalization[types[center] as usize],
                        d.rcut_smth,
                        d.rcut,
                    );
                    let q = self.rows.len() / 4;
                    self.rows.extend(r);
                    if let Some(neighbor) = neighbor {
                        self.records.push((q, neighbor.clone(), jac));
                    }
                }
            }
        }
        self.profile.row_capacity = self.rows.capacity();
        self.profile.record_capacity = self.records.capacity();
        self.profile.prepare_ms = phase.elapsed().as_secs_f64() * 1000.0;
        self.runtime
            .evaluate(&self.rows, &types, s, self.packed.width)?;
        self.profile.type_uploads = self.runtime.type_uploads;
        self.profile.upload_bytes = self.runtime.last_upload_bytes;
        let energies = &self.runtime.energies;
        let dr = &self.runtime.gradients;
        let [upload, kernels, download] = self.runtime.last_timing_ms;
        self.profile.upload_and_allocation_ms = upload;
        self.profile.kernels_and_sync_ms = kernels;
        self.profile.download_ms = download;
        let phase = std::time::Instant::now();
        let mut out = Evaluation {
            energy: 0.0,
            forces: vec![[0.0; 3]; n],
            virial: [[0.0; 3]; 3],
            units: Units::EvAngstrom,
        };
        for (i, energy) in energies.iter().enumerate() {
            out.energy += energy + self.cpu.model.atom_energy_bias[types[i] as usize];
        }
        for (q, neighbor, jac) in &self.records {
            let center = q / s;
            for c in 0..3 {
                let grad = (0..4).map(|a| dr[q * 4 + a] * jac[a][c]).sum::<f64>();
                out.forces[center][c] += grad;
                out.forces[neighbor.atom][c] -= grad;
                for a in 0..3 {
                    out.virial[a][c] -= neighbor.delta[a] * grad;
                }
            }
        }
        out.validate(n)?;
        self.profile.reduction_ms = phase.elapsed().as_secs_f64() * 1000.0;
        Ok(out)
    }
}
impl Potential for CudaPotential {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }
    fn eval(&mut self, frame: &Frame) -> Result<Evaluation> {
        if self.poisoned {
            return Err(Error(
                "CUDA instance stopped after an earlier evaluation failure; no automatic retry"
                    .into(),
            ));
        }
        let result = self.evaluate(frame);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packing_is_strict_and_does_not_require_cuda() {
        let mut model: Model =
            serde_json::from_str(include_str!("../../fixtures/full-tanh-v1.json")).unwrap();
        let packed = PackedModel::new(&model).unwrap();
        assert_eq!(packed.weights.len(), 112);
        assert_eq!(packed.slots, 16);
        assert!(packed.source.contains("#define MW 3"));
        assert_eq!(packed.slot_types, [[0; 8], [1; 8]].concat());
        model.descriptor.embedding[0][0].layers[0].activation = Activation::Softplus;
        assert!(PackedModel::new(&model).is_err());
    }
}
