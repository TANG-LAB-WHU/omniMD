//! Optional real CUDA FP64 network/descriptor inference; CPU geometry/reduction.
mod runtime;
use crate::{
    neighbor::{Geometry, Neighbor},
    Activation, Model, NativePotential, Residual,
};
use omnimd_ml_contract::{Error, Evaluation, Frame, ModelMetadata, Potential, Result, Units};
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
        for (net_index, net) in nets.iter().enumerate() {
            if net.layers.len() > 8 {
                return Err(Error(
                    "CUDA subset supports at most 8 layers per network".into(),
                ));
            }
            ml = ml.max(net.layers.len());
            metadata[net_index] = metadata.len() as i32;
            metadata.push(net.layers.len() as i32);
            for layer in &net.layers {
                let ni = layer.weights.len();
                let no = layer.bias.len();
                mw = mw.max(ni).max(no);
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
        let source=format!("#define NT {nt}\n#define NSLOTS {slots}\n#define EW {width}\n#define AXIS {}\n#define MW {mw}\n#define ML {ml}\n{}",d.axis_neuron,include_str!("kernels.cu"));
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

/// Deliberately explicit backend selection. A CUDA failure returns an error and
/// poisons this instance; use NativePotential explicitly for CPU fallback.
pub struct CudaPotential {
    cpu: NativePotential,
    metadata: ModelMetadata,
    packed: PackedModel,
    runtime: Runtime,
    poisoned: bool,
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
        })
    }
    pub fn device_info(&self) -> &str {
        &self.runtime.device_info
    }
    fn evaluate(&mut self, frame: &Frame) -> Result<Evaluation> {
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
        let s = self.packed.slots;
        let mut rows = Vec::with_capacity(n * s * 4);
        let mut records: Vec<(usize, Neighbor, [[f64; 3]; 4])> = Vec::new();
        for center in 0..n {
            let groups = geometry.neighbors(frame, &types, center, &d.sel)?;
            for (species, group) in groups.iter().enumerate() {
                for slot in 0..d.sel[species] {
                    let neighbor = group.get(slot);
                    let (r, jac) = crate::environment(
                        neighbor,
                        &d.normalization[types[center] as usize],
                        d.rcut_smth,
                        d.rcut,
                    );
                    let q = rows.len() / 4;
                    rows.extend(r);
                    if let Some(neighbor) = neighbor {
                        records.push((q, neighbor.clone(), jac));
                    }
                }
            }
        }
        let kinds: Vec<i32> = types.iter().map(|v| *v as i32).collect();
        let (energies, dr) = self.runtime.evaluate(&rows, &kinds, s, self.packed.width)?;
        let mut out = Evaluation {
            energy: 0.0,
            forces: vec![[0.0; 3]; n],
            virial: [[0.0; 3]; 3],
            units: Units::EvAngstrom,
        };
        for (i, energy) in energies.iter().enumerate() {
            out.energy += energy + self.cpu.model.atom_energy_bias[types[i] as usize];
        }
        for (q, neighbor, jac) in records {
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
