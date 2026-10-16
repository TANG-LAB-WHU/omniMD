//! Exact affine readout columns using a private reusable CUDA potential.
use super::{CudaPotential, PackedModel};
use crate::{
    readout::{ReadoutBasis, ReadoutFeatures, ReadoutParameter},
    Model,
};
use omnimd_ml_contract::{Error, Frame, Potential, Result};

// Map the already validated final affine layers into the packed resident weights.
fn coefficient_offsets(
    packed: &PackedModel,
    model: &Model,
    parameters: &[ReadoutParameter],
) -> Vec<usize> {
    let nt = model.type_map.len();
    parameters
        .iter()
        .map(|parameter| {
            let species = match *parameter {
                ReadoutParameter::Weight { species, .. } | ReadoutParameter::Bias { species } => {
                    species
                }
            };
            let base = packed.metadata[nt * nt + species] as usize;
            let layers = packed.metadata[base] as usize;
            let last = base + 1 + (layers - 1) * 7;
            match *parameter {
                ReadoutParameter::Weight { input, .. } => {
                    packed.metadata[last + 2] as usize + input
                }
                ReadoutParameter::Bias { .. } => packed.metadata[last + 3] as usize,
            }
        })
        .collect()
}

/// GPU-assisted frozen-final-readout features, not full-network GPU gradients.
/// Temporary one-hot weights are private and never exposed as a usable model.
pub struct CudaReadoutBasis {
    layout: ReadoutBasis,
    gpu: CudaPotential,
    offsets: Vec<usize>,
    selected: Option<usize>,
}
impl CudaReadoutBasis {
    pub fn new(model: &Model, fit_bias: bool, visible_device: i32) -> Result<Self> {
        if visible_device < 0 {
            return Err(Error(
                "CUDA visible device index must be nonnegative".into(),
            ));
        }
        let layout = ReadoutBasis::new(model, fit_bias)?;
        // Packed validation precedes any runtime/driver creation.
        let gpu = CudaPotential::new(layout.zero_model().clone(), visible_device)?;
        let offsets = coefficient_offsets(&gpu.packed, layout.zero_model(), &layout.parameters);
        Ok(Self {
            layout,
            gpu,
            offsets,
            selected: None,
        })
    }
    pub fn device_info(&self) -> &str {
        self.gpu.device_info()
    }
    pub fn features(&mut self, frame: &Frame) -> Result<ReadoutFeatures> {
        if self.gpu.poisoned {
            return Err(Error(
                "CUDA readout instance stopped after an earlier failure; no automatic retry".into(),
            ));
        }
        let result = self.evaluate(frame);
        if result.is_err() {
            self.gpu.poisoned = true;
        }
        result
    }
    fn evaluate(&mut self, frame: &Frame) -> Result<ReadoutFeatures> {
        let fixed_energy = self.layout.fixed_energy(frame)?;
        let mut columns = Vec::with_capacity(self.offsets.len());
        for &offset in &self.offsets {
            self.gpu
                .runtime
                .select_readout_basis(self.selected, offset)?;
            self.selected = Some(offset);
            columns.push(self.gpu.eval(frame)?);
        }
        Ok(ReadoutFeatures {
            fixed_energy,
            columns,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resident_basis_offsets_reconstruct_cpu_columns_without_a_driver() {
        let mut model: Model =
            serde_json::from_str(include_str!("../../fixtures/full-tanh-v1.json")).unwrap();
        model.atom_energy_bias = vec![-9e4, 3e4];
        let count = model.fitting[1].layers.len();
        let hidden = &mut model.fitting[1].layers[count - 2];
        for row in &mut hidden.weights {
            row.truncate(2);
        }
        hidden.bias.truncate(2);
        if let Some(timestep) = &mut hidden.timestep {
            timestep.truncate(2);
        }
        hidden.residual = crate::Residual::None;
        model.fitting[1]
            .layers
            .last_mut()
            .unwrap()
            .weights
            .truncate(2);
        let frame: Frame =
            serde_json::from_str(include_str!("../../fixtures/frame.jsonl")).unwrap();
        for fit_bias in [false, true] {
            let mut basis = ReadoutBasis::new(&model, fit_bias).unwrap();
            let expected = basis.features(&frame).unwrap();
            let zero = basis.zero_model().clone();
            let packed = PackedModel::new(&zero).unwrap();
            let offsets = coefficient_offsets(&packed, &zero, &basis.parameters);
            assert_eq!(offsets.len(), expected.columns.len());
            // Independent packing of each CPU one-hot model checks exact offset,
            // species and bias selection, including all untouched weight values.
            for (column, &offset) in offsets.iter().enumerate() {
                let mut values = vec![0.0; offsets.len()];
                values[column] = 1.0;
                let mut one = basis.model_with_coefficients(&values).unwrap();
                one.atom_energy_bias.fill(0.0);
                if !fit_bias {
                    for net in &mut one.fitting {
                        net.layers.last_mut().unwrap().bias[0] = 0.0;
                    }
                }
                let actual = PackedModel::new(&one).unwrap();
                let mut selected = packed.weights.clone();
                selected[offset] = 1.0;
                assert_eq!(selected, actual.weights);
            }
        }
    }
}
