//! Scratch planning for the opt-in cached forward / reverse dual executor.
use super::PackedModel;
use crate::Model;
use omnimd_ml_contract::{Error, Result};
const MEMORY_LIMIT: usize = 64 * 1024 * 1024;
#[derive(Clone, Debug, serde::Serialize)]
pub struct ReverseResources {
    pub atoms: usize,
    pub cache_bytes: usize,
    pub descriptor_adjoint_bytes: usize,
    pub slot_gradient_bytes: usize,
    pub peak_device_buffer_bytes: usize,
    pub policy_limit_bytes: usize,
}
pub(super) struct ReversePlan {
    pub cache_stride: usize,
    pub embedding_parameters: usize,
    pub parameter_network: Vec<i32>,
    slots: usize,
    width: usize,
    parameters: usize,
    metadata_bytes: usize,
    pub source: String,
}
fn count(net: &crate::Network) -> usize {
    net.layers
        .iter()
        .map(|l| {
            l.weights.iter().map(Vec::len).sum::<usize>()
                + l.bias.len()
                + l.timestep.as_ref().map_or(0, Vec::len)
        })
        .sum()
}
fn bytes_product(values: &[usize]) -> Result<usize> {
    values
        .iter()
        .try_fold(1usize, |a, b| a.checked_mul(*b))
        .ok_or_else(|| Error("CUDA reverse scratch size overflow".into()))
}
impl ReversePlan {
    pub fn new(model: &Model, packed: &PackedModel, reference_source: &str) -> Result<Self> {
        let nets: Vec<_> = model
            .descriptor
            .embedding
            .iter()
            .flatten()
            .chain(&model.fitting)
            .collect();
        let layers = nets.iter().map(|n| n.layers.len()).max().unwrap();
        let width = nets
            .iter()
            .flat_map(|n| &n.layers)
            .map(|l| l.weights.len().max(l.bias.len()))
            .max()
            .unwrap();
        let cache_stride = (4 * layers + 2) * width;
        let embedding_parameters = model
            .descriptor
            .embedding
            .iter()
            .flatten()
            .map(count)
            .max()
            .unwrap();
        let parameter_network = nets
            .iter()
            .enumerate()
            .flat_map(|(i, n)| std::iter::repeat_n(i as i32, count(n)))
            .collect::<Vec<_>>();
        if parameter_network.len() != packed.weights.len() {
            return Err(Error("CUDA reverse parameter routing mismatch".into()));
        }
        // Compile a separate module. The reference source/identity stays unchanged.
        let source=format!("{reference_source}\n#define RL {layers}\n#define RC {cache_stride}\n#define EP {embedding_parameters}\n{}",include_str!("reverse_kernels.cu"));
        Ok(Self {
            cache_stride,
            embedding_parameters,
            parameter_network,
            slots: packed.slots,
            width: packed.width,
            parameters: packed.weights.len(),
            metadata_bytes: packed.metadata.len() * 4,
            source,
        })
    }
    pub fn tiled_source(&self) -> Result<String> {
        let argument = "int atoms,int parameters,double energy_seed,";
        let seed = "adj[0]=energy_seed;";
        if self.source.matches(argument).count() != 1 || self.source.matches(seed).count() != 1 {
            return Err(Error(
                "CUDA tiled seed specialization source mismatch".into(),
            ));
        }
        Ok(self
            .source
            .replacen(
                argument,
                "int atoms,int parameters,const double *energy_seed,",
                1,
            )
            .replacen(seed, "adj[0]=energy_seed[atom];", 1))
    }
    pub fn resources(&self, n: usize) -> Result<ReverseResources> {
        self.resources_impl(n, false)
    }
    pub fn tiled_resources(&self, n: usize) -> Result<ReverseResources> {
        self.resources_impl(n, true)
    }
    fn resources_impl(&self, n: usize, tiled: bool) -> Result<ReverseResources> {
        if n == 0 || n > if tiled { 1024 } else { 64 } {
            return Err(Error(
                "CUDA reverse atom count exceeds execution bound".into(),
            ));
        }
        let cache_bytes = bytes_product(&[n, self.slots + 1, self.cache_stride, 8])?;
        let descriptor_adjoint_bytes = bytes_product(&[n, self.width, 8, 8])?;
        let slot_gradient_bytes = bytes_product(&[n, self.slots, self.embedding_parameters, 8])?;
        let parts = [
            cache_bytes,
            descriptor_adjoint_bytes,
            slot_gradient_bytes,
            bytes_product(&[n, self.slots, 4, 8, 2])?, // environment value and directional rows
            bytes_product(&[n, self.parameters, 8])?,
            n * 8,                   // atom gradients and energies
            self.parameters * 8 * 4, // weights, two moments, uploaded batch gradient
            self.metadata_bytes,
            self.parameters * 4,
            n * 4,
            self.slots * 4,
            if tiled { n * 8 } else { 0 }, // per-atom energy seeds
            4,
        ];
        let peak = parts
            .iter()
            .try_fold(0usize, |a, b| a.checked_add(*b))
            .ok_or_else(|| Error("CUDA reverse resident size overflow".into()))?;
        if peak > MEMORY_LIMIT {
            return Err(Error(format!("CUDA reverse device buffer budget exceeded: {peak} bytes > {MEMORY_LIMIT}; no fallback")));
        }
        Ok(ReverseResources {
            atoms: n,
            cache_bytes,
            descriptor_adjoint_bytes,
            slot_gradient_bytes,
            peak_device_buffer_bytes: peak,
            policy_limit_bytes: MEMORY_LIMIT,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checked_peak_sizes_and_parameter_routing_without_cuda() {
        assert!(bytes_product(&[usize::MAX, 8]).is_err());
        let m: Model =
            serde_json::from_str(include_str!("../../fixtures/full-tanh-v1.json")).unwrap();
        let (p, _, s) = super::super::full::packed(&m).unwrap();
        let mut plan = ReversePlan::new(&m, &p, &s).unwrap();
        assert_eq!(plan.parameter_network.len(), 112);
        assert_eq!(plan.embedding_parameters, 14);
        let r = plan.resources(64).unwrap();
        assert!(r.peak_device_buffer_bytes < 1024 * 1024);
        assert!(
            r.peak_device_buffer_bytes
                > r.cache_bytes + r.slot_gradient_bytes + r.descriptor_adjoint_bytes
        );
        plan.cache_stride = usize::MAX;
        assert!(plan.resources(64).is_err());
        plan.cache_stride = 1_000_000;
        assert!(plan.resources(64).unwrap_err().0.contains("budget"));
    }
}
