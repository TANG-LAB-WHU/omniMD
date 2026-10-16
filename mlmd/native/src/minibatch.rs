//! Deterministic epoch-local sampling. No model, loss, or CUDA state lives here.
use omnimd_ml_contract::{Error, Result};
use serde::{Deserialize, Serialize};
const ALGORITHM: &str = "omnimd-epoch-splitmix64-v1";
const GAMMA: u64 = 0x9e3779b97f4a7c15;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchOrder {
    Sequential,
    ShuffleEachEpoch,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MiniBatchConfig {
    pub batch_size: usize,
    pub seed: u64,
    pub order: BatchOrder,
}
impl MiniBatchConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.batch_size == 0 || self.batch_size > 1024 {
            return Err(Error("minibatch batch_size must be 1..1024".into()));
        }
        Ok(())
    }
}
/// Canonical next-batch state, saved only with a completed optimizer update.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplerState {
    pub algorithm: String,
    pub seed: u64,
    pub order_mode: BatchOrder,
    pub epoch: u64,
    pub cursor: usize,
    pub order: Vec<usize>,
    /// Derived SplitMix64 state after generating this epoch's permutation.
    pub rng_state: u64,
}
fn epoch_state(c: &MiniBatchConfig, training: &[usize], epoch: u64) -> SamplerState {
    let mut source = c.seed.wrapping_add(GAMMA.wrapping_mul(epoch));
    let mut rng_state = crate::training::next(&mut source);
    let mut order = training.to_vec();
    if c.order == BatchOrder::ShuffleEachEpoch {
        for i in (1..order.len()).rev() {
            let bound = (i + 1) as u64;
            let threshold = bound.wrapping_neg() % bound;
            let value = loop {
                let v = crate::training::next(&mut rng_state);
                if v >= threshold {
                    break v;
                }
            };
            order.swap(i, (value % bound) as usize);
        }
    }
    SamplerState {
        algorithm: ALGORITHM.into(),
        seed: c.seed,
        order_mode: c.order,
        epoch,
        cursor: 0,
        order,
        rng_state,
    }
}
impl SamplerState {
    pub fn for_step(c: &MiniBatchConfig, training: &[usize], step: u64) -> Result<Self> {
        c.validate()?;
        if training.is_empty() || training.len() > 1024 || step > 1_000_000_000 {
            return Err(Error("invalid minibatch training count/global step".into()));
        }
        if training.windows(2).any(|x| x[0] >= x[1]) {
            return Err(Error(
                "minibatch training indices must be strictly increasing".into(),
            ));
        }
        let batches = training.len().div_ceil(c.batch_size) as u64;
        let mut state = epoch_state(c, training, step / batches);
        state.cursor = (step % batches) as usize * c.batch_size;
        Ok(state)
    }
    pub fn validate(&self, c: &MiniBatchConfig, training: &[usize], step: u64) -> Result<()> {
        if *self != Self::for_step(c, training, step)? {
            return Err(Error(
                "checkpoint sampler algorithm, seed, epoch, cursor, order or RNG state mismatch"
                    .into(),
            ));
        }
        Ok(())
    }
    pub fn batch(&self, c: &MiniBatchConfig) -> Result<&[usize]> {
        c.validate()?;
        if self.cursor >= self.order.len() {
            return Err(Error("invalid minibatch cursor".into()));
        }
        let end = self
            .cursor
            .saturating_add(c.batch_size)
            .min(self.order.len());
        Ok(&self.order[self.cursor..end])
    }
    pub(crate) fn advance(&mut self, c: &MiniBatchConfig, training: &[usize]) -> Result<()> {
        let count = self.batch(c)?.len();
        if self.cursor + count == self.order.len() {
            let epoch = self
                .epoch
                .checked_add(1)
                .ok_or_else(|| Error("minibatch epoch overflow".into()))?;
            *self = epoch_state(c, training, epoch);
        } else {
            self.cursor += count;
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct MiniBatchReport {
    pub normalization: String,
    pub start: SamplerState,
    pub end: SamplerState,
    pub frames_processed: u64,
    pub partial_batches: u64,
    pub batch_size_counts: std::collections::BTreeMap<usize, u64>,
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn golden_shuffle_vectors() {
        let c = MiniBatchConfig {
            batch_size: 2,
            seed: 20261016,
            order: BatchOrder::ShuffleEachEpoch,
        };
        for (epoch, order, rng) in [
            (0, [4, 2, 7, 5, 0], 13274578833983480863),
            (1, [0, 2, 5, 7, 4], 18024458041679499801),
            (2, [4, 5, 2, 7, 0], 1678407417306061623),
        ] {
            let s = SamplerState::for_step(&c, &[0, 2, 4, 5, 7], epoch * 3).unwrap();
            assert_eq!(s.order, order);
            assert_eq!(s.rng_state, rng);
        }
    }
    #[test]
    fn canonical_partial_batches_and_bounded_validation() {
        let indices = [0, 2, 4, 5, 7];
        for size in [1, 2, 5, 8] {
            let c = MiniBatchConfig {
                batch_size: size,
                seed: u64::MAX,
                order: BatchOrder::Sequential,
            };
            let mut s = SamplerState::for_step(&c, &indices, 0).unwrap();
            for step in 0..20 {
                s.validate(&c, &indices, step).unwrap();
                let batch = s.batch(&c).unwrap();
                assert!(!batch.is_empty());
                assert!(batch.len() <= size);
                s.advance(&c, &indices).unwrap();
            }
        }
        for size in [0, 1025] {
            assert!(SamplerState::for_step(
                &MiniBatchConfig {
                    batch_size: size,
                    seed: 0,
                    order: BatchOrder::Sequential
                },
                &indices,
                0
            )
            .is_err());
        }
    }
}
