//! Explicit native optimizer-update schedules; no DeepMD default equivalence.
use omnimd_ml_contract::{Error, Result};
use serde::{Deserialize, Serialize};

pub(crate) const MAX_UPDATES: u64 = 1_000_000_000;

/// k is the number of completed Adam updates BEFORE the next update.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LearningRateSchedule {
    Constant {},
    /// lr(k) = base * decay_rate ^ floor(k / decay_steps).
    Exponential {
        decay_steps: u64,
        decay_rate: f64,
    },
}
impl LearningRateSchedule {
    pub(crate) fn validate(&self) -> Result<()> {
        if let Self::Exponential {
            decay_steps,
            decay_rate,
        } = self
        {
            if *decay_steps == 0
                || *decay_steps > MAX_UPDATES
                || !decay_rate.is_finite()
                || *decay_rate <= 0.0
                || *decay_rate > 1.0
            {
                return Err(Error("exponential learning-rate schedule requires decay_steps in 1..1000000000 and finite decay_rate in (0,1]".into()));
            }
        }
        Ok(())
    }
    pub(crate) fn rate(&self, base: f64, completed_updates: u64) -> Result<f64> {
        self.validate()?;
        if completed_updates > MAX_UPDATES || !base.is_finite() || base <= 0.0 {
            return Err(Error(
                "invalid learning-rate base or completed update count".into(),
            ));
        }
        let rate = match self {
            Self::Constant {} => base,
            Self::Exponential {
                decay_steps,
                decay_rate,
            } => {
                let exponent = completed_updates / decay_steps;
                // The global step bound makes this conversion exact and safe.
                // Integer exponent, no interpolation or repeated mutable LR state.
                base * powi_portable(*decay_rate, exponent)
            }
        };
        if !rate.is_finite() || rate <= 0.0 {
            return Err(Error(format!("learning-rate schedule is nonpositive/nonfinite at completed update {completed_updates} (possible float64 underflow)")));
        }
        Ok(rate)
    }
}

/// `f64::powi` has unspecified precision and differs between toolchains (MSVC
/// does not underflow 0.5^1075 to zero). This is compiler-rt's `__powidf2`
/// square-and-multiply order, which the gnu toolchain used for all recorded
/// schedules, written out so every platform rounds identically.
fn powi_portable(mut factor: f64, mut exponent: u64) -> f64 {
    let mut result = 1.0;
    loop {
        if exponent & 1 == 1 {
            result *= factor;
        }
        exponent /= 2;
        if exponent == 0 {
            return result;
        }
        factor *= factor;
    }
}

#[derive(Serialize)]
pub struct LearningRateReport {
    pub contract: &'static str,
    pub schedule: LearningRateSchedule,
    pub base_rate: f64,
    /// One-based update numbers actually performed in this invocation.
    pub first_update: u64,
    pub last_update: u64,
    pub first_rate: f64,
    pub last_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn staircase_boundaries_and_strict_configuration() {
        let schedule = LearningRateSchedule::Exponential {
            decay_steps: 3,
            decay_rate: 0.5,
        };
        let expected = [0.125, 0.125, 0.125, 0.0625, 0.0625, 0.0625, 0.03125];
        for (k, value) in expected.into_iter().enumerate() {
            assert_eq!(schedule.rate(0.125, k as u64).unwrap(), value);
        }
        for bad in [
            r#"{"kind":"cosine"}"#,
            r#"{"kind":"constant","decay_rate":0.5}"#,
            r#"{"kind":"exponential","decay_steps":2}"#,
            r#"{"kind":"exponential","decay_steps":2.5,"decay_rate":0.5}"#,
            r#"{"kind":"exponential","decay_steps":2,"decay_rate":0.5,"staircase":false}"#,
        ] {
            assert!(
                serde_json::from_str::<LearningRateSchedule>(bad).is_err(),
                "{bad}"
            );
        }
        for (decay_steps, decay_rate) in [
            (0, 0.5),
            (MAX_UPDATES + 1, 0.5),
            (1, 0.0),
            (1, -0.1),
            (1, 1.01),
            (1, f64::NAN),
            (1, f64::INFINITY),
        ] {
            assert!(LearningRateSchedule::Exponential {
                decay_steps,
                decay_rate
            }
            .rate(0.01, 0)
            .is_err());
        }
    }
    #[test]
    fn bounds_underflow_and_constant_preserve_float64_values() {
        let decay = LearningRateSchedule::Exponential {
            decay_steps: 1,
            decay_rate: 0.5,
        };
        assert_eq!(decay.rate(1.0, 1022).unwrap(), f64::MIN_POSITIVE);
        assert!(decay.rate(1.0, 1074).unwrap() > 0.0);
        assert!(decay.rate(1.0, 1075).is_err());
        for value in [f64::MIN_POSITIVE, f64::from_bits(1), 0.005, f64::MAX] {
            assert_eq!(
                LearningRateSchedule::Constant {}
                    .rate(value, MAX_UPDATES)
                    .unwrap()
                    .to_bits(),
                value.to_bits()
            );
            assert_eq!(
                LearningRateSchedule::Exponential {
                    decay_steps: 1,
                    decay_rate: 1.0
                }
                .rate(value, MAX_UPDATES)
                .unwrap()
                .to_bits(),
                value.to_bits()
            );
        }
        assert!(LearningRateSchedule::Constant {}
            .rate(1.0, MAX_UPDATES + 1)
            .is_err());
    }
}
