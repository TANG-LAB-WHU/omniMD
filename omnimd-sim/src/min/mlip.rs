//! Fixed-cell, fallible steepest descent using the upstream backtracking policy.
use super::steepest_descent::DescentStep;
use omnimd_ml_contract::{Error, Evaluation, Frame, ModelMetadata, Potential, Result};

#[derive(Debug, Clone)]
/// Explicit units and finite work limits for fixed-cell minimization.
pub struct MlipMinimizationSettings {
    /// Maximum per-atom force norm tolerance, eV/angstrom.
    pub force_tolerance_ev_per_angstrom: f64,
    /// Absolute total-energy change tolerance, eV.
    pub energy_tolerance_ev: f64,
    /// Initial multiplier of the force direction, angstrom squared/eV.
    pub initial_step_angstrom2_per_ev: f64,
    /// Maximum displacement norm of any atom in one trial.
    pub max_displacement_angstrom: f64,
    /// Maximum halvings after the first trial, in 1..=128.
    pub max_backtracks: u32,
    /// Maximum number of accepted moves.
    pub max_iterations: u64,
    /// Sufficient-decrease coefficient, strictly between zero and one.
    pub armijo: f64,
}
impl MlipMinimizationSettings {
    /// Reject invalid settings before evaluating a model.
    pub fn validate(&self) -> Result<()> {
        if [
            self.force_tolerance_ev_per_angstrom,
            self.energy_tolerance_ev,
            self.initial_step_angstrom2_per_ev,
            self.max_displacement_angstrom,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.)
            || !self.armijo.is_finite()
            || self.armijo <= 0.
            || self.armijo >= 1.
            || self.max_iterations == 0
            || self.max_backtracks == 0
            || self.max_backtracks > 128
        {
            return Err(Error("minimization tolerances/step/displacement must be positive finite; armijo in (0,1); iterations>0; backtracks in 1..=128".into()));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Running state or explicit termination reason.
pub enum MlipMinimizationStatus {
    /// More accepted steps may be attempted.
    Running,
    /// Initial force satisfies tolerance; no energy-change claim.
    AlreadyConverged,
    /// Both force and accepted energy-change criteria hold.
    Converged,
    /// Accepted-step budget exhausted before convergence.
    MaxIterations,
    /// No acceptable energy-decreasing trial within the backtracking budget.
    LineSearchFailed,
    /// No representable coordinate change; convergence is not inferred.
    Stalled,
}
impl MlipMinimizationStatus {
    /// Stable machine-readable status label.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::AlreadyConverged => "already_converged",
            Self::Converged => "converged",
            Self::MaxIterations => "max_iterations",
            Self::LineSearchFailed => "line_search_failed",
            Self::Stalled => "stalled",
        }
    }
    /// Whether this is a verified convergence outcome.
    pub fn converged(self) -> bool {
        matches!(self, Self::AlreadyConverged | Self::Converged)
    }
}
fn max_force(evaluation: &Evaluation) -> Result<f64> {
    let value = evaluation
        .forces
        .iter()
        .map(|f| f[0].hypot(f[1]).hypot(f[2]))
        .fold(0., f64::max);
    if !value.is_finite() {
        return Err(Error("nonfinite force norm".into()));
    }
    Ok(value)
}
/// Owns one potential and atomically updates accepted geometry and E/F/virial.
pub struct MlipMinimizer<P: Potential> {
    potential: P,
    frame: Frame,
    evaluation: Evaluation,
    settings: MlipMinimizationSettings,
    step: DescentStep,
    status: MlipMinimizationStatus,
    iterations: u64,
    evaluations: u64,
    energy_change: Option<f64>,
    max_force: f64,
}
impl<P: Potential> MlipMinimizer<P> {
    /// Validate inputs and evaluate the initial geometry once.
    pub fn new(mut potential: P, frame: Frame, settings: MlipMinimizationSettings) -> Result<Self> {
        settings.validate()?;
        let _ = frame.type_ids(potential.metadata())?;
        let evaluation = potential.eval(&frame)?;
        evaluation.validate(frame.positions.len())?;
        let max_force = max_force(&evaluation)?;
        let status = if max_force <= settings.force_tolerance_ev_per_angstrom {
            MlipMinimizationStatus::AlreadyConverged
        } else {
            MlipMinimizationStatus::Running
        };
        Ok(Self {
            potential,
            frame,
            evaluation,
            step: DescentStep::new(settings.initial_step_angstrom2_per_ev),
            settings,
            status,
            iterations: 0,
            evaluations: 1,
            energy_change: None,
            max_force,
        })
    }
    /// Last accepted geometry.
    pub fn frame(&self) -> &Frame {
        &self.frame
    }
    /// Combined E/F/virial at the last accepted geometry.
    pub fn evaluation(&self) -> &Evaluation {
        &self.evaluation
    }
    /// Metadata of the single loaded potential.
    pub fn metadata(&self) -> &ModelMetadata {
        self.potential.metadata()
    }
    /// Current running or termination status.
    pub fn status(&self) -> MlipMinimizationStatus {
        self.status
    }
    /// Number of accepted coordinate moves.
    pub fn iterations(&self) -> u64 {
        self.iterations
    }
    /// Number of model evaluation attempts, including the initial frame.
    pub fn evaluations(&self) -> u64 {
        self.evaluations
    }
    /// Signed energy change of the last accepted move; initially absent.
    pub fn energy_change_ev(&self) -> Option<f64> {
        self.energy_change
    }
    /// Largest Cartesian force norm over atoms at the accepted geometry.
    pub fn max_force_ev_per_angstrom(&self) -> f64 {
        self.max_force
    }

    /// At most max_backtracks+1 trials; accepted geometry/output changes together.
    /// Model errors abort immediately, preserving the last accepted frame/output.
    pub fn step(&mut self) -> Result<MlipMinimizationStatus> {
        if self.status != MlipMinimizationStatus::Running {
            return Ok(self.status);
        }
        let force2 = self.evaluation.forces.iter().flatten().map(|f| f * f).sum::<f64>();
        if !force2.is_finite() {
            return Err(Error("nonfinite descent slope".into()));
        }
        let mut trial_step = DescentStep::new(
            self.step.value.min(self.settings.max_displacement_angstrom / self.max_force),
        );
        for backtracks in 0..=self.settings.max_backtracks {
            if trial_step.value == 0. {
                self.status = MlipMinimizationStatus::Stalled;
                return Ok(self.status);
            }
            let mut frame = self.frame.clone();
            for (position, force) in frame.positions.iter_mut().zip(&self.evaluation.forces) {
                for axis in 0..3 {
                    position[axis] += trial_step.value * force[axis];
                }
            }
            if frame.positions == self.frame.positions {
                self.status = MlipMinimizationStatus::Stalled;
                return Ok(self.status);
            }
            frame.validate()?;
            self.evaluations += 1;
            let evaluation = self.potential.eval(&frame)?;
            evaluation.validate(frame.positions.len())?;
            let delta = evaluation.energy - self.evaluation.energy;
            let required = self.settings.armijo * trial_step.value * force2;
            if !delta.is_finite() || !required.is_finite() {
                return Err(Error("nonfinite line-search energy difference".into()));
            }
            // Difference form avoids cancellation when forming E0 - required.
            if delta <= -required && delta < 0. {
                let max_force = max_force(&evaluation)?;
                self.frame = frame;
                self.evaluation = evaluation;
                self.energy_change = Some(delta);
                self.max_force = max_force;
                self.iterations += 1;
                self.step = trial_step;
                if backtracks == 0 {
                    self.step.grow();
                }
                if max_force <= self.settings.force_tolerance_ev_per_angstrom
                    && delta.abs() <= self.settings.energy_tolerance_ev
                {
                    self.status = MlipMinimizationStatus::Converged;
                } else if self.iterations >= self.settings.max_iterations {
                    self.status = MlipMinimizationStatus::MaxIterations;
                }
                return Ok(self.status);
            }
            trial_step.shrink();
        }
        self.status = MlipMinimizationStatus::LineSearchFailed;
        Ok(self.status)
    }
}
