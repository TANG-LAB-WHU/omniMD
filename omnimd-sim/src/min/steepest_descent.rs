// OmniMD, an extensible molecular simulation engine
// Copyright (C) OmniMD's contributors — MIT license

use soa_derive::soa_zip;
use std::f64;

use super::{Minimizer, Tolerance};
use omnimd_core::{units, DegreesOfFreedom, System};

/// Steepest descent minimization algorithm.
///
/// This method propagates the system along the gradient of energy to find a
/// minimum. Although easy to use, it will not converge in all situations.
pub struct SteepestDescent {
    /// Damping factor
    step: DescentStep,
}

impl SteepestDescent {
    /// Create a new `SteepestDescent` minimizer
    pub fn new() -> SteepestDescent {
        SteepestDescent {
            step: DescentStep::new(units::from(0.1, "fs^2/u").expect("bad unit")),
        }
    }
}

impl Minimizer for SteepestDescent {
    fn degrees_of_freedom(&self, _: &System) -> DegreesOfFreedom {
        DegreesOfFreedom::Particles
    }

    fn minimize(&mut self, system: &mut System) -> Tolerance {
        // Store the current coordinates
        let prevpos = system.particles().position.to_vec();

        let mut gamma_changed = false;
        let forces = system.forces();
        let initial_energy = system.potential_energy();
        let mut energy;
        // Update coordinates, reducing gamma until we find a configuration of
        // lower energy
        loop {
            for (position, prevpos, force) in
                soa_zip!(system.particles_mut(), [mut position], &prevpos, &forces)
            {
                *position = prevpos + self.step.value * force;
            }

            energy = system.potential_energy();
            if energy <= initial_energy {
                break;
            }
            self.step.shrink();
            gamma_changed = true;
        }

        // If we had a successful iteration without needing to reduce gamma,
        // we can increase it slightly
        if !gamma_changed {
            self.step.grow();
        }

        return Tolerance {
            energy: energy,
            force2: forces.iter().map(|&f| f.norm2()).fold(f64::NAN, f64::max),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use omnimd_core::{Harmonic, PairInteraction};
    use omnimd_core::{Molecule, Particle, System, UnitCell};

    use crate::min::Minimization;
    use crate::propagator::Propagator;

    use approx::assert_relative_eq;

    fn testing_system() -> System {
        let mut system = System::with_cell(UnitCell::cubic(20.0));
        system.add_molecule(Molecule::new(Particle::with_position("Cl", [0.0, 0.0, 0.0].into())));
        system.add_molecule(Molecule::new(Particle::with_position("Cl", [0.0, 0.0, 2.0].into())));

        let pair = PairInteraction::new(Box::new(Harmonic { x0: 2.3, k: 0.1 }), 10.0);
        system.set_pair_potential(("Cl", "Cl"), pair);
        return system;
    }

    #[test]
    fn minization() {
        let mut system = testing_system();

        let mut minization = Minimization::new(
            Box::new(SteepestDescent::new()),
            Tolerance {
                energy: 1e-10,
                force2: 1e-10,
            },
        );
        for _ in 0..100 {
            minization.propagate(&mut system);
        }
        assert!(minization.converged());
        assert_relative_eq!(system.distance(0, 1), 2.3, epsilon = 1e-3);
    }
}

/// Unit-neutral step-size policy shared by legacy and fallible steepest descent.
/// Callers supply the appropriate units and validate positive finite inputs.
pub(crate) struct DescentStep {
    pub(crate) value: f64,
}
impl DescentStep {
    pub(crate) fn new(value: f64) -> Self {
        Self { value }
    }
    pub(crate) fn shrink(&mut self) {
        self.value *= 0.5;
    }
    pub(crate) fn grow(&mut self) {
        self.value = (self.value * 1.1).min(std::primitive::f64::MAX);
    }
}
