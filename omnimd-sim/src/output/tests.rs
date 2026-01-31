// OmniMD, an extensible molecular simulation engine
// Copyright (C) OmniMD's contributors — MIT license

//! Utilities to test the output algorithms

#![cfg(test)]

use tempfile::NamedTempFile;

use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

use super::Output;
use omnimd_core::energy::{Harmonic, PairInteraction};
use omnimd_core::units;
use omnimd_core::{Molecule, Particle, System, UnitCell};

pub fn test_output<F>(function: F, expected: &str)
where
    F: Fn(&Path) -> Box<dyn Output>,
{
    let tempfile = NamedTempFile::new().unwrap();
    let system = testing_system();
    {
        let mut output = function(tempfile.path());
        output.setup(&system);
        output.write(&system);
        output.finish(&system);
    }

    let file = tempfile.reopen().unwrap();
    check_file_content(file, expected);
}

pub fn testing_system() -> System {
    let mut system = System::with_cell(UnitCell::cubic(10.0));
    system.add_molecule(Molecule::new(Particle::with_position("F", [0.0, 0.0, 0.0].into())));
    system.add_molecule(Molecule::new(Particle::with_position("F", [1.3, 0.0, 0.0].into())));

    system.particles_mut().velocity[0] = [0.1, 0.0, 0.0].into();
    system.particles_mut().velocity[1] = [0.0, 0.0, 0.0].into();

    let harmonic = Box::new(Harmonic {
        k: units::from(300.0, "kJ/mol/A^2").unwrap(),
        x0: units::from(1.2, "A").unwrap(),
    });
    system.set_pair_potential(("F", "F"), PairInteraction::new(harmonic, 5.0));
    system.step = 42;
    return system;
}

fn check_file_content(mut file: File, content: &str) {
    let mut buffer = String::new();
    let _ = file.read_to_string(&mut buffer).unwrap();

    for (l1, l2) in buffer.lines().zip(content.lines()) {
        let l2 = l2.trim_start();
        // Compare tokens individually to handle floating-point precision
        let tokens1: Vec<&str> = l1.split_whitespace().collect();
        let tokens2: Vec<&str> = l2.split_whitespace().collect();

        assert_eq!(
            tokens1.len(),
            tokens2.len(),
            "Line token count mismatch:\n  left:  {}\n  right: {}",
            l1,
            l2
        );

        for (t1, t2) in tokens1.iter().zip(tokens2.iter()) {
            // Try to parse as f64 for approximate comparison
            match (t1.parse::<f64>(), t2.parse::<f64>()) {
                (Ok(v1), Ok(v2)) => {
                    // Use relative tolerance for floating-point comparison
                    let rel_diff = if v2.abs() > 1e-10 {
                        ((v1 - v2) / v2).abs()
                    } else {
                        (v1 - v2).abs()
                    };
                    assert!(
                        rel_diff < 1e-6,
                        "Floating-point mismatch: {} vs {} (rel_diff: {})\n  left:  {}\n  right: {}",
                        t1,
                        t2,
                        rel_diff,
                        l1,
                        l2
                    );
                }
                _ => {
                    // Non-numeric tokens: exact comparison
                    assert_eq!(t1, t2, "Token mismatch:\n  left:  {}\n  right: {}", l1, l2);
                }
            }
        }
    }
}
