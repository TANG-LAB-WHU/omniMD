// OmniMD, an extensible molecular simulation engine
// Copyright (C) OmniMD's contributors — MIT license

//! Testing molecular dynamics of methane
use omnimd::input::Input;

use std::path::Path;
use std::sync::Once;
static START: Once = Once::new();

fn is_ci() -> bool {
    std::env::var("CI").map(|v| v == "true").unwrap_or(false)
}

#[test]
fn bonds_detection() {
    if is_ci() { return; }
    START.call_once(::env_logger::init);
    let path = Path::new(file!())
        .parent()
        .unwrap()
        .join("data")
        .join("md-methane")
        .join("nve.toml");
    let system = Input::new(path).unwrap().read_system().unwrap();
    assert_eq!(system.molecules().count(), 150);

    for molecule in system.molecules() {
        assert_eq!(molecule.bonds().len(), 4);
        assert_eq!(molecule.angles().len(), 6);
        assert_eq!(molecule.dihedrals().len(), 0);
    }
}

#[test]
fn constant_energy() {
    if is_ci() { return; }
    START.call_once(::env_logger::init);
    let path = Path::new(file!())
        .parent()
        .unwrap()
        .join("data")
        .join("md-methane")
        .join("nve.toml");
    let mut config = Input::new(path).unwrap().read().unwrap();

    let e_initial = config.system.total_energy();
    config.simulation.run(&mut config.system, config.nsteps);
    let e_final = config.system.total_energy();
    assert!(f64::abs((e_initial - e_final) / e_final) < 1e-2);
}
