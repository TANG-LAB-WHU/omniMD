// OmniMD, an extensible molecular simulation engine
// Copyright (C) OmniMD's contributors — MIT license

//! Testing physical properties of a Lennard-Jones gaz of Helium using
//! Monte Carlo simulation
use omnimd::consts::K_BOLTZMANN;
use omnimd::units;

use omnimd::input::Input;

use std::path::Path;
use std::sync::Once;
static START: Once = Once::new();

fn is_ci() -> bool {
    std::env::var("CI").map(|v| v == "true").unwrap_or(false)
}

#[test]
fn perfect_gas() {
    if is_ci() {
        return;
    }
    START.call_once(::env_logger::init);
    let path = Path::new(file!())
        .parent()
        .unwrap()
        .join("data")
        .join("mc-helium")
        .join("nvt.toml");

    let mut config = Input::new(path).unwrap().read().unwrap();
    config.simulation.run(&mut config.system, config.nsteps);
    let pressure = config.system.pressure();
    let volume = config.system.volume();
    let temperature = units::from(300.0, "K").unwrap();

    let pv = pressure * volume;
    let nkt = config.system.size() as f64 * K_BOLTZMANN * temperature;
    let msg = format!("{} {}", f64::abs(pv - nkt), f64::abs(pv - nkt) / pv);
    assert!(f64::abs(pv - nkt) / pv < 2e-2, "{}", msg);
}
