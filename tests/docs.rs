// OmniMD, an extensible molecular simulation engine
// Copyright (C) OmniMD's contributors — MIT license

//! Checking that the documentation tutorials run
//!
//! NOTE: These tests are ignored in CI due to SIGFPE in chemfiles library
//! initialization on GitHub Actions runners. Run locally to test these.
use omnimd::input::Input;

use std::path::Path;
use std::sync::Once;
static START: Once = Once::new();

/// Check if running in CI environment (GitHub Actions sets CI=true)
fn is_ci() -> bool {
    std::env::var("CI").map(|v| v == "true").unwrap_or(false)
}

struct Cleaner {
    files: Vec<&'static str>,
}

impl Cleaner {
    fn new(files: Vec<&'static str>) -> Cleaner {
        Cleaner { files }
    }
}

impl Drop for Cleaner {
    fn drop(&mut self) {
        for file in &self.files {
            let _ = std::fs::remove_file(file);
        }
    }
}

#[test]
fn argon() {
    if is_ci() {
        eprintln!("Skipping test in CI due to chemfiles SIGFPE issue");
        return;
    }
    START.call_once(::env_logger::init);
    let path = Path::new(file!())
        .parent()
        .unwrap()
        .join("..")
        .join("doc")
        .join("src")
        .join("data")
        .join("argon.toml");
    let mut config = Input::new(path).unwrap().read().unwrap();

    let _ = Cleaner::new(vec!["energy.dat", "trajectory.xyz"]);

    config.simulation.run(&mut config.system, 1);
}

#[test]
fn nacl() {
    if is_ci() {
        eprintln!("Skipping test in CI due to chemfiles SIGFPE issue");
        return;
    }
    START.call_once(::env_logger::init);
    let path = Path::new(file!())
        .parent()
        .unwrap()
        .join("..")
        .join("doc")
        .join("src")
        .join("data")
        .join("nacl.toml");
    let mut config = Input::new(path).unwrap().read().unwrap();

    let _ = Cleaner::new(vec!["trajectory.xyz"]);

    config.simulation.run(&mut config.system, 1);
}

#[test]
fn water() {
    if is_ci() {
        eprintln!("Skipping test in CI due to chemfiles SIGFPE issue");
        return;
    }
    START.call_once(::env_logger::init);
    let path = Path::new(file!())
        .parent()
        .unwrap()
        .join("..")
        .join("doc")
        .join("src")
        .join("data")
        .join("water.toml");
    let mut config = Input::new(path).unwrap().read().unwrap();

    let _ = Cleaner::new(vec!["trajectory.xyz"]);

    config.simulation.run(&mut config.system, 1);
}
