// OmniMD, an extensible molecular simulation engine
// Copyright (C) OmniMD's contributors — MIT license

//! OmniMD is a classical molecular simulation engine that provides a solid
//! base for developing new algorithms and methods.
//!
//! Using OmniMD, you can customize the behavior of all the algorithms in a
//! simulation (from force fields to barostats and Monte Carlo moves).
//!
//! OmniMD goals are to be:
//!
//! - **Easy to extend**: the code is modular, object-oriented, well documented,
//!   well tested, open-source and readable;
//! - **Easy to use**: the user interface is nice, with human-oriented input
//!   files;
//! - **Stable**: it will never crash on a good input, and provides helpful
//!   error messages.
//!
//! # Core components
//!
//! OmniMD is built on top of multiple crates:
//! - `omnimd-core` provides the core types and algorithms;
//! - `omnimd-sim` provides the simulation drivers;
//! - `omnimd-input` provides the input file parser.
//!
//! This crate provides the main entry point for the simulation engine.

/// The full version of the crate, containing git state if available
pub static VERSION: &str = env!("OMNIMD_FULL_GIT_VERSION");

pub mod input;
pub mod sim;

pub use omnimd_core::*;
