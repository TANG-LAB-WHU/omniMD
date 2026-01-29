// OmniMD, an extensible molecular simulation engine
// Copyright (C) OmniMD's contributors — MIT license

//! Various internal utilities, which do not have there own module

#[macro_use]
mod macros;

mod thread_vec;
pub use self::thread_vec::ThreadLocalVec;

#[cfg(test)]
mod xyz;
#[cfg(test)]
pub use self::xyz::system_from_xyz;
