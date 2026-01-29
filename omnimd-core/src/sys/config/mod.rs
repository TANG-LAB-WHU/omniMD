// OmniMD, an extensible molecular simulation engine
// Copyright (C) 2015-2016 OmniMD's contributors — MIT license

//! Configuration and related types

mod mass;
pub use self::mass::get_atomic_mass;

mod particles;
pub use self::particles::{Particle, ParticleKind};
pub use self::particles::{ParticlePtr, ParticlePtrMut};
pub use self::particles::{ParticleRef, ParticleRefMut};
pub use self::particles::{ParticleSlice, ParticleSliceMut, ParticleVec};

mod composition;
pub use self::composition::Composition;

mod cells;
pub use self::cells::{CellShape, UnitCell};

mod connect;
pub use self::connect::BondDistances;
pub use self::connect::{Angle, Bond, Dihedral};

mod bonding;
pub use self::bonding::Bonding;

mod molecules;
pub use self::molecules::{Molecule, MoleculeHash, MoleculeRef, MoleculeRefMut};

mod configuration;
pub use self::configuration::Configuration;
pub use self::configuration::Permutation;
pub use self::configuration::{MoleculeIter, MoleculeIterMut};
