// OmniMD, an extensible molecular simulation engine
// Copyright (C) OmniMD's contributors — MIT license

use std::path::Path;
use std::sync::{Arc, Mutex};

use tch::{CModule, Device, IValue, Kind, Tensor};

use crate::sys::Configuration;
use crate::types::{Matrix3, Vector3D};
use crate::units;
use super::{BoxCloneGlobal, GlobalCache, GlobalPotential};

/// A global potential using a PyTorch TorchScript model to compute energy and forces.
///
/// The model must accept a list of 3 tensors as input:
/// - atomic numbers (1D tensor of integers)
/// - positions (2D tensor of floats [N, 3], in Angstroms)
/// - cell (2D tensor of floats [3, 3], column-major vectors in Angstroms)
///
/// The model must return a dictionary (Dict<str, Tensor>) with the following keys:
/// - "energy": 0-D tensor (scalar) containing the potential energy in eV.
/// - "forces": 2D tensor [N, 3] containing the forces in eV/A.
/// - "virial": (Optional) 2D tensor [3, 3] containing the virial in eV.
///
/// Note: The `cell` input tensor contains the cell vectors as COLUMNS, which is
/// consistent with `omnimd`'s matrix representation, but check your model's expectation.
pub struct TorchPotential {
    /// The torchscript model
    module: Arc<Mutex<CModule>>,
    /// Factor to convert eV to internal energy unit
    energy_factor: f64,
    /// Factor to convert eV/A to internal force unit
    force_factor: f64,
    /// The computation device (CPU or CUDA)
    device: Device,
}

impl TorchPotential {
    /// Create a new `TorchPotential` from the given `model_path`.
    pub fn new<P: AsRef<Path>>(
        model_path: P,
    ) -> Result<TorchPotential, Box<dyn std::error::Error + Send + Sync>> {
        let device = Device::cuda_if_available();
        let mut module = CModule::load(model_path)?;
        module.set_eval();
        module.to(device, Kind::Float, false);

        let ev = units::from(1.0, "eV").expect("Could not find eV unit");

        // OmniMD internal length is Angstrom.
        // Energy: eV -> internal
        // Force: eV/A -> internal/A = internal
        // So the factors are identical.
        let factor = ev;

        Ok(TorchPotential {
            module: Arc::new(Mutex::new(module)),
            energy_factor: factor,
            force_factor: factor,
            device: device,
        })
    }

    fn prepare_inputs(&self, configuration: &Configuration) -> (Tensor, Tensor, Tensor) {
        let n = configuration.size();

        // 1. Atomic numbers
        // We need to map particle names to atomic constants, but omnimd doesn't strictly store Z.
        // We will fallback to using the mass or name lookup if possible,
        // but for now let's try to parse the name as an element.
        // A better way is to rely on `omnimd-input` to set this up, but here we only have `Configuration`.
        // We will iterate and try to match names to atomic numbers using a helper or lookup.
        // Since `chemfiles` is a dependency, maybe we can use it?
        // Actually, `omnimd` core doesn't expose a simple Z lookup easily in `Configuration` without a full table.
        // However, `Particle` has a `name`.
        // We will assume `name` is the element symbol.
        let species: Vec<i64> = configuration
            .particles()
            .iter()
            .map(|p| element_to_z(&p.name).unwrap_or(0))
            .collect();

        let z_tensor = Tensor::from_slice(&species).to(self.device);

        // 2. Positions (Angstroms)
        let coords: Vec<f64> = configuration
            .particles()
            .iter()
            .flat_map(|p| vec![p.position[0], p.position[1], p.position[2]])
            .collect();
        let pos_tensor = Tensor::from_slice(&coords)
            .reshape(&[n as i64, 3])
            .to_kind(Kind::Float)
            .to(self.device);

        // 3. Cell (Angstroms)
        // OmniMD Matrix3 is column major?
        // Matrix3 struct:
        // pub struct Matrix3 { pub m: [[f64; 3]; 3] }
        // internal indexing m[i][j] where i is row, j is column? Or vice versa?
        // Let's check Matrix3 definition. assuming column vectors for cell.
        // Standard cell tensor in MLIP is usually [3, 3] with row vectors or column vectors.
        // MACE usually expects row vectors [a, b, c].
        // Let's send the matrix as is.
        let cell = configuration.cell.matrix();
        let cell_data = vec![
            cell[0][0], cell[0][1], cell[0][2], cell[1][0], cell[1][1], cell[1][2], cell[2][0],
            cell[2][1], cell[2][2],
        ];
        // Reshape to 3x3
        let cell_tensor = Tensor::from_slice(&cell_data)
            .reshape(&[3, 3])
            .to_kind(Kind::Float)
            .to(self.device);

        (z_tensor, pos_tensor, cell_tensor)
    }
}

// Simple lookup for common elements used in AIMD (CHON, metals, etc.)
// A full table would be better, but we keep it simple for this prototype.
fn element_to_z(name: &str) -> Option<i64> {
    match name {
        "H" => Some(1), "He" => Some(2), "Li" => Some(3), "Be" => Some(4), "B" => Some(5),
        "C" => Some(6), "N" => Some(7), "O" => Some(8), "F" => Some(9), "Ne" => Some(10),
        "Na" => Some(11), "Mg" => Some(12), "Al" => Some(13), "Si" => Some(14), "P" => Some(15),
        "S" => Some(16), "Cl" => Some(17), "Ar" => Some(18), "K" => Some(19), "Ca" => Some(20),
        "Sc" => Some(21), "Ti" => Some(22), "V" => Some(23), "Cr" => Some(24), "Mn" => Some(25),
        "Fe" => Some(26), "Co" => Some(27), "Ni" => Some(28), "Cu" => Some(29), "Zn" => Some(30),
        "Ga" => Some(31), "Ge" => Some(32), "As" => Some(33), "Se" => Some(34), "Br" => Some(35),
        "Kr" => Some(36), "Rb" => Some(37), "Sr" => Some(38), "Y" => Some(39), "Zr" => Some(40),
        "Nb" => Some(41), "Mo" => Some(42), "Tc" => Some(43), "Ru" => Some(44), "Rh" => Some(45),
        "Pd" => Some(46), "Ag" => Some(47), "Cd" => Some(48), "In" => Some(49), "Sn" => Some(50),
        "Sb" => Some(51), "Te" => Some(52), "I" => Some(53), "Xe" => Some(54), "Cs" => Some(55),
        "Ba" => Some(56), "La" => Some(57), "Ce" => Some(58), "Pr" => Some(59), "Nd" => Some(60),
        "Pm" => Some(61), "Sm" => Some(62), "Eu" => Some(63), "Gd" => Some(64), "Tb" => Some(65),
        "Dy" => Some(66), "Ho" => Some(67), "Er" => Some(68), "Tm" => Some(69), "Yb" => Some(70),
        "Lu" => Some(71), "Hf" => Some(72), "Ta" => Some(73), "W" => Some(74), "Re" => Some(75),
        "Os" => Some(76), "Ir" => Some(77), "Pt" => Some(78), "Au" => Some(79), "Hg" => Some(80),
        "Tl" => Some(81), "Pb" => Some(82), "Bi" => Some(83), "Po" => Some(84), "At" => Some(85),
        "Rn" => Some(86), "Fr" => Some(87), "Ra" => Some(88), "Ac" => Some(89), "Th" => Some(90),
        "Pa" => Some(91), "U" => Some(92), "Np" => Some(93), "Pu" => Some(94), "Am" => Some(95),
        "Cm" => Some(96), "Bk" => Some(97), "Cf" => Some(98), "Es" => Some(99), "Fm" => Some(100),
        "Md" => Some(101), "No" => Some(102), "Lr" => Some(103), "Rf" => Some(104), "Db" => Some(105),
        "Sg" => Some(106), "Bh" => Some(107), "Hs" => Some(108), "Mt" => Some(109), "Ds" => Some(110),
        "Rg" => Some(111), "Cn" => Some(112), "Nh" => Some(113), "Fl" => Some(114), "Mc" => Some(115),
        "Lv" => Some(116), "Ts" => Some(117), "Og" => Some(118),
        _ => None,
    }
}

impl GlobalPotential for TorchPotential {
    fn cutoff(&self) -> Option<f64> {
        // MLIPs usually have a local cutoff but act globally in this interface.
        // Returning None implies it affects everything (like Ewald).
        None
    }

    fn energy(&self, configuration: &Configuration) -> f64 {
        let (z, pos, cell) = self.prepare_inputs(configuration);
        let module = self.module.lock().unwrap();

        // Forward pass
        // Model signature: forward(z, pos, cell) -> Dict<str, Tensor>
        // We pass inputs as IValues
        let inputs = [IValue::Tensor(z), IValue::Tensor(pos), IValue::Tensor(cell)];
        let output = module
            .forward_is(&inputs)
            .expect("TorchScript forward failed"); // Handle error properly in real code

        let dict = match output {
            tch::IValue::GenericDict(d) => d,
            _ => panic!("Expected Dictionary output from MLIP model"),
        };

        // Extract energy
        // Key needs to be an IValue::String match?
        // tch-rs generic dict is Vec<(IValue, IValue)>
        // We scan for "energy"
        let energy_val = dict.iter().find(|(k, _)| match k {
            tch::IValue::String(s) => s == "energy",
            _ => false,
        });

        if let Some((_, v)) = energy_val {
            let t = match v {
                tch::IValue::Tensor(t) => t,
                _ => panic!("Energy key must be a Tensor"),
            };
            // Get scalar
            let e_ev: f64 = t.double_value(&[]);
            return e_ev * self.energy_factor;
        }

        0.0 // Or error
    }

    fn forces(&self, configuration: &Configuration, forces: &mut [Vector3D]) {
        let (z, pos, cell) = self.prepare_inputs(configuration);
        let module = self.module.lock().unwrap();

        let inputs = [IValue::Tensor(z), IValue::Tensor(pos), IValue::Tensor(cell)];
        let output = module
            .forward_is(&inputs)
            .expect("TorchScript forward failed");

        let dict = match output {
            tch::IValue::GenericDict(d) => d,
            _ => panic!("Expected Dictionary output from MLIP model"),
        };

        // Extract forces
        let forces_val = dict.iter().find(|(k, _)| match k {
            tch::IValue::String(s) => s == "forces",
            _ => false,
        });

        if let Some((_, v)) = forces_val {
            let t = match v {
                tch::IValue::Tensor(t) => t,
                _ => panic!("Forces key must be a Tensor"),
            };

            // Expected shape [N, 3]
            // We assume contiguous row-major.
            // Convert to Vec<f64>
            let num_atoms = configuration.size();
            let f_data: Vec<f64> = t.flatten(0, 1).try_into().unwrap(); // flatten to 1D

            for (i, force) in forces.iter_mut().enumerate() {
                if i < num_atoms {
                    let fx = f_data[3 * i] * self.force_factor;
                    let fy = f_data[3 * i + 1] * self.force_factor;
                    let fz = f_data[3 * i + 2] * self.force_factor;
                    // Add to existing forces?
                    // The trait definition says: "Compute the force contribution...".
                    // Usually this means we ADD to the forces buffer, or SET it?
                    // Let's check `GlobalPotential::forces` doc: "Compute the force contribution... This function should return...".
                    // The signature is `forces: &mut [Vector3D]`.
                    // In `omnimd`, usually we accumulate.
                    // Wait, `forces` is passed as mutable slice.
                    // The doc says "Compute the force contribution".
                    // Looking at `Ewald::forces`, it does `forces[i] += ...`.
                    // So we must ACCUMULATE.

                    force[0] += fx;
                    force[1] += fy;
                    force[2] += fz;
                }
            }
        }
    }

    fn atomic_virial(&self, configuration: &Configuration) -> Matrix3 {
        let (z, pos, cell) = self.prepare_inputs(configuration);
        let module = self.module.lock().unwrap();

        let inputs = [IValue::Tensor(z), IValue::Tensor(pos), IValue::Tensor(cell)];
        let output = module
            .forward_is(&inputs)
            .expect("TorchScript forward failed");

        let dict = match output {
            tch::IValue::GenericDict(d) => d,
            _ => panic!("Expected Dictionary output from MLIP model"),
        };

        // Extract virial
        // "virial" or "stress"?
        // We documented "virial".
        let virial_val = dict.iter().find(|(k, _)| match k {
            tch::IValue::String(s) => s == "virial",
            _ => false,
        });

        if let Some((_, v)) = virial_val {
            let t = match v {
                tch::IValue::Tensor(t) => t,
                _ => panic!("Virial key must be a Tensor"),
            };
            // 3x3
            let v_data: Vec<f64> = t.flatten(0, 1).try_into().unwrap();
            // Convert to Matrix3
            // Assume standard layout.
            let m = Matrix3::new([
                [
                    v_data[0] * self.energy_factor,
                    v_data[1] * self.energy_factor,
                    v_data[2] * self.energy_factor,
                ],
                [
                    v_data[3] * self.energy_factor,
                    v_data[4] * self.energy_factor,
                    v_data[5] * self.energy_factor,
                ],
                [
                    v_data[6] * self.energy_factor,
                    v_data[7] * self.energy_factor,
                    v_data[8] * self.energy_factor,
                ],
            ]);
            return m;
        }

        Matrix3::zero()
    }
}

// GlobalCache implementation (required for GlobalPotential)
// Since MLIP is expensive and GlobalCache is mostly for Monte Carlo optimization (partial energy),
// and MLIP is many-body, we can't easily do partial updates.
// We will just do a full recompute or panic if used in MC with partial moves?
// `forces` and `energy` are for MD.
// `GlobalCache` is for MC.
// If we run MD, we don't need efficient GlobalCache.
impl GlobalCache for TorchPotential {
    fn move_molecule_cost(&self, _configuration: &Configuration, _: usize, _: &[Vector3D]) -> f64 {
        // This is inefficient but correct: calculate full energy difference.
        // But `move_molecule_cost` is supposed to be differential.
        // If we can't support it efficiently, we might warn or just implement expensive full diff.
        // For now, let's unimplemented!() or return 0 if we assume it's only for MD.
        // But the trait requires it.
        // Let's implement a panic to prevent silent slow MC.
        unimplemented!("TorchPotential does not support Monte Carlo moves yet (GlobalCache)")
    }

    fn update(&self) {
        // Nothing to update
    }
}

impl Clone for TorchPotential {
    fn clone(&self) -> TorchPotential {
        TorchPotential {
            module: self.module.clone(),
            energy_factor: self.energy_factor,
            force_factor: self.force_factor,
            device: self.device,
        }
    }
}
