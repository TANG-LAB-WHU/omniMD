//! Errors of a native model against the labels of a dataset, on the
//! omnimd-gpu runtime (device 0): the counterpart of `dp test`.
//!
//! ```text
//! omnimd-native-gpu-test MODEL.json DATASET.jsonl [--precision float64|float32]
//! ```
//!
//! Prints one JSON object with the errors over all frames and per group,
//! where a frame's group is its id up to the first `/` (the system name for
//! datasets made from DeePMD systems). Reported per group and overall:
//!
//! - `energy_rmse_per_atom`, `energy_mae_per_atom`: over frames, of
//!   `(E - E*) / N` (eV/atom);
//! - `force_rmse`, `force_mae`: over all force components (eV/Å);
//! - `virial_rmse_per_atom`: over the nine components of frames that have a
//!   virial label, of `(V - V*) / N` (eV/atom).
use omnimd_gpu::{Device, Scalar};
use omnimd_ml_contract::Potential;
use omnimd_native::gpu_model::GpuPotential;
use omnimd_native::training_data::Dataset;
use omnimd_native::NativePotential;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Sums of errors of one group of frames.
#[derive(Clone, Debug, Default)]
struct Errors {
    frames: usize,
    atoms: usize,
    energy_square: f64,
    energy_absolute: f64,
    force_components: usize,
    force_square: f64,
    force_absolute: f64,
    virial_components: usize,
    virial_square: f64,
}

impl Errors {
    fn add(
        &mut self,
        natoms: usize,
        energy: (f64, f64),
        forces: (&[[f64; 3]], &[[f64; 3]]),
        virial: Option<(&[[f64; 3]; 3], &[[f64; 3]; 3])>,
    ) {
        let n = natoms as f64;
        self.frames += 1;
        self.atoms += natoms;
        let per_atom = (energy.0 - energy.1) / n;
        self.energy_square += per_atom * per_atom;
        self.energy_absolute += per_atom.abs();
        for (predicted, label) in forces.0.iter().zip(forces.1) {
            for a in 0..3 {
                let difference = predicted[a] - label[a];
                self.force_square += difference * difference;
                self.force_absolute += difference.abs();
            }
        }
        self.force_components += 3 * natoms;
        if let Some((predicted, label)) = virial {
            for a in 0..3 {
                for b in 0..3 {
                    let difference = (predicted[a][b] - label[a][b]) / n;
                    self.virial_square += difference * difference;
                }
            }
            self.virial_components += 9;
        }
    }

    fn merge(&mut self, other: &Errors) {
        self.frames += other.frames;
        self.atoms += other.atoms;
        self.energy_square += other.energy_square;
        self.energy_absolute += other.energy_absolute;
        self.force_components += other.force_components;
        self.force_square += other.force_square;
        self.force_absolute += other.force_absolute;
        self.virial_components += other.virial_components;
        self.virial_square += other.virial_square;
    }

    fn report(&self) -> serde_json::Value {
        let frames = self.frames as f64;
        let components = self.force_components as f64;
        serde_json::json!({
            "frames": self.frames,
            "atoms": self.atoms,
            "energy_rmse_per_atom": (self.energy_square / frames).sqrt(),
            "energy_mae_per_atom": self.energy_absolute / frames,
            "force_rmse": (self.force_square / components).sqrt(),
            "force_mae": self.force_absolute / components,
            "virial_rmse_per_atom": (self.virial_components > 0)
                .then(|| (self.virial_square / self.virial_components as f64).sqrt()),
            "virial_frames": self.virial_components / 9,
        })
    }
}

fn group(id: &str) -> &str {
    id.split_once('/').map_or(id, |(prefix, _)| prefix)
}

fn test<P: Scalar>(
    model: &NativePotential,
    data: &Dataset,
    device: Arc<Device>,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let mut potential = GpuPotential::<P>::new(model.model(), device)?;
    let mut groups: BTreeMap<String, Errors> = BTreeMap::new();
    let mut truncated = 0usize;
    for frame in &data.frames {
        let prediction = potential.eval(&frame.frame)?;
        truncated += usize::from(potential.last_truncated());
        let (Some(energy), Some(forces)) = (frame.labels.energy, &frame.labels.forces) else {
            return Err(format!("frame {} lacks an energy or force label", frame.id).into());
        };
        groups.entry(group(&frame.id).to_string()).or_default().add(
            frame.frame.positions.len(),
            (prediction.energy, energy),
            (&prediction.forces, forces),
            frame
                .labels
                .virial
                .as_ref()
                .map(|label| (&prediction.virial, label)),
        );
    }
    let mut all = Errors::default();
    for errors in groups.values() {
        all.merge(errors);
    }
    Ok(serde_json::json!({
        "dataset_sha256": data.dataset_sha256,
        "precision": P::SUFFIX,
        "frames_exceeding_sel": truncated,
        "all": all.report(),
        "groups": groups.iter().map(|(name, errors)| (name.clone(), errors.report()))
            .collect::<BTreeMap<_, _>>(),
    }))
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    const USAGE: &str =
        "usage: omnimd-native-gpu-test MODEL.json DATASET.jsonl [--precision float64|float32]";
    let args: Vec<String> = std::env::args().skip(1).collect();
    let float32 = match args.as_slice() {
        [_, _] => false,
        [_, _, flag, value] if flag == "--precision" => match value.as_str() {
            "float32" => true,
            "float64" => false,
            _ => return Err(USAGE.into()),
        },
        _ => return Err(USAGE.into()),
    };
    let model = NativePotential::load(&args[0])?;
    let data = Dataset::load(&args[1])?;
    if data.header.type_map != model.model().type_map {
        return Err("dataset and model type maps differ".into());
    }
    let device = Arc::new(Device::new(0)?);
    let report = if float32 {
        test::<f32>(&model, &data, device)?
    } else {
        test::<f64>(&model, &data, device)?
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("test stopped: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::{group, Errors};

    #[test]
    fn errors_follow_their_definitions() {
        let mut errors = Errors::default();
        // Two atoms: energy off by 0.4 (0.2 per atom), one force component
        // off by 0.3, virial diagonal off by 0.2 (0.1 per atom).
        let label_forces = [[0.0; 3]; 2];
        let mut predicted_forces = label_forces;
        predicted_forces[1][2] = 0.3;
        let label_virial = [[0.0; 3]; 3];
        let mut predicted_virial = label_virial;
        for a in 0..3 {
            predicted_virial[a][a] = 0.2;
        }
        errors.add(
            2,
            (1.4, 1.0),
            (&predicted_forces, &label_forces),
            Some((&predicted_virial, &label_virial)),
        );
        // A second frame with exact energy and forces and no virial label.
        errors.add(2, (5.0, 5.0), (&label_forces, &label_forces), None);
        let report = errors.report();
        let close = |key: &str, value: f64| {
            let got = report[key].as_f64().unwrap();
            assert!((got - value).abs() < 1e-12, "{key}: {got} vs {value}");
        };
        close("energy_rmse_per_atom", (0.04f64 / 2.0).sqrt());
        close("energy_mae_per_atom", 0.1);
        close("force_rmse", (0.09f64 / 12.0).sqrt());
        close("force_mae", 0.3 / 12.0);
        close("virial_rmse_per_atom", (3.0 * 0.01f64 / 9.0).sqrt());
        assert_eq!(report["virial_frames"], 1);
        assert_eq!(
            (report["frames"].as_u64(), report["atoms"].as_u64()),
            (Some(2), Some(4))
        );

        let mut merged = Errors::default();
        merged.merge(&errors);
        merged.merge(&errors);
        assert_eq!(merged.report()["force_rmse"], report["force_rmse"]);
        assert!(Errors::default().report()["virial_rmse_per_atom"].is_null());
        assert_eq!(group("hap_bulk/set.000/3"), "hap_bulk");
        assert_eq!(group("frame-7"), "frame-7");
    }
}
