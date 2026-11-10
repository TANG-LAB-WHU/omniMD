//! Dataset statistics with DeePMD-kit semantics, as one JSON object.
//!
//! ```text
//! omnimd-native-stat DATASET.jsonl --rcut R [--rcut-smth S] [--sel N,N,...|auto[:FACTOR]]
//! ```
//!
//! Always prints `neighbor_stat` (`dp neighbor-stat`) and `energy_bias`.
//! With `--rcut-smth` and `--sel` it also prints the descriptor input
//! statistics `davg`/`dstd` per center type; `--sel auto` takes the largest
//! neighbor counts times 1.1 (or FACTOR), rounded up to a multiple of 4.
use omnimd_ml_contract::Frame;
use omnimd_native::stats::{energy_bias, env_mat_stat, neighbor_stat};
use omnimd_native::training_data::Dataset;

const USAGE: &str = "usage: omnimd-native-stat DATASET.jsonl --rcut R [--rcut-smth S] \
                     [--sel N,N,...|auto[:FACTOR]]";

enum Sel {
    Explicit(Vec<usize>),
    Auto(f64),
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or(USAGE)?;
    let (mut rcut, mut rcut_smth, mut sel) = (None, None, None);
    while let Some(flag) = args.next() {
        let value = args.next().ok_or(USAGE)?;
        match flag.as_str() {
            "--rcut" => rcut = Some(value.parse::<f64>()?),
            "--rcut-smth" => rcut_smth = Some(value.parse::<f64>()?),
            "--sel" => {
                sel = Some(match value.strip_prefix("auto") {
                    Some("") => Sel::Auto(1.1),
                    Some(rest) => Sel::Auto(rest.strip_prefix(':').ok_or(USAGE)?.parse()?),
                    None => {
                        Sel::Explicit(value.split(',').map(str::parse).collect::<Result<_, _>>()?)
                    }
                })
            }
            _ => return Err(USAGE.into()),
        }
    }
    let rcut = rcut.ok_or(USAGE)?;
    let dataset = Dataset::load(&path)?;
    let type_map = &dataset.header.type_map;
    let frames: Vec<&Frame> = dataset.frames.iter().map(|f| &f.frame).collect();
    let energies: Vec<f64> = dataset
        .frames
        .iter()
        .map(|f| f.labels.energy.ok_or("frame without an energy label"))
        .collect::<Result<_, _>>()?;

    let neighbors = neighbor_stat(&frames, type_map, rcut)?;
    let mut report = serde_json::json!({
        "dataset_sha256": dataset.dataset_sha256,
        "frames": frames.len(),
        "type_map": type_map,
        "rcut": rcut,
        "neighbor_stat": neighbors,
        "energy_bias": energy_bias(&frames, &energies, type_map)?,
    });
    if !neighbors.min_nbor_dist.is_finite() {
        // JSON has no infinity: no pair within rcut is reported as null.
        report["neighbor_stat"]["min_nbor_dist"] = serde_json::Value::Null;
    }
    match (rcut_smth, sel) {
        (Some(rcut_smth), Some(sel)) => {
            let sel = match sel {
                Sel::Explicit(sel) => sel,
                Sel::Auto(factor) => neighbors.auto_sel(factor),
            };
            let stats = env_mat_stat(&frames, type_map, rcut, rcut_smth, &sel)?;
            report["rcut_smth"] = rcut_smth.into();
            report["sel"] = sel.into();
            report["davg"] = stats
                .iter()
                .map(|n| n.mean.to_vec())
                .collect::<Vec<_>>()
                .into();
            report["dstd"] = stats
                .iter()
                .map(|n| n.std.to_vec())
                .collect::<Vec<_>>()
                .into();
        }
        (None, None) => {}
        _ => return Err("--rcut-smth and --sel go together".into()),
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("statistics stopped: {e}");
        std::process::exit(1);
    }
}
