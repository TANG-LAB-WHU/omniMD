//! JSONL evaluator on the omnimd-gpu se_e2_a path (device 0).
//!
//! ```text
//! omnimd-native-gpu-eval MODEL.json [--precision float64|float32] [--bench K]
//!                        [--profile] [--chunk ATOMS] [--compress RMIN[:STEP]]
//!                        [--member MODEL.json]... [--batch FRAMES] < frames.jsonl
//! ```
//!
//! `--precision float32` runs the networks in float32 (geometry, reductions,
//! forces and virial stay float64). `--bench K` times K in-process
//! evaluations of each frame after one warmup and prints one JSON line per
//! frame to stderr; `--profile` adds per-stage times from one extra,
//! synchronized evaluation. `--chunk` overrides the per-chunk atom count.
//! `--compress RMIN` tabulates the embedding nets for neighbor distances
//! down to RMIN Å (table step 0.01 unless given); frames with closer pairs
//! fall back to the exact networks. `--fused-fitting ATOMS` sets the largest
//! chunk whose fitting nets run in the single fused kernel instead of cuBLAS
//! (0 disables it); the default suits an RTX 4090.
//!
//! `--member` adds a model of the same architecture: the models are then
//! evaluated together as a committee, and each output line is the array of
//! their evaluations of a frame instead of one evaluation. `--batch FRAMES`
//! evaluates that many frames in one device pass (the last batch may be
//! smaller); the output is the same, and `--bench` and `--profile` then time
//! batches and report the first frame of each.
use omnimd_gpu::{Device, Scalar};
use omnimd_ml_contract::Frame;
use omnimd_native::{
    gpu_model::{GpuPotential, StageTimes},
    Model, NativePotential,
};
use std::io::{BufRead, Write};
use std::sync::Arc;
use std::time::Instant;

const USAGE: &str = "usage: omnimd-native-gpu-eval MODEL.json [--precision float64|float32] \
                     [--bench K] [--profile] [--chunk ATOMS] [--compress RMIN[:STEP]] \
                     [--fused-fitting ATOMS] [--tiled-fitting on|off] [--member MODEL.json]... \
                     [--batch FRAMES] < frames.jsonl";

struct Options {
    model: String,
    members: Vec<String>,
    batch: usize,
    float32: bool,
    bench: usize,
    profile: bool,
    chunk: Option<usize>,
    compress: Option<(f64, f64)>,
    fused_fitting: Option<usize>,
    tiled_fitting: Option<bool>,
}

fn options() -> Result<Options, Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut options = Options {
        model: args.next().ok_or(USAGE)?,
        members: Vec::new(),
        batch: 1,
        float32: false,
        bench: 0,
        profile: false,
        chunk: None,
        compress: None,
        fused_fitting: None,
        tiled_fitting: None,
    };
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or(USAGE);
        match flag.as_str() {
            "--precision" => {
                options.float32 = match value()?.as_str() {
                    "float32" => true,
                    "float64" => false,
                    other => return Err(format!("unknown precision {other:?}").into()),
                }
            }
            "--bench" => options.bench = value()?.parse()?,
            "--chunk" => options.chunk = Some(value()?.parse()?),
            "--member" => options.members.push(value()?),
            "--batch" => {
                options.batch = value()?.parse()?;
                if options.batch == 0 {
                    return Err("--batch takes at least one frame".into());
                }
            }
            "--fused-fitting" => options.fused_fitting = Some(value()?.parse()?),
            "--tiled-fitting" => {
                options.tiled_fitting = Some(match value()?.as_str() {
                    "on" => true,
                    "off" => false,
                    other => {
                        return Err(format!("--tiled-fitting takes on or off, not {other:?}").into())
                    }
                })
            }
            "--compress" => {
                let text = value()?;
                let (rmin, step) = text.split_once(':').unwrap_or((&text, "0.01"));
                options.compress = Some((rmin.parse()?, step.parse()?));
            }
            "--profile" => options.profile = true,
            _ => return Err(USAGE.into()),
        }
    }
    Ok(options)
}

fn evaluate<P: Scalar>(
    options: &Options,
    models: &[&Model],
    device: Arc<Device>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = GpuPotential::<P>::committee(models, device)?;
    let members = gpu.models();
    if let Some(chunk) = options.chunk {
        gpu.set_chunk_atoms(chunk);
    }
    if let Some(atoms) = options.fused_fitting {
        gpu.set_fused_fitting_max_atoms(atoms);
    }
    if let Some((min_distance, step)) = options.compress {
        gpu.compress(min_distance, step)?;
    }
    if let Some(on) = options.tiled_fitting {
        if gpu.set_tiled_fitting(on) != on {
            return Err("this model's fitting nets have no tiled first layer".into());
        }
    }
    eprintln!(
        "precision: network {}, chunk {} atoms, embedding {}{}",
        P::SUFFIX,
        gpu.chunk_atoms(),
        if gpu.is_compressed() {
            "tabulated"
        } else {
            "exact"
        },
        match (members, options.batch) {
            (1, 1) => String::new(),
            (members, batch) => format!(", {members} models, batches of {batch} frames"),
        }
    );
    let input = std::io::stdin();
    let mut lines = input.lock().lines().enumerate();
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    loop {
        // The next batch of frames, with the line of its first.
        let mut frames: Vec<Frame> = Vec::with_capacity(options.batch);
        let mut first_line = 0;
        while frames.len() < options.batch {
            let Some((line_number, line)) = lines.next() else {
                break;
            };
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            if frames.is_empty() {
                first_line = line_number;
            }
            frames.push(
                serde_json::from_str(&line)
                    .map_err(|e| format!("frame {}: {e}", line_number + 1))?,
            );
        }
        if frames.is_empty() {
            break;
        }
        let batch: Vec<&Frame> = frames.iter().collect();
        let natoms: usize = frames.iter().map(|frame| frame.positions.len()).sum();
        let evaluations = gpu.evaluate_frames(&batch, members)?;
        if gpu.table_fallbacks() > 0 && first_line == 0 {
            eprintln!("note: frame 1 left the embedding table range; the exact networks were used");
        }
        if gpu.last_truncated() {
            eprintln!(
                "warning: frame {} has more neighbors than sel; nearest kept",
                first_line + 1
            );
        }
        if options.bench > 0 {
            gpu.evaluate_frames(&batch, members)?;
            let start = Instant::now();
            for _ in 0..options.bench {
                gpu.evaluate_frames(&batch, members)?;
            }
            let ms = start.elapsed().as_secs_f64() * 1e3 / options.bench as f64;
            let mut record = serde_json::json!({"frame": first_line + 1, "natoms": natoms,
                "precision": P::SUFFIX, "ms_per_eval": ms, "evaluations": options.bench,
                "table_fallbacks": gpu.table_fallbacks()});
            if members > 1 || options.batch > 1 {
                record["frames"] = frames.len().into();
                record["models"] = members.into();
                // One model's evaluation of one frame.
                record["ms_per_frame_and_model"] = (ms / (frames.len() * members) as f64).into();
            }
            eprintln!("{record}");
        }
        if options.profile {
            // Mean over as many evaluations as the benchmark, each stage
            // timed up to a device sync of its own.
            let rounds = options.bench.max(1);
            let mut sum = StageTimes::default();
            for _ in 0..rounds {
                gpu.set_profiling(true);
                gpu.evaluate_frames(&batch, members)?;
                if let Some(times) = gpu.stage_times() {
                    sum.add(times);
                }
            }
            gpu.set_profiling(false);
            eprintln!(
                "{}",
                serde_json::json!({"frame": first_line + 1, "evaluations": rounds,
                    "stages_ms": sum.mean(rounds)})
            );
        }
        for of_frame in &evaluations {
            match of_frame.as_slice() {
                [only] if members == 1 => serde_json::to_writer(&mut out, only)?,
                all => serde_json::to_writer(&mut out, all)?,
            }
            writeln!(out)?;
        }
        out.flush()?;
    }
    Ok(())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let options = options()?;
    let loaded = std::iter::once(&options.model)
        .chain(&options.members)
        .map(NativePotential::load)
        .collect::<Result<Vec<_>, _>>()?;
    let models: Vec<&Model> = loaded.iter().map(|potential| potential.model()).collect();
    let device = Arc::new(Device::new(0)?);
    eprintln!(
        "device: {} (sm_{}{})",
        device.name(),
        device.compute_capability().0,
        device.compute_capability().1
    );
    if options.float32 {
        evaluate::<f32>(&options, &models, device)
    } else {
        evaluate::<f64>(&options, &models, device)
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("GPU evaluation stopped: {e}");
        std::process::exit(1);
    }
}
