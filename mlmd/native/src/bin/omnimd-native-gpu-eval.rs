//! JSONL evaluator on the omnimd-gpu se_e2_a path (device 0).
//!
//! ```text
//! omnimd-native-gpu-eval MODEL.json [--precision float64|float32] [--bench K]
//!                        [--profile] [--chunk ATOMS] [--compress RMIN[:STEP]] < frames.jsonl
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
use omnimd_gpu::{Device, Scalar};
use omnimd_ml_contract::{Frame, Potential};
use omnimd_native::{gpu_model::GpuPotential, NativePotential};
use std::io::{BufRead, Write};
use std::sync::Arc;
use std::time::Instant;

const USAGE: &str = "usage: omnimd-native-gpu-eval MODEL.json [--precision float64|float32] \
                     [--bench K] [--profile] [--chunk ATOMS] [--compress RMIN[:STEP]] \
                     [--fused-fitting ATOMS] < frames.jsonl";

struct Options {
    model: String,
    float32: bool,
    bench: usize,
    profile: bool,
    chunk: Option<usize>,
    compress: Option<(f64, f64)>,
    fused_fitting: Option<usize>,
}

fn options() -> Result<Options, Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut options = Options {
        model: args.next().ok_or(USAGE)?,
        float32: false,
        bench: 0,
        profile: false,
        chunk: None,
        compress: None,
        fused_fitting: None,
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
            "--fused-fitting" => options.fused_fitting = Some(value()?.parse()?),
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
    model: &NativePotential,
    device: Arc<Device>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = GpuPotential::<P>::new(model.model(), device)?;
    if let Some(chunk) = options.chunk {
        gpu.set_chunk_atoms(chunk);
    }
    if let Some(atoms) = options.fused_fitting {
        gpu.set_fused_fitting_max_atoms(atoms);
    }
    if let Some((min_distance, step)) = options.compress {
        gpu.compress(min_distance, step)?;
    }
    eprintln!(
        "precision: network {}, chunk {} atoms, embedding {}",
        P::SUFFIX,
        gpu.chunk_atoms(),
        if gpu.is_compressed() {
            "tabulated"
        } else {
            "exact"
        }
    );
    let input = std::io::stdin();
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for (line_number, line) in input.lock().lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let frame: Frame =
            serde_json::from_str(&line).map_err(|e| format!("frame {}: {e}", line_number + 1))?;
        let evaluation = gpu.eval(&frame)?;
        if gpu.table_fallbacks() > 0 && line_number == 0 {
            eprintln!("note: frame 1 left the embedding table range; the exact networks were used");
        }
        if gpu.last_truncated() {
            eprintln!(
                "warning: frame {} has more neighbors than sel; nearest kept",
                line_number + 1
            );
        }
        if options.bench > 0 {
            gpu.eval(&frame)?;
            let start = Instant::now();
            for _ in 0..options.bench {
                gpu.eval(&frame)?;
            }
            let ms = start.elapsed().as_secs_f64() * 1e3 / options.bench as f64;
            eprintln!(
                "{}",
                serde_json::json!({"frame": line_number + 1, "natoms": frame.positions.len(),
                    "precision": P::SUFFIX, "ms_per_eval": ms, "evaluations": options.bench,
                    "table_fallbacks": gpu.table_fallbacks()})
            );
        }
        if options.profile {
            gpu.set_profiling(true);
            gpu.eval(&frame)?;
            eprintln!(
                "{}",
                serde_json::json!({"frame": line_number + 1, "stages_ms": gpu.stage_times()})
            );
            gpu.set_profiling(false);
        }
        serde_json::to_writer(&mut out, &evaluation)?;
        writeln!(out)?;
        out.flush()?;
    }
    Ok(())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let options = options()?;
    let model = NativePotential::load(&options.model)?;
    let device = Arc::new(Device::new(0)?);
    eprintln!(
        "device: {} (sm_{}{})",
        device.name(),
        device.compute_capability().0,
        device.compute_capability().1
    );
    if options.float32 {
        evaluate::<f32>(&options, &model, device)
    } else {
        evaluate::<f64>(&options, &model, device)
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("GPU evaluation stopped: {e}");
        std::process::exit(1);
    }
}
