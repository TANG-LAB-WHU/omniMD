//! JSONL evaluator on the omnimd-gpu runtime (device 0).
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
//! frame to stderr; `--profile` adds the times of the stages, means over as
//! many further evaluations: the host's time to issue each stage and the
//! device's time over it, between marks in its stream (see `StageTimes`).
//! `--chunk` overrides the per-chunk atom count.
//! `--compress RMIN` replaces the embedding nets by tables in pieces of
//! uneven length, each within a tolerance of its net (that of a table with
//! a knot every 0.01 on a smooth net, or of another step if `:STEP` is
//! given). RMIN is a distance in Å below the shortest expected: the tables
//! reach to a fifth of it, frames with closer pairs fall back to the exact
//! networks, and the size of the tables is printed. `--fused-fitting ROWS`
//! sets the most rows (atoms of a chunk times models) whose fitting nets run
//! in the single fused kernel instead of cuBLAS (0 disables it); the default
//! suits an RTX 4090. `--grouped-fitting on|off` has the cuBLAS path take
//! all nets in grouped products, or one net at a time, always instead of by
//! the default rule (which goes by the nets and rows of a chunk).
//! `--wide-forces CENTERS` sets the most centers (atoms of a chunk times
//! models) of a pass whose two-body force kernel gets its wider blocks (0:
//! none); the default suits that card too. `--block-sums CENTERS` sets, for
//! every descriptor, the most centers of a pass whose force kernel forms the
//! sums of a block across its warps in shared memory (0: every warp reduces
//! its own, as before 2026-11-19); without it se_e2_a has a limit and the
//! others none. `--inline-tables on|off` has the descriptor and force
//! kernels of se_e2_a look the embedding values up in the tables themselves
//! (on), or read what a kernel of the tables stored for every row (off),
//! instead of by the precision (on in float32, off in float64).
//!
//! `--member` adds a model of the same architecture: the models are then
//! evaluated together as a committee, and each output line is the array of
//! their evaluations of a frame instead of one evaluation. `--batch FRAMES`
//! evaluates that many frames in one device pass (the last batch may be
//! smaller); the output is the same, and `--bench` and `--profile` then time
//! batches and report the first frame of each. `--bench-batches N,N,...`
//! (with `--bench K`) instead reads all frames and times passes over the
//! first N of them for each N in turn, about K evaluations of a frame by a
//! model each; it writes no evaluations.
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
                     [--fused-fitting ROWS] [--grouped-fitting on|off] [--tiled-fitting on|off] [--wide-forces CENTERS] \
                     [--block-sums CENTERS] [--inline-tables on|off] [--member MODEL.json]... [--batch FRAMES] \
                     [--bench-batches N,N,...] < frames.jsonl";

struct Options {
    model: String,
    members: Vec<String>,
    batch: usize,
    bench_batches: Vec<usize>,
    float32: bool,
    bench: usize,
    profile: bool,
    chunk: Option<usize>,
    compress: Option<(f64, f64)>,
    fused_fitting: Option<usize>,
    grouped_fitting: Option<bool>,
    tiled_fitting: Option<bool>,
    wide_forces: Option<usize>,
    block_sums: Option<usize>,
    inline_tables: Option<bool>,
}

fn options() -> Result<Options, Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut options = Options {
        model: args.next().ok_or(USAGE)?,
        members: Vec::new(),
        batch: 1,
        bench_batches: Vec::new(),
        float32: false,
        bench: 0,
        profile: false,
        chunk: None,
        compress: None,
        fused_fitting: None,
        grouped_fitting: None,
        tiled_fitting: None,
        wide_forces: None,
        block_sums: None,
        inline_tables: None,
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
            "--bench-batches" => {
                for size in value()?.split(',') {
                    options.bench_batches.push(size.parse()?);
                }
                if options.bench_batches.contains(&0) {
                    return Err("--bench-batches takes sizes of at least one frame".into());
                }
            }
            "--fused-fitting" => options.fused_fitting = Some(value()?.parse()?),
            "--wide-forces" => options.wide_forces = Some(value()?.parse()?),
            "--block-sums" => options.block_sums = Some(value()?.parse()?),
            "--grouped-fitting" => {
                options.grouped_fitting = Some(match value()?.as_str() {
                    "on" => true,
                    "off" => false,
                    other => {
                        return Err(
                            format!("--grouped-fitting takes on or off, not {other:?}").into()
                        )
                    }
                })
            }
            "--inline-tables" => {
                options.inline_tables = Some(match value()?.as_str() {
                    "on" => true,
                    "off" => false,
                    other => {
                        return Err(format!("--inline-tables takes on or off, not {other:?}").into())
                    }
                })
            }
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
    if !options.bench_batches.is_empty() && options.bench == 0 {
        return Err("--bench-batches needs --bench K".into());
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
    if let Some(rows) = options.fused_fitting {
        gpu.set_fused_fitting_max_rows(rows);
    }
    if let Some(on) = options.grouped_fitting {
        if gpu.set_grouped_fitting(on) != on {
            return Err("this model's fitting nets or this cuBLAS take no grouped products".into());
        }
    }
    if let Some(centers) = options.wide_forces {
        gpu.set_wide_force_centers(centers);
    }
    if let Some(centers) = options.block_sums {
        gpu.set_block_sums_centers(Some(centers));
    }
    if options.inline_tables.is_some() {
        gpu.set_inline_tables(options.inline_tables);
    }
    let mut build_seconds = 0.0;
    if let Some((min_distance, step)) = options.compress {
        let start = Instant::now();
        gpu.compress(min_distance, step)?;
        build_seconds = start.elapsed().as_secs_f64();
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
        match (gpu.is_compressed(), gpu.looks_up_tables()) {
            (true, true) => "tabulated, looked up by the kernels",
            (true, false) => "tabulated",
            (false, _) => "exact",
        },
        match (members, options.batch) {
            (1, 1) => String::new(),
            (members, batch) => format!(", {members} models, batches of {batch} frames"),
        }
    );
    if gpu.is_compressed() {
        let (bytes, pieces) = gpu.table_size();
        eprintln!(
            "tables: {:.1} MiB on the device, {pieces} pieces, made in {build_seconds:.2} s",
            bytes as f64 / (1 << 20) as f64
        );
    }
    let input = std::io::stdin();
    if !options.bench_batches.is_empty() {
        let mut frames: Vec<Frame> = Vec::new();
        for (line_number, line) in input.lock().lines().enumerate() {
            let line = line?;
            if !line.trim().is_empty() {
                frames.push(
                    serde_json::from_str(&line)
                        .map_err(|e| format!("frame {}: {e}", line_number + 1))?,
                );
            }
        }
        for &size in &options.bench_batches {
            let Some(batch) = frames.get(..size) else {
                return Err(format!("a batch of {size} needs as many frames").into());
            };
            let batch: Vec<&Frame> = batch.iter().collect();
            let natoms: usize = batch.iter().map(|frame| frame.positions.len()).sum();
            // Twice before the clock starts: the first pass of a size sets up
            // buffers and tables, the second runs with them.
            gpu.evaluate_frames(&batch, members)?;
            gpu.evaluate_frames(&batch, members)?;
            let passes = (options.bench / (size * members)).max(20);
            let start = Instant::now();
            for _ in 0..passes {
                gpu.evaluate_frames(&batch, members)?;
            }
            let ms = start.elapsed().as_secs_f64() * 1e3 / passes as f64;
            eprintln!(
                "{}",
                serde_json::json!({"frames": size, "natoms": natoms, "models": members,
                    "precision": P::SUFFIX, "ms_per_eval": ms, "evaluations": passes,
                    "ms_per_frame_and_model": ms / (size * members) as f64,
                    "table_fallbacks": gpu.table_fallbacks(),
                    "tables_looked_up": gpu.tables_looked_up(),
                    "count_downloads": gpu.count_downloads()})
            );
        }
        return Ok(());
    }
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
                "table_fallbacks": gpu.table_fallbacks(),
                "tables_looked_up": gpu.tables_looked_up(),
                "count_downloads": gpu.count_downloads()});
            if members > 1 || options.batch > 1 {
                record["frames"] = frames.len().into();
                record["models"] = members.into();
                // One model's evaluation of one frame.
                record["ms_per_frame_and_model"] = (ms / (frames.len() * members) as f64).into();
            }
            eprintln!("{record}");
        }
        if options.profile {
            // Means over as many evaluations as the benchmark: each as it
            // runs otherwise, with a mark in the device's stream after every
            // stage.
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
