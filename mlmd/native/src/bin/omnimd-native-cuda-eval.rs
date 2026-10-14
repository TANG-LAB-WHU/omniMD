//! Explicit CUDA JSONL evaluator; never silently substitutes CPU output.
use omnimd_ml_contract::{Frame, Potential};
use omnimd_native::{cuda::CudaPotential, NativePotential};
use std::io::{BufRead, Write};
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 {
        return Err("usage: omnimd-native-cuda-eval MODEL.json < frames.jsonl (device 0)".into());
    }
    let model = NativePotential::load(&args[1])?;
    let mut gpu = CudaPotential::new(model.model().clone(), 0)?;
    eprintln!("{}", gpu.device_info());
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
        serde_json::to_writer(&mut out, &evaluation)?;
        writeln!(out)?;
        out.flush()?;
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("CUDA evaluation stopped: {e}");
        std::process::exit(1);
    }
}
