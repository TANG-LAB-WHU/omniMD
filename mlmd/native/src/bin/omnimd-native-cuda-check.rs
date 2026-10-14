//! Bounded same-model parity and warmed end-to-end timing. Opt-in GPU execution.
use omnimd_ml_contract::{Evaluation, Frame, Potential};
use omnimd_native::{
    cuda::{CudaPotential, PackedModel},
    NativePotential,
};
use std::time::Instant;
fn compare(
    a: &Evaluation,
    b: &Evaluation,
    max: &mut [f64; 3],
) -> Result<(), Box<dyn std::error::Error>> {
    let fields = [
        (vec![a.energy], vec![b.energy]),
        (
            a.forces.iter().flatten().copied().collect(),
            b.forces.iter().flatten().copied().collect(),
        ),
        (
            a.virial.iter().flatten().copied().collect(),
            b.virial.iter().flatten().copied().collect(),
        ),
    ];
    for (k, (x, y)) in fields.iter().enumerate() {
        for (x, y) in x.iter().zip(y) {
            let e = (x - y).abs();
            max[k] = max[k].max(e);
            if e > 1e-9 + 1e-9 * x.abs() {
                return Err(format!("CPU/CUDA component mismatch {x} vs {y}, error {e}").into());
            }
        }
    }
    Ok(())
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() == 3 && args[1] == "--compile-only" {
        let model = NativePotential::load(&args[2])?;
        let bytes = PackedModel::new(model.model())?.compile_only()?;
        println!(
            "{}",
            serde_json::json!({"compile_only":true,"gpu_executed":false,"ptx_bytes":bytes,"target":"compute_75"})
        );
        return Ok(());
    }
    if args.len() != 4 {
        return Err("usage: omnimd-native-cuda-check MODEL.json FRAMES.jsonl REPEATS(1..20); or --compile-only MODEL.json".into());
    }
    let repeats: usize = args[3].parse()?;
    if !(1..=20).contains(&repeats) {
        return Err("repeats must be 1..20".into());
    }
    let text = std::fs::read_to_string(&args[2])?;
    let frames: Vec<Frame> = text
        .lines()
        .filter(|s| !s.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    if frames.is_empty() || frames.len() > 16 {
        return Err("check requires 1..16 frames".into());
    }
    let mut cpu = NativePotential::load(&args[1])?;
    let before = Instant::now();
    let mut gpu = CudaPotential::new(cpu.model().clone(), 0)?;
    let startup_ms = before.elapsed().as_secs_f64() * 1000.0;
    let mut max = [0.0_f64; 3];
    for frame in &frames {
        compare(&cpu.eval(frame)?, &gpu.eval(frame)?, &mut max)?;
    }
    // Above parity evaluations warm both paths; model compilation/upload excluded,
    // all per-evaluation geometry, allocations/copies/kernels/reduction included.
    let before = Instant::now();
    for _ in 0..repeats {
        for frame in &frames {
            std::hint::black_box(cpu.eval(frame)?);
        }
    }
    let cpu_ms = before.elapsed().as_secs_f64() * 1000.0;
    let before = Instant::now();
    for _ in 0..repeats {
        for frame in &frames {
            std::hint::black_box(gpu.eval(frame)?);
        }
    }
    let cuda_ms = before.elapsed().as_secs_f64() * 1000.0;
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"device":gpu.device_info(),"frames":frames.len(),"atoms":frames.iter().map(|f|f.positions.len()).collect::<Vec<_>>(),"repeats":repeats,
        "max_abs_energy_force_virial":max,"comparison_tolerance":"1e-9 absolute + 1e-9 relative","startup_ms":startup_ms,"warmed_cpu_total_ms":cpu_ms,"warmed_cuda_total_ms":cuda_ms,"cpu_over_cuda_speed_ratio":cpu_ms/cuda_ms,
        "timing_scope":"same-model end-to-end EFV; excludes initialization, includes CPU geometry, transfers, kernels and force/virial reduction; single bounded run, not a general benchmark"})
        )?
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("CUDA check stopped: {e}");
        std::process::exit(1);
    }
}
