//! JSON-lines evaluation: one shared-contract Frame per input line.
use omnimd_ml_contract::{Frame, Potential};
use omnimd_native::NativePotential;
use std::io::{BufRead, Write};

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        return Err("usage: omnimd-native-eval MODEL.json < frames.jsonl".into());
    }
    let mut potential = NativePotential::load(&args[1])?;
    let stdin = std::io::stdin();
    let mut stdout = std::io::BufWriter::new(std::io::stdout().lock());
    for (line_number, line) in stdin.lock().lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let frame: Frame = serde_json::from_str(&line)
            .map_err(|e| format!("frame line {}: {e}", line_number + 1))?;
        let evaluation = potential
            .eval(&frame)
            .map_err(|e| format!("frame line {}: {e}", line_number + 1))?;
        serde_json::to_writer(&mut stdout, &evaluation)?;
        writeln!(stdout)?;
        stdout.flush()?;
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
