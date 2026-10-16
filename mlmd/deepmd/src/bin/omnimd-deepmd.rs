//! JSON evaluator and inspector with a fresh device-isolated worker per request.
//! Complete-result publication requires an output filesystem supporting hard links.
use omnimd_deepmd::{
    AtomicEvaluation, DeepMd, DetailedEvaluation, EvalParameters, LoadOptions, ModelDetails,
};
use omnimd_ml_contract::{Evaluation, Frame, ModelMetadata, Potential};
use serde::{Deserialize, Serialize};
use std::{
    error::Error,
    ffi::OsString,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    options: LoadOptions,
    frames: Vec<Frame>,
    #[serde(default)]
    parameters: Option<Vec<EvalParameters>>,
    #[serde(default)]
    atomic: bool,
    #[serde(default = "default_batch_size")]
    batch_size: usize,
    timing: Option<TimingRequest>,
}

fn default_batch_size() -> usize {
    1
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TimingRequest {
    warmup: usize,
    iterations: usize,
}

impl Request {
    fn validate(&self) -> Result<(), Box<dyn Error>> {
        if self.frames.is_empty() {
            return Err("at least one frame is required".into());
        }
        if self.batch_size == 0 {
            return Err("batch_size must be positive".into());
        }
        if self
            .parameters
            .as_ref()
            .is_some_and(|p| p.len() != self.frames.len())
        {
            return Err("parameters must contain exactly one entry per frame".into());
        }
        for frame in &self.frames {
            frame.validate()?;
        }
        if let Some(timing) = &self.timing {
            if timing.warmup == 0
                || timing.iterations == 0
                || timing.warmup > 100
                || timing.iterations > 1000
            {
                return Err("timing requires warmup=1..100 and iterations=1..1000".into());
            }
            self.frames
                .len()
                .checked_mul(1 + timing.warmup + timing.iterations)
                .ok_or("timing evaluation count overflow")?;
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct Response {
    metadata: ModelMetadata,
    model_details: ModelDetails,
    evaluations: Vec<Evaluation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    atomic_evaluations: Option<Vec<AtomicEvaluation>>,
    batch_size: usize,
    capi_calls: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    cuda_kernel_launches: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timing: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct Inspection {
    metadata: ModelMetadata,
    model_details: ModelDetails,
    #[serde(skip_serializing_if = "Option::is_none")]
    cuda_kernel_launches: Option<u64>,
}

// Both normal and timed passes use this same chunk/parameter pairing. Timed
// passes discard each completed chunk to avoid retaining another full result.
fn evaluate_pass<F>(
    request: &Request,
    collect: bool,
    mut evaluate: F,
) -> Result<(Vec<DetailedEvaluation>, usize), Box<dyn Error>>
where
    F: FnMut(
        &[Frame],
        Option<&[EvalParameters]>,
        bool,
    ) -> omnimd_ml_contract::Result<Vec<DetailedEvaluation>>,
{
    let mut output = Vec::new();
    let mut offset = 0;
    let mut calls = 0;
    for frames in request.frames.chunks(request.batch_size) {
        let end = offset + frames.len();
        let parameters = request.parameters.as_ref().map(|p| &p[offset..end]);
        let values = evaluate(frames, parameters, request.atomic)?;
        if values.len() != frames.len() {
            return Err("DeepMD returned an unexpected number of frame evaluations".into());
        }
        if request.atomic && values.iter().any(|value| value.atomic.is_none()) {
            return Err("DeepMD did not return requested atomic evaluations".into());
        }
        if collect {
            output.extend(values);
        }
        offset = end;
        calls += 1;
    }
    Ok((output, calls))
}

fn worker(input: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    let request: Request = serde_json::from_str(&fs::read_to_string(input)?)?;
    request.validate()?;
    // The parent configures visibility before this fresh worker loads a runtime.
    let start = Instant::now();
    let mut model = unsafe { DeepMd::load(&request.options) }?;
    let load_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let (detailed, first_pass_capi_calls) =
        evaluate_pass(&request, true, |f, p, a| model.evaluate(f, p, a))?;
    let first_pass_seconds = start.elapsed().as_secs_f64();
    let mut capi_calls = first_pass_capi_calls;
    let timing = if let Some(timing) = &request.timing {
        let mut warmup_capi_calls = 0;
        let start = Instant::now();
        for _ in 0..timing.warmup {
            warmup_capi_calls +=
                evaluate_pass(&request, false, |f, p, a| model.evaluate(f, p, a))?.1;
        }
        let warmup_seconds = start.elapsed().as_secs_f64();
        let mut steady_capi_calls = 0;
        let start = Instant::now();
        for _ in 0..timing.iterations {
            steady_capi_calls +=
                evaluate_pass(&request, false, |f, p, a| model.evaluate(f, p, a))?.1;
        }
        let steady_seconds = start.elapsed().as_secs_f64();
        let steady_evaluations = timing.iterations * request.frames.len();
        capi_calls += warmup_capi_calls + steady_capi_calls;
        Some(serde_json::json!({
            "load_seconds": load_seconds,
            "first_pass_seconds": first_pass_seconds,
            "warmup_seconds": warmup_seconds,
            "warmup_passes": timing.warmup,
            "steady_seconds": steady_seconds,
            "steady_passes": timing.iterations,
            "steady_evaluations": steady_evaluations,
            "seconds_per_evaluation": steady_seconds / steady_evaluations as f64,
            "batch_size": request.batch_size,
            "first_pass_capi_calls": first_pass_capi_calls,
            "warmup_capi_calls": warmup_capi_calls,
            "steady_capi_calls": steady_capi_calls,
            "total_capi_calls": capi_calls,
            "scope": "host validation, packing, combined CAPI EFV and requested atomic outputs, transfers, GPU attestation/synchronization and output validation; excludes process startup and JSON serialization"
        }))
    } else {
        None
    };
    let mut evaluations = Vec::with_capacity(detailed.len());
    let mut atomic_evaluations = request.atomic.then(|| Vec::with_capacity(detailed.len()));
    for value in detailed {
        evaluations.push(value.evaluation);
        if let Some(atomic) = &mut atomic_evaluations {
            atomic.push(value.atomic.ok_or("missing requested atomic evaluation")?);
        }
    }
    publish_json(
        output,
        &Response {
            metadata: model.metadata().clone(),
            model_details: model.details().clone(),
            evaluations,
            atomic_evaluations,
            batch_size: request.batch_size,
            capi_calls,
            cuda_kernel_launches: model.cuda_kernel_launches(),
            timing,
        },
    )
}

fn inspect_worker(input: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    let options: LoadOptions = serde_json::from_str(&fs::read_to_string(input)?)?;
    let model = unsafe { DeepMd::load(&options) }?;
    // No evaluation occurs here. CUDA metadata therefore remains pending.
    publish_json(
        output,
        &Inspection {
            metadata: model.metadata().clone(),
            model_details: model.details().clone(),
            cuda_kernel_launches: model.cuda_kernel_launches(),
        },
    )
}

// Publish a complete file with atomic no-clobber semantics. The sibling file is
// on the output filesystem; hard_link fails rather than replacing any existing
// destination. Filesystems without hard links fail without publishing a result.
fn publish_json<T: Serialize>(output: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    let mut encoded = serde_json::to_vec_pretty(value)?;
    encoded.push(b'\n');
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    static TEMP_ID: AtomicU64 = AtomicU64::new(0);
    let (temporary, mut file) = (0..64)
        .find_map(|_| {
            let path = parent.join(format!(
                ".omnimd-deepmd-{}-{}.tmp",
                std::process::id(),
                TEMP_ID.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(file) => Some(Ok((path, file))),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => Some(Err(e)),
            }
        })
        .ok_or("could not create a unique output temporary file")??;
    let written = file.write_all(&encoded).and_then(|()| file.sync_all());
    drop(file);
    let result = written.and_then(|()| fs::hard_link(&temporary, output));
    let _ = fs::remove_file(&temporary);
    result.map_err(|e| {
        format!(
            "publish complete result to {} (output filesystem must support hard links): {e}",
            output.display()
        )
    })?;
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
enum Operation {
    Evaluate,
    Inspect,
}

#[derive(Debug)]
struct Cli {
    operation: Operation,
    worker: bool,
    input: PathBuf,
    output: PathBuf,
}

fn parse_cli(args: &[OsString]) -> Result<Cli, Box<dyn Error>> {
    let (operation, worker, paths) = match args {
        [input, output] => (Operation::Evaluate, false, [input, output]),
        [mode, input, output] if mode == "inspect" => (Operation::Inspect, false, [input, output]),
        [mode, input, output] if mode == "--worker" => (Operation::Evaluate, true, [input, output]),
        [mode, input, output] if mode == "--inspect-worker" => (Operation::Inspect, true, [input, output]),
        _ => return Err("usage: omnimd-deepmd REQUEST.json NEW_OUTPUT.json | omnimd-deepmd inspect OPTIONS.json NEW_INFO.json (isolated worker; output must not exist)".into()),
    };
    Ok(Cli {
        operation,
        worker,
        input: paths[0].into(),
        output: paths[1].into(),
    })
}

fn run() -> Result<(), Box<dyn Error>> {
    let cli = parse_cli(&std::env::args_os().skip(1).collect::<Vec<_>>())?;
    if cli.output.exists() {
        return Err("output already exists".into());
    }
    if cli.worker {
        return match cli.operation {
            Operation::Evaluate => worker(&cli.input, &cli.output),
            Operation::Inspect => inspect_worker(&cli.input, &cli.output),
        };
    }
    let input = fs::read_to_string(&cli.input)?;
    let options = match cli.operation {
        Operation::Evaluate => {
            let request: Request = serde_json::from_str(&input)?;
            request.validate()?;
            request.options
        }
        Operation::Inspect => serde_json::from_str::<LoadOptions>(&input)?,
    };
    let mut command = Command::new(std::env::current_exe()?);
    command
        .arg(match cli.operation {
            Operation::Evaluate => "--worker",
            Operation::Inspect => "--inspect-worker",
        })
        .arg(&cli.input)
        .arg(&cli.output);
    options.device.configure_worker(&mut command)?;
    let status = command.status()?;
    if !status.success() {
        return Err(format!("DeepMD worker failed: {status}").into());
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("omnimd-deepmd: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use omnimd_ml_contract::Units;

    fn request_json() -> serde_json::Value {
        serde_json::json!({
            "options": {"library": "runtime", "model": "model.pb", "type_map": ["H"], "units": "EvAngstrom"},
            "frames": [{"positions": [[0.0, 0.0, 0.0]], "species": ["H"], "cell": null, "units": "EvAngstrom"}]
        })
    }

    #[test]
    fn legacy_defaults_and_invalid_controls() {
        let request: Request = serde_json::from_value(request_json()).unwrap();
        request.validate().unwrap();
        assert_eq!(request.batch_size, 1);
        assert!(!request.atomic);
        assert!(request.parameters.is_none());
        for (key, value) in [
            ("batch_size", serde_json::json!(0)),
            ("parameters", serde_json::json!([])),
            ("timing", serde_json::json!({"warmup": 0, "iterations": 1})),
        ] {
            let mut json = request_json();
            json[key] = value;
            let request: Request = serde_json::from_value(json).unwrap();
            assert!(request.validate().is_err());
        }
        let mut json = request_json();
        json["atomic_typo"] = true.into();
        assert!(serde_json::from_value::<Request>(json).is_err());
    }

    #[test]
    fn inspect_and_worker_commands_are_distinct() {
        for (flag, expected_worker) in [("inspect", false), ("--inspect-worker", true)] {
            let cli = parse_cli(&[flag.into(), "options.json".into(), "info.json".into()]).unwrap();
            assert_eq!(cli.operation, Operation::Inspect);
            assert_eq!(cli.worker, expected_worker);
            assert_eq!(cli.input, PathBuf::from("options.json"));
        }
        assert!(parse_cli(&["unknown".into(), "in".into(), "out".into()]).is_err());
    }

    #[test]
    fn normal_and_timed_passes_preserve_parameter_chunks() {
        let mut request: Request = serde_json::from_value(request_json()).unwrap();
        request.frames = vec![request.frames[0].clone(); 5];
        request.parameters = Some(
            (0..5)
                .map(|i| EvalParameters {
                    fparam: vec![i as f64],
                    aparam: vec![],
                })
                .collect(),
        );
        request.batch_size = 2;
        request.atomic = true;
        request.validate().unwrap();
        for collect in [true, false] {
            let mut seen = vec![];
            let (output, calls) = evaluate_pass(&request, collect, |frames, params, atomic| {
                assert!(atomic);
                seen.push(
                    params
                        .unwrap()
                        .iter()
                        .map(|p| p.fparam[0] as usize)
                        .collect::<Vec<_>>(),
                );
                Ok(frames
                    .iter()
                    .map(|_| DetailedEvaluation {
                        evaluation: Evaluation {
                            energy: 0.0,
                            forces: vec![[0.0; 3]],
                            virial: [[0.0; 3]; 3],
                            units: Units::EvAngstrom,
                        },
                        atomic: Some(AtomicEvaluation {
                            energies: vec![0.0],
                            virials: vec![[[0.0; 3]; 3]],
                        }),
                    })
                    .collect())
            })
            .unwrap();
            assert_eq!(seen, vec![vec![0, 1], vec![2, 3], vec![4]]);
            assert_eq!(calls, 3);
            assert_eq!(output.len(), if collect { 5 } else { 0 });
        }
    }

    #[test]
    fn complete_publication_preserves_existing_output() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory =
            std::env::temp_dir().join(format!("omnimd-publish-{}-{unique}", std::process::id()));
        fs::create_dir(&directory).unwrap();
        let output = directory.join("result.json");
        let value = serde_json::json!({"complete": true});
        publish_json(&output, &value).unwrap();
        let original = fs::read(&output).unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&original).unwrap(),
            value
        );
        assert!(publish_json(&output, &serde_json::json!({"overwrite":true})).is_err());
        assert_eq!(fs::read(&output).unwrap(), original);
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
        fs::remove_file(&output).unwrap();
        fs::remove_dir(&directory).unwrap();
    }
}
