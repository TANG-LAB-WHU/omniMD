//! CPU ensemble CLI, using existing model loaders and dataset records.
use super::{Backend, BackendOptions};
use omnimd_deepmd::Device;
use omnimd_ml_contract::Potential;
use omnimd_mlmd::{
    dataset::{self, DatasetReader},
    deviation::{self, ForceWindow, MAX_MODELS},
};
use serde::Deserialize;
use serde_json::json;
use std::{
    collections::HashSet,
    error::Error,
    ffi::OsString,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Member {
    id: String,
    options: BackendOptions,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: String,
    models: Vec<Member>,
    dataset: PathBuf,
    #[serde(default)]
    per_atom: bool,
    force_window: Option<ForceWindow>,
}
impl Request {
    fn read(path: &Path) -> Result<Self> {
        let request: Self = serde_json::from_str(&fs::read_to_string(path)?)?;
        if request.schema != "omnimd-model-deviation-v1" {
            return Err("unsupported model-deviation request schema".into());
        }
        if !(2..=MAX_MODELS).contains(&request.models.len()) {
            return Err(format!("model deviation requires 2..={MAX_MODELS} models").into());
        }
        if !request.dataset.is_absolute() {
            return Err("dataset path must be absolute".into());
        }
        if let Some(window) = request.force_window {
            window.validate()?;
        }
        let mut ids = HashSet::new();
        let mut library = None;
        for member in &request.models {
            if member.id.trim().is_empty() || member.id.len() > 128 || !ids.insert(&member.id) {
                return Err("model IDs must be nonempty, unique and at most 128 bytes".into());
            }
            if !matches!(member.options.device(), Device::Cpu) {
                return Err(
                    "model-deviation currently requires explicit/default CPU for every member"
                        .into(),
                );
            }
            member.options.validate()?;
            if let BackendOptions::DeepMd(options) = &member.options {
                let path = options.library.canonicalize()?;
                if library.as_ref().is_some_and(|old| *old != path) {
                    return Err(
                        "all C API ensemble members must use the same runtime library".into(),
                    );
                }
                library = Some(path);
            }
        }
        Ok(request)
    }
}
fn worker(input: &Path, output: &Path) -> Result<()> {
    let request = Request::read(input)?;
    if output.exists() {
        return Err("output already exists".into());
    }
    let mut reader = DatasetReader::open_hashed(&request.dataset)?;
    if reader.header.parameter_layout.is_some() {
        return Err("model-deviation does not yet accept parameterized dataset v2".into());
    }
    let mut models = Vec::with_capacity(request.models.len());
    let mut provenance = Vec::with_capacity(request.models.len());
    let mut hashes = HashSet::new();
    for member in &request.models {
        let hash = dataset::sha256(member.options.model_path())?;
        hashes.insert(hash.clone());
        let model = member.options.load()?;
        if let Backend::DeepMd(deepmd) = &model {
            let details = deepmd.details();
            if details.frame_parameter_dim != 0 || details.atom_parameter_dim != 0 {
                return Err(format!("model {} requires parameters; use the standalone parameter-aware C API instead", member.id).into());
            }
        }
        if model.metadata().type_map != reader.header.type_map
            || model.metadata().units != reader.header.units
        {
            return Err(format!(
                "model {} and dataset must have identical ordered type maps and units",
                member.id
            )
            .into());
        }
        let library_hash = match &member.options {
            BackendOptions::DeepMd(options) => Some(dataset::sha256(&options.library)?),
            BackendOptions::Native(_) => None,
        };
        provenance.push(json!({"id":member.id,"model":member.options.model_path(),
            "model_sha256":hash,"library_sha256":library_hash,"metadata":model.metadata()}));
        models.push(model);
    }
    let mut writer = dataset::new_output(output)?;
    dataset::write_line(
        &mut writer,
        &json!({"kind":"header","schema":"omnimd-model-deviation-jsonl-v1",
        "dataset_header":reader.header,"models":provenance,"distinct_model_byte_hashes":hashes.len(),
        "device":"cpu","per_atom":request.per_atom,"force_window":request.force_window,
        "statistics":"absolute population std over models; E and all 9 V components per atom; F vector std norm; force mean is arithmetic, virial aggregate is RMS",
        "selection":"min <= max force deviation < max; candidate=null when no force window",
        "interpretation":"Model disagreement only; no calibrated uncertainty, physical accuracy, energy-offset alignment or use of reference labels. Duplicate byte hashes reduce ensemble diversity.",
        "provenance":"exact read-byte dataset SHA256 is in the terminal completed record; source hashes are declarations, not verified source files"}),
    )?;
    let mut frames = 0_u64;
    let mut candidates = 0_u64;
    let mut maximum_force_deviation = 0_f64;
    while let Some(data) = reader.next_frame()? {
        let statistics = deviation::evaluate_frame(&mut models, &data.frame, request.per_atom)
            .map_err(|e| format!("dataset frame {} ({}): {e}", data.index, data.id))?;
        let max_force = statistics.force_ev_per_angstrom.max;
        let candidate = request
            .force_window
            .map(|window| window.contains(max_force));
        frames = frames.checked_add(1).ok_or("frame count overflow")?;
        if candidate == Some(true) {
            candidates = candidates
                .checked_add(1)
                .ok_or("candidate count overflow")?;
        }
        maximum_force_deviation = maximum_force_deviation.max(max_force);
        dataset::write_line(
            &mut writer,
            &json!({"kind":"deviation","index":data.index,"id":data.id,
            "natoms":data.frame.positions.len(),"statistics":statistics,"candidate":candidate}),
        )?;
    }
    let fingerprint = reader.completed_provenance()?;
    dataset::write_line(
        &mut writer,
        &json!({"kind":"completed","frames":frames,
        "candidate_frames":request.force_window.map(|_| candidates),
        "max_force_deviation_ev_per_angstrom":maximum_force_deviation,
        "model_loads":models.len(),"combined_evaluations":frames.checked_mul(models.len() as u64).ok_or("evaluation count overflow")?,
        "dataset_sha256":fingerprint.dataset_sha256,"declared_source_sha256":fingerprint.declared_source_sha256,
        "source_hashes_verified":false}),
    )?;
    writer.flush()?;
    Ok(())
}
pub fn dispatch(args: &[OsString]) -> Result<bool> {
    if args.len() == 3 && args[0] == "--deviation-worker" {
        worker(Path::new(&args[1]), Path::new(&args[2]))?;
        return Ok(true);
    }
    if args.len() == 3 && args[0] == "model-deviation" {
        if Path::new(&args[2]).exists() {
            return Err("output already exists".into());
        }
        Request::read(Path::new(&args[1]))?;
        let mut command = Command::new(std::env::current_exe()?);
        command.arg("--deviation-worker").args(&args[1..]);
        Device::Cpu.configure_worker(&mut command)?;
        let status = command.status()?;
        if !status.success() {
            return Err(format!("CPU model-deviation worker failed: {status}; partial output without completed is invalid").into());
        }
        return Ok(true);
    }
    Ok(false)
}
