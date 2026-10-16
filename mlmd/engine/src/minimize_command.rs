//! Fixed-cell minimization command. Geometry exports never create reference labels.
use super::BackendOptions;
use omnimd_ml_contract::Frame;
use omnimd_mlmd::{
    dataset::{
        self, DatasetFrame, DatasetHeader, FrameReference, Labels, Record, VirialConvention, SCHEMA,
    },
    xyz,
};
use omnimd_sim::min::{MlipMinimizationSettings, MlipMinimizationStatus, MlipMinimizer};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::BTreeMap,
    error::Error,
    fs,
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    force_tolerance_ev_per_angstrom: f64,
    energy_tolerance_ev: f64,
    initial_step_angstrom2_per_ev: f64,
    max_displacement_angstrom: f64,
    max_backtracks: u32,
    max_iterations: u64,
    armijo: f64,
}
impl Settings {
    fn algorithm(&self) -> MlipMinimizationSettings {
        MlipMinimizationSettings {
            force_tolerance_ev_per_angstrom: self.force_tolerance_ev_per_angstrom,
            energy_tolerance_ev: self.energy_tolerance_ev,
            initial_step_angstrom2_per_ev: self.initial_step_angstrom2_per_ev,
            max_displacement_angstrom: self.max_displacement_angstrom,
            max_backtracks: self.max_backtracks,
            max_iterations: self.max_iterations,
            armijo: self.armijo,
        }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    options: BackendOptions,
    frame: Option<Frame>,
    dataset_frame: Option<FrameReference>,
    minimization: Settings,
}

fn suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(suffix);
    PathBuf::from(name)
}
fn snapshot(
    writer: &mut impl Write,
    min: &MlipMinimizer<super::Backend>,
) -> Result<(), Box<dyn Error>> {
    dataset::write_line(
        writer,
        &json!({
            "kind":"iteration","iteration":min.iterations(),"evaluations":min.evaluations(),
            "frame":min.frame(),"evaluation":min.evaluation(),"max_force_ev_per_angstrom":min.max_force_ev_per_angstrom(),
            "energy_change_ev":min.energy_change_ev()
        }),
    )
}
pub fn worker(input: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    let request: Request = serde_json::from_str(&fs::read_to_string(input)?)?;
    let extxyz = suffix(output, ".final.extxyz");
    let dataset_path = suffix(output, ".final.dataset.jsonl");
    if [output, extxyz.as_path(), dataset_path.as_path()]
        .iter()
        .any(|p| p.exists())
    {
        return Err("minimization output or final structure already exists".into());
    }
    let frame = match (request.frame, request.dataset_frame.as_ref()) {
        (Some(frame), None) => frame,
        (None, Some(reference)) => reference.resolve()?,
        _ => return Err("minimize requires exactly one frame or dataset_frame".into()),
    };
    let hash = dataset::sha256(request.options.model_path())?;
    let mut min = MlipMinimizer::new(
        request.options.load()?,
        frame,
        request.minimization.algorithm(),
    )?;
    // Reserve every output name before optimization; never replace existing files.
    let mut writer = dataset::new_output(output)?;
    let mut xyz_writer = dataset::new_output(&extxyz)?;
    let mut frame_writer = dataset::new_output(&dataset_path)?;
    dataset::write_line(
        &mut writer,
        &json!({
            "kind":"header","schema":"omnimd-minimize-jsonl-v1","model_sha256":hash,
            "metadata":min.metadata(),"minimization":request.minimization,
            "dataset_frame":request.dataset_frame,"cell_policy":"fixed","method":"bounded upstream steepest descent with Armijo backtracking"
        }),
    )?;
    snapshot(&mut writer, &min)?;
    let initial_energy = min.evaluation().energy;
    let mut failure = None;
    while min.status() == MlipMinimizationStatus::Running {
        match min.step() {
            Ok(_) => snapshot(&mut writer, &min)?,
            Err(error) => {
                failure = Some(error);
                break;
            }
        }
    }
    let status = if failure.is_some() {
        "evaluation_or_numerical_error"
    } else {
        min.status().as_str()
    };
    xyz::export_extxyz(&mut xyz_writer, min.frame(), status)?;
    xyz_writer.flush()?;
    dataset::write_line(
        &mut frame_writer,
        &Record::Header {
            header: DatasetHeader {
                parameter_layout: None,
                schema: SCHEMA.into(),
                type_map: min.metadata().type_map.clone(),
                units: min.frame().units,
                virial_convention: VirialConvention::MinusDEpsilonRow,
                source: format!("minimization last accepted geometry; status={status}"),
                selected_sets: vec![],
            },
        },
    )?;
    dataset::write_line(
        &mut frame_writer,
        &Record::Frame {
            data: DatasetFrame {
                parameters: None,
                index: 0,
                id: "minimized/0".into(),
                frame: min.frame().clone(),
                labels: Labels::default(),
            },
        },
    )?;
    dataset::write_line(
        &mut frame_writer,
        &Record::Completed {
            frames: 1,
            source_sha256: BTreeMap::from([("model".into(), hash)]),
        },
    )?;
    frame_writer.flush()?;
    if let Some(error) = failure {
        dataset::write_line(
            &mut writer,
            &json!({"kind":"failed","status":status,"error":error.to_string(),"final_extxyz":extxyz,"final_dataset":dataset_path}),
        )?;
        writer.flush()?;
        return Err(error.into());
    }
    dataset::write_line(
        &mut writer,
        &json!({
            "kind":"completed","status":status,"converged":min.status().converged(),"iterations":min.iterations(),
            "evaluations":min.evaluations(),"initial_energy_ev":initial_energy,"final_energy_ev":min.evaluation().energy,
            "energy_change_ev":min.energy_change_ev(),"max_force_ev_per_angstrom":min.max_force_ev_per_angstrom(),
            "force_criterion_met":min.max_force_ev_per_angstrom()<=request.minimization.force_tolerance_ev_per_angstrom,
            "energy_change_criterion_met":min.energy_change_ev().map(|delta| delta.abs()<=request.minimization.energy_tolerance_ev),
            "final_extxyz":extxyz,"final_dataset":dataset_path,
            "initial_stationary_policy":"already_converged verifies force only; energy-change criterion is null without an accepted step"
        }),
    )?;
    writer.flush()?;
    if !min.status().converged() {
        return Err(format!(
            "minimization did not converge: {status}; last accepted structure was exported"
        )
        .into());
    }
    Ok(())
}
