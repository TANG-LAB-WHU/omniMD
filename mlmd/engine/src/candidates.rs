//! The candidates of an exploration. A run with model deviation writes, with
//! every sampled frame, how far the members of the ensemble disagree on it.
//! This reads any number of such trajectories and gives the two things an
//! active-learning loop goes on with: the frames whose largest force
//! deviation lies in a window, at most so many of one trajectory, as one
//! unlabeled dataset; and the share of the frames the members agree on.
//!
//! Every trajectory is read and checked completely before anything is
//! written, and read a second time for the frames that were picked.
use crate::{
    dataset::{
        self, DatasetFrame, DatasetHeader, Labels, Record, Result, VirialConvention, SCHEMA,
    },
    deviation::ForceWindow,
};
use omnimd_ml_contract::{Frame, ModelMetadata, Units};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    fs::{self, File},
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
};

pub const REQUEST_SCHEMA: &str = "omnimd-candidates-v1";
pub const SUMMARY_SCHEMA: &str = "omnimd-candidates-summary-v1";
/// Most trajectories of one collection.
pub const MAX_TRAJECTORIES: usize = 4096;
/// The dataset of the picked frames, in the output directory.
pub const DATASET: &str = "candidates.jsonl";
/// The summary, written last: a directory with it is complete.
pub const SUMMARY: &str = "summary.json";
const MAX_LINE_BYTES: usize = 16 * 1024 * 1024;
const REPLICAS_SCHEMA: &str = "omnimd-replicas-run-v1";

/// One trajectory of a run with `deviation`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    /// Written by `omnimd-mlmd run` or `run-replicas`, with its `completed`
    /// record.
    pub trajectory: PathBuf,
    /// Its frames get the IDs `ID_PREFIX/step-N` with the absolute step N.
    pub id_prefix: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidatesRequest {
    pub schema: String,
    /// Trajectories one by one.
    #[serde(default)]
    pub trajectories: Vec<Source>,
    /// Output directories of `run-replicas`: every trajectory of each, the
    /// name of the replica as its ID prefix. They follow `trajectories`.
    #[serde(default)]
    pub replica_runs: Vec<PathBuf>,
    /// Leave out frames before this absolute step (the equilibration, say):
    /// they are neither counted nor picked.
    #[serde(default)]
    pub first_step: u64,
    /// At most this many candidates of one trajectory. Of more, the first,
    /// the last and those evenly spaced between them in time are taken.
    pub max_per_trajectory: Option<u64>,
    /// The window to judge every frame by. Without it each trajectory is
    /// judged by the window its run was made with.
    pub force_window: Option<ForceWindow>,
}

/// A picked frame.
#[derive(Debug, Serialize)]
pub struct Picked {
    pub id: String,
    pub step: u64,
    pub max_force_deviation_ev_per_angstrom: f64,
}

/// What one trajectory held at and after `first_step`.
#[derive(Debug, Serialize)]
pub struct TrajectorySummary {
    pub id_prefix: String,
    pub trajectory: PathBuf,
    pub trajectory_sha256: String,
    pub model_sha256: Option<String>,
    /// The ensemble of the run, as its header states it.
    pub ensemble: serde_json::Value,
    /// The window its frames were judged by.
    pub force_window: ForceWindow,
    pub frames: u64,
    /// Frames below, inside and at or above the window.
    pub accurate_frames: u64,
    pub candidate_frames: u64,
    pub failed_frames: u64,
    pub accurate_fraction: f64,
    pub max_force_deviation_ev_per_angstrom: f64,
    pub picked: Vec<Picked>,
}

#[derive(Debug, Serialize)]
pub struct CandidatesSummary {
    pub schema: &'static str,
    pub first_step: u64,
    pub max_per_trajectory: Option<u64>,
    /// The window of the request; null when every trajectory was judged by
    /// its own.
    pub force_window: Option<ForceWindow>,
    pub trajectories: Vec<TrajectorySummary>,
    pub frames: u64,
    pub accurate_frames: u64,
    pub candidate_frames: u64,
    pub failed_frames: u64,
    /// Accurate frames over all frames.
    pub accurate_fraction: f64,
    /// The mean of the trajectories' accurate fractions: every run counts
    /// the same, however many frames it sampled.
    pub mean_accurate_fraction: f64,
    pub picked_frames: u64,
    /// The dataset of the picked frames beside this summary; null when no
    /// frame was picked (a dataset holds at least one frame).
    pub dataset: Option<&'static str>,
    pub dataset_sha256: Option<String>,
}

/// A trajectory to read and the prefix of the IDs of its frames.
struct Input {
    trajectory: PathBuf,
    id_prefix: String,
}

#[derive(Deserialize)]
struct Largest {
    max: f64,
}

/// Of the model deviation of a frame, what the window is compared with.
#[derive(Deserialize)]
struct Disagreement {
    force_ev_per_angstrom: Largest,
}

/// Of the `deviation` entry of a trajectory header.
#[derive(Deserialize)]
struct RunDeviation {
    force_window: Option<ForceWindow>,
}

/// The records of a trajectory, with the fields this collection reads.
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Line {
    Header {
        schema: String,
        metadata: ModelMetadata,
        #[serde(default)]
        ensemble: serde_json::Value,
        #[serde(default)]
        model_sha256: Option<String>,
        #[serde(default)]
        deviation: Option<RunDeviation>,
    },
    Frame {
        step: u64,
        frame: Frame,
        #[serde(default)]
        candidate: Option<bool>,
        #[serde(default)]
        model_deviation: Option<Disagreement>,
    },
    Completed {
        final_step: u64,
    },
}

/// The records of a file, one after the other, and the hash of what was read.
struct Lines {
    reader: BufReader<File>,
    hash: Sha256,
    line: Vec<u8>,
}

impl Lines {
    fn open(path: &Path) -> Result<Self> {
        Ok(Self {
            reader: BufReader::new(File::open(path)?),
            hash: Sha256::new(),
            line: Vec::new(),
        })
    }

    fn next(&mut self) -> Result<Option<Line>> {
        self.line.clear();
        let limit = MAX_LINE_BYTES as u64 + 1;
        (&mut self.reader)
            .take(limit)
            .read_until(b'\n', &mut self.line)?;
        if self.line.is_empty() {
            return Ok(None);
        }
        if self.line.len() > MAX_LINE_BYTES || self.line.last() != Some(&b'\n') {
            return Err("trajectory record is unterminated or exceeds 16 MiB".into());
        }
        self.hash.update(&self.line);
        Ok(Some(serde_json::from_slice(&self.line)?))
    }

    fn sha256(self) -> String {
        format!("{:x}", self.hash.finalize())
    }
}

/// What the first reading of a trajectory found.
struct Scan {
    sha256: String,
    type_map: Vec<String>,
    units: Units,
    ensemble: serde_json::Value,
    model_sha256: Option<String>,
    window: ForceWindow,
    frames: u64,
    accurate: u64,
    failed: u64,
    maximum: f64,
    /// Step and largest force deviation of every candidate, in order.
    candidates: Vec<(u64, f64)>,
}

fn scan(input: &Input, request: &CandidatesRequest) -> Result<Scan> {
    let mut lines = Lines::open(&input.trajectory)?;
    let Some(Line::Header {
        schema,
        metadata,
        ensemble,
        model_sha256,
        deviation,
    }) = lines.next()?
    else {
        return Err("trajectory must start with its header".into());
    };
    if !matches!(
        schema.as_str(),
        "omnimd-mlmd-jsonl-v2" | "omnimd-mlmd-jsonl-v4"
    ) {
        return Err(format!(
            "trajectory schema {schema} is not supported (fixed-parameter trajectories carry conditioning that a dataset frame would drop)"
        )
        .into());
    }
    metadata.validate()?;
    let Some(deviation) = deviation else {
        return Err(
            "the run was made without deviation: its frames carry no model deviation".into(),
        );
    };
    // The window of the request, or else the one the run marked its frames
    // by; those marks are then checked against the deviations they came from.
    let (window, marked) = match (request.force_window, deviation.force_window) {
        (Some(window), _) => (window, false),
        (None, Some(window)) => (window, true),
        (None, None) => {
            return Err(
                "the run was made without a force window; give force_window in the request".into(),
            )
        }
    };
    window.validate()?;
    let mut found = Scan {
        sha256: String::new(),
        type_map: metadata.type_map.clone(),
        units: metadata.units,
        ensemble,
        model_sha256,
        window,
        frames: 0,
        accurate: 0,
        failed: 0,
        maximum: 0.0,
        candidates: Vec::new(),
    };
    let mut last_step = None;
    let mut completed = None;
    while let Some(record) = lines.next()? {
        if completed.is_some() {
            return Err("trajectory continues after its completed record".into());
        }
        match record {
            Line::Header { .. } => return Err("trajectory has a second header".into()),
            Line::Completed { final_step } => completed = Some(final_step),
            Line::Frame {
                step,
                frame,
                candidate,
                model_deviation,
            } => {
                if last_step.is_some_and(|last| step <= last) {
                    return Err("trajectory steps must increase".into());
                }
                last_step = Some(step);
                frame.type_ids(&metadata)?;
                if step < request.first_step {
                    continue;
                }
                let Some(disagreement) = model_deviation else {
                    return Err(
                        format!("the frame of step {step} carries no model deviation").into(),
                    );
                };
                let largest = disagreement.force_ev_per_angstrom.max;
                if !largest.is_finite() || largest < 0.0 {
                    return Err(format!(
                        "the frame of step {step} carries a force deviation that is not a finite, nonnegative number"
                    )
                    .into());
                }
                let inside = window.contains(largest);
                if marked && candidate != Some(inside) {
                    return Err(format!(
                        "the candidate mark of the frame of step {step} contradicts its force deviation and the window of the run"
                    )
                    .into());
                }
                found.frames += 1;
                found.maximum = found.maximum.max(largest);
                found.accurate += u64::from(largest < window.min_ev_per_angstrom);
                found.failed += u64::from(largest >= window.max_ev_per_angstrom);
                if inside {
                    found.candidates.push((step, largest));
                }
            }
        }
    }
    if completed.is_none() || completed != last_step {
        return Err(
            "trajectory has no completed record matching its last frame; a failed or interrupted run is not collected"
                .into(),
        );
    }
    if found.frames == 0 {
        return Err("no trajectory frame at or after first_step".into());
    }
    found.sha256 = lines.sha256();
    Ok(found)
}

/// Which of `count` candidates in order of time are taken when at most
/// `most` may be: all of them, or else the first, the last and those evenly
/// spaced between, the k-th at position `k (count - 1) / (most - 1)` rounded
/// to the nearest, a half to the even one. A single one is the first.
fn spread(count: u64, most: Option<u64>) -> Vec<u64> {
    match most {
        Some(1) if count > 1 => vec![0],
        Some(most) if most < count => (0..most)
            .map(|k| {
                let numerator = u128::from(k) * u128::from(count - 1);
                let denominator = u128::from(most - 1);
                let (whole, rest) = (numerator / denominator, numerator % denominator);
                let up = 2 * rest > denominator || (2 * rest == denominator && whole % 2 == 1);
                (whole + u128::from(up)) as u64
            })
            .collect(),
        _ => (0..count).collect(),
    }
}

/// Reads the trajectory again and writes the frames of the picked steps,
/// numbered from `index` on. The file must be the one that was scanned.
fn write_picked(
    input: &Input,
    scanned: &Scan,
    picked: &HashSet<u64>,
    writer: &mut impl Write,
    index: &mut u64,
) -> Result<()> {
    let mut lines = Lines::open(&input.trajectory)?;
    let mut written = 0;
    while let Some(record) = lines.next()? {
        if let Line::Frame { step, frame, .. } = record {
            if !picked.contains(&step) {
                continue;
            }
            dataset::write_line(
                writer,
                &Record::Frame {
                    data: DatasetFrame {
                        parameters: None,
                        index: *index,
                        id: format!("{}/step-{step}", input.id_prefix),
                        frame,
                        labels: Labels::default(),
                    },
                },
            )?;
            *index += 1;
            written += 1;
        }
    }
    if lines.sha256() != scanned.sha256 || written != picked.len() {
        return Err("the trajectory changed while it was read".into());
    }
    Ok(())
}

fn plain_prefix(prefix: &str) -> bool {
    !prefix.is_empty()
        && prefix.len() <= 128
        && prefix
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
}

/// The trajectories of the request in its order: those listed, then those
/// of every replica run.
fn inputs(request: &CandidatesRequest) -> Result<Vec<Input>> {
    #[derive(Deserialize)]
    struct Replica {
        name: String,
    }
    #[derive(Deserialize)]
    struct ReplicaRun {
        schema: String,
        replicas: Vec<Replica>,
    }
    let mut all = Vec::new();
    for source in &request.trajectories {
        if !source.trajectory.is_absolute() {
            return Err("trajectory paths must be absolute".into());
        }
        all.push(Input {
            trajectory: source.trajectory.clone(),
            id_prefix: source.id_prefix.clone(),
        });
    }
    for directory in &request.replica_runs {
        if !directory.is_absolute() {
            return Err("replica run directories must be absolute".into());
        }
        let record = directory.join("replicas.json");
        let run: ReplicaRun = serde_json::from_str(&fs::read_to_string(&record).map_err(|e| {
            format!(
                "{}: {e} (a replica run that finished has this file)",
                record.display()
            )
        })?)
        .map_err(|e| format!("{}: {e}", record.display()))?;
        if run.schema != REPLICAS_SCHEMA {
            return Err(format!("{}: not the record of a replica run", record.display()).into());
        }
        for replica in run.replicas {
            all.push(Input {
                trajectory: directory.join(format!("{}.jsonl", replica.name)),
                id_prefix: replica.name,
            });
        }
    }
    if !(1..=MAX_TRAJECTORIES).contains(&all.len()) {
        return Err(format!("a collection takes 1..={MAX_TRAJECTORIES} trajectories").into());
    }
    let mut prefixes = HashSet::new();
    let mut paths = HashSet::new();
    for input in &all {
        if !plain_prefix(&input.id_prefix) {
            return Err(format!(
                "ID prefix {:?}: prefixes are 1..=128 characters of [A-Za-z0-9_.-]",
                input.id_prefix
            )
            .into());
        }
        if !prefixes.insert(input.id_prefix.as_str()) {
            return Err(format!(
                "ID prefix {:?} occurs twice; list trajectories of the same name one by one with prefixes of their own",
                input.id_prefix
            )
            .into());
        }
        if !paths.insert(input.trajectory.as_path()) {
            return Err(format!("{} is listed twice", input.trajectory.display()).into());
        }
    }
    Ok(all)
}

fn fraction(part: u64, whole: u64) -> f64 {
    part as f64 / whole as f64
}

/// Collects the candidates of the request's trajectories into the new
/// directory `output`: `candidates.jsonl` when a frame was picked, and
/// `summary.json`, which is also returned. Nothing is written before every
/// trajectory has been read and checked.
pub fn collect_candidates(request: &CandidatesRequest, output: &Path) -> Result<CandidatesSummary> {
    if request.schema != REQUEST_SCHEMA {
        return Err("unsupported candidates request schema".into());
    }
    if request.max_per_trajectory == Some(0) {
        return Err("max_per_trajectory must be positive".into());
    }
    if let Some(window) = request.force_window {
        window.validate()?;
    }
    if !output.is_absolute() || output.exists() {
        return Err("collect-candidates requires a new absolute output directory".into());
    }
    let inputs = inputs(request)?;
    let named = |input: &Input, e: Box<dyn std::error::Error>| -> Box<dyn std::error::Error> {
        format!("{}: {e}", input.trajectory.display()).into()
    };
    let mut scans = Vec::with_capacity(inputs.len());
    for input in &inputs {
        scans.push(scan(input, request).map_err(|e| named(input, e))?);
    }
    for (input, found) in inputs.iter().zip(&scans).skip(1) {
        if found.type_map != scans[0].type_map || found.units != scans[0].units {
            return Err(format!(
                "{}: type map and units must equal those of the first trajectory",
                input.trajectory.display()
            )
            .into());
        }
    }
    let mut trajectories = Vec::with_capacity(inputs.len());
    for (input, found) in inputs.iter().zip(&scans) {
        let count = found.candidates.len() as u64;
        let picked = spread(count, request.max_per_trajectory)
            .into_iter()
            .map(|position| {
                let (step, largest) = found.candidates[position as usize];
                Picked {
                    id: format!("{}/step-{step}", input.id_prefix),
                    step,
                    max_force_deviation_ev_per_angstrom: largest,
                }
            })
            .collect();
        trajectories.push(TrajectorySummary {
            id_prefix: input.id_prefix.clone(),
            trajectory: input.trajectory.clone(),
            trajectory_sha256: found.sha256.clone(),
            model_sha256: found.model_sha256.clone(),
            ensemble: found.ensemble.clone(),
            force_window: found.window,
            frames: found.frames,
            accurate_frames: found.accurate,
            candidate_frames: count,
            failed_frames: found.failed,
            accurate_fraction: fraction(found.accurate, found.frames),
            max_force_deviation_ev_per_angstrom: found.maximum,
            picked,
        });
    }
    let total = |of: fn(&TrajectorySummary) -> u64| trajectories.iter().map(of).sum::<u64>();
    let frames = total(|t| t.frames);
    let accurate = total(|t| t.accurate_frames);
    let picked_frames = total(|t| t.picked.len() as u64);
    let mut summary = CandidatesSummary {
        schema: SUMMARY_SCHEMA,
        first_step: request.first_step,
        max_per_trajectory: request.max_per_trajectory,
        force_window: request.force_window,
        frames,
        accurate_frames: accurate,
        candidate_frames: total(|t| t.candidate_frames),
        failed_frames: total(|t| t.failed_frames),
        accurate_fraction: fraction(accurate, frames),
        mean_accurate_fraction: trajectories
            .iter()
            .map(|t| t.accurate_fraction)
            .sum::<f64>()
            / trajectories.len() as f64,
        picked_frames,
        dataset: None,
        dataset_sha256: None,
        trajectories,
    };
    fs::create_dir(output)?;
    if picked_frames > 0 {
        let path = output.join(DATASET);
        let mut writer = dataset::new_output(&path)?;
        dataset::write_line(
            &mut writer,
            &Record::Header {
                header: DatasetHeader {
                    parameter_layout: None,
                    schema: SCHEMA.into(),
                    type_map: scans[0].type_map.clone(),
                    units: scans[0].units,
                    virial_convention: VirialConvention::MinusDEpsilonRow,
                    source: format!("candidates of {} trajectories", inputs.len()),
                    selected_sets: vec![],
                },
            },
        )?;
        let mut index = 0;
        let mut sources = BTreeMap::new();
        for ((input, found), trajectory) in inputs.iter().zip(&scans).zip(&summary.trajectories) {
            let steps: HashSet<u64> = trajectory.picked.iter().map(|pick| pick.step).collect();
            if steps.is_empty() {
                continue;
            }
            write_picked(input, found, &steps, &mut writer, &mut index)
                .map_err(|e| named(input, e))?;
            sources.insert(format!("{}.jsonl", input.id_prefix), found.sha256.clone());
        }
        dataset::write_line(
            &mut writer,
            &Record::Completed {
                frames: index,
                source_sha256: sources,
            },
        )?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        summary.dataset = Some(DATASET);
        summary.dataset_sha256 = Some(dataset::sha256(&path)?);
    }
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.join(SUMMARY))?;
    serde_json::to_writer_pretty(&mut file, &summary)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::{scan, spread, write_picked, CandidatesRequest, Input, REQUEST_SCHEMA};
    use std::collections::HashSet;

    #[test]
    fn a_trajectory_that_changed_between_its_two_readings_is_refused() {
        let path = std::env::temp_dir().join(format!(
            "omnimd-candidates-readings-{}.jsonl",
            std::process::id()
        ));
        // One frame, at step 0, with this largest force deviation.
        let write = |deviation: f64| {
            let text = format!(
                concat!(
                    r#"{{"kind":"header","schema":"omnimd-mlmd-jsonl-v2","metadata":{{"type_map":["O","H"],"#,
                    r#""cutoff":3.0,"backend":"a test","precision":"float64","units":"EvAngstrom"}},"#,
                    r#""deviation":{{"force_window":{{"min_ev_per_angstrom":0.1,"max_ev_per_angstrom":0.3}}}}}}"#,
                    "\n",
                    r#"{{"kind":"frame","step":0,"frame":{{"positions":[[0.0,0.0,0.0],[0.96,0.0,0.0]],"#,
                    r#""species":["O","H"],"cell":null,"units":"EvAngstrom"}},"candidate":true,"#,
                    r#""model_deviation":{{"force_ev_per_angstrom":{{"max":{}}}}}}}"#,
                    "\n",
                    r#"{{"kind":"completed","final_step":0}}"#,
                    "\n"
                ),
                deviation
            );
            std::fs::write(&path, text).unwrap();
        };
        write(0.2);
        let input = Input {
            trajectory: path.clone(),
            id_prefix: "run".into(),
        };
        let request = CandidatesRequest {
            schema: REQUEST_SCHEMA.into(),
            trajectories: vec![],
            replica_runs: vec![],
            first_step: 0,
            max_per_trajectory: None,
            force_window: None,
        };
        let scanned = scan(&input, &request).unwrap();
        assert_eq!(scanned.candidates, [(0, 0.2)]);
        let picked = HashSet::from([0]);
        let (mut written, mut index) = (Vec::new(), 0);
        write_picked(&input, &scanned, &picked, &mut written, &mut index).unwrap();
        assert_eq!(index, 1);
        assert!(String::from_utf8(written)
            .unwrap()
            .contains(r#""id":"run/step-0""#));
        // The same step with another deviation is another file.
        write(0.25);
        let error = write_picked(&input, &scanned, &picked, &mut Vec::new(), &mut index);
        assert!(error
            .unwrap_err()
            .to_string()
            .contains("changed while it was read"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn picks_are_spread_from_the_first_candidate_to_the_last() {
        // Fewer candidates than the most, or no most: all of them.
        assert_eq!(spread(0, Some(3)), Vec::<u64>::new());
        assert_eq!(spread(3, Some(3)), [0, 1, 2]);
        assert_eq!(spread(2, Some(5)), [0, 1]);
        assert_eq!(spread(4, None), [0, 1, 2, 3]);
        // A single one is the first.
        assert_eq!(spread(1, Some(1)), [0]);
        assert_eq!(spread(9, Some(1)), [0]);
        // Whole positions.
        assert_eq!(spread(10, Some(4)), [0, 3, 6, 9]);
        assert_eq!(spread(3, Some(2)), [0, 2]);
        // Positions 0, 5/3, 10/3, 5: to the nearest.
        assert_eq!(spread(6, Some(4)), [0, 2, 3, 5]);
        // A half goes to the even position: 1.5 to 2, 2.5 to 2, 3.5 to 4.
        assert_eq!(spread(4, Some(3)), [0, 2, 3]);
        assert_eq!(spread(6, Some(3)), [0, 2, 5]);
        assert_eq!(spread(8, Some(3)), [0, 4, 7]);
        // As many picks as allowed, each one once, in order, ends included.
        for count in 2..60u64 {
            for most in 2..count {
                let picks = spread(count, Some(most));
                assert_eq!(picks.len() as u64, most, "{count} {most}");
                assert_eq!((picks[0], picks[picks.len() - 1]), (0, count - 1));
                assert!(
                    picks.windows(2).all(|pair| pair[0] < pair[1]),
                    "{count} {most}"
                );
                // No pick is further than half a step from its even place.
                for (k, pick) in picks.iter().enumerate() {
                    let exact = k as f64 * (count - 1) as f64 / (most - 1) as f64;
                    assert!((*pick as f64 - exact).abs() <= 0.5, "{count} {most} {k}");
                }
            }
        }
    }
}
