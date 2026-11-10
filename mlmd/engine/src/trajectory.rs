//! Frames of a finished MD trajectory as an unlabeled dataset, the input of
//! model deviation and selection. Model predictions stored in the trajectory
//! are not copied: they are not reference labels.
use crate::dataset::{
    self, DatasetFrame, DatasetHeader, Labels, Record, Result, VirialConvention, SCHEMA,
};
use omnimd_ml_contract::{Frame, ModelMetadata};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
};

pub const REQUEST_SCHEMA: &str = "omnimd-trajectory-import-v1";
const MAX_LINE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrajectoryRequest {
    pub schema: String,
    /// A trajectory written by `omnimd-mlmd run`, with its `completed` record.
    pub trajectory: PathBuf,
    /// Frames get the IDs `ID_PREFIX/step-N` with the absolute step N.
    pub id_prefix: String,
    /// Skip frames before this absolute step (for example the equilibration).
    #[serde(default)]
    pub first_step: u64,
    /// Keep every this many of the remaining sampled frames.
    #[serde(default = "one")]
    pub stride: u64,
}

fn one() -> u64 {
    1
}

/// The records of a trajectory, with the fields this import reads.
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Line {
    Header {
        schema: String,
        metadata: ModelMetadata,
    },
    Frame {
        step: u64,
        frame: Frame,
    },
    Completed {
        final_step: u64,
    },
}

/// Writes the selected frames as a new dataset and returns their number.
/// The whole trajectory is read and checked before the completed record of
/// the dataset is written; a failure leaves an incomplete output file that
/// dataset readers reject.
pub fn import_trajectory(request: &TrajectoryRequest, output: &Path) -> Result<u64> {
    if request.schema != REQUEST_SCHEMA {
        return Err("unsupported trajectory import request schema".into());
    }
    if !request.trajectory.is_absolute() {
        return Err("trajectory path must be absolute".into());
    }
    if request.stride == 0 {
        return Err("stride must be positive".into());
    }
    let prefix = request.id_prefix.as_str();
    if prefix.is_empty()
        || prefix.len() > 128
        || !prefix
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
    {
        return Err("id_prefix must be 1..=128 characters of [A-Za-z0-9_.-]".into());
    }
    let mut reader = BufReader::new(File::open(&request.trajectory)?);
    let mut hash = Sha256::new();
    let mut line = Vec::new();
    let mut next = |line: &mut Vec<u8>| -> Result<Option<Line>> {
        line.clear();
        let limit = MAX_LINE_BYTES as u64 + 1;
        (&mut reader).take(limit).read_until(b'\n', line)?;
        if line.is_empty() {
            return Ok(None);
        }
        if line.len() > MAX_LINE_BYTES || line.last() != Some(&b'\n') {
            return Err("trajectory record is unterminated or exceeds 16 MiB".into());
        }
        hash.update(&*line);
        Ok(Some(serde_json::from_slice(line)?))
    };
    let Some(Line::Header { schema, metadata }) = next(&mut line)? else {
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
    let mut writer = dataset::new_output(output)?;
    dataset::write_line(
        &mut writer,
        &Record::Header {
            header: DatasetHeader {
                parameter_layout: None,
                schema: SCHEMA.into(),
                type_map: metadata.type_map.clone(),
                units: metadata.units,
                virial_convention: VirialConvention::MinusDEpsilonRow,
                source: request.trajectory.display().to_string(),
                selected_sets: vec![],
            },
        },
    )?;
    let (mut written, mut eligible) = (0u64, 0u64);
    let mut last_step = None;
    let mut completed = None;
    while let Some(record) = next(&mut line)? {
        if completed.is_some() {
            return Err("trajectory continues after its completed record".into());
        }
        match record {
            Line::Header { .. } => return Err("trajectory has a second header".into()),
            Line::Completed { final_step } => completed = Some(final_step),
            Line::Frame { step, frame } => {
                if last_step.is_some_and(|last| step <= last) {
                    return Err("trajectory steps must increase".into());
                }
                last_step = Some(step);
                frame.type_ids(&metadata)?;
                if step < request.first_step {
                    continue;
                }
                eligible += 1;
                if (eligible - 1) % request.stride != 0 {
                    continue;
                }
                dataset::write_line(
                    &mut writer,
                    &Record::Frame {
                        data: DatasetFrame {
                            parameters: None,
                            index: written,
                            id: format!("{prefix}/step-{step}"),
                            frame,
                            labels: Labels::default(),
                        },
                    },
                )?;
                written += 1;
            }
        }
    }
    if completed.is_none() || completed != last_step {
        return Err(
            "trajectory has no completed record matching its last frame; a failed or interrupted run is not imported"
                .into(),
        );
    }
    if written == 0 {
        return Err("no trajectory frame at or after first_step".into());
    }
    dataset::write_line(
        &mut writer,
        &Record::Completed {
            frames: written,
            source_sha256: BTreeMap::from([(
                "trajectory.jsonl".into(),
                format!("{:x}", hash.finalize()),
            )]),
        },
    )?;
    writer.flush()?;
    Ok(written)
}
