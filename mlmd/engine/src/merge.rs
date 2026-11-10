//! Concatenation of datasets that share type map and units, so a model can
//! be trained on the frames of several iterations. Frames, labels and IDs
//! are copied unchanged; only the indices are renumbered.
use crate::dataset::{self, DatasetHeader, DatasetReader, Record, Result, SCHEMA};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, HashSet},
    io::Write,
    path::{Path, PathBuf},
};

pub const REQUEST_SCHEMA: &str = "omnimd-dataset-merge-v1";
const MAX_DATASETS: usize = 256;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MergeRequest {
    pub schema: String,
    /// Complete datasets, in the order their frames appear in the output.
    pub datasets: Vec<PathBuf>,
}

/// Writes the frames of every dataset, in order, as one new dataset and
/// returns their number. Every input is read and validated completely; its
/// SHA-256 is recorded as `dataset-K`. A failure leaves an incomplete output
/// file that dataset readers reject.
pub fn merge_datasets(request: &MergeRequest, output: &Path) -> Result<u64> {
    if request.schema != REQUEST_SCHEMA {
        return Err("unsupported dataset merge request schema".into());
    }
    if !(2..=MAX_DATASETS).contains(&request.datasets.len()) {
        return Err(format!("a merge takes 2..={MAX_DATASETS} datasets").into());
    }
    let unique: HashSet<_> = request.datasets.iter().collect();
    if unique.len() != request.datasets.len() {
        return Err("a dataset is listed twice".into());
    }
    // Every header is checked before anything is written.
    let mut readers = Vec::with_capacity(request.datasets.len());
    for path in &request.datasets {
        let reader =
            DatasetReader::open_hashed(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if reader.header.parameter_layout.is_some() {
            return Err(
                format!("{}: parameterized datasets are not merged", path.display()).into(),
            );
        }
        readers.push(reader);
    }
    let first = readers[0].header.clone();
    for (reader, path) in readers.iter().zip(&request.datasets).skip(1) {
        let header = &reader.header;
        if header.type_map != first.type_map
            || header.units != first.units
            || header.virial_convention != first.virial_convention
        {
            return Err(format!(
                "{}: type map, units and virial convention must equal those of the first dataset",
                path.display()
            )
            .into());
        }
    }
    let mut writer = dataset::new_output(output)?;
    dataset::write_line(
        &mut writer,
        &Record::Header {
            header: DatasetHeader {
                parameter_layout: None,
                schema: SCHEMA.into(),
                type_map: first.type_map,
                units: first.units,
                virial_convention: first.virial_convention,
                source: format!("merge of {} datasets", readers.len()),
                selected_sets: vec![],
            },
        },
    )?;
    let mut ids = HashSet::new();
    let mut sources = BTreeMap::new();
    let mut frames = 0u64;
    for (number, (mut reader, path)) in readers.into_iter().zip(&request.datasets).enumerate() {
        while let Some(mut data) = reader
            .next_frame()
            .map_err(|e| format!("{}: {e}", path.display()))?
        {
            if !ids.insert(data.id.clone()) {
                return Err(format!(
                    "{}: frame ID {:?} already occurs in an earlier dataset",
                    path.display(),
                    data.id
                )
                .into());
            }
            data.index = frames;
            dataset::write_line(&mut writer, &Record::Frame { data })?;
            frames = frames.checked_add(1).ok_or("frame count overflow")?;
        }
        sources.insert(
            format!("dataset-{number}"),
            reader.completed_provenance()?.dataset_sha256,
        );
    }
    dataset::write_line(
        &mut writer,
        &Record::Completed {
            frames,
            source_sha256: sources,
        },
    )?;
    writer.flush()?;
    Ok(frames)
}
