//! Content-preserving subset extraction bound to the exact immediate input bytes.
use crate::dataset::{self, DatasetReader, Record, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};
const MAX_REQUEST_BYTES: u64 = 4 * 1024 * 1024;
const MAX_SELECTION_BYTES: usize = 256 * 1024 * 1024;
const MAX_IDS: usize = 10_000;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: String,
    dataset: PathBuf,
    expected_source_sha256: String,
    ids: Vec<String>,
}
#[derive(Serialize)]
pub struct Mapping {
    pub id: String,
    pub source_index: u64,
    pub output_index: u64,
}
#[derive(Serialize)]
pub struct SelectionReport {
    pub schema: &'static str,
    pub status: &'static str,
    pub input_dataset: PathBuf,
    pub input_header: dataset::DatasetHeader,
    pub input_dataset_sha256: String,
    pub request_sha256: String,
    pub output_dataset_sha256: String,
    pub input_frames: u64,
    pub selected_frames: usize,
    pub requested_ids: Vec<String>,
    pub mapping: Vec<Mapping>,
    pub declared_upstream_source_sha256: BTreeMap<String, String>,
    pub upstream_source_hashes_verified: bool,
    pub ordering: &'static str,
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn save(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}
/// Validate the entire input before creating the output bundle. The final receipt is
/// its completion marker. An IO failure may leave an incomplete new bundle; never reuse it.
pub fn select_dataset(request_path: &Path, output: &Path) -> Result<SelectionReport> {
    if !output.is_absolute() || output.exists() {
        return Err("selection requires a new absolute output directory".into());
    }
    let mut request_bytes = Vec::new();
    File::open(request_path)?
        .take(MAX_REQUEST_BYTES + 1)
        .read_to_end(&mut request_bytes)?;
    if request_bytes.len() as u64 > MAX_REQUEST_BYTES {
        return Err("selection request exceeds 4 MiB".into());
    }
    let request: Request = serde_json::from_slice(&request_bytes)?;
    if request.schema != "omnimd-select-dataset-v1" {
        return Err("unsupported selection request schema".into());
    }
    if request.expected_source_sha256.len() != 64
        || !request
            .expected_source_sha256
            .bytes()
            .all(|b| b.is_ascii_hexdigit())
    {
        return Err("expected_source_sha256 must be a 64-digit hexadecimal SHA256".into());
    }
    if request.ids.is_empty() || request.ids.len() > MAX_IDS {
        return Err("selection requires 1..10000 explicit IDs".into());
    }
    let mut wanted = HashSet::new();
    for id in &request.ids {
        if id.is_empty() || id.len() > 256 || !wanted.insert(id.clone()) {
            return Err("selection IDs must be unique, nonempty and at most 256 bytes".into());
        }
    }
    let mut reader = DatasetReader::open_hashed(&request.dataset)?;
    let input_header = reader.header.clone();
    let mut output_header = input_header.clone();
    output_header.source = format!("selection from {}", request.dataset.display());
    let mut bytes = Vec::new();
    dataset::write_line(
        &mut bytes,
        &Record::Header {
            header: output_header,
        },
    )?;
    let mut mapping = Vec::new();
    let mut input_frames = 0_u64;
    while let Some(mut data) = reader.next_frame()? {
        input_frames = input_frames.checked_add(1).ok_or("frame count overflow")?;
        if wanted.remove(&data.id) {
            let output_index = mapping.len() as u64;
            mapping.push(Mapping {
                id: data.id.clone(),
                source_index: data.index,
                output_index,
            });
            data.index = output_index;
            dataset::write_line(&mut bytes, &Record::Frame { data })?;
            if bytes.len() > MAX_SELECTION_BYTES {
                return Err("selected dataset exceeds 256 MiB serialized bound".into());
            }
        }
    }
    let fingerprint = reader.completed_provenance()?;
    if !fingerprint
        .dataset_sha256
        .eq_ignore_ascii_case(&request.expected_source_sha256)
    {
        return Err("input dataset SHA256 does not match expected_source_sha256".into());
    }
    if !wanted.is_empty() {
        let first = request.ids.iter().find(|id| wanted.contains(*id)).unwrap();
        return Err(format!(
            "{} requested selection IDs were not found; first: {first:?}",
            wanted.len()
        )
        .into());
    }
    // Namespace all inherited declarations to avoid colliding with observed inputs.
    let mut sources: BTreeMap<_, _> = fingerprint
        .declared_source_sha256
        .iter()
        .map(|(key, value)| (format!("upstream:{key}"), value.clone()))
        .collect();
    let request_sha256 = hash(&request_bytes);
    sources.insert("input_dataset".into(), fingerprint.dataset_sha256.clone());
    sources.insert("selection_request".into(), request_sha256.clone());
    dataset::write_line(
        &mut bytes,
        &Record::Completed {
            frames: mapping.len() as u64,
            source_sha256: sources,
        },
    )?;
    if bytes.len() > MAX_SELECTION_BYTES {
        return Err("selected dataset exceeds 256 MiB serialized bound".into());
    }
    let report = SelectionReport {
        schema: "omnimd-selection-receipt-v1", status: "completed", input_dataset: request.dataset,
        input_header, input_dataset_sha256: fingerprint.dataset_sha256, request_sha256,
        output_dataset_sha256: hash(&bytes), input_frames, selected_frames: mapping.len(),
        requested_ids: request.ids, mapping, declared_upstream_source_sha256: fingerprint.declared_source_sha256,
        upstream_source_hashes_verified: false, ordering: "source order; requested IDs are membership only; source_index refers to immediate input",
    };
    let mut report_bytes = serde_json::to_vec_pretty(&report)?;
    report_bytes.push(b'\n');
    fs::create_dir(output)?;
    // Request copy + dataset first, receipt last. No atomic-directory or power-loss claim.
    let result = (|| -> Result<()> {
        save(&output.join("request.json"), &request_bytes)?;
        save(&output.join("dataset.jsonl"), &bytes)?;
        save(&output.join("receipt.json"), &report_bytes)?;
        Ok(())
    })();
    result.map_err(|e| format!("selection bundle write failed: {e}; treat {} as incomplete unless its receipt parses and all recorded file hashes match", output.display()))?;
    Ok(report)
}
