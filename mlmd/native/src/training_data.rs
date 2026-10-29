//! Native reader for A's exact omnimd-dataset-jsonl-v1 wire contract.
//! Types stay here temporarily: native must not depend on engine.
use omnimd_ml_contract::{Error, Frame, ModelMetadata, Result, Units};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    fs::File,
    io::{BufRead, BufReader, Read},
    path::Path,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetHeader {
    pub schema: String,
    pub type_map: Vec<String>,
    pub units: Units,
    pub virial_convention: VirialConvention,
    pub source: String,
    pub selected_sets: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VirialConvention {
    MinusDEpsilonRow,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Labels {
    pub energy: Option<f64>,
    pub forces: Option<Vec<[f64; 3]>>,
    pub virial: Option<[[f64; 3]; 3]>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetFrame {
    pub index: u64,
    pub id: String,
    pub frame: Frame,
    pub labels: Labels,
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Record {
    Header {
        header: DatasetHeader,
    },
    Frame {
        data: DatasetFrame,
    },
    Completed {
        frames: u64,
        source_sha256: BTreeMap<String, String>,
    },
}
#[derive(Debug)]
pub struct Dataset {
    pub header: DatasetHeader,
    pub frames: Vec<DatasetFrame>,
    pub source_sha256: BTreeMap<String, String>,
    pub dataset_sha256: String,
}
fn record(reader: &mut impl BufRead, hash: &mut Sha256) -> Result<Option<Record>> {
    let mut line = Vec::new();
    reader
        .take(16 * 1024 * 1024 + 1)
        .read_until(b'\n', &mut line)
        .map_err(|e| Error(e.to_string()))?;
    if line.is_empty() {
        return Ok(None);
    }
    if line.len() > 16 * 1024 * 1024 || line.last() != Some(&b'\n') {
        return Err(Error(
            "dataset record exceeds 16 MiB or lacks terminating newline".into(),
        ));
    }
    hash.update(&line);
    serde_json::from_slice(&line)
        .map(Some)
        .map_err(|e| Error(format!("dataset record: {e}")))
}
impl Dataset {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path).map_err(|e| Error(format!("open dataset: {e}")))?;
        if file.metadata().map_err(|e| Error(e.to_string()))?.len() > 128 * 1024 * 1024 {
            return Err(Error(
                "small readout trainer dataset exceeds 128 MiB".into(),
            ));
        }
        let mut reader = BufReader::new(file);
        let mut hash = Sha256::new();
        let Some(Record::Header { header }) = record(&mut reader, &mut hash)? else {
            return Err(Error("dataset must begin with one header".into()));
        };
        if header.schema != "omnimd-dataset-jsonl-v1" {
            return Err(Error("unsupported dataset schema".into()));
        }
        let metadata = ModelMetadata {
            type_map: header.type_map.clone(),
            cutoff: 1.0,
            backend: "dataset".into(),
            precision: "float64".into(),
            units: header.units,
        };
        metadata.validate()?;
        let mut frames = Vec::new();
        let mut ids = HashSet::new();
        let source_sha256 = loop {
            match record(&mut reader, &mut hash)? {
                Some(Record::Frame { data }) => {
                    if data.index != frames.len() as u64
                        || data.id.is_empty()
                        || data.id.len() > 256
                        || !ids.insert(data.id.clone())
                        || frames.len() >= 10_000
                    {
                        return Err(Error("dataset requires consecutive indices, unique IDs and at most 10000 frames".into()));
                    }
                    data.frame.type_ids(&metadata)?;
                    validate_labels(&data)?;
                    frames.push(data);
                }
                Some(Record::Completed {
                    frames: count,
                    source_sha256,
                }) => {
                    if count == 0
                        || count != frames.len() as u64
                        || source_sha256
                            .values()
                            .any(|s| s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()))
                    {
                        return Err(Error(
                            "invalid completed frame count or source SHA256".into(),
                        ));
                    }
                    if record(&mut reader, &mut hash)?.is_some() {
                        return Err(Error("trailing record after completed".into()));
                    }
                    break source_sha256;
                }
                _ => {
                    return Err(Error(
                        "incomplete dataset or misplaced header: completed record required".into(),
                    ))
                }
            }
        };
        Ok(Self {
            header,
            frames,
            source_sha256,
            dataset_sha256: format!("{:x}", hash.finalize()),
        })
    }
}
pub(crate) fn validate_labels(data: &DatasetFrame) -> Result<()> {
    let n = data.frame.positions.len();
    match (&data.labels.energy, &data.labels.forces) {
        (Some(e), Some(f))
            if e.is_finite() && f.len() == n && f.iter().flatten().all(|x| x.is_finite()) => {}
        _ => {
            return Err(Error(format!(
                "{}: readout training requires finite energy and all force labels",
                data.id
            )))
        }
    }
    if data
        .labels
        .virial
        .as_ref()
        .is_some_and(|v| v.iter().flatten().any(|x| !x.is_finite()))
    {
        return Err(Error(format!("{}: nonfinite virial label", data.id)));
    }
    Ok(())
}

/// Storage access only; numerical ordering and dataset identity are unchanged.
/// A returned owned frame lives only for the current loop iteration.
pub trait DatasetAccess: private::Sealed {
    fn header(&self) -> &DatasetHeader;
    fn source_sha256(&self) -> &BTreeMap<String, String>;
    fn dataset_sha256(&self) -> &str;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn index(&self, index: usize) -> Result<u64>;
    fn id(&self, index: usize) -> Result<&str>;
    fn atoms(&self, index: usize) -> Result<usize>;
    fn frame(&self, index: usize) -> Result<std::borrow::Cow<'_, DatasetFrame>>;
    fn indexed(&self) -> bool {
        false
    }
    fn verify(&self) -> Result<()> {
        Ok(())
    }
    fn access_report(&self) -> Option<crate::indexed_dataset::AccessReport> {
        None
    }
}
pub(crate) mod private {
    pub trait Sealed {}
}
impl private::Sealed for Dataset {}
impl DatasetAccess for Dataset {
    fn header(&self) -> &DatasetHeader {
        &self.header
    }
    fn source_sha256(&self) -> &BTreeMap<String, String> {
        &self.source_sha256
    }
    fn dataset_sha256(&self) -> &str {
        &self.dataset_sha256
    }
    fn len(&self) -> usize {
        self.frames.len()
    }
    fn index(&self, index: usize) -> Result<u64> {
        self.frames
            .get(index)
            .map(|f| f.index)
            .ok_or_else(|| Error("dataset index out of range".into()))
    }
    fn id(&self, index: usize) -> Result<&str> {
        self.frames
            .get(index)
            .map(|f| f.id.as_str())
            .ok_or_else(|| Error("dataset index out of range".into()))
    }
    fn atoms(&self, index: usize) -> Result<usize> {
        self.frames
            .get(index)
            .map(|f| f.frame.positions.len())
            .ok_or_else(|| Error("dataset index out of range".into()))
    }
    fn frame(&self, index: usize) -> Result<std::borrow::Cow<'_, DatasetFrame>> {
        self.frames
            .get(index)
            .map(std::borrow::Cow::Borrowed)
            .ok_or_else(|| Error("dataset index out of range".into()))
    }
}
