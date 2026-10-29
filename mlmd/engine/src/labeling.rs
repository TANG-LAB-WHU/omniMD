//! External label roundtrip. No calculator, model, GPU, unit conversion or atom remapping.
use crate::dataset::{
    self, DatasetFrame, DatasetHeader, DatasetReader, Labels, Record, Result, VirialConvention,
};
use crate::parameters::{FramePolicy, Layout, Values};
use omnimd_ml_contract::{Frame, Units};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};
const REQUEST_LIMIT: u64 = 4 * 1024 * 1024;
const DATA_LIMIT: u64 = 256 * 1024 * 1024;
const RETURN_LIMIT: u64 = 128 * 1024 * 1024;
const LINE_LIMIT: usize = 16 * 1024 * 1024;
const FRAME_LIMIT: usize = 10_000;
const IDENTITY_VERSION: &str = "omnimd-label-identity-v1";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PrepareRequest {
    schema: String,
    dataset: PathBuf,
    expected_source_sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImportRequest {
    schema: String,
    dataset: PathBuf,
    expected_source_sha256: String,
    labels: PathBuf,
    expected_labels_sha256: String,
}
#[derive(Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Context {
    type_map: Vec<String>,
    units: Units,
    virial_convention: VirialConvention,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parameter_layout: Option<Layout>,
}
impl From<&DatasetHeader> for Context {
    fn from(h: &DatasetHeader) -> Self {
        Self {
            type_map: h.type_map.clone(),
            units: h.units,
            virial_convention: h.virial_convention,
            parameter_layout: h.parameter_layout.clone(),
        }
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Identity {
    index: u64,
    id: String,
    identity_sha256: String,
    frame: Frame,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parameters: Option<Values>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    provider: String,
    method: String,
    #[serde(deserialize_with = "unique_hashes")]
    source_sha256: BTreeMap<String, String>,
}
fn unique_hashes<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<BTreeMap<String, String>, D::Error> {
    struct Visitor;
    impl<'de> serde::de::Visitor<'de> for Visitor {
        type Value = BTreeMap<String, String>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("unique declared source hashes")
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut a: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            let mut out = BTreeMap::new();
            while let Some((key, value)) = a.next_entry::<String, String>()? {
                if out.len() >= 1000
                    || key.is_empty()
                    || key.len() > 256
                    || out.insert(key, value).is_some()
                {
                    return Err(serde::de::Error::custom(
                        "duplicate/invalid provenance key or more than 1000 entries",
                    ));
                }
            }
            Ok(out)
        }
    }
    d.deserialize_map(Visitor)
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum ReturnRecord {
    Header {
        schema: String,
        identity_schema: String,
        source_dataset_sha256: String,
        context: Context,
        provenance: Provenance,
    },
    Frame {
        identity: Identity,
        labels: Labels,
    },
    Completed {
        frames: u64,
    },
}
#[derive(Serialize)]
pub struct Binding {
    pub index: u64,
    pub id: String,
    pub identity_sha256: String,
    pub added_fields: Vec<&'static str>,
}
#[derive(Serialize)]
pub struct Receipt {
    pub schema: &'static str,
    pub status: &'static str,
    pub frames: usize,
    pub input_dataset_sha256: String,
    pub request_sha256: String,
    pub output_file: &'static str,
    pub output_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label_return_sha256: Option<String>,
    pub bindings: Vec<Binding>,
    pub declared_upstream_source_sha256: BTreeMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub declared_calculator: Option<Provenance>,
    pub declared_sources_verified: bool,
    pub identity_schema: &'static str,
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn check_hash(s: &str) -> Result<()> {
    if s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("expected a 64-digit hexadecimal SHA256".into());
    }
    Ok(())
}
fn read_bounded(path: &Path, max: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)?.take(max + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max {
        return Err(format!("{} exceeds {max} byte limit", path.display()).into());
    }
    Ok(bytes)
}
fn new_bundle(output: &Path) -> Result<()> {
    if !output.is_absolute() || output.exists() {
        return Err("label workflow requires a new absolute output directory".into());
    }
    Ok(())
}
fn load_source(
    path: &Path,
    expected: &str,
) -> Result<(DatasetHeader, Vec<DatasetFrame>, dataset::DatasetProvenance)> {
    check_hash(expected)?;
    let mut reader = DatasetReader::open_hashed_bounded(path, DATA_LIMIT)?;
    let mut frames = Vec::new();
    let mut serialized = 0usize;
    while let Some(frame) = reader.next_frame()? {
        serialized = serialized
            .checked_add(serde_json::to_vec(&frame)?.len())
            .ok_or("size overflow")?;
        if frames.len() >= FRAME_LIMIT || serialized > DATA_LIMIT as usize {
            return Err("label workflow exceeds 10000 frames or 256 MiB normalized content".into());
        }
        frames.push(frame);
    }
    let provenance = reader.completed_provenance()?;
    if !provenance.dataset_sha256.eq_ignore_ascii_case(expected) {
        return Err("source dataset SHA256 mismatch".into());
    }
    Ok((reader.header, frames, provenance))
}
struct Hash(Sha256);
impl Hash {
    fn n(&mut self, v: u64) {
        self.0.update(v.to_le_bytes());
    }
    fn s(&mut self, v: &str) {
        self.n(v.len() as u64);
        self.0.update(v.as_bytes());
    }
    fn x(&mut self, v: f64) {
        self.0.update(v.to_bits().to_le_bytes());
    }
    fn vector(&mut self, v: &[f64]) {
        self.n(v.len() as u64);
        for x in v {
            self.x(*x);
        }
    }
}
// Versioned binary encoding; no JSON formatting, decimal serialization or float tolerance.
fn identity_hash(
    context: &Context,
    source: &str,
    index: u64,
    id: &str,
    frame: &Frame,
    parameters: Option<&Values>,
) -> Result<String> {
    frame.validate()?;
    if frame.units != context.units {
        return Err("frame and label context units differ".into());
    }
    if frame.species.iter().any(|s| !context.type_map.contains(s)) {
        return Err("unknown label frame species".into());
    }
    match (&context.parameter_layout, parameters) {
        (None, None) => {}
        (Some(l), Some(v)) => v.validate(l, frame.positions.len())?,
        _ => return Err("label parameter context omitted or introduced".into()),
    }
    let mut h = Hash(Sha256::new());
    h.s(IDENTITY_VERSION);
    h.s(&source.to_ascii_lowercase());
    h.n(index);
    h.s(id);
    h.s("EvAngstrom");
    h.s("MinusDEpsilonRow");
    h.n(context.type_map.len() as u64);
    for s in &context.type_map {
        h.s(s);
    }
    h.n(frame.positions.len() as u64);
    for (s, r) in frame.species.iter().zip(&frame.positions) {
        h.s(s);
        for v in r {
            h.x(*v);
        }
    }
    h.n(frame.cell.is_some() as u64);
    if let Some(cell) = frame.cell {
        for row in cell {
            for v in row {
                h.x(v);
            }
        }
    }
    h.n(context.parameter_layout.is_some() as u64);
    if let Some(layout) = &context.parameter_layout {
        for fields in [&layout.fparam, &layout.aparam] {
            h.n(fields.len() as u64);
            for f in fields {
                h.s(&f.name);
                h.s(&f.unit);
            }
        }
    }
    h.n(parameters.is_some() as u64);
    if let Some(p) = parameters {
        h.s(match p.frame_parameter_policy {
            FramePolicy::Explicit => "explicit",
            FramePolicy::ModelDefault => "model_default",
        });
        h.vector(&p.fparam);
        h.n(p.aparam.len() as u64);
        for row in &p.aparam {
            h.vector(row);
        }
    }
    Ok(format!("{:x}", h.0.finalize()))
}
fn bounded_line(bytes: &mut Vec<u8>, v: &impl Serialize) -> Result<()> {
    dataset::write_line(bytes, v)?;
    if bytes.len() > DATA_LIMIT as usize {
        return Err("label workflow output exceeds 256 MiB".into());
    }
    Ok(())
}
fn save(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}
fn save_bundle(
    output: &Path,
    request: &[u8],
    payload: &[u8],
    returned: Option<&[u8]>,
    receipt: &Receipt,
) -> Result<()> {
    let mut receipt_bytes = serde_json::to_vec_pretty(receipt)?;
    receipt_bytes.push(b'\n');
    fs::create_dir(output)?;
    let result = (|| -> Result<()> {
        save(&output.join("request.json"), request)?;
        if let Some(bytes) = returned {
            save(&output.join("labels.jsonl"), bytes)?;
        }
        save(&output.join(receipt.output_file), payload)?;
        save(&output.join("receipt.json"), &receipt_bytes)?;
        Ok(())
    })();
    result.map_err(|e|format!("label bundle write failed: {e}; preserve the incomplete directory; receipt and hashes are required for reuse").into())
}
pub fn prepare_labels(request_path: &Path, output: &Path) -> Result<Receipt> {
    new_bundle(output)?;
    let request_bytes = read_bounded(request_path, REQUEST_LIMIT)?;
    let request: PrepareRequest = serde_json::from_slice(&request_bytes)?;
    if request.schema != "omnimd-prepare-labels-v1" {
        return Err("unsupported prepare-labels schema".into());
    }
    let (header, frames, source) = load_source(&request.dataset, &request.expected_source_sha256)?;
    let context = Context::from(&header);
    let mut bytes = Vec::new();
    let mut bindings = Vec::new();
    bounded_line(
        &mut bytes,
        &serde_json::json!({"kind":"header","schema":"omnimd-label-job-v1","identity_schema":IDENTITY_VERSION,"source_dataset_sha256":source.dataset_sha256,"context":context}),
    )?;
    for data in frames {
        let hash = identity_hash(
            &context,
            &source.dataset_sha256,
            data.index,
            &data.id,
            &data.frame,
            data.parameters.as_ref(),
        )?;
        bindings.push(Binding {
            index: data.index,
            id: data.id.clone(),
            identity_sha256: hash.clone(),
            added_fields: vec![],
        });
        let identity = Identity {
            index: data.index,
            id: data.id,
            identity_sha256: hash,
            frame: data.frame,
            parameters: data.parameters,
        };
        bounded_line(
            &mut bytes,
            &serde_json::json!({"kind":"frame","identity":identity,"existing_labels":data.labels}),
        )?;
    }
    bounded_line(
        &mut bytes,
        &serde_json::json!({"kind":"completed","frames":bindings.len()}),
    )?;
    let receipt = Receipt {
        schema: "omnimd-label-job-receipt-v1",
        status: "completed",
        frames: bindings.len(),
        input_dataset_sha256: source.dataset_sha256,
        request_sha256: digest(&request_bytes),
        output_file: "job.jsonl",
        output_sha256: digest(&bytes),
        label_return_sha256: None,
        bindings,
        declared_upstream_source_sha256: source.declared_source_sha256,
        declared_calculator: None,
        declared_sources_verified: false,
        identity_schema: IDENTITY_VERSION,
    };
    save_bundle(output, &request_bytes, &bytes, None, &receipt)?;
    Ok(receipt)
}
fn same_numbers(a: impl Iterator<Item = f64>, b: impl Iterator<Item = f64>) -> bool {
    a.map(f64::to_bits).eq(b.map(f64::to_bits))
}
fn merge_labels(
    current: &mut Labels,
    returned: Labels,
    natoms: usize,
) -> Result<Vec<&'static str>> {
    returned.validate(natoms)?;
    if returned.energy.is_none() || returned.forces.is_none() {
        return Err("external label return requires both energy and forces".into());
    }
    let mut added = Vec::new();
    if let Some(old) = current.energy {
        if old.to_bits() != returned.energy.unwrap().to_bits() {
            return Err("conflicting existing energy label".into());
        }
    } else {
        current.energy = returned.energy;
        added.push("energy");
    }
    if let Some(old) = &current.forces {
        if !same_numbers(
            old.iter().flatten().copied(),
            returned.forces.as_ref().unwrap().iter().flatten().copied(),
        ) {
            return Err("conflicting existing force label".into());
        }
    } else {
        current.forces = returned.forces;
        added.push("forces");
    }
    if let Some(new) = returned.virial {
        if let Some(old) = current.virial {
            if !same_numbers(old.into_iter().flatten(), new.into_iter().flatten()) {
                return Err("conflicting existing virial label".into());
            }
        } else {
            current.virial = Some(new);
            added.push("virial");
        }
    }
    Ok(added)
}
pub fn import_labels(request_path: &Path, output: &Path) -> Result<Receipt> {
    new_bundle(output)?;
    let request_bytes = read_bounded(request_path, REQUEST_LIMIT)?;
    let request: ImportRequest = serde_json::from_slice(&request_bytes)?;
    if request.schema != "omnimd-import-labels-v1" || !request.labels.is_absolute() {
        return Err("unsupported import-labels schema or nonabsolute labels path".into());
    }
    check_hash(&request.expected_labels_sha256)?;
    let returned_bytes = read_bounded(&request.labels, RETURN_LIMIT)?;
    let return_hash = digest(&returned_bytes);
    if !return_hash.eq_ignore_ascii_case(&request.expected_labels_sha256) {
        return Err("label-return SHA256 mismatch".into());
    }
    let (header, mut frames, source) =
        load_source(&request.dataset, &request.expected_source_sha256)?;
    let context = Context::from(&header);
    let lookup: HashMap<_, _> = frames
        .iter()
        .enumerate()
        .map(|(i, d)| (d.id.clone(), i))
        .collect();
    let mut lines = returned_bytes.split_inclusive(|b| *b == b'\n');
    let mut next = || -> Result<Option<ReturnRecord>> {
        lines
            .next()
            .map(|line| {
                if line.len() > LINE_LIMIT || line.last() != Some(&b'\n') {
                    return Err(
                        "label return needs newline-terminated records at most 16 MiB".into(),
                    );
                }
                Ok(serde_json::from_slice(line)?)
            })
            .transpose()
    };
    let Some(ReturnRecord::Header {
        schema,
        identity_schema,
        source_dataset_sha256,
        context: echoed,
        provenance,
    }) = next()?
    else {
        return Err("label return must start with one header".into());
    };
    if schema != "omnimd-label-return-v1"
        || identity_schema != IDENTITY_VERSION
        || !source_dataset_sha256.eq_ignore_ascii_case(&source.dataset_sha256)
        || echoed != context
    {
        return Err("label-return schema/source/context mismatch".into());
    }
    if provenance.provider.trim().is_empty()
        || provenance.method.trim().is_empty()
        || provenance.provider.len() > 1024
        || provenance.method.len() > 4096
    {
        return Err(
            "calculator provider/method must be explicit bounded nonempty declarations".into(),
        );
    }
    for value in provenance.source_sha256.values() {
        check_hash(value)?;
    }
    let mut seen = HashSet::new();
    let mut bindings = BTreeMap::new();
    loop {
        match next()? {
            Some(ReturnRecord::Frame { identity, labels }) => {
                if !seen.insert(identity.id.clone()) {
                    return Err("duplicate returned frame ID".into());
                }
                let index = *lookup
                    .get(&identity.id)
                    .ok_or("unknown returned frame ID")?;
                let data = &mut frames[index];
                if data.index != identity.index {
                    return Err("returned index does not match source ID".into());
                }
                let expected = identity_hash(
                    &context,
                    &source.dataset_sha256,
                    data.index,
                    &data.id,
                    &data.frame,
                    data.parameters.as_ref(),
                )?;
                let actual = identity_hash(
                    &context,
                    &source.dataset_sha256,
                    identity.index,
                    &identity.id,
                    &identity.frame,
                    identity.parameters.as_ref(),
                )?;
                if !identity.identity_sha256.eq_ignore_ascii_case(&expected) || actual != expected {
                    return Err("returned coordinate/atom/parameter identity mismatch".into());
                }
                let added = merge_labels(&mut data.labels, labels, data.frame.positions.len())
                    .map_err(|e| format!("frame {:?}: {e}", data.id))?;
                bindings.insert(
                    index,
                    Binding {
                        index: data.index,
                        id: data.id.clone(),
                        identity_sha256: expected,
                        added_fields: added,
                    },
                );
            }
            Some(ReturnRecord::Completed { frames: count }) => {
                if count != frames.len() as u64 || seen.len() != frames.len() {
                    return Err("missing returned frames or completed count mismatch".into());
                }
                if next()?.is_some() {
                    return Err("label completed record must be last".into());
                }
                break;
            }
            _ => return Err("incomplete label return or misplaced header".into()),
        }
    }
    let request_hash = digest(&request_bytes);
    let mut declared: BTreeMap<_, _> = source
        .declared_source_sha256
        .iter()
        .map(|(k, v)| (format!("upstream:{k}"), v.clone()))
        .collect();
    for (k, v) in &provenance.source_sha256 {
        declared.insert(format!("calculator:{k}"), v.clone());
    }
    declared.insert("input_dataset".into(), source.dataset_sha256.clone());
    declared.insert("label_return".into(), return_hash.clone());
    declared.insert("label_import_request".into(), request_hash.clone());
    let mut bytes = Vec::new();
    bounded_line(&mut bytes, &Record::Header { header })?;
    for data in frames {
        bounded_line(&mut bytes, &Record::Frame { data })?;
    }
    bounded_line(
        &mut bytes,
        &Record::Completed {
            frames: bindings.len() as u64,
            source_sha256: declared,
        },
    )?;
    let receipt = Receipt {
        schema: "omnimd-label-import-receipt-v1",
        status: "completed",
        frames: bindings.len(),
        input_dataset_sha256: source.dataset_sha256,
        request_sha256: request_hash,
        output_file: "dataset.jsonl",
        output_sha256: digest(&bytes),
        label_return_sha256: Some(return_hash),
        bindings: bindings.into_values().collect(),
        declared_upstream_source_sha256: source.declared_source_sha256,
        declared_calculator: Some(provenance),
        declared_sources_verified: false,
        identity_schema: IDENTITY_VERSION,
    };
    save_bundle(
        output,
        &request_bytes,
        &bytes,
        Some(&returned_bytes),
        &receipt,
    )?;
    Ok(receipt)
}
