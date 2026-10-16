//! Bounded, explicit dataset IO and evaluation statistics. No training or splitting.
use omnimd_ml_contract::{Evaluation, Frame, ModelMetadata, Units};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    error::Error,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Cursor, Read, Write},
    path::{Path, PathBuf},
};
pub type Result<T> = std::result::Result<T, Box<dyn Error>>;
pub const SCHEMA: &str = "omnimd-dataset-jsonl-v1";
const MAX_ARRAY_BYTES: u64 = 256 * 1024 * 1024;
const MAX_LINE_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VirialConvention {
    MinusDEpsilonRow,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Pbc {
    Periodic,
    NonPeriodic,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportRequest {
    pub system: PathBuf,
    pub sets: Vec<String>,
    pub units: Units,
    pub pbc: Pbc,
}
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
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Labels {
    pub energy: Option<f64>,
    pub forces: Option<Vec<[f64; 3]>>,
    pub virial: Option<[[f64; 3]; 3]>,
}
impl Labels {
    pub fn validate(&self, natoms: usize) -> Result<()> {
        if self.energy.is_some_and(|v| !v.is_finite())
            || self
                .forces
                .as_ref()
                .is_some_and(|v| v.len() != natoms || v.iter().flatten().any(|v| !v.is_finite()))
            || self
                .virial
                .as_ref()
                .is_some_and(|v| v.iter().flatten().any(|v| !v.is_finite()))
        {
            return Err("label shape mismatch or nonfinite label".into());
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetFrame {
    pub index: u64,
    pub id: String,
    pub frame: Frame,
    pub labels: Labels,
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Record {
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

fn type_metadata(type_map: &[String]) -> ModelMetadata {
    ModelMetadata {
        type_map: type_map.to_vec(),
        cutoff: 1.0,
        backend: "dataset-validation".into(),
        precision: "float64".into(),
        units: Units::EvAngstrom,
    }
}

pub fn new_output(path: &Path) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(
        OpenOptions::new().write(true).create_new(true).open(path)?,
    ))
}
pub fn write_line(writer: &mut impl Write, record: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec(record)?;
    if bytes.len() as u64 + 1 > MAX_LINE_BYTES {
        return Err("record exceeds dataset line limit".into());
    }
    writer.write_all(&bytes)?;
    writer.write_all(b"\n")?;
    Ok(())
}
pub fn sha256(path: &Path) -> Result<String> {
    let mut reader = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}
fn read_record(reader: &mut impl BufRead, hash: &mut Option<Sha256>) -> Result<Option<Record>> {
    let mut line = Vec::new();
    reader
        .take(MAX_LINE_BYTES + 1)
        .read_until(b'\n', &mut line)?;
    if line.is_empty() {
        return Ok(None);
    }
    if line.len() as u64 > MAX_LINE_BYTES {
        return Err("dataset record exceeds the 16 MiB line limit".into());
    }
    if line.last() != Some(&b'\n') {
        return Err("dataset record is missing its terminating newline".into());
    }
    if let Some(hash) = hash {
        hash.update(&line);
    }
    Ok(Some(serde_json::from_slice(&line)?))
}
pub struct DatasetReader {
    reader: BufReader<File>,
    pub header: DatasetHeader,
    metadata: ModelMetadata,
    next_index: u64,
    ids: HashSet<String>,
    completed: bool,
    hash: Option<Sha256>,
    source_sha256: BTreeMap<String, String>,
}
impl DatasetReader {
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_inner(path, false)
    }
    fn open_inner(path: &Path, hash_bytes: bool) -> Result<Self> {
        if !path.is_absolute() {
            return Err("dataset path must be absolute".into());
        }
        let mut reader = BufReader::new(File::open(path)?);
        let mut hash = hash_bytes.then(Sha256::new);
        let Some(Record::Header { header }) = read_record(&mut reader, &mut hash)? else {
            return Err("dataset must start with one header".into());
        };
        if header.schema != SCHEMA {
            return Err("unsupported dataset schema".into());
        }
        let metadata = type_metadata(&header.type_map);
        metadata.validate()?;
        Ok(Self {
            reader,
            header,
            metadata,
            next_index: 0,
            ids: HashSet::new(),
            completed: false,
            hash,
            source_sha256: BTreeMap::new(),
        })
    }
    /// Read until None to verify the terminal record and reject truncated/trailing data.
    pub fn next_frame(&mut self) -> Result<Option<DatasetFrame>> {
        if self.completed {
            return Ok(None);
        }
        match read_record(&mut self.reader, &mut self.hash)? {
            Some(Record::Frame { data }) => {
                if data.index != self.next_index
                    || data.id.is_empty()
                    || data.id.len() > 256
                    || !self.ids.insert(data.id.clone())
                {
                    return Err(
                        "dataset frame index must be consecutive and id unique/nonempty".into(),
                    );
                }
                data.frame.type_ids(&self.metadata)?;
                data.labels.validate(data.frame.positions.len())?;
                self.next_index = self
                    .next_index
                    .checked_add(1)
                    .ok_or("frame count overflow")?;
                Ok(Some(data))
            }
            Some(Record::Completed {
                frames,
                source_sha256,
            }) => {
                if frames == 0
                    || frames != self.next_index
                    || source_sha256
                        .values()
                        .any(|s| s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()))
                {
                    return Err("invalid completed count or source SHA256".into());
                }
                if read_record(&mut self.reader, &mut self.hash)?.is_some() {
                    return Err("completed must be the last dataset record".into());
                }
                self.source_sha256 = source_sha256;
                self.completed = true;
                Ok(None)
            }
            _ => Err("incomplete dataset or misplaced header; completed record required".into()),
        }
    }
}
/// Statistics for fully validated input. Source hashes are declarations, not reverified files.
#[derive(Debug, Serialize)]
pub struct DatasetInspection {
    pub schema: &'static str,
    pub dataset_sha256: String,
    pub header: DatasetHeader,
    pub declared_source_sha256: BTreeMap<String, String>,
    pub source_hashes_verified: bool,
    pub frames: u64,
    pub atom_occurrences: u64,
    pub min_atoms_per_frame: u64,
    pub max_atoms_per_frame: u64,
    pub species_atom_occurrences: BTreeMap<String, u64>,
    pub periodic_frames: u64,
    pub nonperiodic_frames: u64,
    pub labels: LabelCoverage,
    pub scope: &'static str,
}
#[derive(Debug, Default, Serialize)]
pub struct LabelCoverage {
    pub energy_frames: u64,
    pub force_frames: u64,
    pub virial_frames: u64,
    pub energy_and_force_frames: u64,
    pub all_frames_have_energy_and_forces: bool,
    pub missing_energy_or_force_frames: u64,
    pub missing_examples: Vec<MissingLabels>,
    pub missing_examples_truncated: bool,
}
#[derive(Debug, Serialize)]
pub struct MissingLabels {
    pub index: u64,
    pub id: String,
    pub missing_energy: bool,
    pub missing_forces: bool,
}
fn increment(count: &mut u64, amount: u64) -> Result<()> {
    *count = count
        .checked_add(amount)
        .ok_or("inspection count overflow")?;
    Ok(())
}
/// One file handle, one streaming pass. Hash original bytes including newlines.
/// Existing unique-ID validation retains IDs; frame contents are not accumulated.
pub fn inspect_dataset(path: &Path) -> Result<DatasetInspection> {
    let mut reader = DatasetReader::open_inner(path, true)?;
    let mut report = DatasetInspection {
        schema: "omnimd-dataset-inspection-v1",
        dataset_sha256: String::new(),
        header: reader.header.clone(),
        declared_source_sha256: BTreeMap::new(),
        source_hashes_verified: false,
        frames: 0,
        atom_occurrences: 0,
        min_atoms_per_frame: u64::MAX,
        max_atoms_per_frame: 0,
        species_atom_occurrences: BTreeMap::new(),
        periodic_frames: 0,
        nonperiodic_frames: 0,
        labels: LabelCoverage::default(),
        scope: "Structural validation and label coverage only; no model compatibility, trainer size/rank/split checks, source-file verification, or physical accuracy assessment.",
    };
    while let Some(data) = reader.next_frame()? {
        let atoms = u64::try_from(data.frame.positions.len())?;
        increment(&mut report.frames, 1)?;
        increment(&mut report.atom_occurrences, atoms)?;
        report.min_atoms_per_frame = report.min_atoms_per_frame.min(atoms);
        report.max_atoms_per_frame = report.max_atoms_per_frame.max(atoms);
        for species in &data.frame.species {
            increment(
                report
                    .species_atom_occurrences
                    .entry(species.clone())
                    .or_default(),
                1,
            )?;
        }
        if data.frame.cell.is_some() {
            increment(&mut report.periodic_frames, 1)?;
        } else {
            increment(&mut report.nonperiodic_frames, 1)?;
        }
        let labels = &mut report.labels;
        let energy = data.labels.energy.is_some();
        let forces = data.labels.forces.is_some();
        increment(&mut labels.energy_frames, u64::from(energy))?;
        increment(&mut labels.force_frames, u64::from(forces))?;
        increment(
            &mut labels.virial_frames,
            u64::from(data.labels.virial.is_some()),
        )?;
        increment(
            &mut labels.energy_and_force_frames,
            u64::from(energy && forces),
        )?;
        if !energy || !forces {
            increment(&mut labels.missing_energy_or_force_frames, 1)?;
            if labels.missing_examples.len() < 32 {
                labels.missing_examples.push(MissingLabels {
                    index: data.index,
                    id: data.id,
                    missing_energy: !energy,
                    missing_forces: !forces,
                });
            } else {
                labels.missing_examples_truncated = true;
            }
        }
    }
    // The reader rejects zero-frame, incomplete or trailing records before this point.
    report.labels.all_frames_have_energy_and_forces =
        report.labels.energy_and_force_frames == report.frames;
    report.dataset_sha256 = format!(
        "{:x}",
        reader
            .hash
            .take()
            .ok_or("inspection hash missing")?
            .finalize()
    );
    report.declared_source_sha256 = reader.source_sha256;
    Ok(report)
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrameReference {
    pub dataset: PathBuf,
    pub index: u64,
}
impl FrameReference {
    pub fn resolve(&self) -> Result<Frame> {
        let mut reader = DatasetReader::open(&self.dataset)?;
        let mut found = None;
        while let Some(data) = reader.next_frame()? {
            if data.index == self.index {
                found = Some(data.frame);
            }
        }
        found.ok_or_else(|| "dataset frame index out of range".into())
    }
}

fn source_bytes(
    root: &Path,
    relative: &str,
    hashes: &mut BTreeMap<String, String>,
) -> Result<Vec<u8>> {
    let path = root.join(relative).canonicalize()?;
    if !path.starts_with(root) || !path.is_file() {
        return Err(
            format!("source escapes the selected system or is not a file: {relative}").into(),
        );
    }
    let file = File::open(&path)?;
    if file.metadata()?.len() > MAX_ARRAY_BYTES {
        return Err(format!("source too large: {relative}").into());
    }
    let mut bytes = Vec::new();
    file.take(MAX_ARRAY_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_ARRAY_BYTES {
        return Err("source grew beyond size limit".into());
    }
    hashes.insert(relative.to_owned(), format!("{:x}", Sha256::digest(&bytes)));
    Ok(bytes)
}
struct Array {
    shape: Vec<u64>,
    values: Vec<f64>,
}
fn array(root: &Path, relative: &str, hashes: &mut BTreeMap<String, String>) -> Result<Array> {
    let bytes = source_bytes(root, relative, hashes)?;
    // Bound the Python-literal header before asking the NPY parser to read it.
    if bytes.len() < 12 || &bytes[..6] != b"\x93NUMPY" {
        return Err("invalid NPY magic/header".into());
    }
    let header_len = match (bytes[6], bytes[7]) {
        (1, 0) => u16::from_le_bytes([bytes[8], bytes[9]]) as usize,
        (2 | 3, 0) => u32::from_le_bytes(bytes[8..12].try_into()?) as usize,
        _ => return Err("unsupported NPY version".into()),
    };
    if header_len > 65536 {
        return Err("NPY header exceeds 64 KiB".into());
    }
    let offset = if bytes[6] == 1 { 10 } else { 12 };
    let header_bytes = bytes
        .get(offset..offset + header_len)
        .ok_or("truncated NPY header")?;
    // Standard scalar f64 headers need only a dict and shape tuple. Bound nesting
    // before invoking the generic Python-literal parser (which never executes code).
    let mut depth = 0usize;
    for byte in header_bytes {
        match byte {
            b'{' | b'[' | b'(' => {
                depth += 1;
                if depth > 4 {
                    return Err("unsupported NPY header nesting".into());
                }
            }
            b'}' | b']' | b')' => {
                depth = depth.checked_sub(1).ok_or("unbalanced NPY header")?;
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Err("unbalanced NPY header".into());
    }
    // npyz 0.9.1 multiplies dimensions while constructing its header. Preflight
    // the safe literal first, with checked sizes, so malformed shapes cannot
    // overflow inside that dependency before our payload validation.
    use py_literal::Value;
    let Value::Dict(fields) = std::str::from_utf8(header_bytes)?.trim().parse::<Value>()? else {
        return Err("NPY header must be a dictionary".into());
    };
    let mut seen = HashSet::new();
    let mut checked_shape = None;
    for (key, value) in fields {
        let Value::String(key) = key else {
            return Err("NPY keys must be strings".into());
        };
        if !seen.insert(key.clone()) {
            return Err("duplicate NPY header key".into());
        }
        match (key.as_str(), value) {
            ("descr", Value::String(dtype)) if matches!(dtype.as_str(), "<f8" | ">f8" | "=f8") => {}
            ("fortran_order", Value::Boolean(false)) => {}
            ("shape", Value::Tuple(dims)) if (1..=2).contains(&dims.len()) => {
                let mut shape = Vec::new();
                let mut elements = 1u64;
                for dim in dims {
                    let Value::Integer(dim) = dim else {
                        return Err("NPY dimensions must be integers".into());
                    };
                    let dim: u64 = dim.to_string().parse()?;
                    if dim == 0 {
                        return Err("NPY dimensions must be positive".into());
                    }
                    elements = elements.checked_mul(dim).ok_or("NPY size overflow")?;
                    shape.push(dim);
                }
                if elements > MAX_ARRAY_BYTES / 8 {
                    return Err("NPY dimensions exceed byte limit".into());
                }
                checked_shape = Some(shape);
            }
            _ => return Err("unsupported NPY header field, dtype, order or shape".into()),
        }
    }
    if seen.len() != 3 || checked_shape.is_none() {
        return Err("missing NPY header field".into());
    }
    let mut cursor = Cursor::new(&bytes);
    let header = npyz::NpyHeader::from_reader(&mut cursor)?;
    if header.order() != npyz::Order::C || header.uses_pickled_array() {
        return Err("only C-order float64 NPY arrays are supported; no objects/pickle".into());
    }
    let shape = header.shape().to_vec();
    if shape.is_empty() || shape.len() > 2 || shape.contains(&0) {
        return Err(format!("unsupported or empty NPY shape in {relative}").into());
    }
    let elements = shape
        .iter()
        .try_fold(1u64, |n, d| n.checked_mul(*d))
        .ok_or("NPY size overflow")?;
    let payload_bytes = elements.checked_mul(8).ok_or("NPY byte count overflow")?;
    if payload_bytes > MAX_ARRAY_BYTES
        || cursor.position().checked_add(payload_bytes) != Some(bytes.len() as u64)
    {
        return Err(
            format!("NPY must contain exactly shape*8 bytes of float64 data: {relative}").into(),
        );
    }
    let values = npyz::NpyFile::with_header(header, cursor).into_vec::<f64>()?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(format!("nonfinite NPY value in {relative}").into());
    }
    Ok(Array { shape, values })
}
fn matrix_rows(values: &[f64]) -> Vec<[f64; 3]> {
    values.chunks_exact(3).map(|v| [v[0], v[1], v[2]]).collect()
}
fn matrix3(values: &[f64]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| values[3 * i + j]))
}
fn check_shape(array: &Array, n: u64, width: u64, energy: bool) -> Result<()> {
    if array.shape != [n, width] && !(energy && array.shape == [n]) {
        return Err(format!(
            "invalid array shape {:?}, expected [{n},{width}]",
            array.shape
        )
        .into());
    }
    Ok(())
}
pub fn import_deepmd(request: &ImportRequest, output: &Path) -> Result<u64> {
    if !request.system.is_absolute() || !request.system.is_dir() || request.sets.is_empty() {
        return Err(
            "system must be an absolute directory and sets must be explicit/nonempty".into(),
        );
    }
    let root = request.system.canonicalize()?;
    let mut selected = HashSet::new();
    for set in &request.sets {
        if !set.starts_with("set.")
            || set.len() <= 4
            || !set[4..].bytes().all(|b| b.is_ascii_digit())
            || !selected.insert(set)
        {
            return Err("sets must be unique set.<digits> names, in desired input order".into());
        }
        if !root.join(set).canonicalize()?.starts_with(&root) {
            return Err("set escapes system".into());
        }
    }
    for entry in fs::read_dir(&root)? {
        let name = entry?.file_name().to_string_lossy().into_owned();
        if (name.ends_with(".raw") && !matches!(name.as_str(), "type.raw" | "type_map.raw"))
            || name.ends_with(".npy")
        {
            return Err(format!("unsupported root data field: {name}").into());
        }
    }
    let nonperiodic = root.join("nopbc").exists();
    if nonperiodic != (request.pbc == Pbc::NonPeriodic) {
        return Err("explicit PBC policy conflicts with presence/absence of nopbc".into());
    }
    let mut hashes = BTreeMap::new();
    if nonperiodic {
        source_bytes(&root, "nopbc", &mut hashes)?;
    }
    let type_map: Vec<String> =
        String::from_utf8(source_bytes(&root, "type_map.raw", &mut hashes)?)?
            .split_whitespace()
            .map(str::to_owned)
            .collect();
    let metadata = type_metadata(&type_map);
    metadata.validate()?;
    let types = String::from_utf8(source_bytes(&root, "type.raw", &mut hashes)?)?;
    let species: Vec<String> = types
        .split_whitespace()
        .map(|s| -> Result<String> {
            let i: usize = s.parse()?;
            Ok(type_map
                .get(i)
                .ok_or("type.raw index outside type_map")?
                .clone())
        })
        .collect::<Result<_>>()?;
    if species.is_empty() || species.len() > i32::MAX as usize / 9 {
        return Err("invalid atom count".into());
    }
    let width = species.len() as u64 * 3;
    let header = DatasetHeader {
        schema: SCHEMA.into(),
        type_map,
        units: request.units,
        virial_convention: VirialConvention::MinusDEpsilonRow,
        source: request.system.display().to_string(),
        selected_sets: request.sets.clone(),
    };
    let mut writer = new_output(output)?;
    write_line(&mut writer, &Record::Header { header })?;
    let mut index = 0u64;
    for set in &request.sets {
        for entry in fs::read_dir(root.join(set))? {
            let name = entry?.file_name().to_string_lossy().into_owned();
            if (name.ends_with(".npy")
                && !matches!(
                    name.as_str(),
                    "coord.npy" | "box.npy" | "energy.npy" | "force.npy" | "virial.npy"
                ))
                || name.ends_with(".raw")
            {
                return Err(format!("unsupported data field: {set}/{name}").into());
            }
        }
        let coords = array(&root, &format!("{set}/coord.npy"), &mut hashes)?;
        let n = coords.shape[0];
        check_shape(&coords, n, width, false)?;
        let mut optional = |name: &str, width: u64, energy: bool| -> Result<Option<Array>> {
            let relative = format!("{set}/{name}.npy");
            if !root.join(&relative).exists() {
                return Ok(None);
            }
            let value = array(&root, &relative, &mut hashes)?;
            check_shape(&value, n, width, energy)?;
            Ok(Some(value))
        };
        let boxes = optional("box", 9, false)?;
        if boxes.is_some() != (request.pbc == Pbc::Periodic) {
            return Err(
                "periodic data requires box.npy; nonperiodic data must omit box.npy".into(),
            );
        }
        let energies = optional("energy", 1, true)?;
        let forces = optional("force", width, false)?;
        let virials = optional("virial", 9, false)?;
        for i in 0..n as usize {
            let width = width as usize;
            let frame = Frame {
                positions: matrix_rows(&coords.values[i * width..(i + 1) * width]),
                species: species.clone(),
                cell: boxes
                    .as_ref()
                    .map(|a| matrix3(&a.values[i * 9..(i + 1) * 9])),
                units: request.units,
            };
            frame.type_ids(&metadata)?;
            let labels = Labels {
                energy: energies.as_ref().map(|a| a.values[i]),
                forces: forces
                    .as_ref()
                    .map(|a| matrix_rows(&a.values[i * width..(i + 1) * width])),
                virial: virials
                    .as_ref()
                    .map(|a| matrix3(&a.values[i * 9..(i + 1) * 9])),
            };
            labels.validate(species.len())?;
            let data = DatasetFrame {
                index,
                id: format!("{set}/{i}"),
                frame,
                labels,
            };
            // Keep the writer's record bound identical to the reader's.
            let bytes = serde_json::to_vec(&Record::Frame { data })?;
            if bytes.len() as u64 + 1 > MAX_LINE_BYTES {
                return Err("frame exceeds dataset line limit".into());
            }
            writer.write_all(&bytes)?;
            writer.write_all(b"\n")?;
            index = index.checked_add(1).ok_or("frame count overflow")?;
        }
        writer.flush()?;
    }
    write_line(
        &mut writer,
        &Record::Completed {
            frames: index,
            source_sha256: hashes,
        },
    )?;
    writer.flush()?;
    Ok(index)
}

#[derive(Default)]
pub struct Metric {
    count: u64,
    mean_absolute: f64,
    scale: f64,
    scaled_squares: f64,
}
impl Metric {
    pub fn add(&mut self, difference: f64) -> Result<()> {
        if !difference.is_finite() {
            return Err("nonfinite prediction-label difference".into());
        }
        let x = difference.abs();
        self.count = self.count.checked_add(1).ok_or("metric count overflow")?;
        self.mean_absolute += (x - self.mean_absolute) / self.count as f64;
        if x > self.scale {
            self.scaled_squares = self.scaled_squares * (self.scale / x).powi(2) + 1.0;
            self.scale = x;
        } else if self.scale != 0.0 {
            self.scaled_squares += (x / self.scale).powi(2);
        }
        Ok(())
    }
    pub fn report(&self, units: &str) -> serde_json::Value {
        serde_json::json!({
            "count": self.count, "units": units,
            "mae": (self.count > 0).then_some(self.mean_absolute),
            "rmse": (self.count > 0).then(|| self.scale * (self.scaled_squares / self.count as f64).sqrt()),
            "max_absolute": (self.count > 0).then_some(self.scale)
        })
    }
}
#[derive(Default)]
pub struct Metrics {
    frames: u64,
    energy_frames: u64,
    force_frames: u64,
    virial_frames: u64,
    energy: Metric,
    energy_per_atom: Metric,
    forces: Metric,
    virial: Metric,
}
impl Metrics {
    pub fn add(&mut self, prediction: &Evaluation, labels: &Labels, natoms: usize) -> Result<()> {
        prediction.validate(natoms)?;
        labels.validate(natoms)?;
        self.frames += 1;
        if let Some(value) = labels.energy {
            let difference = prediction.energy - value;
            self.energy.add(difference)?;
            self.energy_per_atom.add(difference / natoms as f64)?;
            self.energy_frames += 1;
        }
        if let Some(values) = &labels.forces {
            for (p, y) in prediction
                .forces
                .iter()
                .flatten()
                .zip(values.iter().flatten())
            {
                self.forces.add(p - y)?;
            }
            self.force_frames += 1;
        }
        if let Some(values) = &labels.virial {
            for (p, y) in prediction
                .virial
                .iter()
                .flatten()
                .zip(values.iter().flatten())
            {
                self.virial.add(p - y)?;
            }
            self.virial_frames += 1;
        }
        Ok(())
    }
    pub fn report(&self) -> serde_json::Value {
        serde_json::json!({
            "frames":self.frames,
            "labeled_frames":{"energy":self.energy_frames,"forces":self.force_frames,"virial":self.virial_frames},
            "energy_total":self.energy.report("eV"), "energy_per_atom":self.energy_per_atom.report("eV/atom"),
            "force_components":self.forces.report("eV/angstrom"), "virial_components":self.virial.report("eV")
        })
    }
}
