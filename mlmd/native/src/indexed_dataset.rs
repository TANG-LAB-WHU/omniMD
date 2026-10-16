//! Bounded random access to the existing JSONL wire format; no on-disk sidecar.
use crate::training_data::{
    private, validate_labels, DatasetAccess, DatasetFrame, DatasetHeader, Labels,
};
use omnimd_ml_contract::{Error, Frame, ModelMetadata, Result, Units};
use serde::{
    de::{self, SeqAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};
use sha2::{Digest, Sha256};
use std::{
    borrow::Cow,
    cell::{Cell, RefCell},
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, Read, Seek, SeekFrom},
    mem::size_of,
    path::Path,
    time::{Instant, SystemTime},
};

pub const MAX_FRAMES: usize = 100_000;
pub const MAX_RECORD_BYTES: usize = 256 * 1024;
pub const INDEX_PAYLOAD_LIMIT: usize = 64 * 1024 * 1024;
pub const MAX_FILE_BYTES: u64 = 16 * 1024 * 1024 * 1024;
const IO_BUFFER: usize = 64 * 1024;
fn fail(s: impl Into<String>) -> Error {
    Error(s.into())
}

fn bounded_vec<'de, D, T, const N: usize>(d: D) -> std::result::Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct V<T, const N: usize>(std::marker::PhantomData<T>);
    impl<'de, T: Deserialize<'de>, const N: usize> Visitor<'de> for V<T, N> {
        type Value = Vec<T>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "at most {N} elements")
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<Vec<T>, A::Error> {
            let mut v = Vec::new();
            while let Some(x) = seq.next_element()? {
                if v.len() == N {
                    return Err(de::Error::custom("indexed record array bound exceeded"));
                }
                v.push(x);
            }
            Ok(v)
        }
    }
    d.deserialize_seq(V::<T, N>(std::marker::PhantomData))
}
fn bounded_string<'de, D: Deserializer<'de>, const N: usize>(
    d: D,
) -> std::result::Result<String, D::Error> {
    struct V<const N: usize>;
    impl<'de, const N: usize> Visitor<'de> for V<N> {
        type Value = String;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "string at most {N} bytes")
        }
        fn visit_str<E: de::Error>(self, s: &str) -> std::result::Result<String, E> {
            if s.len() > N {
                Err(E::custom("indexed string bound exceeded"))
            } else {
                Ok(s.to_owned())
            }
        }
    }
    d.deserialize_str(V::<N>)
}
#[derive(Deserialize)]
#[serde(remote = "Frame", deny_unknown_fields)]
struct BoundedFrame {
    #[serde(deserialize_with = "bounded_vec::<_,_,64>")]
    positions: Vec<[f64; 3]>,
    #[serde(deserialize_with = "bounded_vec::<_,_,64>")]
    species: Vec<String>,
    cell: Option<[[f64; 3]; 3]>,
    units: Units,
}
fn bounded_forces<'de, D: Deserializer<'de>>(
    d: D,
) -> std::result::Result<Option<Vec<[f64; 3]>>, D::Error> {
    #[derive(Deserialize)]
    struct Forces(#[serde(deserialize_with = "bounded_vec::<_,_,64>")] Vec<[f64; 3]>);
    Ok(Option::<Forces>::deserialize(d)?.map(|v| v.0))
}
#[derive(Deserialize)]
#[serde(remote = "Labels", deny_unknown_fields)]
struct BoundedLabels {
    energy: Option<f64>,
    #[serde(default, deserialize_with = "bounded_forces")]
    forces: Option<Vec<[f64; 3]>>,
    virial: Option<[[f64; 3]; 3]>,
}
#[derive(Deserialize)]
#[serde(remote = "DatasetFrame", deny_unknown_fields)]
struct BoundedData {
    index: u64,
    #[serde(deserialize_with = "bounded_string::<_,256>")]
    id: String,
    #[serde(with = "BoundedFrame")]
    frame: Frame,
    #[serde(with = "BoundedLabels")]
    labels: Labels,
}
#[derive(Deserialize)]
#[serde(remote = "DatasetHeader", deny_unknown_fields)]
struct BoundedHeader {
    schema: String,
    #[serde(deserialize_with = "bounded_vec::<_,_,118>")]
    type_map: Vec<String>,
    units: Units,
    virial_convention: crate::training_data::VirialConvention,
    #[serde(deserialize_with = "bounded_string::<_,65536>")]
    source: String,
    #[serde(deserialize_with = "bounded_vec::<_,_,4096>")]
    selected_sets: Vec<String>,
}
fn bounded_hashes<'de, D: Deserializer<'de>>(
    d: D,
) -> std::result::Result<BTreeMap<String, String>, D::Error> {
    struct V;
    impl<'de> Visitor<'de> for V {
        type Value = BTreeMap<String, String>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "at most 4096 provenance hashes")
        }
        fn visit_map<A: de::MapAccess<'de>>(
            self,
            mut a: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            let mut result = BTreeMap::new();
            let mut count = 0;
            while let Some((key, value)) = a.next_entry::<String, String>()? {
                count += 1;
                if count > 4096 || key.len() > 256 || value.len() != 64 {
                    return Err(de::Error::custom("indexed provenance map bound exceeded"));
                }
                result.insert(key, value);
            }
            Ok(result)
        }
    }
    d.deserialize_map(V)
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Record {
    Header {
        #[serde(with = "BoundedHeader")]
        header: DatasetHeader,
    },
    Frame {
        #[serde(with = "BoundedData")]
        data: DatasetFrame,
    },
    Completed {
        frames: u64,
        #[serde(deserialize_with = "bounded_hashes")]
        source_sha256: BTreeMap<String, String>,
    },
}
#[derive(Debug)]
struct Entry {
    offset: u64,
    len: u32,
    hash: [u8; 32],
    id: Box<str>,
    atoms: usize,
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct AccessReport {
    pub mode: String,
    pub frames: usize,
    pub file_bytes: u64,
    pub index_payload_bytes: usize,
    pub index_payload_limit_bytes: usize,
    pub parser_record_limit_bytes: usize,
    pub largest_record_bytes: usize,
    pub record_buffer_peak_capacity_bytes: usize,
    pub io_buffer_bytes: usize,
    pub duplicate_check_scratch_bytes: usize,
    pub peak_decoded_frame_payload_bytes: usize,
    pub frames_decoded_per_read: usize,
    pub frame_reads: u64,
    pub index_build_ms: f64,
    pub frame_read_ms: f64,
    pub integrity_verifications: u64,
    pub integrity_verification_ms: f64,
}
pub struct IndexedDataset {
    file: RefCell<File>,
    header: DatasetHeader,
    source_sha256: BTreeMap<String, String>,
    dataset_sha256: String,
    entries: Vec<Entry>,
    file_bytes: u64,
    modified: Option<SystemTime>,
    poisoned: Cell<bool>,
    report: RefCell<AccessReport>,
}
fn line(reader: &mut impl BufRead) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader
        .take((MAX_RECORD_BYTES + 1) as u64)
        .read_until(b'\n', &mut bytes)
        .map_err(|e| fail(format!("indexed read: {e}")))?;
    if !bytes.is_empty() && (bytes.len() > MAX_RECORD_BYTES || bytes.last() != Some(&b'\n')) {
        return Err(fail(
            "indexed record exceeds 256 KiB or lacks terminating newline",
        ));
    }
    Ok(bytes)
}
fn metadata_payload(h: &DatasetHeader, s: &BTreeMap<String, String>) -> usize {
    size_of::<DatasetHeader>() + h.schema.capacity()+h.type_map.capacity()*size_of::<String>()+h.type_map.iter().map(String::capacity).sum::<usize>()+h.source.capacity()+h.selected_sets.capacity()*size_of::<String>()+h.selected_sets.iter().map(String::capacity).sum::<usize>()
 // Conservative BTree node allowance per retained source entry; allocator overhead excluded.
 +s.iter().map(|(k,v)|k.capacity()+v.capacity()+128).sum::<usize>()
}
fn decoded_bytes(r: &DatasetFrame) -> usize {
    size_of::<DatasetFrame>()
        + r.id.capacity()
        + r.frame.positions.capacity() * size_of::<[f64; 3]>()
        + r.frame.species.capacity() * size_of::<String>()
        + r.frame.species.iter().map(String::capacity).sum::<usize>()
        + r.labels
            .forces
            .as_ref()
            .map_or(0, |f| f.capacity() * size_of::<[f64; 3]>())
}
fn enforce_index(capacity: usize, id_bytes: usize, metadata: usize) -> Result<usize> {
    let n = capacity
        .checked_mul(size_of::<Entry>())
        .and_then(|n| n.checked_add(id_bytes))
        .and_then(|n| n.checked_add(metadata))
        .and_then(|n| n.checked_add(size_of::<IndexedDataset>() + 64 + "indexed_jsonl_v1".len()))
        .ok_or_else(|| fail("indexed payload overflow"))?;
    if n > INDEX_PAYLOAD_LIMIT {
        return Err(fail("indexed payload exceeds 64 MiB"));
    }
    Ok(n)
}
impl IndexedDataset {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            options.share_mode(1);
        }
        let file = options
            .open(path)
            .map_err(|e| fail(format!("open immutable indexed dataset: {e}")))?;
        Self::from_file(file)
    }
    fn from_file(file: File) -> Result<Self> {
        let start = Instant::now();
        let meta = file.metadata().map_err(|e| fail(e.to_string()))?;
        if !meta.is_file() || meta.len() > MAX_FILE_BYTES {
            return Err(fail("indexed dataset requires a regular file <=16 GiB"));
        }
        let file_bytes = meta.len();
        let modified = meta.modified().ok();
        let mut reader = BufReader::with_capacity(IO_BUFFER, file);
        let mut digest = Sha256::new();
        let mut offset = 0u64;
        let first = line(&mut reader)?;
        let mut largest = first.len();
        let mut record_buffer_peak = first.capacity();
        digest.update(&first);
        offset += first.len() as u64;
        let Record::Header { header } =
            serde_json::from_slice(&first).map_err(|e| fail(format!("indexed header: {e}")))?
        else {
            return Err(fail("indexed dataset must begin with one header"));
        };
        drop(first);
        if header.schema != "omnimd-dataset-jsonl-v1" {
            return Err(fail("unsupported dataset schema"));
        }
        let metadata = ModelMetadata {
            type_map: header.type_map.clone(),
            cutoff: 1.,
            backend: "indexed dataset".into(),
            precision: "float64".into(),
            units: header.units,
        };
        metadata.validate()?;
        let mut entries = Vec::new();
        let mut id_bytes = 0usize;
        let mut peak_decoded = 0;
        let source_sha256 = loop {
            let bytes = line(&mut reader)?;
            if bytes.is_empty() {
                return Err(fail(
                    "incomplete indexed dataset: completed record required",
                ));
            }
            largest = largest.max(bytes.len());
            record_buffer_peak = record_buffer_peak.max(bytes.capacity());
            digest.update(&bytes);
            let here = offset;
            offset = offset
                .checked_add(bytes.len() as u64)
                .filter(|n| *n <= MAX_FILE_BYTES)
                .ok_or_else(|| fail("indexed offset/file limit exceeded"))?;
            let record: Record = serde_json::from_slice(&bytes)
                .map_err(|e| fail(format!("indexed record at {here}: {e}")))?;
            match record {
                Record::Frame { data } => {
                    if entries.len() >= MAX_FRAMES
                        || data.index != entries.len() as u64
                        || data.id.is_empty()
                    {
                        return Err(fail("indexed dataset requires consecutive indices, nonempty IDs and <=100000 frames"));
                    }
                    data.frame.type_ids(&metadata)?;
                    validate_labels(&data)?;
                    if data.frame.positions.len() > 64 {
                        return Err(fail("indexed training supports at most 64 atoms/frame"));
                    }
                    peak_decoded = peak_decoded.max(decoded_bytes(&data));
                    id_bytes += data.id.len();
                    if entries.len() == entries.capacity() {
                        let new = (entries.capacity().max(128) * 2).min(MAX_FRAMES);
                        enforce_index(new, id_bytes, metadata_payload(&header, &BTreeMap::new()))?;
                        entries.reserve_exact(new - entries.len());
                    }
                    enforce_index(
                        entries.capacity(),
                        id_bytes,
                        metadata_payload(&header, &BTreeMap::new()),
                    )?;
                    entries.push(Entry {
                        offset: here,
                        len: bytes.len() as u32,
                        hash: Sha256::digest(&bytes).into(),
                        atoms: data.frame.positions.len(),
                        id: data.id.into_boxed_str(),
                    });
                }
                Record::Completed {
                    frames,
                    source_sha256,
                } => {
                    if frames == 0
                        || frames != entries.len() as u64
                        || source_sha256
                            .values()
                            .any(|s| s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()))
                    {
                        return Err(fail("invalid completed count/source SHA256"));
                    }
                    if !line(&mut reader)?.is_empty() {
                        return Err(fail("trailing record after completed"));
                    }
                    break source_sha256;
                }
                _ => return Err(fail("misplaced indexed header")),
            }
        };
        if offset != file_bytes {
            return Err(fail("indexed dataset length changed while scanning"));
        }
        // Borrowed IDs only: no duplicate string hash table is retained or constructed.
        let mut ids: Vec<_> = entries.iter().map(|e| e.id.as_ref()).collect();
        let duplicate_scratch = ids.capacity() * size_of::<&str>();
        ids.sort_unstable();
        if ids.windows(2).any(|w| w[0] == w[1]) {
            return Err(fail("duplicate indexed frame ID"));
        }
        drop(ids);
        let payload = enforce_index(
            entries.capacity(),
            id_bytes,
            metadata_payload(&header, &source_sha256),
        )?;
        let file = reader.into_inner();
        let after = file.metadata().map_err(|e| fail(e.to_string()))?;
        if after.len() != file_bytes || after.modified().ok() != modified {
            return Err(fail("indexed dataset changed while scanning"));
        }
        let frames = entries.len();
        Ok(Self {
            file: RefCell::new(file),
            header,
            source_sha256,
            dataset_sha256: format!("{:x}", digest.finalize()),
            entries,
            file_bytes,
            modified,
            poisoned: Cell::new(false),
            report: RefCell::new(AccessReport {
                mode: "indexed_jsonl_v1".into(),
                frames,
                file_bytes,
                index_payload_bytes: payload,
                index_payload_limit_bytes: INDEX_PAYLOAD_LIMIT,
                parser_record_limit_bytes: MAX_RECORD_BYTES,
                largest_record_bytes: largest,
                record_buffer_peak_capacity_bytes: record_buffer_peak,
                io_buffer_bytes: IO_BUFFER,
                duplicate_check_scratch_bytes: duplicate_scratch,
                peak_decoded_frame_payload_bytes: peak_decoded,
                frames_decoded_per_read: 1,
                index_build_ms: start.elapsed().as_secs_f64() * 1000.,
                ..Default::default()
            }),
        })
    }
    fn unchanged(&self, file: &File) -> Result<()> {
        if self.poisoned.get() {
            return Err(fail("indexed dataset invalidated; reopen required"));
        }
        let meta = file.metadata().map_err(|e| fail(e.to_string()))?;
        if meta.len() != self.file_bytes || meta.modified().ok() != self.modified {
            return Err(fail("indexed dataset length/mtime changed"));
        }
        Ok(())
    }
    fn read(&self, index: usize) -> Result<DatasetFrame> {
        let start = Instant::now();
        let result = (|| {
            let entry = self
                .entries
                .get(index)
                .ok_or_else(|| fail("indexed frame index out of range"))?;
            let mut file = self.file.borrow_mut();
            self.unchanged(&file)?;
            file.seek(SeekFrom::Start(entry.offset))
                .map_err(|e| fail(e.to_string()))?;
            let mut bytes = vec![0; entry.len as usize];
            file.read_exact(&mut bytes)
                .map_err(|e| fail(format!("indexed frame truncated: {e}")))?;
            if <[u8; 32]>::from(Sha256::digest(&bytes)) != entry.hash {
                return Err(fail("indexed frame bytes changed"));
            }
            let Record::Frame { data } = serde_json::from_slice(&bytes)
                .map_err(|e| fail(format!("indexed frame parse: {e}")))?
            else {
                return Err(fail("indexed offset is not a frame"));
            };
            if data.index != index as u64
                || data.id.as_str() != entry.id.as_ref()
                || data.frame.positions.len() != entry.atoms
            {
                return Err(fail("indexed frame identity changed"));
            }
            self.unchanged(&file)?;
            Ok(data)
        })();
        let mut r = self.report.borrow_mut();
        r.frame_read_ms += start.elapsed().as_secs_f64() * 1000.;
        r.frame_reads += 1;
        if result.is_err() {
            self.poisoned.set(true);
        }
        result
    }
    pub fn verify_content(&self) -> Result<()> {
        let start = Instant::now();
        let result = (|| {
            let mut file = self.file.borrow_mut();
            self.unchanged(&file)?;
            file.seek(SeekFrom::Start(0))
                .map_err(|e| fail(e.to_string()))?;
            let mut hash = Sha256::new();
            let mut buf = [0u8; IO_BUFFER];
            let mut total = 0u64;
            loop {
                let count = file.read(&mut buf).map_err(|e| fail(e.to_string()))?;
                if count == 0 {
                    break;
                }
                total += count as u64;
                if total > self.file_bytes {
                    return Err(fail("indexed dataset grew"));
                }
                hash.update(&buf[..count]);
            }
            self.unchanged(&file)?;
            if total != self.file_bytes || format!("{:x}", hash.finalize()) != self.dataset_sha256 {
                return Err(fail("indexed dataset content changed before publication"));
            }
            Ok(())
        })();
        let mut r = self.report.borrow_mut();
        r.integrity_verifications += 1;
        r.integrity_verification_ms += start.elapsed().as_secs_f64() * 1000.;
        if result.is_err() {
            self.poisoned.set(true);
        }
        result
    }
}
impl private::Sealed for IndexedDataset {}
impl DatasetAccess for IndexedDataset {
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
        self.entries.len()
    }
    fn index(&self, i: usize) -> Result<u64> {
        self.entries
            .get(i)
            .map(|_| i as u64)
            .ok_or_else(|| fail("indexed index out of range"))
    }
    fn id(&self, i: usize) -> Result<&str> {
        self.entries
            .get(i)
            .map(|e| e.id.as_ref())
            .ok_or_else(|| fail("indexed ID out of range"))
    }
    fn atoms(&self, i: usize) -> Result<usize> {
        self.entries
            .get(i)
            .map(|e| e.atoms)
            .ok_or_else(|| fail("indexed atom count out of range"))
    }
    fn frame(&self, i: usize) -> Result<Cow<'_, DatasetFrame>> {
        self.read(i).map(Cow::Owned)
    }
    fn indexed(&self) -> bool {
        true
    }
    fn verify(&self) -> Result<()> {
        self.verify_content()
    }
    fn access_report(&self) -> Option<AccessReport> {
        Some(self.report.borrow().clone())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    static NEXT_TEMP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    struct Temp(std::path::PathBuf);
    impl Temp {
        fn new() -> Self {
            let p = std::env::temp_dir().join(format!(
                "omnimd-index-integrity-{}-{}-{}.jsonl",
                NEXT_TEMP.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                std::process::id(),
                SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            Self(p)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    fn fixture() -> Temp {
        let p = Temp::new();
        let frame: Frame = serde_json::from_str(include_str!("../fixtures/frame.jsonl")).unwrap();
        let header = DatasetHeader {
            schema: "omnimd-dataset-jsonl-v1".into(),
            type_map: vec!["O".into(), "H".into()],
            units: Units::EvAngstrom,
            virial_convention: crate::training_data::VirialConvention::MinusDEpsilonRow,
            source: "original integrity test".into(),
            selected_sets: vec![],
        };
        let mut f = File::create(&p.0).unwrap();
        writeln!(
            f,
            "{}",
            serde_json::json!({"kind":"header","header":header})
        )
        .unwrap();
        for i in 0..3 {
            let data = DatasetFrame {
                index: i,
                id: format!("frame/{i}"),
                frame: frame.clone(),
                labels: Labels {
                    energy: Some(0.),
                    forces: Some(vec![[0.; 3]; 4]),
                    virial: None,
                },
            };
            writeln!(f, "{}", serde_json::json!({"kind":"frame","data":data})).unwrap();
        }
        writeln!(
            f,
            "{}",
            serde_json::json!({"kind":"completed","frames":3,"source_sha256":{}})
        )
        .unwrap();
        p
    }
    fn change(p: &Path, offset: u64, modified: SystemTime) {
        let mut writer = OpenOptions::new().write(true).open(p).unwrap();
        writer.seek(SeekFrom::Start(offset)).unwrap();
        writer.write_all(b" ").unwrap();
        writer.flush().unwrap();
        writer
            .set_times(std::fs::FileTimes::new().set_modified(modified))
            .unwrap();
    }
    #[test]
    fn same_size_same_time_frame_and_unread_header_mutation_are_rejected() {
        // Use an unlocked handle only inside this test to exercise the portable hash
        // guards. Public Windows open additionally denies write/delete sharing.
        let p = fixture();
        let d = IndexedDataset::from_file(File::open(&p.0).unwrap()).unwrap();
        change(&p.0, d.entries[1].offset, d.modified.unwrap());
        assert_eq!(
            d.file.borrow().metadata().unwrap().modified().ok(),
            d.modified
        );
        assert!(d.frame(0).is_ok());
        assert!(d.frame(1).unwrap_err().0.contains("bytes changed"));
        assert!(d.frame(0).unwrap_err().0.contains("invalidated"));
        let q = fixture();
        let e = IndexedDataset::from_file(File::open(&q.0).unwrap()).unwrap();
        change(&q.0, 0, e.modified.unwrap());
        assert!(e.frame(0).is_ok());
        assert!(e
            .verify_content()
            .unwrap_err()
            .0
            .contains("content changed"));
    }
    #[test]
    fn truncation_append_and_index_budget_fail_closed() {
        for truncate in [true, false] {
            let p = fixture();
            let d = IndexedDataset::from_file(File::open(&p.0).unwrap()).unwrap();
            let f = OpenOptions::new().write(true).open(&p.0).unwrap();
            f.set_len(if truncate {
                d.file_bytes - 1
            } else {
                d.file_bytes + 1
            })
            .unwrap();
            drop(f);
            assert!(d.frame(0).is_err());
            assert!(d.verify_content().is_err());
        }
        assert!(enforce_index(usize::MAX, 0, 0).is_err());
        assert!(enforce_index(0, INDEX_PAYLOAD_LIMIT, 0).is_err());
        assert!(
            enforce_index(MAX_FRAMES, MAX_FRAMES * 256, 8 * 1024 * 1024).unwrap()
                < INDEX_PAYLOAD_LIMIT
        );
    }
    #[cfg(windows)]
    #[test]
    fn public_windows_handle_denies_mutation_and_replacement() {
        let p = fixture();
        let d = IndexedDataset::open(&p.0).unwrap();
        assert!(OpenOptions::new().write(true).open(&p.0).is_err());
        assert!(std::fs::rename(&p.0, p.0.with_extension("moved")).is_err());
        d.verify_content().unwrap();
    }
}
