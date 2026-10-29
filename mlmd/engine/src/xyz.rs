//! Strict plain XYZ / common extended-XYZ profile. No inferred labels or units.
use crate::dataset::{
    self, DatasetFrame, DatasetHeader, Labels, Pbc, Record, Result, VirialConvention, SCHEMA,
};
use omnimd_ml_contract::{Frame, ModelMetadata, Units};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
};

#[derive(Debug, Deserialize, PartialEq, Eq)]
pub enum XyzFormat {
    Xyz,
    Extxyz,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct XyzRequest {
    pub input: PathBuf,
    pub format: XyzFormat,
    pub units: Units,
    pub pbc: Pbc,
    pub type_map: Vec<String>,
    pub cell: Option<[[f64; 3]; 3]>,
}
fn number(token: &str) -> Result<f64> {
    let value: f64 = token.trim().replace(['d', 'D'], "e").parse()?;
    if !value.is_finite() {
        return Err("XYZ contains a nonfinite number".into());
    }
    Ok(value)
}
fn numbers(text: &str, count: usize) -> Result<Vec<f64>> {
    let values = text
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|s| !s.is_empty())
        .map(number)
        .collect::<Result<Vec<_>>>()?;
    if values.len() != count {
        return Err(format!("expected {count} numeric values").into());
    }
    Ok(values)
}
fn fields(line: &str) -> Result<BTreeMap<String, String>> {
    let mut chars = line.chars().peekable();
    let mut result = BTreeMap::new();
    while chars.peek().is_some() {
        while chars.peek().is_some_and(|c| c.is_whitespace()) {
            chars.next();
        }
        if chars.peek().is_none() {
            break;
        }
        let mut key = String::new();
        while chars
            .peek()
            .is_some_and(|c| !c.is_whitespace() && *c != '=')
        {
            key.push(chars.next().unwrap());
        }
        if key.is_empty() || !key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err("unsupported extxyz key".into());
        }
        while chars.peek().is_some_and(|c| c.is_whitespace()) {
            chars.next();
        }
        if chars.next() != Some('=') {
            return Err("extxyz requires explicit key=value metadata".into());
        }
        while chars.peek().is_some_and(|c| c.is_whitespace()) {
            chars.next();
        }
        let first = chars.next().ok_or("missing extxyz value")?;
        let closing = match first {
            '"' => Some('"'),
            '\'' => Some('\''),
            '[' => Some(']'),
            '{' => Some('}'),
            _ => None,
        };
        let mut value = String::new();
        if let Some(closing) = closing {
            let mut closed = false;
            while let Some(c) = chars.next() {
                if c == closing {
                    closed = true;
                    break;
                }
                if c == '\\' {
                    value.push(chars.next().ok_or("unfinished escape")?);
                } else {
                    value.push(c);
                }
            }
            if !closed || chars.peek().is_some_and(|c| !c.is_whitespace()) {
                return Err("unterminated or unseparated extxyz value".into());
            }
        } else {
            value.push(first);
            while chars.peek().is_some_and(|c| !c.is_whitespace()) {
                value.push(chars.next().unwrap());
            }
        }
        if result.insert(key, value).is_some() {
            return Err("duplicate extxyz metadata key".into());
        }
    }
    Ok(result)
}
fn matrix(values: &[f64], column_major: bool) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| values[if column_major { i + 3 * j } else { 3 * i + j }])
    })
}
fn parse_pbc(text: &str) -> Result<Pbc> {
    let values = text
        .split_whitespace()
        .map(|s| match s {
            "T" | "true" | "True" | "TRUE" => Ok(true),
            "F" | "false" | "False" | "FALSE" => Ok(false),
            _ => Err("pbc must contain three standard logical values"),
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    match values.as_slice() {
        [true, true, true] => Ok(Pbc::Periodic),
        [false, false, false] => Ok(Pbc::NonPeriodic),
        _ => Err("partial PBC or malformed pbc is unsupported".into()),
    }
}
pub fn import_xyz(request: &XyzRequest, output: &Path) -> Result<u64> {
    if !request.input.is_absolute() {
        return Err("XYZ input path must be absolute".into());
    }
    if request.format == XyzFormat::Extxyz && request.cell.is_some() {
        return Err("extxyz cell comes from Lattice; cell override is forbidden".into());
    }
    if request.format == XyzFormat::Xyz && request.cell.is_some() != (request.pbc == Pbc::Periodic)
    {
        return Err("plain XYZ requires explicit cell only for Periodic".into());
    }
    let mut bytes = Vec::new();
    File::open(&request.input)?
        .take(256 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 256 * 1024 * 1024 {
        return Err("XYZ input exceeds 256 MiB".into());
    }
    let source_hash = format!("{:x}", Sha256::digest(&bytes));
    let text = std::str::from_utf8(&bytes)?;
    if text.lines().any(|s| s.len() > 1024 * 1024) {
        return Err("XYZ line exceeds 1 MiB".into());
    }
    let metadata = ModelMetadata {
        type_map: request.type_map.clone(),
        cutoff: 1.,
        backend: "xyz-validation".into(),
        precision: "float64".into(),
        units: request.units,
    };
    metadata.validate()?;
    let mut writer = dataset::new_output(output)?;
    dataset::write_line(
        &mut writer,
        &Record::Header {
            header: DatasetHeader {
                parameter_layout: None,
                schema: SCHEMA.into(),
                type_map: request.type_map.clone(),
                units: request.units,
                virial_convention: VirialConvention::MinusDEpsilonRow,
                source: request.input.display().to_string(),
                selected_sets: vec![],
            },
        },
    )?;
    let mut lines = text.lines().peekable();
    let mut index = 0u64;
    while let Some(line) = lines.next() {
        if line.trim().is_empty() {
            if lines.all(|line| line.trim().is_empty()) {
                break;
            }
            return Err("blank lines are permitted only after the final XYZ frame".into());
        }
        let natoms: usize = line.trim().parse()?;
        if natoms == 0 || natoms > 1_000_000 {
            return Err("XYZ atom count must be in 1..=1000000".into());
        }
        let comment = lines.next().ok_or("missing XYZ comment line")?;
        let mut cell = request.cell;
        let mut labels = Labels::default();
        let mut descriptors = vec![("species".to_owned(), 1usize), ("pos".to_owned(), 3usize)];
        if request.format == XyzFormat::Extxyz {
            let mut values = fields(comment)?;
            let properties = values
                .remove("Properties")
                .ok_or("extxyz requires Properties; no plain-XYZ fallback")?;
            let fields: Vec<_> = properties.split(':').collect();
            if fields.len() % 3 != 0 {
                return Err("malformed Properties triplets".into());
            }
            descriptors.clear();
            let mut seen = std::collections::HashSet::new();
            for triple in fields.chunks_exact(3) {
                let name = if triple[0] == "force" {
                    "forces"
                } else {
                    triple[0]
                };
                let width = match (name, triple[1], triple[2]) {
                    ("species", "S", "1") => 1,
                    ("pos", "R", "3") | ("forces", "R", "3") => 3,
                    _ => {
                        return Err(
                            format!("unsupported extxyz property: {}", triple.join(":")).into()
                        )
                    }
                };
                if !seen.insert(name) {
                    return Err("duplicate extxyz property".into());
                }
                descriptors.push((name.to_owned(), width));
            }
            if !seen.contains("species") || !seen.contains("pos") {
                return Err("species:S:1 and pos:R:3 are required".into());
            }
            if seen.contains("forces") {
                labels.forces = Some(Vec::with_capacity(natoms));
            }
            if let Some(pbc) = values.remove("pbc") {
                if parse_pbc(&pbc)? != request.pbc {
                    return Err("extxyz pbc conflicts with request".into());
                }
            }
            if let Some(lattice) = values.remove("Lattice") {
                cell = Some(matrix(&numbers(&lattice, 9)?, false)); // concatenated a,b,c vectors
            }
            if cell.is_some() != (request.pbc == Pbc::Periodic) {
                return Err(
                    "periodic extxyz requires Lattice; nonperiodic input must omit it".into(),
                );
            }
            if let Some(units) = values.remove("units") {
                if units != "EvAngstrom" {
                    return Err("extxyz units conflict with EvAngstrom request".into());
                }
            }
            if let Some(energy) = values.remove("energy") {
                labels.energy = Some(number(&energy)?);
            }
            if let Some(virial) = values.remove("virial") {
                labels.virial = Some(matrix(&numbers(&virial, 9)?, true)); // standard extxyz Fortran tensor order
            }
            // Explicitly nonphysical metadata only; original bytes remain fingerprinted.
            for key in [
                "config_type",
                "comment",
                "model_sha256",
                "minimization_status",
            ] {
                values.remove(key);
            }
            if !values.is_empty() {
                return Err(format!(
                    "unsupported extxyz metadata: {:?}",
                    values.keys().collect::<Vec<_>>()
                )
                .into());
            }
        } else if [
            "Properties",
            "Lattice",
            "pbc=",
            "energy=",
            "virial=",
            "stress=",
        ]
        .iter()
        .any(|key| comment.contains(key))
        {
            return Err("structured metadata found in plain XYZ; select Extxyz explicitly".into());
        }
        let columns: usize = descriptors.iter().map(|(_, n)| n).sum();
        let mut positions = Vec::with_capacity(natoms);
        let mut species = Vec::with_capacity(natoms);
        for _ in 0..natoms {
            let line = lines.next().ok_or("truncated XYZ atom records")?;
            let values: Vec<_> = line.split_whitespace().collect();
            if values.len() != columns {
                return Err("XYZ atom column count does not match Properties".into());
            }
            let mut offset = 0;
            for (name, width) in &descriptors {
                match name.as_str() {
                    "species" => species.push(values[offset].to_owned()),
                    "pos" => positions.push([
                        number(values[offset])?,
                        number(values[offset + 1])?,
                        number(values[offset + 2])?,
                    ]),
                    "forces" => labels.forces.as_mut().unwrap().push([
                        number(values[offset])?,
                        number(values[offset + 1])?,
                        number(values[offset + 2])?,
                    ]),
                    _ => unreachable!(),
                }
                offset += width;
            }
        }
        let frame = Frame {
            positions,
            species,
            cell,
            units: request.units,
        };
        frame.type_ids(&metadata)?;
        labels.validate(natoms)?;
        dataset::write_line(
            &mut writer,
            &Record::Frame {
                data: DatasetFrame {
                    parameters: None,
                    index,
                    id: format!("xyz/{index}"),
                    frame,
                    labels,
                },
            },
        )?;
        index += 1;
    }
    if index == 0 {
        return Err("XYZ contains no frames".into());
    }
    dataset::write_line(
        &mut writer,
        &Record::Completed {
            frames: index,
            source_sha256: BTreeMap::from([("input.xyz".into(), source_hash)]),
        },
    )?;
    writer.flush()?;
    Ok(index)
}

/// Export geometry only: predictions remain in the minimization report, never
/// silently relabeled as training references in this structure artifact.
pub fn export_extxyz(writer: &mut impl Write, frame: &Frame, status: &str) -> Result<()> {
    frame.validate()?;
    writeln!(writer, "{}", frame.positions.len())?;
    if let Some(cell) = frame.cell {
        write!(writer, "Lattice=\"")?;
        for (i, x) in cell.iter().flatten().enumerate() {
            if i != 0 {
                write!(writer, " ")?;
            }
            write!(writer, "{x:.17e}")?;
        }
        write!(writer, "\" ")?;
    }
    writeln!(
        writer,
        "Properties=species:S:1:pos:R:3 pbc=\"{}\" units=EvAngstrom minimization_status={status}",
        if frame.cell.is_some() {
            "T T T"
        } else {
            "F F F"
        }
    )?;
    for (symbol, pos) in frame.species.iter().zip(&frame.positions) {
        writeln!(
            writer,
            "{symbol} {:.17e} {:.17e} {:.17e}",
            pos[0], pos[1], pos[2]
        )?;
    }
    Ok(())
}
