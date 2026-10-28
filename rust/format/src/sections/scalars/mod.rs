//! chunk_scalars (0x0D): named, typed columns parallel to chunk order, the
//! fields a corpus carries per chunk beyond its text (a card's power, a
//! paper's year, a diagnosis code). OPTIONAL and EXCLUDED from
//! content_hash, so adding columns never moves a urna:// citation. the
//! manifest's `schema` says what each column means; this module owns only
//! the bytes.
//!
//! wire encoding `raw`, every integer le:
//!
//! ```text
//! u32 version=1 · u64 n_rows · u64 n_columns
//! per column: u32 name_len · name · u8 kind · u8 has_validity
//!             [validity bitset, ceil(n_rows/8) bytes, bit set = present]
//!             u32 body_len · body
//! body by kind:
//!   int       intpack of zigzag(i64), one per row (absent rows 0)
//!   real      f64 per row (absent rows 0.0)
//!   bool      bitset, ceil(n_rows/8) bytes
//!   text      intpack offsets (n_rows + 1) · utf-8 bytes
//!   uuid      16 bytes per row (absent rows zero)
//!   enum      dictionary · intpack codes, one per row (absent rows 0,
//!             so a column with no value has an empty dictionary)
//!   enum_list dictionary · intpack offsets (n_rows + 1) · intpack codes
//! dictionary: u32 n · (u32 len · utf-8) × n · u8 has_numeric ·
//!             [f64 × n, NaN = no value]
//! intpack blobs inside a body are prefixed by their u32 byte length.
//! ```
//!
//! the per-128-block frame-of-reference of `intpack` makes a constant run
//! cost zero bits per value, which is the run-length behaviour a column of
//! repeated codes needs. a map of enum to enum (a card's legality per
//! format) is one enum column per key, `legal.pauper`, by the writer.

mod decode;
pub mod from_json;

use crate::encoding::pack_u64s;
use crate::error::UrnaError;
use crate::layout::SECTION_CHUNK_SCALARS;

pub use decode::decode_chunk_scalars;

pub const CHUNK_SCALARS_PAYLOAD_VERSION: u32 = 1;

/// the column kinds on the wire, by their kind byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ColumnKind {
    Int = 0,
    Real = 1,
    Bool = 2,
    Text = 3,
    Uuid = 4,
    Enum = 5,
    EnumList = 6,
}

impl ColumnKind {
    pub fn from_byte(b: u8) -> Option<Self> {
        Some(match b {
            0 => Self::Int,
            1 => Self::Real,
            2 => Self::Bool,
            3 => Self::Text,
            4 => Self::Uuid,
            5 => Self::Enum,
            6 => Self::EnumList,
            _ => return None,
        })
    }

    /// the name the manifest schema uses for the kind.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Int => "int",
            Self::Real => "real",
            Self::Bool => "bool",
            Self::Text => "text",
            Self::Uuid => "uuid",
            Self::Enum => "enum",
            Self::EnumList => "enum_list",
        }
    }
}

/// an enum column's value table. `numeric` gives a value a number when it
/// has one (a card's power "3" is 3, "*" has none), so a range predicate
/// can run over an enum column.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Dictionary {
    pub values: Vec<String>,
    pub numeric: Option<Vec<Option<f64>>>,
}

impl Dictionary {
    pub fn code_of(&self, value: &str) -> Option<u32> {
        self.values
            .iter()
            .position(|v| v == value)
            .map(|i| i as u32)
    }

    pub fn numeric_of(&self, code: u32) -> Option<f64> {
        self.numeric.as_ref()?.get(code as usize).copied().flatten()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ColumnData {
    Int(Vec<i64>),
    Real(Vec<f64>),
    Bool(Vec<bool>),
    Text(Vec<String>),
    Uuid(Vec<[u8; 16]>),
    Enum {
        dict: Dictionary,
        codes: Vec<u32>,
    },
    EnumList {
        dict: Dictionary,
        /// `n_rows + 1` offsets into `codes`.
        offsets: Vec<u32>,
        codes: Vec<u32>,
    },
}

impl ColumnData {
    pub fn kind(&self) -> ColumnKind {
        match self {
            Self::Int(_) => ColumnKind::Int,
            Self::Real(_) => ColumnKind::Real,
            Self::Bool(_) => ColumnKind::Bool,
            Self::Text(_) => ColumnKind::Text,
            Self::Uuid(_) => ColumnKind::Uuid,
            Self::Enum { .. } => ColumnKind::Enum,
            Self::EnumList { .. } => ColumnKind::EnumList,
        }
    }

    /// rows the data holds (offsets minus one for a list column).
    pub fn len(&self) -> usize {
        match self {
            Self::Int(v) => v.len(),
            Self::Real(v) => v.len(),
            Self::Bool(v) => v.len(),
            Self::Text(v) => v.len(),
            Self::Uuid(v) => v.len(),
            Self::Enum { codes, .. } => codes.len(),
            Self::EnumList { offsets, .. } => offsets.len().saturating_sub(1),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dictionary(&self) -> Option<&Dictionary> {
        match self {
            Self::Enum { dict, .. } | Self::EnumList { dict, .. } => Some(dict),
            _ => None,
        }
    }

    /// the enum codes of `row` (one for an enum column, the list's for a
    /// list column), empty for the other kinds or out of range.
    pub fn codes_at(&self, row: usize) -> &[u32] {
        match self {
            Self::Enum { codes, .. } => codes.get(row..row + 1).unwrap_or(&[]),
            Self::EnumList { offsets, codes, .. } => {
                let (Some(&a), Some(&b)) = (offsets.get(row), offsets.get(row + 1)) else {
                    return &[];
                };
                codes.get(a as usize..b as usize).unwrap_or(&[])
            }
            _ => &[],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Column {
    pub name: String,
    /// `None` when every row has a value; else one flag per row.
    pub validity: Option<Vec<bool>>,
    pub data: ColumnData,
}

impl Column {
    pub fn present(&self, row: usize) -> bool {
        match &self.validity {
            Some(v) => v.get(row).copied().unwrap_or(false),
            None => row < self.data.len(),
        }
    }
}

/// the decoded section: every column, all `n_rows` long.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScalarTable {
    pub n_rows: usize,
    pub columns: Vec<Column>,
}

impl ScalarTable {
    pub fn column(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|c| c.name == name)
    }
}

pub(crate) fn malformed(reason: impl Into<String>) -> UrnaError {
    UrnaError::MalformedSectionPayload {
        section_id: SECTION_CHUNK_SCALARS,
        reason: reason.into(),
    }
}

fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

pub(crate) fn unzigzag(u: u64) -> i64 {
    ((u >> 1) as i64) ^ -((u & 1) as i64)
}

fn bitset(bits: impl Iterator<Item = bool>, n: usize) -> Vec<u8> {
    let mut out = vec![0u8; n.div_ceil(8)];
    for (i, b) in bits.enumerate().take(n) {
        if b {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    out
}

fn push_packed(out: &mut Vec<u8>, values: &[u64]) {
    let packed = pack_u64s(values);
    out.extend_from_slice(&(packed.len() as u32).to_le_bytes());
    out.extend_from_slice(&packed);
}

fn push_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn push_dictionary(out: &mut Vec<u8>, dict: &Dictionary) -> Result<(), UrnaError> {
    out.extend_from_slice(&(dict.values.len() as u32).to_le_bytes());
    for v in &dict.values {
        push_str(out, v);
    }
    match &dict.numeric {
        Some(nums) => {
            if nums.len() != dict.values.len() {
                return Err(malformed("dictionary numeric values do not match entries"));
            }
            out.push(1);
            for n in nums {
                out.extend_from_slice(&n.unwrap_or(f64::NAN).to_le_bytes());
            }
        }
        None => out.push(0),
    }
    Ok(())
}

/// encode `table` into the 0x0D payload. deterministic: the same table
/// gives the same bytes. every column must hold exactly `n_rows` rows,
/// every code must index its dictionary, every offset must be monotonic.
pub fn encode_chunk_scalars(table: &ScalarTable) -> Result<Vec<u8>, UrnaError> {
    let n = table.n_rows;
    let mut out = Vec::new();
    out.extend_from_slice(&CHUNK_SCALARS_PAYLOAD_VERSION.to_le_bytes());
    out.extend_from_slice(&(n as u64).to_le_bytes());
    out.extend_from_slice(&(table.columns.len() as u64).to_le_bytes());
    for c in &table.columns {
        if c.data.len() != n {
            return Err(malformed(format!(
                "column {} has {} rows, table has {n}",
                c.name,
                c.data.len()
            )));
        }
        if table.columns.iter().filter(|o| o.name == c.name).count() > 1 {
            return Err(malformed(format!("column {} given twice", c.name)));
        }
        if c.name.is_empty() {
            return Err(malformed("a column has an empty name"));
        }
        push_str(&mut out, &c.name);
        out.push(c.data.kind() as u8);
        match &c.validity {
            Some(v) => {
                if v.len() != n {
                    return Err(malformed(format!("column {} validity length", c.name)));
                }
                out.push(1);
                out.extend_from_slice(&bitset(v.iter().copied(), n));
            }
            None => out.push(0),
        }
        let body = encode_body(&c.data, n, c.validity.as_deref())?;
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&body);
    }
    Ok(out)
}

fn encode_body(
    data: &ColumnData,
    n: usize,
    validity: Option<&[bool]>,
) -> Result<Vec<u8>, UrnaError> {
    let mut body = Vec::new();
    match data {
        ColumnData::Int(v) => {
            let packed: Vec<u64> = v.iter().map(|&x| zigzag(x)).collect();
            push_packed(&mut body, &packed);
        }
        ColumnData::Real(v) => {
            for x in v {
                body.extend_from_slice(&x.to_le_bytes());
            }
        }
        ColumnData::Bool(v) => body.extend_from_slice(&bitset(v.iter().copied(), n)),
        ColumnData::Text(v) => {
            let mut offsets: Vec<u64> = Vec::with_capacity(n + 1);
            let mut total = 0u64;
            offsets.push(0);
            for s in v {
                total += s.len() as u64;
                offsets.push(total);
            }
            push_packed(&mut body, &offsets);
            for s in v {
                body.extend_from_slice(s.as_bytes());
            }
        }
        ColumnData::Uuid(v) => {
            for u in v {
                body.extend_from_slice(u);
            }
        }
        ColumnData::Enum { dict, codes } => {
            check_codes(dict, codes, validity)?;
            push_dictionary(&mut body, dict)?;
            let packed: Vec<u64> = codes.iter().map(|&c| c as u64).collect();
            push_packed(&mut body, &packed);
        }
        ColumnData::EnumList {
            dict,
            offsets,
            codes,
        } => {
            check_codes(dict, codes, None)?;
            if offsets.len() != n + 1
                || offsets.windows(2).any(|w| w[0] > w[1])
                || offsets.last().is_some_and(|&l| l as usize != codes.len())
            {
                return Err(malformed(
                    "enum_list offsets are not monotonic over the codes",
                ));
            }
            push_dictionary(&mut body, dict)?;
            let packed: Vec<u64> = offsets.iter().map(|&o| o as u64).collect();
            push_packed(&mut body, &packed);
            let packed: Vec<u64> = codes.iter().map(|&c| c as u64).collect();
            push_packed(&mut body, &packed);
        }
    }
    Ok(body)
}

/// a present row's code indexes the dictionary; an absent row's is 0, the
/// only code a column with no value at all (an empty dictionary) carries.
pub(super) fn check_codes(
    dict: &Dictionary,
    codes: &[u32],
    validity: Option<&[bool]>,
) -> Result<(), UrnaError> {
    let n = dict.values.len() as u32;
    let present = |i: usize| validity.is_none_or(|v| v[i]);
    for (i, &c) in codes.iter().enumerate() {
        if if present(i) { c >= n } else { c != 0 } {
            return Err(malformed(format!("code {c} outside a dictionary of {n}")));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn sample() -> ScalarTable {
        let dict = Dictionary {
            values: vec!["1".into(), "*".into(), "3".into()],
            numeric: Some(vec![Some(1.0), None, Some(3.0)]),
        };
        let kinds = Dictionary {
            values: vec!["card".into(), "token".into()],
            numeric: None,
        };
        ScalarTable {
            n_rows: 4,
            columns: vec![
                Column {
                    name: "edhrec_rank".into(),
                    validity: Some(vec![true, true, false, true]),
                    data: ColumnData::Int(vec![12, -3, 0, 1 << 40]),
                },
                Column {
                    name: "cmc".into(),
                    validity: None,
                    data: ColumnData::Real(vec![1.0, 2.5, 0.0, 7.0]),
                },
                Column {
                    name: "foil".into(),
                    validity: None,
                    data: ColumnData::Bool(vec![true, false, false, true]),
                },
                Column {
                    name: "name".into(),
                    validity: None,
                    data: ColumnData::Text(vec![
                        "Lightning Bolt".into(),
                        "".into(),
                        "Relâmpago".into(),
                        "x".into(),
                    ]),
                },
                Column {
                    name: "oracle_id".into(),
                    validity: None,
                    data: ColumnData::Uuid(vec![[1; 16], [2; 16], [0; 16], [9; 16]]),
                },
                Column {
                    name: "power".into(),
                    validity: Some(vec![true, true, true, false]),
                    data: ColumnData::Enum {
                        dict,
                        codes: vec![0, 1, 2, 0],
                    },
                },
                Column {
                    name: "kinds".into(),
                    validity: None,
                    data: ColumnData::EnumList {
                        dict: kinds,
                        offsets: vec![0, 2, 2, 3, 3],
                        codes: vec![0, 1, 1],
                    },
                },
            ],
        }
    }

    #[test]
    fn roundtrips_every_kind_with_and_without_nulls() {
        let table = sample();
        let bytes = encode_chunk_scalars(&table).expect("encodes");
        let back = decode_chunk_scalars(&bytes, 4).expect("decodes");
        assert_eq!(back, table);
        assert_eq!(bytes, encode_chunk_scalars(&back).expect("encodes again"));
        let power = back.column("power").expect("power");
        assert!(power.present(0) && !power.present(3));
        assert_eq!(power.data.dictionary().and_then(|d| d.numeric_of(1)), None);
        assert_eq!(
            power.data.dictionary().and_then(|d| d.numeric_of(2)),
            Some(3.0)
        );
        assert_eq!(
            back.column("kinds").expect("kinds").data.codes_at(0),
            &[0, 1]
        );
        assert!(
            back.column("kinds")
                .expect("kinds")
                .data
                .codes_at(1)
                .is_empty()
        );
    }

    #[test]
    fn refuses_a_table_whose_columns_disagree() {
        let mut table = sample();
        table.n_rows = 3;
        assert!(encode_chunk_scalars(&table).is_err());
        let mut table = sample();
        if let ColumnData::Enum { codes, .. } = &mut table.columns[5].data {
            codes[0] = 9;
        }
        assert!(encode_chunk_scalars(&table).is_err());
    }
}
