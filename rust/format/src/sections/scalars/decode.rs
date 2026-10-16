//! decoding the chunk_scalars (0x0D) payload. every read is bounds-checked
//! and every claimed count is bounded by the bytes that back it before any
//! allocation, so a hostile payload gets a typed error, never a panic or a
//! huge reservation.

use super::{
    CHUNK_SCALARS_PAYLOAD_VERSION, Column, ColumnData, ColumnKind, Dictionary, ScalarTable,
    malformed, unzigzag,
};
use crate::bytes::{le_u32, le_u64};
use crate::encoding::unpack_u64s;
use crate::error::UrnaError;
use crate::layout::SECTION_CHUNK_SCALARS;

/// smallest encoded column: name_len, one name byte, kind, validity flag,
/// body_len.
const MIN_COLUMN_SIZE: usize = 4 + 1 + 1 + 1 + 4;

/// decode the payload into a table of exactly `n_rows` rows (the file's
/// chunk count; a table of another length is malformed).
pub fn decode_chunk_scalars(bytes: &[u8], n_rows: usize) -> Result<ScalarTable, UrnaError> {
    let mut cur = Cursor::new(bytes);
    let version = cur.u32()?;
    if version != CHUNK_SCALARS_PAYLOAD_VERSION {
        return Err(UrnaError::UnsupportedSectionVersion {
            section_id: SECTION_CHUNK_SCALARS,
            version,
        });
    }
    let rows = cur.u64()? as usize;
    if rows != n_rows {
        return Err(malformed(format!(
            "chunk_scalars has {rows} rows, the file has {n_rows} chunks"
        )));
    }
    let n_columns = cur.u64()? as usize;
    if n_columns > cur.remaining() / MIN_COLUMN_SIZE {
        return Err(malformed("chunk_scalars: column count exceeds payload"));
    }
    let mut columns = Vec::with_capacity(n_columns);
    for _ in 0..n_columns {
        let name = cur.string()?;
        if name.is_empty() {
            return Err(malformed("chunk_scalars: empty column name"));
        }
        let kind = ColumnKind::from_byte(cur.u8()?)
            .ok_or_else(|| malformed(format!("chunk_scalars: unknown kind in column {name}")))?;
        let validity = match cur.u8()? {
            0 => None,
            1 => Some(read_bits(cur.take(n_rows.div_ceil(8))?, n_rows)),
            other => {
                return Err(malformed(format!(
                    "chunk_scalars: bad validity flag {other} in column {name}"
                )));
            }
        };
        let body_len = cur.u32()? as usize;
        let body = cur.take(body_len)?;
        let data = decode_body(kind, body, n_rows).map_err(|e| tag(e, &name))?;
        columns.push(Column {
            name,
            validity,
            data,
        });
    }
    if cur.pos != bytes.len() {
        return Err(malformed("chunk_scalars: trailing bytes after columns"));
    }
    Ok(ScalarTable { n_rows, columns })
}

fn tag(e: UrnaError, column: &str) -> UrnaError {
    match e {
        UrnaError::MalformedSectionPayload { section_id, reason } => {
            UrnaError::MalformedSectionPayload {
                section_id,
                reason: format!("column {column}: {reason}"),
            }
        }
        other => other,
    }
}

fn decode_body(kind: ColumnKind, body: &[u8], n: usize) -> Result<ColumnData, UrnaError> {
    let mut cur = Cursor::new(body);
    let data = match kind {
        ColumnKind::Int => {
            let packed = cur.packed(n)?;
            ColumnData::Int(packed.into_iter().map(unzigzag).collect())
        }
        ColumnKind::Real => {
            let raw = cur.take(
                n.checked_mul(8)
                    .ok_or_else(|| malformed("real: overflow"))?,
            )?;
            ColumnData::Real(
                raw.chunks_exact(8)
                    .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect(),
            )
        }
        ColumnKind::Bool => ColumnData::Bool(read_bits(cur.take(n.div_ceil(8))?, n)),
        ColumnKind::Text => {
            let offsets = cur.packed(n + 1)?;
            let total = offsets.last().copied().unwrap_or(0) as usize;
            if offsets.windows(2).any(|w| w[0] > w[1]) {
                return Err(malformed("text offsets are not monotonic"));
            }
            let text = cur.take(total)?;
            let mut out = Vec::with_capacity(n);
            for w in offsets.windows(2) {
                let s = std::str::from_utf8(&text[w[0] as usize..w[1] as usize])
                    .map_err(|_| malformed("text is not utf-8"))?;
                out.push(s.to_string());
            }
            ColumnData::Text(out)
        }
        ColumnKind::Uuid => {
            let raw = cur.take(
                n.checked_mul(16)
                    .ok_or_else(|| malformed("uuid: overflow"))?,
            )?;
            ColumnData::Uuid(
                raw.chunks_exact(16)
                    .map(|c| {
                        let mut u = [0u8; 16];
                        u.copy_from_slice(c);
                        u
                    })
                    .collect(),
            )
        }
        ColumnKind::Enum => {
            let dict = cur.dictionary()?;
            let codes = cur.codes(n, &dict)?;
            ColumnData::Enum { dict, codes }
        }
        ColumnKind::EnumList => {
            let dict = cur.dictionary()?;
            let offsets = cur.packed(n + 1)?;
            if offsets.windows(2).any(|w| w[0] > w[1]) {
                return Err(malformed("enum_list offsets are not monotonic"));
            }
            let total = offsets.last().copied().unwrap_or(0);
            if total > u32::MAX as u64 {
                return Err(malformed("enum_list offsets overflow"));
            }
            let codes = cur.codes(total as usize, &dict)?;
            ColumnData::EnumList {
                dict,
                offsets: offsets.into_iter().map(|o| o as u32).collect(),
                codes,
            }
        }
    };
    if cur.pos != body.len() {
        return Err(malformed("trailing bytes after the column body"));
    }
    Ok(data)
}

fn read_bits(bytes: &[u8], n: usize) -> Vec<bool> {
    (0..n).map(|i| bytes[i / 8] & (1 << (i % 8)) != 0).collect()
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], UrnaError> {
        if n > self.remaining() {
            return Err(malformed("unexpected EOF"));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, UrnaError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, UrnaError> {
        le_u32(self.take(4)?)
    }
    fn u64(&mut self) -> Result<u64, UrnaError> {
        le_u64(self.take(8)?)
    }
    fn string(&mut self) -> Result<String, UrnaError> {
        let len = self.u32()? as usize;
        let raw = self.take(len)?;
        std::str::from_utf8(raw)
            .map(str::to_string)
            .map_err(|_| malformed("string is not utf-8"))
    }
    /// a length-prefixed intpack blob holding exactly `want` values.
    fn packed(&mut self, want: usize) -> Result<Vec<u64>, UrnaError> {
        let len = self.u32()? as usize;
        let values = unpack_u64s(self.take(len)?)?;
        if values.len() != want {
            return Err(malformed(format!(
                "packed column holds {} values, expected {want}",
                values.len()
            )));
        }
        Ok(values)
    }
    fn dictionary(&mut self) -> Result<Dictionary, UrnaError> {
        let n = self.u32()? as usize;
        if n > self.remaining() / 4 {
            return Err(malformed("dictionary entry count exceeds payload"));
        }
        let mut values = Vec::with_capacity(n);
        for _ in 0..n {
            values.push(self.string()?);
        }
        let numeric = match self.u8()? {
            0 => None,
            1 => {
                let raw = self.take(
                    n.checked_mul(8)
                        .ok_or_else(|| malformed("numeric overflow"))?,
                )?;
                Some(
                    raw.chunks_exact(8)
                        .map(|c| {
                            let v = f64::from_le_bytes([
                                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                            ]);
                            v.is_finite().then_some(v)
                        })
                        .collect(),
                )
            }
            other => return Err(malformed(format!("bad numeric flag {other}"))),
        };
        Ok(Dictionary { values, numeric })
    }
    fn codes(&mut self, want: usize, dict: &Dictionary) -> Result<Vec<u32>, UrnaError> {
        let n = dict.values.len() as u64;
        let packed = self.packed(want)?;
        let mut codes = Vec::with_capacity(want);
        for c in packed {
            if c >= n {
                return Err(malformed(format!("code {c} outside a dictionary of {n}")));
            }
            codes.push(c as u32);
        }
        Ok(codes)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{encode_chunk_scalars, tests::sample};
    use super::*;

    #[test]
    fn truncation_wrong_row_count_and_hostile_claims_are_typed_errors() {
        let bytes = encode_chunk_scalars(&sample()).expect("encodes");
        for cut in [0, 4, 12, 20, 40, bytes.len() - 1] {
            assert!(decode_chunk_scalars(&bytes[..cut], 4).is_err(), "cut {cut}");
        }
        assert!(decode_chunk_scalars(&bytes, 5).is_err());
        let mut hostile = bytes.clone();
        hostile[12..20].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(
            decode_chunk_scalars(&hostile, 4),
            Err(UrnaError::MalformedSectionPayload { .. })
        ));
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(decode_chunk_scalars(&trailing, 4).is_err());
        let mut versioned = bytes;
        versioned[0] = 9;
        assert!(matches!(
            decode_chunk_scalars(&versioned, 4),
            Err(UrnaError::UnsupportedSectionVersion { .. })
        ));
    }

    #[test]
    fn a_code_outside_the_dictionary_is_refused() {
        let table = sample();
        let mut bytes = encode_chunk_scalars(&table).expect("encodes");
        // the enum column's codes are packed last in its body; flipping the
        // dictionary count to 1 makes code 2 out of range.
        let marker = b"power";
        let at = bytes
            .windows(marker.len())
            .position(|w| w == marker)
            .expect("column");
        let dict_count_at = at + marker.len() + 1 + 1 + 1 + 4;
        bytes[dict_count_at..dict_count_at + 4].copy_from_slice(&1u32.to_le_bytes());
        assert!(decode_chunk_scalars(&bytes, 4).is_err());
    }
}
