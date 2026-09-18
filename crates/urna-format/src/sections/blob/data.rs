//! blob_data (0x17): inlined media bytes for the self-contained twin of
//! the 0x14 catalog.
//!
//! OPTIONAL and EXCLUDED from content_hash. The payload opens with an
//! offset table PARALLEL to the blob_refs (0x14) record order — entry i
//! describes where record i's bytes live — followed by the concatenated
//! blob bytes. A record kept out-of-line (`inlined = false`) has a
//! (0, 0) table entry. Offsets are relative to the first data byte, so
//! the table can be decoded alone and the heavy bytes sliced lazily off
//! the mmap; the section is never copied whole.
//!
//! wire encoding: `raw` (media bytes are already codec-compressed).
//! all integers le.

use crate::bytes::{le_u32, le_u64};
use crate::error::UrnaError;
use crate::layout::SECTION_BLOB_DATA;

pub const BLOB_DATA_PAYLOAD_VERSION: u32 = 1;

/// fixed prelude: u32 version + u64 entry count.
const HEADER_SIZE: usize = 12;
/// per-entry cost in the offset table: u64 offset + u64 len.
const ENTRY_SIZE: usize = 16;

/// decoded 0x17 offset table. `entries[i]` is `(offset, len)` of record
/// i's bytes RELATIVE to `data_start`, which is the byte position of the
/// first data byte within the section payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlobDataTable {
    pub entries: Vec<(u64, u64)>,
    pub data_start: usize,
}

fn malformed(reason: impl Into<String>) -> UrnaError {
    UrnaError::MalformedSectionPayload {
        section_id: SECTION_BLOB_DATA,
        reason: reason.into(),
    }
}

/// encode the 0x17 payload from per-record byte slices; `None` marks a
/// record that stays out-of-line. deterministic: same blobs in the same
/// order, same bytes.
pub fn encode_blob_data(blobs: &[Option<&[u8]>]) -> Result<Vec<u8>, UrnaError> {
    let data_len: usize = blobs.iter().flatten().map(|b| b.len()).sum();
    let mut out = Vec::with_capacity(HEADER_SIZE + blobs.len() * ENTRY_SIZE + data_len);
    out.extend_from_slice(&BLOB_DATA_PAYLOAD_VERSION.to_le_bytes());
    out.extend_from_slice(&(blobs.len() as u64).to_le_bytes());
    let mut offset = 0u64;
    for b in blobs {
        match b {
            Some(bytes) => {
                out.extend_from_slice(&offset.to_le_bytes());
                out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
                offset += bytes.len() as u64;
            }
            None => out.extend_from_slice(&[0u8; ENTRY_SIZE]),
        }
    }
    for b in blobs.iter().flatten() {
        out.extend_from_slice(b);
    }
    Ok(out)
}

/// decode ONLY the offset table off the section payload, bounds-checking
/// every entry against the physical data region so a later lazy slice can
/// never read past the section. typed errors, never panics.
pub fn decode_blob_data_table(bytes: &[u8]) -> Result<BlobDataTable, UrnaError> {
    if bytes.len() < HEADER_SIZE {
        return Err(malformed("blob_data: payload shorter than header"));
    }
    let version = le_u32(&bytes[0..4])?;
    if version != BLOB_DATA_PAYLOAD_VERSION {
        return Err(UrnaError::UnsupportedSectionVersion {
            section_id: SECTION_BLOB_DATA,
            version,
        });
    }
    let n = le_u64(&bytes[4..12])? as usize;
    // bound the claim against the physical payload before allocating.
    if n > (bytes.len() - HEADER_SIZE) / ENTRY_SIZE {
        return Err(malformed("blob_data: entry count exceeds payload"));
    }
    let data_start = HEADER_SIZE + n * ENTRY_SIZE;
    let data_len = (bytes.len() - data_start) as u64;
    let mut entries = Vec::with_capacity(n);
    for i in 0..n {
        let at = HEADER_SIZE + i * ENTRY_SIZE;
        let offset = le_u64(&bytes[at..at + 8])?;
        let len = le_u64(&bytes[at + 8..at + 16])?;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| malformed(format!("blob_data: entry {} offset overflow", i)))?;
        if end > data_len {
            return Err(malformed(format!(
                "blob_data: entry {} spans past payload ({} > {})",
                i, end, data_len
            )));
        }
        entries.push((offset, len));
    }
    Ok(BlobDataTable {
        entries,
        data_start,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_with_gaps() {
        let a = b"av1-shard-bytes".as_slice();
        let b = b"x".as_slice();
        let payload = encode_blob_data(&[Some(a), None, Some(b)]).unwrap();
        let table = decode_blob_data_table(&payload).unwrap();
        assert_eq!(table.entries, vec![(0, 15), (0, 0), (15, 1)]);
        let data = &payload[table.data_start..];
        assert_eq!(&data[0..15], a);
        assert_eq!(&data[15..16], b);
    }

    #[test]
    fn empty_table_roundtrips() {
        let payload = encode_blob_data(&[]).unwrap();
        let table = decode_blob_data_table(&payload).unwrap();
        assert!(table.entries.is_empty());
        assert_eq!(table.data_start, payload.len());
    }

    #[test]
    fn hostile_count_is_rejected() {
        let mut payload = encode_blob_data(&[Some(b"abc".as_slice())]).unwrap();
        payload[4..12].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(decode_blob_data_table(&payload).is_err());
    }

    #[test]
    fn entry_past_payload_is_rejected() {
        let mut payload = encode_blob_data(&[Some(b"abc".as_slice())]).unwrap();
        // inflate the entry length past the physical data region.
        payload[20..28].copy_from_slice(&1000u64.to_le_bytes());
        assert!(decode_blob_data_table(&payload).is_err());
    }

    #[test]
    fn bad_version_is_typed() {
        let mut payload = encode_blob_data(&[]).unwrap();
        payload[0..4].copy_from_slice(&9u32.to_le_bytes());
        assert!(matches!(
            decode_blob_data_table(&payload),
            Err(UrnaError::UnsupportedSectionVersion { .. })
        ));
    }
}
