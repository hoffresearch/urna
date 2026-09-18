//! `provenance` section (`SECTION_PROVENANCE = 0x05`). A single
//! versioned JSON blob carrying ingestion metadata. v1 keeps it
//! schema-free so callers can include whatever they need (legacy
//! source, ingestion timestamps, per-document labels). The reader
//! only enforces that it parses as JSON.

use super::codec::Cursor;
use crate::error::UrnaError;
use crate::layout::{SECTION_PAYLOAD_PREFIX_SIZE, SECTION_PAYLOAD_VERSION, SECTION_PROVENANCE};

pub fn encode_provenance(value: &serde_json::Value) -> crate::Result<Vec<u8>> {
    let json = serde_json::to_vec(value)?;
    let json_len = u32::try_from(json.len())
        .map_err(|_| UrnaError::InvalidInput("provenance JSON too large".into()))?;
    let mut buf = Vec::with_capacity(SECTION_PAYLOAD_PREFIX_SIZE + json.len());
    buf.extend_from_slice(&SECTION_PAYLOAD_VERSION.to_le_bytes());
    buf.extend_from_slice(&(json_len as u64).to_le_bytes());
    buf.extend_from_slice(&json);
    Ok(buf)
}

pub fn decode_provenance(data: &[u8]) -> crate::Result<serde_json::Value> {
    let mut c = Cursor::new(data, SECTION_PROVENANCE);
    if c.data.len() < SECTION_PAYLOAD_PREFIX_SIZE {
        return Err(c.malformed("payload shorter than prefix"));
    }
    let version = c.read_u32()?;
    if version != SECTION_PAYLOAD_VERSION {
        return Err(UrnaError::UnsupportedSectionVersion {
            section_id: SECTION_PROVENANCE,
            version,
        });
    }
    let json_len = c.read_u64()? as usize;
    let json = c.read_bytes(json_len)?;
    let value: serde_json::Value = serde_json::from_slice(json).map_err(UrnaError::Json)?;
    c.finish()?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let v = serde_json::json!({"source": "demo", "ingested_at": "2026-01-01"});
        let bytes = encode_provenance(&v).unwrap();
        let back = decode_provenance(&bytes).unwrap();
        assert_eq!(v, back);
    }
}
