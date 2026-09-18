//! `chunk_ids` section (`SECTION_CHUNK_IDS = 0x01`). Length-prefixed
//! UTF-8 strings of the form `sha256:<64 hex>` — one per chunk.
//!
//! the raw encoding stores each id as 71 ascii bytes. the `intpack`
//! repack (encoding id 4, kind 0) stores only the 32 raw digest bytes
//! per id (~2.3x smaller) and reconstructs the exact ascii payload on
//! decode, so `content_hash` (computed over the decoded bytes) is
//! byte-identical to the raw form and citations stay stable.

use super::REPACK_KIND_CHUNK_IDS;
use super::codec::{Cursor, read_prefix, write_lp_str, write_prefix};
use crate::bytes::le_u32;
use crate::error::UrnaError;
use crate::layout::SECTION_CHUNK_IDS;

/// canonical chunk-id prefix; the rest is 64 lowercase hex digits.
const SHA256_PREFIX: &str = "sha256:";
const DIGEST_LEN: usize = 32;

pub fn encode_chunk_ids(ids: &[String]) -> crate::Result<Vec<u8>> {
    let mut buf = Vec::new();
    write_prefix(&mut buf, ids.len() as u64);
    for id in ids {
        write_lp_str(&mut buf, id)?;
    }
    Ok(buf)
}

pub fn decode_chunk_ids(data: &[u8], expected_count: usize) -> crate::Result<Vec<String>> {
    let mut c = Cursor::new(data, SECTION_CHUNK_IDS);
    let count = read_prefix(&mut c)? as usize;
    if count != expected_count {
        return Err(UrnaError::SectionCountMismatch {
            section_id: SECTION_CHUNK_IDS,
            expected: expected_count,
            got: count,
        });
    }
    let mut ids = Vec::with_capacity(count);
    for _ in 0..count {
        ids.push(c.read_lp_str()?);
    }
    c.finish()?;
    Ok(ids)
}

/// encode the chunk-ids section as an `intpack` repack payload: a kind
/// byte, the count, then the 32 raw digest bytes per id. returns `None`
/// when any id is not a canonical `sha256:<64 lowercase hex>` string, so
/// the writer falls back to the raw encoding and reconstruction stays
/// guaranteed byte-exact.
pub fn encode_chunk_ids_intpack(ids: &[String]) -> Option<Vec<u8>> {
    let mut digests: Vec<u8> = Vec::with_capacity(ids.len() * DIGEST_LEN);
    for id in ids {
        let hex_part = id.strip_prefix(SHA256_PREFIX)?;
        if hex_part.len() != DIGEST_LEN * 2 {
            return None;
        }
        let d = hex::decode(hex_part).ok()?;
        // guard against any non-canonical (e.g. uppercase) hex so the
        // round-trip reproduces the original ascii byte-for-byte.
        if hex::encode(&d) != hex_part {
            return None;
        }
        digests.extend_from_slice(&d);
    }
    let mut out = Vec::with_capacity(1 + 4 + digests.len());
    out.push(REPACK_KIND_CHUNK_IDS);
    out.extend_from_slice(&(ids.len() as u32).to_le_bytes());
    out.extend_from_slice(&digests);
    Some(out)
}

/// reconstruct the canonical (raw-encoding) chunk-ids payload from the
/// body of an `intpack` repack (the bytes after the kind byte). the
/// output is byte-identical to [`encode_chunk_ids`] so `content_hash`
/// is preserved.
pub fn decode_chunk_ids_intpack(rest: &[u8]) -> crate::Result<Vec<u8>> {
    let malformed = |reason: &str| UrnaError::MalformedSectionPayload {
        section_id: SECTION_CHUNK_IDS,
        reason: reason.into(),
    };
    if rest.len() < 4 {
        return Err(malformed("chunk_ids intpack: truncated count"));
    }
    let count = le_u32(&rest[0..4])? as usize;
    let body = &rest[4..];
    if body.len() != count * DIGEST_LEN {
        return Err(malformed("chunk_ids intpack: digest body size mismatch"));
    }
    let mut buf = Vec::with_capacity(12 + count * (4 + 71));
    write_prefix(&mut buf, count as u64);
    for d in body.chunks_exact(DIGEST_LEN) {
        let s = format!("{}{}", SHA256_PREFIX, hex::encode(d));
        write_lp_str(&mut buf, &s)?;
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::SECTION_CHUNK_IDS;

    #[test]
    fn roundtrip() {
        let ids = vec!["sha256:aaa".to_string(), "sha256:bbb".to_string()];
        let bytes = encode_chunk_ids(&ids).unwrap();
        let back = decode_chunk_ids(&bytes, 2).unwrap();
        assert_eq!(ids, back);
    }

    #[test]
    fn count_mismatch() {
        let ids = vec!["a".to_string()];
        let bytes = encode_chunk_ids(&ids).unwrap();
        let err = decode_chunk_ids(&bytes, 5).unwrap_err();
        assert!(matches!(err, UrnaError::SectionCountMismatch { .. }));
    }

    fn sample_ids() -> Vec<String> {
        (0u8..4)
            .map(|i| format!("sha256:{}", hex::encode([i; 32])))
            .collect()
    }

    #[test]
    fn intpack_decodes_byte_identical_to_raw() {
        // the whole point: the packed form must decode to the exact raw
        // payload so content_hash (over decoded bytes) is unchanged.
        let ids = sample_ids();
        let packed = encode_chunk_ids_intpack(&ids).unwrap();
        assert_eq!(packed[0], REPACK_KIND_CHUNK_IDS);
        // packed is ~2.3x smaller than the raw ascii payload.
        let raw = encode_chunk_ids(&ids).unwrap();
        assert!(packed.len() < raw.len());
        let reconstructed = decode_chunk_ids_intpack(&packed[1..]).unwrap();
        assert_eq!(reconstructed, raw, "intpack repack must rebuild raw bytes");
        // and the reconstructed payload decodes back to the same ids.
        assert_eq!(decode_chunk_ids(&reconstructed, ids.len()).unwrap(), ids);
    }

    #[test]
    fn intpack_rejects_non_canonical_ids() {
        assert!(encode_chunk_ids_intpack(&["not-a-hash".to_string()]).is_none());
        assert!(encode_chunk_ids_intpack(&["sha256:ABCD".to_string()]).is_none());
        // uppercase hex is not the canonical lowercase form.
        let up = format!("sha256:{}", "A".repeat(64));
        assert!(encode_chunk_ids_intpack(&[up]).is_none());
    }

    #[test]
    fn intpack_empty_roundtrips() {
        let packed = encode_chunk_ids_intpack(&[]).unwrap();
        let reconstructed = decode_chunk_ids_intpack(&packed[1..]).unwrap();
        assert_eq!(reconstructed, encode_chunk_ids(&[]).unwrap());
    }

    #[test]
    fn intpack_body_size_mismatch_errors() {
        assert!(decode_chunk_ids_intpack(&[]).is_err());
        let mut bad = 2u32.to_le_bytes().to_vec();
        bad.extend_from_slice(&[0u8; 40]); // 40 != 2*32
        assert!(decode_chunk_ids_intpack(&bad).is_err());
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&99u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        let err = decode_chunk_ids(&buf, 0).unwrap_err();
        assert!(matches!(
            err,
            UrnaError::UnsupportedSectionVersion {
                section_id: SECTION_CHUNK_IDS,
                version: 99,
            }
        ));
    }
}
