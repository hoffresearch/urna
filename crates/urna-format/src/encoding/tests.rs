//! unit tests for the encoding dispatch (`encoding/mod.rs`), carved out so
//! the module stays under the 300-line src guard.

use super::*;

#[test]
#[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
fn zstd_roundtrip_preserves_bytes() {
    let original = b"hello hello hello world world world".repeat(64);
    let compressed = zstd_encode(&original).unwrap();
    assert!(
        compressed.len() < original.len(),
        "zstd should shrink repetitive text"
    );
    let decoded = decode_payload(SECTION_ENCODING_ZSTD, &compressed).unwrap();
    assert_eq!(decoded.as_ref(), original.as_slice());
}

#[test]
fn raw_decode_borrows() {
    let bytes = b"plain";
    let decoded = decode_payload(SECTION_ENCODING_RAW, bytes).unwrap();
    assert!(matches!(decoded, std::borrow::Cow::Borrowed(_)));
}

#[test]
fn unknown_encoding_rejected() {
    let res = decode_payload(99, &[]);
    assert!(matches!(
        res,
        Err(UrnaError::UnsupportedSectionEncoding { encoding: 99, .. })
    ));
}

#[test]
fn intpack_decode_payload_rebuilds_canonical_bytes() {
    // intpack (id 4) repacks chunk_ids/spans; decode_payload must
    // rebuild the BYTE-IDENTICAL raw payload so content_hash (hashed
    // over decoded bytes) is unchanged and citations stay stable.
    use crate::sections::{encode_chunk_ids, encode_chunk_ids_intpack};
    let ids: Vec<String> = (0u8..3)
        .map(|i| format!("sha256:{}", hex::encode([i; 32])))
        .collect();
    let packed = encode_chunk_ids_intpack(&ids).unwrap();
    let decoded = decode_payload(SECTION_ENCODING_INTPACK, &packed).unwrap();
    assert_eq!(decoded.as_ref(), encode_chunk_ids(&ids).unwrap().as_slice());
    // a malformed intpack payload is a typed error, never a panic.
    assert!(decode_payload(SECTION_ENCODING_INTPACK, &[]).is_err());
}

#[test]
fn wire_codec_registry_maps_only_implemented_ids() {
    use crate::layout::{
        SECTION_ENCODING_FRONTCODE, SECTION_ENCODING_FSST, SECTION_ENCODING_INT4,
        SECTION_ENCODING_INTPACK, SECTION_ENCODING_RABITQ, SECTION_ENCODING_RAW,
        SECTION_ENCODING_TXT_STREAMS, SECTION_ENCODING_ZSTD, SECTION_ENCODING_ZSTD_DICT,
    };
    assert!(WireCodec::from_id(SECTION_ENCODING_RAW).is_some());
    assert!(WireCodec::from_id(SECTION_ENCODING_ZSTD).is_some());
    // intpack (id 4) is now implemented and in the registry.
    assert!(WireCodec::from_id(SECTION_ENCODING_INTPACK).is_some());
    // int4 (id 7) is now implemented and in the registry.
    assert!(WireCodec::from_id(SECTION_ENCODING_INT4).is_some());
    // txt_streams (id 10) is now implemented and in the registry.
    assert!(WireCodec::from_id(SECTION_ENCODING_TXT_STREAMS).is_some());
    // fsst (id 9) is now implemented and in the registry (self-contained).
    assert!(WireCodec::from_id(SECTION_ENCODING_FSST).is_some());
    // zstd_dict (id 5) is implemented but needs the shared dictionary
    // from section 0x0A, so it is NOT in the context-free registry: it is
    // decoded via `decode_payload_with_dict`, not `decode_payload`.
    assert!(WireCodec::from_id(SECTION_ENCODING_ZSTD_DICT).is_none());
    // still-reserved-but-unimplemented ids stay rejected: frontcode(6),
    // rabitq(8), and any unknown id.
    assert!(WireCodec::from_id(SECTION_ENCODING_FRONTCODE).is_none());
    assert!(WireCodec::from_id(SECTION_ENCODING_RABITQ).is_none());
    assert!(WireCodec::from_id(0xFF).is_none());
}

#[test]
#[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
fn encode_smallest_picks_the_winner_and_records_its_id() {
    use crate::layout::{SECTION_ENCODING_RAW, SECTION_ENCODING_ZSTD};
    let candidates = [SECTION_ENCODING_RAW, SECTION_ENCODING_ZSTD];

    // highly repetitive payload: zstd wins, and the chosen id is recorded.
    let compressible = b"abcabcabcabc".repeat(64);
    let (id, bytes) = encode_smallest(&candidates, &compressible).unwrap();
    assert_eq!(id, SECTION_ENCODING_ZSTD);
    assert!(bytes.len() < compressible.len());

    // tiny payload: zstd framing overhead loses, raw wins (first on ties).
    let (id2, _) = encode_smallest(&candidates, b"x").unwrap();
    assert_eq!(id2, SECTION_ENCODING_RAW);
}

#[test]
fn encode_smallest_rejects_embedding_and_empty_candidates() {
    use crate::layout::SECTION_ENCODING_INT8;
    assert!(encode_smallest(&[SECTION_ENCODING_INT8], b"data").is_err());
    assert!(encode_smallest(&[], b"data").is_err());
}
