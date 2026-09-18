//! negative paths for the blob_refs (0x14) and blob_span_overlay (0x16)
//! codecs: every malformed or hostile payload must return a typed
//! `UrnaError`, NEVER panic (mirrors negative_graph_adjacency discipline).
//! plus an exhaustive prefix-truncation fuzz on both.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use urna_format::{
    BLOB_REF_NONE, BlobRefRecord, BlobSpanEntry, UrnaError, decode_blob_refs,
    decode_blob_span_overlay, encode_blob_refs, encode_blob_span_overlay,
};

fn good_records() -> Vec<BlobRefRecord> {
    vec![
        BlobRefRecord {
            content_hash: [1; 32],
            original_uri: "media/a.av1".into(),
            byte_len: 1_000_000,
            inlined: true,
        },
        BlobRefRecord {
            content_hash: [2; 32],
            original_uri: "s3://bucket/b.av1".into(),
            byte_len: 2_000_000,
            inlined: false,
        },
    ]
}

fn good_overlay() -> Vec<BlobSpanEntry> {
    vec![
        BlobSpanEntry {
            blob_ref_index: 0,
            byte_start: 0,
            byte_end: 100,
        },
        BlobSpanEntry {
            blob_ref_index: BLOB_REF_NONE,
            byte_start: 0,
            byte_end: 0,
        },
    ]
}

#[test]
fn baseline_decodes_cleanly() {
    let packed = encode_blob_refs(&good_records()).unwrap();
    assert!(decode_blob_refs(&packed).is_ok());
    let overlay = encode_blob_span_overlay(&good_overlay()).unwrap();
    assert!(decode_blob_span_overlay(&overlay).is_ok());
}

#[test]
fn empty_payload_errors() {
    assert!(decode_blob_refs(&[]).is_err());
    assert!(decode_blob_span_overlay(&[]).is_err());
}

#[test]
fn bad_version_errors() {
    let mut packed = encode_blob_refs(&good_records()).unwrap();
    packed[0] = packed[0].wrapping_add(9);
    match decode_blob_refs(&packed) {
        Err(UrnaError::UnsupportedSectionVersion { .. }) => {}
        other => panic!("expected UnsupportedSectionVersion, got {:?}", other),
    }
    let mut overlay = encode_blob_span_overlay(&good_overlay()).unwrap();
    overlay[0] = overlay[0].wrapping_add(9);
    match decode_blob_span_overlay(&overlay) {
        Err(UrnaError::UnsupportedSectionVersion { .. }) => {}
        other => panic!("expected UnsupportedSectionVersion, got {:?}", other),
    }
}

#[test]
fn hostile_count_does_not_allocate() {
    // tamper the u64 count (bytes [4..12)) to u64::MAX: the
    // count-vs-payload bound must reject BEFORE any allocation, never
    // panic, never try to reserve gigabytes from the claim alone.
    let mut packed = encode_blob_refs(&good_records()).unwrap();
    packed[4..12].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(decode_blob_refs(&packed).is_err());
    let mut overlay = encode_blob_span_overlay(&good_overlay()).unwrap();
    overlay[4..12].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(decode_blob_span_overlay(&overlay).is_err());
}

#[test]
fn hostile_uri_length_errors() {
    // the first record's uri length sits right after the 12-byte header
    // and the 32-byte hash: bytes [44..48). claim a huge uri; the
    // length-vs-remaining bound must reject.
    let mut packed = encode_blob_refs(&good_records()).unwrap();
    packed[44..48].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(decode_blob_refs(&packed).is_err());
}

#[test]
fn bad_inlined_flag_errors() {
    // first record: 12 header + 32 hash + 4 uri-len + 11 uri + 8 byte-len
    // puts the inlined flag at byte 67. anything but 0/1 must error.
    let mut packed = encode_blob_refs(&good_records()).unwrap();
    assert_eq!(packed[67], 1, "test layout assumption");
    packed[67] = 7;
    assert!(decode_blob_refs(&packed).is_err());
}

#[test]
fn non_utf8_uri_errors() {
    let mut packed = encode_blob_refs(&good_records()).unwrap();
    // first uri starts at byte 48 (12 header + 32 hash + 4 len).
    packed[48] = 0xFF;
    assert!(decode_blob_refs(&packed).is_err());
}

#[test]
fn trailing_bytes_error() {
    let mut packed = encode_blob_refs(&good_records()).unwrap();
    packed.push(0);
    assert!(decode_blob_refs(&packed).is_err());
    let mut overlay = encode_blob_span_overlay(&good_overlay()).unwrap();
    overlay.push(0);
    assert!(decode_blob_span_overlay(&overlay).is_err());
}

#[test]
fn fuzz_every_truncation_never_panics() {
    // exhaustive prefix truncation: the core no-panic-on-hostile-mmap
    // guarantee (mirrors negative_graph_adjacency).
    let packed = encode_blob_refs(&good_records()).unwrap();
    for cut in 0..=packed.len() {
        let _ = decode_blob_refs(&packed[..cut]);
    }
    let overlay = encode_blob_span_overlay(&good_overlay()).unwrap();
    for cut in 0..=overlay.len() {
        let _ = decode_blob_span_overlay(&overlay[..cut]);
    }
}

#[test]
fn fuzz_byte_flips_never_panic() {
    // flip every byte past the header: every result must be a typed error
    // or a (different but valid) parse, never a panic.
    let packed = encode_blob_refs(&good_records()).unwrap();
    for i in 12..packed.len() {
        let mut evil = packed.clone();
        evil[i] ^= 0xFF;
        let _ = decode_blob_refs(&evil);
    }
    let overlay = encode_blob_span_overlay(&good_overlay()).unwrap();
    for i in 12..overlay.len() {
        let mut evil = overlay.clone();
        evil[i] ^= 0xFF;
        let _ = decode_blob_span_overlay(&evil);
    }
}
