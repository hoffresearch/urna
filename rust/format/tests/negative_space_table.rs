//! negative paths for the space_table (0x15) codec: every malformed or
//! hostile payload must return a typed `UrnaError`, NEVER panic (mirrors
//! negative_blob_refs discipline). plus an exhaustive prefix-truncation
//! fuzz.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use urna_format::{SPACE_DTYPE_F32, SpaceEntry, UrnaError, decode_space_table, encode_space_table};

fn good_entries() -> Vec<SpaceEntry> {
    vec![
        SpaceEntry {
            space_index: 1,
            name: "vision".into(),
            dim: 512,
            dtype: SPACE_DTYPE_F32,
            model_hash: format!("sha256:{}", "1".repeat(64)),
            n_vectors: 10015,
        },
        SpaceEntry {
            space_index: 2,
            name: "audio".into(),
            dim: 128,
            dtype: 1,
            model_hash: format!("sha256:{}", "2".repeat(64)),
            n_vectors: 10015,
        },
    ]
}

#[test]
fn baseline_decodes_cleanly() {
    let packed = encode_space_table(&good_entries()).unwrap();
    assert!(decode_space_table(&packed).is_ok());
}

#[test]
fn empty_payload_errors() {
    assert!(decode_space_table(&[]).is_err());
}

#[test]
fn bad_version_errors() {
    let mut packed = encode_space_table(&good_entries()).unwrap();
    packed[0] = packed[0].wrapping_add(5);
    match decode_space_table(&packed) {
        Err(UrnaError::UnsupportedSectionVersion { .. }) => {}
        other => panic!("expected UnsupportedSectionVersion, got {:?}", other),
    }
}

#[test]
fn hostile_count_does_not_allocate() {
    // tamper the u64 count (bytes [4..12)) to u64::MAX: the
    // count-vs-payload bound must reject BEFORE any allocation.
    let mut packed = encode_space_table(&good_entries()).unwrap();
    packed[4..12].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(decode_space_table(&packed).is_err());
}

#[test]
fn hostile_name_length_errors() {
    // first entry: 12 header + 1 index, then the u32 name length at
    // bytes [13..17). claim a huge name; the bound must reject.
    let mut packed = encode_space_table(&good_entries()).unwrap();
    packed[13..17].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(decode_space_table(&packed).is_err());
}

#[test]
fn reserved_space_zero_rejected_at_decode() {
    // space_index 0 is the canonical text space and must never be listed;
    // even a hand-crafted payload is rejected at decode, not just encode.
    let mut packed = encode_space_table(&good_entries()).unwrap();
    assert_eq!(packed[12], 1, "test layout assumption");
    packed[12] = 0;
    assert!(decode_space_table(&packed).is_err());
}

#[test]
fn unknown_dtype_rejected_at_decode() {
    // first entry's dtype byte: 12 header + 1 index + 4 len + 6 name + 4
    // dim = byte 27.
    let mut packed = encode_space_table(&good_entries()).unwrap();
    assert_eq!(packed[27], 0, "test layout assumption");
    packed[27] = 42;
    assert!(decode_space_table(&packed).is_err());
}

#[test]
fn duplicate_index_rejected_at_decode() {
    // two entries, second hand-patched to reuse index 1: byte layout of
    // entry 2 starts at 12 + (1+4+6+4+1+4+71+8) = 111; its index byte.
    let packed = encode_space_table(&good_entries()).unwrap();
    let mut evil = packed.clone();
    assert_eq!(evil[111], 2, "test layout assumption");
    evil[111] = 1;
    assert!(decode_space_table(&evil).is_err());
}

#[test]
fn trailing_bytes_error() {
    let mut packed = encode_space_table(&good_entries()).unwrap();
    packed.push(0);
    assert!(decode_space_table(&packed).is_err());
}

#[test]
fn fuzz_every_truncation_never_panics() {
    let packed = encode_space_table(&good_entries()).unwrap();
    for cut in 0..=packed.len() {
        let _ = decode_space_table(&packed[..cut]);
    }
}

#[test]
fn fuzz_byte_flips_never_panic() {
    let packed = encode_space_table(&good_entries()).unwrap();
    for i in 12..packed.len() {
        let mut evil = packed.clone();
        evil[i] ^= 0xFF;
        let _ = decode_space_table(&evil);
    }
}
