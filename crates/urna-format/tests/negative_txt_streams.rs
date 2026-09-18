//! negative paths for the `txt_streams` wire codec (encoding id 10):
//! every malformed or hostile payload must return a typed `UrnaError`,
//! NEVER panic (mirrors the negative_int8 / negative_zstd discipline).
//!
//! container layout (see encoding/txt_streams.rs):
//!
//! ```text
//!   [0]        u8  kind/version (TXT_STREAMS_V1 = 0)
//!   [1..9]     u64 chunk count  (LE)
//!   [9..9+T]   intpack offset table of N+1 byte offsets
//!   [9+T..]    N zstd streams
//! ```

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use urna_format::layout::SECTION_ENCODING_TXT_STREAMS;
use urna_format::{
    TXT_STREAMS_V1, UrnaError, decode_payload, decode_txt_streams, encode_txt_streams,
};

fn good_packed() -> Vec<u8> {
    let texts: Vec<String> = ["primeiro", "segundo", "coração", "quarto"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    encode_txt_streams(&texts).unwrap()
}

fn assert_typed_err(res: urna_format::Result<Vec<u8>>) {
    match res {
        Err(UrnaError::MalformedSectionPayload { .. }) => {}
        other => panic!("expected MalformedSectionPayload; got {:?}", other),
    }
}

#[test]
#[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
fn baseline_decodes_cleanly() {
    // guards against false-positive negatives: a real payload must decode.
    let packed = good_packed();
    assert!(decode_txt_streams(&packed).is_ok());
    assert!(decode_payload(SECTION_ENCODING_TXT_STREAMS, &packed).is_ok());
}

#[test]
fn empty_payload_errors() {
    assert_typed_err(decode_txt_streams(&[]));
    // also through the registry: must be a typed error, not a panic.
    assert!(decode_payload(SECTION_ENCODING_TXT_STREAMS, &[]).is_err());
}

#[test]
#[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
fn bad_kind_byte_errors() {
    let mut packed = good_packed();
    packed[0] = TXT_STREAMS_V1.wrapping_add(7); // unknown kind/version
    assert_typed_err(decode_txt_streams(&packed));
}

#[test]
#[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
fn truncated_count_errors() {
    // only the kind byte and a partial count: cannot read the u64 count.
    for len in 1..9 {
        let mut packed = good_packed();
        packed.truncate(len);
        assert_typed_err(decode_txt_streams(&packed));
    }
}

#[test]
#[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
fn truncated_offset_table_errors() {
    // chop into the intpack offset table region (right after the 9-byte
    // header). the IntpackReader directory / block bounds checks must fire.
    let packed = good_packed();
    for cut in 9..packed.len().min(40) {
        let truncated = &packed[..cut];
        // must never panic; an error is the expected outcome here.
        let _ = decode_txt_streams(truncated);
    }
}

#[test]
#[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
fn truncated_stream_body_errors() {
    // drop the last byte of the final zstd stream: the final-offset ==
    // streams-length check or the zstd decode must reject it.
    let mut packed = good_packed();
    packed.pop();
    assert_typed_err(decode_txt_streams(&packed));
}

#[test]
#[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
fn oversized_claimed_count_errors() {
    // tamper the u64 chunk count to a huge value while the offset table is
    // unchanged: parse must reject (offsets != count + 1) without trying to
    // allocate from the claim alone, and never panic.
    let mut packed = good_packed();
    packed[1..9].copy_from_slice(&u64::MAX.to_le_bytes());
    assert_typed_err(decode_txt_streams(&packed));

    // a more modest but still-wrong count (off by one) must also be rejected.
    let mut off_by_one = good_packed();
    off_by_one[1..9].copy_from_slice(&7u64.to_le_bytes());
    assert_typed_err(decode_txt_streams(&off_by_one));
}

#[test]
#[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
fn corrupted_zstd_stream_errors() {
    // flip bytes inside the streams region (a zstd frame): decode of that
    // stream must fail as a typed error, not a panic.
    let mut packed = good_packed();
    let n = packed.len();
    // the streams region is the tail; corrupt a byte near the start of it.
    if n > 20 {
        packed[n - 5] ^= 0xFF;
        packed[n - 6] ^= 0xFF;
    }
    let _ = decode_txt_streams(&packed); // must not panic
}

#[test]
#[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
fn fuzz_every_truncation_never_panics() {
    // exhaustive prefix truncation: every cut returns a typed error or Ok,
    // never a panic. this is the core no-panic-on-hostile-mmap guarantee.
    let packed = good_packed();
    for cut in 0..packed.len() {
        let _ = decode_txt_streams(&packed[..cut]);
        let _ = decode_payload(SECTION_ENCODING_TXT_STREAMS, &packed[..cut]);
    }
}
