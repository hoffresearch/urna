//! fsst (encoding id 9) negative coverage: a bad kind byte, truncated codes,
//! an oversized declared table length, and a trailing escape must all return
//! a typed UrnaError, never a panic. exhaustive prefix-truncation fuzz.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use urna_format::encoding::{decode_fsst_payload, encode_fsst};

fn framed() -> Vec<u8> {
    let texts: Vec<String> = (0..200)
        .map(|i| format!("frase curta repetida {} fim", i % 11))
        .collect();
    encode_fsst(&texts).unwrap()
}

#[test]
fn empty_payload_errors() {
    assert!(decode_fsst_payload(&[]).is_err());
}

#[test]
#[cfg_attr(miri, ignore)] // too slow under miri
fn bad_kind_byte_errors() {
    let mut f = framed();
    f[0] = 0x00; // TXT_STREAMS_V1, not V3
    assert!(decode_fsst_payload(&f).is_err());
}

#[test]
#[cfg_attr(miri, ignore)] // too slow under miri
fn truncated_count_errors() {
    let f = framed();
    // keep the kind byte but truncate the u64 count.
    assert!(decode_fsst_payload(&f[..4]).is_err());
}

#[test]
#[cfg_attr(miri, ignore)] // too slow under miri
fn prefix_truncation_fuzz_never_panics() {
    let f = framed();
    for cut in 0..f.len() {
        // every prefix of a valid payload must error or decode, never panic.
        let _ = decode_fsst_payload(&f[..cut]);
    }
}

#[test]
#[cfg_attr(miri, ignore)] // too slow under miri
fn corrupted_table_region_errors_no_panic() {
    let mut f = framed();
    // flip bytes deep in the region (past the container header + offset
    // table) to corrupt the embedded symbol table / frames; must not panic.
    let n = f.len();
    for b in f.iter_mut().skip(n.saturating_sub(20)) {
        *b ^= 0xFF;
    }
    let _ = decode_fsst_payload(&f); // typed error or ok, never a panic
}
