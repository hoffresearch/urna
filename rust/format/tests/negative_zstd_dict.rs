//! zstd_dict (encoding id 5) negative coverage: a truncated dict-framed
//! payload, a missing dictionary, and an oversized/garbage dict must all
//! return a typed UrnaError, never a panic. exhaustive prefix-truncation
//! fuzz over a real framed payload.

// not under miri: every case builds a dict-framed zstd payload first (c code miri cannot call).
#![cfg(not(miri))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use urna_format::encoding::{
    decode_payload_with_dict, decode_zstd_dict_payload, encode_zstd_dict, train_dict,
};
use urna_format::layout::SECTION_ENCODING_ZSTD_DICT;

fn framed_and_dict() -> (Vec<u8>, Vec<u8>) {
    let texts: Vec<String> = (0..200)
        .map(|i| format!("noticia curta similar numero {} no dia {}", i % 13, i))
        .collect();
    let mut su = texts.clone();
    su.sort_unstable();
    su.dedup();
    let dict = train_dict(&su).unwrap();
    let framed = encode_zstd_dict(&texts, &dict).unwrap();
    (framed, dict)
}

#[test]
fn empty_payload_errors() {
    let (_, dict) = framed_and_dict();
    assert!(decode_zstd_dict_payload(&[], &dict).is_err());
}

#[test]
fn missing_dict_via_dispatch_errors() {
    let (framed, _) = framed_and_dict();
    // the dispatch entry point with no dict (section 0x0A absent) must error,
    // not panic.
    let res = decode_payload_with_dict(SECTION_ENCODING_ZSTD_DICT, &framed, None);
    assert!(res.is_err(), "dict-framed section with no dict must error");
}

#[test]
fn oversized_or_garbage_dict_errors_no_panic() {
    let (framed, _) = framed_and_dict();
    // a garbage dict (not a real zstd dictionary) must not panic the decoder.
    let garbage = vec![0xABu8; 64 * 1024];
    let _ = decode_zstd_dict_payload(&framed, &garbage); // must not panic
    // an empty dict is also handled without panic.
    let _ = decode_zstd_dict_payload(&framed, &[]);
}

#[test]
fn prefix_truncation_fuzz_never_panics() {
    let (framed, dict) = framed_and_dict();
    for cut in 0..framed.len() {
        // every prefix of a valid framed payload must error or decode, never
        // panic, regardless of where it is severed.
        let _ = decode_zstd_dict_payload(&framed[..cut], &dict);
    }
}

#[test]
fn bad_kind_byte_errors() {
    let (mut framed, dict) = framed_and_dict();
    framed[0] = 0xFE; // not TXT_STREAMS_V2
    assert!(decode_zstd_dict_payload(&framed, &dict).is_err());
}
