//! fsst (encoding id 9) positive coverage: deterministic table build,
//! encode -> decode byte-identical to the raw chunks_canonical payload over
//! empty/single/many/multibyte-utf8 pt-br corpora and the escape path, two
//! builds byte-identical.

// not under miri: the fsst symbol-table build over whole corpora takes minutes and gigabytes under miri; the fsst decoder is covered natively and by negative_fsst under miri.
#![cfg(not(miri))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use urna_format::encoding::{decode_fsst_payload, encode_fsst};
use urna_format::sections::encode_chunks_canonical;

fn texts(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

fn assert_byte_identical(items: &[String]) {
    let framed = encode_fsst(items).unwrap();
    let decoded = decode_fsst_payload(&framed).unwrap();
    let raw = encode_chunks_canonical(items).unwrap();
    assert_eq!(
        decoded, raw,
        "fsst decode must rebuild the raw canonical bytes"
    );
}

#[test]
fn byte_identical_across_corpora() {
    assert_byte_identical(&texts(&[]));
    assert_byte_identical(&texts(&["only one"]));
    assert_byte_identical(&texts(&["primeiro", "segundo", "terceiro"]));
    // multibyte utf-8 (pt-br accents) plus an empty string.
    assert_byte_identical(&texts(&[
        "cora\u{e7}\u{e3}o",
        "informa\u{e7}\u{e3}o",
        "a\u{e7}a\u{ed} \u{e9} \u{f3}timo",
        "",
    ]));
}

#[test]
fn many_short_similar_chunks_roundtrip() {
    let many: Vec<String> = (0..1000)
        .map(|i| format!("registro {} sobre o mesmo assunto repetido", i % 23))
        .collect();
    assert_byte_identical(&many);
}

#[test]
fn escape_path_arbitrary_bytes() {
    // strings whose bytes are unlikely to be table symbols force the 0xFF
    // escape path; they must still round-trip exactly. (valid utf-8 only,
    // since the canonical section is utf-8.)
    let weird = texts(&["\u{0}\u{1}\u{2}\u{7f}", "\u{ffff}\u{10000}", "\t\n\r"]);
    assert_byte_identical(&weird);
}

#[test]
fn deterministic_two_builds_identical() {
    let t: Vec<String> = (0..300)
        .map(|i| format!("texto determin\u{ed}stico {}", i % 7))
        .collect();
    let a = encode_fsst(&t).unwrap();
    let b = encode_fsst(&t).unwrap();
    assert_eq!(
        a, b,
        "the symbol table + frames are a pure function of input"
    );
}
