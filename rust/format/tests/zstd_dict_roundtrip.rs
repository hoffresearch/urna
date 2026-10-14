//! zstd_dict (encoding id 5) positive coverage: train a deterministic dict
//! over sorted samples, encode -> decode byte-identical to the raw
//! chunks_canonical payload (the content_hash invariant), two trains over
//! the same sorted samples give a byte-identical dict, and a wrong-dict
//! decode errors cleanly (never a panic).

// not under miri: zstd dictionary training and framing are c code miri cannot call.
#![cfg(not(miri))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use urna_format::encoding::{decode_zstd_dict_payload, encode_zstd_dict, train_dict};
use urna_format::sections::encode_chunks_canonical;

fn corpus() -> Vec<String> {
    // many short, mutually-similar pt-br chunks: the dict regime. >= 8 so the
    // trainer is offered material.
    (0..400)
        .map(|i| {
            format!(
                "Bras\u{ed}lia, {} - segundo informa\u{e7}\u{e3}o do dia, o caso {} segue em an\u{e1}lise.",
                i % 17,
                i
            )
        })
        .collect()
}

fn sorted_unique(texts: &[String]) -> Vec<String> {
    let mut v = texts.to_vec();
    v.sort_unstable();
    v.dedup();
    v
}

#[test]
fn encode_decode_byte_identical() {
    let texts = corpus();
    let dict = train_dict(&sorted_unique(&texts)).expect("dict trains on a similar corpus");
    let framed = encode_zstd_dict(&texts, &dict).unwrap();
    let decoded = decode_zstd_dict_payload(&framed, &dict).unwrap();
    let raw = encode_chunks_canonical(&texts).unwrap();
    assert_eq!(
        decoded, raw,
        "dict decode must rebuild the raw canonical bytes"
    );
}

#[test]
fn multibyte_utf8_pt_br_roundtrips() {
    // accents and an empty string must survive the dict path byte-for-byte.
    let mut texts: Vec<String> = (0..40)
        .map(|i| {
            format!(
                "cora\u{e7}\u{e3}o e informa\u{e7}\u{e3}o a\u{e7}a\u{ed} {}",
                i
            )
        })
        .collect();
    texts.push(String::new());
    let dict = train_dict(&sorted_unique(&texts)).expect("dict trains");
    let framed = encode_zstd_dict(&texts, &dict).unwrap();
    let decoded = decode_zstd_dict_payload(&framed, &dict).unwrap();
    assert_eq!(decoded, encode_chunks_canonical(&texts).unwrap());
}

#[test]
fn same_sorted_samples_give_byte_identical_dict() {
    let texts = corpus();
    let su = sorted_unique(&texts);
    let d1 = train_dict(&su).unwrap();
    let d2 = train_dict(&su).unwrap();
    assert_eq!(d1, d2, "ZDICT is a pure function of the sorted samples");
    // and the whole framed payload is deterministic too.
    assert_eq!(
        encode_zstd_dict(&texts, &d1).unwrap(),
        encode_zstd_dict(&texts, &d2).unwrap()
    );
}

#[test]
fn wrong_dict_decode_errors_cleanly() {
    let texts = corpus();
    let dict = train_dict(&sorted_unique(&texts)).unwrap();
    let framed = encode_zstd_dict(&texts, &dict).unwrap();
    // a dict trained on unrelated bytes is the wrong dict: decode must return
    // a typed error, not panic and not silently produce garbage.
    let other: Vec<String> = (0..40)
        .map(|i| format!("completely unrelated english sentence {}", i))
        .collect();
    let wrong = train_dict(&sorted_unique(&other)).unwrap();
    let res = decode_zstd_dict_payload(&framed, &wrong);
    assert!(res.is_err(), "wrong-dict decode must error");
}

#[test]
fn train_declines_on_tiny_corpus() {
    // too few samples to train a useful dict: the build simply does not offer
    // the dict candidate (None), never an error.
    assert!(train_dict(&[]).is_none());
    assert!(train_dict(&["a".into(), "b".into()]).is_none());
}
