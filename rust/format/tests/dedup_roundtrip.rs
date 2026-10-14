//! dedup (section 0x0B) positive coverage at the codec level: the first-seen
//! dedup pass over all-unique / all-duplicate / mixed corpora, the back-ref
//! map round-trips through its serialized form, and expand rebuilds the exact
//! original ordered list (the content_hash invariant: re-expanded bytes equal
//! the non-deduped bytes).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use urna_format::encoding::{decode_dedup_map, dedup, encode_dedup_map, expand_dedup};
use urna_format::sections::encode_chunks_canonical;

fn texts(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

fn assert_rebuilds(corpus: &[String]) {
    let d = dedup(corpus);
    // serialized map round-trips.
    let blob = encode_dedup_map(&d.back_refs);
    assert_eq!(decode_dedup_map(&blob).unwrap(), d.back_refs);
    // expand rebuilds the exact ordered list.
    let back = expand_dedup(&d.unique, &d.back_refs).unwrap();
    assert_eq!(&back, corpus, "expand must rebuild the original order");
    // and therefore the canonical bytes are byte-identical to a non-deduped
    // build (this is what keeps content_hash stable).
    assert_eq!(
        encode_chunks_canonical(&back).unwrap(),
        encode_chunks_canonical(corpus).unwrap()
    );
}

#[test]
fn rebuilds_all_unique() {
    assert_rebuilds(&texts(&["a", "b", "c", "d"]));
}

#[test]
fn rebuilds_all_duplicate() {
    let d = texts(&["same", "same", "same", "same", "same"]);
    assert_rebuilds(&d);
    let pass = dedup(&d);
    assert_eq!(pass.unique.len(), 1, "all-duplicate collapses to one");
}

#[test]
fn rebuilds_mixed() {
    assert_rebuilds(&texts(&["a", "b", "a", "c", "b", "a", "d", "d"]));
}

#[test]
fn rebuilds_empty() {
    assert_rebuilds(&texts(&[]));
}

#[test]
fn rebuilds_multibyte_utf8() {
    assert_rebuilds(&texts(&[
        "cora\u{e7}\u{e3}o",
        "informa\u{e7}\u{e3}o",
        "cora\u{e7}\u{e3}o",
        "",
        "",
    ]));
}

#[test]
fn dedup_shrinks_repeated_corpus() {
    // a corpus that is 4x the same boilerplate dedups to a quarter the unique
    // entries: the dedup-as-compression mechanism (nix store-optimise).
    let block = "BOILERPLATE LEGAL FOOTER REPETIDO".to_string();
    let corpus: Vec<String> = (0..400).map(|_| block.clone()).collect();
    let d = dedup(&corpus);
    assert_eq!(d.unique.len(), 1);
    assert_eq!(d.back_refs.len(), 400);
    // the unique pool stored once is far smaller than storing all 400 copies.
    let pool = encode_chunks_canonical(&d.unique).unwrap();
    let full = encode_chunks_canonical(&corpus).unwrap();
    assert!(pool.len() < full.len() / 100);
}
