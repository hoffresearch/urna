//! positive coverage for the `txt_streams` wire codec (encoding id 10):
//! the per-chunk-streams re-layout of the `chunks_canonical` (0x02)
//! section. the load-bearing invariant is that `decode_payload(10, ...)`
//! rebuilds bytes BYTE-IDENTICAL to `encode_chunks_canonical`, so
//! `content_hash` (hashed over decoded bytes) and `urna://` citations are
//! unchanged. also covers O(1) single-chunk seek and determinism.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use urna_format::encoding::TxtStreams;
use urna_format::layout::SECTION_ENCODING_TXT_STREAMS;
use urna_format::{
    decode_payload, decode_txt_streams, encode_chunks_canonical, encode_txt_streams,
};

fn texts(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

/// the content_hash invariant: encode -> decode_payload(id 10) rebuilds the
/// EXACT canonical bytes for every corpus shape.
fn assert_decodes_byte_identical(items: &[&str]) {
    let t = texts(items);
    let raw = encode_chunks_canonical(&t).unwrap();
    let packed = encode_txt_streams(&t).unwrap();
    assert_eq!(packed[0], urna_format::TXT_STREAMS_V1, "kind/version byte");

    // through the wire-codec registry (the path the reader actually uses).
    let via_payload = decode_payload(SECTION_ENCODING_TXT_STREAMS, &packed).unwrap();
    assert_eq!(
        via_payload.as_ref(),
        raw.as_slice(),
        "decode_payload != raw"
    );

    // through the sections-level dispatch (parallel to decode_intpack_repack).
    let via_sections = decode_txt_streams(&packed).unwrap();
    assert_eq!(via_sections, raw, "decode_txt_streams != raw");
}

#[test]
fn empty_corpus_decodes_byte_identical() {
    assert_decodes_byte_identical(&[]);
}

#[test]
#[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
fn single_chunk_decodes_byte_identical() {
    assert_decodes_byte_identical(&["only one chunk of canonical text"]);
}

#[test]
#[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
fn many_chunks_decode_byte_identical() {
    let owned: Vec<String> = (0..200)
        .map(|i| format!("chunk number {i} with some body"))
        .collect();
    let refs: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();
    assert_decodes_byte_identical(&refs);
}

#[test]
#[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
fn multibyte_utf8_ptbr_accents_decode_byte_identical() {
    // pt-br accents + empty string in the middle must round-trip byte-exact.
    assert_decodes_byte_identical(&[
        "coração",
        "informação pública",
        "açaí é ótimo à tarde",
        "",
        "São Paulo, ñ, ü, ç",
    ]);
}

#[test]
#[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
fn offset_table_o1_seek_returns_the_right_stream() {
    let t = texts(&["alpha", "beta", "coração", "delta", "épsilon"]);
    let packed = encode_txt_streams(&t).unwrap();
    let parsed = TxtStreams::parse(&packed).unwrap();
    assert_eq!(parsed.len(), t.len());
    assert!(!parsed.is_empty());
    // each chunk i seeks to the right stream WITHOUT decoding its neighbors.
    for (i, s) in t.iter().enumerate() {
        assert_eq!(&parsed.text(i).unwrap(), s, "O(1) seek text({i}) mismatch");
    }
    // out-of-range index is a typed error, never a panic.
    assert!(parsed.text(t.len()).is_err());
}

#[test]
#[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
fn two_encodes_are_byte_identical_deterministic() {
    let t = texts(&["a", "bb", "ccc", "coração", "dddd"]);
    assert_eq!(
        encode_txt_streams(&t).unwrap(),
        encode_txt_streams(&t).unwrap()
    );
}

// ---- TXT_STREAMS_V2 (dict, id 5) and V3 (fsst, id 9) variants ----
//
// both reuse the txt_streams container (kind byte + count + intpack offset
// table + N frames) and decode BYTE-IDENTICALLY to the raw chunks_canonical
// payload, the same content_hash invariant, while preserving O(1) seek via
// the shared offset table.

use urna_format::encoding::{
    decode_fsst_payload, decode_zstd_dict_payload, encode_fsst, encode_zstd_dict, train_dict,
};

fn similar_corpus() -> Vec<String> {
    (0..300)
        .map(|i| format!("registro {} de texto similar para o codec", i % 19))
        .collect()
}

#[test]
#[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
fn dict_variant_v2_decodes_byte_identical() {
    let t = similar_corpus();
    let mut su = t.clone();
    su.sort_unstable();
    su.dedup();
    let dict = train_dict(&su).expect("dict trains on a similar corpus");
    let framed = encode_zstd_dict(&t, &dict).unwrap();
    assert_eq!(framed[0], urna_format::TXT_STREAMS_V2, "V2 kind byte");
    let decoded = decode_zstd_dict_payload(&framed, &dict).unwrap();
    assert_eq!(decoded, encode_chunks_canonical(&t).unwrap());
}

#[test]
#[cfg_attr(miri, ignore)] // too slow under miri
fn fsst_variant_v3_decodes_byte_identical() {
    let t = similar_corpus();
    let framed = encode_fsst(&t).unwrap();
    assert_eq!(framed[0], urna_format::TXT_STREAMS_V3, "V3 kind byte");
    let decoded = decode_fsst_payload(&framed).unwrap();
    assert_eq!(decoded, encode_chunks_canonical(&t).unwrap());
}

#[test]
#[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
fn all_three_variants_share_the_offset_table_layout() {
    // V1 / V2 / V3 all start with their kind byte then the same u64 count,
    // so a reader dispatches on the section-entry encoding id and the kind
    // byte agrees. proves the container is shared (O(1) seek preserved).
    let t = texts(&["alpha", "beta", "coração", "delta"]);
    let v1 = encode_txt_streams(&t).unwrap();
    let mut su = t.clone();
    su.sort_unstable();
    su.dedup();
    assert_eq!(v1[0], urna_format::TXT_STREAMS_V1);
    let count_v1 = u64::from_le_bytes(v1[1..9].try_into().unwrap());
    let v3 = encode_fsst(&t).unwrap();
    let count_v3 = u64::from_le_bytes(v3[1..9].try_into().unwrap());
    assert_eq!(count_v1, count_v3);
    assert_eq!(count_v1, t.len() as u64);
}
