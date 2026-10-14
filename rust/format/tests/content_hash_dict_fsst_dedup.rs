//! the load-bearing content_hash invariant for the three text levers: a
//! file whose chunks_canonical is dict-framed (5), fsst-framed (9), or
//! deduped (0x0B) must carry the SAME content_hash as its raw-text twin, so
//! `urna://` citations are unchanged. proven two ways:
//!
//! 1. file level: a redundant corpus where the dedup candidate wins yields a
//!    file with the 0x0B aux section AND the same content_hash as the raw
//!    twin (the writer auto-chooses; the test asserts the lever engaged).
//! 2. codec level: dict/fsst/dedup decode byte-identically to the raw
//!    chunks_canonical payload, so the content_hash preimage is unchanged
//!    regardless of which codec the chooser picks.

// not under miri: the twins are built through the zstd dictionary codec (c code miri cannot call) and the fsst table build is too heavy under miri.
#![cfg(not(miri))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use urna_format::encoding::{
    decode_dedup_map, decode_fsst_payload, decode_zstd_dict_payload, dedup, encode_dedup_map,
    encode_fsst, encode_zstd_dict, expand_dedup, train_dict,
};
use urna_format::layout::{SECTION_CHUNKS_CANONICAL, SECTION_DEDUP_MAP, SECTION_ENCODING_ZSTD};
use urna_format::manifest::{Capabilities, Manifest};
use urna_format::sections::encode_chunks_canonical;
use urna_format::writer::{SectionEncoding, UrnaFileBuilder};
use urna_format::{ChunkInput, UrnaView};

fn manifest_for(n: u64) -> Manifest {
    Manifest {
        embedding_model: "demo".into(),
        embedding_dim: 4,
        n_chunks: n,
        model_hash: format!("sha256:{}", "0".repeat(64)),
        chunker_version: "demo-chunker/1".into(),
        capabilities: Capabilities::default(),
        ..Default::default()
    }
}

fn build(enc: SectionEncoding, texts: &[String]) -> Vec<u8> {
    let chunks: Vec<ChunkInput> = texts
        .iter()
        .enumerate()
        .map(|(i, t)| ChunkInput {
            canonical_text: t.clone(),
            source_uri: format!("doc://{}", i % 5),
            byte_start: 0,
            byte_end: t.len() as u64,
            embedding: {
                let mut v = vec![0.0f32; 4];
                v[i % 4] = 1.0;
                v
            },
        })
        .collect();
    UrnaFileBuilder::new(manifest_for(texts.len() as u64))
        .text_encoding(enc)
        .reproducible(true)
        .add_chunks(chunks)
        .build_bytes()
        .unwrap()
}

/// a corpus of many far-apart repeated blocks that exceed a single zstd
/// window, so dedup-before-zstd genuinely beats single-frame zstd and the
/// 0x0B aux section is emitted. kept just large enough to clear the window
/// without making the build slow.
fn dedup_winning_corpus() -> Vec<String> {
    // 60 distinct ~8 KiB pseudo-random-but-deterministic blocks (zstd cannot
    // shrink the unique residue much), each repeated 40x and interleaved so
    // duplicates are spread across > 16 MiB, beyond an easy LZ window reach.
    let uniques: Vec<String> = (0..60)
        .map(|i| {
            (0..8192)
                .map(|j| char::from(b'a' + (((i * 7919 + j * 31) % 26) as u8)))
                .collect::<String>()
        })
        .collect();
    let mut out = Vec::new();
    for _ in 0..40 {
        for u in &uniques {
            out.push(u.clone());
        }
    }
    out
}

#[test]
fn dedup_file_shares_content_hash_with_raw_twin() {
    let texts = dedup_winning_corpus();
    let raw = build(SectionEncoding::Raw, &texts);
    let zst = build(SectionEncoding::Zstd, &texts);

    let v_raw = UrnaView::from_bytes(&raw).unwrap();
    let v_zst = UrnaView::from_bytes(&zst).unwrap();

    // the dedup lever must have engaged on this far-apart-repeats corpus: the
    // 0x0B aux section is present and chunks_canonical holds the unique pool.
    assert!(
        v_zst.entry(SECTION_DEDUP_MAP).is_ok(),
        "dedup candidate must win on a far-apart heavily-repeated corpus"
    );
    let zst_canonical = v_zst.entry(SECTION_CHUNKS_CANONICAL).unwrap();
    assert_eq!(zst_canonical.encoding, SECTION_ENCODING_ZSTD);

    // the invariant: same content_hash as the raw twin (citations stable),
    // different file_hash (different bytes on disk).
    assert_eq!(
        v_raw.content_hash_hex().unwrap(),
        v_zst.content_hash_hex().unwrap(),
        "deduped file must share content_hash with its raw twin"
    );
    assert_ne!(v_raw.file_hash_hex(), v_zst.file_hash_hex());

    // and the decoded chunks_canonical bytes are byte-identical (the 0x0B
    // re-expansion rebuilt the full ordered byte stream).
    assert_eq!(
        v_raw.decoded_section(SECTION_CHUNKS_CANONICAL).unwrap(),
        v_zst.decoded_section(SECTION_CHUNKS_CANONICAL).unwrap()
    );
}

#[test]
fn dict_codec_preserves_content_hash_preimage() {
    // codec-level: a dict-framed payload decodes to the exact raw canonical
    // bytes, so the content_hash preimage (decoded bytes) is unchanged.
    let texts: Vec<String> = (0..300)
        .map(|i| format!("noticia similar e curta numero {} no dia {}", i % 19, i))
        .collect();
    let mut su = texts.clone();
    su.sort_unstable();
    su.dedup();
    let dict = train_dict(&su).unwrap();
    let framed = encode_zstd_dict(&texts, &dict).unwrap();
    assert_eq!(
        decode_zstd_dict_payload(&framed, &dict).unwrap(),
        encode_chunks_canonical(&texts).unwrap()
    );
}

#[test]
fn fsst_codec_preserves_content_hash_preimage() {
    let texts: Vec<String> = (0..300)
        .map(|i| format!("frase {} repetida e curta", i % 11))
        .collect();
    let framed = encode_fsst(&texts).unwrap();
    assert_eq!(
        decode_fsst_payload(&framed).unwrap(),
        encode_chunks_canonical(&texts).unwrap()
    );
}

#[test]
fn dedup_codec_preserves_content_hash_preimage() {
    let texts: Vec<String> = (0..300).map(|i| format!("bloco {}", i % 7)).collect();
    let d = dedup(&texts);
    let map = encode_dedup_map(&d.back_refs);
    let refs = decode_dedup_map(&map).unwrap();
    let full = expand_dedup(&d.unique, &refs).unwrap();
    assert_eq!(
        encode_chunks_canonical(&full).unwrap(),
        encode_chunks_canonical(&texts).unwrap()
    );
}
