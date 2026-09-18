//! Dual integrity guarantees: section checksum (physical bytes) and
//! `content_hash` (decoded bytes) catch different failure classes.
//!
//! Phase 0 wired `decoded_section()` to feed `content_hash` through the
//! decompressed bytes when a section is zstd-encoded. This test files
//! that decision in three concrete scenarios:
//!
//! - **Encoding-invariant content_hash**: a corpus stored with zstd
//!   text encoding and the same corpus stored raw must share
//!   `content_hash` (so citations are stable across wire encodings)
//!   but differ in `file_hash` (different bytes on disk).
//! - **Section checksums are physical, not logical**: A raw and a zstd
//!   variant with the same logical content have *different* section
//!   checksums for every text-heavy section. The checksum is what
//!   catches a flipped byte after the fact, regardless of encoding.
//! - **Different content yields different content_hash**: a corpus
//!   where one chunk's canonical text is changed must produce a
//!   different `content_hash` than the original — even though both
//!   files individually pass `validate()`.
//!
//! Together: physical and semantic guarantees are independent.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use urna_format::layout::{
    SECTION_CHUNK_IDS, SECTION_CHUNKS_CANONICAL, SECTION_CHUNKS_ORIGINAL_SPANS,
    SECTION_ENCODING_INTPACK, SECTION_ENCODING_RAW, SECTION_ENCODING_ZSTD, SECTION_PROVENANCE,
    SECTION_SEARCH_CONTRACT,
};
use urna_format::manifest::{Capabilities, Manifest};
use urna_format::writer::{SectionEncoding, UrnaFileBuilder};
use urna_format::{ChunkInput, UrnaView};

fn manifest_for(n: u64) -> Manifest {
    Manifest {
        format_version: 1,
        schema_version: 1,
        embedding_model: "demo".into(),
        embedding_dim: 4,
        n_chunks: n,
        dtype: "float32".into(),
        metric: "ip".into(),
        score_type: "cosine".into(),
        normalize: "l2".into(),
        index_type: "exact".into(),
        rerank_policy: "none".into(),
        model_hash: format!("sha256:{}", "0".repeat(64)),
        chunker_version: "demo-chunker/1".into(),
        capabilities: Capabilities {
            supports_exact: true,
            supports_reproducible_build: true,
            supports_ann: false,
            supports_bm25: false,
            supports_citations: true,
        },
        title: None,
        version: None,
        created: None,
        description: None,
        authors: None,
        license: None,
        mrl_dim: None,
        full_dim: None,
        capabilities_ext: None,
        extra: Default::default(),
    }
}

fn chunks(texts: &[&str]) -> Vec<ChunkInput> {
    texts
        .iter()
        .enumerate()
        .map(|(i, t)| ChunkInput {
            canonical_text: (*t).to_string(),
            source_uri: "doc.txt".into(),
            byte_start: (i * 100) as u64,
            byte_end: (i * 100 + t.len()) as u64,
            embedding: {
                let mut v = vec![0.0; 4];
                v[i % 4] = 1.0;
                v
            },
        })
        .collect()
}

fn build(text_encoding: SectionEncoding, texts: &[&str]) -> Vec<u8> {
    UrnaFileBuilder::new(manifest_for(texts.len() as u64))
        .text_encoding(text_encoding)
        .reproducible(true)
        .add_chunks(chunks(texts))
        .build_bytes()
        .unwrap()
}

#[test]
#[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
fn raw_and_zstd_share_content_hash_but_not_file_hash() {
    let texts = &[
        "primeiro paragrafo",
        "segundo paragrafo",
        "terceiro paragrafo",
    ];
    let raw = build(SectionEncoding::Raw, texts);
    let zst = build(SectionEncoding::Zstd, texts);

    let v_raw = UrnaView::from_bytes(&raw).unwrap();
    let v_zst = UrnaView::from_bytes(&zst).unwrap();

    assert_eq!(
        v_raw.content_hash_hex().unwrap(),
        v_zst.content_hash_hex().unwrap(),
        "content_hash must be identical across wire encodings"
    );
    assert_ne!(
        v_raw.file_hash_hex(),
        v_zst.file_hash_hex(),
        "file_hash must differ when wire encoding differs"
    );
}

#[test]
#[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
fn section_checksums_track_physical_bytes_not_decoded() {
    // Same logical content, two encodings. Physical checksums for the
    // text-heavy sections must differ (different bytes on disk =
    // different SHA-256 over those bytes). This is what catches a
    // bit-flip even when the logical content is unchanged.
    let texts = &["alfa", "beta", "gama", "delta"];
    let raw = build(SectionEncoding::Raw, texts);
    let zst = build(SectionEncoding::Zstd, texts);

    let v_raw = UrnaView::from_bytes(&raw).unwrap();
    let v_zst = UrnaView::from_bytes(&zst).unwrap();

    for sid in [
        SECTION_CHUNKS_CANONICAL,
        SECTION_CHUNKS_ORIGINAL_SPANS,
        SECTION_PROVENANCE,
        SECTION_SEARCH_CONTRACT,
    ] {
        let raw_entry = v_raw
            .section_table
            .iter()
            .find(|e| e.section_id == sid)
            .unwrap();
        let zst_entry = v_zst
            .section_table
            .iter()
            .find(|e| e.section_id == sid)
            .unwrap();
        assert_eq!(raw_entry.encoding, SECTION_ENCODING_RAW);
        // under the compressed preset every text section is compressed: the
        // text/provenance/contract sections zstd, while spans takes the
        // smaller of zstd or the intpack repack (deduped uri pool + bitpacked
        // offsets). either way it is physically distinct from the raw variant.
        let compressed_enc = if sid == SECTION_CHUNKS_ORIGINAL_SPANS {
            matches!(
                zst_entry.encoding,
                SECTION_ENCODING_ZSTD | SECTION_ENCODING_INTPACK
            )
        } else {
            zst_entry.encoding == SECTION_ENCODING_ZSTD
        };
        assert!(
            compressed_enc,
            "section 0x{:02x} compressed variant has unexpected encoding {}",
            sid, zst_entry.encoding
        );
        assert_ne!(
            raw_entry.checksum, zst_entry.checksum,
            "physical checksums must differ between raw and compressed for section 0x{:02x}",
            sid
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
fn compressed_preset_repacks_chunk_ids_with_intpack() {
    // chunk_ids are high-entropy sha-256, so the 32-raw intpack repack
    // always beats the ascii form. it must be chosen under the compressed
    // preset, shrink the section, and leave content_hash unchanged.
    let texts = &["alfa", "beta", "gama"];
    let raw = build(SectionEncoding::Raw, texts);
    let zst = build(SectionEncoding::Zstd, texts);
    let v_raw = UrnaView::from_bytes(&raw).unwrap();
    let v_zst = UrnaView::from_bytes(&zst).unwrap();

    let raw_ci = v_raw
        .section_table
        .iter()
        .find(|e| e.section_id == SECTION_CHUNK_IDS)
        .unwrap();
    let zst_ci = v_zst
        .section_table
        .iter()
        .find(|e| e.section_id == SECTION_CHUNK_IDS)
        .unwrap();
    assert_eq!(raw_ci.encoding, SECTION_ENCODING_RAW);
    assert_eq!(zst_ci.encoding, SECTION_ENCODING_INTPACK);
    assert!(zst_ci.size < raw_ci.size, "intpack must shrink chunk_ids");
    assert_eq!(
        v_raw.content_hash_hex().unwrap(),
        v_zst.content_hash_hex().unwrap(),
        "the chunk_ids repack must not move content_hash"
    );
}

#[test]
fn content_hash_diverges_when_canonical_text_changes() {
    let original = &["alfa", "beta", "gama"];
    let mutated = &["alfa", "BETA-CHANGED", "gama"];

    let a = build(SectionEncoding::Raw, original);
    let b = build(SectionEncoding::Raw, mutated);

    let va = UrnaView::from_bytes(&a).unwrap();
    let vb = UrnaView::from_bytes(&b).unwrap();

    // Both pass physical validation: section checksums are over each
    // file's own bytes, both internally consistent.
    va.validate_embeddings_values().unwrap();
    vb.validate_embeddings_values().unwrap();

    let ha = va.content_hash_hex().unwrap();
    let hb = vb.content_hash_hex().unwrap();
    assert_ne!(
        ha, hb,
        "content_hash must change when canonical text changes (got {} for both)",
        ha
    );
}

#[test]
#[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
fn three_files_three_distinct_classifications() {
    // A complete proof: file A (raw, X), file B (zstd, X), file C (raw, Y).
    //   A.content_hash == B.content_hash (semantic equivalence)
    //   A.content_hash != C.content_hash (semantic divergence)
    //   A.file_hash != B.file_hash != C.file_hash (all three differ physically)
    let x = &["alfa", "beta", "gama"];
    let y = &["alfa", "beta-different", "gama"];
    let a = build(SectionEncoding::Raw, x);
    let b = build(SectionEncoding::Zstd, x);
    let c = build(SectionEncoding::Raw, y);

    let va = UrnaView::from_bytes(&a).unwrap();
    let vb = UrnaView::from_bytes(&b).unwrap();
    let vc = UrnaView::from_bytes(&c).unwrap();

    let ha_c = va.content_hash_hex().unwrap();
    let hb_c = vb.content_hash_hex().unwrap();
    let hc_c = vc.content_hash_hex().unwrap();
    assert_eq!(ha_c, hb_c);
    assert_ne!(ha_c, hc_c);

    let ha_f = va.file_hash_hex();
    let hb_f = vb.file_hash_hex();
    let hc_f = vc.file_hash_hex();
    assert_ne!(ha_f, hb_f);
    assert_ne!(ha_f, hc_f);
    assert_ne!(hb_f, hc_f);
}
