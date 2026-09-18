//! multimodal space band integration (0x15 + 0x20):
//!
//! - search_space("vision") scores the vision band with real exact cosine
//!   and returns the expected order.
//! - ISOLATION: the text path (search) only reads the canonical 0x04 slab
//!   and the space path only reads the band; a text-dim query against the
//!   vision space fails with DimensionMismatch, a wrong expected
//!   model_hash fails with SpaceModelMismatch, an unknown space fails
//!   with SpaceNotFound.
//! - content_hash equality: a fixed corpus WITH vs WITHOUT the multimodal
//!   sections has IDENTICAL content_hash (citations stay stable).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use std::path::PathBuf;
use urna_format::manifest::Manifest;
use urna_format::writer::UrnaFileBuilder;
use urna_format::{
    ChunkInput, SECTION_ENCODING_RAW, SPACE_DTYPE_F32, SpaceEntry, UrnaView, encode_space_table,
};
use urna_runtime::{MmapUrnaFile, RuntimeError};

const TEXT_DIM: usize = 4;
const VIS_DIM: usize = 2;
const VIS_HASH: &str = "sha256:vis111";

fn tmp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(name);
    p
}

fn vision_vectors() -> Vec<[f32; VIS_DIM]> {
    // chunk 0 -> +x, chunk 1 -> +y, chunk 2 -> -x (normalized rows).
    vec![[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
}

fn build_multimodal(path: &PathBuf, with_spaces: bool) {
    let n = 3usize;
    let manifest = Manifest {
        embedding_model: "demo".into(),
        embedding_dim: TEXT_DIM as u32,
        n_chunks: n as u64,
        chunker_version: "demo-chunker/1".into(),
        model_hash: format!("sha256:{}", "0".repeat(64)),
        ..Default::default()
    };
    let mut builder = UrnaFileBuilder::new(manifest).reproducible(true);
    for i in 0..n {
        let mut emb = vec![0.0f32; TEXT_DIM];
        emb[i % TEXT_DIM] = 1.0;
        builder = builder.add_chunk(ChunkInput {
            canonical_text: format!("chunk {i}"),
            source_uri: "doc.txt".into(),
            byte_start: (i * 10) as u64,
            byte_end: ((i + 1) * 10) as u64,
            embedding: emb,
        });
    }
    if with_spaces {
        let entries = vec![SpaceEntry {
            space_index: 1,
            name: "vision".into(),
            dim: VIS_DIM as u32,
            dtype: SPACE_DTYPE_F32,
            model_hash: VIS_HASH.into(),
            n_vectors: n as u64,
        }];
        builder = builder.space_table(encode_space_table(&entries).unwrap());
        let mut band = Vec::new();
        for v in vision_vectors() {
            for x in v {
                band.extend_from_slice(&x.to_le_bytes());
            }
        }
        builder = builder.space_band(1, SECTION_ENCODING_RAW, band);
    }
    builder.write_to_path(path).unwrap();
}

#[test]
fn search_space_scores_the_vision_band_exactly() {
    let path = tmp_path("rt_space.urna");
    let _ = std::fs::remove_file(&path);
    build_multimodal(&path, true);

    let rt = MmapUrnaFile::open(&path).unwrap();
    assert!(rt.has_spaces());
    assert_eq!(rt.space_names(), vec!["vision"]);

    // query +x: chunk 0 (cos 1) first, chunk 2 (cos -1) last.
    let res = rt
        .search_space("vision", &[1.0, 0.0], 3, Some(VIS_HASH))
        .unwrap();
    assert_eq!(res.hits.len(), 3);
    assert_eq!(res.index_type, "space");
    assert_eq!(res.recall, 1.0);
    assert!((res.hits[0].score - 1.0).abs() < 1e-6);
    assert!(res.hits[0].chunk_id.ends_with("chunk 0") || res.hits[0].offset_start == 0);
    assert!(res.hits[2].score < -0.99);
    // model gate: the matching hash passes (above), a wrong one errors.
    let err = rt
        .search_space("vision", &[1.0, 0.0], 3, Some("sha256:wrong"))
        .unwrap_err();
    assert!(matches!(err, RuntimeError::SpaceModelMismatch { .. }));

    let _ = std::fs::remove_file(&path);
}

#[test]
fn text_and_vision_spaces_are_isolated() {
    let path = tmp_path("rt_space_iso.urna");
    let _ = std::fs::remove_file(&path);
    build_multimodal(&path, true);
    let rt = MmapUrnaFile::open(&path).unwrap();

    // the text path scores the 0x04 slab: a text +x query hits chunk 0
    // with cos 1 against TEXT_DIM vectors, unaware of the vision band.
    let text = rt.search(&[1.0, 0.0, 0.0, 0.0], 3).unwrap();
    assert_eq!(text.index_type, "exact");
    assert!((text.hits[0].score - 1.0).abs() < 1e-6);
    assert_eq!(text.hits[0].offset_start, 0);

    // a TEXT_DIM query against the vision band fails loudly (dim gate),
    // so a text query can never be scored against vision by accident.
    let err = rt
        .search_space("vision", &[1.0, 0.0, 0.0, 0.0], 3, None)
        .unwrap_err();
    assert!(matches!(
        err,
        RuntimeError::DimensionMismatch {
            expected: VIS_DIM,
            got: TEXT_DIM
        }
    ));
    // and a vision-dim query against the text path fails the same way.
    assert!(rt.search(&[1.0, 0.0], 3).is_err());

    // unknown space: typed error, never a silent fallback to text.
    let err = rt.search_space("audio", &[1.0, 0.0], 3, None).unwrap_err();
    assert!(matches!(err, RuntimeError::SpaceNotFound(_)));

    let _ = std::fs::remove_file(&path);
}

#[test]
fn multimodal_sections_do_not_change_content_hash() {
    let mut a = std::env::temp_dir();
    a.push("space_ch_without.urna");
    let mut b = std::env::temp_dir();
    b.push("space_ch_with.urna");
    let _ = std::fs::remove_file(&a);
    let _ = std::fs::remove_file(&b);
    build_multimodal(&a, false);
    build_multimodal(&b, true);

    let da = std::fs::read(&a).unwrap();
    let db = std::fs::read(&b).unwrap();
    let va = UrnaView::from_bytes(&da).unwrap();
    let vb = UrnaView::from_bytes(&db).unwrap();
    assert_eq!(
        va.content_hash_hex().unwrap(),
        vb.content_hash_hex().unwrap(),
        "the 0x15/0x20 multimodal sections must NOT change content_hash"
    );
    assert_ne!(va.file_hash_hex(), vb.file_hash_hex());
    let ext = vb.manifest.capabilities_ext.as_ref().unwrap();
    assert_eq!(ext.supports_multimodal, Some(true));

    // a corpus without the capability has no spaces and a typed error.
    let rt = MmapUrnaFile::open(&a).unwrap();
    assert!(!rt.has_spaces());
    assert!(rt.space_names().is_empty());
    assert!(rt.search_space("vision", &[1.0, 0.0], 1, None).is_err());

    let _ = std::fs::remove_file(&a);
    let _ = std::fs::remove_file(&b);
}
