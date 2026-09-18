//! Integration-style tests for the builder. Exercise the full
//! build_bytes flow + reader roundtrip, plus the encoding / dtype
//! invariants the runtime relies on.

use super::*;
use crate::chunk::ChunkInput;
use crate::error::UrnaError;
use crate::layout::{
    SECTION_EMBEDDINGS, SECTION_ENCODING_FLOAT16, SECTION_ENCODING_INT8, SECTION_ENCODING_RAW,
};
use crate::manifest::Manifest;
use crate::reader::UrnaView;

fn good_manifest() -> Manifest {
    Manifest {
        embedding_model: "demo".into(),
        embedding_dim: 4,
        n_chunks: 1,
        chunker_version: "demo-chunker/1".into(),
        model_hash: format!("sha256:{}", "0".repeat(64)),
        ..Default::default()
    }
}

fn one_chunk() -> ChunkInput {
    ChunkInput {
        canonical_text: "hello world".into(),
        source_uri: "doc.txt".into(),
        byte_start: 0,
        byte_end: 11,
        embedding: vec![1.0, 0.0, 0.0, 0.0],
    }
}

#[test]
fn deterministic_build() {
    let b1 = UrnaFileBuilder::new(good_manifest())
        .add_chunk(one_chunk())
        .build_bytes()
        .unwrap();
    let b2 = UrnaFileBuilder::new(good_manifest())
        .add_chunk(one_chunk())
        .build_bytes()
        .unwrap();
    assert_eq!(b1, b2);
}

#[test]
fn reproducible_mode_overrides_created() {
    let mut m1 = good_manifest();
    m1.created = Some("2026-04-27T13:30:00Z".into());
    let mut m2 = good_manifest();
    m2.created = Some("2030-12-31T23:59:59Z".into());

    let b1 = UrnaFileBuilder::new(m1)
        .reproducible(true)
        .add_chunk(one_chunk())
        .build_bytes()
        .unwrap();
    let b2 = UrnaFileBuilder::new(m2)
        .reproducible(true)
        .add_chunk(one_chunk())
        .build_bytes()
        .unwrap();
    assert_eq!(b1, b2, "reproducible builds must be byte-equal");

    let view = UrnaView::from_bytes(&b1).unwrap();
    assert_eq!(view.manifest.created.as_deref(), Some(REPRODUCIBLE_CREATED));
}

#[test]
fn non_reproducible_preserves_created() {
    let mut m = good_manifest();
    m.created = Some("2026-04-27T13:30:00Z".into());
    let bytes = UrnaFileBuilder::new(m.clone())
        .add_chunk(one_chunk())
        .build_bytes()
        .unwrap();
    let view = UrnaView::from_bytes(&bytes).unwrap();
    assert_eq!(
        view.manifest.created.as_deref(),
        Some("2026-04-27T13:30:00Z")
    );
}

#[test]
fn rejects_chunk_count_mismatch() {
    let mut m = good_manifest();
    m.n_chunks = 2;
    let err = UrnaFileBuilder::new(m).add_chunk(one_chunk()).build_bytes();
    assert!(matches!(err, Err(UrnaError::ManifestInvalid(_))));
}

#[test]
fn rejects_bad_dim() {
    let mut c = one_chunk();
    c.embedding = vec![1.0, 0.0]; // wrong dim
    let err = UrnaFileBuilder::new(good_manifest())
        .add_chunk(c)
        .build_bytes();
    assert!(matches!(err, Err(UrnaError::DimensionMismatch { .. })));
}

#[test]
#[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
fn zstd_encoding_preserves_content_hash() {
    let mut m = good_manifest();
    m.n_chunks = 3;
    let chunks: Vec<ChunkInput> = (0..3)
        .map(|i| ChunkInput {
            canonical_text: format!("paragraph number {}", i),
            source_uri: "doc.txt".into(),
            byte_start: i * 10,
            byte_end: i * 10 + 5,
            embedding: {
                let mut v = vec![0.0; 4];
                v[(i % 4) as usize] = 1.0;
                v
            },
        })
        .collect();

    let raw = UrnaFileBuilder::new(m.clone())
        .add_chunks(chunks.clone())
        .build_bytes()
        .unwrap();
    let zst = UrnaFileBuilder::new(m)
        .text_encoding(SectionEncoding::Zstd)
        .add_chunks(chunks)
        .build_bytes()
        .unwrap();

    let v_raw = UrnaView::from_bytes(&raw).unwrap();
    let v_zst = UrnaView::from_bytes(&zst).unwrap();
    assert_eq!(
        v_raw.content_hash_hex().unwrap(),
        v_zst.content_hash_hex().unwrap(),
        "content_hash must equal across wire encodings"
    );
    assert_ne!(
        v_raw.file_hash_hex(),
        v_zst.file_hash_hex(),
        "file_hash must differ between raw and zstd"
    );
}

#[test]
#[cfg_attr(all(miri, target_arch = "aarch64"), ignore)] // half converts f16 with inline asm on aarch64, which miri cannot run
fn float16_embeddings_roundtrip() {
    let mut m = good_manifest();
    m.n_chunks = 2;
    let bytes = UrnaFileBuilder::new(m)
        .embedding_dtype(EmbeddingDType::Float16)
        .add_chunk(ChunkInput {
            canonical_text: "a".into(),
            source_uri: "d".into(),
            byte_start: 0,
            byte_end: 1,
            embedding: vec![1.0, 0.0, 0.0, 0.0],
        })
        .add_chunk(ChunkInput {
            canonical_text: "b".into(),
            source_uri: "d".into(),
            byte_start: 1,
            byte_end: 2,
            embedding: vec![0.0, 1.0, 0.0, 0.0],
        })
        .build_bytes()
        .unwrap();
    let view = UrnaView::from_bytes(&bytes).unwrap();
    assert_eq!(view.manifest.dtype, "float16");
    let entry = view.entry(SECTION_EMBEDDINGS).unwrap();
    assert_eq!(entry.encoding, SECTION_ENCODING_FLOAT16);
    assert_eq!(entry.size as usize, 2 * 4 * 2); // n*dim*2
}

#[test]
fn int8_embeddings_roundtrip() {
    let mut m = good_manifest();
    m.n_chunks = 2;
    let bytes = UrnaFileBuilder::new(m)
        .embedding_dtype(EmbeddingDType::Int8)
        .add_chunk(ChunkInput {
            canonical_text: "a".into(),
            source_uri: "d".into(),
            byte_start: 0,
            byte_end: 1,
            embedding: vec![1.0, 0.0, 0.0, 0.0],
        })
        .add_chunk(ChunkInput {
            canonical_text: "b".into(),
            source_uri: "d".into(),
            byte_start: 1,
            byte_end: 2,
            embedding: vec![0.0, 1.0, 0.0, 0.0],
        })
        .build_bytes()
        .unwrap();
    let view = UrnaView::from_bytes(&bytes).unwrap();
    assert_eq!(view.manifest.dtype, "int8");
    let entry = view.entry(SECTION_EMBEDDINGS).unwrap();
    assert_eq!(entry.encoding, SECTION_ENCODING_INT8);
    // 8-byte prefix + n*4 (scales) + n*dim
    assert_eq!(entry.size as usize, 8 + 2 * 4 + 2 * 4);
}

#[test]
#[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
fn two_compressed_builds_byte_identical_with_text_levers() {
    // a heavily-repeated corpus engages the dict/dedup text levers under the
    // compressed preset. two reproducible builds must still be byte-identical
    // (the dict trainer + dedup pass + chooser are all pure functions).
    let mut m = good_manifest();
    let uniques: Vec<String> = (0..30)
        .map(|i| {
            (0..300)
                .map(|j| char::from(b'a' + (((i * 13 + j * 5) % 26) as u8)))
                .collect::<String>()
        })
        .collect();
    let mut texts: Vec<String> = Vec::new();
    for _ in 0..12 {
        for u in &uniques {
            texts.push(u.clone());
        }
    }
    m.n_chunks = texts.len() as u64;
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

    let b1 = UrnaFileBuilder::new(m.clone())
        .text_encoding(SectionEncoding::Zstd)
        .reproducible(true)
        .add_chunks(chunks.clone())
        .build_bytes()
        .unwrap();
    let b2 = UrnaFileBuilder::new(m)
        .text_encoding(SectionEncoding::Zstd)
        .reproducible(true)
        .add_chunks(chunks)
        .build_bytes()
        .unwrap();
    assert_eq!(b1, b2, "two compressed builds must be byte-identical");
    // the file is valid and re-openable.
    let view = UrnaView::from_bytes(&b1).unwrap();
    view.content_hash_hex().unwrap();
}

#[test]
#[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
fn rejects_zstd_embeddings() {
    // Embeddings can never be zstd-compressed (we want SIMD-friendly
    // mmap reads). text_encoding does not apply to embeddings.
    let m = good_manifest();
    let bytes = UrnaFileBuilder::new(m)
        .text_encoding(SectionEncoding::Zstd)
        .add_chunk(one_chunk())
        .build_bytes()
        .unwrap();
    let view = UrnaView::from_bytes(&bytes).unwrap();
    let entry = view.entry(SECTION_EMBEDDINGS).unwrap();
    assert_eq!(entry.encoding, SECTION_ENCODING_RAW);
}
