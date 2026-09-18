//! Matryoshka truncate-then-renormalize roundtrip (WO EMB-matryoshka).
//!
//! The python builder (crates/urna-python/src/build_fn.rs) does the prefix
//! slice + L2-renorm before handing chunks to UrnaFileBuilder. These tests
//! mirror that pure deterministic op at the format level: a file whose chunks
//! were truncated+renormalized stores EXACTLY the renormalized prefix bytes
//! in the embeddings section (raw f32), and two such builds are byte-
//! identical (file_hash equal). They also confirm the additive manifest
//! disclosure fields (mrl_dim/full_dim) travel through the writer/reader.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use urna_format::layout::*;
use urna_format::manifest::Manifest;
use urna_format::{ChunkInput, UrnaView};

fn manifest(full_dim: u32, mrl_dim: u32, n: u64) -> Manifest {
    Manifest {
        embedding_model: "demo".into(),
        embedding_dim: mrl_dim,
        n_chunks: n,
        chunker_version: "demo-chunker/1".into(),
        model_hash: format!("sha256:{}", "0".repeat(64)),
        mrl_dim: Some(mrl_dim),
        full_dim: Some(full_dim),
        ..Default::default()
    }
}

/// Deterministic per-row source vector at the full dim. Spread mass across
/// dims so a prefix is not trivially the whole vector.
fn full_vec(i: usize, full_dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; full_dim];
    for (j, x) in v.iter_mut().enumerate() {
        *x = (((i * 31 + j * 7) % 23) as f32 - 11.0) * 0.05 + 0.01;
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut v {
        *x /= norm;
    }
    v
}

/// The pure op the builder applies: slice to the prefix, re-L2-normalize.
fn truncate_renorm(v: &[f32], mrl_dim: usize) -> Vec<f32> {
    let mut p = v[..mrl_dim].to_vec();
    let norm: f32 = p.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut p {
        *x /= norm;
    }
    p
}

fn chunk(i: usize, emb: Vec<f32>) -> ChunkInput {
    ChunkInput {
        canonical_text: format!("text {i} alpha"),
        source_uri: "doc.txt".into(),
        byte_start: (i * 10) as u64,
        byte_end: ((i + 1) * 10) as u64,
        embedding: emb,
    }
}

fn build_truncated(full_dim: usize, mrl_dim: usize, n: usize) -> Vec<u8> {
    use urna_format::writer::UrnaFileBuilder;
    UrnaFileBuilder::new(manifest(full_dim as u32, mrl_dim as u32, n as u64))
        .reproducible(true)
        .add_chunks((0..n).map(|i| chunk(i, truncate_renorm(&full_vec(i, full_dim), mrl_dim))))
        .build_bytes()
        .unwrap()
}

#[test]
fn truncated_embeddings_section_equals_manual_prefix_renorm_byte_for_byte() {
    let full_dim = 384usize;
    let mrl_dim = 128usize;
    let n = 5usize;

    let bytes = build_truncated(full_dim, mrl_dim, n);
    let view = UrnaView::from_bytes(&bytes).unwrap();

    // header/manifest stride by the prefix dim.
    assert_eq!(view.header.embedding_dim, mrl_dim as u32);
    assert_eq!(view.manifest.embedding_dim, mrl_dim as u32);
    assert_eq!(view.manifest.mrl_dim, Some(mrl_dim as u32));
    assert_eq!(view.manifest.full_dim, Some(full_dim as u32));
    assert_eq!(view.manifest.dtype, "float32");

    // the raw f32 embeddings section is EXACTLY the renormalized prefixes,
    // concatenated, little-endian, with no reordering.
    let section = view.get_section_data(SECTION_EMBEDDINGS).unwrap();
    let mut expected: Vec<u8> = Vec::with_capacity(n * mrl_dim * 4);
    for i in 0..n {
        for x in truncate_renorm(&full_vec(i, full_dim), mrl_dim) {
            expected.extend_from_slice(&x.to_le_bytes());
        }
    }
    assert_eq!(
        section,
        &expected[..],
        "truncation is a pure deterministic slice + renorm: bytes must match"
    );
}

#[test]
fn two_truncated_builds_are_byte_identical() {
    let a = build_truncated(384, 128, 5);
    let b = build_truncated(384, 128, 5);
    assert_eq!(a, b, "same chunks + same mrl_dim => byte-identical file");

    let va = UrnaView::from_bytes(&a).unwrap();
    let vb = UrnaView::from_bytes(&b).unwrap();
    assert_eq!(va.file_hash_hex(), vb.file_hash_hex());
}

#[test]
fn truncated_content_hash_differs_from_full_dim() {
    use urna_format::writer::UrnaFileBuilder;
    let full_dim = 384usize;
    let n = 5usize;

    let full = UrnaFileBuilder::new(Manifest {
        embedding_model: "demo".into(),
        embedding_dim: full_dim as u32,
        n_chunks: n as u64,
        chunker_version: "demo-chunker/1".into(),
        model_hash: format!("sha256:{}", "0".repeat(64)),
        ..Default::default()
    })
    .reproducible(true)
    .add_chunks((0..n).map(|i| chunk(i, full_vec(i, full_dim))))
    .build_bytes()
    .unwrap();

    let truncated = build_truncated(full_dim, 128, n);

    let vf = UrnaView::from_bytes(&full).unwrap();
    let vt = UrnaView::from_bytes(&truncated).unwrap();
    // content_hash is over the decoded embeddings bytes: a truncated file
    // legitimately differs from full-dim, so citations are tied to a given
    // mrl_dim (never claimed stable across dims).
    assert_ne!(
        vf.content_hash_hex().unwrap(),
        vt.content_hash_hex().unwrap()
    );
}
