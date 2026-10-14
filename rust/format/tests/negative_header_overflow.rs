//! Regression for the first mutation-fuzz finding: a header whose
//! `n_embeddings * embedding_dim` does not fit in `usize` must be a typed
//! rejection, never an arithmetic-overflow panic (debug) or a wrapped
//! product that matches a tiny section (release).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]

mod common;

use common::mutation::reseal;
use urna_format::encoding::expected_embeddings_size;
use urna_format::layout::{URNA_HEADER_SIZE, UrnaHeader};
use urna_format::manifest::Manifest;
use urna_format::writer::UrnaFileBuilder;
use urna_format::{ChunkInput, UrnaView};

fn minimal() -> Vec<u8> {
    let path = std::env::temp_dir().join(format!("urna_overflow_{}.urna", std::process::id()));
    UrnaFileBuilder::new(Manifest {
        embedding_model: "demo".into(),
        embedding_dim: 4,
        n_chunks: 1,
        chunker_version: "demo/1".into(),
        model_hash: format!("sha256:{}", "0".repeat(64)),
        ..Default::default()
    })
    .add_chunk(ChunkInput {
        canonical_text: "alpha".into(),
        source_uri: "doc.txt".into(),
        byte_start: 0,
        byte_end: 5,
        embedding: vec![1.0, 0.0, 0.0, 0.0],
    })
    .write_to_path(&path)
    .unwrap();
    let b = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    b
}

#[test]
fn expected_size_reports_overflow_as_none() {
    for dtype in ["float32", "float16", "int8", "int4"] {
        assert!(
            expected_embeddings_size(dtype, usize::MAX / 2, 64).is_none(),
            "{dtype}"
        );
        assert!(
            expected_embeddings_size(dtype, 1 << 40, 1 << 40).is_none(),
            "{dtype}"
        );
    }
    assert_eq!(expected_embeddings_size("float32", 3, 4), Some(48));
}

#[test]
fn header_with_overflowing_shape_is_rejected_not_panicked() {
    let mut bytes = minimal();
    let mut h = UrnaHeader::default();
    h.as_bytes_mut().copy_from_slice(&bytes[..URNA_HEADER_SIZE]);
    // n * dim * 4 wraps to a small number on 64-bit; resealed so the
    // rejection comes from the shape check, not from a checksum.
    h.n_embeddings = (1u64 << 62) + 1;
    h.embedding_dim = 16;
    bytes[..URNA_HEADER_SIZE].copy_from_slice(h.as_bytes());
    reseal(&mut bytes);
    let Err(err) = UrnaView::from_bytes(&bytes) else {
        panic!("an overflowing shape must not parse");
    };
    assert!(!err.to_string().is_empty(), "must be a typed error");
}
