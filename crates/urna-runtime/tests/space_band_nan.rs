//! Regression for the second mutation-fuzz finding: a multimodal band with
//! a NaN lane reached the exact-cosine sort as a NaN score, and a NaN in
//! `partial_cmp(..).unwrap_or(Equal)` is not a total order (panic since
//! rust 1.81). Two fixes, both pinned here: the band is value-validated at
//! open like the canonical embeddings, and every runtime sort is a NaN-last
//! total order, so even a value that slips past validation can only rank
//! last, never panic.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]

mod common;

use common::mutation::reseal;
use std::path::PathBuf;
use urna_format::manifest::Manifest;
use urna_format::writer::UrnaFileBuilder;
use urna_format::{
    ChunkInput, SECTION_ENCODING_RAW, SECTION_SPACE_EMBEDDINGS_BASE, SPACE_DTYPE_F32, SpaceEntry,
    UrnaView, encode_space_table,
};
use urna_runtime::{MmapUrnaFile, RuntimeError};

const N: usize = 3;

fn build(path: &PathBuf) -> Vec<u8> {
    let mut b = UrnaFileBuilder::new(Manifest {
        embedding_model: "demo".into(),
        embedding_dim: 4,
        n_chunks: N as u64,
        chunker_version: "demo/1".into(),
        model_hash: format!("sha256:{}", "0".repeat(64)),
        ..Default::default()
    });
    for i in 0..N {
        let mut e = vec![0.0f32; 4];
        e[i] = 1.0;
        b = b.add_chunk(ChunkInput {
            canonical_text: format!("chunk {i}"),
            source_uri: "doc.txt".into(),
            byte_start: 0,
            byte_end: 1,
            embedding: e,
        });
    }
    let spaces = vec![SpaceEntry {
        space_index: 1,
        name: "vision".into(),
        dim: 2,
        dtype: SPACE_DTYPE_F32,
        model_hash: "sha256:vis".into(),
        n_vectors: N as u64,
    }];
    let mut band = Vec::new();
    for _ in 0..N {
        band.extend_from_slice(&0.6f32.to_le_bytes());
        band.extend_from_slice(&0.8f32.to_le_bytes());
    }
    b.space_table(encode_space_table(&spaces).unwrap())
        .space_band(1, SECTION_ENCODING_RAW, band)
        .write_to_path(path)
        .unwrap();
    std::fs::read(path).unwrap()
}

#[test]
fn nan_lane_in_a_space_band_is_rejected_at_open() {
    let path = std::env::temp_dir().join(format!("urna_band_nan_{}.urna", std::process::id()));
    let mut bytes = build(&path);
    let view = UrnaView::from_bytes(&bytes).unwrap();
    let band = *view.entry(SECTION_SPACE_EMBEDDINGS_BASE + 1).unwrap();
    drop(view);
    let off = band.offset as usize + 4; // second lane of row 0
    bytes[off..off + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    reseal(&mut bytes);
    std::fs::write(&path, &bytes).unwrap();
    let Err(err) = MmapUrnaFile::open(&path) else {
        panic!("a NaN band lane must not open");
    };
    let _ = std::fs::remove_file(&path);
    assert!(
        matches!(
            err,
            RuntimeError::Format(urna_format::UrnaError::InvalidEmbeddingValue)
        ),
        "expected InvalidEmbeddingValue, got {err:?}"
    );
}
