//! Files written before the rename carry the `NEST` magic. The reader
//! accepts them unchanged (same layout, same hashes); the writer never
//! emits that magic; any other magic is still rejected.
//!
//! The fixture is the 0.4.0 golden file, byte for byte, taken from the
//! last commit before the rename.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use urna_format::layout::{LEGACY_MAGIC, URNA_MAGIC};
use urna_format::manifest::Manifest;
use urna_format::sections::decode_chunks_canonical;
use urna_format::{ChunkInput, SECTION_CHUNKS_CANONICAL, UrnaError, UrnaFileBuilder, UrnaView};

const LEGACY: &[u8] = include_bytes!("fixtures/legacy_v040_minimal.nest");

#[test]
fn legacy_fixture_carries_the_old_magic() {
    assert_eq!(&LEGACY[..4], LEGACY_MAGIC);
    assert_eq!(LEGACY.len(), 1366);
}

#[test]
fn legacy_magic_opens_and_validates() {
    let view = UrnaView::from_bytes(LEGACY).unwrap();
    assert_eq!(&view.header.magic, LEGACY_MAGIC);
    assert_eq!(view.header.n_chunks, 1);
    assert_eq!(
        view.file_hash_hex(),
        "sha256:7fe82b9d2ee5b8f7b5535cb3b9b4736a66a11051d44a1894c160a7cf9bd4a799"
    );
    assert_eq!(
        view.content_hash_hex().unwrap(),
        "sha256:8d9904cf3689ffc9a0e0f9b387dd0ec41d0c0e4dc3c511b31ef909651000e1b8"
    );
    let texts =
        decode_chunks_canonical(view.get_section_data(SECTION_CHUNKS_CANONICAL).unwrap(), 1)
            .unwrap();
    assert_eq!(texts, vec!["hi".to_string()]);
}

#[test]
fn writer_emits_only_the_new_magic() {
    let m = Manifest {
        embedding_model: "demo".into(),
        embedding_dim: 4,
        n_chunks: 1,
        chunker_version: "demo-chunker/1".into(),
        model_hash: format!("sha256:{}", "0".repeat(64)),
        ..Default::default()
    };
    let bytes = UrnaFileBuilder::new(m)
        .add_chunk(ChunkInput {
            canonical_text: "hi".into(),
            source_uri: "doc.txt".into(),
            byte_start: 0,
            byte_end: 2,
            embedding: vec![1.0, 0.0, 0.0, 0.0],
        })
        .build_bytes()
        .unwrap();
    assert_eq!(&bytes[..4], URNA_MAGIC);
}

#[test]
fn any_other_magic_is_still_rejected() {
    let mut bad = LEGACY.to_vec();
    bad[..4].copy_from_slice(b"NEXT");
    match UrnaView::from_bytes(&bad).map(|_| ()) {
        Err(UrnaError::MagicMismatch { got, .. }) => assert_eq!(&got, b"NEXT"),
        other => panic!("expected MagicMismatch, got {other:?}"),
    }
}
