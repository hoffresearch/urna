//! `urna cite` and `urna validate` on a media corpus, against real files
//! built with the blob pair (0x14 blob_refs, 0x17 blob_data) and the 0x16
//! span overlay.
//!
//! - cite prints the overlay span (the blob uri and its byte range), the
//!   same span search hits and `retrieve` report; a chunk whose overlay
//!   entry is BLOB_REF_NONE keeps its stored 0x03 span.
//! - validate proves every inlined blob before it prints anything: a blob
//!   that fails its content_hash exits non-zero with no "OK:" on stdout.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::Command;
use urna_format::manifest::Manifest;
use urna_format::writer::UrnaFileBuilder;
use urna_format::{
    BLOB_REF_NONE, BlobRefRecord, BlobSpanEntry, ChunkInput, encode_blob_data, encode_blob_refs,
    encode_blob_span_overlay,
};

const BLOB: &[u8] = b"not really av1, but bytes with a hash";

fn tmp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(name);
    p
}

/// three chunks: two point into the inlined blob through the overlay, the
/// third is a text chunk (BLOB_REF_NONE). `blob_hash` is what 0x14 records
/// for the inlined bytes, so a wrong one builds a file whose blob fails.
fn build_media_corpus(path: &Path, blob_hash: [u8; 32]) {
    let dim = 4usize;
    let manifest = Manifest {
        embedding_model: "demo".into(),
        embedding_dim: dim as u32,
        n_chunks: 3,
        chunker_version: "demo-chunker/1".into(),
        model_hash: format!("sha256:{}", "0".repeat(64)),
        ..Default::default()
    };
    let mut builder = UrnaFileBuilder::new(manifest).reproducible(true);
    for i in 0..3usize {
        let mut emb = vec![0.0f32; dim];
        emb[i % dim] = 1.0;
        builder = builder.add_chunk(ChunkInput {
            canonical_text: format!("frame {i}"),
            // the forge writes row ordinals into 0x03 for a media corpus
            source_uri: "rows".into(),
            byte_start: i as u64,
            byte_end: (i + 1) as u64,
            embedding: emb,
        });
    }
    let records = vec![BlobRefRecord {
        content_hash: blob_hash,
        original_uri: "media/corpus.av1".into(),
        byte_len: BLOB.len() as u64,
        inlined: true,
    }];
    let entries = vec![
        BlobSpanEntry {
            blob_ref_index: 0,
            byte_start: 0,
            byte_end: 16,
        },
        BlobSpanEntry {
            blob_ref_index: 0,
            byte_start: 16,
            byte_end: BLOB.len() as u64,
        },
        BlobSpanEntry {
            blob_ref_index: BLOB_REF_NONE,
            byte_start: 0,
            byte_end: 0,
        },
    ];
    builder
        .blob_refs(encode_blob_refs(&records).unwrap())
        .blob_data(encode_blob_data(&[Some(BLOB)]).unwrap())
        .blob_span_overlay(encode_blob_span_overlay(&entries).unwrap())
        .write_to_path(path)
        .unwrap();
}

fn good_hash() -> [u8; 32] {
    Sha256::digest(BLOB).into()
}

/// the `urna://content_hash/chunk_id` citation of chunk `i`, read from the
/// file the way `retrieve` would hand it out.
fn citation(path: &Path, i: usize) -> String {
    let rt = urna_engine::MmapUrnaFile::open(path).unwrap();
    let mut q = vec![0.0f32; 4];
    q[i] = 1.0;
    let hit = rt.search(&q, 1).unwrap().hits.remove(0);
    format!("urna://{}/{}", rt.content_hash(), hit.chunk_id)
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_urna"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn cite_prints_the_overlay_span_like_retrieve() {
    let path = tmp_path("cli_media_cite.urna");
    build_media_corpus(&path, good_hash());
    let p = path.to_str().unwrap();

    let out = run(&["cite", p, &citation(&path, 1)]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("source_uri:   media/corpus.av1"),
        "{stdout}"
    );
    assert!(stdout.contains("byte_start:   16"), "{stdout}");
    assert!(
        stdout.contains(&format!("byte_end:     {}", BLOB.len())),
        "{stdout}"
    );
    assert!(stdout.contains("frame 1"), "{stdout}");

    // the runtime reports the same span for the same chunk
    let rt = urna_engine::MmapUrnaFile::open(&path).unwrap();
    let hit = rt.search(&[0.0, 1.0, 0.0, 0.0], 1).unwrap().hits.remove(0);
    assert_eq!(hit.source_uri, "media/corpus.av1");
    assert_eq!(hit.offset_start, 16);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn cite_keeps_the_stored_span_for_a_text_chunk() {
    let path = tmp_path("cli_media_cite_text.urna");
    build_media_corpus(&path, good_hash());
    let p = path.to_str().unwrap();

    let out = run(&["cite", p, &citation(&path, 2)]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("source_uri:   rows"), "{stdout}");
    assert!(stdout.contains("byte_start:   2"), "{stdout}");
    assert!(stdout.contains("byte_end:     3"), "{stdout}");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn validate_reports_inlined_blobs_on_a_good_file() {
    let path = tmp_path("cli_media_validate_ok.urna");
    build_media_corpus(&path, good_hash());

    let out = run(&["validate", path.to_str().unwrap()]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.starts_with("OK: "), "{stdout}");
    assert!(
        stdout.contains("Inlined blobs:      1 verified"),
        "{stdout}"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn validate_prints_no_ok_when_an_inlined_blob_fails() {
    let path = tmp_path("cli_media_validate_bad.urna");
    // every checksum is right; only the recorded blob hash is wrong
    build_media_corpus(&path, [7; 32]);

    let out = run(&["validate", path.to_str().unwrap()]);
    assert!(!out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stdout.contains("OK:"), "stdout must stay empty: {stdout}");
    assert!(stderr.contains("fails its content_hash"), "{stderr}");
    let _ = std::fs::remove_file(&path);
}
