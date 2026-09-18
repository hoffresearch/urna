//! blob overlay integration: a media corpus built with the 0x14/0x16 pair
//! opens with its 0x03 placeholder spans REPLACED by the blob-relative
//! spans (uri + byte range from the overlay), so search hits and
//! cite/retrieve report the real media coordinates. BLOB_REF_NONE entries
//! keep their 0x03 span. a dangling blob_ref_index fails open with a
//! typed error, never a silent fallback.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use std::path::PathBuf;
use urna_format::manifest::Manifest;
use urna_format::writer::UrnaFileBuilder;
use urna_format::{
    BLOB_REF_NONE, BlobRefRecord, BlobSpanEntry, ChunkInput, encode_blob_refs,
    encode_blob_span_overlay,
};
use urna_runtime::MmapUrnaFile;

fn tmp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(name);
    p
}

fn build_media_corpus(path: &PathBuf, dangling: bool) {
    let n = 3usize;
    let dim = 4usize;
    let manifest = Manifest {
        embedding_model: "demo".into(),
        embedding_dim: dim as u32,
        n_chunks: n as u64,
        chunker_version: "demo-chunker/1".into(),
        model_hash: format!("sha256:{}", "0".repeat(64)),
        ..Default::default()
    };
    let mut builder = UrnaFileBuilder::new(manifest).reproducible(true);
    for i in 0..n {
        let mut emb = vec![0.0f32; dim];
        emb[i % dim] = 1.0;
        builder = builder.add_chunk(ChunkInput {
            canonical_text: format!("frame {i}"),
            // 0x03 carries ordinal placeholders in a media corpus.
            source_uri: "frames".into(),
            byte_start: i as u64,
            byte_end: (i + 1) as u64,
            embedding: emb,
        });
    }
    let records = vec![BlobRefRecord {
        content_hash: [9; 32],
        original_uri: "media/corpus.av1".into(),
        byte_len: 1_892_352,
        inlined: true,
    }];
    builder = builder.blob_refs(encode_blob_refs(&records).unwrap());
    let entries = vec![
        BlobSpanEntry {
            blob_ref_index: if dangling { 7 } else { 0 },
            byte_start: 0,
            byte_end: 4096,
        },
        BlobSpanEntry {
            blob_ref_index: 0,
            byte_start: 4096,
            byte_end: 8192,
        },
        // legacy/text chunk: keeps its 0x03 span.
        BlobSpanEntry {
            blob_ref_index: BLOB_REF_NONE,
            byte_start: 0,
            byte_end: 0,
        },
    ];
    builder = builder.blob_span_overlay(encode_blob_span_overlay(&entries).unwrap());
    builder.write_to_path(path).unwrap();
}

#[test]
fn overlay_spans_replace_placeholders_at_open() {
    let path = tmp_path("rt_blob_overlay.urna");
    let _ = std::fs::remove_file(&path);
    build_media_corpus(&path, false);

    let rt = MmapUrnaFile::open(&path).unwrap();
    assert!(rt.has_blobs());
    let refs = rt.blob_refs().unwrap();
    assert_eq!(refs.len(), 1);
    assert_eq!(refs[0].original_uri, "media/corpus.av1");
    assert!(refs[0].inlined);

    // query hits chunk 0: its span must be the overlay's blob-relative
    // range against the blob uri, not the 0x03 ordinal placeholder.
    let res = rt.search(&[1.0f32, 0.0, 0.0, 0.0], 3).unwrap();
    let hit0 = res
        .hits
        .iter()
        .find(|h| h.offset_end == 4096)
        .expect("chunk 0");
    assert_eq!(hit0.source_uri, "media/corpus.av1");
    assert_eq!(hit0.offset_start, 0);
    // chunk 2 is BLOB_REF_NONE: keeps its 0x03 span (uri "frames", 2..3).
    let hit2 = res
        .hits
        .iter()
        .find(|h| h.source_uri == "frames")
        .expect("chunk 2");
    assert_eq!(hit2.offset_start, 2);
    assert_eq!(hit2.offset_end, 3);

    // inspect lists the blob table.
    let doc: serde_json::Value = serde_json::from_str(&rt.inspect_json().unwrap()).unwrap();
    assert_eq!(doc["blobs"][0]["original_uri"], "media/corpus.av1");
    assert_eq!(doc["blobs"][0]["byte_len"], 1_892_352u64);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn dangling_blob_ref_fails_open_with_typed_error() {
    let path = tmp_path("rt_blob_dangling.urna");
    let _ = std::fs::remove_file(&path);
    build_media_corpus(&path, true);
    assert!(MmapUrnaFile::open(&path).is_err());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn corpus_without_blob_capability_has_no_blobs() {
    // a plain text corpus (no blob sections, no capability) opens with
    // has_blobs() == false and blob_refs() == None.
    let path = tmp_path("rt_blob_none.urna");
    let _ = std::fs::remove_file(&path);
    let manifest = Manifest {
        embedding_model: "demo".into(),
        embedding_dim: 4,
        n_chunks: 1,
        chunker_version: "demo-chunker/1".into(),
        model_hash: format!("sha256:{}", "0".repeat(64)),
        ..Default::default()
    };
    UrnaFileBuilder::new(manifest)
        .add_chunk(ChunkInput {
            canonical_text: "plain text".into(),
            source_uri: "doc.txt".into(),
            byte_start: 0,
            byte_end: 10,
            embedding: vec![1.0, 0.0, 0.0, 0.0],
        })
        .write_to_path(&path)
        .unwrap();
    let rt = MmapUrnaFile::open(&path).unwrap();
    assert!(!rt.has_blobs());
    assert!(rt.blob_refs().is_none());
    let doc: serde_json::Value = serde_json::from_str(&rt.inspect_json().unwrap()).unwrap();
    assert!(doc["blobs"].is_null());
    let _ = std::fs::remove_file(&path);
}
