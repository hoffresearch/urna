//! blob_refs (0x14) + blob_span_overlay (0x16) positive coverage:
//!
//! - decode(encode(records)) == records over empty / single / many-entry
//!   tables, with utf-8 uris and both storage modes.
//! - deterministic re-encode: two builds of the same table are byte-
//!   identical (entry order is the contract, no sorting).
//! - decode(encode(entries)) == entries for the span overlay, including
//!   BLOB_REF_NONE sentinels.
//! - content_hash equality: a fixed corpus WITH vs WITHOUT the blob pair
//!   has IDENTICAL content_hash (citations stay stable), because both
//!   sections are excluded from CANONICAL_SECTIONS.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use urna_format::manifest::Manifest;
use urna_format::writer::UrnaFileBuilder;
use urna_format::{
    BLOB_REF_NONE, BlobRefRecord, BlobSpanEntry, ChunkInput, UrnaView, decode_blob_refs,
    decode_blob_span_overlay, encode_blob_refs, encode_blob_span_overlay,
};

fn record(seed: u8, uri: &str, byte_len: u64, inlined: bool) -> BlobRefRecord {
    BlobRefRecord {
        content_hash: [seed; 32],
        original_uri: uri.into(),
        byte_len,
        inlined,
    }
}

fn roundtrip(records: Vec<BlobRefRecord>) {
    let payload = encode_blob_refs(&records).unwrap();
    let got = decode_blob_refs(&payload).unwrap();
    assert_eq!(got, records, "record table mismatch");
    let payload2 = encode_blob_refs(&records).unwrap();
    assert_eq!(payload, payload2, "two builds must be byte-identical");
    let payload3 = encode_blob_refs(&got).unwrap();
    assert_eq!(payload, payload3, "re-encode of decoded table must match");
}

fn overlay_roundtrip(entries: Vec<BlobSpanEntry>) {
    let payload = encode_blob_span_overlay(&entries).unwrap();
    let got = decode_blob_span_overlay(&payload).unwrap();
    assert_eq!(got, entries, "overlay entries mismatch");
    let payload2 = encode_blob_span_overlay(&entries).unwrap();
    assert_eq!(payload, payload2, "two builds must be byte-identical");
}

#[test]
fn empty_table_roundtrips() {
    roundtrip(vec![]);
    overlay_roundtrip(vec![]);
}

#[test]
fn single_record_roundtrips() {
    roundtrip(vec![record(7, "media/corpus.av1", 1_892_352, true)]);
}

#[test]
fn many_records_roundtrips() {
    let records = vec![
        record(1, "media/shard-000.av1", 10_000_000, true),
        record(2, "s3://bucket/external.av1", 250_000_000, false),
        record(3, "media/图像是很好的.avif", 512, true),
        record(4, "", 0, false),
    ];
    roundtrip(records);
}

#[test]
fn overlay_entries_roundtrip() {
    let entries = vec![
        BlobSpanEntry {
            blob_ref_index: 0,
            byte_start: 0,
            byte_end: 4096,
        },
        BlobSpanEntry {
            blob_ref_index: 1,
            byte_start: 4096,
            byte_end: 8192,
        },
        // legacy/text chunk: no blob, spans fall back to 0x03.
        BlobSpanEntry {
            blob_ref_index: BLOB_REF_NONE,
            byte_start: 0,
            byte_end: 0,
        },
    ];
    overlay_roundtrip(entries);
}

// ---- content_hash equality: the honesty anchor for citations ----

fn build_corpus(path: &std::path::Path, with_blobs: bool) {
    let n = 4usize;
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
            canonical_text: format!("frame {i} label text"),
            source_uri: "frames".into(),
            byte_start: i as u64,
            byte_end: (i + 1) as u64,
            embedding: emb,
        });
    }
    if with_blobs {
        let records = vec![record(9, "media/corpus.av1", 1_892_352, true)];
        builder = builder.blob_refs(encode_blob_refs(&records).unwrap());
        let entries: Vec<BlobSpanEntry> = (0..n)
            .map(|i| BlobSpanEntry {
                blob_ref_index: 0,
                byte_start: (i * 4096) as u64,
                byte_end: ((i + 1) * 4096) as u64,
            })
            .collect();
        builder = builder.blob_span_overlay(encode_blob_span_overlay(&entries).unwrap());
    }
    builder.write_to_path(path).unwrap();
}

#[test]
fn blob_sections_do_not_change_content_hash() {
    let mut a = std::env::temp_dir();
    a.push("blob_ch_without.urna");
    let mut b = std::env::temp_dir();
    b.push("blob_ch_with.urna");
    let _ = std::fs::remove_file(&a);
    let _ = std::fs::remove_file(&b);
    build_corpus(&a, false);
    build_corpus(&b, true);

    let da = std::fs::read(&a).unwrap();
    let db = std::fs::read(&b).unwrap();
    let va = UrnaView::from_bytes(&da).unwrap();
    let vb = UrnaView::from_bytes(&db).unwrap();

    assert_eq!(
        va.content_hash_hex().unwrap(),
        vb.content_hash_hex().unwrap(),
        "the 0x14/0x16 blob sections must NOT change content_hash (citations stable)"
    );
    // the blob build legitimately moves file_hash (the bytes differ).
    assert_ne!(va.file_hash_hex(), vb.file_hash_hex());
    // the with-blobs file genuinely carries both sections and the capability.
    assert!(vb.entry(0x14).is_ok(), "with-blobs file must carry 0x14");
    assert!(vb.entry(0x16).is_ok(), "with-blobs file must carry 0x16");
    assert!(va.entry(0x14).is_err(), "without-blobs file must not");
    assert!(va.entry(0x16).is_err(), "without-blobs file must not");
    let ext = vb.manifest.capabilities_ext.as_ref().unwrap();
    assert_eq!(ext.blobs_present, Some(true));
    // and the payloads decode back to exactly what was written.
    let recs = decode_blob_refs(vb.get_section_data(0x14).unwrap()).unwrap();
    assert_eq!(recs.len(), 1);
    assert_eq!(recs[0].original_uri, "media/corpus.av1");
    let entries = decode_blob_span_overlay(vb.get_section_data(0x16).unwrap()).unwrap();
    assert_eq!(entries.len(), 4);
    assert_eq!(entries[2].byte_start, 8192);

    let _ = std::fs::remove_file(&a);
    let _ = std::fs::remove_file(&b);
}
