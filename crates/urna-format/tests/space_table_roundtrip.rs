//! space_table (0x15) positive coverage:
//!
//! - decode(encode(entries)) == entries over empty / single / many-space
//!   tables and every dtype code.
//! - deterministic re-encode: two builds of the same table are byte-
//!   identical (entry order is the contract).
//! - encode rejects invalid entries up front (space 0, out-of-band index,
//!   unknown dtype, empty name, non-sha256 hash, duplicate index/name).
//! - the reader rejects a band whose size disagrees with the table, and
//!   rejects a zstd-encoded band (bands are fixed-stride slabs, never
//!   zstd) — both through real .urna files.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use urna_format::manifest::Manifest;
use urna_format::writer::UrnaFileBuilder;
use urna_format::{
    ChunkInput, SECTION_ENCODING_ZSTD, SPACE_DTYPE_F16, SPACE_DTYPE_F32, SPACE_DTYPE_I4,
    SPACE_DTYPE_I8, SpaceEntry, UrnaError, UrnaView, decode_space_table, encode_space_table,
};

fn entry(idx: u8, name: &str, dim: u32, dtype: u8) -> SpaceEntry {
    SpaceEntry {
        space_index: idx,
        name: name.into(),
        dim,
        dtype,
        model_hash: format!("sha256:{:066}", idx),
        n_vectors: 2,
    }
}

fn roundtrip(entries: Vec<SpaceEntry>) {
    let payload = encode_space_table(&entries).unwrap();
    let got = decode_space_table(&payload).unwrap();
    assert_eq!(got, entries, "space table mismatch");
    let payload2 = encode_space_table(&entries).unwrap();
    assert_eq!(payload, payload2, "two builds must be byte-identical");
    let payload3 = encode_space_table(&got).unwrap();
    assert_eq!(payload, payload3, "re-encode of decoded table must match");
}

#[test]
fn empty_table_roundtrips() {
    roundtrip(vec![]);
}

#[test]
fn single_space_roundtrips() {
    roundtrip(vec![entry(1, "vision", 512, SPACE_DTYPE_F32)]);
}

#[test]
fn many_spaces_all_dtypes_roundtrip() {
    roundtrip(vec![
        entry(1, "vision", 512, SPACE_DTYPE_F32),
        entry(2, "audio", 128, SPACE_DTYPE_F16),
        entry(3, "depth", 64, SPACE_DTYPE_I8),
        entry(15, "thermal", 64, SPACE_DTYPE_I4),
    ]);
}

#[test]
fn invalid_entries_rejected_at_encode() {
    assert!(encode_space_table(&[entry(0, "text", 4, SPACE_DTYPE_F32)]).is_err());
    assert!(encode_space_table(&[entry(16, "x", 4, SPACE_DTYPE_F32)]).is_err());
    assert!(encode_space_table(&[entry(1, "x", 4, 9)]).is_err());
    assert!(encode_space_table(&[entry(1, "", 4, SPACE_DTYPE_F32)]).is_err());
    let mut bad_hash = entry(1, "x", 4, SPACE_DTYPE_F32);
    bad_hash.model_hash = "deadbeef".into();
    assert!(encode_space_table(&[bad_hash]).is_err());
    let dup = vec![
        entry(1, "a", 4, SPACE_DTYPE_F32),
        entry(1, "b", 4, SPACE_DTYPE_F32),
    ];
    assert!(encode_space_table(&dup).is_err());
    let dup_name = vec![
        entry(1, "a", 4, SPACE_DTYPE_F32),
        entry(2, "a", 4, SPACE_DTYPE_F32),
    ];
    assert!(encode_space_table(&dup_name).is_err());
}

// ---- band validation through real .urna files ----

fn base_builder(n: usize) -> UrnaFileBuilder {
    let manifest = Manifest {
        embedding_model: "demo".into(),
        embedding_dim: 4,
        n_chunks: n as u64,
        chunker_version: "demo-chunker/1".into(),
        model_hash: format!("sha256:{}", "0".repeat(64)),
        ..Default::default()
    };
    let mut b = UrnaFileBuilder::new(manifest).reproducible(true);
    for i in 0..n {
        let mut emb = vec![0.0f32; 4];
        emb[i % 4] = 1.0;
        b = b.add_chunk(ChunkInput {
            canonical_text: format!("chunk {i}"),
            source_uri: "doc.txt".into(),
            byte_start: (i * 10) as u64,
            byte_end: ((i + 1) * 10) as u64,
            embedding: emb,
        });
    }
    b
}

/// build and parse; only the verdict leaves the function (the view borrows
/// the bytes, so it is dropped here rather than leaked into 'static, which
/// miri's leak checker would flag).
fn open(builder: UrnaFileBuilder) -> Result<(), UrnaError> {
    let bytes = builder.build_bytes().unwrap();
    UrnaView::from_bytes(&bytes).map(|_| ())
}

fn open_err(builder: UrnaFileBuilder) -> UrnaError {
    match open(builder) {
        Err(e) => e,
        Ok(_) => panic!("expected the reader to reject the file"),
    }
}

#[test]
fn band_size_mismatch_rejected() {
    // the table claims 2 vectors of dim 4 f32 (32 bytes) but the band
    // carries only one vector (16 bytes): the reader must reject.
    let entries = vec![entry(1, "vision", 4, SPACE_DTYPE_F32)];
    let mut band = Vec::new();
    for x in [1.0f32, 0.0, 0.0, 0.0] {
        band.extend_from_slice(&x.to_le_bytes());
    }
    let b = base_builder(2)
        .space_table(encode_space_table(&entries).unwrap())
        .space_band(1, urna_format::SECTION_ENCODING_RAW, band);
    let err = open_err(b);
    assert!(matches!(err, UrnaError::EmbeddingSizeMismatch { .. }));
}

#[test]
#[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
fn zstd_band_rejected() {
    // bands are fixed-stride slabs scored by the simd kernels: the zstd
    // encoding is illegal for band ids and the reader rejects it at parse.
    let entries = vec![entry(1, "vision", 4, SPACE_DTYPE_F32)];
    let mut band = Vec::new();
    for i in 0..2 {
        for (j, x) in [1.0f32, 0.0, 0.0, 0.0].iter().enumerate() {
            let v = if i == j { *x } else { 0.0 };
            band.extend_from_slice(&v.to_le_bytes());
        }
    }
    let compressed = urna_format::zstd_encode(&band).unwrap();
    let b = base_builder(2)
        .space_table(encode_space_table(&entries).unwrap())
        .space_band(1, SECTION_ENCODING_ZSTD, compressed);
    let err = open_err(b);
    assert!(matches!(err, UrnaError::UnsupportedSectionEncoding { .. }));
}

#[test]
fn vector_count_mismatch_rejected() {
    // the table claims 3 vectors but the corpus has 2 chunks: bands are
    // parallel per-chunk embeddings, so the reader must reject.
    let mut e = entry(1, "vision", 4, SPACE_DTYPE_F32);
    e.n_vectors = 3;
    let mut band = Vec::new();
    for _ in 0..3 * 4 {
        band.extend_from_slice(&0.0f32.to_le_bytes());
    }
    let b = base_builder(2)
        .space_table(encode_space_table(&[e]).unwrap())
        .space_band(1, urna_format::SECTION_ENCODING_RAW, band);
    assert!(open(b).is_err());
}
