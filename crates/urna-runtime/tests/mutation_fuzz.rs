//! Deterministic mutation fuzz over the runtime open + search path. The
//! format crate's twin (`crates/urna-format/tests/mutation_fuzz.rs`) covers
//! the reader and the section decoders; this one covers what only the
//! runtime does with a file: the HNSW / BM25 / graph codecs, the blob store,
//! the multimodal spaces, and every search entry point, all through a real
//! `mmap` of a real file. Same rules: a corrupted file may be rejected with
//! a typed error, never with a panic (doc/SECURITY.md scope).
//!
//! `URNA_MUTATION_ITERS` overrides the per-fixture count (default 250, it is
//! file-backed so slower than the format twin). `URNA_FUZZ_SEED_DIR` dumps
//! the base fixtures as seeds for `cargo fuzz`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]

mod common;

use common::mutation::{Rng, mutate, reseal, sha256};
use std::path::PathBuf;
use urna_format::manifest::Manifest;
use urna_format::writer::{EmbeddingDType, SectionEncoding, UrnaFileBuilder};
use urna_format::{
    BLOB_REF_NONE, BlobRefRecord, BlobSpanEntry, ChunkInput, EDGE_TYPE_NEXT_CHUNK,
    EDGE_TYPE_SEMANTIC, Edge, SECTION_ENCODING_RAW, SPACE_DTYPE_F32, SpaceEntry, encode_blob_data,
    encode_blob_refs, encode_blob_span_overlay, encode_graph_adjacency, encode_space_table,
};
use urna_runtime::MmapUrnaFile;
use urna_runtime::ann::HnswIndex;
use urna_runtime::bm25::Bm25Index;

const N: usize = 40;
const DIM: usize = 64;

fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "urna_rt_mutation_{name}_{}.urna",
        std::process::id()
    ))
}

/// Every optional section the runtime knows how to open, over one corpus.
fn fixture(dtype: EmbeddingDType, text: SectionEncoding) -> Vec<u8> {
    let mut rng = Rng::new(0xC0FFEE);
    let mut vectors = vec![0.0f32; N * DIM];
    for row in vectors.chunks_mut(DIM) {
        row.iter_mut().for_each(|x| *x = rng.f32() - 0.5);
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        row.iter_mut().for_each(|x| *x /= norm);
    }
    let texts: Vec<String> = (0..N)
        .map(|i| format!("doc {i} alpha beta term{} shared{}", i, i % 7))
        .collect();
    let manifest = Manifest {
        embedding_model: "demo".into(),
        embedding_dim: DIM as u32,
        n_chunks: N as u64,
        chunker_version: "demo-chunker/1".into(),
        model_hash: format!("sha256:{}", "0".repeat(64)),
        ..Default::default()
    };
    let hnsw = HnswIndex::build(vectors.clone(), N, DIM, 8, 64, 42);
    let bm25 = Bm25Index::build(&texts, 1.2, 0.75);
    let mut b = UrnaFileBuilder::new(manifest)
        .embedding_dtype(dtype)
        .text_encoding(text);
    for i in 0..N {
        b = b.add_chunk(ChunkInput {
            canonical_text: texts[i].clone(),
            source_uri: "doc.txt".into(),
            byte_start: (i * 10) as u64,
            byte_end: ((i + 1) * 10) as u64,
            embedding: vectors[i * DIM..(i + 1) * DIM].to_vec(),
        });
    }
    let mut edges = Vec::new();
    for i in 0..N - 1 {
        for (s, d) in [(i, i + 1), (i + 1, i)] {
            edges.push(Edge {
                src: s as u32,
                dst: d as u32,
                edge_type: EDGE_TYPE_NEXT_CHUNK,
            });
        }
        edges.push(Edge {
            src: i as u32,
            dst: ((i + 11) % N) as u32,
            edge_type: EDGE_TYPE_SEMANTIC,
        });
    }
    let blobs: [&[u8]; 2] = [b"blob-zero-bytes", b"blob-one-bytes-longer"];
    let refs: Vec<BlobRefRecord> = blobs
        .iter()
        .enumerate()
        .map(|(i, bytes)| BlobRefRecord {
            content_hash: sha256(bytes),
            original_uri: format!("media/{i}.bin"),
            byte_len: bytes.len() as u64,
            inlined: true,
        })
        .collect();
    let overlay: Vec<BlobSpanEntry> = (0..N)
        .map(|i| BlobSpanEntry {
            blob_ref_index: if i % 4 == 3 {
                BLOB_REF_NONE
            } else {
                (i % 2) as u32
            },
            byte_start: 0,
            byte_end: 8,
        })
        .collect();
    let spaces = vec![SpaceEntry {
        space_index: 1,
        name: "vision".into(),
        dim: 2,
        dtype: SPACE_DTYPE_F32,
        model_hash: "sha256:vis111".into(),
        n_vectors: N as u64,
    }];
    let mut band = Vec::new();
    for i in 0..N {
        band.extend_from_slice(&(i as f32 * 0.37).cos().to_le_bytes());
        band.extend_from_slice(&(i as f32 * 0.37).sin().to_le_bytes());
    }
    let path = tmp("fixture");
    b.hnsw_index(hnsw.to_bytes())
        .bm25_index(bm25.to_bytes())
        .graph_adjacency(encode_graph_adjacency(N, &edges).unwrap())
        .blob_refs(encode_blob_refs(&refs).unwrap())
        .blob_data(encode_blob_data(&[Some(blobs[0]), Some(blobs[1])]).unwrap())
        .blob_span_overlay(encode_blob_span_overlay(&overlay).unwrap())
        .space_table(encode_space_table(&spaces).unwrap())
        .space_band(1, SECTION_ENCODING_RAW, band)
        .hybrid()
        .write_to_path(&path)
        .unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    bytes
}

/// Open through mmap and run every search verb; errors are fine, panics not.
fn exercise(bytes: &[u8], path: &PathBuf) -> bool {
    std::fs::write(path, bytes).unwrap();
    let Ok(f) = MmapUrnaFile::open(path) else {
        return false;
    };
    let dim = f.embedding_dim();
    let q: Vec<f32> = (0..dim).map(|j| ((j as f32) * 0.11).sin()).collect();
    let _ = f.search(&q, 5);
    let _ = f.search_ann(&q, 5, 32);
    let _ = f.search_hybrid(&q, "alpha term3 shared2", 5, 16);
    let _ = f.search_graph(&q, 5, 2, 32);
    let _ = f.search_space("vision", &[0.6, 0.8], 5, None);
    let _ = f.search_space("vision", &[0.6, 0.8], 5, Some("sha256:vis111"));
    let _ = f.blob_bytes(0);
    let _ = f.blob_bytes(1);
    let _ = f.blob_bytes(7);
    let _ = f.inspect_json();
    let _ = f.revalidate();
    true
}

fn iters() -> usize {
    std::env::var("URNA_MUTATION_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(250)
}

#[test]
fn mutated_files_never_panic_the_runtime() {
    let combos = [
        (EmbeddingDType::Float32, SectionEncoding::Raw, "f32_raw"),
        (EmbeddingDType::Float16, SectionEncoding::Zstd, "f16_zstd"),
        (EmbeddingDType::Int8, SectionEncoding::Zstd, "i8_zstd"),
        (EmbeddingDType::Int4, SectionEncoding::Raw, "i4_raw"),
    ];
    let seed_dir = std::env::var_os("URNA_FUZZ_SEED_DIR");
    let mut failures: Vec<String> = Vec::new();
    for (dtype, text, name) in combos {
        let base = fixture(dtype, text);
        let path = tmp(name);
        assert!(exercise(&base, &path), "base fixture {name} must open");
        if let Some(dir) = &seed_dir {
            std::fs::write(
                std::path::Path::new(dir).join(format!("runtime_{name}.bin")),
                &base,
            )
            .unwrap();
        }
        for seed in 0..iters() as u64 {
            let mut rng = Rng::new(seed ^ 0x7777_0000);
            let mut m = mutate(&base, &mut rng);
            if seed % 2 == 1 {
                reseal(&mut m);
            }
            let r = std::panic::catch_unwind(|| exercise(&m, &path));
            if r.is_err() {
                failures.push(format!("{name} seed={seed} resealed={}", seed % 2 == 1));
            }
        }
        let _ = std::fs::remove_file(&path);
    }
    assert!(failures.is_empty(), "runtime panicked on: {failures:?}");
}
