//! Deterministic mutation fuzz over the reader: build real `.urna` files in
//! every stored-dtype / text-encoding combination the writer supports,
//! then hammer `UrnaView::from_bytes` and every section decoder with
//! thousands of corrupted variants. The contract under test is the one in
//! doc/SECURITY.md: a malformed file may be REJECTED with a typed error but
//! must never panic, loop, or read out of bounds.
//!
//! Two mutation regimes, both seeded, so a failure is reproducible from
//! its printed seed:
//!
//! - raw: the mutation lands as-is, so most variants die at the header /
//!   section / footer checksum layer (that layer is what is being tested).
//! - resealed: after mutating, every checksum and the footer hash are
//!   recomputed, so the variant passes integrity and reaches the section
//!   decoders, the manifest parser and the embeddings validators. This is
//!   the regime that finds decoder bugs.
//!
//! `URNA_MUTATION_ITERS` overrides the per-fixture iteration count (default
//! 1500; CI runs the default, a local soak can run 100000). `cargo fuzz`
//! targets under `fuzz/` run the same exercise with coverage guidance; set
//! `URNA_FUZZ_SEED_DIR` to have this test dump its base fixtures there as
//! seeds.

// not under miri: thousands of mutated decodes, most through zstd (c code miri cannot call); the deterministic harness runs natively in ci.
#![cfg(not(miri))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]

mod common;

use common::mutation::{Rng, mutate, reseal};
use urna_format::layout::*;
use urna_format::manifest::Manifest;
use urna_format::sections::{
    decode_chunk_ids, decode_chunks_canonical, decode_chunks_original_spans, decode_provenance,
    decode_search_contract,
};
use urna_format::writer::{EmbeddingDType, SectionEncoding, UrnaFileBuilder};
use urna_format::{
    BLOB_REF_NONE, BlobRefRecord, BlobSpanEntry, ChunkInput, EDGE_TYPE_NEXT_CHUNK, Edge,
    Int4EmbeddingsView, Int8EmbeddingsView, SPACE_DTYPE_F32, SpaceEntry, UrnaView,
    decode_blob_data_table, decode_blob_refs, decode_blob_span_overlay, decode_graph_adjacency,
    decode_space_table, encode_blob_data, encode_blob_refs, encode_blob_span_overlay,
    encode_graph_adjacency, encode_space_table,
};

const N: usize = 24;
const DIM: usize = 64; // int4 needs a multiple of 64

fn fixture(dtype: EmbeddingDType, text: SectionEncoding) -> Vec<u8> {
    let mut rng = Rng::new(0x5EED);
    let manifest = Manifest {
        embedding_model: "demo".into(),
        embedding_dim: DIM as u32,
        n_chunks: N as u64,
        chunker_version: "demo-chunker/1".into(),
        model_hash: format!("sha256:{}", "0".repeat(64)),
        ..Default::default()
    };
    let mut b = UrnaFileBuilder::new(manifest)
        .embedding_dtype(dtype)
        .text_encoding(text);
    for i in 0..N {
        let mut emb: Vec<f32> = (0..DIM).map(|_| rng.f32() - 0.5).collect();
        let norm = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        emb.iter_mut().for_each(|x| *x /= norm);
        b = b.add_chunk(ChunkInput {
            canonical_text: format!("chunk {i} alpha beta term{} shared{}", i, i % 5),
            source_uri: format!("doc{}.txt", i % 3),
            byte_start: (i * 10) as u64,
            byte_end: (i * 10 + 9) as u64,
            embedding: emb,
        });
    }
    let edges: Vec<Edge> = (0..N - 1)
        .map(|i| Edge {
            src: i as u32,
            dst: (i + 1) as u32,
            edge_type: EDGE_TYPE_NEXT_CHUNK,
        })
        .collect();
    let blobs: [&[u8]; 2] = [b"blob-zero-bytes", b"blob-one-bytes-longer"];
    let refs: Vec<BlobRefRecord> = blobs
        .iter()
        .enumerate()
        .map(|(i, bytes)| BlobRefRecord {
            content_hash: common::mutation::sha256(bytes),
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
        let a = (i as f32 * 0.37).cos();
        let c = (i as f32 * 0.37).sin();
        band.extend_from_slice(&a.to_le_bytes());
        band.extend_from_slice(&c.to_le_bytes());
    }
    let path = std::env::temp_dir().join(format!(
        "urna_mutation_{}_{}.urna",
        dtype.manifest_str(),
        text.id()
    ));
    b.graph_adjacency(encode_graph_adjacency(N, &edges).unwrap())
        .blob_refs(encode_blob_refs(&refs).unwrap())
        .blob_data(encode_blob_data(&[Some(blobs[0]), Some(blobs[1])]).unwrap())
        .blob_span_overlay(encode_blob_span_overlay(&overlay).unwrap())
        .space_table(encode_space_table(&spaces).unwrap())
        .space_band(1, SECTION_ENCODING_RAW, band)
        .write_to_path(&path)
        .unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    bytes
}

/// Everything a reader can do with a file; errors are fine, panics are not.
fn exercise(bytes: &[u8]) {
    let Ok(view) = UrnaView::from_bytes(bytes) else {
        return;
    };
    let n = view.header.n_chunks as usize;
    let dim = view.header.embedding_dim as usize;
    let _ = view.validate_embeddings_values();
    let _ = view.content_hash_hex();
    let _ = view.file_hash_hex();
    let _ = view.search_contract();
    for entry in view.section_table.clone() {
        let id = entry.section_id;
        let Ok(payload) = view.decoded_section(id) else {
            continue;
        };
        let p: &[u8] = &payload;
        match id {
            SECTION_CHUNK_IDS => {
                let _ = decode_chunk_ids(p, n);
            }
            SECTION_CHUNKS_CANONICAL => {
                let _ = decode_chunks_canonical(p, n);
            }
            SECTION_CHUNKS_ORIGINAL_SPANS => {
                let _ = decode_chunks_original_spans(p, n);
            }
            SECTION_PROVENANCE => {
                let _ = decode_provenance(p);
            }
            SECTION_SEARCH_CONTRACT => {
                let _ = decode_search_contract(p);
            }
            SECTION_EMBEDDINGS => {
                if let Ok(v) = Int8EmbeddingsView::parse(p, n, dim) {
                    for i in 0..v.n {
                        let _ = (v.row(i), v.scale(i));
                    }
                }
                if let Ok(v) = Int4EmbeddingsView::parse(p, n, dim) {
                    let mut scales = vec![0.0f32; v.blocks];
                    for i in 0..v.n {
                        v.row_scales_into(i, &mut scales);
                        let _ = v.row_codes(i);
                    }
                }
            }
            SECTION_GRAPH_ADJACENCY => {
                let _ = decode_graph_adjacency(p);
            }
            SECTION_BLOB_REFS => {
                let _ = decode_blob_refs(p);
            }
            SECTION_BLOB_SPAN_OVERLAY => {
                let _ = decode_blob_span_overlay(p);
            }
            SECTION_BLOB_DATA => {
                let _ = decode_blob_data_table(p);
            }
            SECTION_SPACE_TABLE => {
                let _ = decode_space_table(p);
            }
            _ => {}
        }
    }
}

fn iters() -> usize {
    std::env::var("URNA_MUTATION_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1500)
}

#[test]
fn mutated_files_never_panic_the_reader() {
    let combos = [
        (EmbeddingDType::Float32, SectionEncoding::Raw),
        (EmbeddingDType::Float16, SectionEncoding::Zstd),
        (EmbeddingDType::Int8, SectionEncoding::Zstd),
        (EmbeddingDType::Int4, SectionEncoding::Raw),
    ];
    let seed_dir = std::env::var_os("URNA_FUZZ_SEED_DIR");
    let mut failures: Vec<String> = Vec::new();
    for (dtype, text) in combos {
        let base = fixture(dtype, text);
        exercise(&base); // sanity: the base fixture must be fully readable
        assert!(
            UrnaView::from_bytes(&base).is_ok(),
            "base fixture must parse"
        );
        if let Some(dir) = &seed_dir {
            let p = std::path::Path::new(dir).join(format!(
                "format_{}_{}.bin",
                dtype.manifest_str(),
                text.id()
            ));
            std::fs::write(p, &base).unwrap();
        }
        let name = format!("{}/{}", dtype.manifest_str(), text.id());
        for seed in 0..iters() as u64 {
            let mut rng = Rng::new(seed ^ 0xA5A5_0000);
            let mut m = mutate(&base, &mut rng);
            if seed % 2 == 1 {
                reseal(&mut m);
            }
            let r = std::panic::catch_unwind(|| exercise(&m));
            if r.is_err() {
                failures.push(format!("{name} seed={seed} resealed={}", seed % 2 == 1));
            }
        }
    }
    assert!(failures.is_empty(), "reader panicked on: {failures:?}");
}
