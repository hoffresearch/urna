//! `search_routed` picks the path by what the file carries, not by the
//! manifest's `index_type`. the case that motivated it: a file built with
//! the python `hybrid` preset declares `index_type = "hnsw"` and carries its
//! bm25 section as the `supports_bm25` capability; routing by the declared
//! string sent `ask`, `retrieve` and `search-text` down the hnsw path and
//! the lexical index was never read. the contract here:
//!
//!   bm25 section + query text  -> "hybrid" (bm25 candidates > 0)
//!   bm25 section, no text      -> "hnsw" when an hnsw section exists, else "exact"
//!   hnsw section only          -> "hnsw"
//!   neither                    -> "exact"
//!
//! and the graph is never routed to automatically (`search_graph` is its
//! entry point).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use std::path::PathBuf;
use urna_engine::MmapUrnaFile;
use urna_engine::ann::{DEFAULT_EF_CONSTRUCTION, DEFAULT_M, HnswIndex};
use urna_engine::bm25::Bm25Index;
use urna_format::ChunkInput;
use urna_format::manifest::Manifest;
use urna_format::writer::UrnaFileBuilder;

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(
            seed.wrapping_mul(2862933555777941757)
                .wrapping_add(3037000493),
        )
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))) as f32
    }
}

fn random_l2(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    let mut v = vec![0.0f32; n * dim];
    for x in v.iter_mut() {
        *x = rng.next_f32() - 0.5;
    }
    for i in 0..n {
        let row = &mut v[i * dim..(i + 1) * dim];
        let norm = row
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(f32::EPSILON);
        for x in row.iter_mut() {
            *x /= norm;
        }
    }
    v
}

/// one synthetic corpus, with the optional sections the case asks for. the
/// builder is used the way `urna.build` uses it: `hnsw_index` and
/// `bm25_index` only, never `.hybrid()`, so `index_type` stays "exact" or
/// "hnsw" exactly like a preset file.
fn build(path: &PathBuf, n: usize, dim: usize, hnsw: bool, bm25: bool) -> Vec<f32> {
    let vectors = random_l2(n, dim, 0xC0FFEE);
    let texts: Vec<String> = (0..n)
        .map(|i| format!("doc {i} alpha beta term{i} shared{}", i % 7))
        .collect();
    let manifest = Manifest {
        embedding_model: "demo".into(),
        embedding_dim: dim as u32,
        n_chunks: n as u64,
        chunker_version: "demo-chunker/1".into(),
        model_hash: format!("sha256:{}", "0".repeat(64)),
        ..Default::default()
    };
    let mut builder = UrnaFileBuilder::new(manifest);
    for i in 0..n {
        builder = builder.add_chunk(ChunkInput {
            canonical_text: texts[i].clone(),
            source_uri: "doc.txt".into(),
            byte_start: (i * 10) as u64,
            byte_end: ((i + 1) * 10) as u64,
            embedding: vectors[i * dim..(i + 1) * dim].to_vec(),
        });
    }
    if hnsw {
        let idx = HnswIndex::build(
            vectors.clone(),
            n,
            dim,
            DEFAULT_M,
            DEFAULT_EF_CONSTRUCTION,
            42,
        );
        builder = builder.hnsw_index(idx.to_bytes());
    }
    if bm25 {
        builder = builder.bm25_index(Bm25Index::build(&texts, 1.2, 0.75).to_bytes());
    }
    let _ = std::fs::remove_file(path);
    builder.write_to_path(path).unwrap();
    vectors
}

fn tmp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(name);
    p
}

#[test]
fn hybrid_preset_shape_routes_to_hybrid_with_text_and_hnsw_without() {
    let (n, dim) = (300usize, 32usize);
    let path = tmp_path("rt_search_routed_hnsw_bm25.urna");
    build(&path, n, dim, true, true);
    let rt = MmapUrnaFile::open(&path).unwrap();
    // the preset shape: the vector index is declared, the lexical one is a
    // capability.
    assert_eq!(rt.declared_index_type(), "hnsw");
    assert!(rt.has_bm25());

    let q = random_l2(1, dim, 0xABCD);
    let with_text = rt
        .search_routed(&q, Some("alpha shared3 term12"), 10, 64)
        .unwrap();
    assert_eq!(with_text.index_type, "hybrid");
    assert_eq!(with_text.explain.route, "hybrid");
    assert!(
        with_text.explain.bm25_candidates > 0,
        "the lexical leg must run when the text is given"
    );
    assert!(
        with_text.explain.ann_candidates > 0,
        "the vector leg is the ann shortlist"
    );

    let no_text = rt.search_routed(&q, None, 10, 64).unwrap();
    assert_eq!(no_text.index_type, "hnsw");
    assert_eq!(no_text.explain.bm25_candidates, 0);

    let blank = rt.search_routed(&q, Some("   "), 10, 64).unwrap();
    assert_eq!(blank.index_type, "hnsw", "blank text is no text");

    // the scores stay the exact rerank value whatever the route (the
    // rerank_contract test covers the byte-for-byte claim; this checks the
    // routed path is one of those paths).
    let exact = rt.search(&q, n as i32).unwrap();
    for h in &with_text.hits {
        let truth = exact
            .hits
            .iter()
            .find(|e| e.chunk_id == h.chunk_id)
            .unwrap();
        assert_eq!(h.score.to_bits(), truth.score.to_bits());
    }
}

#[test]
fn bm25_without_hnsw_routes_to_hybrid_over_the_exact_shortlist() {
    let (n, dim) = (120usize, 16usize);
    let path = tmp_path("rt_search_routed_bm25_only.urna");
    build(&path, n, dim, false, true);
    let rt = MmapUrnaFile::open(&path).unwrap();
    assert_eq!(rt.declared_index_type(), "exact");
    assert!(!rt.has_ann());
    assert!(rt.has_bm25());

    let q = random_l2(1, dim, 0x5EED);
    let with_text = rt.search_routed(&q, Some("term7 shared2"), 5, 32).unwrap();
    assert_eq!(with_text.index_type, "hybrid");
    assert!(with_text.explain.bm25_candidates > 0);
    assert!(
        with_text.explain.exact_candidates > 0 && with_text.explain.ann_candidates == 0,
        "no hnsw section: the vector leg is the exact-flat shortlist"
    );

    let no_text = rt.search_routed(&q, None, 5, 32).unwrap();
    assert_eq!(no_text.index_type, "exact");
    assert_eq!(no_text.recall, 1.0);
}

#[test]
fn plain_files_route_to_hnsw_or_exact() {
    let (n, dim) = (120usize, 16usize);
    let q = random_l2(1, dim, 0x5EED);

    let path = tmp_path("rt_search_routed_hnsw_only.urna");
    build(&path, n, dim, true, false);
    let rt = MmapUrnaFile::open(&path).unwrap();
    assert!(!rt.has_bm25());
    let r = rt.search_routed(&q, Some("alpha"), 5, 32).unwrap();
    assert_eq!(
        r.index_type, "hnsw",
        "text without a bm25 section changes nothing"
    );

    let path = tmp_path("rt_search_routed_exact_only.urna");
    build(&path, n, dim, false, false);
    let rt = MmapUrnaFile::open(&path).unwrap();
    let r = rt.search_routed(&q, Some("alpha"), 5, 32).unwrap();
    assert_eq!(r.index_type, "exact");
    assert_eq!(r.recall, 1.0);
}
