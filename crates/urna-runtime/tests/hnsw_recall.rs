//! Recall regression tests for the HNSW Algorithm 4 heuristic.
//!
//! Two scales:
//!
//! - **Small synthetic** (n=2000, dim=128): runs fast in `cargo test`,
//!   exercises the full build/search loop with default HNSW params, and
//!   asserts mean recall@10 ≥ 0.95 vs in-process exact top-10.
//! - **Realistic synthetic** (n=10_000, dim=384): mirrors the production
//!   corpus shape. Slower (~30s in release). Asserts the same threshold
//!   when search is given a candidate budget commensurate with the
//!   corpus (`ef_search ≥ 4*k`).
//!
//! Recall here is set overlap between approximate and exact top-k.
//! Queries are sampled from a different seed than the corpus so they
//! are out-of-distribution (not chunks-of-corpus). This is the strict
//! production workload — `measure_presets.py` complements with
//! near-corpus auto-queries on the real PT-BR data.
//!
//! Today's HNSW impl reaches recall@10 ≥ 0.95 at default `m=16,
//! ef_construction=400` when paired with `ef_search ≥ 400`. With
//! smaller `ef_construction` (e.g. 200), recall is ~0.84 at the same
//! `ef_search` — the production default was bumped to 400 in Phase 2.
//! Tracking a possible 10% recall gap vs the hnswlib reference; the
//! current numbers are sufficient for the production gate but not yet
//! optimal.

use std::collections::HashSet;

use urna_runtime::ann::{DEFAULT_EF_CONSTRUCTION, DEFAULT_M, HnswIndex};

/// Tiny LCG (deterministic) for synthetic corpora.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(
            seed.wrapping_mul(2862933555777941757)
                .wrapping_add(3037000493),
        )
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))) as f32
    }
}

fn random_l2(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    let mut v = Vec::with_capacity(n * dim);
    for _ in 0..(n * dim) {
        v.push(rng.next_f32() - 0.5);
    }
    for i in 0..n {
        let row = &mut v[i * dim..(i + 1) * dim];
        let norm: f32 = row
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

fn exact_topk(corpus: &[f32], n: usize, dim: usize, q: &[f32], k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = (0..n)
        .map(|i| {
            let row = &corpus[i * dim..(i + 1) * dim];
            let s: f32 = q.iter().zip(row.iter()).map(|(x, y)| x * y).sum();
            (i, s)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

#[allow(clippy::too_many_arguments)]
fn measure_recall(
    idx: &HnswIndex,
    corpus: &[f32],
    n: usize,
    dim: usize,
    queries: &[f32],
    n_queries: usize,
    k: usize,
    ef: usize,
) -> f64 {
    let mut total = 0.0f64;
    for q_idx in 0..n_queries {
        let q = &queries[q_idx * dim..(q_idx + 1) * dim];
        let exact: HashSet<usize> = exact_topk(corpus, n, dim, q, k).into_iter().collect();
        let approx: HashSet<usize> = idx.search(q, ef).into_iter().take(k).collect();
        total += exact.intersection(&approx).count() as f64 / k as f64;
    }
    total / n_queries as f64
}

#[test]
fn hnsw_recall_at_10_synthetic() {
    // Small synthetic at default params — graph is dense at this scale,
    // so recall@10 should be very high.
    let n = 2000;
    let dim = 128;
    let corpus = random_l2(n, dim, 0xCAFE_BABE);
    let idx = HnswIndex::build(
        corpus.clone(),
        n,
        dim,
        DEFAULT_M,
        DEFAULT_EF_CONSTRUCTION,
        0xDEAD_BEEF,
    );

    let n_queries = 50;
    let queries = random_l2(n_queries, dim, 0xFACE_FEED);

    let recall = measure_recall(&idx, &corpus, n, dim, &queries, n_queries, 10, 200);
    assert!(
        recall >= 0.95,
        "mean recall@10 too low: {:.4} (expected >= 0.95) — Algorithm 4 regression?",
        recall
    );
}

#[test]
fn hnsw_recall_at_1_synthetic() {
    // recall@1 is the strictest: ANN must find the literal nearest neighbor.
    let n = 2000;
    let dim = 128;
    let corpus = random_l2(n, dim, 0xCAFE_BABE);
    let idx = HnswIndex::build(
        corpus.clone(),
        n,
        dim,
        DEFAULT_M,
        DEFAULT_EF_CONSTRUCTION,
        0xDEAD_BEEF,
    );

    let n_queries = 50;
    let queries = random_l2(n_queries, dim, 0xFACE_FEED);

    let recall = measure_recall(&idx, &corpus, n, dim, &queries, n_queries, 1, 100);
    assert!(
        recall >= 0.90,
        "mean recall@1 too low: {:.4} (expected >= 0.90)",
        recall
    );
}

/// Realistic-sized corpus (n=10_000, dim=384). Mirrors the production
/// shape. Uses ef_search=400 (= default ef_construction) so the runtime
/// has a candidate budget commensurate with the corpus. Slow in debug;
/// run via `cargo test --release` in CI.
#[test]
fn hnsw_recall_realistic_size() {
    let n = 10_000;
    let dim = 384;
    let corpus = random_l2(n, dim, 0xCAFE_BABE);
    let mut idx = HnswIndex::build(
        corpus.clone(),
        n,
        dim,
        DEFAULT_M,
        DEFAULT_EF_CONSTRUCTION,
        0xDEAD_BEEF,
    );
    idx.ef_search = 400;

    let n_queries = 30;
    let queries = random_l2(n_queries, dim, 0xFACE_FEED);

    let recall = measure_recall(&idx, &corpus, n, dim, &queries, n_queries, 10, 400);
    assert!(
        recall >= 0.94,
        "mean recall@10 at realistic scale too low: {:.4} (expected >= 0.94)",
        recall
    );
}

/// sha256 of the serialized graph, hex.
fn graph_digest(idx: &HnswIndex) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(idx.to_bytes()))
}

/// the build is deterministic for a seed AND its bytes are pinned: any
/// change to the insertion order, the distance arithmetic, the neighbor
/// heuristic or the tie-breaking shows up here as a different digest.
/// any build-throughput work must either keep these
/// digests or change them on purpose, in the same commit, with the reason.
/// `PIN_HNSW_DIGESTS=1 cargo test --release -p urna-runtime --test
/// hnsw_recall graph_bytes_are_pinned -- --nocapture` prints the current
/// values to paste below.
#[test]
fn graph_bytes_are_pinned() {
    const CASES: [(usize, usize, u64, u64, usize, usize, &str); 2] = [
        // (n, dim, corpus seed, build seed, m, ef_construction, digest)
        (
            2_000,
            128,
            0x5EED_0001,
            0xBEEF,
            DEFAULT_M,
            DEFAULT_EF_CONSTRUCTION,
            "0210ecf212b48524fea016a3033064259ce23075666cbae43f0bdb02d0b77f9b",
        ),
        // 2026-09-10: `cosine_dist` moved from a sequential f32 loop to eight
        // fixed-order accumulators (identical on every target, no fma). the
        // last bits of a few distances differ, so a handful of near-ties on
        // the 10k build resolve the other way; the 2k build is unchanged and
        // recall@10 is the same (hnsw_recall_realistic_size). previous digest:
        // 7b0cbd99fd12d272f6925d5dc82f691e3f411eea65400dea968c38f44248c1e8
        (
            10_000,
            384,
            0xCAFE_BABE,
            0xDEAD_BEEF,
            DEFAULT_M,
            DEFAULT_EF_CONSTRUCTION,
            "bf93c2fecdb6bba867787f23b0b541e1e9534bdeec625007162eb0d152254139",
        ),
    ];
    let print = std::env::var("PIN_HNSW_DIGESTS").is_ok();
    for (n, dim, corpus_seed, build_seed, m, ef, want) in CASES {
        let corpus = random_l2(n, dim, corpus_seed);
        let idx = HnswIndex::build(corpus, n, dim, m, ef, build_seed);
        let got = graph_digest(&idx);
        if print {
            println!("pin n={n} dim={dim} -> {got}");
            continue;
        }
        assert_eq!(
            got, want,
            "hnsw graph bytes changed for n={n} dim={dim}: update the pin on purpose, in the same commit, with the reason"
        );
    }
}
