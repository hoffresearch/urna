//! Pure-Rust HNSW index for approximate nearest neighbor search.
//!
//! This is a minimal, self-contained HNSW implementation tailored to
//! `.urna`'s contract:
//!
//! - Vectors are L2-normalized → distance is `1 - cosine` (smaller = closer).
//! - Index lives in section `0x07` and is bit-equal across rebuilds.
//! - Search returns a candidate set; the runtime reranks with the exact
//!   dot product against the embeddings section so the final score is
//!   the real cosine.
//!
//! On-disk layout (`encoding=raw`, payload version 1):
//!
//! ```text
//!   u32 LE  payload_version = 1
//!   u32 LE  m                 — out-degree at non-zero levels
//!   u32 LE  m_max0            — out-degree at level 0 (typically 2*m)
//!   u32 LE  ef_construction
//!   u32 LE  entry_point       — node id of the entry vertex
//!   u32 LE  max_level         — highest layer with any node (0-based)
//!   u32 LE  n_nodes           — equal to header.n_embeddings
//!   for each node i in 0..n_nodes:
//!       u32 LE  level_i       — top layer this node lives in
//!       for layer in 0..=level_i:
//!           u32 LE  k_i_l     — neighbor count at this layer
//!           u32 LE * k_i_l    — neighbor ids
//! ```
//!
//! Construction uses HNSW (Malkov & Yashunin, 2018) with a deterministic
//! level distribution so the same input produces the same graph.
//! Neighbor selection lives in `select_neighbors`; today it uses the
//! `_simple` variant (top-m by distance), Phase 2 swaps in the
//! Algorithm 4 heuristic for higher recall.

mod build;
mod codec;
mod search;
pub mod select_neighbors;
mod visited;

use crate::materialize::PackedVectors;

/// on-disk payload version for the hnsw section (`0x07`). v1 stored every
/// neighbour id as a raw u32; v2 bitpacks the level/count/neighbour columns
/// with `intpack` (order-preserving, so the graph and its recall are
/// unchanged). the reader still accepts v1 files. the section is optional
/// and excluded from content_hash, so this bump is additive within v1.
pub const HNSW_PAYLOAD_VERSION: u32 = 2;

/// Default neighbor count at non-zero layers. 16 is a common HNSW sweet
/// spot for ~1M points; for smaller corpora the recall-vs-size curve is
/// flat enough that the default is fine.
pub const DEFAULT_M: usize = 16;
/// Default candidate-list size during construction. Larger = better
/// recall, slower build. 400 is our chosen production default —
/// empirically gives recall@10 ≥ 0.95 at typical corpus sizes
/// (n ≤ 100k, dim ≤ 768) when paired with `ef_search ≥ 400`. Lower
/// values save build time but require larger `ef_search` to match.
pub const DEFAULT_EF_CONSTRUCTION: usize = 400;

#[derive(Clone, Debug)]
pub(super) struct Node {
    /// Top layer this node lives in (0-based).
    pub level: u32,
    /// `neighbors[layer][i]` is the i-th neighbor id at `layer`. Index 0
    /// is the densest layer (level 0).
    pub neighbors: Vec<Vec<u32>>,
}

/// A built HNSW index. Reads borrow from the on-disk payload at open
/// time; the graph is owned (small relative to embeddings).
pub struct HnswIndex {
    pub m: usize,
    pub m_max0: usize,
    pub ef_construction: usize,
    pub entry_point: u32,
    pub max_level: u32,
    pub(super) nodes: Vec<Node>,
    /// The vectors used at search time, kept in their on-disk packing
    /// (int8 stays int8 + scales, f16 stays f16) and decoded one row at a
    /// time. The graph stays dtype-independent (f16/i8 runtimes get the
    /// same recall curve) without the old `n*dim*4` f32 snapshot: the
    /// resident footprint is the packed size, not 4x it.
    pub(super) store: PackedVectors,
    pub(super) dim: usize,
    pub(super) n: usize,
    /// `ef_search` default. Caller can override per query.
    pub ef_search: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Candidate {
    pub id: u32,
    /// `1 - cosine`. Smaller = closer.
    pub dist: f32,
}

impl Eq for Candidate {}

/// `1 - dot` over two l2-normalized rows, with eight independent
/// accumulators combined in a fixed tree. plain rust on purpose: the
/// compiler vectorizes the eight lanes on every target (sse2 on x86_64,
/// neon on aarch64) and, without fma contraction, the arithmetic is the
/// same on all of them, so a graph built on one machine is byte-identical
/// to the same build on another. the simd kernels in `crate::simd` are
/// NOT used here: their reduction trees differ per backend, which would
/// make the graph bytes depend on the cpu that built them.
#[inline]
pub(super) fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for j in 0..8 {
            acc[j] += x[j] * y[j];
        }
    }
    let mut tail = 0.0f32;
    for (x, y) in ra.iter().zip(rb.iter()) {
        tail += x * y;
    }
    let dot =
        ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7])) + tail;
    1.0 - dot
}

/// Distance between an f32 query and stored row `i`, decoding `i` through
/// `store` into `scratch` first. `scratch` is empty for the f32 store.
#[inline]
pub(super) fn dist_q(
    store: &PackedVectors,
    q: &[f32],
    i: usize,
    dim: usize,
    scratch: &mut [f32],
) -> f32 {
    cosine_dist(q, store.row(i, dim, scratch))
}

/// Distance between two stored rows `a` and `b`, each decoded through
/// `store` into its own scratch buffer (`sa`, `sb`).
#[inline]
pub(super) fn dist_rr(
    store: &PackedVectors,
    a: usize,
    b: usize,
    dim: usize,
    sa: &mut [f32],
    sb: &mut [f32],
) -> f32 {
    cosine_dist(store.row(a, dim, sa), store.row(b, dim, sb))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ann::build::LcgRng;
    use std::collections::HashSet;

    pub(super) fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut v = Vec::with_capacity(n * dim);
        for _ in 0..(n * dim) {
            v.push((rng.next_f64() as f32) - 0.5);
        }
        // L2-normalize each row.
        for i in 0..n {
            let row = &mut v[i * dim..(i + 1) * dim];
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in row.iter_mut() {
                    *x /= norm;
                }
            }
        }
        v
    }

    #[test]
    fn small_index_recall_against_exact() {
        // 200 random vectors, dim 32. Recall@10 vs exact should be very
        // high — small enough that the graph is fully connected.
        let n = 200;
        let dim = 32;
        let vecs = random_vectors(n, dim, 0xDEAD_BEEF);
        let idx = HnswIndex::build(vecs.clone(), n, dim, 8, 50, 42);

        let q = random_vectors(1, dim, 0xCAFEBABE);
        // Exact top-10.
        let mut exact: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let row = &vecs[i * dim..(i + 1) * dim];
                let mut s = 0.0f32;
                for j in 0..dim {
                    s += q[j] * row[j];
                }
                (i, s)
            })
            .collect();
        exact.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let exact_top: HashSet<usize> = exact.iter().take(10).map(|p| p.0).collect();
        let approx = idx.search(&q, 50);
        let approx_set: HashSet<usize> = approx.into_iter().take(10).collect();
        let overlap = exact_top.intersection(&approx_set).count();
        assert!(
            overlap >= 7,
            "recall@10 too low: {} of 10 (expected >= 7)",
            overlap
        );
    }

    #[test]
    fn serialize_roundtrip() {
        let n = 50;
        let dim = 16;
        let vecs = random_vectors(n, dim, 7);
        let idx = HnswIndex::build(vecs.clone(), n, dim, 8, 30, 42);
        let bytes = idx.to_bytes();
        let mut decoded = HnswIndex::from_bytes(&bytes, n, dim).unwrap();
        decoded.attach_vectors(vecs);
        // Same query should produce a similar candidate set.
        let q: Vec<f32> = vec![1.0 / (dim as f32).sqrt(); dim];
        let a = idx.search(&q, 20);
        let b = decoded.search(&q, 20);
        let a_set: HashSet<usize> = a.into_iter().collect();
        let b_set: HashSet<usize> = b.into_iter().collect();
        assert_eq!(a_set, b_set, "candidate sets must match after roundtrip");
    }

    #[test]
    fn deterministic_build_same_seed() {
        let n = 30;
        let dim = 8;
        let vecs = random_vectors(n, dim, 0xABCD);
        let a = HnswIndex::build(vecs.clone(), n, dim, 4, 20, 123);
        let b = HnswIndex::build(vecs, n, dim, 4, 20, 123);
        assert_eq!(a.to_bytes(), b.to_bytes(), "same seed => same graph");
    }
}
