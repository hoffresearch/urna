//! Reciprocal-rank fusion of two ranked candidate lists. Used by the
//! hybrid search path to merge BM25 and vector candidates before exact
//! cosine rerank.

use std::collections::HashMap;

/// RRF k constant. 60 is the value from the original Cormack et al.
/// paper; small enough that the top items dominate, large enough to
/// not be a brittle tie-breaker.
const RRF_K: f32 = 60.0;

/// Reciprocal-rank fusion of two ranked candidate lists. Returns the
/// union sorted by RRF score descending. The runtime then reranks with
/// the exact dot product so the final scores are real cosine.
pub fn rrf_union(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut scores: HashMap<usize, f32> = HashMap::new();
    for (rank, &id) in a.iter().enumerate() {
        *scores.entry(id).or_insert(0.0) += 1.0 / (RRF_K + rank as f32 + 1.0);
    }
    for (rank, &id) in b.iter().enumerate() {
        *scores.entry(id).or_insert(0.0) += 1.0 / (RRF_K + rank as f32 + 1.0);
    }
    let mut all: Vec<(usize, f32)> = scores.into_iter().collect();
    all.sort_by(|x, y| crate::order::cmp_score_desc(*x, *y));
    all.into_iter().map(|p| p.0).collect()
}
