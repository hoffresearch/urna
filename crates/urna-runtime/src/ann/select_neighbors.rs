//! neighbor selection during HNSW construction.
//!
//! Two variants:
//!
//! - `select_neighbors_simple` (Algorithm 3 in Malkov-Yashunin 2018):
//!   sort candidates ascending by distance to the inserted node, keep
//!   top-m. Cheap. Picks geometrically-closest neighbors but tends to
//!   create dense clusters that hurt recall on non-uniform corpora.
//!
//! - `select_neighbors_heuristic` (Algorithm 4): iteratively pick the
//!   closest remaining candidate, but reject it if there is already a
//!   chosen neighbor that is closer to the candidate than the candidate
//!   is to the query — meaning the candidate is "shadowed" by an
//!   already-chosen point. This gives angular diversity. Refills any
//!   shortfall from the pruned set so neighbor lists actually reach m.
//!
//! Build (`super::build`) calls the heuristic. Simple is kept for unit
//! tests and as the lower-bound reference baseline.

use super::{Candidate, cosine_dist};
use crate::materialize::PackedVectors;

/// Algorithm 3: keep the `m` closest candidates by distance ascending.
/// Stable: ties break by ascending id so the same candidate set + same
/// `m` always returns the same neighbor list (required for reproducible
/// builds). Kept as a reference baseline and for unit tests; build calls
/// the heuristic.
#[allow(dead_code)]
pub(super) fn select_neighbors_simple(candidates: &[Candidate], m: usize) -> Vec<u32> {
    let mut sorted: Vec<Candidate> = candidates.to_vec();
    sorted.sort_by(|a, b| crate::order::cmp_dist_asc(a.dist, b.dist).then_with(|| a.id.cmp(&b.id)));
    sorted.truncate(m);
    sorted.into_iter().map(|c| c.id).collect()
}

/// Algorithm 4 (Malkov & Yashunin 2018): heuristic neighbor selection
/// with diversity.
///
/// `candidates` are pre-scored against the query (the node being inserted
/// or the node whose backlink list overflowed). `store` decodes rows on
/// the fly (into `sa`/`sb`) so we can compute candidate-to-candidate
/// distances without an f32 snapshot. The `keep_pruned` refill is on by
/// default in build to guarantee neighbor lists actually reach `m` even
/// when most candidates get shadowed.
///
/// Determinism: ties in distance break by ascending id so the same
/// inputs always produce the same output ordering.
#[allow(clippy::too_many_arguments)]
pub(super) fn select_neighbors_heuristic(
    candidates: &[Candidate],
    m: usize,
    store: &PackedVectors,
    dim: usize,
    sa: &mut [f32],
    sb: &mut [f32],
    keep_pruned: bool,
) -> Vec<u32> {
    if candidates.is_empty() || m == 0 {
        return Vec::new();
    }

    let mut working: Vec<Candidate> = candidates.to_vec();
    working
        .sort_by(|a, b| crate::order::cmp_dist_asc(a.dist, b.dist).then_with(|| a.id.cmp(&b.id)));

    let mut chosen: Vec<Candidate> = Vec::with_capacity(m);
    let mut pruned: Vec<Candidate> = Vec::new();
    for cand in working {
        if chosen.len() >= m {
            break;
        }
        let cand_row = store.row(cand.id as usize, dim, sa);
        let mut shadowed = false;
        for chosen_one in &chosen {
            let chosen_row = store.row(chosen_one.id as usize, dim, sb);
            let d_cc = cosine_dist(cand_row, chosen_row);
            // Candidate is shadowed: an already-chosen neighbor is
            // closer to it than the query is. Adding it gives no new
            // angular coverage.
            if d_cc < cand.dist {
                shadowed = true;
                break;
            }
        }
        if shadowed {
            pruned.push(cand);
        } else {
            chosen.push(cand);
        }
    }

    if keep_pruned && chosen.len() < m {
        // Pruned came in distance-ascending order via `working`; preserve
        // that order so the closest-shadowed are picked first.
        for cand in pruned {
            if chosen.len() >= m {
                break;
            }
            chosen.push(cand);
        }
    }

    chosen.into_iter().map(|c| c.id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(id: u32, dist: f32) -> Candidate {
        Candidate { id, dist }
    }

    #[test]
    fn simple_keeps_top_m_by_distance() {
        let cs = vec![cand(0, 0.5), cand(1, 0.1), cand(2, 0.3), cand(3, 0.2)];
        assert_eq!(select_neighbors_simple(&cs, 2), vec![1, 3]);
        assert_eq!(select_neighbors_simple(&cs, 4), vec![1, 3, 2, 0]);
    }

    #[test]
    fn simple_breaks_ties_by_id() {
        let cs = vec![cand(2, 0.1), cand(0, 0.1), cand(1, 0.1)];
        assert_eq!(select_neighbors_simple(&cs, 2), vec![0, 1]);
    }

    #[test]
    fn heuristic_returns_at_most_m_and_includes_closest() {
        // 4 unit vectors at the four 2D axes. Query is implicitly the
        // node being inserted; we encode dist-to-query directly via
        // cand.dist. Cand 0 is the closest by construction.
        let vectors = vec![
            1.0, 0.0, // 0: +x
            0.0, 1.0, // 1: +y
            -1.0, 0.0, // 2: -x
            0.0, -1.0, // 3: -y
        ];
        let cs = vec![cand(0, 0.0), cand(1, 0.5), cand(2, 1.0), cand(3, 1.5)];
        let store = PackedVectors::F32(vectors);
        let mut sa = store.scratch(2);
        let mut sb = store.scratch(2);
        let picked = select_neighbors_heuristic(&cs, 2, &store, 2, &mut sa, &mut sb, true);
        assert!(picked.len() <= 2, "len={} > 2", picked.len());
        assert!(picked.contains(&0), "closest must always be picked");
    }

    #[test]
    fn heuristic_drops_shadowed_candidate() {
        // 3 vectors: 0 and 1 are nearly identical, 2 is orthogonal.
        // Query is closer to 0 than to 1 by a hair. With m=2, simple
        // would pick {0, 1} (the two closest). Heuristic picks {0, 2}
        // because 1 is shadowed by 0 (their mutual distance is tiny,
        // smaller than 1's distance to the query).
        let vectors = vec![
            1.0, 0.0, // 0: +x
            0.99, 0.01, // 1: nearly +x
            0.0, 1.0, // 2: +y
        ];
        let cs = vec![cand(0, 0.05), cand(1, 0.06), cand(2, 1.0)];
        let store = PackedVectors::F32(vectors);
        let mut sa = store.scratch(2);
        let mut sb = store.scratch(2);
        let picked = select_neighbors_heuristic(&cs, 2, &store, 2, &mut sa, &mut sb, true);
        assert_eq!(picked.len(), 2);
        assert!(picked.contains(&0));
        assert!(
            picked.contains(&2),
            "heuristic should reject shadowed 1 in favor of orthogonal 2: got {:?}",
            picked
        );
    }

    #[test]
    fn heuristic_refills_when_under_m() {
        // All 3 vectors collinear → 1 and 2 are shadowed by 0. With
        // keep_pruned=true and m=3, all three must come back (in
        // closest-first order).
        let vectors = vec![
            1.0, 0.0, // 0
            0.99, 0.01, // 1: shadowed by 0
            0.98, 0.02, // 2: shadowed by 0 and 1
        ];
        let cs = vec![cand(0, 0.05), cand(1, 0.06), cand(2, 0.07)];
        let store = PackedVectors::F32(vectors);
        let mut sa = store.scratch(2);
        let mut sb = store.scratch(2);
        let picked = select_neighbors_heuristic(&cs, 3, &store, 2, &mut sa, &mut sb, true);
        assert_eq!(picked.len(), 3);
        assert_eq!(picked, vec![0, 1, 2]);
    }
}
