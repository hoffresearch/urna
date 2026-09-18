//! HNSW search: greedy descent on upper layers, ef-bounded beam search
//! on layer 0. Returns candidates sorted ascending by distance; the
//! runtime reranks with the exact dot product to produce the final
//! cosine score.

use std::collections::{BinaryHeap, HashSet};

use super::visited::VisitSet;
use super::{Candidate, HnswIndex, Node, dist_q};
use crate::materialize::PackedVectors;

impl HnswIndex {
    /// Attach f32 vectors as the search store. Kept for callers that
    /// already hold an f32 buffer (build, tests); the runtime open path
    /// uses `attach_store` to avoid an f32 expansion.
    pub fn attach_vectors(&mut self, vectors: Vec<f32>) {
        debug_assert_eq!(vectors.len(), self.n * self.dim);
        self.store = PackedVectors::F32(vectors);
    }

    /// Attach a packed vector store at open time. Keeps int8/f16 rows in
    /// their on-disk packing so the resident footprint is the packed size,
    /// not the old `n*dim*4` f32 snapshot.
    pub(crate) fn attach_store(&mut self, store: PackedVectors) {
        self.store = store;
    }

    /// Level-0 (densest layer) out-neighbors of node `i`, or an empty slice
    /// if `i` is out of range. Read-only accessor the graph build path uses to
    /// derive top-m semantic edges from the already-built hnsw graph without
    /// re-running an O(n^2) exact knn. The neighbor SET is the build-order
    /// hnsw adjacency; the graph builder sorts canonically before writing.
    pub fn level0_neighbors(&self, i: usize) -> &[u32] {
        self.nodes
            .get(i)
            .and_then(|node| node.neighbors.first())
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Search for the `ef` closest candidates to `q`. Returns ids only —
    /// the runtime reranks with the exact dot product to produce the
    /// final cosine score.
    pub fn search(&self, q: &[f32], ef: usize) -> Vec<usize> {
        if self.n == 0 {
            return Vec::new();
        }
        if !self.store.is_attached() {
            // Index is loaded but vectors haven't been attached. Return
            // an empty candidate set; runtime falls back to exact.
            return Vec::new();
        }
        let mut curr = self.entry_point;
        for layer in (1..=self.max_level).rev() {
            curr = greedy_search(curr, q, layer, &self.nodes, &self.store, self.dim);
        }
        let mut visited: HashSet<u32> = HashSet::new();
        let candidates = layer_search(
            &[curr],
            q,
            0,
            ef.max(self.ef_search),
            &self.nodes,
            &self.store,
            self.dim,
            u32::MAX,
            &mut visited,
        );
        candidates.into_iter().map(|c| c.id as usize).collect()
    }
}

/// Greedy descent on a single layer until no neighbor is closer.
pub(super) fn greedy_search(
    entry: u32,
    q: &[f32],
    layer: u32,
    nodes: &[Node],
    store: &PackedVectors,
    dim: usize,
) -> u32 {
    let mut scratch = store.scratch(dim);
    let mut curr = entry;
    let mut curr_dist = dist_q(store, q, curr as usize, dim, &mut scratch);
    loop {
        let layer_idx = layer as usize;
        if layer_idx >= nodes[curr as usize].neighbors.len() {
            return curr;
        }
        let nbrs = &nodes[curr as usize].neighbors[layer_idx];
        let mut best = curr;
        let mut best_dist = curr_dist;
        for &nbr in nbrs {
            let d = dist_q(store, q, nbr as usize, dim, &mut scratch);
            if d < best_dist {
                best = nbr;
                best_dist = d;
            }
        }
        if best == curr {
            return curr;
        }
        curr = best;
        curr_dist = best_dist;
    }
}

/// Search a single layer with a candidate list of size `ef`. Returns
/// the best `ef` candidates sorted ascending by distance.
#[allow(clippy::too_many_arguments)]
pub(super) fn layer_search(
    entries: &[u32],
    q: &[f32],
    layer: u32,
    ef: usize,
    nodes: &[Node],
    store: &PackedVectors,
    dim: usize,
    skip_id: u32,
    visited: &mut impl VisitSet,
) -> Vec<Candidate> {
    let mut scratch = store.scratch(dim);
    // BinaryHeap orderings:
    //   `frontier` — min-heap by distance (closest first to expand).
    //   `result`   — max-heap by distance (so we can prune the farthest).
    let mut frontier: BinaryHeap<ByDistAsc> = BinaryHeap::new();
    let mut result: BinaryHeap<ByDistDesc> = BinaryHeap::new();

    for &e in entries {
        if e == skip_id {
            continue;
        }
        let d = dist_q(store, q, e as usize, dim, &mut scratch);
        let c = Candidate { id: e, dist: d };
        frontier.push(ByDistAsc(c));
        result.push(ByDistDesc(c));
        visited.insert(e);
    }

    while let Some(ByDistAsc(curr)) = frontier.pop() {
        let worst_in_result = result.peek().map(|r| r.0.dist).unwrap_or(f32::INFINITY);
        if curr.dist > worst_in_result && result.len() >= ef {
            break;
        }
        let layer_idx = layer as usize;
        if layer_idx >= nodes[curr.id as usize].neighbors.len() {
            continue;
        }
        let nbrs = &nodes[curr.id as usize].neighbors[layer_idx];
        for &nbr in nbrs {
            if nbr == skip_id || !visited.insert(nbr) {
                continue;
            }
            let d = dist_q(store, q, nbr as usize, dim, &mut scratch);
            let worst = result.peek().map(|r| r.0.dist).unwrap_or(f32::INFINITY);
            if result.len() < ef || d < worst {
                let c = Candidate { id: nbr, dist: d };
                frontier.push(ByDistAsc(c));
                result.push(ByDistDesc(c));
                if result.len() > ef {
                    result.pop();
                }
            }
        }
    }

    let mut out: Vec<Candidate> = result.into_iter().map(|w| w.0).collect();
    out.sort_by(|a, b| crate::order::cmp_dist_asc(a.dist, b.dist));
    out
}

// `BinaryHeap` is a max-heap; we want closest-first / farthest-first
// orderings. Define orderings explicitly.
#[derive(Clone, Copy)]
struct ByDistAsc(Candidate);
#[derive(Clone, Copy)]
struct ByDistDesc(Candidate);

impl PartialEq for ByDistAsc {
    fn eq(&self, other: &Self) -> bool {
        self.0.dist == other.0.dist
    }
}
impl Eq for ByDistAsc {}
impl Ord for ByDistAsc {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse so BinaryHeap (max-heap) pops smallest distance.
        other
            .0
            .dist
            .partial_cmp(&self.0.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}
impl PartialOrd for ByDistAsc {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ByDistDesc {
    fn eq(&self, other: &Self) -> bool {
        self.0.dist == other.0.dist
    }
}
impl Eq for ByDistDesc {}
impl Ord for ByDistDesc {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .dist
            .partial_cmp(&other.0.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}
impl PartialOrd for ByDistDesc {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
