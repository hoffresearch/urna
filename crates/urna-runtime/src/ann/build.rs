//! HNSW construction: insert nodes one at a time, link bidirectionally,
//! and call `select_neighbors_heuristic` (Algorithm 4) at every neighbor
//! selection point. Build is deterministic given the same seed: level
//! distribution comes from a fixed LCG, neighbor lists are sorted by id
//! when serialized.

use urna_format::Int8EmbeddingsView;

use super::search::{greedy_search, layer_search};
use super::select_neighbors::select_neighbors_heuristic;
use super::visited::VisitedList;
use super::{Candidate, HnswIndex, Node, dist_rr};
use crate::materialize::PackedVectors;

impl HnswIndex {
    /// Build an HNSW index from f32 vectors. Deterministic given the
    /// same `seed`. `vectors` is row-major `n*dim`.
    pub fn build(
        vectors: Vec<f32>,
        n: usize,
        dim: usize,
        m: usize,
        ef_construction: usize,
        seed: u64,
    ) -> Self {
        assert_eq!(vectors.len(), n * dim);
        // Build distances run over the f32 build store; `row()` borrows
        // each row directly so the graph is byte-identical to the old
        // `&[f32]` path, while the same `PackedVectors` type lets the
        // runtime search path stay packed (no f32 snapshot).
        let store = PackedVectors::F32(vectors);
        let m_max0 = m * 2;
        let mut rng = LcgRng::new(seed);

        // one visited list for the whole build, cleared per layer search
        // (an epoch bump), instead of a fresh hash set per call.
        let mut visited = VisitedList::new(n);
        let mut nodes: Vec<Node> = Vec::with_capacity(n);
        for _ in 0..n {
            let level = sample_level(&mut rng, m);
            let mut neighbors = Vec::with_capacity((level as usize) + 1);
            for _ in 0..=level {
                neighbors.push(Vec::new());
            }
            nodes.push(Node { level, neighbors });
        }

        // Insert nodes in id order for determinism.
        let mut entry_point: u32 = 0;
        let mut max_level: u32 = 0;

        for i in 0..n {
            let level = nodes[i].level;
            if i == 0 {
                entry_point = 0;
                max_level = level;
                continue;
            }

            // 1. Greedy walk from entry_point down to layer (level+1) to
            //    find the closest entry on the layer above this node.
            //    `store` is the f32 build store, so `q` borrows it directly
            //    and `q_scratch` stays empty (zero overhead at build time).
            let mut q_scratch = store.scratch(dim);
            let mut sa = store.scratch(dim);
            let mut sb = store.scratch(dim);
            let q = store.row(i, dim, &mut q_scratch);
            let mut curr = entry_point;
            for layer in (level + 1..=max_level).rev() {
                curr = greedy_search(curr, q, layer, &nodes, &store, dim);
            }

            // 2. From `curr`, do an `ef_construction`-sized search on
            //    each layer from `min(level, max_level)` down to 0;
            //    select top neighbors and link bidirectionally.
            let mut entry = curr;
            let start_layer = level.min(max_level);
            for layer in (0..=start_layer).rev() {
                visited.clear();
                let candidates = layer_search(
                    &[entry],
                    q,
                    layer,
                    ef_construction,
                    &nodes,
                    &store,
                    dim,
                    i as u32,
                    &mut visited,
                );
                // The new node always picks `m` neighbors (Algorithm 4 with
                // M=m). The asymmetry — layer 0 allowing up to `m_max0`
                // neighbors per node — only kicks in via backlinks: when
                // neighbor's list overflows `cap_layer`, we re-select with
                // the heuristic to bring it back down to `cap_layer`.
                let cap_layer = if layer == 0 { m_max0 } else { m };
                let neighbor_ids =
                    select_neighbors_heuristic(&candidates, m, &store, dim, &mut sa, &mut sb, true);
                nodes[i].neighbors[layer as usize] = neighbor_ids.clone();

                // Backlinks. Insert i into each neighbor's list and prune
                // if it overflows `cap_layer`.
                for &nbr in &neighbor_ids {
                    let nbr_idx = nbr as usize;
                    let layer_idx = layer as usize;
                    if layer_idx >= nodes[nbr_idx].neighbors.len() {
                        continue;
                    }
                    if nodes[nbr_idx].neighbors[layer_idx].len() >= cap_layer {
                        // Prune: keep the cap_layer best of (existing + i).
                        // Clone the ids first so the per-row decode borrows
                        // do not collide with the later reassignment.
                        let existing = nodes[nbr_idx].neighbors[layer_idx].clone();
                        let mut all: Vec<Candidate> = Vec::with_capacity(existing.len() + 1);
                        for id in existing {
                            let dist = dist_rr(&store, nbr_idx, id as usize, dim, &mut sa, &mut sb);
                            all.push(Candidate { id, dist });
                        }
                        let dist = dist_rr(&store, nbr_idx, i, dim, &mut sa, &mut sb);
                        all.push(Candidate { id: i as u32, dist });
                        nodes[nbr_idx].neighbors[layer_idx] = select_neighbors_heuristic(
                            &all, cap_layer, &store, dim, &mut sa, &mut sb, true,
                        );
                    } else if !nodes[nbr_idx].neighbors[layer_idx].contains(&(i as u32)) {
                        nodes[nbr_idx].neighbors[layer_idx].push(i as u32);
                    }
                }

                // Next-layer entry is the closest *selected* neighbor (post-
                // heuristic). hnswlib does the same. Falls back to the raw
                // beam closest if the heuristic returned nothing.
                if let Some(&first) = neighbor_ids.first() {
                    entry = first;
                } else if !candidates.is_empty() {
                    entry = candidates[0].id;
                }
            }

            if level > max_level {
                max_level = level;
                entry_point = i as u32;
            }
        }

        Self {
            m,
            m_max0,
            ef_construction,
            entry_point,
            max_level,
            nodes,
            store,
            dim,
            n,
            ef_search: ef_construction,
        }
    }

    /// Build from int8 quantized embeddings. Dequantizes once (lossy) and
    /// hands off to `build`. Recall ends up bounded by quantization
    /// noise; in practice still well above 0.95 @ k=10 for real corpora.
    pub fn build_from_int8(view: &Int8EmbeddingsView<'_>, m: usize, ef: usize, seed: u64) -> Self {
        let mut vectors = vec![0.0f32; view.n * view.dim];
        for i in 0..view.n {
            let scale = view.scale(i);
            let row = view.row(i);
            for j in 0..view.dim {
                vectors[i * view.dim + j] = row[j] as f32 * scale;
            }
        }
        Self::build(vectors, view.n, view.dim, m, ef, seed)
    }

    /// Build from float16 LE bytes. Decodes once into f32 and builds.
    pub fn build_from_f16(
        bytes: &[u8],
        n: usize,
        dim: usize,
        m: usize,
        ef: usize,
        seed: u64,
    ) -> Self {
        let vectors = urna_format::f16_bytes_to_f32(bytes);
        Self::build(vectors, n, dim, m, ef, seed)
    }

    /// Build from raw f32 LE bytes. Copies into an owned buffer.
    pub fn build_from_f32(
        bytes: &[u8],
        n: usize,
        dim: usize,
        m: usize,
        ef: usize,
        seed: u64,
    ) -> Self {
        let mut vectors = vec![0.0f32; n * dim];
        for (i, slot) in vectors.iter_mut().enumerate() {
            let off = i * 4;
            *slot =
                f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        }
        Self::build(vectors, n, dim, m, ef, seed)
    }
}

/// Geometric level distribution. Deterministic via the LCG state.
pub(super) fn sample_level(rng: &mut LcgRng, m: usize) -> u32 {
    let m_l = 1.0 / (m as f64).ln();
    let r = rng.next_f64();
    if r <= 0.0 {
        return 0;
    }
    let level = (-(r.ln()) * m_l).floor() as i64;
    level.clamp(0, 31) as u32 // cap at 31 layers
}

/// Tiny LCG (deterministic, no rand dep). `pub(super)` so tests in
/// `super::tests` can reuse it for synthetic vector generation.
pub(super) struct LcgRng {
    state: u64,
}
impl LcgRng {
    pub(super) fn new(seed: u64) -> Self {
        Self {
            state: seed
                .wrapping_mul(2862933555777941757)
                .wrapping_add(3037000493),
        }
    }
    pub(super) fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
    pub(super) fn next_f64(&mut self) -> f64 {
        // Map upper 53 bits to [0, 1).
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}
