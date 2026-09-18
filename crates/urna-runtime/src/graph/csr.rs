//! `CsrIndex`: the runtime reader over the graph_adjacency (0x0C) csr
//! payload. parses the offsets + neighbor columns off the section bytes
//! once at open time, then exposes `neighbors(node)` as a single contiguous
//! slice walked with zero allocation, mirroring `ann::HnswIndex::from_bytes`.
//! typed errors, never panics on a truncated mmap; the graph only ever
//! GENERATES candidates, it never scores.

use urna_format::error::UrnaError;
use urna_format::sections::graph::parse_csr_parts;

use crate::error::RuntimeError;

/// flat csr adjacency over chunk ordinals. `offsets[node..node+1]` bounds
/// node's run in `neighbors` (and `edge_types`), so one node's neighbors are
/// a contiguous `&[u32]` slice.
pub struct CsrIndex {
    n_nodes: usize,
    /// row pointers, len `n_nodes + 1`.
    offsets: Vec<u32>,
    /// absolute (decoded) neighbor ids, concatenated per node.
    neighbors: Vec<u32>,
    /// edge type per neighbor, aligned 1:1 with `neighbors`.
    edge_types: Vec<u8>,
}

impl CsrIndex {
    /// parse a graph_adjacency csr payload. `n_embeddings` is the corpus
    /// chunk count; the csr's `n_nodes` must match it so neighbor ids index
    /// the same ordinal space as the embeddings.
    pub fn from_bytes(bytes: &[u8], n_embeddings: usize) -> Result<Self, RuntimeError> {
        let parts = parse_csr_parts(bytes).map_err(RuntimeError::Format)?;
        if parts.n_nodes != n_embeddings {
            return Err(RuntimeError::Format(UrnaError::MalformedSectionPayload {
                section_id: urna_format::layout::SECTION_GRAPH_ADJACENCY,
                reason: format!(
                    "graph node count {} != n_embeddings {}",
                    parts.n_nodes, n_embeddings
                ),
            }));
        }
        // narrow the validated u64 columns to u32 (ids index a u32 ordinal
        // space; parse_csr_parts already range-checks dst < n_nodes and the
        // monotone offsets, so these casts cannot lose information).
        let offsets: Vec<u32> = parts.offsets.iter().map(|&o| o as u32).collect();
        Ok(Self {
            n_nodes: parts.n_nodes,
            offsets,
            neighbors: parts.neighbors,
            edge_types: parts.edge_types,
        })
    }

    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// node `node`'s out-neighbors as a contiguous slice. empty (not an
    /// error) for an out-of-range node so the bfs can probe freely.
    #[inline]
    pub fn neighbors(&self, node: usize) -> &[u32] {
        if node + 1 >= self.offsets.len() {
            return &[];
        }
        let start = self.offsets[node] as usize;
        let end = self.offsets[node + 1] as usize;
        &self.neighbors[start..end]
    }

    /// edge type of node's `i`-th out-edge (0-based within the node's run).
    #[inline]
    pub fn edge_type(&self, node: usize, i: usize) -> Option<u8> {
        if node + 1 >= self.offsets.len() {
            return None;
        }
        let start = self.offsets[node] as usize;
        let end = self.offsets[node + 1] as usize;
        let idx = start + i;
        if idx < end {
            Some(self.edge_types[idx])
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use urna_format::sections::graph::{
        EDGE_TYPE_NEXT_CHUNK, EDGE_TYPE_SEMANTIC, Edge, encode_graph_adjacency,
    };

    fn edge(src: u32, dst: u32, t: u8) -> Edge {
        Edge {
            src,
            dst,
            edge_type: t,
        }
    }

    #[test]
    fn neighbors_are_contiguous_and_correct() {
        // node 0 -> {1,2}, node 1 -> {2}, node 2 -> {}, node 3 -> {0}.
        let edges = vec![
            edge(0, 2, EDGE_TYPE_SEMANTIC),
            edge(0, 1, EDGE_TYPE_NEXT_CHUNK),
            edge(1, 2, EDGE_TYPE_NEXT_CHUNK),
            edge(3, 0, EDGE_TYPE_NEXT_CHUNK),
        ];
        let payload = encode_graph_adjacency(4, &edges).unwrap();
        let idx = CsrIndex::from_bytes(&payload, 4).unwrap();
        assert_eq!(idx.n_nodes(), 4);
        // canonical sort is (src, edge_type, dst): node 0's edges become
        // [NEXT_CHUNK->1, SEMANTIC->2].
        assert_eq!(idx.neighbors(0), &[1, 2]);
        assert_eq!(idx.neighbors(1), &[2]);
        assert_eq!(idx.neighbors(2), &[] as &[u32]);
        assert_eq!(idx.neighbors(3), &[0]);
        assert_eq!(idx.edge_type(0, 0), Some(EDGE_TYPE_NEXT_CHUNK));
        assert_eq!(idx.edge_type(0, 1), Some(EDGE_TYPE_SEMANTIC));
        assert_eq!(idx.edge_type(0, 2), None);
        // out-of-range node is empty, not a panic.
        assert_eq!(idx.neighbors(99), &[] as &[u32]);
    }

    #[test]
    fn rejects_node_count_mismatch() {
        let edges = vec![edge(0, 1, EDGE_TYPE_NEXT_CHUNK)];
        let payload = encode_graph_adjacency(2, &edges).unwrap();
        assert!(CsrIndex::from_bytes(&payload, 3).is_err());
    }

    #[test]
    fn truncated_payload_errors_never_panics() {
        let edges = vec![
            edge(0, 1, EDGE_TYPE_NEXT_CHUNK),
            edge(1, 2, EDGE_TYPE_SEMANTIC),
            edge(2, 0, EDGE_TYPE_SEMANTIC),
        ];
        let payload = encode_graph_adjacency(3, &edges).unwrap();
        for cut in 0..payload.len() {
            // every prefix must error or parse, never panic.
            let _ = CsrIndex::from_bytes(&payload[..cut], 3);
        }
    }

    #[test]
    fn empty_graph_parses() {
        let payload = encode_graph_adjacency(0, &[]).unwrap();
        let idx = CsrIndex::from_bytes(&payload, 0).unwrap();
        assert_eq!(idx.n_nodes(), 0);
        assert_eq!(idx.neighbors(0), &[] as &[u32]);
    }
}
