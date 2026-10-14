//! graph section codecs (chunk-to-chunk csr adjacency and, later, the
//! entity/property-graph sections). all are OPTIONAL and EXCLUDED from
//! content_hash, additive within frozen format v1, and never touch the
//! embedding hot path. G1 ships only `adjacency` (SECTION_GRAPH_ADJACENCY,
//! 0x0C); the entity sections (0x11..=0x13) land in G2.

mod adjacency;

pub use adjacency::{
    CsrParts, EDGE_TYPE_CITATION, EDGE_TYPE_NEXT_CHUNK, EDGE_TYPE_SEMANTIC, Edge,
    GRAPH_ADJACENCY_PAYLOAD_VERSION, MAX_DEGREE as GRAPH_MAX_DEGREE, decode_graph_adjacency,
    encode_graph_adjacency, parse_csr_parts,
};
