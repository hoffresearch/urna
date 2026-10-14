//! Entities and typed edges. Extraction is a forge-belt concern (later,
//! and allowed to be non-deterministic), but the .fci stores entities and
//! edges in a fixed shape so the deterministic serialize is the
//! reproducibility anchor: a fixed entity/edge set serializes
//! byte-identically regardless of how it was extracted.

use serde::{Deserialize, Serialize};

/// A mention of an entity inside a chunk, as a byte span into that
/// chunk's canonical text, so an entity resolves back to a citable span.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MentionSpan {
    /// Index into `FciBundle::chunks`.
    pub chunk_index: u64,
    pub byte_start: u64,
    pub byte_end: u64,
}

/// An extracted entity: a stable id, a type/kind, a canonical name, and
/// the spans where it is mentioned. ids are assigned by the producer and
/// are the join key for edges.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entity {
    pub id: u64,
    pub kind: String,
    pub canonical_name: String,
    pub mentions: Vec<MentionSpan>,
}

/// A typed, weighted edge between two entities (by id). `weight` is f32
/// so this type is `PartialEq` only, not `Eq`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub src: u64,
    pub dst: u64,
    pub edge_type: String,
    pub weight: f32,
}
