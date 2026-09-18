//! the .fci (forge canonical intermediate) schema, FROZEN at
//! `FCI_SCHEMA_VERSION`. it is the stable contract between forge (which
//! produces it from messy inputs) and the python adapter (which feeds it
//! into the existing builder.Pipeline + urna.build). the schema is
//! versioned INDEPENDENTLY of URNA_FORMAT_VERSION because .fci is a forge
//! artifact, never a .urna section, so it can evolve without touching the
//! frozen container.

mod blob_ref;
mod embedding_request;
mod entity;
mod record;
mod serialize;

pub use blob_ref::BlobRef;
pub use embedding_request::{EmbeddingRequest, PayloadRef, SpaceTag};
pub use entity::{Edge, Entity, MentionSpan};
pub use record::ChunkRecord;

use crate::error::ForgeError;
use serde::{Deserialize, Serialize};

/// Frozen .fci schema version, independent of URNA_FORMAT_VERSION. bumped
/// only when the .fci layout changes meaning; readers fail closed on an
/// unknown version.
pub const FCI_SCHEMA_VERSION: u32 = 1;

/// A forge canonical intermediate bundle: canonical-text shards, the
/// per-modality embedding requests over them, extracted entities + typed
/// edges, and the blob manifest. serialized deterministically so the same
/// canonical input yields byte-identical .fci, the upstream half of urna's
/// reproducible build.
///
/// field order is fixed by declaration and the serializer is compact, so
/// `to_canonical_bytes` is stable across machines for the same content.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FciBundle {
    pub schema_version: u32,
    pub chunks: Vec<ChunkRecord>,
    pub embedding_requests: Vec<EmbeddingRequest>,
    pub entities: Vec<Entity>,
    pub edges: Vec<Edge>,
    pub blobs: Vec<BlobRef>,
}

impl Default for FciBundle {
    fn default() -> Self {
        Self {
            schema_version: FCI_SCHEMA_VERSION,
            chunks: Vec::new(),
            embedding_requests: Vec::new(),
            entities: Vec::new(),
            edges: Vec::new(),
            blobs: Vec::new(),
        }
    }
}

impl FciBundle {
    /// An empty bundle stamped with the current schema version.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check the bundle is internally consistent and at a supported schema
    /// version. every cross-reference (request -> chunk, mention -> chunk,
    /// edge -> entity) must point at an existing target; otherwise the
    /// downstream adapter would silently mis-map a span. fail closed with a
    /// typed error, never panic.
    pub fn validate(&self) -> Result<(), ForgeError> {
        if self.schema_version != FCI_SCHEMA_VERSION {
            return Err(ForgeError::UnsupportedSchemaVersion {
                found: self.schema_version,
                supported: FCI_SCHEMA_VERSION,
            });
        }
        let n_chunks = self.chunks.len() as u64;
        for c in &self.chunks {
            if c.byte_end < c.byte_start {
                return Err(ForgeError::Invalid(format!(
                    "chunk span {}..{} is reversed",
                    c.byte_start, c.byte_end
                )));
            }
        }
        for r in &self.embedding_requests {
            if r.chunk_index >= n_chunks {
                return Err(ForgeError::Invalid(format!(
                    "embedding_request chunk_index {} out of range (n_chunks={n_chunks})",
                    r.chunk_index
                )));
            }
        }
        let mut entity_ids = std::collections::BTreeSet::new();
        for e in &self.entities {
            if !entity_ids.insert(e.id) {
                return Err(ForgeError::Invalid(format!("duplicate entity id {}", e.id)));
            }
            for m in &e.mentions {
                if m.chunk_index >= n_chunks {
                    return Err(ForgeError::Invalid(format!(
                        "entity {} mention chunk_index {} out of range",
                        e.id, m.chunk_index
                    )));
                }
            }
        }
        for edge in &self.edges {
            if !entity_ids.contains(&edge.src) || !entity_ids.contains(&edge.dst) {
                return Err(ForgeError::Invalid(format!(
                    "edge {}->{} references an unknown entity id",
                    edge.src, edge.dst
                )));
            }
        }
        Ok(())
    }
}
