//! Deterministic serialization for the .fci bundle.
//!
//! struct fields serialize in declaration order and the formatter is
//! compact (no whitespace), so the same logical bundle produces
//! byte-identical bytes on any machine. strings are stored VERBATIM:
//! extractors are responsible for producing NFC canonical text, and
//! re-normalizing here would desync the chunk_id the adapter derives from
//! the same text.

use super::FciBundle;
use crate::error::ForgeError;
use serde::Serialize;

impl FciBundle {
    /// Serialize to canonical, deterministic bytes (compact json,
    /// declaration-order fields). same bundle -> same bytes everywhere.
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, ForgeError> {
        let mut buf = Vec::new();
        let mut ser =
            serde_json::Serializer::with_formatter(&mut buf, serde_json::ser::CompactFormatter);
        self.serialize(&mut ser)
            .map_err(|e| ForgeError::Serialize(e.to_string()))?;
        Ok(buf)
    }

    /// Parse a bundle from canonical bytes and validate it. fail closed on
    /// malformed bytes, an unknown schema version, or a dangling reference.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ForgeError> {
        let bundle: FciBundle =
            serde_json::from_slice(bytes).map_err(|e| ForgeError::Deserialize(e.to_string()))?;
        bundle.validate()?;
        Ok(bundle)
    }
}

#[cfg(test)]
mod tests {
    use crate::fci::{
        BlobRef, ChunkRecord, Edge, EmbeddingRequest, Entity, FCI_SCHEMA_VERSION, FciBundle,
        MentionSpan, PayloadRef, SpaceTag,
    };

    fn sample() -> FciBundle {
        let mut b = FciBundle::new();
        b.chunks.push(ChunkRecord {
            canonical_text: "vacina contra a covid".into(),
            source_uri: "doc.txt".into(),
            byte_start: 0,
            byte_end: 21,
        });
        b.chunks.push(ChunkRecord {
            canonical_text: "fonte oficial".into(),
            source_uri: "doc.txt".into(),
            byte_start: 21,
            byte_end: 34,
        });
        b.embedding_requests.push(EmbeddingRequest {
            chunk_index: 0,
            space: SpaceTag::Text,
            modelhash: format!("sha256:{}", "0".repeat(64)),
            payload_ref: PayloadRef::InlineText,
        });
        b.embedding_requests.push(EmbeddingRequest {
            chunk_index: 1,
            space: SpaceTag::Image,
            modelhash: format!("sha256:{}", "1".repeat(64)),
            payload_ref: PayloadRef::BlobHash([7u8; 32]),
        });
        b.entities.push(Entity {
            id: 1,
            kind: "topic".into(),
            canonical_name: "covid".into(),
            mentions: vec![MentionSpan {
                chunk_index: 0,
                byte_start: 13,
                byte_end: 18,
            }],
        });
        b.entities.push(Entity {
            id: 2,
            kind: "source".into(),
            canonical_name: "oficial".into(),
            mentions: vec![],
        });
        b.edges.push(Edge {
            src: 1,
            dst: 2,
            edge_type: "cited_by".into(),
            weight: 0.5,
        });
        b.blobs.push(BlobRef {
            content_hash: [3u8; 32],
            original_uri: "doc.txt".into(),
            byte_len: 34,
            inlined: true,
        });
        b
    }

    #[test]
    fn round_trips_byte_identical() {
        let b = sample();
        let bytes = b.to_canonical_bytes().unwrap();
        let back = FciBundle::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(b, back, "deserialize must reconstruct the bundle exactly");
        assert_eq!(
            back.to_canonical_bytes().unwrap(),
            bytes,
            "re-serialization must be byte-identical"
        );
    }

    #[test]
    fn two_builds_of_same_content_are_byte_identical() {
        assert_eq!(
            sample().to_canonical_bytes().unwrap(),
            sample().to_canonical_bytes().unwrap(),
            "same canonical content -> byte-identical .fci"
        );
    }

    #[test]
    fn empty_bundle_is_stamped_with_schema_version() {
        let b = FciBundle::new();
        assert_eq!(b.schema_version, FCI_SCHEMA_VERSION);
        b.validate().unwrap();
    }

    #[test]
    fn validate_rejects_dangling_chunk_index() {
        let mut b = FciBundle::new();
        b.embedding_requests.push(EmbeddingRequest {
            chunk_index: 0,
            space: SpaceTag::Text,
            modelhash: "sha256:x".into(),
            payload_ref: PayloadRef::InlineText,
        });
        assert!(
            b.validate().is_err(),
            "request with no chunk must be rejected"
        );
    }

    #[test]
    fn validate_rejects_edge_to_unknown_entity() {
        let mut b = FciBundle::new();
        b.edges.push(Edge {
            src: 1,
            dst: 99,
            edge_type: "x".into(),
            weight: 1.0,
        });
        assert!(
            b.validate().is_err(),
            "edge to unknown entity must be rejected"
        );
    }

    #[test]
    fn validate_rejects_unsupported_schema_version() {
        let mut b = FciBundle::new();
        b.schema_version = FCI_SCHEMA_VERSION + 1;
        let bytes = serde_json::to_vec(&b).unwrap();
        assert!(
            FciBundle::from_canonical_bytes(&bytes).is_err(),
            "future schema version must fail closed"
        );
    }
}
