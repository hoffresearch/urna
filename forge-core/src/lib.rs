//! forge-core: the forge canonical intermediate (.fci) schema and the
//! deterministic core of the forge ingestion layer.
//!
//! forge is the messy half of urna: it turns raw heterogeneous files
//! (pdf, image, font, dataset, dump) into a deterministic canonical
//! intermediate that the sovereign urna build path already eats. it lives
//! in its OWN cargo workspace, OUTSIDE crates/, so its dependency tree
//! never touches urna-format or urna-runtime and the byte-identical
//! container guarantee stays untainted.
//!
//! this crate is FORGE-0a: the FROZEN .fci schema only. it owns the stable
//! contract between forge (produces) and the python adapter (consumes).
//! it deliberately does NOT:
//!   - chunk text. there is ONE authoritative chunker, builder.chunk_text,
//!     called by the python adapter; forge-core never duplicates it.
//!   - run models or heavy extraction. normalization, native extractors,
//!     and the toolbelt land later and stay behind a subprocess contract.
//!
//! the determinism anchor is the CANONICAL EXTRACTED TEXT plus the
//! per-space model fingerprint: same canonical text + same fingerprints
//! => byte-identical .fci => byte-identical .urna. never panic in this
//! library; every failure is a typed `ForgeError`.

pub mod error;
pub mod fci;

pub use error::ForgeError;
pub use fci::{
    BlobRef, ChunkRecord, Edge, EmbeddingRequest, Entity, FCI_SCHEMA_VERSION, FciBundle,
    MentionSpan, PayloadRef, SpaceTag,
};
