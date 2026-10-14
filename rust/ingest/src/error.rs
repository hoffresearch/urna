//! Typed errors for urna-ingest. Never panic in library code.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeError {
    /// The bundle could not be serialized to canonical bytes.
    #[error("fci serialize: {0}")]
    Serialize(String),
    /// The bytes could not be parsed as an .fci bundle.
    #[error("fci deserialize: {0}")]
    Deserialize(String),
    /// The bundle parsed but violates a schema invariant (a dangling
    /// chunk_index, an edge to an unknown entity id, etc.).
    #[error("fci invalid: {0}")]
    Invalid(String),
    /// The bundle declares a schema version this build does not support.
    /// Fail closed rather than guess at unknown layout.
    #[error("unsupported fci schema version {found}, this build supports {supported}")]
    UnsupportedSchemaVersion { found: u32, supported: u32 },
}
