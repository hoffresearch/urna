//! Typed errors for urna-ingest. Never panic in library code.

#[derive(Debug)]
pub enum ForgeError {
    /// The bundle could not be serialized to canonical bytes.
    Serialize(String),
    /// The bytes could not be parsed as an .fci bundle.
    Deserialize(String),
    /// The bundle parsed but violates a schema invariant (a dangling
    /// chunk_index, an edge to an unknown entity id, etc.).
    Invalid(String),
    /// The bundle declares a schema version this build does not support.
    /// Fail closed rather than guess at unknown layout.
    UnsupportedSchemaVersion { found: u32, supported: u32 },
}

impl std::fmt::Display for ForgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Serialize(v0) => write!(f, "fci serialize: {v0}"),
            Self::Deserialize(v0) => write!(f, "fci deserialize: {v0}"),
            Self::Invalid(v0) => write!(f, "fci invalid: {v0}"),
            Self::UnsupportedSchemaVersion { found, supported } => write!(
                f,
                "unsupported fci schema version {found}, this build supports {supported}"
            ),
        }
    }
}

impl std::error::Error for ForgeError {}
