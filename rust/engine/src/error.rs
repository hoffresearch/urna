use thiserror::Error;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Format(#[from] urna_format::UrnaError),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("invalid k: {0}")]
    InvalidK(i32),
    #[error("empty query")]
    EmptyQuery,
    #[error("zero-norm query")]
    ZeroNormQuery,
    #[error("NaN or Inf in query")]
    InvalidQueryValue,
    #[error("embedding space not found: {0}")]
    SpaceNotFound(String),
    #[error(
        "blob {index} is not inlined in this file: open the media sidecar \
         named by its blob_refs uri, or rebuild with [output] embed_media"
    )]
    BlobNotInlined { index: usize },
    #[error(
        "model_hash mismatch in space {space}: the query was embedded with {expected}, \
         but the space vectors were embedded with {actual}"
    )]
    SpaceModelMismatch {
        space: String,
        expected: String,
        actual: String,
    },
}
