#[derive(Debug)]
pub enum RuntimeError {
    Format(urna_format::UrnaError),
    Io(std::io::Error),
    DimensionMismatch {
        expected: usize,
        got: usize,
    },
    InvalidK(i32),
    EmptyQuery,
    ZeroNormQuery,
    InvalidQueryValue,
    SpaceNotFound(String),
    Filter(String),
    BlobNotInlined {
        index: usize,
    },
    SpaceModelMismatch {
        space: String,
        expected: String,
        actual: String,
    },
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Format(v0) => std::fmt::Display::fmt(v0, f),
            Self::Io(v0) => write!(f, "{v0}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::InvalidK(v0) => write!(f, "invalid k: {v0}"),
            Self::EmptyQuery => f.write_str("empty query"),
            Self::ZeroNormQuery => f.write_str("zero-norm query"),
            Self::InvalidQueryValue => f.write_str("NaN or Inf in query"),
            Self::SpaceNotFound(v0) => write!(f, "embedding space not found: {v0}"),
            Self::Filter(v0) => write!(f, "filter: {v0}"),
            Self::BlobNotInlined { index } => write!(
                f,
                "blob {index} is not inlined in this file: open the media sidecar \
                 named by its blob_refs uri, or rebuild with [output] embed_media"
            ),
            Self::SpaceModelMismatch {
                space,
                expected,
                actual,
            } => write!(
                f,
                "model_hash mismatch in space {space}: the query was embedded with {expected}, \
                 but the space vectors were embedded with {actual}"
            ),
        }
    }
}

impl std::error::Error for RuntimeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Format(e) => std::error::Error::source(e),
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<urna_format::UrnaError> for RuntimeError {
    fn from(e: urna_format::UrnaError) -> Self {
        Self::Format(e)
    }
}

impl From<std::io::Error> for RuntimeError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
