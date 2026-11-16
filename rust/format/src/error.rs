#[derive(Debug)]
pub enum UrnaError {
    Io(std::io::Error),
    MagicMismatch {
        expected: [u8; 4],
        got: [u8; 4],
    },
    UnsupportedVersion(u16, u16),
    UnsupportedFormatVersion(u32),
    UnsupportedSchemaVersion(u32),
    InvalidHeaderChecksum,
    SectionOffsetOutOfBounds {
        section_id: u32,
        offset: u64,
    },
    SectionChecksumMismatch(u32),
    SectionNotFound(u32),
    UnsupportedSectionVersion {
        section_id: u32,
        version: u32,
    },
    UnsupportedSectionEncoding {
        section_id: u32,
        encoding: u32,
    },
    SectionMisaligned {
        section_id: u32,
        offset: u64,
        alignment: u64,
    },
    MalformedSectionPayload {
        section_id: u32,
        reason: String,
    },
    FooterHashMismatch,
    FileSizeMismatch {
        expected: u64,
        got: u64,
    },
    FileTruncated,
    ManifestInvalid(String),
    MissingRequiredSection(&'static str),
    EmbeddingSizeMismatch {
        expected: usize,
        got: usize,
    },
    SectionCountMismatch {
        section_id: u32,
        expected: usize,
        got: usize,
    },
    UnsupportedDType(String),
    UnsupportedMetric(String),
    UnsupportedScoreType(String),
    UnsupportedNormalize(String),
    UnsupportedIndexType(String),
    UnsupportedRerankPolicy(String),
    InvalidModelHash(String),
    InvalidEmbeddingValue,
    Json(serde_json::Error),
    QueryValidation(String),
    DimensionMismatch {
        expected: usize,
        got: usize,
    },
    InvalidK(i32),
    EmptyQuery,
    ZeroNormQuery,
    FileNotFound(String),
    UnexpectedEof,
    UnsupportedFeature(String),
    InvalidInput(String),
}

impl std::fmt::Display for UrnaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(v0) => write!(f, "I/O error: {v0}"),
            Self::MagicMismatch { expected, got } => {
                write!(f, "magic mismatch: expected {expected:?}, got {got:?}")
            }
            Self::UnsupportedVersion(v0, v1) => write!(f, "unsupported version: {v0}.{v1}"),
            Self::UnsupportedFormatVersion(v0) => write!(f, "unsupported format_version: {v0}"),
            Self::UnsupportedSchemaVersion(v0) => write!(f, "unsupported schema_version: {v0}"),
            Self::InvalidHeaderChecksum => f.write_str("invalid header checksum"),
            Self::SectionOffsetOutOfBounds { section_id, offset } => {
                write!(f, "section {section_id} offset out of bounds: {offset}")
            }
            Self::SectionChecksumMismatch(v0) => write!(f, "section {v0} checksum mismatch"),
            Self::SectionNotFound(v0) => write!(f, "section {v0} not found"),
            Self::UnsupportedSectionVersion {
                section_id,
                version,
            } => write!(
                f,
                "unsupported section payload version: section={section_id} version={version}"
            ),
            Self::UnsupportedSectionEncoding {
                section_id,
                encoding,
            } => write!(
                f,
                "unsupported section encoding: section={section_id} encoding={encoding}"
            ),
            Self::SectionMisaligned {
                section_id,
                offset,
                alignment,
            } => write!(
                f,
                "section {section_id} offset not aligned: offset={offset} alignment={alignment}"
            ),
            Self::MalformedSectionPayload { section_id, reason } => write!(
                f,
                "malformed section payload: section={section_id} reason={reason}"
            ),
            Self::FooterHashMismatch => f.write_str("footer hash mismatch"),
            Self::FileSizeMismatch { expected, got } => {
                write!(f, "file size mismatch: expected {expected}, got {got}")
            }
            Self::FileTruncated => f.write_str("file truncated"),
            Self::ManifestInvalid(v0) => write!(f, "manifest invalid: {v0}"),
            Self::MissingRequiredSection(v0) => write!(f, "missing required section: {v0}"),
            Self::EmbeddingSizeMismatch { expected, got } => {
                write!(f, "embedding size mismatch: expected {expected}, got {got}")
            }
            Self::SectionCountMismatch {
                section_id,
                expected,
                got,
            } => write!(
                f,
                "section count mismatch: section={section_id} expected={expected} got={got}"
            ),
            Self::UnsupportedDType(v0) => write!(f, "unsupported dtype: {v0}"),
            Self::UnsupportedMetric(v0) => write!(f, "unsupported metric: {v0}"),
            Self::UnsupportedScoreType(v0) => write!(f, "unsupported score_type: {v0}"),
            Self::UnsupportedNormalize(v0) => write!(f, "unsupported normalize: {v0}"),
            Self::UnsupportedIndexType(v0) => write!(f, "unsupported index_type: {v0}"),
            Self::UnsupportedRerankPolicy(v0) => write!(f, "unsupported rerank_policy: {v0}"),
            Self::InvalidModelHash(v0) => write!(f, "invalid model_hash: {v0}"),
            Self::InvalidEmbeddingValue => f.write_str("NaN or Inf detected in embeddings"),
            Self::Json(v0) => write!(f, "JSON error: {v0}"),
            Self::QueryValidation(v0) => write!(f, "query validation failed: {v0}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::InvalidK(v0) => write!(f, "invalid k parameter: {v0}"),
            Self::EmptyQuery => f.write_str("empty query"),
            Self::ZeroNormQuery => f.write_str("zero-norm query"),
            Self::FileNotFound(v0) => write!(f, "file not found: {v0}"),
            Self::UnexpectedEof => f.write_str("unexpected end of file"),
            Self::UnsupportedFeature(v0) => write!(f, "unsupported feature: {v0}"),
            Self::InvalidInput(v0) => write!(f, "invalid input: {v0}"),
        }
    }
}

impl std::error::Error for UrnaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Json(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for UrnaError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for UrnaError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

pub type Result<T> = std::result::Result<T, UrnaError>;
