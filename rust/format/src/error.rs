use thiserror::Error;

#[derive(Debug, Error)]
pub enum UrnaError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("magic mismatch: expected {expected:?}, got {got:?}")]
    MagicMismatch { expected: [u8; 4], got: [u8; 4] },
    #[error("unsupported version: {0}.{1}")]
    UnsupportedVersion(u16, u16),
    #[error("unsupported format_version: {0}")]
    UnsupportedFormatVersion(u32),
    #[error("unsupported schema_version: {0}")]
    UnsupportedSchemaVersion(u32),
    #[error("invalid header checksum")]
    InvalidHeaderChecksum,
    #[error("section {section_id} offset out of bounds: {offset}")]
    SectionOffsetOutOfBounds { section_id: u32, offset: u64 },
    #[error("section {0} checksum mismatch")]
    SectionChecksumMismatch(u32),
    #[error("section {0} not found")]
    SectionNotFound(u32),
    #[error("unsupported section payload version: section={section_id} version={version}")]
    UnsupportedSectionVersion { section_id: u32, version: u32 },
    #[error("unsupported section encoding: section={section_id} encoding={encoding}")]
    UnsupportedSectionEncoding { section_id: u32, encoding: u32 },
    #[error("section {section_id} offset not aligned: offset={offset} alignment={alignment}")]
    SectionMisaligned {
        section_id: u32,
        offset: u64,
        alignment: u64,
    },
    #[error("malformed section payload: section={section_id} reason={reason}")]
    MalformedSectionPayload { section_id: u32, reason: String },
    #[error("footer hash mismatch")]
    FooterHashMismatch,
    #[error("file size mismatch: expected {expected}, got {got}")]
    FileSizeMismatch { expected: u64, got: u64 },
    #[error("file truncated")]
    FileTruncated,
    #[error("manifest invalid: {0}")]
    ManifestInvalid(String),
    #[error("missing required section: {0}")]
    MissingRequiredSection(&'static str),
    #[error("embedding size mismatch: expected {expected}, got {got}")]
    EmbeddingSizeMismatch { expected: usize, got: usize },
    #[error("section count mismatch: section={section_id} expected={expected} got={got}")]
    SectionCountMismatch {
        section_id: u32,
        expected: usize,
        got: usize,
    },
    #[error("unsupported dtype: {0}")]
    UnsupportedDType(String),
    #[error("unsupported metric: {0}")]
    UnsupportedMetric(String),
    #[error("unsupported score_type: {0}")]
    UnsupportedScoreType(String),
    #[error("unsupported normalize: {0}")]
    UnsupportedNormalize(String),
    #[error("unsupported index_type: {0}")]
    UnsupportedIndexType(String),
    #[error("unsupported rerank_policy: {0}")]
    UnsupportedRerankPolicy(String),
    #[error("invalid model_hash: {0}")]
    InvalidModelHash(String),
    #[error("NaN or Inf detected in embeddings")]
    InvalidEmbeddingValue,
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("query validation failed: {0}")]
    QueryValidation(String),
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("invalid k parameter: {0}")]
    InvalidK(i32),
    #[error("empty query")]
    EmptyQuery,
    #[error("zero-norm query")]
    ZeroNormQuery,
    #[error("file not found: {0}")]
    FileNotFound(String),
    #[error("unexpected end of file")]
    UnexpectedEof,
    #[error("unsupported feature: {0}")]
    UnsupportedFeature(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, UrnaError>;
