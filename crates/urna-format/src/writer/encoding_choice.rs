//! Wire-encoding and dtype choices exposed by the builder. The two
//! enums are 1:1 mapped to `SECTION_ENCODING_*` values; this layer
//! keeps the public API enum-friendly and the layout module pure.

use crate::layout::{
    SECTION_ENCODING_FLOAT16, SECTION_ENCODING_INT4, SECTION_ENCODING_INT8, SECTION_ENCODING_RAW,
    SECTION_ENCODING_ZSTD,
};

/// Timestamp written to the manifest when the builder is in reproducible
/// mode. Chosen so that two builds with identical inputs produce
/// identical bytes regardless of when they ran.
pub const REPRODUCIBLE_CREATED: &str = "1970-01-01T00:00:00Z";

/// Wire encoding choice for a non-embedding section. Embedding encoding
/// is controlled by `EmbeddingDType` since dtype and encoding are
/// intertwined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SectionEncoding {
    Raw,
    Zstd,
}

impl SectionEncoding {
    pub fn id(self) -> u32 {
        match self {
            Self::Raw => SECTION_ENCODING_RAW,
            Self::Zstd => SECTION_ENCODING_ZSTD,
        }
    }
}

/// Embedding dtype + on-disk encoding. The two are 1:1 in v1 — float32
/// implies raw f32 LE, float16 implies raw f16 LE, int8 implies the
/// quantized prefix layout (see `encoding::encode_int8_embeddings`), int4
/// implies the block-64 per-group layout (see `encode_int4_embeddings`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddingDType {
    Float32,
    Float16,
    Int8,
    Int4,
}

impl EmbeddingDType {
    pub fn manifest_str(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Float16 => "float16",
            Self::Int8 => "int8",
            Self::Int4 => "int4",
        }
    }
    pub fn encoding(self) -> u32 {
        match self {
            Self::Float32 => SECTION_ENCODING_RAW,
            Self::Float16 => SECTION_ENCODING_FLOAT16,
            Self::Int8 => SECTION_ENCODING_INT8,
            Self::Int4 => SECTION_ENCODING_INT4,
        }
    }
}
