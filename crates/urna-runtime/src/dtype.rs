//! Runtime view of the embeddings section dtype, carved out of
//! `mmap_file.rs` so that file stays under the 300-line crate guard.

use urna_format::UrnaError;

use crate::error::RuntimeError;

/// Runtime view of the embeddings section dtype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    Float32,
    Float16,
    Int8,
    Int4,
}

impl DType {
    pub(crate) fn from_str(s: &str) -> Result<Self, RuntimeError> {
        match s {
            "float32" => Ok(Self::Float32),
            "float16" => Ok(Self::Float16),
            "int8" => Ok(Self::Int8),
            "int4" => Ok(Self::Int4),
            other => Err(RuntimeError::Format(UrnaError::UnsupportedDType(
                other.into(),
            ))),
        }
    }
    /// Nominal on-disk bytes per stored embedding value. int4 packs two
    /// codes per byte (rounds to 0 here); the exact section size, with the
    /// f16 group scales, is `expected_embeddings_size`.
    pub fn bytes_per_value(self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Float16 => 2,
            Self::Int8 => 1,
            Self::Int4 => 0,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Float16 => "float16",
            Self::Int8 => "int8",
            Self::Int4 => "int4",
        }
    }
}
