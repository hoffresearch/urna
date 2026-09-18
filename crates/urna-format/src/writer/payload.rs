//! Section payload preparation: encode embeddings per dtype, optionally
//! zstd-wrap a text section. Pure functions; no state from the builder.

use super::{EmbeddingDType, SectionEncoding};
use crate::chunk::ChunkInput;
use crate::encoding::{
    encode_int4_embeddings, encode_int8_embeddings, f32_to_f16_bytes, zstd_encode,
};
use crate::layout::{SECTION_ENCODING_RAW, SECTION_ENCODING_ZSTD};

/// Encode the embeddings section payload for the given dtype.
/// `chunks` must already have been validated against `embedding_dim`.
pub(super) fn encode_embeddings_payload(
    dtype: EmbeddingDType,
    chunks: &[ChunkInput],
    embedding_dim: usize,
) -> crate::Result<Vec<u8>> {
    let n = chunks.len();
    Ok(match dtype {
        EmbeddingDType::Float32 => {
            let mut buf: Vec<u8> = Vec::with_capacity(n * embedding_dim * 4);
            for c in chunks {
                for v in &c.embedding {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
            buf
        }
        EmbeddingDType::Float16 => {
            let mut flat: Vec<f32> = Vec::with_capacity(n * embedding_dim);
            for c in chunks {
                flat.extend_from_slice(&c.embedding);
            }
            f32_to_f16_bytes(&flat)
        }
        EmbeddingDType::Int8 => {
            let mut flat: Vec<f32> = Vec::with_capacity(n * embedding_dim);
            for c in chunks {
                flat.extend_from_slice(&c.embedding);
            }
            encode_int8_embeddings(&flat, n, embedding_dim)?
        }
        EmbeddingDType::Int4 => {
            let mut flat: Vec<f32> = Vec::with_capacity(n * embedding_dim);
            for c in chunks {
                flat.extend_from_slice(&c.embedding);
            }
            encode_int4_embeddings(&flat, n, embedding_dim)?
        }
    })
}

/// Wrap `payload` according to `enc`. Used for text-heavy sections that
/// can be either raw or zstd; embedding section bypasses this and goes
/// straight through `encode_embeddings_payload`.
pub(super) fn maybe_zstd(
    section_id: u32,
    enc: SectionEncoding,
    payload: Vec<u8>,
) -> crate::Result<(u32, u32, Vec<u8>)> {
    Ok(match enc {
        SectionEncoding::Raw => (section_id, SECTION_ENCODING_RAW, payload),
        SectionEncoding::Zstd => (section_id, SECTION_ENCODING_ZSTD, zstd_encode(&payload)?),
    })
}
