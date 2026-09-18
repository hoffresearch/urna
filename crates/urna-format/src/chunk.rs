//! Canonical chunk types and chunk_id derivation.
//!
//! `chunk_id` is a deterministic SHA-256 over the canonical text plus the
//! original location and the chunker version. The exact preimage is fixed
//! by spec — anything else would let two runs of the same chunker produce
//! different IDs for the same content, which breaks reproducibility.

use crate::error::UrnaError;
use sha2::{Digest, Sha256};

/// One chunk's worth of input that the writer needs to produce a v1 file.
///
/// `embedding` must be a normalized f32 vector of the embedding dimension
/// declared in the manifest. The writer validates dimensions and rejects
/// NaN/Inf — callers do not get to pass garbage in.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkInput {
    pub canonical_text: String,
    pub source_uri: String,
    pub byte_start: u64,
    pub byte_end: u64,
    pub embedding: Vec<f32>,
}

/// Compute the canonical chunk_id for the given inputs.
///
/// Preimage layout (UTF-8, no separators ambiguity — all length-prefixed):
///
/// ```text
///   "urna:chunk_id:v1\n"
///   u32 LE   len(canonical_text)
///   bytes    canonical_text
///   u32 LE   len(source_uri)
///   bytes    source_uri
///   u64 LE   byte_start
///   u64 LE   byte_end
///   u32 LE   len(chunker_version)
///   bytes    chunker_version
/// ```
///
/// Output: `sha256:<64 hex chars>`.
pub fn chunk_id(
    canonical_text: &str,
    source_uri: &str,
    byte_start: u64,
    byte_end: u64,
    chunker_version: &str,
) -> String {
    const DOMAIN: &[u8] = b"urna:chunk_id:v1\n";
    let mut h = Sha256::new();
    h.update(DOMAIN);
    write_lp(&mut h, canonical_text.as_bytes());
    write_lp(&mut h, source_uri.as_bytes());
    h.update(byte_start.to_le_bytes());
    h.update(byte_end.to_le_bytes());
    write_lp(&mut h, chunker_version.as_bytes());
    let digest = h.finalize();
    format!("sha256:{}", hex::encode(digest))
}

fn write_lp(h: &mut Sha256, bytes: &[u8]) {
    let len = bytes.len() as u32;
    h.update(len.to_le_bytes());
    h.update(bytes);
}

/// Validate a `ChunkInput` against the embedding dimension declared in the
/// manifest. Returns the validated reference for ergonomics.
pub fn validate_chunk(c: &ChunkInput, embedding_dim: usize) -> crate::Result<()> {
    if c.embedding.len() != embedding_dim {
        return Err(UrnaError::DimensionMismatch {
            expected: embedding_dim,
            got: c.embedding.len(),
        });
    }
    for v in &c.embedding {
        if v.is_nan() || v.is_infinite() {
            return Err(UrnaError::InvalidEmbeddingValue);
        }
    }
    if c.byte_end < c.byte_start {
        return Err(UrnaError::InvalidInput(format!(
            "byte_end ({}) < byte_start ({})",
            c.byte_end, c.byte_start
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_id_is_deterministic() {
        let a = chunk_id("hello", "doc.txt", 0, 5, "v1");
        let b = chunk_id("hello", "doc.txt", 0, 5, "v1");
        assert_eq!(a, b);
        assert!(a.starts_with("sha256:"));
        assert_eq!(a.len(), "sha256:".len() + 64);
    }

    #[test]
    fn chunk_id_changes_with_inputs() {
        let base = chunk_id("hello", "doc.txt", 0, 5, "v1");
        assert_ne!(base, chunk_id("HELLO", "doc.txt", 0, 5, "v1"));
        assert_ne!(base, chunk_id("hello", "other.txt", 0, 5, "v1"));
        assert_ne!(base, chunk_id("hello", "doc.txt", 1, 5, "v1"));
        assert_ne!(base, chunk_id("hello", "doc.txt", 0, 6, "v1"));
        assert_ne!(base, chunk_id("hello", "doc.txt", 0, 5, "v2"));
    }

    #[test]
    fn validate_chunk_dim_mismatch() {
        let c = ChunkInput {
            canonical_text: "x".into(),
            source_uri: "y".into(),
            byte_start: 0,
            byte_end: 1,
            embedding: vec![1.0, 0.0],
        };
        assert!(validate_chunk(&c, 4).is_err());
    }

    #[test]
    fn validate_chunk_nan() {
        let c = ChunkInput {
            canonical_text: "x".into(),
            source_uri: "y".into(),
            byte_start: 0,
            byte_end: 1,
            embedding: vec![f32::NAN, 0.0, 0.0, 0.0],
        };
        assert!(matches!(
            validate_chunk(&c, 4),
            Err(UrnaError::InvalidEmbeddingValue)
        ));
    }

    #[test]
    fn validate_chunk_bad_span() {
        let c = ChunkInput {
            canonical_text: "x".into(),
            source_uri: "y".into(),
            byte_start: 10,
            byte_end: 5,
            embedding: vec![1.0, 0.0, 0.0, 0.0],
        };
        assert!(matches!(
            validate_chunk(&c, 4),
            Err(UrnaError::InvalidInput(_))
        ));
    }
}
