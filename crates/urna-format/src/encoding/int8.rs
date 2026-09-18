//! int8 quantized embeddings (`encoding=3`).
//!
//! On disk:
//! ```text
//!   u32 LE  payload_version = 1
//!   u32 LE  scale_kind      = 0  (per-vector f32)
//!   f32 LE * n              scales (one per vector)
//!   i8     * (n * dim)      quantized embeddings, row-major
//! ```
//!
//! The scale is the multiplier such that `f32_value ≈ i8_value * scale`.
//! We pick per-vector scales so a single outlier vector cannot crush
//! resolution for the whole corpus.

use crate::bytes::le_u32;
use crate::error::UrnaError;

pub const INT8_PAYLOAD_VERSION: u32 = 1;
pub const INT8_SCALE_KIND_PER_VECTOR: u32 = 0;
pub const INT8_PREFIX_SIZE: usize = 8;

/// Quantize an L2-normalized float32 embedding to int8 with a per-vector
/// scale. Returns `(scale, i8_bytes)` where `f32_value ≈ i8 * scale`.
///
/// L2-normalized vectors live in `[-1, 1]`; in practice the largest
/// component is well below 1 so we map `max(|v|)` to 127 to use the
/// full int8 range. Re-normalization at query time accumulates in f32.
pub fn quantize_f32_to_i8(values: &[f32]) -> (f32, Vec<i8>) {
    let max_abs = values.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    if max_abs == 0.0 {
        // Pathological zero vector — quantize to all zeros with scale 1.
        // The reader's zero-norm guard will reject queries against this.
        return (1.0, vec![0i8; values.len()]);
    }
    let scale = max_abs / 127.0;
    let inv_scale = 1.0 / scale;
    let q: Vec<i8> = values
        .iter()
        .map(|&v| {
            let scaled = (v * inv_scale).round();
            scaled.clamp(-127.0, 127.0) as i8
        })
        .collect();
    (scale, q)
}

/// Encode the int8 embeddings section payload. Layout matches
/// `INT8_PAYLOAD_VERSION` / `INT8_SCALE_KIND_PER_VECTOR`.
///
/// `embeddings` is `n * dim` row-major f32 values. Returns a buffer
/// ready to write as the embeddings section (encoding = INT8).
pub fn encode_int8_embeddings(embeddings: &[f32], n: usize, dim: usize) -> crate::Result<Vec<u8>> {
    if embeddings.len() != n * dim {
        return Err(UrnaError::InvalidInput(format!(
            "encode_int8_embeddings: got {} f32 values for n={} dim={}",
            embeddings.len(),
            n,
            dim
        )));
    }
    let mut out = Vec::with_capacity(INT8_PREFIX_SIZE + n * 4 + n * dim);
    out.extend_from_slice(&INT8_PAYLOAD_VERSION.to_le_bytes());
    out.extend_from_slice(&INT8_SCALE_KIND_PER_VECTOR.to_le_bytes());
    let mut scales: Vec<u8> = Vec::with_capacity(n * 4);
    let mut bodies: Vec<u8> = Vec::with_capacity(n * dim);
    for i in 0..n {
        let row = &embeddings[i * dim..(i + 1) * dim];
        let (scale, q) = quantize_f32_to_i8(row);
        scales.extend_from_slice(&scale.to_le_bytes());
        // i8 -> u8 is a bitcast (two's complement preserved).
        bodies.extend(q.iter().map(|&v| v as u8));
    }
    out.extend_from_slice(&scales);
    out.extend_from_slice(&bodies);
    Ok(out)
}

/// Decoded view over an int8 embeddings payload. The slices borrow
/// from the input bytes (no copy).
pub struct Int8EmbeddingsView<'a> {
    pub scales: &'a [u8], // n * 4 bytes (f32 LE)
    pub bodies: &'a [u8], // n * dim bytes (i8, bitcast to u8)
    pub n: usize,
    pub dim: usize,
}

impl<'a> Int8EmbeddingsView<'a> {
    pub fn parse(bytes: &'a [u8], n: usize, dim: usize) -> crate::Result<Self> {
        // checked: `n` / `dim` are header-controlled; an overflowed product
        // must be a typed mismatch, never a wrapped "match".
        let want = super::expected_embeddings_size("int8", n, dim).unwrap_or(usize::MAX);
        if bytes.len() != want {
            return Err(UrnaError::EmbeddingSizeMismatch {
                expected: want,
                got: bytes.len(),
            });
        }
        let version = le_u32(&bytes[0..4])?;
        if version != INT8_PAYLOAD_VERSION {
            return Err(UrnaError::UnsupportedSectionVersion {
                section_id: crate::layout::SECTION_EMBEDDINGS,
                version,
            });
        }
        let kind = le_u32(&bytes[4..8])?;
        if kind != INT8_SCALE_KIND_PER_VECTOR {
            return Err(UrnaError::MalformedSectionPayload {
                section_id: crate::layout::SECTION_EMBEDDINGS,
                reason: format!("int8 scale_kind {} not supported", kind),
            });
        }
        let scales_end = INT8_PREFIX_SIZE + n * 4;
        Ok(Self {
            scales: &bytes[INT8_PREFIX_SIZE..scales_end],
            bodies: &bytes[scales_end..],
            n,
            dim,
        })
    }

    /// Read scale[i] as f32.
    #[inline]
    pub fn scale(&self, i: usize) -> f32 {
        let off = i * 4;
        f32::from_le_bytes([
            self.scales[off],
            self.scales[off + 1],
            self.scales[off + 2],
            self.scales[off + 3],
        ])
    }

    /// Borrow row[i] as `&[i8]`.
    #[inline]
    pub fn row(&self, i: usize) -> &'a [i8] {
        let start = i * self.dim;
        let end = start + self.dim;
        // i8 has the same size and alignment as u8; bytemuck checks both at
        // compile time, so this reinterpretation carries no `unsafe`.
        bytemuck::cast_slice(&self.bodies[start..end])
    }
}
