//! int4 block-64 quantized embeddings (`encoding=7`).
//!
//! On disk:
//! ```text
//!   u32 LE  payload_version = 1
//!   u32 LE  scale_kind      = 1  (per-group, block-64)
//!   f16 LE * (n * dim/64)   per-group block absmax scales, row-major
//!   u8     * (n * dim/2)    packed 4-bit signed codes, two nibbles/byte
//! ```
//!
//! Each row is split into `dim/64` contiguous 64-dim blocks; `dim` must be
//! divisible by 64. Every block carries one f16 absmax scale, so an outlier
//! in one block cannot crush another (per-group idea, the int8 per-vector
//! step taken finer). A code `c` in `[-7, 7]` reconstructs as
//! `f32_value ~= c * scale`; the range is symmetric and reserves `-8`.
//!
//! STORED-PRECISION codec: like int8, the returned cosine is real AT THIS
//! PRECISION (disclosed via the dtype). It never zstds/shuffles, so the
//! runtime scores it off mmap with the fused dequant+dot kernel.

use crate::bytes::le_u32;
use crate::error::UrnaError;

pub const INT4_PAYLOAD_VERSION: u32 = 1;
pub const INT4_SCALE_KIND_PER_GROUP: u32 = 1;
pub const INT4_PREFIX_SIZE: usize = 8;
/// Block size for the per-group absmax scale. `dim` must be a multiple.
pub const INT4_BLOCK: usize = 64;

/// Number of 64-dim blocks per row. `dim` must be divisible by `INT4_BLOCK`.
#[inline]
pub fn int4_blocks_per_row(dim: usize) -> usize {
    dim / INT4_BLOCK
}

/// Quantize one L2-normalized f32 row to int4 with per-64-block absmax
/// scales. Returns `(scales, codes)` where `scales[g]` is the f16 absmax
/// scale of block `g` and `codes[j]` in `[-7, 7]` reconstructs as
/// `codes[j] as f32 * scales[j / 64]`. Mirrors `quantize_f32_to_i8` but per
/// 64-dim group; a zero block maps to all-zero codes with scale 1.
pub fn quantize_f32_to_i4(values: &[f32], dim: usize) -> (Vec<half::f16>, Vec<i8>) {
    let blocks = int4_blocks_per_row(dim);
    let mut scales: Vec<half::f16> = Vec::with_capacity(blocks);
    let mut codes: Vec<i8> = Vec::with_capacity(dim);
    for g in 0..blocks {
        let blk = &values[g * INT4_BLOCK..(g + 1) * INT4_BLOCK];
        let max_abs = blk.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        // Quantize against the f16-rounded scale so the codes match the
        // stored (f16) scale the reader sees, keeping round-trip tight.
        let scale_f16 = half::f16::from_f32(if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 });
        let scale = scale_f16.to_f32();
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for &v in blk {
            let q = (v * inv).round().clamp(-7.0, 7.0);
            codes.push(q as i8);
        }
        scales.push(scale_f16);
    }
    (scales, codes)
}

/// Pack signed 4-bit codes (`[-7, 7]`) into bytes, two nibbles per byte,
/// low nibble first. The nibble is the two's-complement low 4 bits, so
/// `-7..=7` maps to `0x9..=0x7` and unpacks back exactly via sign extension.
#[inline]
pub fn pack_nibbles(codes: &[i8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(codes.len().div_ceil(2));
    for pair in codes.chunks(2) {
        let lo = (pair[0] as u8) & 0x0F;
        let hi = pair.get(1).map(|&c| (c as u8) & 0x0F).unwrap_or(0);
        out.push(lo | (hi << 4));
    }
    out
}

/// Sign-extend a 4-bit nibble (low 4 bits of `b`) to an `i8` in `[-8, 7]`.
#[inline]
pub fn nibble_to_i4(b: u8) -> i8 {
    let n = b & 0x0F;
    if n & 0x08 != 0 {
        (n | 0xF0) as i8
    } else {
        n as i8
    }
}

/// Encode the int4 embeddings section payload. `embeddings` is `n * dim`
/// row-major f32; `dim` must be divisible by `INT4_BLOCK`.
pub fn encode_int4_embeddings(embeddings: &[f32], n: usize, dim: usize) -> crate::Result<Vec<u8>> {
    if dim == 0 || dim % INT4_BLOCK != 0 {
        return Err(UrnaError::InvalidInput(format!(
            "encode_int4_embeddings: dim={dim} must be a nonzero multiple of {INT4_BLOCK}"
        )));
    }
    if embeddings.len() != n * dim {
        return Err(UrnaError::InvalidInput(format!(
            "encode_int4_embeddings: got {} f32 values for n={n} dim={dim}",
            embeddings.len()
        )));
    }
    let blocks = int4_blocks_per_row(dim);
    let mut out = Vec::with_capacity(INT4_PREFIX_SIZE + n * blocks * 2 + n * dim / 2);
    out.extend_from_slice(&INT4_PAYLOAD_VERSION.to_le_bytes());
    out.extend_from_slice(&INT4_SCALE_KIND_PER_GROUP.to_le_bytes());
    let mut scale_bytes: Vec<u8> = Vec::with_capacity(n * blocks * 2);
    let mut body: Vec<u8> = Vec::with_capacity(n * dim / 2);
    for i in 0..n {
        let row = &embeddings[i * dim..(i + 1) * dim];
        let (scales, codes) = quantize_f32_to_i4(row, dim);
        for s in &scales {
            scale_bytes.extend_from_slice(&s.to_le_bytes());
        }
        body.extend_from_slice(&pack_nibbles(&codes));
    }
    out.extend_from_slice(&scale_bytes);
    out.extend_from_slice(&body);
    Ok(out)
}

/// Decoded view over an int4 embeddings payload. Slices borrow the input
/// bytes (no copy); accessors decode scales/codes on demand.
pub struct Int4EmbeddingsView<'a> {
    /// f16 LE group scales, `n * blocks` of them, row-major.
    pub scales: &'a [u8],
    /// packed nibble codes, `n * dim/2` bytes.
    pub codes: &'a [u8],
    pub n: usize,
    pub dim: usize,
    pub blocks: usize,
}

impl<'a> Int4EmbeddingsView<'a> {
    pub fn parse(bytes: &'a [u8], n: usize, dim: usize) -> crate::Result<Self> {
        if dim == 0 || dim % INT4_BLOCK != 0 {
            return Err(UrnaError::MalformedSectionPayload {
                section_id: crate::layout::SECTION_EMBEDDINGS,
                reason: format!("int4 dim={dim} must be a nonzero multiple of {INT4_BLOCK}"),
            });
        }
        let blocks = int4_blocks_per_row(dim);
        // checked: `n` / `dim` are header-controlled; an overflowed product
        // must be a typed mismatch, never a wrapped "match".
        let want = super::expected_embeddings_size("int4", n, dim).unwrap_or(usize::MAX);
        if bytes.len() != want {
            return Err(UrnaError::EmbeddingSizeMismatch {
                expected: want,
                got: bytes.len(),
            });
        }
        let version = le_u32(&bytes[0..4])?;
        if version != INT4_PAYLOAD_VERSION {
            return Err(UrnaError::UnsupportedSectionVersion {
                section_id: crate::layout::SECTION_EMBEDDINGS,
                version,
            });
        }
        let kind = le_u32(&bytes[4..8])?;
        if kind != INT4_SCALE_KIND_PER_GROUP {
            return Err(UrnaError::MalformedSectionPayload {
                section_id: crate::layout::SECTION_EMBEDDINGS,
                reason: format!("int4 scale_kind {kind} not supported"),
            });
        }
        let scales_end = INT4_PREFIX_SIZE + n * blocks * 2;
        Ok(Self {
            scales: &bytes[INT4_PREFIX_SIZE..scales_end],
            codes: &bytes[scales_end..],
            n,
            dim,
            blocks,
        })
    }

    /// Read the f16 group scale `g` of row `i` as f32.
    #[inline]
    pub fn group_scale(&self, i: usize, g: usize) -> f32 {
        let off = (i * self.blocks + g) * 2;
        half::f16::from_le_bytes([self.scales[off], self.scales[off + 1]]).to_f32()
    }

    /// Borrow row `i`'s packed nibble bytes (`dim/2` of them).
    #[inline]
    pub fn row_codes(&self, i: usize) -> &'a [u8] {
        let rs = self.dim / 2;
        let start = i * rs;
        &self.codes[start..start + rs]
    }

    /// Decode row `i`'s f16 group scales into a fresh `Vec<f32>` (one per
    /// 64-dim block). One-shot use only (the packed ANN store materializes
    /// every row once); the per-candidate rerank uses `row_scales_into`.
    #[inline]
    pub fn row_scales_f32(&self, i: usize) -> Vec<f32> {
        (0..self.blocks).map(|g| self.group_scale(i, g)).collect()
    }

    /// Decode row `i`'s f16 group scales into a caller-owned buffer of
    /// exactly `blocks` f32s, so a rerank over thousands of candidates
    /// reuses one allocation instead of one `Vec` per row.
    ///
    /// # Panics
    ///
    /// When `out.len() != self.blocks`.
    #[inline]
    pub fn row_scales_into(&self, i: usize, out: &mut [f32]) {
        assert_eq!(out.len(), self.blocks, "int4: one scale slot per block");
        for (g, slot) in out.iter_mut().enumerate() {
            *slot = self.group_scale(i, g);
        }
    }
}

// Unit + round-trip coverage lives in `tests/int4_roundtrip.rs` (pack /
// unpack, quantize clamping, the section view, and the typed malformed-
// payload rejections) so this codec source stays under the 300-line rust
// src guard. Negative file-level paths live in `tests/negative_int4.rs`.
