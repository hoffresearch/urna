//! Rerank-source handle: the single place the mandatory exact-cosine
//! rerank reads its vectors from.
//!
//! The honesty contract is that every non-exact search path (ann, hybrid,
//! and the future graph/space/cross paths) only *generates candidates*;
//! the returned `score` is always a real cosine recomputed here against a
//! fixed-stride embeddings slab. Routing that recompute through one
//! explicit handle (instead of hard-dispatching on a single `self.dtype`
//! and the single `embeddings` section) is what lets per-space and
//! sub-int8 paths plug into the SAME gate later without weakening it.
//!
//! Two precisions:
//!
//! - no fp source: the rerank reads the STORED dtype slab, so the score
//!   is "real cosine at stored precision" (full precision for float32,
//!   slightly lossy for float16/int8). this is today's behavior.
//! - with an `embeddings_fp` (0x09) source present: the rerank reads the
//!   full-precision fp slab instead, so a sub-int8 candidate slab still
//!   yields a real cosine. the fp source is mandatory for any sub-int8
//!   section (enforced where that section is written, phase 3); here we
//!   only consume it.
//!
//! The fp slab is a fixed-stride 64-byte-aligned raw slab (float32 or
//! float16), NEVER zstd, so the existing simd kernels score it byte-for-
//! byte unchanged; its dtype is read back from the section stride.

use urna_format::layout::{SECTION_EMBEDDINGS_FP, SectionEntry};
use urna_format::{INT4_BLOCK, Int4EmbeddingsView, Int8EmbeddingsView, UrnaError};

use crate::dtype::DType;
use crate::error::RuntimeError;
use crate::simd;

/// The optional full-precision rerank slab (`embeddings_fp`, section
/// `0x09`): a fixed-stride 64-byte-aligned raw slab, NEVER zstd, one row
/// per embedding. Present only when a sub-int8 candidate slab needs an
/// honest exact rerank. The dtype (float32 or float16 only) is read back
/// from the section stride. The writer side lands with the sub-int8
/// codecs (phase 3); the runtime read path is here so the rerank handle
/// can route to it byte-for-byte unchanged when it appears.
#[derive(Clone, Copy)]
pub(crate) struct FpSlab {
    pub(crate) offset: usize,
    pub(crate) size: usize,
    pub(crate) dtype: DType,
}

impl FpSlab {
    /// Detect an `embeddings_fp` (0x09) section in the table and infer its
    /// dtype from stride: 4 bytes/value -> float32, 2 -> float16 (an fp
    /// source is never int8). A bad stride is a malformed file, not a
    /// silent skip. Returns `None` when the section is absent.
    pub(crate) fn detect(
        entries: &[SectionEntry],
        n: usize,
        dim: usize,
    ) -> Result<Option<Self>, RuntimeError> {
        let Some(e) = entries
            .iter()
            .find(|e| e.section_id == SECTION_EMBEDDINGS_FP)
        else {
            return Ok(None);
        };
        let size = e.size as usize;
        let stride = dim.checked_mul(n).filter(|d| *d > 0).map(|d| size / d);
        let dtype = match stride {
            Some(4) => DType::Float32,
            Some(2) => DType::Float16,
            _ => {
                return Err(RuntimeError::Format(UrnaError::MalformedSectionPayload {
                    section_id: SECTION_EMBEDDINGS_FP,
                    reason: format!("fp slab {size} bytes is not f32/f16 for n={n} dim={dim}"),
                }));
            }
        };
        Ok(Some(Self {
            offset: e.offset as usize,
            size,
            dtype,
        }))
    }
}

/// Where one rerank reads its vectors from. Built once per search call,
/// then `score`d per candidate. Borrows the mmap slab; no copy.
///
/// The int4 path needs two small f32 rows per candidate (the unpacked
/// nibbles and the per-block scales). They live HERE, allocated once in
/// `new`, so `score` is allocation-free: a rerank over thousands of
/// candidates used to pay a `malloc`/`free` pair per row for each. That
/// is why `score` takes `&mut self`.
pub(crate) struct RerankSource<'a> {
    rows: RerankRows<'a>,
    dim: usize,
    /// `dim` f32s, int4 only (empty otherwise).
    scratch: Vec<f32>,
    /// `dim / INT4_BLOCK` f32s, int4 only (empty otherwise).
    scales: Vec<f32>,
}

enum RerankRows<'a> {
    F32(&'a [u8]),
    F16(&'a [u8]),
    Int8(Int8EmbeddingsView<'a>),
    Int4(Int4EmbeddingsView<'a>),
}

impl<'a> RerankSource<'a> {
    /// Build a rerank source over a fixed-stride embeddings slab. `bytes`
    /// is the raw section payload for `dtype` (for int8 that includes the
    /// per-vector scale prefix, parsed here once).
    pub(crate) fn new(
        dtype: DType,
        bytes: &'a [u8],
        n: usize,
        dim: usize,
    ) -> Result<Self, RuntimeError> {
        let (rows, scratch, scales) = match dtype {
            DType::Float32 => (RerankRows::F32(bytes), Vec::new(), Vec::new()),
            DType::Float16 => (RerankRows::F16(bytes), Vec::new(), Vec::new()),
            DType::Int8 => (
                RerankRows::Int8(
                    Int8EmbeddingsView::parse(bytes, n, dim).map_err(RuntimeError::Format)?,
                ),
                Vec::new(),
                Vec::new(),
            ),
            DType::Int4 => {
                let view =
                    Int4EmbeddingsView::parse(bytes, n, dim).map_err(RuntimeError::Format)?;
                let blocks = view.blocks;
                (RerankRows::Int4(view), vec![0.0; dim], vec![0.0; blocks])
            }
        };
        Ok(Self {
            rows,
            dim,
            scratch,
            scales,
        })
    }

    /// Real cosine of `qnorm` against row `i` at this source's precision.
    /// `qnorm` is L2-normalized and rows are L2-normalized at write time,
    /// so the dot product IS the cosine. Allocation-free on every dtype.
    #[inline]
    pub(crate) fn score(&mut self, qnorm: &[f32], i: usize) -> f32 {
        match &self.rows {
            RerankRows::F32(b) => {
                let rs = self.dim * 4;
                let off = i * rs;
                simd::dot_f32_bytes(qnorm, &b[off..off + rs])
            }
            RerankRows::F16(b) => {
                let rs = self.dim * 2;
                let off = i * rs;
                simd::dot_f32_f16_bytes(qnorm, &b[off..off + rs])
            }
            RerankRows::Int8(view) => simd::dot_f32_i8(qnorm, view.row(i), view.scale(i)),
            RerankRows::Int4(view) => {
                view.row_scales_into(i, &mut self.scales);
                simd::dot_f32_i4_blocked(
                    qnorm,
                    view.row_codes(i),
                    &self.scales,
                    self.dim,
                    INT4_BLOCK,
                    &mut self.scratch,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use urna_format::{encode_int4_embeddings, encode_int8_embeddings, f32_to_f16_bytes};

    fn f32_slab(rows: &[[f32; 4]]) -> Vec<u8> {
        let mut b = Vec::new();
        for row in rows {
            for v in row {
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
        b
    }

    #[test]
    fn f32_source_dot_is_cosine() {
        let rows = [[1.0f32, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
        let bytes = f32_slab(&rows);
        let mut src = RerankSource::new(DType::Float32, &bytes, 2, 4).unwrap();
        let q = [1.0f32, 0.0, 0.0, 0.0];
        assert!((src.score(&q, 0) - 1.0).abs() < 1e-6);
        assert!(src.score(&q, 1).abs() < 1e-6);
    }

    /// The handle scores whatever slab it is handed. A full-precision fp
    /// slab returns the exact dot; the int8 slab over the SAME logical
    /// vector returns a quantized approximation. They differ, which is
    /// exactly why an fp rerank source makes a sub-int8 score honest.
    #[test]
    fn fp_source_is_more_precise_than_int8_for_same_vectors() {
        // One L2-normalized row with values int8 cannot represent exactly.
        let raw = [0.0312f32, 0.5007, -0.2013, 0.8401];
        let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        let row: [f32; 4] = std::array::from_fn(|j| raw[j] / norm);
        let q: [f32; 4] = row; // query == the row, exact cosine is 1.0

        let f32_bytes = f32_slab(&[row]);
        let mut fp = RerankSource::new(DType::Float32, &f32_bytes, 1, 4).unwrap();
        let int8_bytes = encode_int8_embeddings(&row, 1, 4).unwrap();
        let mut q8 = RerankSource::new(DType::Int8, &int8_bytes, 1, 4).unwrap();

        let s_fp = fp.score(&q, 0);
        let s_i8 = q8.score(&q, 0);
        assert!((s_fp - 1.0).abs() < 1e-6, "fp source must be exact: {s_fp}");
        assert!(
            (s_i8 - 1.0).abs() > 1e-4,
            "int8 must lose precision so the fp source is the honest one: {s_i8}"
        );

        // f16 is between the two: lossy but far closer than int8.
        let f16_bytes = f32_to_f16_bytes(&row);
        let mut f16 = RerankSource::new(DType::Float16, &f16_bytes, 1, 4).unwrap();
        assert!((f16.score(&q, 0) - 1.0).abs() < 1e-2);
    }

    /// The int4 rerank source scores a known row, and an fp source over the
    /// same logical vector is strictly more precise. int4 is the coarser
    /// stored precision, so its self-similarity drifts further from 1.0 than
    /// the exact fp source does, which is exactly why int4 must disclose
    /// "real cosine at stored precision".
    #[test]
    fn int4_source_scores_row_and_is_less_precise_than_fp() {
        // dim = 64 (one block); a row int4 cannot represent exactly.
        let dim = 64;
        let raw: Vec<f32> = (0..dim)
            .map(|j| ((j as f32 * 0.021).cos()) * 0.3 + 0.01)
            .collect();
        let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        let row: Vec<f32> = raw.iter().map(|x| x / norm).collect();
        let q = row.clone(); // query == row; exact cosine is 1.0

        let f32_bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut fp = RerankSource::new(DType::Float32, &f32_bytes, 1, dim).unwrap();
        let int4_bytes = encode_int4_embeddings(&row, 1, dim).unwrap();
        let mut q4 = RerankSource::new(DType::Int4, &int4_bytes, 1, dim).unwrap();

        let s_fp = fp.score(&q, 0);
        let s_i4 = q4.score(&q, 0);
        assert!((s_fp - 1.0).abs() < 1e-6, "fp source must be exact: {s_fp}");
        assert!(
            (s_i4 - 1.0).abs() > 1e-4,
            "int4 must lose precision so the fp source is the honest one: {s_i4}"
        );
        // int4 still returns a real, finite cosine near 1.0 (a valid score,
        // just at the coarser stored precision).
        assert!(s_i4.is_finite() && s_i4 > 0.9, "int4 score sane: {s_i4}");
    }
}
