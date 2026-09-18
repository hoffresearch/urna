//! Packed vector store for the ANN graph.
//!
//! The HNSW graph needs to read embedding rows as f32 to compute cosine
//! distance, but the on-disk dtype may be float16 or int8. The old path
//! expanded the WHOLE section to `n * dim * 4` f32 at open time and held
//! it for the life of the file: a 4x RAM blow-up over an int8 slab, on
//! top of the mmap. `PackedVectors` keeps the data in its on-disk packing
//! (int8 stays int8 + per-vector scales, float16 stays f16) and decodes
//! ONE row at a time into a caller-provided scratch buffer, reproducing
//! the exact `int8 * scale` / `f16 -> f32` arithmetic the whole-buffer
//! path used, so the graph and the search results are byte-for-byte
//! identical while the resident footprint drops to the packed size.

use urna_format::{INT4_BLOCK, Int4EmbeddingsView, Int8EmbeddingsView, UrnaError, nibble_to_i4};

use crate::error::RuntimeError;

/// Embedding rows in their on-disk packing, decoded to f32 one row at a
/// time. `float32` is kept as-is (it is not the leak); `float16` and
/// `int8` stay packed and decode into a scratch buffer on demand.
pub(crate) enum PackedVectors {
    /// Not yet attached (graph parsed from disk, vectors pending).
    Empty,
    F32(Vec<f32>),
    F16(Vec<u8>),
    Int8 {
        data: Vec<i8>,
        scales: Vec<f32>,
    },
    /// int4 block-64: codes stay as i8 in `[-7, 7]` (one per dim) and the
    /// per-group f16 scales are kept as f32 (`blocks` per row). A row
    /// decodes to `code * group_scale`, matching the int4 view exactly.
    Int4 {
        codes: Vec<i8>,
        scales: Vec<f32>,
        blocks: usize,
    },
}

impl PackedVectors {
    pub(crate) fn empty() -> Self {
        PackedVectors::Empty
    }

    pub(crate) fn is_attached(&self) -> bool {
        !matches!(self, PackedVectors::Empty)
    }

    /// Build a packed store from a decoded embeddings section. Copies the
    /// section into its packed form (compact for int8/float16); never the
    /// `n * dim * 4` f32 expansion the old `materialize_f32_vectors` did.
    pub(crate) fn from_section(
        dtype: &str,
        bytes: &[u8],
        n: usize,
        dim: usize,
    ) -> Result<Self, RuntimeError> {
        match dtype {
            "float32" => {
                let mut out = vec![0.0f32; n * dim];
                for (i, slot) in out.iter_mut().enumerate() {
                    let off = i * 4;
                    *slot = f32::from_le_bytes([
                        bytes[off],
                        bytes[off + 1],
                        bytes[off + 2],
                        bytes[off + 3],
                    ]);
                }
                Ok(PackedVectors::F32(out))
            }
            "float16" => Ok(PackedVectors::F16(bytes.to_vec())),
            "int8" => {
                let view =
                    Int8EmbeddingsView::parse(bytes, n, dim).map_err(RuntimeError::Format)?;
                let data: Vec<i8> = (0..n).flat_map(|i| view.row(i).to_vec()).collect();
                let scales: Vec<f32> = (0..n).map(|i| view.scale(i)).collect();
                Ok(PackedVectors::Int8 { data, scales })
            }
            "int4" => {
                let view =
                    Int4EmbeddingsView::parse(bytes, n, dim).map_err(RuntimeError::Format)?;
                // unpack nibbles to i8 codes (one per dim) and flatten the
                // per-group f16 scales to f32 (blocks per row), so the row
                // decode is `code * group_scale`, identical to the view.
                let mut codes: Vec<i8> = Vec::with_capacity(n * dim);
                for i in 0..n {
                    let packed = view.row_codes(i);
                    for j in 0..dim {
                        let byte = packed[j / 2];
                        let nib = if j % 2 == 0 { byte } else { byte >> 4 };
                        codes.push(nibble_to_i4(nib));
                    }
                }
                let scales: Vec<f32> = (0..n).flat_map(|i| view.row_scales_f32(i)).collect();
                Ok(PackedVectors::Int4 {
                    codes,
                    scales,
                    blocks: view.blocks,
                })
            }
            other => Err(RuntimeError::Format(UrnaError::UnsupportedDType(
                other.into(),
            ))),
        }
    }

    /// A scratch buffer sized for one decoded row, or empty for the f32
    /// store (which borrows its rows directly and never touches scratch).
    pub(crate) fn scratch(&self, dim: usize) -> Vec<f32> {
        match self {
            PackedVectors::F32(_) | PackedVectors::Empty => Vec::new(),
            _ => vec![0.0f32; dim],
        }
    }

    /// Row `i` decoded to f32. For `F32` this borrows the stored slice and
    /// ignores `scratch`; for `F16`/`Int8` it decodes into `scratch` and
    /// returns that, using the SAME arithmetic the whole-buffer path used
    /// (`f16::to_f32`, `int8 as f32 * scale`) so distances are unchanged.
    #[inline]
    pub(crate) fn row<'s>(&'s self, i: usize, dim: usize, scratch: &'s mut [f32]) -> &'s [f32] {
        match self {
            PackedVectors::F32(v) => &v[i * dim..(i + 1) * dim],
            PackedVectors::F16(b) => {
                let off = i * dim * 2;
                for (j, slot) in scratch.iter_mut().enumerate().take(dim) {
                    let lo = b[off + j * 2];
                    let hi = b[off + j * 2 + 1];
                    *slot = half::f16::from_le_bytes([lo, hi]).to_f32();
                }
                &scratch[..dim]
            }
            PackedVectors::Int8 { data, scales } => {
                let scale = scales[i];
                let base = i * dim;
                for (j, slot) in scratch.iter_mut().enumerate().take(dim) {
                    *slot = data[base + j] as f32 * scale;
                }
                &scratch[..dim]
            }
            PackedVectors::Int4 {
                codes,
                scales,
                blocks,
            } => {
                let base = i * dim;
                let scale_base = i * blocks;
                for (j, slot) in scratch.iter_mut().enumerate().take(dim) {
                    let g = j / INT4_BLOCK;
                    *slot = codes[base + j] as f32 * scales[scale_base + g];
                }
                &scratch[..dim]
            }
            PackedVectors::Empty => &scratch[..0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use urna_format::{encode_int4_embeddings, encode_int8_embeddings, f32_to_f16_bytes};

    fn f32_bytes(rows: &[f32]) -> Vec<u8> {
        rows.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn f32_store_borrows_rows_without_scratch() {
        let rows = vec![1.0f32, 0.0, 0.0, 1.0];
        let store = PackedVectors::from_section("float32", &f32_bytes(&rows), 2, 2).unwrap();
        let mut s = store.scratch(2);
        assert!(s.is_empty(), "f32 needs no scratch");
        assert_eq!(store.row(0, 2, &mut s), &[1.0, 0.0]);
        assert_eq!(store.row(1, 2, &mut s), &[0.0, 1.0]);
    }

    #[test]
    fn int8_row_decode_matches_quantization() {
        // A known row; decode must equal int8 * scale exactly.
        let row = [0.5f32, -0.25, 0.125, 1.0];
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        let unit: Vec<f32> = row.iter().map(|x| x / norm).collect();
        let bytes = encode_int8_embeddings(&unit, 1, 4).unwrap();
        let store = PackedVectors::from_section("int8", &bytes, 1, 4).unwrap();
        let view = Int8EmbeddingsView::parse(&bytes, 1, 4).unwrap();
        let scale = view.scale(0);
        let mut s = store.scratch(4);
        let got = store.row(0, 4, &mut s);
        for (j, &g) in got.iter().enumerate() {
            let want = view.row(0)[j] as f32 * scale;
            assert_eq!(g.to_bits(), want.to_bits(), "row[{j}] decode drift");
        }
    }

    #[test]
    fn int4_row_decode_matches_view_group_scale_times_code() {
        // dim = 128 -> 2 blocks. Decoded row must equal the int4 view's
        // group_scale * code bit-for-bit, so HNSW distances are stable.
        let dim = 128;
        let row: Vec<f32> = {
            let raw: Vec<f32> = (0..dim)
                .map(|j| ((j as f32 * 0.017).sin()) + 0.05)
                .collect();
            let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
            raw.iter().map(|x| x / norm).collect()
        };
        let bytes = encode_int4_embeddings(&row, 1, dim).unwrap();
        let store = PackedVectors::from_section("int4", &bytes, 1, dim).unwrap();
        let view = urna_format::Int4EmbeddingsView::parse(&bytes, 1, dim).unwrap();
        let mut s = store.scratch(dim);
        let got = store.row(0, dim, &mut s);
        let packed = view.row_codes(0);
        for (j, &g) in got.iter().enumerate() {
            let byte = packed[j / 2];
            let nib = if j % 2 == 0 { byte } else { byte >> 4 };
            let code = urna_format::nibble_to_i4(nib);
            let want = code as f32 * view.group_scale(0, j / urna_format::INT4_BLOCK);
            assert_eq!(g.to_bits(), want.to_bits(), "int4 row[{j}] decode drift");
        }
    }

    #[test]
    fn f16_row_decode_matches_whole_buffer() {
        let row = [0.3f32, -0.7, 0.1, 0.64];
        let bytes = f32_to_f16_bytes(&row);
        let whole = urna_format::f16_bytes_to_f32(&bytes);
        let store = PackedVectors::from_section("float16", &bytes, 1, 4).unwrap();
        let mut s = store.scratch(4);
        let got = store.row(0, 4, &mut s);
        for (j, (&g, &w)) in got.iter().zip(whole.iter()).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "f16 row[{j}] drift");
        }
    }
}
