//! float16 conversions for the `encoding=2` embeddings path. The
//! runtime never accumulates in f16 — quantize at write time, decode
//! lane-by-lane into f32 at read time.

/// Convert an L2-normalized float32 embedding to float16, returning the
/// little-endian byte representation. Not a checked NaN/Inf path —
/// callers must validate inputs upstream.
pub fn f32_to_f16_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        let half = half::f16::from_f32(v);
        out.extend_from_slice(&half.to_le_bytes());
    }
    out
}

/// Decode a float16 byte slice into f32 values, accumulating in f32
/// for downstream dot-product accuracy.
#[inline]
pub fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let h = half::f16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(h.to_f32());
    }
    out
}
