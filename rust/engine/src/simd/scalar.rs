//! Scalar fallback dot products. Auto-vectorizes well in release mode
//! and is the path used when SIMD detection fails or
//! `URNA_FORCE_SCALAR=1`. Also the reference implementation the SIMD
//! parity tests compare against.

#[inline]
pub fn dot_f32_scalar(q: &[f32], row_bytes: &[u8]) -> f32 {
    let mut acc = 0.0f32;
    for (i, qv) in q.iter().enumerate() {
        let off = i * 4;
        let v = f32::from_le_bytes([
            row_bytes[off],
            row_bytes[off + 1],
            row_bytes[off + 2],
            row_bytes[off + 3],
        ]);
        acc += *qv * v;
    }
    acc
}

#[inline]
pub fn dot_f32_f16_scalar(q: &[f32], row_bytes: &[u8]) -> f32 {
    let mut acc = 0.0f32;
    for (i, qv) in q.iter().enumerate() {
        let off = i * 2;
        let h = half::f16::from_le_bytes([row_bytes[off], row_bytes[off + 1]]);
        acc += *qv * h.to_f32();
    }
    acc
}

#[inline]
pub fn dot_f32_i8_scalar(q: &[f32], row: &[i8]) -> f32 {
    let mut acc = 0.0f32;
    for (qv, &iv) in q.iter().zip(row.iter()) {
        acc += *qv * (iv as f32);
    }
    acc
}

/// Sign-extend a 4-bit nibble (low 4 bits of `b`) to f32 in `[-8, 7]`.
#[inline]
fn nib_to_f32(b: u8) -> f32 {
    let n = b & 0x0F;
    let s = if n & 0x08 != 0 {
        (n | 0xF0) as i8
    } else {
        n as i8
    };
    s as f32
}

/// Fused dequant + dot for int4 block-`block` codes against an f32 query.
/// `codes` is `dim/2` packed bytes (two nibbles each, low nibble first);
/// `group_scales` is one f32 per `block`-dim group. Each component
/// contributes `q[j] * code[j] * group_scales[j / block]`, accumulated in
/// f32 per group so the per-group scale multiplies the group's partial sum
/// once (matching the SIMD backends bit-for-bit).
#[inline]
pub fn dot_f32_i4_blocked_scalar(
    q: &[f32],
    codes: &[u8],
    group_scales: &[f32],
    dim: usize,
    block: usize,
) -> f32 {
    let mut acc = 0.0f32;
    let _ = dim;
    for (g, &scale) in group_scales.iter().enumerate() {
        let mut part = 0.0f32;
        let base = g * block;
        for j in 0..block {
            let idx = base + j;
            let byte = codes[idx / 2];
            let nib = if idx % 2 == 0 { byte } else { byte >> 4 };
            part += q[idx] * nib_to_f32(nib);
        }
        acc += part * scale;
    }
    acc
}
