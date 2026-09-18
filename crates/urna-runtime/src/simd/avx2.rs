//! x86_64 AVX2+FMA implementations. Gated by `cfg(target_arch =
//! "x86_64")`; the dispatcher only calls these after
//! `is_x86_feature_detected!("avx2")` and `"fma"` both return true.
//!
//! Every kernel here is `unsafe fn` for two reasons the caller must
//! discharge: the `target_feature` (the CPU must have avx2+fma, or the
//! instructions fault), and the raw-pointer loads, which are bounded only
//! by the slice lengths the safe dispatcher in `super` asserts. The
//! `// SAFETY:` comment on each block names exactly those two invariants.

/// # Safety
///
/// avx2+fma must be available, and `row_bytes.len() == q.len() * 4`.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn dot_f32_avx2(q: &[f32], row_bytes: &[u8]) -> f32 {
    // SAFETY: `chunks = dim / 8`, so every `add(i * 8)` load reads 8 f32
    // (32 bytes) that end at or before `dim` floats / `dim * 4` bytes, which
    // the caller guarantees both slices hold; `loadu` tolerates any
    // alignment; the tail is indexed through the bounds-checked slice.
    unsafe {
        use std::arch::x86_64::*;
        let dim = q.len();
        let mut acc = _mm256_setzero_ps();
        let row_ptr = row_bytes.as_ptr() as *const f32;
        let chunks = dim / 8;
        for i in 0..chunks {
            let qv = _mm256_loadu_ps(q.as_ptr().add(i * 8));
            let rv = _mm256_loadu_ps(row_ptr.add(i * 8));
            acc = _mm256_fmadd_ps(qv, rv, acc);
        }
        let mut tail = 0.0f32;
        for (i, &qv) in q.iter().enumerate().skip(chunks * 8) {
            let off = i * 4;
            let v = f32::from_le_bytes([
                row_bytes[off],
                row_bytes[off + 1],
                row_bytes[off + 2],
                row_bytes[off + 3],
            ]);
            tail += qv * v;
        }
        let mut buf = [0.0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), acc);
        buf.iter().sum::<f32>() + tail
    }
}

/// Fused dequant + dot for int4 block-`block` codes. AVX2 unpacks the
/// nibbles 16 packed bytes (32 nibbles) at a time into the caller's f32
/// `scratch` row (no allocation per call), then runs the IDENTICAL
/// per-group scalar reduction over it (float add is not associative, so a
/// lane-parallel reduction would diverge from the scalar backend in the
/// last ulp; the win here is the vectorized unpack).
///
/// # Safety
///
/// avx2+fma must be available; `q.len() == scratch.len() == dim`,
/// `codes.len() == dim / 2`, `group_scales.len() == dim / block`.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn dot_f32_i4_avx2(
    q: &[f32],
    codes: &[u8],
    group_scales: &[f32],
    dim: usize,
    block: usize,
    scratch: &mut [f32],
) -> f32 {
    // SAFETY: `chunks = (dim/2) / 16`, so each 16-byte `loadu_si128` at
    // `c * 16` ends at or before `dim / 2 == codes.len()`, and the two
    // 16-lane stores it feeds cover `scratch[c*32 .. c*32+32]`, which ends
    // at or before `dim == scratch.len()` (the store helper is handed a
    // sub-slice, so a wrong length would fail its own bounds check). The
    // tail loop uses only bounds-checked slice indexing.
    unsafe {
        use std::arch::x86_64::*;
        let nbytes = dim / 2;
        let chunks = nbytes / 16;
        let lo_mask = _mm_set1_epi8(0x0F);
        for c in 0..chunks {
            let packed = _mm_loadu_si128(codes.as_ptr().add(c * 16) as *const __m128i);
            // low nibbles: mask -> shift left 4 -> arithmetic shift right 4.
            let lo_masked = _mm_and_si128(packed, lo_mask);
            let lo = mm_srai_epi8_4(_mm_slli_epi16(lo_masked, 4));
            // high nibbles: arithmetic shift right by 4 (sign-extends bit 7).
            let hi = mm_srai_epi8_4(packed);
            // interleave lo/hi so component order is lo0,hi0,lo1,hi1,...
            let lo_part = _mm_unpacklo_epi8(lo, hi); // components 0..16
            let hi_part = _mm_unpackhi_epi8(lo, hi); // components 16..32
            store_i8x16_as_f32(lo_part, &mut scratch[c * 32..]);
            store_i8x16_as_f32(hi_part, &mut scratch[c * 32 + 16..]);
        }
        for idx in (chunks * 32)..dim {
            let byte = codes[idx / 2];
            let nib = if idx % 2 == 0 { byte } else { byte >> 4 };
            let n = nib & 0x0F;
            let s = if n & 0x08 != 0 {
                (n | 0xF0) as i8
            } else {
                n as i8
            };
            scratch[idx] = s as f32;
        }
        dot_scratch_blocked(q, scratch, group_scales, dim, block)
    }
}

/// Arithmetic shift right by 4 on packed i8 lanes (AVX2 has no epi8 SRA),
/// emulated via the epi16 SRA plus a byte-wise blend of even/odd lanes.
///
/// # Safety
///
/// avx2 must be available. Pure register arithmetic, no memory access.
#[target_feature(enable = "avx2,fma")]
// This body is pure register-only intrinsics (no memory I/O). On our MSRV
// (1.85) those are `unsafe` and the block is required under edition 2024's
// `unsafe_op_in_unsafe_fn`; newer toolchains reclassify them as safe under the
// enabled target feature, making the block redundant (`unused_unsafe`). Allow
// the lint so the same source compiles clean on both.
#[allow(unused_unsafe)]
unsafe fn mm_srai_epi8_4(v: std::arch::x86_64::__m128i) -> std::arch::x86_64::__m128i {
    // SAFETY: register-only intrinsics; the only requirement is the target
    // feature, which the caller's own `target_feature` guarantees.
    unsafe {
        use std::arch::x86_64::*;
        // shift each 16-bit lane right by 4 arithmetically: this is correct
        // for the HIGH byte of each pair; the LOW byte gets the high byte's
        // low bits shifted in, so mask it out and recompute the low byte
        // from a separately-shifted copy.
        let sra16 = _mm_srai_epi16(v, 4);
        // high bytes (odd lanes) are now correct; keep them.
        let hi_bytes = _mm_and_si128(sra16, _mm_set1_epi16(0xFF00u16 as i16));
        // for low bytes: shift the byte into the high half of its 16-bit
        // lane first (<<8), SRA by 4, then SRA the result back down 8, so the
        // low byte is arithmetically shifted in isolation.
        let lo16 = _mm_srai_epi16(_mm_srai_epi16(_mm_slli_epi16(v, 8), 4), 8);
        let lo_bytes = _mm_and_si128(lo16, _mm_set1_epi16(0x00FF));
        _mm_or_si128(hi_bytes, lo_bytes)
    }
}

/// Widen an i8x16 vector to f32 and store into `out[..16]`.
///
/// # Safety
///
/// avx2 must be available and `out.len() >= 16`.
#[target_feature(enable = "avx2,fma")]
unsafe fn store_i8x16_as_f32(v: std::arch::x86_64::__m128i, out: &mut [f32]) {
    assert!(out.len() >= 16, "store_i8x16_as_f32: need 16 lanes");
    // SAFETY: the assert above guarantees the two 8-lane `storeu` writes
    // (`out[0..8]` and `out[8..16]`) stay inside `out`; unaligned stores.
    unsafe {
        use std::arch::x86_64::*;
        let lo = _mm256_cvtepi8_epi32(v);
        let hi = _mm256_cvtepi8_epi32(_mm_srli_si128(v, 8));
        _mm256_storeu_ps(out.as_mut_ptr(), _mm256_cvtepi32_ps(lo));
        _mm256_storeu_ps(out.as_mut_ptr().add(8), _mm256_cvtepi32_ps(hi));
    }
}

/// Per-group reduction over an already-dequantized f32 scratch row,
/// matching `scalar::dot_f32_i4_blocked` operation-for-operation.
#[inline]
fn dot_scratch_blocked(
    q: &[f32],
    scratch: &[f32],
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
            part += q[base + j] * scratch[base + j];
        }
        acc += part * scale;
    }
    acc
}

/// # Safety
///
/// avx2+fma must be available, and `row.len() == q.len()`.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn dot_f32_i8_avx2(q: &[f32], row: &[i8]) -> f32 {
    // SAFETY: `chunks = dim / 8`, so every 8-byte `read_unaligned` of `row`
    // and 8-lane `loadu` of `q` at `i * 8` ends at or before `dim`, which
    // both slices hold by the caller's contract; the tail is bounds-checked.
    unsafe {
        use std::arch::x86_64::*;
        let dim = q.len();
        let mut acc = _mm256_setzero_ps();
        let chunks = dim / 8;
        for i in 0..chunks {
            let i8_ptr = row.as_ptr().add(i * 8) as *const i64;
            // Load 8 i8s (8 bytes) into the low half of an xmm.
            // SAFETY: `row` is an `&[i8]` (1-byte alignment), so the `*const
            // i64` is not guaranteed 8-byte aligned; a plain `*i8_ptr` deref
            // would be UB. `read_unaligned` performs a well-defined unaligned
            // load. `chunks = dim/8` keeps `i*8 + 8 <= dim <= row.len()`.
            let raw = _mm_set1_epi64x(i8_ptr.read_unaligned());
            // Widen i8 -> i32 (8 lanes).
            let widened = _mm256_cvtepi8_epi32(raw);
            let f = _mm256_cvtepi32_ps(widened);
            let qv = _mm256_loadu_ps(q.as_ptr().add(i * 8));
            acc = _mm256_fmadd_ps(qv, f, acc);
        }
        let mut tail = 0.0f32;
        for i in (chunks * 8)..dim {
            tail += q[i] * (row[i] as f32);
        }
        let mut buf = [0.0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), acc);
        buf.iter().sum::<f32>() + tail
    }
}
