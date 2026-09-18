//! SIMD parity tests. Compare each backend's dispatched output to the
//! scalar reference; tolerance is 1e-4 for f32 / i8 paths and 1e-3 for
//! f16 (which loses ~5e-4 per component to rounding).

use super::*;

fn random_normalized(seed: u64, dim: usize) -> Vec<f32> {
    // Linear congruential — deterministic, no rand dep needed.
    let mut state = seed
        .wrapping_mul(2862933555777941757)
        .wrapping_add(3037000493);
    let mut v = Vec::with_capacity(dim);
    for _ in 0..dim {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (state >> 32) as u32;
        let u = (bits as f32) / (u32::MAX as f32);
        v.push(u - 0.5);
    }
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in &mut v {
            *x /= n;
        }
    }
    v
}

fn f32_to_le_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

#[test]
fn detect_returns_a_backend() {
    let b = detect_backend();
    // Cached after first call.
    assert_eq!(b, detect_backend());
}

#[test]
fn f32_simd_matches_scalar() {
    for dim in [4, 7, 16, 31, 64, 384] {
        let q = random_normalized(0xDEAD_BEEF, dim);
        let row = random_normalized(0xC0DE_FACE, dim);
        let row_bytes = f32_to_le_bytes(&row);
        let scalar = dot_f32_scalar(&q, &row_bytes);
        let dispatched = dot_f32_bytes(&q, &row_bytes);
        assert!(
            (scalar - dispatched).abs() < 1e-4,
            "dim={} scalar={} dispatched={} diff={}",
            dim,
            scalar,
            dispatched,
            (scalar - dispatched).abs()
        );
    }
}

#[test]
fn f16_simd_matches_scalar() {
    for dim in [4, 8, 31, 64, 384] {
        let q = random_normalized(0x1234, dim);
        let row = random_normalized(0x5678, dim);
        let f16_bytes = urna_format::f32_to_f16_bytes(&row);
        let scalar = dot_f32_f16_scalar(&q, &f16_bytes);
        let dispatched = dot_f32_f16_bytes(&q, &f16_bytes);
        assert!(
            (scalar - dispatched).abs() < 1e-3,
            "dim={} scalar={} dispatched={}",
            dim,
            scalar,
            dispatched
        );
    }
}

#[test]
fn i8_simd_matches_scalar() {
    for dim in [8, 31, 64, 384] {
        let q = random_normalized(0xABCD, dim);
        let row = random_normalized(0x4321, dim);
        let (scale, q_row) = urna_format::quantize_f32_to_i8(&row);
        let scalar = dot_f32_i8_scalar(&q, &q_row) * scale;
        let dispatched = dot_f32_i8(&q, &q_row, scale);
        assert!(
            (scalar - dispatched).abs() < 1e-4,
            "dim={} scalar={} dispatched={} diff={}",
            dim,
            scalar,
            dispatched,
            (scalar - dispatched).abs()
        );
    }
}

#[test]
fn i4_blocked_dispatched_matches_scalar_bit_for_bit() {
    // The dispatched backend (neon/avx2/scalar) MUST equal the scalar
    // reference bit-for-bit on random block-64 rows. dims cover one block
    // (64), the corpus dim (384 -> 192 packed bytes -> 12 full 16B SIMD
    // chunks), and 320 (160 packed bytes -> 10 full 16B chunks): every
    // valid int4 dim is a multiple of 64 so dim/2 is a multiple of 16 and
    // the SIMD nibble loop has no partial 16-byte chunk, but the wider
    // sweep still exercises the chunk count beyond a single step.
    let block = urna_format::INT4_BLOCK;
    for dim in [64usize, 128, 320, 384] {
        let q = random_normalized(0xF00D, dim);
        let row = random_normalized(0xBEEF, dim);
        let (scales, codes) = urna_format::quantize_f32_to_i4(&row, dim);
        let packed = urna_format::pack_nibbles(&codes);
        let scales_f32: Vec<f32> = scales.iter().map(|s| s.to_f32()).collect();
        let scalar = dot_f32_i4_blocked_scalar(&q, &packed, &scales_f32, dim, block);
        let mut scratch = vec![0.0f32; dim];
        let dispatched = dot_f32_i4_blocked(&q, &packed, &scales_f32, dim, block, &mut scratch);
        assert_eq!(
            scalar.to_bits(),
            dispatched.to_bits(),
            "dim={dim}: scalar={scalar} dispatched={dispatched} not bit-identical",
        );
    }
}

#[test]
fn i4_blocked_matches_reference_dequant_dot() {
    // The fused kernel equals an explicit f32 dequant + dot within 1e-5.
    let block = urna_format::INT4_BLOCK;
    for dim in [64usize, 128, 320, 384] {
        let q = random_normalized(0x1357, dim);
        let row = random_normalized(0x2468, dim);
        let (scales, codes) = urna_format::quantize_f32_to_i4(&row, dim);
        let packed = urna_format::pack_nibbles(&codes);
        let scales_f32: Vec<f32> = scales.iter().map(|s| s.to_f32()).collect();

        // reference: dequant each code to f32 then a plain dot.
        let mut reference = 0.0f32;
        for (j, &c) in codes.iter().enumerate() {
            reference += q[j] * (c as f32 * scales_f32[j / block]);
        }
        let mut scratch = vec![0.0f32; dim];
        let fused = dot_f32_i4_blocked(&q, &packed, &scales_f32, dim, block, &mut scratch);
        assert!(
            (fused - reference).abs() < 1e-5,
            "dim={dim}: fused={fused} reference={reference}",
        );
    }
}
