//! x86_64 AVX2+FMA kernels. Each is `unsafe fn` because of the
//! `target_feature` and the raw-pointer lanes; the dispatcher in `native`
//! calls them only after detecting avx2 and fma and checking every shape.

/// # Safety
///
/// avx2 and fma must be available; `rows.len()` is a non-zero multiple of
/// `dim` and `out.len() == dim`.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn mean_pool(rows: &[f32], dim: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n_rows = rows.len() / dim;
    let chunks = dim / 8;
    // SAFETY: every 8-lane unaligned load at `row + i * 8` ends at or
    // before `dim` floats inside a row the caller proved whole; the tail
    // goes through the bounds-checked slice.
    unsafe {
        let inv = _mm256_set1_ps(1.0 / n_rows as f32);
        for c in 0..chunks {
            let mut acc = _mm256_setzero_ps();
            for r in 0..n_rows {
                acc = _mm256_add_ps(acc, _mm256_loadu_ps(rows.as_ptr().add(r * dim + c * 8)));
            }
            _mm256_storeu_ps(out.as_mut_ptr().add(c * 8), _mm256_mul_ps(acc, inv));
        }
    }
    for i in chunks * 8..dim {
        let mut s = 0.0f32;
        for r in 0..n_rows {
            s += rows[r * dim + i];
        }
        out[i] = s / n_rows as f32;
    }
}

/// # Safety
///
/// avx2 and fma must be available.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn l2_normalize(v: &mut [f32]) {
    use std::arch::x86_64::*;
    let chunks = v.len() / 8;
    let mut sumsq = 0.0f32;
    // SAFETY: every 8-lane unaligned load and store at `i * 8` ends at or
    // before `v.len()`; the tail goes through the bounds-checked slice.
    unsafe {
        let mut acc = _mm256_setzero_ps();
        for i in 0..chunks {
            let x = _mm256_loadu_ps(v.as_ptr().add(i * 8));
            acc = _mm256_fmadd_ps(x, x, acc);
        }
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
        sumsq += lanes.iter().sum::<f32>();
        for x in &v[chunks * 8..] {
            sumsq += x * x;
        }
        let norm = sumsq.sqrt();
        if norm <= 0.0 {
            return;
        }
        let inv = _mm256_set1_ps(1.0 / norm);
        for i in 0..chunks {
            let x = _mm256_loadu_ps(v.as_ptr().add(i * 8));
            _mm256_storeu_ps(v.as_mut_ptr().add(i * 8), _mm256_mul_ps(x, inv));
        }
        for x in &mut v[chunks * 8..] {
            *x /= norm;
        }
    }
}
