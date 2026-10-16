//! aarch64 NEON kernels. Each is `unsafe fn` because of the
//! `target_feature` and the raw-pointer lanes; the dispatcher in `native`
//! calls them only after detecting neon and checking every shape.

/// # Safety
///
/// neon must be available; `rows.len()` is a non-zero multiple of `dim`
/// and `out.len() == dim`.
#[target_feature(enable = "neon")]
pub(super) unsafe fn mean_pool(rows: &[f32], dim: usize, out: &mut [f32]) {
    use std::arch::aarch64::*;
    let n_rows = rows.len() / dim;
    let chunks = dim / 4;
    // SAFETY: every 4-lane load at `row + i * 4` ends at or before `dim`
    // floats inside a row the caller proved whole; the tail goes through
    // the bounds-checked slice.
    unsafe {
        for c in 0..chunks {
            let mut acc = vdupq_n_f32(0.0);
            for r in 0..n_rows {
                acc = vaddq_f32(acc, vld1q_f32(rows.as_ptr().add(r * dim + c * 4)));
            }
            let inv = vdupq_n_f32(1.0 / n_rows as f32);
            vst1q_f32(out.as_mut_ptr().add(c * 4), vmulq_f32(acc, inv));
        }
    }
    for i in chunks * 4..dim {
        let mut s = 0.0f32;
        for r in 0..n_rows {
            s += rows[r * dim + i];
        }
        out[i] = s / n_rows as f32;
    }
}

/// # Safety
///
/// neon must be available.
#[target_feature(enable = "neon")]
pub(super) unsafe fn l2_normalize(v: &mut [f32]) {
    use std::arch::aarch64::*;
    let chunks = v.len() / 4;
    let mut sumsq = 0.0f32;
    // SAFETY: every 4-lane load and store at `i * 4` ends at or before
    // `v.len()`; the tail goes through the bounds-checked slice.
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let x = vld1q_f32(v.as_ptr().add(i * 4));
            acc = vfmaq_f32(acc, x, x);
        }
        sumsq += vaddvq_f32(acc);
        for x in &v[chunks * 4..] {
            sumsq += x * x;
        }
        let norm = sumsq.sqrt();
        if norm <= 0.0 {
            return;
        }
        let inv = vdupq_n_f32(1.0 / norm);
        for i in 0..chunks {
            let x = vld1q_f32(v.as_ptr().add(i * 4));
            vst1q_f32(v.as_mut_ptr().add(i * 4), vmulq_f32(x, inv));
        }
        for x in &mut v[chunks * 4..] {
            *x /= norm;
        }
    }
}
