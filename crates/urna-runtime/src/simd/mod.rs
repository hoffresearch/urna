//! SIMD-accelerated dot products for the search hot path.
//!
//! Three dtype paths, three SIMD targets:
//!
//! ```text
//!                AVX2 (x86_64)         NEON (aarch64)        Scalar
//!   f32 · f32    8 lanes (f32x8)       4 lanes (f32x4)       autovec
//!   f32 · f16    8 lanes (load+cvt)    4 lanes (load+cvt)    autovec
//!   f32 · i8     16 lanes (i8 -> i32)  16 lanes (i8 -> i32)  autovec
//!   f32 · i4     32-nib unpack/step    32-nib unpack/step    autovec
//! ```
//!
//! The int4 kernel is block-64 (per-group f16 absmax scale). SIMD
//! vectorizes the nibble unpack but reduces per group identically to the
//! scalar path, so all three backends agree bit-for-bit.
//!
//! Accumulators are always f32. The query is f32 (L2-normalized), the
//! database is f32 / f16 / i8 (i8 with a per-vector scale). Final score
//! is the real cosine.
//!
//! Detection happens once at module load via `OnceLock`. The dispatch
//! function is a function pointer chosen at first call, so the per-query
//! cost is one indirect call, not a CPUID check per vector.
//!
//! Safety boundary. Every `unsafe` kernel below reads through raw pointers
//! sized by `q.len()` / `dim`, so the safe dispatchers in this module are
//! the ONLY place the length invariants are checked, and they check with
//! `assert!` (kept in release builds), never `debug_assert!`. The reader
//! validates section sizes at open time too, but a defence three layers
//! away from the pointer is not a defence; the cost is one integer compare
//! per row, invisible next to the dot product itself.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "aarch64")]
mod neon;
mod scalar;

#[cfg(test)]
mod tests;

use std::sync::OnceLock;

use urna_format::Int8EmbeddingsView;

pub use scalar::{
    dot_f32_f16_scalar, dot_f32_i4_blocked_scalar, dot_f32_i8_scalar, dot_f32_scalar,
};

/// What backend is the runtime using right now? Useful for `urna stats`
/// / benchmarks so the user can see whether SIMD is active.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdBackend {
    Scalar,
    Avx2,
    Neon,
}

impl SimdBackend {
    pub fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
            Self::Neon => "neon",
        }
    }
}

static BACKEND: OnceLock<SimdBackend> = OnceLock::new();

/// The SIMD backend selected at runtime. Cached after the first call.
///
/// Set `URNA_FORCE_SCALAR=1` to disable SIMD entirely — useful for
/// before/after SIMD benchmarks on the same binary.
pub fn detect_backend() -> SimdBackend {
    *BACKEND.get_or_init(|| {
        if std::env::var("URNA_FORCE_SCALAR")
            .map(|v| v != "0")
            .unwrap_or(false)
        {
            return SimdBackend::Scalar;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                return SimdBackend::Avx2;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return SimdBackend::Neon;
            }
        }
        SimdBackend::Scalar
    })
}

/// Dot product between an f32 query and an f32 row stored as little-endian
/// bytes (the way embeddings live in mmap).
#[inline]
pub fn dot_f32_bytes(q: &[f32], row_bytes: &[u8]) -> f32 {
    assert_eq!(
        row_bytes.len(),
        q.len() * 4,
        "dot_f32_bytes: row byte length must be 4 * query dim"
    );
    match detect_backend() {
        // SAFETY: the assert above bounds every `add(i * 8)` load in the
        // kernel (`i < dim / 8`) inside both `q` and `row_bytes`; the backend
        // was selected only after `avx2` + `fma` were detected on this CPU.
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx2 => unsafe { avx2::dot_f32_avx2(q, row_bytes) },
        // SAFETY: same length invariant (lanes of 4 here); `neon` detected.
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => unsafe { neon::dot_f32_neon(q, row_bytes) },
        _ => scalar::dot_f32_scalar(q, row_bytes),
    }
}

/// Dot product between an f32 query and an f16 row stored as little-endian
/// bytes. Accumulates in f32. The query stays f32 (it is normalized once
/// per call, no need to drop precision there).
#[inline]
pub fn dot_f32_f16_bytes(q: &[f32], row_bytes: &[u8]) -> f32 {
    assert_eq!(
        row_bytes.len(),
        q.len() * 2,
        "dot_f32_f16_bytes: row byte length must be 2 * query dim"
    );
    match detect_backend() {
        // the neon f16 arm needs rustc >= 1.94 (cfg(neon_f16), see build.rs);
        // on older toolchains only this f16 path falls back to scalar.
        // SAFETY: the assert above bounds every 4-lane u16 load (`i < dim / 4`)
        // inside `row_bytes` and every f32 load inside `q`; `neon` detected.
        #[cfg(all(target_arch = "aarch64", neon_f16))]
        SimdBackend::Neon => unsafe { neon::dot_f32_f16_neon(q, row_bytes) },
        // AVX2 has no native f16->f32 unless F16C is present; our cutoff
        // is "AVX2 + FMA" which usually pulls F16C along. Using a portable
        // unpack here keeps the AVX2 path simple and avoids the F16C
        // detection branch.
        _ => scalar::dot_f32_f16_scalar(q, row_bytes),
    }
}

/// Dot product between an f32 query and a single i8 row, multiplied by
/// the row's f32 scale. `q` stays f32; the i8 row is widened to i32 in
/// the inner loop, multiplied by f32 lanes of `q`, accumulated in f32.
///
/// `f32_value ≈ i8_value * scale`, so:
///   `q · v = scale * sum_i(q_i * i8_i)`.
#[inline]
pub fn dot_f32_i8(q: &[f32], row: &[i8], scale: f32) -> f32 {
    assert_eq!(
        row.len(),
        q.len(),
        "dot_f32_i8: int8 row length must equal query dim"
    );
    let acc = match detect_backend() {
        // SAFETY: the assert above bounds every 8-lane load (`i < dim / 8`)
        // inside both `q` and `row`; the backend was selected only after
        // `avx2` + `fma` were detected on this CPU.
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx2 => unsafe { avx2::dot_f32_i8_avx2(q, row) },
        // SAFETY: same length invariant; `neon` detected.
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => unsafe { neon::dot_f32_i8_neon(q, row) },
        _ => scalar::dot_f32_i8_scalar(q, row),
    };
    acc * scale
}

/// Fused dequant + dot for an int4 block-`block` row against an f32 query.
/// `codes` is `dim/2` packed nibble bytes (low nibble first), `group_scales`
/// is one f32 per `block`-dim group. The SIMD backends vectorize the nibble
/// unpack but reduce per-group identically to scalar, so the result is
/// bit-for-bit equal across all three backends (float add is not
/// associative; a lane-parallel reduction would diverge in the last ulp).
///
/// `scratch` is the caller-owned `dim`-long f32 row the SIMD backends unpack
/// the nibbles into. It is an argument, not a local, so a rerank over
/// thousands of candidates reuses ONE allocation instead of paying a
/// `malloc`/`free` pair per row (the scalar backend never touches it).
/// Its contents on return are unspecified.
///
/// `f32_value ~= code * group_scales[group]`, so the cosine is
/// `sum_g scale_g * sum_{j in g} q_j * code_j` accumulated in f32.
///
/// # Panics
///
/// When any length disagrees with `dim` / `block`: these asserts are the
/// safety boundary for the raw-pointer kernels and stay on in release.
#[inline]
pub fn dot_f32_i4_blocked(
    q: &[f32],
    codes: &[u8],
    group_scales: &[f32],
    dim: usize,
    block: usize,
    scratch: &mut [f32],
) -> f32 {
    assert!(
        block > 0 && dim % block == 0,
        "int4: dim must be a multiple of block"
    );
    assert_eq!(q.len(), dim, "int4: query length must equal dim");
    assert_eq!(
        codes.len(),
        dim / 2,
        "int4: packed codes must be dim/2 bytes"
    );
    assert_eq!(group_scales.len(), dim / block, "int4: one scale per block");
    assert_eq!(scratch.len(), dim, "int4: scratch row must be dim long");
    match detect_backend() {
        // SAFETY: the asserts above give the kernel `dim/2` code bytes, a
        // `dim`-long scratch row and `dim/block` scales, so every 16-byte
        // nibble load (`c < (dim/2) / 16`) and every 32-lane scratch store
        // (`c * 32 + 32 <= dim`) is in bounds; `avx2` + `fma` detected.
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx2 => unsafe {
            avx2::dot_f32_i4_avx2(q, codes, group_scales, dim, block, scratch)
        },
        // SAFETY: same invariants; `neon` detected.
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => unsafe {
            neon::dot_f32_i4_neon(q, codes, group_scales, dim, block, scratch)
        },
        _ => scalar::dot_f32_i4_blocked_scalar(q, codes, group_scales, dim, block),
    }
}

/// Score every row of an int8 embeddings section against `q`.
/// `out[i]` is the cosine score; the runtime sorts these.
pub fn score_int8_section(q: &[f32], view: &Int8EmbeddingsView<'_>, out: &mut [f32]) {
    assert_eq!(
        out.len(),
        view.n,
        "score_int8_section: one output slot per row"
    );
    for (i, slot) in out.iter_mut().enumerate().take(view.n) {
        let scale = view.scale(i);
        let row = view.row(i);
        *slot = dot_f32_i8(q, row, scale);
    }
}
