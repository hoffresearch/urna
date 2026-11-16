//! The SHA-256 compression function on the CPU's own instructions: SHA-NI
//! on x86_64 and the ARMv8 SHA2 extension on aarch64, each chosen at run
//! time only after the CPU reports the feature.
//!
//! This is the one module of urna-format allowed `unsafe` (the crate
//! denies it everywhere else): the kernels are `target_feature` functions
//! over raw-pointer vector loads. Each kernel is one `unsafe` block, and
//! its `// SAFETY:` comment names the two invariants it rests on: the
//! feature was detected, and every load and store stays inside a 64-byte
//! block, the 8-word state or the 64-word constant table. Under miri the
//! portable rounds run instead.

/// Compress `blocks` (a multiple of 64 bytes) into `state` on this CPU's
/// SHA instructions. Returns false, with `state` untouched, when the CPU
/// has none, so the caller runs the portable rounds.
pub(super) fn compress(state: &mut [u32; 8], blocks: &[u8]) -> bool {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        if std::is_x86_feature_detected!("sha")
            && std::is_x86_feature_detected!("ssse3")
            && std::is_x86_feature_detected!("sse4.1")
        {
            // SAFETY: sha, ssse3 and sse4.1 were just detected on this CPU,
            // and `blocks` is whole 64-byte blocks (the caller's contract).
            unsafe { x86::compress(state, blocks) };
            return true;
        }
    }
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        if std::arch::is_aarch64_feature_detected!("sha2") {
            // SAFETY: the sha2 extension was just detected on this CPU, and
            // `blocks` is whole 64-byte blocks (the caller's contract).
            unsafe { arm::compress(state, blocks) };
            return true;
        }
    }
    let _ = (state, blocks);
    false
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
mod x86 {
    use super::super::K;
    use std::arch::x86_64::*;

    /// The rounds on SHA-NI, after Intel's reference: the state lives in
    /// two registers as ABEF and CDGH, each `sha256rnds2` runs two rounds
    /// and `sha256msg1`/`sha256msg2` extend the message schedule.
    ///
    /// # Safety
    ///
    /// The CPU must have sha, ssse3 and sse4.1, and `blocks.len()` must be
    /// a multiple of 64.
    #[target_feature(enable = "sha,ssse3,sse4.1")]
    pub(super) unsafe fn compress(state: &mut [u32; 8], blocks: &[u8]) {
        // SAFETY: the caller detected sha, ssse3 and sse4.1. every load is
        // 16 bytes: the two state loads and stores cover words 0..4 and
        // 4..8 of the 8-word state, the four block loads bytes 0..64 of a
        // `chunks_exact(64)` block, and the constant loads words k..k+4 of
        // the 64-word K for k <= 60; `loadu`/`storeu` take any
        // alignment.
        unsafe {
            let mask = _mm_set_epi64x(0x0C0D_0E0F_0809_0A0B, 0x0405_0607_0001_0203);
            let dcba = _mm_loadu_si128(state.as_ptr().cast());
            let hgfe = _mm_loadu_si128(state.as_ptr().add(4).cast());
            let cdab = _mm_shuffle_epi32(dcba, 0xB1);
            let efgh = _mm_shuffle_epi32(hgfe, 0x1B);
            let mut abef = _mm_alignr_epi8(cdab, efgh, 8);
            let mut cdgh = _mm_blend_epi16(efgh, cdab, 0xF0);
            for block in blocks.chunks_exact(64) {
                let (abef_save, cdgh_save) = (abef, cdgh);
                let p = block.as_ptr().cast::<__m128i>();
                let mut w = [
                    _mm_shuffle_epi8(_mm_loadu_si128(p), mask),
                    _mm_shuffle_epi8(_mm_loadu_si128(p.add(1)), mask),
                    _mm_shuffle_epi8(_mm_loadu_si128(p.add(2)), mask),
                    _mm_shuffle_epi8(_mm_loadu_si128(p.add(3)), mask),
                ];
                // four rounds: the schedule words in `$msg` plus K[$k..$k + 4].
                macro_rules! rounds4 {
                    ($k:expr, $msg:expr) => {{
                        let k = _mm_loadu_si128(K.as_ptr().add($k).cast());
                        let t1 = _mm_add_epi32($msg, k);
                        cdgh = _mm_sha256rnds2_epu32(cdgh, abef, t1);
                        let t2 = _mm_shuffle_epi32(t1, 0x0E);
                        abef = _mm_sha256rnds2_epu32(abef, cdgh, t2);
                    }};
                }
                rounds4!(0, w[0]);
                rounds4!(4, w[1]);
                rounds4!(8, w[2]);
                rounds4!(12, w[3]);
                for r in 4..16 {
                    let i = r % 4;
                    let (w1, w2, w3) = (w[(i + 1) % 4], w[(i + 2) % 4], w[(i + 3) % 4]);
                    let t =
                        _mm_add_epi32(_mm_sha256msg1_epu32(w[i], w1), _mm_alignr_epi8(w3, w2, 4));
                    w[i] = _mm_sha256msg2_epu32(t, w3);
                    rounds4!(4 * r, w[i]);
                }
                abef = _mm_add_epi32(abef, abef_save);
                cdgh = _mm_add_epi32(cdgh, cdgh_save);
            }
            let feba = _mm_shuffle_epi32(abef, 0x1B);
            let dchg = _mm_shuffle_epi32(cdgh, 0xB1);
            let dcba = _mm_blend_epi16(feba, dchg, 0xF0);
            let hgef = _mm_alignr_epi8(dchg, feba, 8);
            _mm_storeu_si128(state.as_mut_ptr().cast(), dcba);
            _mm_storeu_si128(state.as_mut_ptr().add(4).cast(), hgef);
        }
    }
}

#[cfg(all(target_arch = "aarch64", not(miri)))]
mod arm {
    use super::super::K;
    use std::arch::aarch64::*;

    /// The rounds on the ARMv8 SHA2 extension: the state lives in two
    /// registers as ABCD and EFGH, `sha256h`/`sha256h2` run four rounds and
    /// `sha256su0`/`sha256su1` extend the message schedule.
    ///
    /// # Safety
    ///
    /// The CPU must have the sha2 extension, and `blocks.len()` must be a
    /// multiple of 64.
    #[target_feature(enable = "neon,sha2")]
    pub(super) unsafe fn compress(state: &mut [u32; 8], blocks: &[u8]) {
        // SAFETY: the caller detected sha2 (which implies neon). every load
        // is 16 bytes: the state loads and stores cover words 0..4 and 4..8
        // of the 8-word state, the four block loads bytes 0..64 of a
        // `chunks_exact(64)` block, and the constant loads words t..t+4 of
        // the 64-word K for t <= 60; `vld1q`/`vst1q` take any alignment.
        unsafe {
            let mut abcd = vld1q_u32(state.as_ptr());
            let mut efgh = vld1q_u32(state.as_ptr().add(4));
            for block in blocks.chunks_exact(64) {
                let (abcd_save, efgh_save) = (abcd, efgh);
                let p = block.as_ptr();
                // the message words are big endian: byte-reverse each one.
                let mut s0 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p)));
                let mut s1 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p.add(16))));
                let mut s2 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p.add(32))));
                let mut s3 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p.add(48))));
                // four rounds: the schedule words in `$msg` plus K[t..t+4].
                macro_rules! rounds4 {
                    ($msg:expr, $t:expr) => {{
                        let tmp = vaddq_u32($msg, vld1q_u32(K.as_ptr().add($t)));
                        let prev = abcd;
                        abcd = vsha256hq_u32(prev, efgh, tmp);
                        efgh = vsha256h2q_u32(efgh, prev, tmp);
                    }};
                }
                rounds4!(s0, 0);
                rounds4!(s1, 4);
                rounds4!(s2, 8);
                rounds4!(s3, 12);
                for t in [16, 32, 48] {
                    s0 = vsha256su1q_u32(vsha256su0q_u32(s0, s1), s2, s3);
                    rounds4!(s0, t);
                    s1 = vsha256su1q_u32(vsha256su0q_u32(s1, s2), s3, s0);
                    rounds4!(s1, t + 4);
                    s2 = vsha256su1q_u32(vsha256su0q_u32(s2, s3), s0, s1);
                    rounds4!(s2, t + 8);
                    s3 = vsha256su1q_u32(vsha256su0q_u32(s3, s0), s1, s2);
                    rounds4!(s3, t + 12);
                }
                abcd = vaddq_u32(abcd, abcd_save);
                efgh = vaddq_u32(efgh, efgh_save);
            }
            vst1q_u32(state.as_mut_ptr(), abcd);
            vst1q_u32(state.as_mut_ptr().add(4), efgh);
        }
    }
}
