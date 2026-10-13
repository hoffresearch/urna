// SPDX-FileCopyrightText: 2026 Brenner Cruvinel, V. Hoff Research
//
// SPDX-License-Identifier: MIT

//! zstd compression for non-embedding sections. Embeddings are never
//! zstd-compressed - they live in mmap and the runtime reads them via
//! SIMD straight from disk.

use crate::error::UrnaError;

/// Default zstd compression level. 19 is in the "high" tier - slow to
/// encode but a one-time cost and yields ~30% smaller text payloads
/// than the default level 3.
pub const DEFAULT_ZSTD_LEVEL: i32 = 19;

/// Compress with zstd at `DEFAULT_ZSTD_LEVEL`. Returns the compressed
/// bytes ready to write as the section payload.
pub fn zstd_encode(bytes: &[u8]) -> crate::Result<Vec<u8>> {
    zstd::encode_all(bytes, DEFAULT_ZSTD_LEVEL)
        .map_err(|e| UrnaError::InvalidInput(format!("zstd compression failed: {}", e)))
}

/// a reusable zstd context for many small frames. [`zstd_encode`] builds a
/// new level-19 context per call, which dominates the cost of encoding tens
/// of thousands of chunk-sized frames. this keeps one context and resets its
/// session between frames, and feeds it the same way `zstd::encode_all`
/// does (unknown pledged size, streaming end), so every frame is
/// byte-identical to `zstd_encode` on the same input.
pub(super) struct ZstdFramer {
    cctx: zstd::zstd_safe::CCtx<'static>,
}

impl ZstdFramer {
    pub(super) fn new() -> Self {
        Self {
            cctx: zstd::zstd_safe::CCtx::create(),
        }
    }

    /// compress `bytes` into one frame equal to `zstd_encode(bytes)`.
    pub(super) fn encode(&mut self, bytes: &[u8]) -> crate::Result<Vec<u8>> {
        let fail = |what: &str, e: String| {
            UrnaError::InvalidInput(format!("zstd compression failed: {}: {}", what, e))
        };
        let code = |c: usize| zstd::zstd_safe::get_error_name(c).to_string();
        self.cctx
            .reset(zstd::zstd_safe::ResetDirective::SessionOnly)
            .map_err(|c| fail("reset", code(c)))?;
        self.cctx
            .set_parameter(zstd::zstd_safe::CParameter::CompressionLevel(
                DEFAULT_ZSTD_LEVEL,
            ))
            .map_err(|c| fail("level", code(c)))?;
        let mut out = Vec::new();
        let mut enc = zstd::stream::write::Encoder::with_context(&mut out, &mut self.cctx);
        std::io::copy(&mut &bytes[..], &mut enc).map_err(|e| fail("write", e.to_string()))?;
        enc.finish().map_err(|e| fail("finish", e.to_string()))?;
        Ok(out)
    }
}

/// hostile-input guard: an attacker can ship a tiny zstd frame that inflates
/// to many GiB and OOM-kills the process at file `open()` (a classic
/// decompression bomb). The decompressed size is bounded to
/// `max(MIN_DECOMPRESS_CAP, MAX_DECOMPRESS_RATIO * compressed_len)`: the floor
/// keeps small legitimate sections working, the ratio bounds amplification on
/// larger inputs. The floor matches the per-stream `STREAM_CAP` the dict codec
/// already uses (`zstd_dict.rs`).
const MIN_DECOMPRESS_CAP: usize = 64 * 1024 * 1024;
const MAX_DECOMPRESS_RATIO: usize = 128;

fn decompress_cap(compressed_len: usize) -> usize {
    compressed_len
        .saturating_mul(MAX_DECOMPRESS_RATIO)
        .max(MIN_DECOMPRESS_CAP)
}

/// Decompress a zstd payload. Internal helper used by `decode_payload`.
/// Bounded by [`decompress_cap`] so a decompression bomb cannot exhaust memory
/// at open time: a frame that declares (or expands past) more than the cap is
/// rejected before the large allocation, never a panic.
pub(super) fn zstd_decode(bytes: &[u8]) -> crate::Result<Vec<u8>> {
    let cap = decompress_cap(bytes.len());
    let bomb = |reason: String| UrnaError::MalformedSectionPayload {
        section_id: 0,
        reason,
    };
    let fail = |e: std::io::Error| bomb(format!("zstd decompression failed: {}", e));
    match zstd::bulk::Decompressor::upper_bound(bytes) {
        // declared frame size already exceeds the cap: refuse before allocating.
        Some(declared) if declared > cap => Err(bomb(format!(
            "zstd frame declares {} bytes, exceeds cap {}",
            declared, cap
        ))),
        // declared size known and within cap: allocate exactly it.
        Some(declared) => zstd::bulk::decompress(bytes, declared).map_err(fail),
        // no size hint: allow up to the cap, erroring if the frame exceeds it.
        None => zstd::bulk::decompress(bytes, cap).map_err(fail),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
    fn roundtrip_within_cap() {
        let original = b"the file is the database ".repeat(256);
        let compressed = zstd_encode(&original).unwrap();
        let decoded = zstd_decode(&compressed).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
    fn decompression_bomb_is_rejected_not_ooming() {
        // a tiny compressed frame declaring more than the cap must error,
        // never inflate. zeros compress to a few bytes but declare their full
        // size in the frame header, which `upper_bound` reads.
        let bomb = vec![0u8; MIN_DECOMPRESS_CAP + 1];
        let compressed = zstd_encode(&bomb).unwrap();
        assert!(compressed.len() < 4096, "bomb should compress tiny");
        let err = zstd_decode(&compressed).unwrap_err();
        assert!(matches!(err, UrnaError::MalformedSectionPayload { .. }));
    }

    #[test]
    #[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
    fn framer_frames_equal_zstd_encode() {
        // one framer reused across inputs of every size class: empty, tiny,
        // and past one 128 KiB zstd block, in an order that would expose any
        // state leaking from one frame into the next.
        let big = b"o arquivo e o banco de dados. ".repeat(10_000);
        let inputs: Vec<&[u8]> = vec![b"", b"a", &big, b"coracao", &big[..131_073], b""];
        let mut framer = ZstdFramer::new();
        for input in inputs {
            assert_eq!(framer.encode(input).unwrap(), zstd_encode(input).unwrap());
        }
    }

    #[test]
    fn cap_has_floor_and_ratio() {
        assert_eq!(decompress_cap(0), MIN_DECOMPRESS_CAP);
        assert_eq!(decompress_cap(usize::MAX), usize::MAX); // saturating, no overflow
        assert_eq!(
            decompress_cap(MIN_DECOMPRESS_CAP),
            MIN_DECOMPRESS_CAP * MAX_DECOMPRESS_RATIO
        );
    }
}
