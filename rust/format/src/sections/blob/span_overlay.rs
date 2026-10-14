//! blob_span_overlay (0x16): per-chunk overlay that replaces the canonical
//! chunks_original_spans (0x03) for out-of-line media (images, video, audio).
//!
//! OPTIONAL and EXCLUDED from content_hash. When present, the runtime uses
//! these spans for `cite` and `retrieve` instead of the spans in 0x03, so
//! a corpus whose chunks index into an AV1 stream or external catalog keeps
//! the SAME content_hash as its text-only twin.
//!
//! wire encoding: `raw` (self-describing payload, no compression).
//! all integers le.

use crate::bytes::{le_u32, le_u64};
use crate::error::UrnaError;
use crate::layout::SECTION_BLOB_SPAN_OVERLAY;

pub const BLOB_SPAN_OVERLAY_PAYLOAD_VERSION: u32 = 1;
pub const BLOB_REF_NONE: u32 = 0xFFFF_FFFF;

/// one entry per chunk. `blob_ref_index` points into the 0x14 blob_refs
/// table (or BLOB_REF_NONE for legacy/text chunks). `byte_start`/`byte_end`
/// are the span inside that blob (or ordinal placeholders for backward
/// compatibility with the current python image builder).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlobSpanEntry {
    pub blob_ref_index: u32,
    pub byte_start: u64,
    pub byte_end: u64,
}

fn malformed(reason: impl Into<String>) -> UrnaError {
    UrnaError::MalformedSectionPayload {
        section_id: SECTION_BLOB_SPAN_OVERLAY,
        reason: reason.into(),
    }
}

/// encode `entries` into the 0x16 payload. deterministic: same entries,
/// same bytes.
pub fn encode_blob_span_overlay(entries: &[BlobSpanEntry]) -> Result<Vec<u8>, UrnaError> {
    let mut out = Vec::with_capacity(12 + entries.len() * 16);
    out.extend_from_slice(&BLOB_SPAN_OVERLAY_PAYLOAD_VERSION.to_le_bytes());
    out.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for e in entries {
        out.extend_from_slice(&e.blob_ref_index.to_le_bytes());
        out.extend_from_slice(&e.byte_start.to_le_bytes());
        out.extend_from_slice(&e.byte_end.to_le_bytes());
    }
    Ok(out)
}

/// decode the payload back to entries. typed errors on truncation or version
/// mismatch; never panics.
pub fn decode_blob_span_overlay(bytes: &[u8]) -> Result<Vec<BlobSpanEntry>, UrnaError> {
    let mut cur = Cursor::new(bytes);
    let version = cur.u32()?;
    if version != BLOB_SPAN_OVERLAY_PAYLOAD_VERSION {
        return Err(UrnaError::UnsupportedSectionVersion {
            section_id: SECTION_BLOB_SPAN_OVERLAY,
            version,
        });
    }
    let n = cur.u64()? as usize;
    // bound the claim against the physical payload before allocating:
    // every entry costs exactly 16 bytes (u32 + u64 + u64).
    if n > (bytes.len() - cur.pos) / 16 {
        return Err(malformed("blob_span_overlay: entry count exceeds payload"));
    }
    let mut entries = Vec::with_capacity(n);
    for _ in 0..n {
        entries.push(BlobSpanEntry {
            blob_ref_index: cur.u32()?,
            byte_start: cur.u64()?,
            byte_end: cur.u64()?,
        });
    }
    if cur.pos != bytes.len() {
        return Err(malformed("trailing bytes after entries"));
    }
    Ok(entries)
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], UrnaError> {
        if n > self.buf.len() - self.pos {
            return Err(malformed("unexpected EOF"));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, UrnaError> {
        le_u32(self.take(4)?)
    }
    fn u64(&mut self) -> Result<u64, UrnaError> {
        le_u64(self.take(8)?)
    }
}
