//! blob_refs (0x14): the content-hash reference table for out-of-line or
//! inlined media blobs (AV1 streams, AVIF images, PDFs, slide scans).
//!
//! OPTIONAL and EXCLUDED from content_hash. Entries are addressed by
//! ordinal from the blob_span_overlay (0x16) `blob_ref_index` column, so
//! entry ORDER is the contract: encode preserves input order, decode
//! returns it unchanged, and two builds of the same table are
//! byte-identical.
//!
//! the record mirrors forge-core's `BlobRef`: a raw 32-byte sha-256 of
//! the original bytes, a uri hint, the original byte length, and whether
//! the heavy bytes are inlined in this .urna (self-contained) or stay
//! out-of-line (catalog sidecar).
//!
//! wire encoding: `raw` (self-describing payload, no compression).
//! all integers le.

use crate::bytes::{array32, le_u32, le_u64};
use crate::error::UrnaError;
use crate::layout::SECTION_BLOB_REFS;

pub const BLOB_REFS_PAYLOAD_VERSION: u32 = 1;

/// smallest possible encoded entry: 32 hash + 4 uri-len + 8 byte-len + 1 flag.
/// used to bound the claimed entry count against the physical payload
/// BEFORE any allocation, so a hostile count never triggers a huge alloc.
const MIN_ENTRY_SIZE: usize = 32 + 4 + 8 + 1;

/// one row of the 0x14 table. `content_hash` is the raw sha-256 of the
/// original blob bytes (a catalog citation can be proven across the
/// reference boundary); `original_uri` is a reopen hint; `inlined` says
/// whether the bytes live inside this .urna.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlobRefRecord {
    pub content_hash: [u8; 32],
    pub original_uri: String,
    pub byte_len: u64,
    pub inlined: bool,
}

fn malformed(reason: impl Into<String>) -> UrnaError {
    UrnaError::MalformedSectionPayload {
        section_id: SECTION_BLOB_REFS,
        reason: reason.into(),
    }
}

/// encode `records` into the 0x14 payload. deterministic: same records in
/// the same order, same bytes.
pub fn encode_blob_refs(records: &[BlobRefRecord]) -> Result<Vec<u8>, UrnaError> {
    let mut size = 12;
    for r in records {
        size += MIN_ENTRY_SIZE + r.original_uri.len();
    }
    let mut out = Vec::with_capacity(size);
    out.extend_from_slice(&BLOB_REFS_PAYLOAD_VERSION.to_le_bytes());
    out.extend_from_slice(&(records.len() as u64).to_le_bytes());
    for r in records {
        out.extend_from_slice(&r.content_hash);
        out.extend_from_slice(&(r.original_uri.len() as u32).to_le_bytes());
        out.extend_from_slice(r.original_uri.as_bytes());
        out.extend_from_slice(&r.byte_len.to_le_bytes());
        out.push(u8::from(r.inlined));
    }
    Ok(out)
}

/// decode the payload back to records. typed errors on truncation, a
/// version mismatch, or a hostile count/uri claim; never panics.
pub fn decode_blob_refs(bytes: &[u8]) -> Result<Vec<BlobRefRecord>, UrnaError> {
    let mut cur = Cursor::new(bytes);
    let version = cur.u32()?;
    if version != BLOB_REFS_PAYLOAD_VERSION {
        return Err(UrnaError::UnsupportedSectionVersion {
            section_id: SECTION_BLOB_REFS,
            version,
        });
    }
    let n = cur.u64()? as usize;
    // bound the claim against the physical payload before allocating:
    // every entry costs at least MIN_ENTRY_SIZE bytes.
    if n > cur.remaining() / MIN_ENTRY_SIZE {
        return Err(malformed("blob_refs: entry count exceeds payload"));
    }
    let mut records = Vec::with_capacity(n);
    for _ in 0..n {
        let content_hash: [u8; 32] = array32(cur.take(32)?)?;
        let uri_len = cur.u32()? as usize;
        if uri_len > cur.remaining() {
            return Err(malformed("blob_refs: uri length exceeds payload"));
        }
        let uri_bytes = cur.take(uri_len)?;
        let original_uri = std::str::from_utf8(uri_bytes)
            .map_err(|_| malformed("blob_refs: uri is not utf-8"))?
            .to_string();
        let byte_len = cur.u64()?;
        let inlined = match cur.u8()? {
            0 => false,
            1 => true,
            other => return Err(malformed(format!("blob_refs: bad inlined flag {}", other))),
        };
        records.push(BlobRefRecord {
            content_hash,
            original_uri,
            byte_len,
            inlined,
        });
    }
    if cur.pos != bytes.len() {
        return Err(malformed("trailing bytes after records"));
    }
    Ok(records)
}

/// light cursor over the payload. every read is bounds-checked and
/// returns a typed error, never a panic on a hostile mmap.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], UrnaError> {
        if n > self.remaining() {
            return Err(malformed("unexpected EOF"));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, UrnaError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, UrnaError> {
        le_u32(self.take(4)?)
    }
    fn u64(&mut self) -> Result<u64, UrnaError> {
        le_u64(self.take(8)?)
    }
}
