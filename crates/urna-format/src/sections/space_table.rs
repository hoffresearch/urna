//! space_table (0x15): the per-space embedding directory for multimodal
//! corpora. space[0] is always the canonical text embeddings (0x04) and is
//! NEVER listed here; every entry describes one non-text space whose
//! vectors live in the fixed-stride band sections 0x20-0x2F (with an
//! optional fp rerank source in 0x30-0x3F).
//!
//! OPTIONAL and EXCLUDED from content_hash. each space carries its own
//! model_hash and dim, so the per-space honesty gate works exactly like
//! the corpus-level one: a query embedded with the wrong model fails
//! loudly instead of silently scoring against the wrong band.
//!
//! wire encoding: `raw` (self-describing payload, no compression).
//! all integers le.

use crate::bytes::{le_u32, le_u64};
use crate::error::UrnaError;
use crate::layout::{SECTION_SPACE_TABLE, SPACE_BAND_LEN};

pub const SPACE_TABLE_PAYLOAD_VERSION: u32 = 1;

/// dtype codes for the band slab (mirror the manifest dtype strings).
pub const SPACE_DTYPE_F32: u8 = 0;
pub const SPACE_DTYPE_F16: u8 = 1;
pub const SPACE_DTYPE_I8: u8 = 2;
pub const SPACE_DTYPE_I4: u8 = 3;

/// smallest possible encoded entry (index, name-len, dim, dtype, hash-len,
/// n_vectors): 22 bytes. the claimed entry count is bounded against the
/// physical payload BEFORE any allocation.
const MIN_ENTRY_SIZE: usize = 1 + 4 + 4 + 1 + 4 + 8;

/// one row of the 0x15 table. `space_index` addresses the band: the
/// vectors live at section 0x20 + space_index (and the optional fp source
/// at 0x30 + space_index). `model_hash` fingerprints the model that
/// produced THIS space's vectors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpaceEntry {
    pub space_index: u8,
    pub name: String,
    pub dim: u32,
    pub dtype: u8,
    pub model_hash: String,
    pub n_vectors: u64,
}

impl SpaceEntry {
    /// the manifest-style dtype string for this space's band slab.
    pub fn dtype_str(&self) -> &'static str {
        match self.dtype {
            SPACE_DTYPE_F32 => "float32",
            SPACE_DTYPE_F16 => "float16",
            SPACE_DTYPE_I8 => "int8",
            SPACE_DTYPE_I4 => "int4",
            _ => "unknown",
        }
    }
}

fn malformed(reason: impl Into<String>) -> UrnaError {
    UrnaError::MalformedSectionPayload {
        section_id: SECTION_SPACE_TABLE,
        reason: reason.into(),
    }
}

/// validate one entry before encode (and after decode): space 0 is the
/// canonical text space and must never be listed; the index must fit the
/// band; the dtype code must be known; name and model_hash non-empty.
fn check_entry(e: &SpaceEntry) -> Result<(), UrnaError> {
    if e.space_index == 0 || e.space_index >= SPACE_BAND_LEN as u8 {
        return Err(malformed(format!(
            "space_table: space_index {} outside 1..{}",
            e.space_index,
            SPACE_BAND_LEN - 1
        )));
    }
    if e.dtype > SPACE_DTYPE_I4 {
        return Err(malformed(format!(
            "space_table: unknown dtype code {}",
            e.dtype
        )));
    }
    if e.name.is_empty() {
        return Err(malformed("space_table: empty space name"));
    }
    if !e.model_hash.starts_with("sha256:") {
        return Err(malformed("space_table: model_hash must be sha256:<hex>"));
    }
    Ok(())
}

/// encode `entries` into the 0x15 payload. deterministic: same entries in
/// the same order, same bytes. rejects duplicate indices/names.
pub fn encode_space_table(entries: &[SpaceEntry]) -> Result<Vec<u8>, UrnaError> {
    for (i, e) in entries.iter().enumerate() {
        check_entry(e)?;
        if entries[..i].iter().any(|p| p.space_index == e.space_index) {
            return Err(malformed(format!(
                "space_table: duplicate space_index {}",
                e.space_index
            )));
        }
        if entries[..i].iter().any(|p| p.name == e.name) {
            return Err(malformed(format!("space_table: duplicate name {}", e.name)));
        }
    }
    let mut out = Vec::new();
    out.extend_from_slice(&SPACE_TABLE_PAYLOAD_VERSION.to_le_bytes());
    out.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for e in entries {
        out.push(e.space_index);
        out.extend_from_slice(&(e.name.len() as u32).to_le_bytes());
        out.extend_from_slice(e.name.as_bytes());
        out.extend_from_slice(&e.dim.to_le_bytes());
        out.push(e.dtype);
        out.extend_from_slice(&(e.model_hash.len() as u32).to_le_bytes());
        out.extend_from_slice(e.model_hash.as_bytes());
        out.extend_from_slice(&e.n_vectors.to_le_bytes());
    }
    Ok(out)
}

/// decode the payload back to entries. typed errors on truncation, a
/// version mismatch, hostile counts, or an invalid entry; never panics.
pub fn decode_space_table(bytes: &[u8]) -> Result<Vec<SpaceEntry>, UrnaError> {
    let mut cur = Cursor::new(bytes);
    let version = cur.u32()?;
    if version != SPACE_TABLE_PAYLOAD_VERSION {
        return Err(UrnaError::UnsupportedSectionVersion {
            section_id: SECTION_SPACE_TABLE,
            version,
        });
    }
    let n = cur.u64()? as usize;
    if n > cur.remaining() / MIN_ENTRY_SIZE {
        return Err(malformed("space_table: entry count exceeds payload"));
    }
    let mut entries = Vec::with_capacity(n);
    for _ in 0..n {
        let space_index = cur.u8()?;
        let name = cur.utf8()?;
        let dim = cur.u32()?;
        let dtype = cur.u8()?;
        let model_hash = cur.utf8()?;
        let n_vectors = cur.u64()?;
        let e = SpaceEntry {
            space_index,
            name,
            dim,
            dtype,
            model_hash,
            n_vectors,
        };
        check_entry(&e)?;
        if entries
            .iter()
            .any(|p: &SpaceEntry| p.space_index == e.space_index || p.name == e.name)
        {
            return Err(malformed("space_table: duplicate index or name"));
        }
        entries.push(e);
    }
    if cur.pos != bytes.len() {
        return Err(malformed("trailing bytes after entries"));
    }
    Ok(entries)
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
    fn utf8(&mut self) -> Result<String, UrnaError> {
        let len = self.u32()? as usize;
        if len > self.remaining() {
            return Err(malformed("space_table: string length exceeds payload"));
        }
        let s = std::str::from_utf8(self.take(len)?)
            .map_err(|_| malformed("space_table: string is not utf-8"))?;
        Ok(s.to_string())
    }
}
