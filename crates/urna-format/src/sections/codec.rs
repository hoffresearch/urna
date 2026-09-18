//! Shared encoding/decoding primitives used by every section payload.
//! Cursor for length-checked reads, prefix writer/reader, length-
//! prefixed UTF-8 strings.

use crate::error::UrnaError;
use crate::layout::{SECTION_PAYLOAD_PREFIX_SIZE, SECTION_PAYLOAD_VERSION};

pub(super) fn write_prefix(buf: &mut Vec<u8>, count: u64) {
    buf.extend_from_slice(&SECTION_PAYLOAD_VERSION.to_le_bytes());
    buf.extend_from_slice(&count.to_le_bytes());
}

pub(super) fn write_lp_str(buf: &mut Vec<u8>, s: &str) -> crate::Result<()> {
    let bytes = s.as_bytes();
    let len = u32::try_from(bytes.len())
        .map_err(|_| UrnaError::InvalidInput(format!("string too long: {} bytes", bytes.len())))?;
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(bytes);
    Ok(())
}

pub(super) struct Cursor<'a> {
    pub data: &'a [u8],
    pub pos: usize,
    pub section_id: u32,
}

impl<'a> Cursor<'a> {
    pub fn new(data: &'a [u8], section_id: u32) -> Self {
        Self {
            data,
            pos: 0,
            section_id,
        }
    }

    pub fn malformed(&self, reason: impl Into<String>) -> UrnaError {
        UrnaError::MalformedSectionPayload {
            section_id: self.section_id,
            reason: reason.into(),
        }
    }

    /// Bounds check written as `n > remaining` on purpose: `pos + n` with an
    /// attacker-chosen `n` (a u64 count cast to usize) wraps in release and
    /// passes a `pos + n > len` check, then the slice panics (found by the
    /// mutation fuzz harness). `pos <= len` is this cursor's invariant.
    pub fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], UrnaError> {
        if n > self.data.len() - self.pos {
            return Err(self.malformed(format!(
                "want {} bytes at offset {}, have {}",
                n,
                self.pos,
                self.data.len()
            )));
        }
        let out = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    pub fn read_u32(&mut self) -> Result<u32, UrnaError> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn read_u64(&mut self) -> Result<u64, UrnaError> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    pub fn read_lp_str(&mut self) -> Result<String, UrnaError> {
        let len = self.read_u32()? as usize;
        let bytes = self.read_bytes(len)?;
        std::str::from_utf8(bytes)
            .map(|s| s.to_string())
            .map_err(|e| self.malformed(format!("invalid utf-8: {}", e)))
    }

    pub fn finish(self) -> Result<(), UrnaError> {
        if self.pos != self.data.len() {
            return Err(self.malformed(format!(
                "trailing bytes: consumed {} of {}",
                self.pos,
                self.data.len()
            )));
        }
        Ok(())
    }
}

pub(super) fn read_prefix(c: &mut Cursor) -> Result<u64, UrnaError> {
    if c.data.len() < SECTION_PAYLOAD_PREFIX_SIZE {
        return Err(c.malformed(format!(
            "payload shorter than {} byte prefix",
            SECTION_PAYLOAD_PREFIX_SIZE
        )));
    }
    let version = c.read_u32()?;
    if version != SECTION_PAYLOAD_VERSION {
        return Err(UrnaError::UnsupportedSectionVersion {
            section_id: c.section_id,
            version,
        });
    }
    c.read_u64()
}
