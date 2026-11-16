//! section table entry (32 bytes per entry, fixed layout).
//!
//! `offset` is 64-byte aligned (see `SECTION_ALIGNMENT`). `size` is the
//! length of the actual payload, NOT including the trailing padding.
//! `encoding` declares the on-disk payload format; v1 supports raw,
//! zstd, float16 and int8.

use crate::bytes::{array8, le_u32, le_u64};
use crate::error::UrnaError;
use crate::sha256::Sha256;

use super::{SECTION_ENCODING_RAW, URNA_SECTION_ENTRY_SIZE};

/// One section table entry as fields. On disk it is serialized field by
/// field, little endian, at the offsets `to_bytes` names, no padding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SectionEntry {
    pub section_id: u32,
    pub encoding: u32,
    pub offset: u64,
    pub size: u64,
    pub checksum: [u8; 8],
}

impl SectionEntry {
    pub fn new(section_id: u32, offset: u64, size: u64) -> Self {
        Self {
            section_id,
            encoding: SECTION_ENCODING_RAW,
            offset,
            size,
            checksum: [0; 8],
        }
    }

    /// The exact on-disk bytes of this record: `section_id` at 0,
    /// `encoding` at 4, `offset` at 8, `size` at 16, `checksum` at 24.
    pub fn to_bytes(&self) -> [u8; URNA_SECTION_ENTRY_SIZE] {
        let mut out = [0u8; URNA_SECTION_ENTRY_SIZE];
        out[0..4].copy_from_slice(&self.section_id.to_le_bytes());
        out[4..8].copy_from_slice(&self.encoding.to_le_bytes());
        out[8..16].copy_from_slice(&self.offset.to_le_bytes());
        out[16..24].copy_from_slice(&self.size.to_le_bytes());
        out[24..32].copy_from_slice(&self.checksum);
        out
    }

    /// Parse the record from the first `URNA_SECTION_ENTRY_SIZE` bytes of
    /// `bytes`.
    pub fn from_bytes(bytes: &[u8]) -> crate::Result<Self> {
        let b = bytes
            .get(..URNA_SECTION_ENTRY_SIZE)
            .ok_or(UrnaError::UnexpectedEof)?;
        Ok(Self {
            section_id: le_u32(&b[0..4])?,
            encoding: le_u32(&b[4..8])?,
            offset: le_u64(&b[8..16])?,
            size: le_u64(&b[16..24])?,
            checksum: array8(&b[24..32])?,
        })
    }

    pub fn compute_checksum(&mut self, data: &[u8]) {
        let hash = Sha256::digest(data);
        self.checksum.copy_from_slice(&hash[..8]);
    }

    pub fn validate_checksum(&self, data: &[u8]) -> crate::Result<()> {
        let hash = Sha256::digest(data);
        if hash[..8] != self.checksum[..] {
            return Err(UrnaError::SectionChecksumMismatch(self.section_id));
        }
        Ok(())
    }
}
