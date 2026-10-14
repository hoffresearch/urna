//! section table entry (32 bytes per entry, fixed layout).
//!
//! `offset` is 64-byte aligned (see `SECTION_ALIGNMENT`). `size` is the
//! length of the actual payload, NOT including the trailing padding.
//! `encoding` declares the on-disk payload format; v1 supports raw,
//! zstd, float16 and int8.

use crate::error::UrnaError;
use sha2::{Digest, Sha256};

use super::SECTION_ENCODING_RAW;

// `Pod` + `Zeroable` are derived, not asserted: the derive fails to
// compile if the struct ever gains padding or a field that is not valid
// for every bit pattern, which is exactly the invariant the on-disk
// byte view below relies on. No `unsafe` in this file.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
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

    /// The exact on-disk bytes of this record (little-endian host only,
    /// which is every supported target; see `layout::tests`).
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    /// Mutable view over the on-disk bytes; `from_bytes`-style readers copy
    /// a slice in here. Sound for any content because every field accepts
    /// every bit pattern (`Pod`).
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        bytemuck::bytes_of_mut(self)
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
