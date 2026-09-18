//! fixed-size binary header (128 bytes). The header contains
//! everything a reader needs to discover the section table and the
//! manifest without reading the file body. The `header_checksum` is
//! the first 8 bytes of `SHA-256` of the header bytes with the
//! checksum field zeroed.

use crate::error::UrnaError;
use sha2::{Digest, Sha256};

use super::{URNA_MAGIC, URNA_VERSION_MAJOR, URNA_VERSION_MINOR};

// `Pod` + `Zeroable` are derived, not asserted: the derive fails to
// compile if the struct ever gains padding or a field that is not valid
// for every bit pattern, which is exactly the invariant the on-disk
// byte view below relies on. No `unsafe` in this file.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UrnaHeader {
    pub magic: [u8; 4],
    pub version_major: u16,
    pub version_minor: u16,
    pub flags: u32,
    pub embedding_dim: u32,
    pub n_chunks: u64,
    pub n_embeddings: u64,
    pub file_size: u64,
    pub section_table_offset: u64,
    pub section_table_count: u64,
    pub manifest_offset: u64,
    pub manifest_size: u64,
    pub header_checksum: [u8; 8],
    pub reserved: [u8; 48],
}

impl UrnaHeader {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        embedding_dim: u32,
        n_chunks: u64,
        n_embeddings: u64,
        file_size: u64,
        section_table_offset: u64,
        section_table_count: u64,
        manifest_offset: u64,
        manifest_size: u64,
    ) -> Self {
        let mut h = Self {
            magic: *URNA_MAGIC,
            version_major: URNA_VERSION_MAJOR,
            version_minor: URNA_VERSION_MINOR,
            flags: 0,
            embedding_dim,
            n_chunks,
            n_embeddings,
            file_size,
            section_table_offset,
            section_table_count,
            manifest_offset,
            manifest_size,
            header_checksum: [0; 8],
            reserved: [0; 48],
        };
        h.compute_checksum();
        h
    }

    pub fn compute_checksum(&mut self) {
        let bytes = self.as_bytes_without_checksum();
        let hash = Sha256::digest(&bytes);
        self.header_checksum.copy_from_slice(&hash[..8]);
    }

    pub fn validate_checksum(&self) -> crate::Result<()> {
        let mut tmp = *self;
        tmp.header_checksum = [0; 8];
        let bytes = tmp.as_bytes_without_checksum();
        let hash = Sha256::digest(&bytes);
        if hash[..8] != self.header_checksum[..] {
            return Err(UrnaError::InvalidHeaderChecksum);
        }
        Ok(())
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

    fn as_bytes_without_checksum(&self) -> Vec<u8> {
        let bytes = self.as_bytes();
        let mut v = bytes[..72].to_vec();
        v.extend_from_slice(&bytes[80..]);
        v
    }
}

impl Default for UrnaHeader {
    fn default() -> Self {
        Self::new(0, 0, 0, 128, 128, 0, 128, 0)
    }
}
