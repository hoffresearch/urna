//! fixed-size binary header (128 bytes). The header contains
//! everything a reader needs to discover the section table and the
//! manifest without reading the file body. The `header_checksum` is
//! the first 8 bytes of `SHA-256` of the header bytes with the
//! checksum field zeroed.

use crate::bytes::{array8, le_u32, le_u64};
use crate::error::UrnaError;
use sha2::{Digest, Sha256};

use super::{URNA_HEADER_SIZE, URNA_MAGIC, URNA_VERSION_MAJOR, URNA_VERSION_MINOR};

/// The header as fields. On disk it is serialized field by field, little
/// endian, at the offsets `to_bytes` names, no padding.
#[derive(Clone, Copy, Debug, PartialEq)]
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

    /// The exact on-disk bytes of this record. Offsets: `magic` 0,
    /// `version_major` 4, `version_minor` 6, `flags` 8, `embedding_dim` 12,
    /// `n_chunks` 16, `n_embeddings` 24, `file_size` 32,
    /// `section_table_offset` 40, `section_table_count` 48,
    /// `manifest_offset` 56, `manifest_size` 64, `header_checksum` 72,
    /// `reserved` 80..128.
    pub fn to_bytes(&self) -> [u8; URNA_HEADER_SIZE] {
        let mut out = [0u8; URNA_HEADER_SIZE];
        out[0..4].copy_from_slice(&self.magic);
        out[4..6].copy_from_slice(&self.version_major.to_le_bytes());
        out[6..8].copy_from_slice(&self.version_minor.to_le_bytes());
        out[8..12].copy_from_slice(&self.flags.to_le_bytes());
        out[12..16].copy_from_slice(&self.embedding_dim.to_le_bytes());
        out[16..24].copy_from_slice(&self.n_chunks.to_le_bytes());
        out[24..32].copy_from_slice(&self.n_embeddings.to_le_bytes());
        out[32..40].copy_from_slice(&self.file_size.to_le_bytes());
        out[40..48].copy_from_slice(&self.section_table_offset.to_le_bytes());
        out[48..56].copy_from_slice(&self.section_table_count.to_le_bytes());
        out[56..64].copy_from_slice(&self.manifest_offset.to_le_bytes());
        out[64..72].copy_from_slice(&self.manifest_size.to_le_bytes());
        out[72..80].copy_from_slice(&self.header_checksum);
        out[80..128].copy_from_slice(&self.reserved);
        out
    }

    /// Parse the record from the first `URNA_HEADER_SIZE` bytes of `bytes`.
    /// No check beyond the length: the reader validates magic, version and
    /// checksum on the parsed fields.
    pub fn from_bytes(bytes: &[u8]) -> crate::Result<Self> {
        let b = bytes
            .get(..URNA_HEADER_SIZE)
            .ok_or(UrnaError::UnexpectedEof)?;
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&b[0..4]);
        let mut reserved = [0u8; 48];
        reserved.copy_from_slice(&b[80..128]);
        Ok(Self {
            magic,
            version_major: u16::from_le_bytes([b[4], b[5]]),
            version_minor: u16::from_le_bytes([b[6], b[7]]),
            flags: le_u32(&b[8..12])?,
            embedding_dim: le_u32(&b[12..16])?,
            n_chunks: le_u64(&b[16..24])?,
            n_embeddings: le_u64(&b[24..32])?,
            file_size: le_u64(&b[32..40])?,
            section_table_offset: le_u64(&b[40..48])?,
            section_table_count: le_u64(&b[48..56])?,
            manifest_offset: le_u64(&b[56..64])?,
            manifest_size: le_u64(&b[64..72])?,
            header_checksum: array8(&b[72..80])?,
            reserved,
        })
    }

    fn as_bytes_without_checksum(&self) -> Vec<u8> {
        let bytes = self.to_bytes();
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
