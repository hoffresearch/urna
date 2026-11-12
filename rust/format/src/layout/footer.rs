//! footer (40 bytes). Sits at `file_size - 40` and carries the
//! `file_hash` (full SHA-256 over the file body) plus the footer's own
//! size for forward compat.

use crate::bytes::{array32, le_u64};
use crate::error::UrnaError;
use crate::sha256::Sha256;

use super::URNA_FOOTER_SIZE;

/// The footer as fields. On disk it is serialized field by field, little
/// endian, at the offsets `to_bytes` names: `footer_size` at 0 and
/// `file_hash` at 8, no padding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UrnaFooter {
    pub footer_size: u64,
    pub file_hash: [u8; 32],
}

impl UrnaFooter {
    pub fn new(file_hash: [u8; 32]) -> Self {
        Self {
            footer_size: URNA_FOOTER_SIZE as u64,
            file_hash,
        }
    }

    /// The exact on-disk bytes of this record.
    pub fn to_bytes(&self) -> [u8; URNA_FOOTER_SIZE] {
        let mut out = [0u8; URNA_FOOTER_SIZE];
        out[0..8].copy_from_slice(&self.footer_size.to_le_bytes());
        out[8..40].copy_from_slice(&self.file_hash);
        out
    }

    pub fn compute_file_hash(data_without_footer: &[u8]) -> [u8; 32] {
        let hash = Sha256::digest(data_without_footer);
        let mut out = [0u8; 32];
        out.copy_from_slice(&hash[..]);
        out
    }

    /// Parse the record from the first `URNA_FOOTER_SIZE` bytes of `bytes`.
    pub fn from_bytes(bytes: &[u8]) -> crate::Result<Self> {
        let b = bytes
            .get(..URNA_FOOTER_SIZE)
            .ok_or(UrnaError::UnexpectedEof)?;
        Ok(Self {
            footer_size: le_u64(&b[0..8])?,
            file_hash: array32(&b[8..40])?,
        })
    }
}
