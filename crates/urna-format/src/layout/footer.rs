//! footer (40 bytes). Sits at `file_size - 40` and carries the
//! `file_hash` (full SHA-256 over the file body) plus the footer's own
//! size for forward compat.

use crate::error::UrnaError;
use sha2::{Digest, Sha256};

use super::URNA_FOOTER_SIZE;

// `Pod` + `Zeroable` are derived, not asserted: the derive fails to
// compile if the struct ever gains padding or a field that is not valid
// for every bit pattern, which is exactly the invariant the on-disk
// byte view below relies on. No `unsafe` in this file.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
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

    pub fn compute_file_hash(data_without_footer: &[u8]) -> [u8; 32] {
        let hash = Sha256::digest(data_without_footer);
        let mut out = [0u8; 32];
        out.copy_from_slice(&hash[..]);
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> crate::Result<Self> {
        if bytes.len() < std::mem::size_of::<Self>() {
            return Err(UrnaError::UnexpectedEof);
        }
        let mut footer = UrnaFooter::new([0; 32]);
        footer.as_bytes_mut().copy_from_slice(bytes);
        Ok(footer)
    }
}
