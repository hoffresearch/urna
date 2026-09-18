//! Little-endian field readers that return a typed error instead of
//! panicking. Every on-disk decoder used to spell `u32::from_le_bytes(
//! b[..4].try_into().unwrap())`; the slice was always pre-checked, but a
//! runtime that opens third-party files should not carry an `unwrap` on
//! its parse path at all (`clippy::unwrap_used` is denied workspace-wide).
//! A wrong length here is a decoder bug surfaced as `UnexpectedEof`.

use crate::error::UrnaError;

/// Read a `u32` from exactly 4 little-endian bytes.
#[inline]
pub fn le_u32(b: &[u8]) -> crate::Result<u32> {
    <[u8; 4]>::try_from(b)
        .map(u32::from_le_bytes)
        .map_err(|_| UrnaError::UnexpectedEof)
}

/// Read a `u64` from exactly 8 little-endian bytes.
#[inline]
pub fn le_u64(b: &[u8]) -> crate::Result<u64> {
    <[u8; 8]>::try_from(b)
        .map(u64::from_le_bytes)
        .map_err(|_| UrnaError::UnexpectedEof)
}

/// Read an `f32` from exactly 4 little-endian bytes (bit pattern only; the
/// caller decides whether NaN/Inf are acceptable).
#[inline]
pub fn le_f32(b: &[u8]) -> crate::Result<f32> {
    <[u8; 4]>::try_from(b)
        .map(f32::from_le_bytes)
        .map_err(|_| UrnaError::UnexpectedEof)
}

/// Copy exactly 32 bytes (a sha256 digest) into an array.
#[inline]
pub fn array32(b: &[u8]) -> crate::Result<[u8; 32]> {
    <[u8; 32]>::try_from(b).map_err(|_| UrnaError::UnexpectedEof)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_lengths_decode_and_wrong_lengths_are_typed_errors() {
        assert_eq!(le_u32(&[1, 0, 0, 0]).unwrap(), 1);
        assert_eq!(le_u64(&[2, 0, 0, 0, 0, 0, 0, 0]).unwrap(), 2);
        assert_eq!(le_f32(&1.5f32.to_le_bytes()).unwrap(), 1.5);
        assert_eq!(array32(&[7u8; 32]).unwrap(), [7u8; 32]);
        assert!(matches!(le_u32(&[1, 0, 0]), Err(UrnaError::UnexpectedEof)));
        assert!(matches!(le_u64(&[0; 9]), Err(UrnaError::UnexpectedEof)));
        assert!(matches!(array32(&[0; 31]), Err(UrnaError::UnexpectedEof)));
    }
}
