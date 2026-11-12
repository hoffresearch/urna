//! the two bit readers of the zstd format (RFC 8878).
//!
//! FSE and Huffman payloads are written forward and read backward
//! ([`BackwardBits`]): the last byte holds a 1 bit above up to 7 zero bits
//! of padding, and the useful stream starts right below that marker. table
//! descriptions (FSE normalized counts) are read forward ([`ForwardBits`]),
//! little-endian, lowest bit first.
//!
//! both readers answer zeros past the end of their data instead of failing:
//! the format itself relies on that (the Huffman weights stream ends on an
//! overflow, a Huffman code near the start of a stream is looked up with
//! zero-filled low bits), and the callers check the position afterwards to
//! tell a well-formed stream from a truncated one.

use super::{Res, corrupt};

/// mask of the low `n` bits, `n <= 64`.
#[inline]
fn mask(n: u32) -> u64 {
    if n >= 64 { u64::MAX } else { (1u64 << n) - 1 }
}

/// little-endian u64 from up to 8 bytes at `at`; bytes past the end are zero.
#[inline]
fn load_le(data: &[u8], at: usize) -> u64 {
    match data.get(at..at + 8) {
        Some(eight) => {
            let mut word = [0u8; 8];
            word.copy_from_slice(eight);
            u64::from_le_bytes(word)
        }
        None => {
            let mut word = [0u8; 8];
            if let Some(tail) = data.get(at..) {
                word[..tail.len()].copy_from_slice(tail);
            }
            u64::from_le_bytes(word)
        }
    }
}

/// a backward bitstream: the useful bits are `[0, pos)` (bit `i` is bit
/// `i % 8` of byte `i / 8`), consumed from the top down. a read of `n` bits
/// returns the next `n` bits with the first one read as the most
/// significant, which is what both FSE (`readBits`) and Huffman (`lookBits`)
/// expect. `pos` may go negative: that is an overflow, the reader then
/// yields zeros and [`BackwardBits::overflowed`] reports it.
pub(super) struct BackwardBits<'a> {
    data: &'a [u8],
    pos: i64,
}

impl std::fmt::Debug for BackwardBits<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackwardBits")
            .field("len", &self.data.len())
            .field("pos", &self.pos)
            .finish()
    }
}

impl<'a> BackwardBits<'a> {
    /// open a stream, skipping the zero padding and the 1-bit end marker of
    /// its last byte. an empty stream or a last byte of zero has no marker
    /// and is corrupt.
    pub(super) fn new(data: &'a [u8]) -> Res<Self> {
        let last = *data.last().ok_or(corrupt("empty bitstream"))?;
        if last == 0 {
            return Err(corrupt("bitstream without its end marker"));
        }
        let padding = i64::from(last.leading_zeros()) + 1;
        let pos = (data.len() as i64) * 8 - padding;
        Ok(Self { data, pos })
    }

    /// the next `n` bits (`n <= 56`) without consuming them; bits below the
    /// start of the stream read as zero.
    #[inline]
    pub(super) fn peek(&self, n: u32) -> u64 {
        if n == 0 || self.pos <= 0 {
            return 0;
        }
        let start = self.pos - i64::from(n);
        if start >= 0 {
            let at = (start / 8) as usize;
            (load_le(self.data, at) >> (start % 8)) & mask(n)
        } else {
            // fewer than `n` bits left: the ones there, then zeros.
            let have = self.pos as u32;
            (load_le(self.data, 0) & mask(have)) << (n - have)
        }
    }

    /// drop `n` bits.
    #[inline]
    pub(super) fn consume(&mut self, n: u32) {
        self.pos -= i64::from(n);
    }

    /// read and consume `n` bits (`n <= 56`).
    #[inline]
    pub(super) fn read(&mut self, n: u32) -> u64 {
        let value = self.peek(n);
        self.consume(n);
        value
    }

    /// more bits were consumed than the stream holds.
    #[inline]
    pub(super) fn overflowed(&self) -> bool {
        self.pos < 0
    }

    /// every bit was consumed, no more and no less.
    #[inline]
    pub(super) fn finished(&self) -> bool {
        self.pos == 0
    }
}

/// a forward little-endian bitstream for the FSE table descriptions: reads
/// past the end yield zeros, and [`ForwardBits::bytes_used`] tells the
/// caller how far the description really went.
pub(super) struct ForwardBits<'a> {
    data: &'a [u8],
    pos: usize,
}

impl std::fmt::Debug for ForwardBits<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ForwardBits")
            .field("len", &self.data.len())
            .field("pos", &self.pos)
            .finish()
    }
}

impl<'a> ForwardBits<'a> {
    pub(super) fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// the next `n` bits (`n <= 32`), lowest first, without consuming them.
    #[inline]
    pub(super) fn peek(&self, n: u32) -> u64 {
        let at = self.pos / 8;
        if at >= self.data.len() {
            return 0;
        }
        (load_le(self.data, at) >> (self.pos % 8)) & mask(n)
    }

    #[inline]
    pub(super) fn consume(&mut self, n: u32) {
        self.pos += n as usize;
    }

    #[inline]
    pub(super) fn read(&mut self, n: u32) -> u64 {
        let value = self.peek(n);
        self.consume(n);
        value
    }

    /// whole bytes the reader went through; a description ends on a byte
    /// boundary, its last partial byte counts.
    pub(super) fn bytes_used(&self) -> usize {
        self.pos.div_ceil(8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backward_skips_padding_and_marker() {
        // RFC 8878 section 4.2.2: "0145" with the example prefix code is
        // 00010000 00001101 (second byte last). marker in byte 1 at bit 3,
        // so 11 useful bits: 1 01 0001 0000 read from the top.
        let data = [0b0001_0000u8, 0b0000_1101];
        let mut bits = BackwardBits::new(&data).unwrap();
        assert_eq!(bits.read(1), 1);
        assert_eq!(bits.read(2), 0b01);
        assert_eq!(bits.read(4), 0b0001);
        assert_eq!(bits.read(4), 0b0000);
        assert!(bits.finished());
        assert_eq!(bits.read(3), 0);
        assert!(bits.overflowed());
    }

    #[test]
    fn backward_zero_fills_below_the_start() {
        let data = [0b0000_0011u8]; // marker at bit 1, one useful bit: 1
        let bits = BackwardBits::new(&data).unwrap();
        assert_eq!(bits.peek(4), 0b1000);
    }

    #[test]
    fn backward_rejects_missing_marker() {
        assert!(BackwardBits::new(&[]).is_err());
        assert!(BackwardBits::new(&[1, 0]).is_err());
    }

    #[test]
    fn forward_reads_lowest_first() {
        let data = [0b1010_0101u8, 0xFF];
        let mut bits = ForwardBits::new(&data);
        assert_eq!(bits.read(4), 0b0101);
        assert_eq!(bits.read(8), 0b1111_1010);
        assert_eq!(bits.bytes_used(), 2);
        assert_eq!(bits.read(8), 0b1111);
        assert_eq!(bits.read(8), 0);
    }
}
