//! shared integer bitpacking primitive (the `intpack` wire codec family,
//! encoding id 4). values are stored frame-of-reference per 128-block:
//! each block records its `min` and a bit-width, then packs `value - min`
//! at that width, lsb-first. a block directory of absolute byte offsets
//! gives O(1) block lookup, so `IntpackReader::get` reaches any element
//! without scanning the whole payload.
//!
//! the primitive is order-preserving: it never sorts, so hnsw neighbour
//! lists keep their exact build order (zero recall change). callers that
//! want delta coding pass pre-differenced, monotonic sequences (e.g.
//! sorted bm25 doc-id gaps); the small gaps shrink the per-block width.
//!
//! every read is bounds-checked and returns a typed `UrnaError` on
//! truncated or malformed input, never a panic on a hostile mmap. one
//! payload holds at most a few GiB of packed body (the block directory uses
//! u32 byte offsets); that is far above any real index or metadata column.

use crate::bytes::{le_u32, le_u64};
use crate::error::UrnaError;

/// values per frame-of-reference block.
pub const INTPACK_BLOCK: usize = 128;
const HEADER: usize = 8; // u32 count + u32 n_blocks

/// bits needed to store the values `0..=range` (0 when `range == 0`).
#[inline]
fn bit_width(range: u64) -> u8 {
    (64 - range.leading_zeros()) as u8
}

#[inline]
fn mask(width: u8) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

fn malformed(reason: impl Into<String>) -> UrnaError {
    UrnaError::MalformedSectionPayload {
        section_id: 0,
        reason: reason.into(),
    }
}

/// pack a slice of u64 with per-128-block frame-of-reference bitpacking.
/// the inverse is [`unpack_u64s`] (and random access via [`IntpackReader`]).
pub fn pack_u64s(values: &[u64]) -> Vec<u8> {
    let count = values.len();
    let n_blocks = count.div_ceil(INTPACK_BLOCK);
    let blocks_start = HEADER + n_blocks * 4;
    let mut dir: Vec<u32> = Vec::with_capacity(n_blocks);
    let mut blocks: Vec<u8> = Vec::new();
    for chunk in values.chunks(INTPACK_BLOCK) {
        dir.push((blocks_start + blocks.len()) as u32);
        // `chunks()` never yields an empty slice, so the fold sees >= 1 value.
        let (min, max) = chunk
            .iter()
            .fold((u64::MAX, 0u64), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        let width = bit_width(max - min);
        blocks.extend_from_slice(&min.to_le_bytes());
        blocks.push(width);
        pack_block(&mut blocks, chunk, min, width);
    }
    let mut out = Vec::with_capacity(blocks_start + blocks.len());
    out.extend_from_slice(&(count as u32).to_le_bytes());
    out.extend_from_slice(&(n_blocks as u32).to_le_bytes());
    for off in &dir {
        out.extend_from_slice(&off.to_le_bytes());
    }
    out.extend_from_slice(&blocks);
    out
}

fn pack_block(out: &mut Vec<u8>, chunk: &[u64], min: u64, width: u8) {
    if width == 0 {
        return;
    }
    let mut acc: u128 = 0;
    let mut nbits: u32 = 0;
    for &v in chunk {
        acc |= ((v - min) as u128 & mask(width) as u128) << nbits;
        nbits += width as u32;
        while nbits >= 8 {
            out.push((acc & 0xff) as u8);
            acc >>= 8;
            nbits -= 8;
        }
    }
    if nbits > 0 {
        out.push((acc & 0xff) as u8);
    }
}

#[inline]
fn read_u32(bytes: &[u8], pos: usize) -> Result<u32, UrnaError> {
    let end = pos + 4;
    if end > bytes.len() {
        return Err(malformed("intpack: truncated u32"));
    }
    le_u32(&bytes[pos..end])
}

#[inline]
fn read_u64(bytes: &[u8], pos: usize) -> Result<u64, UrnaError> {
    let end = pos + 8;
    if end > bytes.len() {
        return Err(malformed("intpack: truncated u64"));
    }
    le_u64(&bytes[pos..end])
}

/// number of packed value-bytes a block of `block_len` values at `width`
/// occupies (after the 9-byte `min`+`width` block header).
#[inline]
fn block_body_len(block_len: usize, width: u8) -> usize {
    (block_len * width as usize).div_ceil(8)
}

/// extract value `idx` (0-based within the block) from a block body of
/// `width`-bit lsb-first packed values. `body` must hold the block.
fn extract(body: &[u8], idx: usize, width: u8) -> u64 {
    if width == 0 {
        return 0;
    }
    let bit = idx * width as usize;
    let mut acc: u128 = 0;
    let first = bit / 8;
    let last = (bit + width as usize - 1) / 8;
    for (k, &b) in body[first..=last].iter().enumerate() {
        acc |= (b as u128) << (k * 8);
    }
    ((acc >> (bit % 8)) as u64) & mask(width)
}

/// decode an entire intpack payload back to its `u64` values.
pub fn unpack_u64s(bytes: &[u8]) -> Result<Vec<u64>, UrnaError> {
    let reader = IntpackReader::parse(bytes)?;
    // cap the up-front reservation: a hostile header can claim a huge count
    // (the per-block bodies still get bounds-checked as we go), so grow on
    // demand rather than reserve gigabytes from a claim alone.
    let mut out = Vec::with_capacity(reader.len().min(1 << 20));
    for b in 0..reader.n_blocks {
        let (min, width, body, block_len) = reader.block(b)?;
        for i in 0..block_len {
            // wrapping: for valid data min + (v - min) == v exactly; on a
            // hostile payload (tampered min/width) this must not panic.
            out.push(min.wrapping_add(extract(body, i, width)));
        }
    }
    Ok(out)
}

/// random-access reader over an intpack payload. `parse` validates the
/// header and directory once; `get` then reaches any element in O(1)
/// block lookups, decoding only the touched block.
pub struct IntpackReader<'a> {
    bytes: &'a [u8],
    count: usize,
    n_blocks: usize,
}

impl<'a> IntpackReader<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, UrnaError> {
        let count = read_u32(bytes, 0)? as usize;
        let n_blocks = read_u32(bytes, 4)? as usize;
        if n_blocks != count.div_ceil(INTPACK_BLOCK) {
            return Err(malformed("intpack: block count inconsistent with count"));
        }
        // directory must be fully present.
        if HEADER + n_blocks * 4 > bytes.len() {
            return Err(malformed("intpack: truncated directory"));
        }
        Ok(Self {
            bytes,
            count,
            n_blocks,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// `(min, width, body, block_len)` for block `b`, all bounds-checked.
    fn block(&self, b: usize) -> Result<(u64, u8, &'a [u8], usize), UrnaError> {
        let off = read_u32(self.bytes, HEADER + b * 4)? as usize;
        let min = read_u64(self.bytes, off)?;
        let width_pos = off + 8;
        if width_pos >= self.bytes.len() {
            return Err(malformed("intpack: truncated block header"));
        }
        let width = self.bytes[width_pos];
        if width > 64 {
            return Err(malformed("intpack: block width out of range"));
        }
        let block_len = (self.count - b * INTPACK_BLOCK).min(INTPACK_BLOCK);
        let body_start = width_pos + 1;
        let body_end = body_start + block_body_len(block_len, width);
        if body_end > self.bytes.len() {
            return Err(malformed("intpack: truncated block body"));
        }
        Ok((min, width, &self.bytes[body_start..body_end], block_len))
    }

    /// value at index `i`, or a typed error if `i` is out of range or the
    /// payload is truncated. never panics.
    pub fn get(&self, i: usize) -> Result<u64, UrnaError> {
        if i >= self.count {
            return Err(malformed("intpack: index out of range"));
        }
        let (min, width, body, _) = self.block(i / INTPACK_BLOCK)?;
        Ok(min.wrapping_add(extract(body, i % INTPACK_BLOCK, width)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(values: &[u64]) {
        let packed = pack_u64s(values);
        let back = unpack_u64s(&packed).unwrap();
        assert_eq!(back, values, "unpack_u64s mismatch");
        let reader = IntpackReader::parse(&packed).unwrap();
        assert_eq!(reader.len(), values.len());
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(reader.get(i).unwrap(), v, "get({}) mismatch", i);
        }
        assert!(reader.get(values.len()).is_err(), "oob index must error");
    }

    #[test]
    fn empty_roundtrips() {
        roundtrip(&[]);
    }

    #[test]
    fn single_block_widths() {
        roundtrip(&[0]);
        roundtrip(&[7, 7, 7, 7]); // width 0 (all equal min)
        roundtrip(&[0, 1, 2, 3, 4, 5]);
        roundtrip(&[5, 1, 9, 2, 30724, 0]); // unsorted, order preserved
    }

    #[test]
    fn crosses_block_boundary() {
        let v: Vec<u64> = (0..300).map(|i| (i * 7) % 31).collect();
        roundtrip(&v);
    }

    #[test]
    fn wide_values_and_for_offset() {
        roundtrip(&[1_000_000, 1_000_001, 1_000_005, 1_000_002]);
        roundtrip(&[u64::MAX, 0, u64::MAX / 2]);
    }

    #[test]
    fn tampered_min_does_not_panic() {
        // hostile payload: a block min near u64::MAX with nonzero packed
        // offsets. decode must wrap, not overflow-panic in debug builds.
        let mut packed = pack_u64s(&[1, 2, 3, 200]);
        // blocks_start = HEADER(8) + n_blocks(1)*4 = 12; min is bytes 12..20.
        for byte in packed.iter_mut().skip(12).take(8) {
            *byte = 0xFF;
        }
        let _ = unpack_u64s(&packed); // must not panic
        if let Ok(r) = IntpackReader::parse(&packed) {
            for i in 0..r.len() {
                let _ = r.get(i);
            }
        }
    }

    #[test]
    fn truncated_inputs_error_never_panic() {
        let packed = pack_u64s(&[1, 2, 3, 4, 5]);
        for cut in 0..packed.len() {
            let _ = unpack_u64s(&packed[..cut]);
            if let Ok(r) = IntpackReader::parse(&packed[..cut]) {
                let _ = r.get(0);
                let _ = r.get(r.len().saturating_sub(1));
            }
        }
        // a header claiming more blocks than bytes back must error, not panic.
        let mut evil = Vec::new();
        evil.extend_from_slice(&1_000_000u32.to_le_bytes());
        evil.extend_from_slice(&7813u32.to_le_bytes());
        assert!(IntpackReader::parse(&evil).is_err());
    }
}
