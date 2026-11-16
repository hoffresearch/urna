//! the literals section of a compressed block, RFC 8878 section 3.1.1.3.1:
//! raw, RLE, Huffman-compressed with a new tree, or treeless (the tree of
//! the previous compressed literals, or of the dictionary).

use super::huffman::{self, HuffTable};
use super::{Res, corrupt};

/// where the Huffman tree for a treeless literals block comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum TreeSource {
    /// no tree yet in this frame: a treeless block is corrupt.
    #[default]
    Missing,
    /// the dictionary's tree.
    Dictionary,
    /// the last tree a block of this frame described.
    Frame,
}

/// the trees a frame can use: the one its blocks last described and the
/// one the dictionary carries.
#[derive(Debug)]
pub(super) struct Trees<'t> {
    pub(super) source: TreeSource,
    pub(super) frame: &'t mut HuffTable,
    pub(super) dictionary: Option<&'t HuffTable>,
}

/// decode the literals section at the start of `block` into `lits`;
/// returns the bytes the section took. `block_max` bounds the literal
/// count, as it bounds the block.
pub(super) fn decode(
    block: &[u8],
    block_max: usize,
    trees: &mut Trees<'_>,
    lits: &mut Vec<u8>,
) -> Res<usize> {
    let b0 = *block.first().ok_or(corrupt("empty literals section"))?;
    let kind = b0 & 3;
    let format = (b0 >> 2) & 3;
    if kind < 2 {
        let (header, size) = match format {
            1 => (2, usize::from(le16(block)?) >> 4),
            3 => (3, le24(block)? >> 4),
            _ => (1, usize::from(b0 >> 3)),
        };
        if size > block_max {
            return Err(corrupt("literals larger than the block"));
        }
        lits.clear();
        if kind == 0 {
            let raw = block
                .get(header..header + size)
                .ok_or(corrupt("raw literals truncated"))?;
            lits.extend_from_slice(raw);
            return Ok(header + size);
        }
        let byte = *block.get(header).ok_or(corrupt("rle literals truncated"))?;
        lits.resize(size, byte);
        return Ok(header + 1);
    }
    if block.len() < 5 {
        return Err(corrupt("compressed literals header truncated"));
    }
    let word = u32::from_le_bytes([block[0], block[1], block[2], block[3]]) as usize;
    let (header, size, compressed, four) = match format {
        0 | 1 => (3, (word >> 4) & 0x3FF, (word >> 14) & 0x3FF, format == 1),
        2 => (4, (word >> 4) & 0x3FFF, word >> 18, true),
        _ => (
            5,
            (word >> 4) & 0x3FFFF,
            (word >> 22) + (usize::from(block[4]) << 10),
            true,
        ),
    };
    if size > block_max {
        return Err(corrupt("literals larger than the block"));
    }
    if four && size < 6 {
        return Err(corrupt("too few literals for four streams"));
    }
    let body = block
        .get(header..header + compressed)
        .ok_or(corrupt("compressed literals truncated"))?;
    let streams = if kind == 2 {
        let used = huffman::read_table(body, trees.frame)?;
        trees.source = TreeSource::Frame;
        body.get(used..)
            .ok_or(corrupt("huffman tree past its literals"))?
    } else {
        body
    };
    let table: &HuffTable = match trees.source {
        TreeSource::Missing => return Err(corrupt("treeless literals without a tree")),
        TreeSource::Frame => &*trees.frame,
        TreeSource::Dictionary => trees
            .dictionary
            .ok_or(corrupt("treeless literals without a tree"))?,
    };
    lits.clear();
    lits.resize(size, 0);
    huffman::decode_literals(table, streams, four, lits)?;
    Ok(header + compressed)
}

fn le16(b: &[u8]) -> Res<u16> {
    match b {
        [a, c, ..] => Ok(u16::from_le_bytes([*a, *c])),
        _ => Err(corrupt("literals header truncated")),
    }
}

fn le24(b: &[u8]) -> Res<usize> {
    match b {
        [a, c, d, ..] => Ok(usize::from(*a) | (usize::from(*c) << 8) | (usize::from(*d) << 16)),
        _ => Err(corrupt("literals header truncated")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(block: &[u8], lits: &mut Vec<u8>) -> Res<usize> {
        let mut frame = HuffTable::default();
        let mut trees = Trees {
            source: TreeSource::Missing,
            frame: &mut frame,
            dictionary: None,
        };
        decode(block, 128 * 1024, &mut trees, lits)
    }

    #[test]
    fn raw_and_rle_headers() {
        let mut lits = Vec::new();
        // raw, 1-byte header: size 3.
        assert_eq!(run(&[3 << 3, b'a', b'b', b'c', 0], &mut lits).unwrap(), 4);
        assert_eq!(lits, b"abc");
        // rle, 2-byte header (format 1): size 0x123.
        let h = (0x123u16 << 4) | (1 << 2) | 1;
        let [lo, hi] = h.to_le_bytes();
        assert_eq!(run(&[lo, hi, b'z', 0], &mut lits).unwrap(), 3);
        assert_eq!(lits.len(), 0x123);
        assert!(lits.iter().all(|&b| b == b'z'));
    }

    #[test]
    fn rejects_truncated_and_treeless() {
        let mut lits = Vec::new();
        assert!(run(&[], &mut lits).is_err());
        assert!(run(&[5 << 3, b'a'], &mut lits).is_err()); // raw truncated
        assert!(run(&[1], &mut lits).is_err()); // rle without its byte
        // treeless (3) with no tree yet.
        assert!(run(&[3, 0x10, 0, 0, 0, 0], &mut lits).is_err());
    }
}
