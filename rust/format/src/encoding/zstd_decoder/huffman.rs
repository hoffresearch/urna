//! Huffman literals, RFC 8878 section 4.2: the tree description (weights
//! FSE-compressed or written directly), the canonical prefix codes it
//! implies, and the decoding of one or four backward streams.

use super::bits::BackwardBits;
use super::fse::decode_weights;
use super::{Res, corrupt};

/// the deepest code libzstd decodes (HUF_TABLELOG_MAX); its encoder writes
/// at most 11.
const MAX_LOG: u32 = 12;

/// a decoding table of `1 << log` entries: peek `log` bits, the entry gives
/// the symbol and how many of those bits its code really uses.
#[derive(Debug, Clone, Default)]
pub(super) struct HuffTable {
    log: u32,
    entries: Vec<(u8, u8)>,
}

/// read a tree description from the start of `src` into `table`; returns
/// the bytes it took.
pub(super) fn read_table(src: &[u8], table: &mut HuffTable) -> Res<usize> {
    let (&header, rest) = src.split_first().ok_or(corrupt("empty huffman tree"))?;
    let mut weights: Vec<u8> = Vec::with_capacity(256);
    let used = if header >= 128 {
        // direct: 4 bits per weight, the first in the high nibble.
        let n = usize::from(header) - 127;
        let packed = rest
            .get(..n.div_ceil(2))
            .ok_or(corrupt("huffman weights truncated"))?;
        for i in 0..n {
            let byte = packed[i / 2];
            weights.push(if i & 1 == 0 { byte >> 4 } else { byte & 15 });
        }
        1 + n.div_ceil(2)
    } else {
        let size = usize::from(header);
        let compressed = rest
            .get(..size)
            .ok_or(corrupt("huffman weights truncated"))?;
        decode_weights(compressed, &mut weights)?;
        1 + size
    };
    build(&mut weights, table)?;
    Ok(used)
}

/// complete the weights with the implied last one and lay the canonical
/// codes out: by weight, then by symbol, the lightest at index 0.
fn build(weights: &mut Vec<u8>, table: &mut HuffTable) -> Res<()> {
    let mut total: u32 = 0;
    for &w in weights.iter() {
        if u32::from(w) > MAX_LOG {
            return Err(corrupt("huffman weight too large"));
        }
        if w > 0 {
            total += 1 << (w - 1);
        }
    }
    if total == 0 {
        return Err(corrupt("huffman tree without symbols"));
    }
    let log = total.ilog2() + 1;
    if log > MAX_LOG {
        return Err(corrupt("huffman tree too deep"));
    }
    let rest = (1u32 << log) - total;
    if !rest.is_power_of_two() {
        return Err(corrupt("huffman weights do not complete a tree"));
    }
    weights.push((rest.ilog2() + 1) as u8);
    if weights.len() > 256 {
        return Err(corrupt("huffman tree has too many symbols"));
    }
    let ones = weights.iter().filter(|&&w| w == 1).count();
    if ones < 2 || ones & 1 == 1 {
        return Err(corrupt("huffman tree with an odd number of deepest codes"));
    }
    table.log = log;
    table.entries.clear();
    table.entries.reserve(1 << log);
    for w in 1..=log {
        let nb_bits = (log + 1 - w) as u8;
        let span = 1usize << (w - 1);
        for (symbol, _) in weights
            .iter()
            .enumerate()
            .filter(|&(_, &x)| u32::from(x) == w)
        {
            let entry = (symbol as u8, nb_bits);
            table.entries.resize(table.entries.len() + span, entry);
        }
    }
    if table.entries.len() != 1 << log {
        return Err(corrupt("huffman codes do not fill the table"));
    }
    Ok(())
}

/// decode one stream into `out`, which must be consumed exactly.
fn decode_stream(table: &HuffTable, src: &[u8], out: &mut [u8]) -> Res<()> {
    let mut bits = BackwardBits::new(src)?;
    let mask = table.entries.len() - 1;
    for slot in out.iter_mut() {
        let (symbol, nb_bits) = table.entries[bits.peek(table.log) as usize & mask];
        *slot = symbol;
        bits.consume(u32::from(nb_bits));
    }
    if !bits.finished() {
        return Err(corrupt("huffman stream not consumed exactly"));
    }
    Ok(())
}

/// decode `out.len()` literals from one stream, or from four behind a
/// 6-byte jump table, each covering a quarter (the last one the rest).
pub(super) fn decode_literals(
    table: &HuffTable,
    src: &[u8],
    four: bool,
    out: &mut [u8],
) -> Res<()> {
    if table.entries.is_empty() {
        return Err(corrupt("huffman table missing"));
    }
    if !four {
        return decode_stream(table, src, out);
    }
    if src.len() < 10 {
        return Err(corrupt("huffman jump table truncated"));
    }
    let size = |i: usize| usize::from(u16::from_le_bytes([src[i], src[i + 1]]));
    let (l1, l2, l3) = (size(0), size(2), size(4));
    let l4 = (src.len() - 6)
        .checked_sub(l1 + l2 + l3)
        .ok_or(corrupt("huffman jump table exceeds the streams"))?;
    let streams = &src[6..];
    let (s1, streams) = streams.split_at(l1);
    let (s2, streams) = streams.split_at(l2);
    let (s3, s4) = streams.split_at(l3);
    debug_assert_eq!(s4.len(), l4);
    let segment = out.len().div_ceil(4);
    if 3 * segment > out.len() {
        return Err(corrupt("too few literals for four streams"));
    }
    let (o1, rest) = out.split_at_mut(segment);
    let (o2, rest) = rest.split_at_mut(segment);
    let (o3, o4) = rest.split_at_mut(segment);
    decode_stream(table, s1, o1)?;
    decode_stream(table, s2, o2)?;
    decode_stream(table, s3, o3)?;
    decode_stream(table, s4, o4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc_prefix_codes() {
        // RFC 8878 tables 23-25: weights 4,3,2,0,1 for literals 0..4, the
        // weight of 5 implied (1). direct header: 128 + 5 - 1 = 132.
        let mut table = HuffTable::default();
        let used = read_table(&[132, 0x43, 0x20, 0x10], &mut table).unwrap();
        assert_eq!(used, 4);
        assert_eq!(table.log, 4);
        // codes 0000 -> 4, 0001 -> 5, 001x -> 2, 01xx -> 1, 1xxx -> 0.
        assert_eq!(table.entries[0], (4, 4));
        assert_eq!(table.entries[1], (5, 4));
        assert_eq!(table.entries[2], (2, 3));
        assert_eq!(table.entries[4], (1, 2));
        assert_eq!(table.entries[8], (0, 1));
        // "0145" with those codes, read from the top: 1 01 0000 0001, then
        // the end marker (the RFC's table 26 swaps the codes of 4 and 5).
        let mut out = [0u8; 4];
        decode_literals(&table, &[0b0000_0001, 0b0000_1101], false, &mut out).unwrap();
        assert_eq!(out, [0, 1, 4, 5]);
    }

    #[test]
    fn rejects_bad_trees() {
        let mut table = HuffTable::default();
        assert!(read_table(&[], &mut table).is_err());
        assert!(read_table(&[130, 0x00, 0x00], &mut table).is_err()); // all zero
        assert!(read_table(&[131, 0x11], &mut table).is_err()); // truncated
        assert!(read_table(&[5, 1, 2], &mut table).is_err()); // fse truncated
    }

    #[test]
    fn stream_must_be_consumed_exactly() {
        let mut table = HuffTable::default();
        read_table(&[132, 0x43, 0x20, 0x10], &mut table).unwrap();
        let mut out = [0u8; 3];
        assert!(decode_literals(&table, &[0b0000_0001, 0b0000_1101], false, &mut out).is_err());
    }
}
