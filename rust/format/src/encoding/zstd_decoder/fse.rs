//! FSE (finite state entropy), RFC 8878 section 4.1: the table description
//! (normalized counts read forward), the decoding table it spreads into, and
//! the two-state decoder of the Huffman weights (section 4.2.1.2).

use super::bits::{BackwardBits, ForwardBits};
use super::{Res, corrupt};

/// the largest accuracy log a description may announce (libzstd's
/// FSE_TABLELOG_ABSOLUTE_MAX); each user caps it lower.
const ABSOLUTE_MAX_LOG: u32 = 15;

/// the normalized distribution of one FSE table: a count per symbol, `-1`
/// for "less than 1", summing to `1 << log` (a `-1` counts as one).
#[derive(Debug, Clone)]
pub(super) struct Distribution {
    pub(super) counts: Vec<i16>,
    pub(super) log: u32,
}

/// one cell of a decoding table: the symbol this state decodes and how to
/// reach the next state (`base` plus `nb_bits` read from the stream).
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Cell {
    pub(super) symbol: u8,
    pub(super) nb_bits: u8,
    pub(super) base: u16,
}

/// read a table description of at most `max_symbol + 1` symbols and an
/// accuracy log of at most `max_log`; returns the distribution and the
/// bytes the description took.
pub(super) fn read_distribution(
    src: &[u8],
    max_symbol: usize,
    max_log: u32,
) -> Res<(Distribution, usize)> {
    let mut bits = ForwardBits::new(src);
    let log = bits.read(4) as u32 + 5;
    if log > ABSOLUTE_MAX_LOG || log > max_log {
        return Err(corrupt("fse accuracy log too large"));
    }
    let mut counts: Vec<i16> = Vec::with_capacity(max_symbol + 1);
    // probability points left plus one, as libzstd counts them.
    let mut remaining: i32 = (1 << log) + 1;
    let mut threshold: i32 = 1 << log;
    let mut nb_bits: u32 = log + 1;
    let mut previous_zero = false;
    loop {
        if previous_zero {
            // a zero count is followed by 2-bit repeat flags of more zeros.
            loop {
                let repeat = bits.read(2) as usize;
                counts.resize(counts.len() + repeat, 0);
                if repeat != 3 {
                    break;
                }
                if counts.len() > max_symbol + 1 {
                    return Err(corrupt("fse description has too many symbols"));
                }
            }
            if counts.len() >= max_symbol + 1 {
                return Err(corrupt("fse description has too many symbols"));
            }
        }
        let max = (2 * threshold - 1) - remaining;
        let low = bits.peek(nb_bits - 1) as i32;
        let mut count = if low < max {
            bits.consume(nb_bits - 1);
            low
        } else {
            let mut value = bits.peek(nb_bits) as i32;
            if value >= threshold {
                value -= max;
            }
            bits.consume(nb_bits);
            value
        };
        count -= 1;
        remaining -= count.abs();
        counts.push(count as i16);
        previous_zero = count == 0;
        if remaining < threshold {
            if remaining <= 1 {
                break;
            }
            nb_bits = remaining.ilog2() + 1;
            threshold = 1 << (nb_bits - 1);
        }
        if counts.len() >= max_symbol + 1 {
            break;
        }
    }
    if remaining != 1 {
        return Err(corrupt("fse counts do not sum to the table size"));
    }
    let used = bits.bytes_used();
    if used > src.len() {
        return Err(corrupt("fse description runs past its input"));
    }
    Ok((Distribution { counts, log }, used))
}

/// spread a distribution into its decoding table (section 4.1.1): the
/// "less than 1" symbols take one cell each from the end, the others are
/// spread with the fixed step, and each state learns its next-state base.
pub(super) fn build_table(dist: &Distribution, cells: &mut Vec<Cell>) -> Res<()> {
    let size = 1usize << dist.log;
    let mask = size - 1;
    cells.clear();
    cells.resize(size, Cell::default());
    if dist.counts.len() > 256 {
        return Err(corrupt("fse alphabet too large"));
    }
    let mut next = [0u16; 256];
    // cells at or past `high` belong to the "less than 1" symbols.
    let mut high = size;
    for (symbol, &count) in dist.counts.iter().enumerate() {
        if count == -1 {
            high = high
                .checked_sub(1)
                .ok_or(corrupt("fse low-probability symbols overflow the table"))?;
            cells[high].symbol = symbol as u8;
            next[symbol] = 1;
        } else {
            next[symbol] = count.max(0) as u16;
        }
    }
    let step = (size >> 1) + (size >> 3) + 3;
    let mut position = 0usize;
    for (symbol, &count) in dist.counts.iter().enumerate() {
        for _ in 0..count.max(0) {
            if position >= high {
                return Err(corrupt("fse counts overflow the table"));
            }
            cells[position].symbol = symbol as u8;
            position = (position + step) & mask;
            while position >= high {
                position = (position + step) & mask;
            }
        }
    }
    if position != 0 {
        return Err(corrupt("fse counts do not fill the table"));
    }
    for cell in cells.iter_mut() {
        let state = &mut next[usize::from(cell.symbol)];
        let s = u32::from(*state);
        if s == 0 {
            return Err(corrupt("fse cell of a symbol without probability"));
        }
        *state += 1;
        let nb_bits = dist
            .log
            .checked_sub(s.ilog2())
            .ok_or(corrupt("fse state out of range"))?;
        let base = (s << nb_bits)
            .checked_sub(size as u32)
            .ok_or(corrupt("fse state out of range"))?;
        cell.nb_bits = nb_bits as u8;
        cell.base = base as u16;
    }
    Ok(())
}

/// decode the FSE-compressed Huffman weights: one description (log at most
/// 6) and one backward stream read by two interleaved states that share the
/// table. the count is not stored: decoding ends when a state update runs
/// past the stream, and the other state then yields the last weight. at
/// most 255 weights (the last symbol's weight is implied).
pub(super) fn decode_weights(src: &[u8], out: &mut Vec<u8>) -> Res<()> {
    const MAX_WEIGHTS: usize = 255;
    let (dist, used) = read_distribution(src, 255, 6)?;
    let mut cells = Vec::new();
    build_table(&dist, &mut cells)?;
    let mask = cells.len() - 1;
    let mut bits = BackwardBits::new(&src[used..])?;
    let mut s1 = bits.read(dist.log) as usize;
    let mut s2 = bits.read(dist.log) as usize;
    if bits.overflowed() {
        return Err(corrupt("huffman weights stream too short"));
    }
    out.clear();
    loop {
        // room for this weight and the final one of the other state.
        if out.len() > MAX_WEIGHTS - 2 {
            return Err(corrupt("too many huffman weights"));
        }
        let c = cells[s1 & mask];
        out.push(c.symbol);
        s1 = usize::from(c.base) + bits.read(u32::from(c.nb_bits)) as usize;
        if bits.overflowed() {
            out.push(cells[s2 & mask].symbol);
            break;
        }
        if out.len() > MAX_WEIGHTS - 2 {
            return Err(corrupt("too many huffman weights"));
        }
        let c = cells[s2 & mask];
        out.push(c.symbol);
        s2 = usize::from(c.base) + bits.read(u32::from(c.nb_bits)) as usize;
        if bits.overflowed() {
            out.push(cells[s1 & mask].symbol);
            break;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dist(counts: &[i16], log: u32) -> Distribution {
        Distribution {
            counts: counts.to_vec(),
            log,
        }
    }

    #[test]
    fn rfc_baseline_example() {
        // RFC 8878 table 21: a symbol of probability 5 in a 128-state table
        // gets widths 32,32,32,16,16 and baselines 32,64,96,0,16 in state
        // order. here symbol 0 has 5 and symbol 1 the other 123.
        let d = dist(&[5, 123], 7);
        let mut cells = Vec::new();
        build_table(&d, &mut cells).unwrap();
        let mine: Vec<(u8, u16)> = cells
            .iter()
            .filter(|c| c.symbol == 0)
            .map(|c| (c.nb_bits, c.base))
            .collect();
        assert_eq!(mine, vec![(5, 32), (5, 64), (5, 96), (4, 0), (4, 16)]);
    }

    #[test]
    fn low_probability_symbols_take_the_last_cells() {
        // 32-state table: symbol 1 is "less than 1", the rest is symbol 0.
        let d = dist(&[31, -1], 5);
        let mut cells = Vec::new();
        build_table(&d, &mut cells).unwrap();
        assert_eq!(cells[31].symbol, 1);
        assert_eq!(cells[31].nb_bits, 5);
        assert_eq!(cells[31].base, 0);
        assert!(cells[..31].iter().all(|c| c.symbol == 0));
    }

    #[test]
    fn rejects_counts_that_do_not_fill_the_table() {
        let d = dist(&[30, 1], 5);
        assert!(build_table(&d, &mut Vec::new()).is_err());
    }

    #[test]
    fn description_by_hand() {
        // log 5 (nibble 0), then symbol 0: 6-bit field, max = 63 - 33 = 30,
        // so 17 fits the short 5-bit form -> count 16, 17 points left. then
        // a 5-bit field with max = 31 - 17 = 14: the low 4 bits 15 >= 14
        // take the long form, 31 - 14 = 17 -> count 16, done.
        // bits lowest first: 0000 | 10001 | 11111 = 0x3F10.
        let (d, used) = read_distribution(&[0x10, 0x3F], 35, 9).unwrap();
        assert_eq!(d.log, 5);
        assert_eq!(d.counts, vec![16, 16]);
        assert_eq!(used, 2);
        assert!(read_distribution(&[0x10], 35, 9).is_err()); // runs past input
        assert!(read_distribution(&[], 35, 9).is_err());
        assert!(read_distribution(&[0x0F], 35, 9).is_err()); // log 20
        assert!(read_distribution(&[0x05], 35, 9).is_err()); // log 10 > 9
    }
}
