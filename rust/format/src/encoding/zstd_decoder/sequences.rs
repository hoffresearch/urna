//! the sequences section of a compressed block, RFC 8878 sections
//! 3.1.1.3.2 to 3.1.1.5: the header and table modes, the interleaved
//! backward stream of codes, the repeat offsets, and the execution that
//! copies literals and matches into the output.

use super::bits::BackwardBits;
use super::codes::{Kind, SeqTable, SeqTables, predefined};
use super::fse::{Cell, read_distribution};
use super::{Res, corrupt, make_room};

/// which table a symbol type uses in the current block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Which {
    /// none yet in this frame: repeat mode is corrupt.
    Missing,
    Predefined,
    Dictionary,
    /// built by a block of this frame (RLE or FSE-compressed mode).
    Frame,
}

/// what the sequences of a frame carry from block to block: the table each
/// type last used and the three repeat offsets.
#[derive(Debug, Clone)]
pub(super) struct SeqState {
    pub(super) which: [Which; 3],
    pub(super) tables: SeqTables,
    pub(super) reps: [u64; 3],
    scratch: Vec<Cell>,
}

impl Default for SeqState {
    fn default() -> Self {
        Self {
            which: [Which::Missing; 3],
            tables: SeqTables::default(),
            reps: [1, 4, 8],
            scratch: Vec::new(),
        }
    }
}

const KINDS: [Kind; 3] = [Kind::LiteralLength, Kind::Offset, Kind::MatchLength];

fn slot(kind: Kind) -> usize {
    match kind {
        Kind::LiteralLength => 0,
        Kind::Offset => 1,
        Kind::MatchLength => 2,
    }
}

impl SeqState {
    /// start a frame: the dictionary's tables and offsets when it has
    /// them, else nothing to repeat and the offsets 1, 4, 8.
    pub(super) fn reset(&mut self, dict: Option<(&SeqTables, [u64; 3])>) {
        match dict {
            Some((_, reps)) => {
                self.which = [Which::Dictionary; 3];
                self.reps = reps;
            }
            None => {
                self.which = [Which::Missing; 3];
                self.reps = [1, 4, 8];
            }
        }
    }

    /// read one table mode; returns the bytes its description took.
    fn read_mode(&mut self, kind: Kind, mode: u8, src: &[u8]) -> Res<usize> {
        let i = slot(kind);
        match mode {
            0 => {
                self.which[i] = Which::Predefined;
                Ok(0)
            }
            1 => {
                let symbol = *src.first().ok_or(corrupt("rle sequence code missing"))?;
                self.tables.get_mut(kind).rle(kind, symbol)?;
                self.which[i] = Which::Frame;
                Ok(1)
            }
            2 => {
                let (dist, used) = read_distribution(src, kind.max_symbol(), kind.max_log())?;
                self.tables
                    .get_mut(kind)
                    .fill(&dist, kind, &mut self.scratch)?;
                self.which[i] = Which::Frame;
                Ok(used)
            }
            _ => {
                if self.which[i] == Which::Missing {
                    return Err(corrupt("repeat mode without a previous table"));
                }
                Ok(0)
            }
        }
    }
}

/// the output side of a block: the decoded bytes so far (all frames),
/// where this frame starts in them, the hard size limit, and the history
/// before the frame (the dictionary content, possibly empty).
#[derive(Debug)]
pub(super) struct Output<'o, 'h> {
    pub(super) out: &'o mut Vec<u8>,
    pub(super) frame_start: usize,
    pub(super) limit: usize,
    pub(super) history: &'h [u8],
}

/// the number of sequences and the bytes that count took.
fn read_count(src: &[u8]) -> Res<(usize, usize)> {
    let short = || corrupt("sequences header truncated");
    let b0 = usize::from(*src.first().ok_or_else(short)?);
    if b0 < 128 {
        return Ok((b0, 1));
    }
    let b1 = usize::from(*src.get(1).ok_or_else(short)?);
    if b0 < 255 {
        return Ok((((b0 - 128) << 8) + b1, 2));
    }
    let b2 = usize::from(*src.get(2).ok_or_else(short)?);
    Ok((b1 + (b2 << 8) + 0x7F00, 3))
}

/// decode the sequences section `src` and execute it with `lits`: every
/// sequence copies its literals then its match; the literals left after the
/// last sequence close the block.
pub(super) fn decode(
    src: &[u8],
    state: &mut SeqState,
    dict: Option<&SeqTables>,
    lits: &[u8],
    output: &mut Output<'_, '_>,
) -> Res<()> {
    let (count, mut pos) = read_count(src)?;
    if count == 0 {
        if pos != src.len() {
            return Err(corrupt("data after an empty sequences section"));
        }
        make_room(output.out, lits.len(), output.limit)?;
        output.out.extend_from_slice(lits);
        return Ok(());
    }
    let modes = *src.get(pos).ok_or(corrupt("sequences modes missing"))?;
    if modes & 3 != 0 {
        return Err(corrupt("sequences modes reserved bits set"));
    }
    pos += 1;
    for (kind, shift) in KINDS.into_iter().zip([6u8, 4, 2]) {
        let rest = src.get(pos..).unwrap_or_default();
        pos += state.read_mode(kind, (modes >> shift) & 3, rest)?;
    }
    let stream = src
        .get(pos..)
        .ok_or(corrupt("sequences tables past the block"))?;
    let sources = Sources {
        frame: &state.tables,
        predefined: predefined()?,
        dict,
    };
    let ll = sources.pick(state.which, Kind::LiteralLength)?;
    let of = sources.pick(state.which, Kind::Offset)?;
    let ml = sources.pick(state.which, Kind::MatchLength)?;
    let mut reps = state.reps;
    execute(stream, count, [ll, of, ml], &mut reps, lits, output)?;
    state.reps = reps;
    Ok(())
}

/// the places a block's table can live.
struct Sources<'a> {
    frame: &'a SeqTables,
    predefined: &'a SeqTables,
    dict: Option<&'a SeqTables>,
}

impl<'a> Sources<'a> {
    fn pick(&self, which: [Which; 3], kind: Kind) -> Res<&'a SeqTable> {
        match which[slot(kind)] {
            Which::Missing => Err(corrupt("sequence table missing")),
            Which::Predefined => Ok(self.predefined.get(kind)),
            Which::Frame => Ok(self.frame.get(kind)),
            Which::Dictionary => self
                .dict
                .map(|d| d.get(kind))
                .ok_or(corrupt("dictionary tables missing")),
        }
    }
}

/// the decode loop proper: per sequence, the offset bits, then the match
/// length bits, then the literals length bits, then (except after the last
/// one) the state updates in the order literals length, match length,
/// offset. the stream must end exactly.
fn execute(
    stream: &[u8],
    count: usize,
    [ll, of, ml]: [&SeqTable; 3],
    reps: &mut [u64; 3],
    lits: &[u8],
    output: &mut Output<'_, '_>,
) -> Res<()> {
    let mut bits = BackwardBits::new(stream)?;
    let (llm, ofm, mlm) = (ll.cells.len() - 1, of.cells.len() - 1, ml.cells.len() - 1);
    let mut s_ll = bits.read(ll.log) as usize & llm;
    let mut s_of = bits.read(of.log) as usize & ofm;
    let mut s_ml = bits.read(ml.log) as usize & mlm;
    let mut lit_pos = 0usize;
    for i in 0..count {
        let (lc, oc, mc) = (ll.cells[s_ll], of.cells[s_of], ml.cells[s_ml]);
        let value = u64::from(oc.base) + bits.read(u32::from(oc.extra));
        let offset = resolve_offset(value, lc.base == 0, reps)?;
        let match_len = mc.base as usize + bits.read(u32::from(mc.extra)) as usize;
        let lit_len = lc.base as usize + bits.read(u32::from(lc.extra)) as usize;
        if i + 1 < count {
            s_ll = (usize::from(lc.next) + bits.read(u32::from(lc.nb_bits)) as usize) & llm;
            s_ml = (usize::from(mc.next) + bits.read(u32::from(mc.nb_bits)) as usize) & mlm;
            s_of = (usize::from(oc.next) + bits.read(u32::from(oc.nb_bits)) as usize) & ofm;
        }
        let lit_end = lit_pos
            .checked_add(lit_len)
            .filter(|&end| end <= lits.len())
            .ok_or(corrupt("sequence takes more literals than decoded"))?;
        make_room(output.out, lit_len + match_len, output.limit)?;
        output.out.extend_from_slice(&lits[lit_pos..lit_end]);
        lit_pos = lit_end;
        copy_match(output, offset, match_len)?;
    }
    if !bits.finished() {
        return Err(corrupt("sequences stream not consumed exactly"));
    }
    let rest = &lits[lit_pos..];
    make_room(output.out, rest.len(), output.limit)?;
    output.out.extend_from_slice(rest);
    Ok(())
}

/// turn an offset value into a distance and update the repeat offsets
/// (section 3.1.1.5). values above 3 are a new distance; 1..=3 name a
/// repeat offset, shifted by one when the literals length is zero, where
/// the fourth case is the most recent offset minus one.
fn resolve_offset(value: u64, no_literals: bool, reps: &mut [u64; 3]) -> Res<u64> {
    if value > 3 {
        let offset = value - 3;
        *reps = [offset, reps[0], reps[1]];
        return Ok(offset);
    }
    let index = value - 1 + u64::from(no_literals);
    let offset = match index {
        0 => reps[0],
        1 => {
            let o = reps[1];
            *reps = [o, reps[0], reps[2]];
            o
        }
        2 => {
            let o = reps[2];
            *reps = [o, reps[0], reps[1]];
            o
        }
        _ => {
            let o = reps[0].wrapping_sub(1);
            *reps = [o, reps[0], reps[1]];
            o
        }
    };
    if offset == 0 || offset == u64::MAX {
        return Err(corrupt("repeat offset of zero"));
    }
    Ok(offset)
}

/// copy `len` bytes from `offset` back: first from the history before the
/// frame when the offset reaches past its start, then from the frame
/// itself, overlapping when the offset is shorter than the match (each
/// pass doubles the bytes it can copy at once).
fn copy_match(output: &mut Output<'_, '_>, offset: u64, len: usize) -> Res<()> {
    let out = &mut *output.out;
    let produced = out.len() - output.frame_start;
    let offset = usize::try_from(offset).map_err(|_| corrupt("match offset too large"))?;
    let mut rest = len;
    if offset > produced {
        let back = offset - produced;
        if back > output.history.len() {
            return Err(corrupt("match offset before the start of the history"));
        }
        let start = output.history.len() - back;
        let take = rest.min(back);
        out.extend_from_slice(&output.history[start..start + take]);
        rest -= take;
    }
    if rest > 0 {
        let start = out.len() - offset;
        while rest > 0 {
            let n = rest.min(out.len() - start);
            out.extend_from_within(start..start + n);
            rest -= n;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc_repeat_offset_table() {
        // RFC 8878 table 18, row by row: (offset_value, literals_length)
        // -> repeat offsets after the sequence.
        let mut reps = [1u64, 4, 8];
        let rows: [(u64, bool, [u64; 3]); 9] = [
            (1114, false, [1111, 1, 4]),
            (1, false, [1111, 1, 4]),
            (2225, false, [2222, 1111, 1]),
            (1114, false, [1111, 2222, 1111]),
            (3336, false, [3333, 1111, 2222]),
            (2, false, [1111, 3333, 2222]),
            (3, false, [2222, 1111, 3333]),
            (3, true, [2221, 2222, 1111]),
            (1, true, [2222, 2221, 1111]),
        ];
        // the RFC's last two rows print offset_value 1 for both and 3333
        // as the third offset at the end; the update rules above them (and
        // libzstd) give value 3 for the "minus one" row and keep 1111.
        for (n, (value, no_lits, want)) in rows.into_iter().enumerate() {
            resolve_offset(value, no_lits, &mut reps).unwrap();
            assert_eq!(reps, want, "row {n}");
        }
    }

    #[test]
    fn rejects_zero_repeat_offset() {
        let mut reps = [1u64, 4, 8];
        assert!(resolve_offset(3, true, &mut reps).is_err());
    }

    #[test]
    fn overlapping_match_repeats_the_pattern() {
        let mut out = b"xab".to_vec();
        let mut output = Output {
            out: &mut out,
            frame_start: 1,
            limit: 100,
            history: b"HIST",
        };
        copy_match(&mut output, 2, 7).unwrap();
        assert_eq!(out, b"xababababa");
        let mut out = b"xab".to_vec();
        let mut output = Output {
            out: &mut out,
            frame_start: 1,
            limit: 100,
            history: b"HIST",
        };
        // 4 back from the end: 2 in the frame, then 2 into the history.
        copy_match(&mut output, 4, 5).unwrap();
        assert_eq!(out, b"xabSTabS");
        let mut output = Output {
            out: &mut b"xab".to_vec(),
            frame_start: 1,
            limit: 100,
            history: b"HIST",
        };
        assert!(copy_match(&mut output, 7, 1).is_err());
    }
}
