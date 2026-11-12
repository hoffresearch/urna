//! the code tables of the sequences section, RFC 8878 sections
//! 3.1.1.3.2.1.1 and 3.1.1.3.2.2: literals length, match length and offset
//! codes with their baselines and extra bits, the predefined distributions,
//! and the decoding tables built from them.

use std::sync::OnceLock;

use super::fse::{self, Cell, Distribution};
use super::{Res, ZstdError, corrupt};

/// literals length codes 0..=35: baseline and extra bits (table 16).
const LL_BASE: [u32; 36] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 28, 32, 40, 48, 64,
    128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
];
const LL_BITS: [u8; 36] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16,
];

/// match length codes 0..=52: baseline and extra bits (table 17).
const ML_BASE: [u32; 53] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027,
    2051, 4099, 8195, 16387, 32771, 65539,
];
const ML_BITS: [u8; 53] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
];

/// the predefined distributions (section 3.1.1.3.2.2).
const LL_DEFAULT: [i16; 36] = [
    4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1,
    -1, -1, -1, -1,
];
const ML_DEFAULT: [i16; 53] = [
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1,
];
const OF_DEFAULT: [i16; 29] = [
    1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
];

/// the three symbol types of a sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Kind {
    LiteralLength,
    Offset,
    MatchLength,
}

impl Kind {
    /// the largest code of this type.
    pub(super) fn max_symbol(self) -> usize {
        match self {
            Self::LiteralLength => 35,
            Self::Offset => 31,
            Self::MatchLength => 52,
        }
    }

    /// the largest accuracy log a block may give this type's table.
    pub(super) fn max_log(self) -> u32 {
        match self {
            Self::Offset => 8,
            Self::LiteralLength | Self::MatchLength => 9,
        }
    }

    /// baseline and extra bits of `code`. an offset code `c` is the value
    /// `(1 << c) + c extra bits`; values 1..=3 are the repeat offsets.
    fn code(self, code: u8) -> Res<(u32, u8)> {
        let i = usize::from(code);
        match self {
            Self::LiteralLength => LL_BASE.get(i).zip(LL_BITS.get(i)).map(|(b, e)| (*b, *e)),
            Self::MatchLength => ML_BASE.get(i).zip(ML_BITS.get(i)).map(|(b, e)| (*b, *e)),
            Self::Offset => 1u32.checked_shl(u32::from(code)).map(|base| (base, code)),
        }
        .ok_or(corrupt("sequence code out of range"))
    }
}

/// one state of a sequence decoding table: the code's baseline and extra
/// bits, and the FSE step to the next state.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct SeqCell {
    pub(super) base: u32,
    pub(super) extra: u8,
    pub(super) nb_bits: u8,
    pub(super) next: u16,
}

/// a sequence decoding table of `1 << log` cells (`log` 0 for RLE).
#[derive(Debug, Clone, Default)]
pub(super) struct SeqTable {
    pub(super) log: u32,
    pub(super) cells: Vec<SeqCell>,
}

impl SeqTable {
    /// the table of a distribution, reusing this table's storage; `scratch`
    /// holds the plain FSE cells between the two steps.
    pub(super) fn fill(
        &mut self,
        dist: &Distribution,
        kind: Kind,
        scratch: &mut Vec<Cell>,
    ) -> Res<()> {
        fse::build_table(dist, scratch)?;
        self.log = dist.log;
        self.cells.clear();
        for c in scratch.iter() {
            let (base, extra) = kind.code(c.symbol)?;
            self.cells.push(SeqCell {
                base,
                extra,
                nb_bits: c.nb_bits,
                next: c.base,
            });
        }
        Ok(())
    }

    /// the one-state table of RLE mode: every sequence uses `symbol`.
    pub(super) fn rle(&mut self, kind: Kind, symbol: u8) -> Res<()> {
        if usize::from(symbol) > kind.max_symbol() {
            return Err(corrupt("rle sequence code out of range"));
        }
        let (base, extra) = kind.code(symbol)?;
        self.log = 0;
        self.cells.clear();
        self.cells.push(SeqCell {
            base,
            extra,
            nb_bits: 0,
            next: 0,
        });
        Ok(())
    }
}

/// the three tables a block or a dictionary can supply.
#[derive(Debug, Clone, Default)]
pub(super) struct SeqTables {
    pub(super) ll: SeqTable,
    pub(super) of: SeqTable,
    pub(super) ml: SeqTable,
}

impl SeqTables {
    pub(super) fn get(&self, kind: Kind) -> &SeqTable {
        match kind {
            Kind::LiteralLength => &self.ll,
            Kind::Offset => &self.of,
            Kind::MatchLength => &self.ml,
        }
    }

    pub(super) fn get_mut(&mut self, kind: Kind) -> &mut SeqTable {
        match kind {
            Kind::LiteralLength => &mut self.ll,
            Kind::Offset => &mut self.of,
            Kind::MatchLength => &mut self.ml,
        }
    }
}

fn build_predefined() -> Res<SeqTables> {
    let mut tables = SeqTables::default();
    let mut scratch = Vec::new();
    for (kind, counts, log) in [
        (Kind::LiteralLength, &LL_DEFAULT[..], 6),
        (Kind::Offset, &OF_DEFAULT[..], 5),
        (Kind::MatchLength, &ML_DEFAULT[..], 6),
    ] {
        let dist = Distribution {
            counts: counts.to_vec(),
            log,
        };
        tables.get_mut(kind).fill(&dist, kind, &mut scratch)?;
    }
    Ok(tables)
}

/// the predefined tables, built once per process.
pub(super) fn predefined() -> Res<&'static SeqTables> {
    static PREDEFINED: OnceLock<Result<SeqTables, ZstdError>> = OnceLock::new();
    match PREDEFINED.get_or_init(build_predefined) {
        Ok(tables) => Ok(tables),
        Err(e) => Err(*e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predefined_tables_match_appendix_a() {
        let t = predefined().unwrap();
        assert_eq!((t.ll.log, t.ll.cells.len()), (6, 64));
        assert_eq!((t.of.log, t.of.cells.len()), (5, 32));
        assert_eq!((t.ml.log, t.ml.cells.len()), (6, 64));
        // RFC 8878 appendix A, first rows: (code baseline, extra bits,
        // nb_bits, next-state base).
        let ll0 = t.ll.cells[0];
        assert_eq!((ll0.base, ll0.extra, ll0.nb_bits, ll0.next), (0, 0, 4, 0));
        let ll1 = t.ll.cells[1];
        assert_eq!((ll1.base, ll1.extra, ll1.nb_bits, ll1.next), (0, 0, 4, 16));
        let ll2 = t.ll.cells[2];
        assert_eq!((ll2.base, ll2.extra, ll2.nb_bits, ll2.next), (1, 0, 5, 32));
        let ll3 = t.ll.cells[3];
        assert_eq!((ll3.base, ll3.extra, ll3.nb_bits, ll3.next), (3, 0, 5, 0));
        let ll10 = t.ll.cells[10];
        assert_eq!((ll10.base, ll10.extra, ll10.nb_bits), (14, 0, 6));
        // the last cell holds the first "less than 1" code, 32: full reset.
        let ll63 = t.ll.cells[63];
        assert_eq!(
            (ll63.base, ll63.extra, ll63.nb_bits, ll63.next),
            (8192, 13, 6, 0)
        );
        let ml0 = t.ml.cells[0];
        assert_eq!((ml0.base, ml0.extra, ml0.nb_bits, ml0.next), (3, 0, 6, 0));
        let ml1 = t.ml.cells[1];
        assert_eq!((ml1.base, ml1.extra, ml1.nb_bits, ml1.next), (4, 0, 4, 0));
        let ml2 = t.ml.cells[2];
        assert_eq!((ml2.base, ml2.extra, ml2.nb_bits, ml2.next), (5, 0, 5, 32));
        let ml63 = t.ml.cells[63];
        assert_eq!((ml63.base, ml63.extra, ml63.nb_bits), (1027, 10, 6)); // code 46
        let of0 = t.of.cells[0];
        assert_eq!((of0.base, of0.extra, of0.nb_bits, of0.next), (1, 0, 5, 0));
        let of1 = t.of.cells[1];
        assert_eq!((of1.base, of1.extra, of1.nb_bits, of1.next), (64, 6, 4, 0));
    }

    #[test]
    fn rle_checks_its_code() {
        let mut t = SeqTable::default();
        assert!(t.rle(Kind::Offset, 32).is_err());
        t.rle(Kind::MatchLength, 52).unwrap();
        assert_eq!((t.cells[0].base, t.cells[0].extra), (65539, 16));
    }
}
