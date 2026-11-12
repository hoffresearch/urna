//! zstd dictionaries, RFC 8878 section 5: a formatted dictionary (magic
//! `0xEC30A437`, an ID, the entropy tables, three repeat offsets, then the
//! content) as `zdict_builder` trains it, or raw content of any other shape.
//!
//! the history a frame may reference is the whole dictionary buffer, as
//! libzstd's `ZSTD_DDict` exposes it: the content is its tail, so every
//! offset into the content is the same either way, and a frame libzstd
//! decodes with a prepared dictionary decodes the same here.

use super::codes::{Kind, SeqTables};
use super::fse::read_distribution;
use super::huffman::{self, HuffTable};
use super::{Res, corrupt};

const MAGIC: u32 = 0xEC30_A437;

/// the entropy a formatted dictionary hands to the first block of a frame.
#[derive(Debug, Clone)]
pub(super) struct Entropy {
    pub(super) huffman: HuffTable,
    pub(super) tables: SeqTables,
    pub(super) reps: [u64; 3],
}

/// a parsed dictionary over borrowed bytes.
pub(super) struct Dictionary<'d> {
    /// 0 for raw content (and allowed in a formatted one).
    pub(super) id: u32,
    pub(super) history: &'d [u8],
    pub(super) entropy: Option<Entropy>,
}

impl std::fmt::Debug for Dictionary<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dictionary")
            .field("id", &self.id)
            .field("len", &self.history.len())
            .field("entropy", &self.entropy.is_some())
            .finish()
    }
}

fn le32(b: &[u8], at: usize) -> Res<u32> {
    let w = b.get(at..at + 4).ok_or(corrupt("dictionary truncated"))?;
    Ok(u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
}

impl<'d> Dictionary<'d> {
    /// parse `bytes` the way libzstd's `ZSTD_createDDict` does: fewer than
    /// 8 bytes or no magic is raw content; with the magic, the entropy
    /// section must be well formed and every repeat offset must point into
    /// the content.
    pub(super) fn parse(bytes: &'d [u8]) -> Res<Self> {
        if bytes.len() < 8 || le32(bytes, 0)? != MAGIC {
            return Ok(Self {
                id: 0,
                history: bytes,
                entropy: None,
            });
        }
        let id = le32(bytes, 4)?;
        if bytes.len() <= 8 {
            return Err(corrupt("dictionary too small"));
        }
        let mut pos = 8;
        let mut huffman = HuffTable::default();
        pos += huffman::read_table(&bytes[pos..], &mut huffman)?;
        let mut tables = SeqTables::default();
        let mut scratch = Vec::new();
        // stored in the order offsets, match lengths, literals lengths.
        for kind in [Kind::Offset, Kind::MatchLength, Kind::LiteralLength] {
            let rest = bytes.get(pos..).ok_or(corrupt("dictionary truncated"))?;
            let (dist, used) = read_distribution(rest, kind.max_symbol(), kind.max_log())?;
            tables.get_mut(kind).fill(&dist, kind, &mut scratch)?;
            pos += used;
        }
        let content = bytes
            .len()
            .checked_sub(pos + 12)
            .ok_or(corrupt("dictionary repeat offsets truncated"))?;
        let mut reps = [0u64; 3];
        for (i, rep) in reps.iter_mut().enumerate() {
            let r = le32(bytes, pos + 4 * i)?;
            if r == 0 || r as usize > content {
                return Err(corrupt("dictionary repeat offset out of its content"));
            }
            *rep = u64::from(r);
        }
        Ok(Self {
            id,
            history: bytes,
            entropy: Some(Entropy {
                huffman,
                tables,
                reps,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_content_dictionaries() {
        let d = Dictionary::parse(b"short").unwrap();
        assert_eq!((d.id, d.history.len(), d.entropy.is_none()), (0, 5, true));
        let d = Dictionary::parse(b"any bytes without the magic").unwrap();
        assert!(d.entropy.is_none());
    }

    #[test]
    fn formatted_dictionary_must_be_well_formed() {
        let mut bad = MAGIC.to_le_bytes().to_vec();
        bad.extend_from_slice(&7u32.to_le_bytes());
        assert!(Dictionary::parse(&bad).is_err()); // nothing after the id
        bad.extend_from_slice(&[0xFF; 16]);
        assert!(Dictionary::parse(&bad).is_err()); // garbage tables
    }
}
