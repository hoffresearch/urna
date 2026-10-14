//! fsst symbol table: deterministic greedy build + byte trie for O(1)
//! longest-match encoding. separated from `fsst.rs`, which keeps the
//! encoder and decoder.

use super::txt_streams::malformed;
use rayon::prelude::*;

/// max symbol length in bytes (codes 0..=254 may map to 1..=8 raw bytes).
const MAX_SYMBOL_LEN: usize = 8;
/// number of assignable codes (0..=254); 255 is the escape.
const N_CODES: usize = 255;

/// a built symbol table: `symbols[code]` is the byte string code `code`
/// expands to. at most 255 entries. round-trips through [`serialize_table`].
pub(super) struct SymbolTable {
    symbols: Vec<Vec<u8>>,
    trie: Trie,
}

impl SymbolTable {
    /// greedily build a static table from a frequency pass over `corpus`.
    /// deterministic: every substring of 1..=8 bytes is counted and ranked by
    /// (saved bytes desc, bytes asc), and the first 255 become the codes, so
    /// two builds match exactly.
    ///
    /// the count runs one length at a time over packed `u64` windows, sorted
    /// and run-length counted, keeping only that length's 255 best: any
    /// substring in the overall first 255 is in its own length's first 255
    /// (everything ranked above it there ranks above it overall), and the
    /// ranking is total, so the table equals ranking every substring at once.
    pub(super) fn build(corpus: &[u8]) -> Self {
        let mut ranked: Vec<Ranked> = Vec::with_capacity(N_CODES * MAX_SYMBOL_LEN);
        for len in 1..=MAX_SYMBOL_LEN.min(corpus.len()) {
            ranked.extend(best_of_length(corpus, len));
        }
        ranked.sort_unstable();
        let symbols: Vec<Vec<u8>> = ranked
            .iter()
            .take(N_CODES)
            .map(|r| unpack(r.packed, r.len))
            .collect();
        let mut trie = Trie::new();
        for (code, sym) in symbols.iter().enumerate() {
            trie.insert(sym, code as u8);
        }
        Self { symbols, trie }
    }

    pub(super) fn longest_match(&self, input: &[u8]) -> Option<(u8, usize)> {
        self.trie.longest_match(input)
    }

    pub(super) fn symbols(&self) -> &Vec<Vec<u8>> {
        &self.symbols
    }
}

/// a counted substring in table order: saved bytes descending, then the
/// bytes ascending. `packed` holds the bytes big-endian, zero-padded on the
/// right, and a shorter string sorts first on an equal `packed`, which is
/// exactly the lexicographic order of the byte strings (a proper prefix sorts
/// first, and the zero padding never outranks a real byte).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Ranked {
    neg_gain: std::cmp::Reverse<u64>,
    packed: u64,
    len: usize,
}

/// the bytes of `w` (1..=8 long) big-endian in a `u64`, zero-padded right.
fn pack(w: &[u8]) -> u64 {
    let mut b = [0u8; MAX_SYMBOL_LEN];
    b[..w.len()].copy_from_slice(w);
    u64::from_be_bytes(b)
}

fn unpack(packed: u64, len: usize) -> Vec<u8> {
    packed.to_be_bytes()[..len].to_vec()
}

/// the 255 best substrings of exactly `len` bytes in `corpus`, by table
/// order. counts every window: sorted packed windows, one run per substring.
fn best_of_length(corpus: &[u8], len: usize) -> Vec<Ranked> {
    let mut windows: Vec<u64> = corpus.par_windows(len).map(pack).collect();
    windows.par_sort_unstable();
    let mut runs: Vec<Ranked> = Vec::new();
    for run in windows.chunk_by(|a, b| a == b) {
        runs.push(Ranked {
            neg_gain: std::cmp::Reverse((len as u64 - 1) * run.len() as u64),
            packed: run[0],
            len,
        });
    }
    if runs.len() > N_CODES {
        runs.select_nth_unstable(N_CODES - 1);
        runs.truncate(N_CODES);
    }
    runs
}

/// compact byte trie for greedy longest-match encoding.
#[derive(Default)]
struct Trie {
    next: Vec<[Option<u16>; 256]>,
    code: Vec<Option<u8>>,
}

impl Trie {
    fn new() -> Self {
        Self {
            next: vec![[None; 256]],
            code: vec![None],
        }
    }

    fn insert(&mut self, sym: &[u8], code: u8) {
        let mut node = 0usize;
        for &b in sym {
            let b = b as usize;
            let child = self.next[node][b].unwrap_or_else(|| {
                let idx = self.next.len() as u16;
                self.next.push([None; 256]);
                self.code.push(None);
                self.next[node][b] = Some(idx);
                idx
            });
            node = child as usize;
        }
        self.code[node] = Some(code);
    }

    fn longest_match(&self, input: &[u8]) -> Option<(u8, usize)> {
        let mut node = 0usize;
        let mut best: Option<(u8, usize)> = None;
        for (i, &b) in input.iter().enumerate().take(MAX_SYMBOL_LEN) {
            let child = self.next[node][b as usize]?;
            node = child as usize;
            if let Some(code) = self.code[node] {
                best = Some((code, i + 1));
            }
        }
        best
    }
}

/// serialize the table: u16 count (LE) + per symbol a u8 length + bytes.
pub(super) fn serialize_table(table: &SymbolTable) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(table.symbols().len() as u16).to_le_bytes());
    for sym in table.symbols() {
        out.push(sym.len() as u8);
        out.extend_from_slice(sym);
    }
    out
}

/// parse a serialized table, bounds-checked. returns the symbols and the byte
/// length consumed so the caller can locate what follows it.
pub(super) fn parse_table(bytes: &[u8]) -> crate::Result<(Vec<Vec<u8>>, usize)> {
    if bytes.len() < 2 {
        return Err(malformed("fsst: truncated table count"));
    }
    let count = u16::from_le_bytes([bytes[0], bytes[1]]) as usize;
    if count > N_CODES {
        return Err(malformed("fsst: table count exceeds 255"));
    }
    let mut pos = 2;
    let mut symbols = Vec::with_capacity(count);
    for _ in 0..count {
        let len = *bytes
            .get(pos)
            .ok_or_else(|| malformed("fsst: truncated symbol len"))? as usize;
        if len == 0 || len > MAX_SYMBOL_LEN {
            return Err(malformed("fsst: symbol len out of range"));
        }
        pos += 1;
        let sym = bytes
            .get(pos..pos + len)
            .ok_or_else(|| malformed("fsst: truncated symbol bytes"))?
            .to_vec();
        pos += len;
        symbols.push(sym);
    }
    Ok((symbols, pos))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// the builder before the per-length count, kept to pin the table: every
    /// substring of 1..=8 bytes in one map, all of them ranked.
    fn build_reference(corpus: &[u8]) -> Vec<Vec<u8>> {
        let mut counts: std::collections::HashMap<Vec<u8>, u64> = Default::default();
        for len in 1..=MAX_SYMBOL_LEN {
            if corpus.len() < len {
                break;
            }
            for w in corpus.windows(len) {
                *counts.entry(w.to_vec()).or_insert(0) += 1;
            }
        }
        let mut ranked: Vec<(Vec<u8>, u64)> = counts.into_iter().collect();
        ranked.sort_by(|a, b| {
            let gain_a = (a.0.len() as u64 - 1) * a.1;
            let gain_b = (b.0.len() as u64 - 1) * b.1;
            gain_b.cmp(&gain_a).then_with(|| a.0.cmp(&b.0))
        });
        ranked.into_iter().take(N_CODES).map(|(s, _)| s).collect()
    }

    #[test]
    fn table_equals_the_full_ranking() {
        // pt-br text, zero bytes next to their prefixes (the padding case),
        // few distinct bytes (ties everywhere) and inputs shorter than 8.
        let text = "a informação do acervo está no arquivo, não no índice. ".repeat(40);
        let zeros: Vec<u8> = (0..3_000u32)
            .map(|i| [0u8, 0, 1, b'a'][(i % 7 % 4) as usize])
            .collect();
        let ties: Vec<u8> = (0..5_000u32)
            .map(|i| b"ab"[(i * i % 3 % 2) as usize])
            .collect();
        let lcg: Vec<u8> = (0..20_000u64)
            .scan(0x10901u64, |x, _| {
                *x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                Some((*x >> 59) as u8)
            })
            .collect();
        for corpus in [text.as_bytes(), &zeros, &ties, &lcg, b"", b"a", b"abcab"] {
            let built = SymbolTable::build(corpus);
            assert_eq!(built.symbols, build_reference(corpus));
        }
    }
}
