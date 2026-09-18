//! fsst symbol table: deterministic greedy build + byte trie for O(1)
//! longest-match encoding. separated from `fsst.rs` so both files stay
//! under the 300-line source limit.

use super::txt_streams::malformed;

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
    /// deterministic: candidate substrings are counted in a hashmap and ranked
    /// by (saved bytes desc, bytes asc), so two builds match exactly.
    pub(super) fn build(corpus: &[u8]) -> Self {
        use std::collections::HashMap;
        let mut counts: HashMap<Vec<u8>, u64> = HashMap::with_capacity(corpus.len());
        let mut key_buf = Vec::with_capacity(MAX_SYMBOL_LEN);
        for len in 1..=MAX_SYMBOL_LEN {
            if corpus.len() < len {
                break;
            }
            for w in corpus.windows(len) {
                key_buf.clear();
                key_buf.extend_from_slice(w);
                *counts.entry(key_buf.clone()).or_insert(0) += 1;
            }
        }
        let mut ranked: Vec<(Vec<u8>, u64)> = counts.into_iter().collect();
        ranked.sort_by(|a, b| {
            let gain_a = (a.0.len() as u64 - 1) * a.1;
            let gain_b = (b.0.len() as u64 - 1) * b.1;
            gain_b.cmp(&gain_a).then_with(|| a.0.cmp(&b.0))
        });
        let mut symbols: Vec<Vec<u8>> = Vec::with_capacity(N_CODES);
        let mut seen: std::collections::BTreeSet<Vec<u8>> = std::collections::BTreeSet::new();
        for (sym, _) in &ranked {
            if symbols.len() >= N_CODES {
                break;
            }
            if seen.insert(sym.clone()) {
                symbols.push(sym.clone());
            }
        }
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
