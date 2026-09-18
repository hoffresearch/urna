//! On-disk encoding/decoding for the BM25 section (`0x08`). Term order:
//! alphabetical (deterministic across builds).
//!
//! Payload version 2 bitpacks the integer-heavy parts with the shared
//! `intpack` codec: the doc lengths, the delta-gapped (sorted) doc ids,
//! and the term frequencies each become one `intpack` column. The decoded
//! index is identical to the built one, so scores are unchanged. Version 1
//! (flat `u32` postings) is still accepted on read. The section is optional
//! and excluded from content_hash.

use std::collections::HashMap;

use urna_format::encoding::{pack_u64s, unpack_u64s};
use urna_format::error::UrnaError;

use super::index::{BM25_PAYLOAD_VERSION, Bm25Index, Posting, TermEntry};
use crate::error::RuntimeError;
use urna_format::bytes::{le_f32, le_u32};

impl Bm25Index {
    /// Encode for storage in section `0x08` (v2).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&BM25_PAYLOAD_VERSION.to_le_bytes());
        out.extend_from_slice(&self.k1.to_le_bytes());
        out.extend_from_slice(&self.b.to_le_bytes());
        out.extend_from_slice(&self.avgdl.to_le_bytes());
        out.extend_from_slice(&(self.n_docs as u32).to_le_bytes());
        out.extend_from_slice(&(self.n_terms as u32).to_le_bytes());

        let dls: Vec<u64> = self.doc_lengths.iter().map(|&x| x as u64).collect();
        write_blob(&mut out, &pack_u64s(&dls));

        // Sorted by term for determinism.
        let mut terms: Vec<(&String, &TermEntry)> = self.terms.iter().collect();
        terms.sort_by(|a, b| a.0.cmp(b.0));
        let mut doc_gaps: Vec<u64> = Vec::new();
        let mut tfs: Vec<u64> = Vec::new();
        for (term, entry) in &terms {
            let bs = term.as_bytes();
            out.extend_from_slice(&(bs.len() as u32).to_le_bytes());
            out.extend_from_slice(bs);
            out.extend_from_slice(&entry.df.to_le_bytes());
            let mut prev = 0u32;
            for p in &entry.postings {
                doc_gaps.push(p.doc.wrapping_sub(prev) as u64);
                prev = p.doc;
                tfs.push(p.tf as u64);
            }
        }
        write_blob(&mut out, &pack_u64s(&doc_gaps));
        write_blob(&mut out, &pack_u64s(&tfs));
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, RuntimeError> {
        let mut cur = ByteCursor::new(bytes);
        let version = cur.u32()?;
        let k1 = cur.f32()?;
        let b = cur.f32()?;
        let avgdl = cur.f32()?;
        let n_docs = cur.u32()? as usize;
        let n_terms = cur.u32()? as usize;
        let (doc_lengths, terms) = match version {
            1 => decode_v1(&mut cur, n_docs, n_terms)?,
            2 => decode_v2(&mut cur, n_docs, n_terms)?,
            other => {
                return Err(RuntimeError::Format(UrnaError::UnsupportedSectionVersion {
                    section_id: urna_format::layout::SECTION_BM25_INDEX,
                    version: other,
                }));
            }
        };
        // semantic validation of an untrusted payload (found by the mutation
        // fuzz harness): a posting whose doc id is outside `doc_lengths`
        // indexed out of bounds at query time, and non-finite parameters
        // would turn every score into NaN.
        if !(k1.is_finite() && b.is_finite() && avgdl.is_finite()) {
            return Err(malformed("bm25: non-finite k1 / b / avgdl"));
        }
        if doc_lengths.len() != n_docs {
            return Err(malformed("bm25: doc-length count != n_docs"));
        }
        for entry in terms.values() {
            if entry.postings.iter().any(|p| p.doc as usize >= n_docs) {
                return Err(malformed("bm25: posting doc id >= n_docs"));
            }
        }
        Ok(Self {
            k1,
            b,
            avgdl,
            n_docs,
            n_terms,
            doc_lengths,
            terms,
        })
    }
}

fn malformed(reason: impl Into<String>) -> RuntimeError {
    RuntimeError::Format(UrnaError::MalformedSectionPayload {
        section_id: urna_format::layout::SECTION_BM25_INDEX,
        reason: reason.into(),
    })
}

fn write_blob(out: &mut Vec<u8>, blob: &[u8]) {
    out.extend_from_slice(&(blob.len() as u32).to_le_bytes());
    out.extend_from_slice(blob);
}

/// hostile-input guard: `n_docs`, `n_terms`, and `df` are raw `u32` read from
/// an untrusted section and would otherwise be passed straight to
/// `Vec::/HashMap::with_capacity`, letting a crafted header (e.g. `df =
/// 0xFFFF_FFFF`) force a multi-GiB reservation and abort the process at
/// `open()`. Cap the capacity HINT and let the collection grow on demand; the
/// cursor's per-read EOF check bounds the actual work to the bytes present.
const MAX_PREALLOC: usize = 1 << 20;

type Decoded = (Vec<u32>, HashMap<String, TermEntry>);

/// v1: flat `u32` doc lengths and `(doc, tf)` postings.
fn decode_v1(cur: &mut ByteCursor, n_docs: usize, n_terms: usize) -> Result<Decoded, RuntimeError> {
    let mut doc_lengths = Vec::with_capacity(n_docs.min(MAX_PREALLOC));
    for _ in 0..n_docs {
        doc_lengths.push(cur.u32()?);
    }
    let mut terms: HashMap<String, TermEntry> = HashMap::with_capacity(n_terms.min(MAX_PREALLOC));
    for _ in 0..n_terms {
        let term = cur.term()?;
        let df = cur.u32()?;
        let mut postings = Vec::with_capacity((df as usize).min(MAX_PREALLOC));
        for _ in 0..df {
            let doc = cur.u32()?;
            let tf = cur.u32()?;
            postings.push(Posting { doc, tf });
        }
        terms.insert(term, TermEntry { df, postings });
    }
    Ok((doc_lengths, terms))
}

/// v2: `intpack` columns for doc lengths, delta-gapped doc ids, and tfs.
fn decode_v2(cur: &mut ByteCursor, n_docs: usize, n_terms: usize) -> Result<Decoded, RuntimeError> {
    let dls = cur.intpack_column()?;
    if dls.len() != n_docs {
        return Err(malformed("bm25 v2: doc-length column mismatch"));
    }
    let doc_lengths: Vec<u32> = dls.iter().map(|&x| x as u32).collect();

    // term dictionary: (term, df) in alphabetical order.
    let mut dict: Vec<(String, u32)> = Vec::with_capacity(n_terms.min(MAX_PREALLOC));
    let mut total: usize = 0;
    for _ in 0..n_terms {
        let term = cur.term()?;
        let df = cur.u32()?;
        total = total.saturating_add(df as usize);
        dict.push((term, df));
    }
    let gaps = cur.intpack_column()?;
    let tfs = cur.intpack_column()?;
    if gaps.len() != total || tfs.len() != total {
        return Err(malformed("bm25 v2: posting column mismatch"));
    }
    let mut terms: HashMap<String, TermEntry> = HashMap::with_capacity(n_terms.min(MAX_PREALLOC));
    let mut k = 0usize;
    for (term, df) in dict {
        let mut postings = Vec::with_capacity((df as usize).min(MAX_PREALLOC));
        let mut prev = 0u32;
        for _ in 0..df {
            let doc = prev.wrapping_add(gaps[k] as u32);
            prev = doc;
            postings.push(Posting {
                doc,
                tf: tfs[k] as u32,
            });
            k += 1;
        }
        terms.insert(term, TermEntry { df, postings });
    }
    Ok((doc_lengths, terms))
}

struct ByteCursor<'a> {
    buf: &'a [u8],
    pos: usize,
}
impl<'a> ByteCursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn u32(&mut self) -> Result<u32, RuntimeError> {
        Ok(le_u32(self.bytes(4)?)?)
    }
    fn f32(&mut self) -> Result<f32, RuntimeError> {
        Ok(le_f32(self.bytes(4)?)?)
    }
    fn bytes(&mut self, n: usize) -> Result<&'a [u8], RuntimeError> {
        if n > self.buf.len() - self.pos {
            return Err(malformed(format!("bm25: unexpected EOF (need {})", n)));
        }
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }
    fn term(&mut self) -> Result<String, RuntimeError> {
        let len = self.u32()? as usize;
        let b = self.bytes(len)?;
        std::str::from_utf8(b)
            .map(|s| s.to_string())
            .map_err(|e| malformed(format!("term utf-8: {}", e)))
    }
    fn intpack_column(&mut self) -> Result<Vec<u64>, RuntimeError> {
        let len = self.u32()? as usize;
        let blob = self.bytes(len)?;
        unpack_u64s(blob).map_err(RuntimeError::Format)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v1_header(n_docs: u32, n_terms: u32) -> Vec<u8> {
        let mut out = 1u32.to_le_bytes().to_vec(); // version 1
        out.extend_from_slice(&1.2f32.to_le_bytes()); // k1
        out.extend_from_slice(&0.75f32.to_le_bytes()); // b
        out.extend_from_slice(&1.0f32.to_le_bytes()); // avgdl
        out.extend_from_slice(&n_docs.to_le_bytes());
        out.extend_from_slice(&n_terms.to_le_bytes());
        out
    }

    // a header claiming ~4.3e9 docs must not reserve ~17 GiB before the read
    // loop hits EOF; it must return a typed error, never OOM/abort.
    #[test]
    fn v1_hostile_n_docs_errors_without_ooming() {
        let p = v1_header(u32::MAX, 0); // no doc-length bytes follow
        assert!(Bm25Index::from_bytes(&p).is_err());
    }

    // a term declaring df = u32::MAX must not reserve ~34 GiB of postings.
    #[test]
    fn v1_hostile_df_errors_without_ooming() {
        let mut p = v1_header(0, 1);
        p.extend_from_slice(&1u32.to_le_bytes()); // term length 1
        p.extend_from_slice(b"a"); // term
        p.extend_from_slice(&u32::MAX.to_le_bytes()); // df = u32::MAX
        // no posting bytes follow
        assert!(Bm25Index::from_bytes(&p).is_err());
    }
}
