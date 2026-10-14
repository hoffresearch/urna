//! the chunks_canonical text-codec chooser (compressed presets only).
//!
//! under a compressed (zstd-text) preset the writer takes the SMALLEST of
//! five candidates for the chunks_canonical (0x02) section, so the build
//! never regresses (single-frame zstd is always in the race):
//!
//!   1. single-frame zstd            (the existing cold form)
//!   2. txt_streams cold             (per-chunk zstd, encoding id 10)
//!   3. txt_streams + trained dict   (encoding id 5, dict in section 0x0A)
//!   4. txt_streams + fsst           (encoding id 9, self-contained table)
//!   5. dedup + single-frame zstd    (unique pool zstd, back-refs in 0x0B)
//!
//! every candidate decodes BYTE-IDENTICALLY to the raw chunks_canonical
//! payload (see `reader::decode`), so content_hash and `urna://` citations
//! are unchanged regardless of which wins. raw-text presets (and the golden)
//! never reach here; they keep raw bytes and stay byte-identical.
//!
//! draws from duckdb (per-section analyze-and-pick over candidate codecs),
//! facebook/zstd + rocksdb (one trained cross-record dictionary), duckdb fsst
//! (static symbol table), and nix/ipfs (content-hash dedup before the entropy
//! coder, on decompressed bytes).

use super::SectionEncoding;
use super::payload::maybe_zstd;
use crate::encoding::{
    dedup, encode_dedup_map, encode_fsst, encode_txt_streams, encode_zstd_dict, train_dict,
};
use crate::layout::{
    SECTION_CHUNKS_CANONICAL, SECTION_DEDUP_MAP, SECTION_DICTIONARY, SECTION_ENCODING_FSST,
    SECTION_ENCODING_RAW, SECTION_ENCODING_TXT_STREAMS, SECTION_ENCODING_ZSTD_DICT,
};
use crate::sections::encode_chunks_canonical;
use sha2::{Digest, Sha256};
use std::sync::Mutex;

/// the chosen chunks_canonical section plus any auxiliary optional sections
/// (the dict 0x0A and/or the dedup map 0x0B) the winning codec needs.
#[derive(Clone)]
pub(super) struct TextChoice {
    /// `(section_id, encoding, payload)` for chunks_canonical.
    pub canonical: (u32, u32, Vec<u8>),
    /// extra `(section_id, encoding, payload)` sections to emit (0x0A/0x0B).
    pub aux: Vec<(u32, u32, Vec<u8>)>,
}

/// the last choice this process made, keyed by the sha-256 of the raw
/// chunks_canonical payload. the choice is a pure function of the texts, so
/// a later build over the same texts (another preset or mrl point of one
/// corpus: the bench ladder builds twelve) reuses it instead of running the
/// five candidates again. one slot: holds one compressed section at most.
static LAST: Mutex<Option<([u8; 32], TextChoice)>> = Mutex::new(None);

/// pick the smallest chunks_canonical encoding among the five candidates,
/// reusing the previous choice when the texts are the same. `compressed`
/// must be true (the caller only invokes this under a zstd-text preset); the
/// cold single-frame zstd is the never-regress floor.
pub(super) fn choose(texts: &[String]) -> crate::Result<TextChoice> {
    let canonical_raw = encode_chunks_canonical(texts)?;
    let key: [u8; 32] = Sha256::digest(&canonical_raw).into();
    // a poisoned lock only means another build panicked mid-update; the slot
    // is then just not used, never trusted.
    if let Ok(last) = LAST.lock() {
        if let Some((k, choice)) = last.as_ref() {
            if *k == key {
                return Ok(choice.clone());
            }
        }
    }
    let choice = compute(texts, canonical_raw)?;
    if let Ok(mut last) = LAST.lock() {
        *last = Some((key, choice.clone()));
    }
    Ok(choice)
}

/// run the five candidates and keep the smallest. they are independent, so
/// they run at the same time (the build costs the slowest, the single-frame
/// zstd-19, not the sum); the race is still judged in candidate order, so
/// ties keep the earlier candidate exactly as a sequential run would.
fn compute(texts: &[String], canonical_raw: Vec<u8>) -> crate::Result<TextChoice> {
    let ((single, streams), (dict, (fsst, dedup))) = rayon::join(
        || rayon::join(|| single_frame(canonical_raw), || txt_streams(texts)),
        || {
            rayon::join(
                || with_dict(texts),
                || rayon::join(|| fsst(texts), || dedup_zstd(texts)),
            )
        },
    );
    let mut best = single?;
    for other in [Some(streams?), dict?, Some(fsst?), dedup?]
        .into_iter()
        .flatten()
    {
        best.consider(other);
    }
    Ok(TextChoice {
        canonical: best.canonical,
        aux: best.aux,
    })
}

/// candidate 1: single-frame zstd (the existing form, always the floor).
fn single_frame(canonical_raw: Vec<u8>) -> crate::Result<Candidate> {
    Ok(Candidate::plain(maybe_zstd(
        SECTION_CHUNKS_CANONICAL,
        SectionEncoding::Zstd,
        canonical_raw,
    )?))
}

/// candidate 2: txt_streams cold (per-chunk zstd + intpack offset table).
fn txt_streams(texts: &[String]) -> crate::Result<Candidate> {
    Ok(Candidate::plain((
        SECTION_CHUNKS_CANONICAL,
        SECTION_ENCODING_TXT_STREAMS,
        encode_txt_streams(texts)?,
    )))
}

/// candidate 3: txt_streams + trained dict. the dict is a separate optional
/// section (0x0A), excluded from content_hash; its size counts toward the
/// candidate total so the chooser is honest about the dict cost.
fn with_dict(texts: &[String]) -> crate::Result<Option<Candidate>> {
    let Some(dict) = train_dict(&sorted_unique(texts)) else {
        return Ok(None);
    };
    let framed = encode_zstd_dict(texts, &dict)?;
    let total = framed.len() + dict.len();
    Ok(Some(Candidate {
        canonical: (SECTION_CHUNKS_CANONICAL, SECTION_ENCODING_ZSTD_DICT, framed),
        aux: vec![(SECTION_DICTIONARY, SECTION_ENCODING_RAW, dict)],
        total,
    }))
}

/// candidate 4: txt_streams + fsst (self-contained static symbol table).
fn fsst(texts: &[String]) -> crate::Result<Candidate> {
    Ok(Candidate::plain((
        SECTION_CHUNKS_CANONICAL,
        SECTION_ENCODING_FSST,
        encode_fsst(texts)?,
    )))
}

/// candidate 5: dedup + single-frame zstd. the dedup pass runs on the
/// DECOMPRESSED canonical texts (the nix/ipfs order rule), then the UNIQUE
/// pool is zstd-compressed; the back-references live in section 0x0B,
/// excluded from content_hash. only competes when the corpus actually
/// repeats (else unique == texts and it can only lose to candidate 1).
fn dedup_zstd(texts: &[String]) -> crate::Result<Option<Candidate>> {
    let d = dedup(texts);
    if d.unique.len() >= texts.len() {
        return Ok(None);
    }
    let unique_raw = encode_chunks_canonical(&d.unique)?;
    let (cid, cenc, cbytes) =
        maybe_zstd(SECTION_CHUNKS_CANONICAL, SectionEncoding::Zstd, unique_raw)?;
    let map = encode_dedup_map(&d.back_refs);
    let total = cbytes.len() + map.len();
    Ok(Some(Candidate {
        canonical: (cid, cenc, cbytes),
        aux: vec![(SECTION_DEDUP_MAP, SECTION_ENCODING_RAW, map)],
        total,
    }))
}

/// the canonical texts, sorted and deduplicated, as the deterministic ZDICT
/// training input (the trainer is a pure function of its sorted samples).
fn sorted_unique(texts: &[String]) -> Vec<String> {
    let mut v: Vec<String> = texts.to_vec();
    v.sort_unstable();
    v.dedup();
    v
}

/// a single chooser candidate: its chunks_canonical tuple, any aux sections,
/// and the total physical bytes it adds (canonical + aux), the comparison key.
struct Candidate {
    canonical: (u32, u32, Vec<u8>),
    aux: Vec<(u32, u32, Vec<u8>)>,
    total: usize,
}

impl Candidate {
    /// a candidate with no auxiliary sections; total is its payload length.
    fn plain(canonical: (u32, u32, Vec<u8>)) -> Self {
        let total = canonical.2.len();
        Self {
            canonical,
            aux: Vec::new(),
            total,
        }
    }

    /// keep `other` if it is strictly smaller (ties keep the incumbent, so
    /// the cheaper-to-decode earlier candidate wins an equal-size race).
    fn consider(&mut self, other: Candidate) {
        if other.total < self.total {
            *self = other;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> Vec<String> {
        // repeats (so dedup competes) and enough samples to train a dict.
        (0..400)
            .map(|i| {
                format!(
                    "trecho {} do acervo: {}",
                    i % 150,
                    "palavra ".repeat(i % 23)
                )
            })
            .collect()
    }

    #[test]
    #[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
    fn memoized_choice_equals_a_fresh_one() {
        let t = corpus();
        let fresh = compute(&t, encode_chunks_canonical(&t).unwrap()).unwrap();
        let first = choose(&t).unwrap();
        let again = choose(&t).unwrap();
        for c in [&first, &again] {
            assert_eq!(c.canonical, fresh.canonical);
            assert_eq!(c.aux, fresh.aux);
        }
        // different texts never get the cached choice.
        let other: Vec<String> = t.iter().map(|s| format!("{} fim", s)).collect();
        let fresh_other = compute(&other, encode_chunks_canonical(&other).unwrap()).unwrap();
        assert_eq!(choose(&other).unwrap().canonical, fresh_other.canonical);
    }
}
