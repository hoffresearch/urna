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

/// the chosen chunks_canonical section plus any auxiliary optional sections
/// (the dict 0x0A and/or the dedup map 0x0B) the winning codec needs.
pub(super) struct TextChoice {
    /// `(section_id, encoding, payload)` for chunks_canonical.
    pub canonical: (u32, u32, Vec<u8>),
    /// extra `(section_id, encoding, payload)` sections to emit (0x0A/0x0B).
    pub aux: Vec<(u32, u32, Vec<u8>)>,
}

/// pick the smallest chunks_canonical encoding among the five candidates.
/// `compressed` must be true (the caller only invokes this under a zstd-text
/// preset); the cold single-frame zstd is the never-regress floor.
pub(super) fn choose(texts: &[String]) -> crate::Result<TextChoice> {
    let canonical_raw = encode_chunks_canonical(texts)?;
    // candidate 1: single-frame zstd (the existing form, always the floor).
    let mut best = Candidate::plain(maybe_zstd(
        SECTION_CHUNKS_CANONICAL,
        SectionEncoding::Zstd,
        canonical_raw.clone(),
    )?);

    // candidate 2: txt_streams cold (per-chunk zstd + intpack offset table).
    let streams = encode_txt_streams(texts)?;
    best.consider(Candidate::plain((
        SECTION_CHUNKS_CANONICAL,
        SECTION_ENCODING_TXT_STREAMS,
        streams,
    )));

    // candidate 3: txt_streams + trained dict. the dict is a separate
    // optional section (0x0A), excluded from content_hash; its size counts
    // toward the candidate total so the chooser is honest about the dict cost.
    if let Some(dict) = train_dict(&sorted_unique(texts)) {
        let framed = encode_zstd_dict(texts, &dict)?;
        let total = framed.len() + dict.len();
        best.consider(Candidate {
            canonical: (SECTION_CHUNKS_CANONICAL, SECTION_ENCODING_ZSTD_DICT, framed),
            aux: vec![(SECTION_DICTIONARY, SECTION_ENCODING_RAW, dict)],
            total,
        });
    }

    // candidate 4: txt_streams + fsst (self-contained static symbol table).
    let fsst = encode_fsst(texts)?;
    best.consider(Candidate::plain((
        SECTION_CHUNKS_CANONICAL,
        SECTION_ENCODING_FSST,
        fsst,
    )));

    // candidate 5: dedup + single-frame zstd. the dedup pass runs on the
    // DECOMPRESSED canonical texts (the nix/ipfs order rule), then the UNIQUE
    // pool is zstd-compressed; the back-references live in section 0x0B,
    // excluded from content_hash. only competes when the corpus actually
    // repeats (else unique == texts and it can only lose to candidate 1).
    let d = dedup(texts);
    if d.unique.len() < texts.len() {
        let unique_raw = encode_chunks_canonical(&d.unique)?;
        let (cid, cenc, cbytes) =
            maybe_zstd(SECTION_CHUNKS_CANONICAL, SectionEncoding::Zstd, unique_raw)?;
        let map = encode_dedup_map(&d.back_refs);
        let total = cbytes.len() + map.len();
        best.consider(Candidate {
            canonical: (cid, cenc, cbytes),
            aux: vec![(SECTION_DEDUP_MAP, SECTION_ENCODING_RAW, map)],
            total,
        });
    }

    Ok(TextChoice {
        canonical: best.canonical,
        aux: best.aux,
    })
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
