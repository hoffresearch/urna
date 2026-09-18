//! content-hash dedup of canonical texts BEFORE zstd (the dedup-map
//! optional section, id 0x0B). a build-time pass over the DECOMPRESSED
//! canonical texts (the load-bearing nix/ipfs order rule: dedup on
//! decompressed bytes, compress the unique pool AFTERWARD) collapses
//! repeated chunks to one stored copy plus a u32 back-reference per chunk.
//!
//! the unique pool is stored in `chunks_canonical` (0x02) under whatever
//! text codec wins (raw / zstd / dict / fsst); the back-reference array
//! lives in `SECTION_DEDUP_MAP` (0x0B), excluded from `content_hash`, so it
//! never moves a `urna://` citation. [`expand`] re-expands the unique pool
//! through the back-references to the EXACT original canonical byte stream
//! before `content_hash` sees it, so a deduped file and its non-deduped twin
//! share the same `content_hash`.
//!
//! draws from nixos/nix store-optimise (content-hash equality dedup, ~25-35%
//! on redundant corpora before any entropy coder) and ipfs/kubo block-level
//! dedup (identical content collapses to one stored block); the "dedup on
//! decompressed bytes, compress afterward" invariant is the nix-casync/tvix
//! lesson encoded as a guardrail.

use super::intpack::{IntpackReader, pack_u64s};
use crate::error::UrnaError;
use crate::layout::SECTION_DEDUP_MAP;
use std::collections::HashMap;

/// dedup-map payload version byte (leads the back-reference array).
pub const DEDUP_MAP_V1: u8 = 0;

fn malformed(reason: impl Into<String>) -> UrnaError {
    UrnaError::MalformedSectionPayload {
        section_id: SECTION_DEDUP_MAP,
        reason: reason.into(),
    }
}

/// the result of a dedup pass: the first-seen unique texts (in first-seen
/// order, deterministic) and a per-chunk back-reference into that pool.
pub struct Deduped {
    pub unique: Vec<String>,
    pub back_refs: Vec<u32>,
}

/// run the first-seen dedup pass over `texts` (the decompressed canonical
/// strings, in chunk order). deterministic: a `HashMap` keyed by the text
/// only decides membership, while first-seen ORDER is driven by the input
/// sequence, so two builds over the same corpus match exactly.
pub fn dedup(texts: &[String]) -> Deduped {
    let mut index: HashMap<&str, u32> = HashMap::with_capacity(texts.len());
    let mut unique: Vec<String> = Vec::new();
    let mut back_refs: Vec<u32> = Vec::with_capacity(texts.len());
    for t in texts {
        match index.get(t.as_str()) {
            Some(&i) => back_refs.push(i),
            None => {
                let i = unique.len() as u32;
                index.insert(t.as_str(), i);
                unique.push(t.clone());
                back_refs.push(i);
            }
        }
    }
    Deduped { unique, back_refs }
}

/// serialize the back-reference array: a version byte then an intpack
/// (encoding-id-4 primitive, reused) packing of the u32 refs as u64. a pure
/// function of the refs, so two builds are byte-identical.
pub fn encode_map(back_refs: &[u32]) -> Vec<u8> {
    let as_u64: Vec<u64> = back_refs.iter().map(|&r| r as u64).collect();
    let packed = pack_u64s(&as_u64);
    let mut out = Vec::with_capacity(1 + packed.len());
    out.push(DEDUP_MAP_V1);
    out.extend_from_slice(&packed);
    out
}

/// parse a dedup-map payload back to the back-reference array, bounds-checked.
pub fn decode_map(bytes: &[u8]) -> crate::Result<Vec<u32>> {
    let (kind, rest) = bytes
        .split_first()
        .ok_or_else(|| malformed("dedup_map: empty"))?;
    if *kind != DEDUP_MAP_V1 {
        return Err(malformed(format!("dedup_map: unknown kind {}", *kind)));
    }
    let reader = IntpackReader::parse(rest)?;
    let mut refs = Vec::with_capacity(reader.len().min(1 << 20));
    for i in 0..reader.len() {
        let v = reader.get(i)?;
        let r = u32::try_from(v).map_err(|_| malformed("dedup_map: back-ref exceeds u32"))?;
        refs.push(r);
    }
    Ok(refs)
}

/// re-expand the unique pool through the back-references to the full ordered
/// list of canonical texts, byte-identical to the original. every ref must
/// index a real unique entry; a hostile map errors, never panics.
pub fn expand(unique: &[String], back_refs: &[u32]) -> crate::Result<Vec<String>> {
    let mut out = Vec::with_capacity(back_refs.len());
    for &r in back_refs {
        let s = unique
            .get(r as usize)
            .ok_or_else(|| malformed("dedup_map: back-ref out of unique-pool range"))?;
        out.push(s.clone());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn dedup_then_expand_roundtrips() {
        for corpus in [
            texts(&[]),
            texts(&["a", "b", "c"]),                // all unique
            texts(&["x", "x", "x", "x"]),           // all duplicate
            texts(&["a", "b", "a", "c", "b", "a"]), // mixed
        ] {
            let d = dedup(&corpus);
            let back = expand(&d.unique, &d.back_refs).unwrap();
            assert_eq!(back, corpus, "expand must rebuild the original order");
            // the map round-trips through its serialized form too.
            let blob = encode_map(&d.back_refs);
            assert_eq!(decode_map(&blob).unwrap(), d.back_refs);
        }
    }

    #[test]
    fn all_duplicate_collapses_to_one() {
        let d = dedup(&texts(&["same", "same", "same"]));
        assert_eq!(d.unique.len(), 1);
        assert_eq!(d.back_refs, vec![0, 0, 0]);
    }

    #[test]
    fn determinism_two_passes_identical() {
        let corpus = texts(&["p", "q", "p", "r", "q"]);
        let a = dedup(&corpus);
        let b = dedup(&corpus);
        assert_eq!(a.unique, b.unique);
        assert_eq!(a.back_refs, b.back_refs);
        assert_eq!(encode_map(&a.back_refs), encode_map(&b.back_refs));
    }
}
