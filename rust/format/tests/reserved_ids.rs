//! Reconciled additive section-id reservation (phase 0, task #02).
//!
//! The four redesign pillars (graph, catalog, multimodal, lens) each
//! independently proposed 0x11. This asserts the single reconciled map is
//! disjoint and content_hash-safe, claimed in ONE pass so no feature
//! branch edits layout/mod.rs into a collision. Every reserved id past the
//! implemented 0x01..=0x08 must be: distinct from every other reserved id,
//! excluded from CANONICAL_SECTIONS (so it never enters content_hash and
//! never invalidates a urna:// citation), and unresolved by section_name
//! until its feature ships.

use urna_format::layout::{
    CANONICAL_SECTIONS, SECTION_BLOB_REFS, SECTION_BLOB_SPAN_OVERLAY, SECTION_BM25_INDEX,
    SECTION_CHUNK_IDS, SECTION_CHUNK_SCALARS, SECTION_CHUNKS_CANONICAL,
    SECTION_CHUNKS_ORIGINAL_SPANS, SECTION_DEDUP_MAP, SECTION_DICTIONARY, SECTION_EDIT_JOURNAL,
    SECTION_EMBEDDINGS, SECTION_EMBEDDINGS_FP, SECTION_GRAPH_ADJACENCY, SECTION_GRAPH_EDGE_PROPS,
    SECTION_GRAPH_ENTITY_MAP, SECTION_GRAPH_NODES, SECTION_HNSW_INDEX, SECTION_PROVENANCE,
    SECTION_REPRO_MANIFEST, SECTION_SEARCH_CONTRACT, SECTION_SPACE_EMBEDDINGS_BASE,
    SECTION_SPACE_EMBEDDINGS_FP_BASE, SECTION_SPACE_TABLE, SECTION_TOKENIZER_MODEL, SPACE_BAND_LEN,
    section_name,
};

fn implemented() -> Vec<u32> {
    vec![
        SECTION_CHUNK_IDS,
        SECTION_CHUNKS_CANONICAL,
        SECTION_CHUNKS_ORIGINAL_SPANS,
        SECTION_EMBEDDINGS,
        SECTION_PROVENANCE,
        SECTION_SEARCH_CONTRACT,
        SECTION_HNSW_INDEX,
        SECTION_BM25_INDEX,
    ]
}

/// Reserved scalar ids (0x09..=0x16) that stay unresolved by section_name
/// until their feature ships. SECTION_DICTIONARY (0x0A) and SECTION_DEDUP_MAP
/// (0x0B) are now EMITTED by the dict/dedup text levers, but stay EXCLUDED
/// from CANONICAL_SECTIONS / content_hash and unresolved by section_name
/// (they are physical-encoding aux sections, never canonical content).
/// SECTION_GRAPH_ADJACENCY (0x0C, G1) is NOT in this list: it now resolves
/// via section_name (OPTIONAL_SECTIONS) yet stays content_hash-excluded; it is
/// covered separately by `graph_adjacency_resolves_but_excluded_from_content_hash`.
/// SECTION_BLOB_REFS (0x14) and SECTION_BLOB_SPAN_OVERLAY (0x16) likewise left
/// this list when the media blob feature shipped; they are covered by
/// `blob_sections_resolve_but_excluded_from_content_hash`.
fn reserved_scalars() -> Vec<u32> {
    vec![
        SECTION_EMBEDDINGS_FP,
        SECTION_DICTIONARY,
        SECTION_DEDUP_MAP,
        SECTION_CHUNK_SCALARS,
        SECTION_TOKENIZER_MODEL,
        SECTION_EDIT_JOURNAL,
        SECTION_REPRO_MANIFEST,
        SECTION_GRAPH_NODES,
        SECTION_GRAPH_EDGE_PROPS,
        SECTION_GRAPH_ENTITY_MAP,
    ]
}

fn space_band() -> Vec<u32> {
    (SECTION_SPACE_EMBEDDINGS_BASE..SECTION_SPACE_EMBEDDINGS_BASE + SPACE_BAND_LEN).collect()
}

fn space_fp_band() -> Vec<u32> {
    (SECTION_SPACE_EMBEDDINGS_FP_BASE..SECTION_SPACE_EMBEDDINGS_FP_BASE + SPACE_BAND_LEN).collect()
}

#[test]
fn all_reserved_bands_are_disjoint() {
    let mut all: Vec<u32> = Vec::new();
    all.extend(implemented());
    all.extend(reserved_scalars());
    // graph_adjacency (0x0C) left reserved_scalars when it shipped (G1), and
    // the blob pair (0x14/0x16) left with the media blob feature; all are
    // still distinct ids that must not collide with any other band.
    all.push(SECTION_GRAPH_ADJACENCY);
    all.push(SECTION_BLOB_REFS);
    all.push(SECTION_BLOB_SPAN_OVERLAY);
    all.push(SECTION_SPACE_TABLE);
    all.extend(space_band());
    all.extend(space_fp_band());

    let mut sorted = all.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        all.len(),
        "every section id (implemented + reserved + per-space bands) must be disjoint"
    );

    // The two per-space bands stay inside their documented ranges.
    assert!(space_band().iter().all(|&x| (0x20..=0x2F).contains(&x)));
    assert!(space_fp_band().iter().all(|&x| (0x30..=0x3F).contains(&x)));
}

#[test]
fn reserved_ids_are_excluded_from_content_hash_and_unresolved() {
    let canonical: Vec<u32> = CANONICAL_SECTIONS.iter().map(|(id, _)| *id).collect();
    let reserved = reserved_scalars();

    for s in reserved {
        assert!(
            !canonical.contains(&s),
            "reserved id {s:#x} must be excluded from content_hash (canonical six only)"
        );
        assert!(
            section_name(s).is_none(),
            "reserved id {s:#x} must not resolve via section_name until its feature ships"
        );
    }
}

#[test]
fn space_sections_resolve_but_excluded_from_content_hash() {
    // the multimodal feature shipped space_table (0x15) and the per-space
    // vector bands: all resolve via section_name (0x15 is an active
    // OPTIONAL_SECTION, the bands resolve through the range check) but MUST
    // stay out of CANONICAL_SECTIONS, so a multimodal corpus keeps the
    // content_hash of its text-only twin.
    assert_eq!(section_name(SECTION_SPACE_TABLE), Some("space_table"));
    let canonical: Vec<u32> = CANONICAL_SECTIONS.iter().map(|(id, _)| *id).collect();
    assert!(!canonical.contains(&SECTION_SPACE_TABLE));
    for s in space_band() {
        assert_eq!(section_name(s), Some("space_embeddings"));
        assert!(
            !canonical.contains(&s),
            "band id {s:#x} must stay excluded from content_hash"
        );
    }
    for s in space_fp_band() {
        assert_eq!(section_name(s), Some("space_embeddings_fp"));
        assert!(
            !canonical.contains(&s),
            "fp band id {s:#x} must stay excluded from content_hash"
        );
    }
}

#[test]
fn graph_adjacency_resolves_but_excluded_from_content_hash() {
    // G1: graph_adjacency (0x0C) now resolves via section_name (it is an active
    // OPTIONAL_SECTION) but MUST stay out of CANONICAL_SECTIONS so it never
    // enters content_hash and never invalidates a urna:// citation.
    assert_eq!(
        section_name(SECTION_GRAPH_ADJACENCY),
        Some("graph_adjacency")
    );
    let canonical: Vec<u32> = CANONICAL_SECTIONS.iter().map(|(id, _)| *id).collect();
    assert!(
        !canonical.contains(&SECTION_GRAPH_ADJACENCY),
        "graph_adjacency must stay excluded from content_hash"
    );
}

#[test]
fn blob_sections_resolve_but_excluded_from_content_hash() {
    // the media blob feature shipped 0x14 (blob_refs) and 0x16
    // (blob_span_overlay): both resolve via section_name (active
    // OPTIONAL_SECTIONS) but MUST stay out of CANONICAL_SECTIONS, so a
    // self-contained media corpus keeps the content_hash of its text twin.
    assert_eq!(section_name(SECTION_BLOB_REFS), Some("blob_refs"));
    assert_eq!(
        section_name(SECTION_BLOB_SPAN_OVERLAY),
        Some("blob_span_overlay")
    );
    let canonical: Vec<u32> = CANONICAL_SECTIONS.iter().map(|(id, _)| *id).collect();
    for s in [SECTION_BLOB_REFS, SECTION_BLOB_SPAN_OVERLAY] {
        assert!(
            !canonical.contains(&s),
            "blob section {s:#x} must stay excluded from content_hash"
        );
    }
}
