//! Backward-compat regression: a v0.1-shape `.urna` (raw encoding,
//! float32 embeddings, six required sections, no optional sections)
//! must continue to load in the post-Phase-3 reader without warnings
//! or behavior changes.
//!
//! `golden.rs` covers the structural invariants (size, header bytes,
//! file_hash, content_hash). This file complements with the
//! semantics that Phase 0–3 introduced:
//!
//! - `decoded_section` returns the same bytes as `get_section_data`
//!   for `encoding = 0` (raw). The decoded path is a Cow that should
//!   borrow the physical slice with zero copy when no decompression
//!   is needed.
//! - Every section in a v0.1-shape file declares `encoding = 0`.
//! - The new `validate_embeddings_layout` accepts raw float32
//!   embeddings.
//! - Two consecutive `decoded_section` calls return identical bytes
//!   (no dedup state, no transient buffer).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use urna_format::layout::{
    CANONICAL_SECTIONS, OPTIONAL_SECTIONS, SECTION_EMBEDDINGS, SECTION_ENCODING_RAW,
};
use urna_format::reader::UrnaView;

const GOLDEN: &[u8] = include_bytes!("fixtures/golden_v1_minimal.urna");

#[test]
fn v01_golden_loads_in_v02_reader() {
    UrnaView::from_bytes(GOLDEN).expect("v0.1 golden must load in v0.2 reader");
}

#[test]
fn v01_all_sections_declare_encoding_raw() {
    let view = UrnaView::from_bytes(GOLDEN).unwrap();
    for entry in &view.section_table {
        assert_eq!(
            entry.encoding, SECTION_ENCODING_RAW,
            "v0.1 file should have only raw-encoded sections; section 0x{:02x} declares encoding={}",
            entry.section_id, entry.encoding,
        );
    }
}

#[test]
fn v01_all_six_required_sections_present_no_optional() {
    let view = UrnaView::from_bytes(GOLDEN).unwrap();
    let ids: std::collections::BTreeSet<u32> =
        view.section_table.iter().map(|e| e.section_id).collect();
    for (id, name) in CANONICAL_SECTIONS {
        assert!(
            ids.contains(id),
            "v0.1 must contain required section {} (0x{:02x})",
            name,
            id
        );
    }
    for (id, name) in OPTIONAL_SECTIONS {
        assert!(
            !ids.contains(id),
            "v0.1 must NOT contain optional section {} (0x{:02x})",
            name,
            id
        );
    }
}

#[test]
fn v01_decoded_section_equals_physical_for_raw() {
    let view = UrnaView::from_bytes(GOLDEN).unwrap();
    for (id, _name) in CANONICAL_SECTIONS {
        let physical = view.get_section_data(*id).unwrap();
        let decoded = view.decoded_section(*id).unwrap();
        assert_eq!(
            physical,
            decoded.as_ref(),
            "raw section 0x{:02x}: decoded must equal physical bytes",
            id
        );
    }
}

#[test]
fn v01_decoded_section_is_borrowed_for_raw() {
    let view = UrnaView::from_bytes(GOLDEN).unwrap();
    for (id, _name) in CANONICAL_SECTIONS {
        let decoded = view.decoded_section(*id).unwrap();
        // Cow::Borrowed for raw — no allocation. Cow::Owned would mean
        // we copied bytes to decompress, which only happens for zstd.
        assert!(
            matches!(decoded, std::borrow::Cow::Borrowed(_)),
            "raw section 0x{:02x}: decoded should borrow the mmap, not allocate",
            id
        );
    }
}

#[test]
fn v01_validate_embeddings_layout_accepts_float32() {
    let view = UrnaView::from_bytes(GOLDEN).unwrap();
    // Phase 1 added per-dtype layout checks; v0.1 (float32 raw) must
    // still pass. Equivalent to validate(): we walk values too.
    view.validate_embeddings_values()
        .expect("v0.1 raw float32 embeddings should validate");
    let entry = view
        .section_table
        .iter()
        .find(|e| e.section_id == SECTION_EMBEDDINGS)
        .unwrap();
    assert_eq!(entry.encoding, SECTION_ENCODING_RAW);
    // 1 chunk × dim 4 × 4 bytes = 16 bytes payload.
    assert_eq!(entry.size, 16);
}

#[test]
fn v01_decoded_section_is_idempotent() {
    let view = UrnaView::from_bytes(GOLDEN).unwrap();
    let first = view.decoded_section(SECTION_EMBEDDINGS).unwrap();
    let second = view.decoded_section(SECTION_EMBEDDINGS).unwrap();
    assert_eq!(
        first.as_ref(),
        second.as_ref(),
        "two decoded_section calls must return identical bytes"
    );
}

#[test]
fn v01_search_contract_decodes() {
    // Confirms the manifest <-> contract cross-check passes under
    // the post-Phase-2 manifest validation (allowed dtypes /
    // index_types / rerank policies expanded).
    let view = UrnaView::from_bytes(GOLDEN).unwrap();
    let contract = view.search_contract().expect("v0.1 contract must decode");
    assert_eq!(contract.metric, "ip");
    assert_eq!(contract.score_type, "cosine");
    assert_eq!(contract.index_type, "exact");
    assert_eq!(contract.rerank_policy, "none");
}
