//! layout tests, carved out of `layout/mod.rs` so the module stays under
//! the 300-line crate guard (same precedent as `ann/codec/mod.rs`).

use super::*;

#[test]
fn header_size_is_128() {
    assert_eq!(std::mem::size_of::<UrnaHeader>(), 128);
}

#[test]
fn section_entry_size_is_32() {
    assert_eq!(std::mem::size_of::<SectionEntry>(), 32);
}

#[test]
fn footer_size_is_40() {
    assert_eq!(std::mem::size_of::<UrnaFooter>(), 40);
}

#[test]
fn header_roundtrip_checksum() {
    let mut h = UrnaHeader::new(384, 100, 100, 1024, 128, 5, 288, 200);
    assert!(h.validate_checksum().is_ok());
    h.n_chunks = 99;
    assert!(h.validate_checksum().is_err());
}

#[test]
fn canonical_sections_are_alphabetical_by_name() {
    let names: Vec<&str> = CANONICAL_SECTIONS.iter().map(|(_, n)| *n).collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    assert_eq!(names, sorted);
}

#[test]
fn section_name_lookup() {
    assert_eq!(section_name(SECTION_CHUNK_IDS), Some("chunk_ids"));
    assert_eq!(section_name(SECTION_EMBEDDINGS), Some("embeddings"));
    assert_eq!(section_name(0xFFFF), None);
}

#[test]
fn reserved_additive_ids_are_in_range_and_distinct() {
    // reserved wire encodings occupy 4..=10, above the four implemented
    // base ids (raw/zstd/float16/int8 are 0..=3).
    let enc = [
        SECTION_ENCODING_INTPACK,
        SECTION_ENCODING_ZSTD_DICT,
        SECTION_ENCODING_FRONTCODE,
        SECTION_ENCODING_INT4,
        SECTION_ENCODING_RABITQ,
        SECTION_ENCODING_FSST,
        SECTION_ENCODING_TXT_STREAMS,
    ];
    for (i, &e) in enc.iter().enumerate() {
        assert_eq!(e, 4 + i as u32);
        assert!(e > SECTION_ENCODING_INT8);
    }
    // reserved optional sections occupy 0x09..=0x10, above the implemented eight.
    let sec = [
        SECTION_EMBEDDINGS_FP,
        SECTION_DICTIONARY,
        SECTION_DEDUP_MAP,
        SECTION_GRAPH_ADJACENCY,
        SECTION_CHUNK_SCALARS,
        SECTION_TOKENIZER_MODEL,
        SECTION_EDIT_JOURNAL,
        SECTION_REPRO_MANIFEST,
    ];
    for (i, &s) in sec.iter().enumerate() {
        assert_eq!(s, 0x09 + i as u32);
        assert!(s > SECTION_BM25_INDEX);
    }
    // reconciled additive ids past 0x10 are contiguous 0x11..=0x16; the
    // per-space bands start at 0x20 and 0x30 with 16 ids each. the
    // exhaustive disjointness + content_hash-exclusion check lives in
    // tests/reserved_ids.rs.
    let recon = [
        SECTION_GRAPH_NODES,
        SECTION_GRAPH_EDGE_PROPS,
        SECTION_GRAPH_ENTITY_MAP,
        SECTION_BLOB_REFS,
        SECTION_SPACE_TABLE,
        SECTION_BLOB_SPAN_OVERLAY,
    ];
    for (i, &s) in recon.iter().enumerate() {
        assert_eq!(s, 0x11 + i as u32);
    }
    assert_eq!(SECTION_SPACE_EMBEDDINGS_BASE, 0x20);
    assert_eq!(SECTION_SPACE_EMBEDDINGS_FP_BASE, 0x30);
    assert_eq!(SPACE_BAND_LEN, 0x10);
    // reserved sections stay section_name-unresolved until their feature
    // ships. EXCEPTIONS: graph_adjacency (0x0C, G1), the blob pair
    // (0x14/0x16, media blobs) and space_table (0x15, multimodal) resolve
    // yet stay content_hash-excluded (see reserved_ids.rs).
    for &s in sec.iter().chain(recon.iter()) {
        match s {
            SECTION_GRAPH_ADJACENCY => assert_eq!(section_name(s), Some("graph_adjacency")),
            SECTION_BLOB_REFS => assert_eq!(section_name(s), Some("blob_refs")),
            SECTION_BLOB_SPAN_OVERLAY => {
                assert_eq!(section_name(s), Some("blob_span_overlay"))
            }
            SECTION_SPACE_TABLE => assert_eq!(section_name(s), Some("space_table")),
            _ => assert!(section_name(s).is_none()),
        }
    }
    // the per-space bands resolve through the range check.
    assert_eq!(section_name(0x20), Some("space_embeddings"));
    assert_eq!(section_name(0x2F), Some("space_embeddings"));
    assert_eq!(section_name(0x30), Some("space_embeddings_fp"));
    assert_eq!(section_name(0x3F), Some("space_embeddings_fp"));
}
