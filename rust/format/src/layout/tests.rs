//! layout tests, carved out of `layout/mod.rs` so the module holds the
//! layout alone (same precedent as `ann/codec/mod.rs`).

use super::*;

// the records are written field by field; these pin every field to its
// on-disk offset and check that parsing reads back what was written.

#[test]
fn header_fields_sit_at_their_offsets() {
    let mut h = UrnaHeader::new(384, 100, 101, 1024, 128, 5, 288, 200);
    h.flags = 0x0102_0304;
    h.reserved[47] = 0xEE;
    let b = h.to_bytes();
    assert_eq!(b.len(), URNA_HEADER_SIZE);
    assert_eq!(&b[0..4], URNA_MAGIC);
    assert_eq!(b[4..6], URNA_VERSION_MAJOR.to_le_bytes());
    assert_eq!(b[6..8], URNA_VERSION_MINOR.to_le_bytes());
    assert_eq!(b[8..12], 0x0102_0304u32.to_le_bytes());
    assert_eq!(b[12..16], 384u32.to_le_bytes());
    assert_eq!(b[16..24], 100u64.to_le_bytes());
    assert_eq!(b[24..32], 101u64.to_le_bytes());
    assert_eq!(b[32..40], 1024u64.to_le_bytes());
    assert_eq!(b[40..48], 128u64.to_le_bytes());
    assert_eq!(b[48..56], 5u64.to_le_bytes());
    assert_eq!(b[56..64], 288u64.to_le_bytes());
    assert_eq!(b[64..72], 200u64.to_le_bytes());
    assert_eq!(b[72..80], h.header_checksum);
    assert_eq!(b[127], 0xEE);
    assert_eq!(UrnaHeader::from_bytes(&b).unwrap(), h);
    assert!(UrnaHeader::from_bytes(&b[..127]).is_err());
}

#[test]
fn section_entry_fields_sit_at_their_offsets() {
    let mut e = SectionEntry::new(0x0D, 4096, 77);
    e.encoding = SECTION_ENCODING_ZSTD;
    e.checksum = [1, 2, 3, 4, 5, 6, 7, 8];
    let b = e.to_bytes();
    assert_eq!(b.len(), URNA_SECTION_ENTRY_SIZE);
    assert_eq!(b[0..4], 0x0Du32.to_le_bytes());
    assert_eq!(b[4..8], SECTION_ENCODING_ZSTD.to_le_bytes());
    assert_eq!(b[8..16], 4096u64.to_le_bytes());
    assert_eq!(b[16..24], 77u64.to_le_bytes());
    assert_eq!(b[24..32], [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(SectionEntry::from_bytes(&b).unwrap(), e);
    assert!(SectionEntry::from_bytes(&b[..31]).is_err());
}

#[test]
fn footer_fields_sit_at_their_offsets() {
    let f = UrnaFooter::new([9; 32]);
    let b = f.to_bytes();
    assert_eq!(b.len(), URNA_FOOTER_SIZE);
    assert_eq!(b[0..8], (URNA_FOOTER_SIZE as u64).to_le_bytes());
    assert_eq!(b[8..40], [9; 32]);
    assert_eq!(UrnaFooter::from_bytes(&b).unwrap(), f);
    assert!(UrnaFooter::from_bytes(&b[..39]).is_err());
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
            SECTION_CHUNK_SCALARS => assert_eq!(section_name(s), Some("chunk_scalars")),
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
