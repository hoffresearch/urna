//! .urna binary file layout (v1)
//!
//! File structure:
//! ```text
//! [0 .. 128)                  UrnaHeader (128 bytes)
//! [128 .. 128+count*32)       SectionTable (32 bytes per entry)
//! [manifest_offset .. ...)    Manifest JSON (JCS canonical)
//! [sections ...]              Required sections, each starting at a
//!                             64-byte aligned offset; padding before
//!                             each section is zero and is NOT part of
//!                             the section's checksum
//! [file_size-40 .. file_size) Footer (40 bytes)
//! ```
//!
//! All multi-byte integers are little-endian, unsigned unless noted.

mod footer;
mod header;
mod section_entry;
#[cfg(test)]
mod tests;

pub use footer::UrnaFooter;
pub use header::UrnaHeader;
pub use section_entry::SectionEntry;

pub const URNA_MAGIC: &[u8; 4] = b"URNA";
pub const URNA_VERSION_MAJOR: u16 = 1;
pub const URNA_VERSION_MINOR: u16 = 0;
pub const URNA_HEADER_SIZE: usize = 128;
pub const URNA_SECTION_ENTRY_SIZE: usize = 32;
pub const URNA_FOOTER_SIZE: usize = 40;

/// Every section's `offset` is aligned to this many bytes. Padding
/// before each section is zero and is NOT covered by the section's
/// checksum. Chosen to match common SIMD widths so embeddings can be
/// loaded directly from mmap.
pub const SECTION_ALIGNMENT: u64 = 64;

/// Round `n` up to the next multiple of `a`. `a` must be a power of two.
#[inline]
pub fn align_up(n: u64, a: u64) -> u64 {
    debug_assert!(a.is_power_of_two(), "alignment must be a power of two");
    (n + a - 1) & !(a - 1)
}

/// Section payload encoding.
///
/// - `0 = raw`: payload is the canonical bytes as the reader consumes them.
///   Used for embeddings (float32) and any non-compressed metadata section.
/// - `1 = zstd`: payload is zstd-compressed canonical bytes. Only valid for
///   non-embedding sections; the reader transparently decompresses.
/// - `2 = float16`: payload is `n * dim * 2` bytes of f16 LE; requires the
///   manifest to declare `dtype = "float16"`. Only valid for the embeddings
///   section.
/// - `3 = int8`: payload is the int8 quantized embeddings section (per-vector
///   f32 scales followed by i8 vectors); requires `dtype = "int8"`. Only
///   valid for the embeddings section.
/// - `7 = int4`: payload is the int4 block-64 quantized embeddings section
///   (per-64-dim-group f16 absmax scales followed by packed 4-bit signed
///   codes, two nibbles per byte, codes in `[-7, 7]`); requires `dtype =
///   "int4"` and `embedding_dim` divisible by 64. Only valid for the
///   embeddings section. STORED-PRECISION cosine, scored straight off mmap
///   by the fused dequant+dot kernel (never zstd/shuffle), the first real
///   sub-int8 size lever (~2x over int8).
///
/// A reader rejects unknown encodings with `UnsupportedSectionEncoding`.
pub const SECTION_ENCODING_RAW: u32 = 0;
pub const SECTION_ENCODING_ZSTD: u32 = 1;
pub const SECTION_ENCODING_FLOAT16: u32 = 2;
pub const SECTION_ENCODING_INT8: u32 = 3;

// reserved additive wire encodings. ids 4-255 are reserved within frozen
// format v1 (see doc/arc/arc.yaml). claimed here as named
// constants so each future codec ships as a small additive diff. NOT yet
// implemented: decode_payload rejects them with UnsupportedSectionEncoding
// until their codec module lands, so old and new readers agree.
pub const SECTION_ENCODING_INTPACK: u32 = 4;
pub const SECTION_ENCODING_ZSTD_DICT: u32 = 5;
pub const SECTION_ENCODING_FRONTCODE: u32 = 6;
pub const SECTION_ENCODING_INT4: u32 = 7;
pub const SECTION_ENCODING_RABITQ: u32 = 8;
pub const SECTION_ENCODING_FSST: u32 = 9;
/// `10 = txt_streams`: the chunks_canonical (0x02) section's COMPRESSED
/// form re-laid-out from one concatenated zstd-19 blob into N independently
/// zstd-encoded streams (one per canonical string) behind an intpack offset
/// table (O(1) single-chunk seek/reopen). decodes BYTE-IDENTICALLY to the
/// raw chunks_canonical payload, so content_hash and urna:// citations are
/// unchanged. only valid for non-embedding sections. this is the named
/// prerequisite layout for the dict(5)/fsst(9) text levers; a per-chunk
/// frame loses cross-chunk LZ context (small ratio cost today) but is where
/// a trained dict/fsst can beat one big zstd-19 blob.
pub const SECTION_ENCODING_TXT_STREAMS: u32 = 10;

/// Format version of the binary layout. Bumped when the on-disk
/// container changes (header/footer/section table layout).
pub const URNA_FORMAT_VERSION: u32 = 1;

/// Schema version of the manifest/contract. Bumped when manifest
/// fields or required section semantics change.
pub const URNA_SCHEMA_VERSION: u32 = 1;

// Section IDs. The first six are required (v1 contract); the rest are
// optional and only present when the manifest's `capabilities` declare
// them.
pub const SECTION_CHUNK_IDS: u32 = 0x01;
pub const SECTION_CHUNKS_CANONICAL: u32 = 0x02;
pub const SECTION_CHUNKS_ORIGINAL_SPANS: u32 = 0x03;
pub const SECTION_EMBEDDINGS: u32 = 0x04;
pub const SECTION_PROVENANCE: u32 = 0x05;
pub const SECTION_SEARCH_CONTRACT: u32 = 0x06;
pub const SECTION_HNSW_INDEX: u32 = 0x07;
pub const SECTION_BM25_INDEX: u32 = 0x08;

// additive optional sections 0x09-0x10, all EXCLUDED from content_hash
// (which covers the canonical six only), so adding any of them never
// invalidates a urna:// citation. status per id:
//   0x09 embeddings_fp: READ by the runtime rerank (`rerank::FpSlab`) as the
//        full-precision source for a sub-int8 candidate slab; the writer
//        does not emit it yet (int4 today reranks at stored precision).
//   0x0A dictionary:    WRITTEN + READ, the trained zstd dictionary the
//        `zstd_dict` text codec decodes chunks_canonical against.
//   0x0B dedup_map:     WRITTEN + READ, the ordinal -> unique-string map the
//        `dedup` text codec expands to the byte-identical canonical payload.
//   0x0C graph_adjacency: WRITTEN + READ, in OPTIONAL_SECTIONS below.
//   0x0D-0x10:          claimed names only, no codec and no manifest
//        capability yet; a reader rejects them via validate_encoding.
// 0x0A / 0x0B resolve through the text codec (never by section_name) on
// purpose: they are private companions of chunks_canonical, not user-facing
// sections.
pub const SECTION_EMBEDDINGS_FP: u32 = 0x09;
pub const SECTION_DICTIONARY: u32 = 0x0A;
pub const SECTION_DEDUP_MAP: u32 = 0x0B;
pub const SECTION_GRAPH_ADJACENCY: u32 = 0x0C;
pub const SECTION_CHUNK_SCALARS: u32 = 0x0D;
pub const SECTION_TOKENIZER_MODEL: u32 = 0x0E;
pub const SECTION_EDIT_JOURNAL: u32 = 0x0F;
pub const SECTION_REPRO_MANIFEST: u32 = 0x10;

// additive optional sections past 0x10 (see doc/arc/arc.yaml), one disjoint
// map so no two features claim the same id. ALL are EXCLUDED from
// content_hash, so adding any of them never invalidates a urna:// citation.
//   0x11-0x13 graph nodes / edge props / entity map: claimed names only.
//   0x14 blob_refs, 0x15 space_table, 0x16 blob_span_overlay, 0x17
//   blob_data: WRITTEN + READ, in OPTIONAL_SECTIONS below.
pub const SECTION_GRAPH_NODES: u32 = 0x11;
pub const SECTION_GRAPH_EDGE_PROPS: u32 = 0x12;
pub const SECTION_GRAPH_ENTITY_MAP: u32 = 0x13;
pub const SECTION_BLOB_REFS: u32 = 0x14;
pub const SECTION_SPACE_TABLE: u32 = 0x15;
// blob-relative span overlay: REPLACES the illegal chunks_original_spans
// v1->2 bump (spans is canonical + content-hashed + required). an excluded
// optional section keyed by chunk ordinal, so self_contained and catalog
// twins keep the SAME content_hash and old readers still open the file.
pub const SECTION_BLOB_SPAN_OVERLAY: u32 = 0x16;
// inlined blob bytes: the self-contained twin of the 0x14 catalog. a small
// offset table parallel to blob_refs order, then the raw media bytes. RAW
// (media is already codec-compressed), EXCLUDED from content_hash so the
// embedded and sidecar twins keep the same citations.
pub const SECTION_BLOB_DATA: u32 = 0x17;

// per-space vector bands. each non-text embedding space gets one fixed-
// stride 64-byte-aligned slab in 0x20-0x2F (NEVER zstd, scored by the
// existing simd kernels) and an optional matching fp rerank source in
// 0x30-0x3F. base + SPACE_BAND_LEN define each band; both excluded from
// content_hash. space[0]=text stays the canonical embeddings(0x04).
pub const SECTION_SPACE_EMBEDDINGS_BASE: u32 = 0x20;
pub const SECTION_SPACE_EMBEDDINGS_FP_BASE: u32 = 0x30;
pub const SPACE_BAND_LEN: u32 = 0x10;

/// Canonical order for content_hash. Sorted alphabetically by name; this
/// order is fixed by spec so adding new section IDs cannot reshuffle the
/// hash. Keep this list and section IDs in sync.
pub const CANONICAL_SECTIONS: &[(u32, &str)] = &[
    (SECTION_CHUNK_IDS, "chunk_ids"),
    (SECTION_CHUNKS_CANONICAL, "chunks_canonical"),
    (SECTION_CHUNKS_ORIGINAL_SPANS, "chunks_original_spans"),
    (SECTION_EMBEDDINGS, "embeddings"),
    (SECTION_PROVENANCE, "provenance"),
    (SECTION_SEARCH_CONTRACT, "search_contract"),
];

/// Required sections for a v1 .urna file. A reader rejects any file
/// missing one of these with `MissingRequiredSection`.
pub const REQUIRED_SECTIONS: &[(u32, &str)] = CANONICAL_SECTIONS;

/// Optional sections — present when their corresponding capability is
/// advertised in the manifest. They do NOT participate in content_hash
/// (which is over the canonical six only) so adding an optional section
/// to a corpus does not invalidate citations.
pub const OPTIONAL_SECTIONS: &[(u32, &str)] = &[
    (SECTION_HNSW_INDEX, "hnsw_index"),
    (SECTION_BM25_INDEX, "bm25_index"),
    // graph_adjacency (0x0C, G1): additive chunk-to-chunk csr. resolves via
    // section_name but stays OUT of CANONICAL_SECTIONS (content_hash-excluded).
    (SECTION_GRAPH_ADJACENCY, "graph_adjacency"),
    // blob_refs (0x14) + blob_span_overlay (0x16): content-hash references to
    // source media blobs and the per-chunk blob-relative span overlay. both
    // resolve via section_name and stay OUT of CANONICAL_SECTIONS, so a
    // self-contained media corpus keeps the content_hash of its text twin.
    (SECTION_BLOB_REFS, "blob_refs"),
    (SECTION_BLOB_SPAN_OVERLAY, "blob_span_overlay"),
    // blob_data (0x17): inlined media bytes for the self-contained twin.
    (SECTION_BLOB_DATA, "blob_data"),
    // space_table (0x15): the multimodal per-space directory. resolves via
    // section_name and stays OUT of CANONICAL_SECTIONS; the vector bands it
    // describes (0x20-0x2F / 0x30-0x3F) resolve through the range check in
    // section_name below and are likewise content_hash-excluded.
    (SECTION_SPACE_TABLE, "space_table"),
];

pub fn section_name(id: u32) -> Option<&'static str> {
    if (SECTION_SPACE_EMBEDDINGS_BASE..SECTION_SPACE_EMBEDDINGS_BASE + SPACE_BAND_LEN).contains(&id)
    {
        return Some("space_embeddings");
    }
    if (SECTION_SPACE_EMBEDDINGS_FP_BASE..SECTION_SPACE_EMBEDDINGS_FP_BASE + SPACE_BAND_LEN)
        .contains(&id)
    {
        return Some("space_embeddings_fp");
    }
    CANONICAL_SECTIONS
        .iter()
        .chain(OPTIONAL_SECTIONS.iter())
        .find(|(sid, _)| *sid == id)
        .map(|(_, name)| *name)
}

/// Common prefix for all internal section payloads (12 bytes):
///   u32 version (LE)
///   u64 entry_count (LE)
pub const SECTION_PAYLOAD_PREFIX_SIZE: usize = 12;
pub const SECTION_PAYLOAD_VERSION: u32 = 1;
