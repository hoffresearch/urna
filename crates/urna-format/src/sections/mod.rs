//! Section payload formats (v1).
//!
//! Each non-binary section starts with a 12-byte header:
//!
//! ```text
//! [0..4)   u32 version  (LE) — currently 1
//! [4..12)  u64 count    (LE) — number of entries
//! ```
//!
//! Then a payload that depends on the section. Embeddings use a different
//! shape (no per-entry header — dim/count come from the file header).
//!
//! All multi-byte integers are little-endian. Strings are raw UTF-8 bytes
//! prefixed by a u32 length (no NUL terminators).

pub mod blob;
mod canonical;
mod chunk_ids;
mod codec;
mod contract;
pub mod graph;
mod provenance;
mod space_table;
mod spans;

/// `intpack` (encoding id 4) repack kinds. the kind byte leads the
/// packed payload so the wire codec can rebuild the exact canonical
/// bytes of the section it was packed from, keeping `content_hash`
/// stable. canonical sections are never version-bumped; this is a wire
/// encoding, not a payload-format change.
pub const REPACK_KIND_CHUNK_IDS: u8 = 0;
pub const REPACK_KIND_SPANS: u8 = 1;

pub use blob::{
    BLOB_DATA_PAYLOAD_VERSION, BLOB_REF_NONE, BLOB_REFS_PAYLOAD_VERSION,
    BLOB_SPAN_OVERLAY_PAYLOAD_VERSION, BlobDataTable, BlobRefRecord, BlobSpanEntry,
    decode_blob_data_table, decode_blob_refs, decode_blob_span_overlay, encode_blob_data,
    encode_blob_refs, encode_blob_span_overlay,
};
pub use canonical::{decode_chunks_canonical, encode_chunks_canonical};
pub use chunk_ids::{
    decode_chunk_ids, decode_chunk_ids_intpack, encode_chunk_ids, encode_chunk_ids_intpack,
};
pub use contract::{SearchContract, decode_search_contract, encode_search_contract};
pub use provenance::{decode_provenance, encode_provenance};
pub use space_table::{
    SPACE_DTYPE_F16, SPACE_DTYPE_F32, SPACE_DTYPE_I4, SPACE_DTYPE_I8, SPACE_TABLE_PAYLOAD_VERSION,
    SpaceEntry, decode_space_table, encode_space_table,
};
pub use spans::{
    OriginalSpan, decode_chunks_original_spans, decode_chunks_original_spans_intpack,
    encode_chunks_original_spans, encode_chunks_original_spans_intpack,
};

/// decode a `txt_streams` payload (the full wire bytes, including the
/// leading kind/version byte) back to the canonical `chunks_canonical`
/// section payload. byte-identical to [`encode_chunks_canonical`], so
/// `content_hash` is unchanged. dispatched by `encoding::decode_payload`
/// for encoding id 10 (parallel to [`decode_intpack_repack`]).
pub fn decode_txt_streams(bytes: &[u8]) -> crate::Result<Vec<u8>> {
    crate::encoding::decode_txt_streams_payload(bytes)
}

/// decode an `intpack` repack payload (the full wire bytes, including the
/// leading kind byte) back to the canonical section payload it was packed
/// from. dispatched by `encoding::decode_payload` for encoding id 4.
pub fn decode_intpack_repack(bytes: &[u8]) -> crate::Result<Vec<u8>> {
    let (kind, rest) =
        bytes
            .split_first()
            .ok_or_else(|| crate::error::UrnaError::MalformedSectionPayload {
                section_id: 0,
                reason: "intpack repack: empty payload".into(),
            })?;
    match *kind {
        REPACK_KIND_CHUNK_IDS => decode_chunk_ids_intpack(rest),
        REPACK_KIND_SPANS => decode_chunks_original_spans_intpack(rest),
        other => Err(crate::error::UrnaError::MalformedSectionPayload {
            section_id: 0,
            reason: format!("intpack repack: unknown kind {}", other),
        }),
    }
}
