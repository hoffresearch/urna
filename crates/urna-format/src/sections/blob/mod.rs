//! blob section codecs (media blob references and the blob-relative span
//! overlay). both are OPTIONAL and EXCLUDED from content_hash, additive
//! within frozen format v1, and never touch the embedding hot path.
//! `refs` is the 0x14 table of content-hash references to source media
//! (self-contained or catalog); `span_overlay` is the 0x16 per-chunk
//! blob-relative span replacement for chunks_original_spans (0x03).

mod data;
mod refs;
mod span_overlay;

pub use data::{
    BLOB_DATA_PAYLOAD_VERSION, BlobDataTable, decode_blob_data_table, encode_blob_data,
};
pub use refs::{BLOB_REFS_PAYLOAD_VERSION, BlobRefRecord, decode_blob_refs, encode_blob_refs};
pub use span_overlay::{
    BLOB_REF_NONE, BLOB_SPAN_OVERLAY_PAYLOAD_VERSION, BlobSpanEntry, decode_blob_span_overlay,
    encode_blob_span_overlay,
};
