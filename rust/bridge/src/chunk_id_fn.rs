//! `chunk_id()` Python helper. Mirrors the Rust `chunk_id` derivation
//! so Python ingestion can dedupe before passing chunks to `build()`.

use pyo3::prelude::*;

/// Compute the canonical chunk_id for inputs that match the writer's
/// derivation. Useful for the Python-side ingestion to deduplicate chunks
/// before passing them in.
#[pyfunction]
pub fn chunk_id(
    canonical_text: &str,
    source_uri: &str,
    byte_start: u64,
    byte_end: u64,
    chunker_version: &str,
) -> String {
    urna_format::chunk_id(
        canonical_text,
        source_uri,
        byte_start,
        byte_end,
        chunker_version,
    )
}
