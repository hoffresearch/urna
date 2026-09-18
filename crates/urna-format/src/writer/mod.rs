//! Builder for .urna files.
//!
//! The builder owns structured chunk inputs and emits all six required
//! sections plus the manifest. Output is fully deterministic given the
//! same inputs and encoding choices.
//!
//! Encoding choices:
//!
//! - `SectionEncoding::Raw` (default) — sections stored verbatim.
//! - `SectionEncoding::Zstd` — text-heavy sections (canonical/spans/
//!   provenance/contract) are zstd-compressed on disk; the reader
//!   decompresses transparently.
//! - `EmbeddingDType::Float32 | Float16 | Int8` — controls the on-disk
//!   representation of the embeddings section. The runtime always
//!   accumulates dot products in f32 regardless of dtype.

mod build;
mod encoding_choice;
mod payload;
#[cfg(test)]
mod tests;
mod text_codec;

pub use encoding_choice::{EmbeddingDType, REPRODUCIBLE_CREATED, SectionEncoding};

use crate::chunk::ChunkInput;
use crate::manifest::Manifest;
use std::path::Path;

/// High-level builder. Accepts canonical chunks plus an optional provenance
/// blob. Computes `chunk_id`s, lays out the file, writes deterministic bytes.
pub struct UrnaFileBuilder {
    pub(super) manifest: Manifest,
    pub(super) chunks: Vec<ChunkInput>,
    pub(super) provenance: serde_json::Value,
    pub(super) reproducible: bool,
    pub(super) text_encoding: SectionEncoding,
    pub(super) dtype: EmbeddingDType,
    /// Optional HNSW index payload, fully encoded by the caller. The
    /// builder doesn't know how to build an HNSW graph itself — that's
    /// the runtime's job.
    pub(super) hnsw_index: Option<Vec<u8>>,
    pub(super) bm25_index: Option<Vec<u8>>,
    /// Optional graph_adjacency (0x0C) csr payload, fully encoded by the
    /// caller (chunk-to-chunk edges). additive, excluded from content_hash.
    pub(super) graph_adjacency: Option<Vec<u8>>,
    /// Optional blob_refs (0x14) payload, fully encoded by the caller
    /// (`encode_blob_refs`). additive, excluded from content_hash.
    pub(super) blob_refs: Option<Vec<u8>>,
    /// Optional blob_data (0x17) payload, fully encoded by the caller
    /// (`encode_blob_data`). additive, excluded from content_hash.
    pub(super) blob_data: Option<Vec<u8>>,
    /// Optional blob_span_overlay (0x16) payload, fully encoded by the
    /// caller (`encode_blob_span_overlay`). additive, excluded from
    /// content_hash; the runtime prefers it over 0x03 spans for cite.
    pub(super) blob_span_overlay: Option<Vec<u8>>,
    /// Optional space_table (0x15) payload, fully encoded by the caller
    /// (`encode_space_table`). additive, excluded from content_hash.
    pub(super) space_table: Option<Vec<u8>>,
    /// Per-space vector band payloads: (band section id, encoding,
    /// payload). section id is 0x20 + space_index; encoding is one of the
    /// dtype encodings (raw/f16/int8/int4), never zstd.
    pub(super) space_bands: Vec<(u32, u32, Vec<u8>)>,
}

impl UrnaFileBuilder {
    pub fn new(manifest: Manifest) -> Self {
        Self {
            manifest,
            chunks: Vec::new(),
            provenance: serde_json::json!({}),
            reproducible: false,
            text_encoding: SectionEncoding::Raw,
            dtype: EmbeddingDType::Float32,
            hnsw_index: None,
            bm25_index: None,
            graph_adjacency: None,
            blob_refs: None,
            blob_data: None,
            blob_span_overlay: None,
            space_table: None,
            space_bands: Vec::new(),
        }
    }

    pub fn add_chunk(mut self, c: ChunkInput) -> Self {
        self.chunks.push(c);
        self
    }

    pub fn add_chunks<I: IntoIterator<Item = ChunkInput>>(mut self, chunks: I) -> Self {
        self.chunks.extend(chunks);
        self
    }

    pub fn with_provenance(mut self, v: serde_json::Value) -> Self {
        self.provenance = v;
        self
    }

    /// Reproducible build mode. When enabled, the writer overrides the
    /// manifest's `created` timestamp to `REPRODUCIBLE_CREATED` so that
    /// two builds with identical inputs produce byte-identical output.
    /// Provenance JSON is not rewritten — callers are responsible for
    /// keeping provenance deterministic if they want bit-for-bit equality.
    pub fn reproducible(mut self, on: bool) -> Self {
        self.reproducible = on;
        self
    }

    /// Encoding for text-heavy sections (chunks_canonical, original_spans,
    /// provenance, search_contract). `Zstd` shrinks PT-BR text by ~3-5×
    /// in practice. chunk_ids stays raw because it is high-entropy and
    /// almost incompressible.
    pub fn text_encoding(mut self, enc: SectionEncoding) -> Self {
        self.text_encoding = enc;
        self
    }

    /// Embedding dtype + on-disk encoding. Mutates the manifest's dtype
    /// to match. Quantized variants (`Float16`, `Int8`) are lossy; the
    /// runtime always accumulates dot products in f32.
    pub fn embedding_dtype(mut self, dt: EmbeddingDType) -> Self {
        self.dtype = dt;
        self.manifest.dtype = dt.manifest_str().to_string();
        self
    }

    /// Attach an HNSW index payload (already encoded by `urna-runtime`).
    /// Sets `index_type=hnsw`, `rerank_policy=exact`, `supports_ann=true`.
    pub fn hnsw_index(mut self, payload: Vec<u8>) -> Self {
        self.hnsw_index = Some(payload);
        self.manifest.index_type = "hnsw".into();
        self.manifest.rerank_policy = "exact".into();
        self.manifest.capabilities.supports_ann = true;
        self
    }

    /// Attach a BM25 index payload.
    pub fn bm25_index(mut self, payload: Vec<u8>) -> Self {
        self.bm25_index = Some(payload);
        self.manifest.capabilities.supports_bm25 = true;
        self
    }

    /// Attach a graph_adjacency (0x0C) csr payload (chunk-to-chunk edges,
    /// already encoded by `encode_graph_adjacency`). Sets the additive
    /// `capabilities_ext.graph_present = Some(true)` flag so the runtime opens
    /// the section behind a capability, exactly like hnsw/bm25. The section is
    /// EXCLUDED from content_hash, so adding a graph never invalidates a
    /// urna:// citation. Emitted raw, like hnsw.
    pub fn graph_adjacency(mut self, payload: Vec<u8>) -> Self {
        self.graph_adjacency = Some(payload);
        let mut ext = self.manifest.capabilities_ext.take().unwrap_or_default();
        ext.graph_present = Some(true);
        self.manifest.capabilities_ext = Some(ext);
        self
    }

    /// Attach a blob_refs (0x14) payload (already encoded by
    /// `encode_blob_refs`): the content-hash reference table for source
    /// media blobs. Sets the additive `capabilities_ext.blobs_present`
    /// flag so the runtime opens the section behind a capability, exactly
    /// like hnsw/bm25/graph. The section is EXCLUDED from content_hash, so
    /// a self-contained media corpus keeps the content_hash (and the
    /// urna:// citations) of its text-only twin. Emitted raw, like hnsw.
    pub fn blob_refs(mut self, payload: Vec<u8>) -> Self {
        self.blob_refs = Some(payload);
        let mut ext = self.manifest.capabilities_ext.take().unwrap_or_default();
        ext.blobs_present = Some(true);
        self.manifest.capabilities_ext = Some(ext);
        self
    }

    /// Attach a blob_data (0x17) payload (already encoded by
    /// `encode_blob_data`): the inlined media bytes whose offset table
    /// parallels the 0x14 record order, making the file self-contained
    /// with no media sidecar. additive and EXCLUDED from content_hash;
    /// implies `blobs_present` (the table indexes the 0x14 records).
    pub fn blob_data(mut self, payload: Vec<u8>) -> Self {
        self.blob_data = Some(payload);
        let mut ext = self.manifest.capabilities_ext.take().unwrap_or_default();
        ext.blobs_present = Some(true);
        self.manifest.capabilities_ext = Some(ext);
        self
    }

    /// Attach a blob_span_overlay (0x16) payload (already encoded by
    /// `encode_blob_span_overlay`): per-chunk blob-relative spans that the
    /// runtime prefers over chunks_original_spans (0x03) for cite/retrieve,
    /// so 0x03 never has to carry an ordinal disguised as a byte range.
    /// additive and EXCLUDED from content_hash; also implies
    /// `blobs_present` (the overlay indexes the 0x14 table).
    pub fn blob_span_overlay(mut self, payload: Vec<u8>) -> Self {
        self.blob_span_overlay = Some(payload);
        let mut ext = self.manifest.capabilities_ext.take().unwrap_or_default();
        ext.blobs_present = Some(true);
        self.manifest.capabilities_ext = Some(ext);
        self
    }

    /// Attach a space_table (0x15) payload (already encoded by
    /// `encode_space_table`): the multimodal per-space directory. Sets the
    /// additive `capabilities_ext.supports_multimodal` flag so the runtime
    /// opens the space bands behind a capability, exactly like the other
    /// optional sections. EXCLUDED from content_hash, so a multimodal
    /// corpus keeps the citations of its text-only twin. Emitted raw.
    pub fn space_table(mut self, payload: Vec<u8>) -> Self {
        self.space_table = Some(payload);
        let mut ext = self.manifest.capabilities_ext.take().unwrap_or_default();
        ext.supports_multimodal = Some(true);
        self.manifest.capabilities_ext = Some(ext);
        self
    }

    /// Attach one per-space vector band: the fixed-stride slab at section
    /// 0x20 + `space_index`. `encoding` is the dtype encoding
    /// (`SECTION_ENCODING_RAW` for f32, or float16/int8/int4), NEVER zstd
    /// (the reader rejects it for band ids). EXCLUDED from content_hash;
    /// also implies `supports_multimodal` (a band without a table is still
    /// a multimodal file, though the runtime only opens listed spaces).
    pub fn space_band(mut self, space_index: u8, encoding: u32, payload: Vec<u8>) -> Self {
        let band_id = crate::layout::SECTION_SPACE_EMBEDDINGS_BASE + space_index as u32;
        self.space_bands.push((band_id, encoding, payload));
        let mut ext = self.manifest.capabilities_ext.take().unwrap_or_default();
        ext.supports_multimodal = Some(true);
        self.manifest.capabilities_ext = Some(ext);
        self
    }

    /// Mark the search path as hybrid (BM25 + cosine). Requires both an
    /// HNSW or exact path and a BM25 index. Caller is responsible for
    /// declaring `score_type=hybrid_rrf` if they want that, otherwise
    /// score_type stays "cosine".
    pub fn hybrid(mut self) -> Self {
        self.manifest.index_type = "hybrid".into();
        self.manifest.rerank_policy = "exact".into();
        self.manifest.score_type = "hybrid_rrf".into();
        self.manifest.capabilities.supports_bm25 = true;
        self
    }

    pub fn write_to_path(self, path: impl AsRef<Path>) -> crate::Result<()> {
        let buf = self.build_bytes()?;
        std::fs::write(path, buf)?;
        Ok(())
    }
}
