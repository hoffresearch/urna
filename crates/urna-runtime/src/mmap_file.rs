//! `MmapUrnaFile`: owns the mmap, parses section metadata once, exposes
//! search/inspect entry points. Search math lives in `super::search`;
//! dtype-agnostic vector materialization lives in `super::materialize`.

use std::path::Path;

use memmap2::Mmap;
use urna_format::layout::{
    SECTION_BM25_INDEX, SECTION_CHUNK_IDS, SECTION_CHUNKS_ORIGINAL_SPANS, SECTION_EMBEDDINGS,
    SECTION_EMBEDDINGS_FP, SECTION_GRAPH_ADJACENCY, SECTION_HNSW_INDEX,
};
use urna_format::reader::{UrnaView, validate_slab_values};
use urna_format::sections::{
    BlobRefRecord, OriginalSpan, decode_chunk_ids, decode_chunks_original_spans,
};

use crate::ann;
use crate::bm25;
use crate::dtype::DType;
use crate::error::RuntimeError;
use crate::graph::CsrIndex;
use crate::materialize::PackedVectors;
use crate::rerank::FpSlab;
use crate::simd::{self, SimdBackend};

pub struct MmapUrnaFile {
    pub(crate) _mmap: Mmap,
    pub(crate) embedding_dim: usize,
    pub(crate) n_embeddings: usize,
    pub(crate) dtype: DType,
    /// Byte offset (within the mmap) of the embeddings section payload.
    pub(crate) embeddings_offset: usize,
    /// Total physical bytes of the embeddings section.
    pub(crate) embeddings_size: usize,
    /// Optional full-precision rerank slab (`embeddings_fp`, 0x09): when
    /// present the exact rerank reads it instead of the stored dtype slab.
    pub(crate) embeddings_fp: Option<FpSlab>,
    pub(crate) chunk_ids: Vec<String>,
    pub(crate) spans: Vec<OriginalSpan>,
    pub(crate) embedding_model: String,
    /// Manifest `model_hash`: the granular fingerprint of the model that
    /// produced the corpus embeddings. Exposed so a caller embedding a query
    /// can verify its embedder matches the corpus (the honesty gate).
    pub(crate) model_hash: String,
    pub(crate) file_hash: String,
    pub(crate) content_hash: String,
    /// Optional ANN index. Built from the HNSW section payload at open
    /// time (eager: build cost is paid once, queries get fast path).
    pub(crate) ann_index: Option<ann::HnswIndex>,
    /// Optional BM25 index. Mostly tiny ints; deserialized eagerly.
    pub(crate) bm25_index: Option<bm25::Bm25Index>,
    /// Optional chunk-to-chunk graph (graph_adjacency 0x0C). Opened behind
    /// the manifest `graph_present` capability, like ann/bm25. A candidate
    /// generator only: its frontier feeds the exact rerank, never a score.
    pub(crate) graph_index: Option<CsrIndex>,
    /// Optional blob_refs (0x14) table, opened behind the additive
    /// `blobs_present` capability. content-hash references to the source
    /// media blobs (self-contained or catalog).
    pub(crate) blob_refs: Option<Vec<BlobRefRecord>>,
    /// Optional blob_data (0x17) offset table; blob bytes are sliced
    /// lazily off the mmap by `blob_bytes` (impl in blobs.rs).
    pub(crate) blob_data: Option<crate::blobs::OpenBlobData>,
    /// Optional multimodal spaces (0x15 + bands), opened behind the
    /// additive `supports_multimodal` capability. each space's band slab
    /// is scored by the per-space exact search, never by the text path.
    pub(crate) spaces: Option<Vec<crate::spaces::OpenSpace>>,
    /// What the manifest says the search path is. The runtime honors
    /// this at search time.
    pub(crate) declared_index_type: String,
    pub(crate) declared_score_type: String,
}

impl MmapUrnaFile {
    pub fn open(path: &Path) -> Result<Self, RuntimeError> {
        let file = std::fs::File::open(path)?;
        // SAFETY: `Mmap::map` is unsafe because a concurrent writer could
        // truncate or rewrite the file under the mapping. The runtime opens
        // corpora read-only and never rewrites a `.urna` in place, so only an
        // external live writer can violate this, which is outside the
        // supported model of every mmap-based store (checksums detect
        // corruption after the fact, they do not guard a live writer).
        let mmap = unsafe { Mmap::map(&file)? };
        let view = UrnaView::from_bytes(&mmap)?;
        view.validate_embeddings_values()?;

        let dim = view.header.embedding_dim as usize;
        let n = view.header.n_embeddings as usize;
        let dtype = DType::from_str(&view.manifest.dtype)?;

        let entry = view.entry(SECTION_EMBEDDINGS)?;
        let embeddings_offset = entry.offset as usize;
        let embeddings_size = entry.size as usize;

        // Optional full-precision rerank slab (0x09); see rerank::FpSlab.
        let embeddings_fp = FpSlab::detect(&view.section_table, n, dim)?;
        if embeddings_fp.is_some() {
            // the rerank scores this slab, so it passes the same NaN gate.
            let e = view.entry(SECTION_EMBEDDINGS_FP)?;
            let fp = view.get_section_data(SECTION_EMBEDDINGS_FP)?;
            validate_slab_values(e.encoding, fp, n, dim)?;
        }

        // Decoded chunk_ids / spans (handles zstd transparently).
        let chunk_ids = decode_chunk_ids(&view.decoded_section(SECTION_CHUNK_IDS)?, n)?;
        let mut spans =
            decode_chunks_original_spans(&view.decoded_section(SECTION_CHUNKS_ORIGINAL_SPANS)?, n)?;

        // Optional ANN section. Materialize f32 vectors from the
        // embeddings section so the graph can compute distances at
        // search time independent of the on-disk dtype.
        let ann_index = if view
            .section_table
            .iter()
            .any(|e| e.section_id == SECTION_HNSW_INDEX)
        {
            let bytes = view.decoded_section(SECTION_HNSW_INDEX)?;
            let mut idx = ann::HnswIndex::from_bytes(&bytes, n, dim)?;
            let emb_bytes = view.get_section_data(SECTION_EMBEDDINGS)?;
            // Keep the vectors in their on-disk packing (no n*dim*4 f32
            // expansion): the graph decodes one row at a time on demand.
            let store = PackedVectors::from_section(&view.manifest.dtype, emb_bytes, n, dim)?;
            idx.attach_store(store);
            Some(idx)
        } else {
            None
        };

        let bm25_index = if view
            .section_table
            .iter()
            .any(|e| e.section_id == SECTION_BM25_INDEX)
        {
            let bytes = view.decoded_section(SECTION_BM25_INDEX)?;
            Some(bm25::Bm25Index::from_bytes(&bytes)?)
        } else {
            None
        };

        // Optional graph_adjacency section (0x0C). gated behind the additive
        // `graph_present` capability (delivered via Option<CapabilitiesExt>),
        // mirroring how ann/bm25 are opened. raw payload; the csr bitpacks its
        // own integer columns with intpack internally.
        let graph_present = view
            .manifest
            .capabilities_ext
            .as_ref()
            .and_then(|e| e.graph_present)
            .unwrap_or(false);
        let graph_index = if graph_present
            && view
                .section_table
                .iter()
                .any(|e| e.section_id == SECTION_GRAPH_ADJACENCY)
        {
            let bytes = view.decoded_section(SECTION_GRAPH_ADJACENCY)?;
            Some(CsrIndex::from_bytes(&bytes, n)?)
        } else {
            None
        };

        // Optional blob pair (0x14/0x16): opens behind the additive
        // `blobs_present` capability; the overlay rewrites blob-pointing
        // spans in place. see blobs.rs.
        let blob_refs = crate::blobs::open_blob_sections(&view, &mut spans)?;
        let blob_data = match &blob_refs {
            Some(refs) => crate::blobs::open_blob_data(&view, refs.len())?,
            None => None,
        };

        // Optional multimodal spaces (0x15 + bands): opens behind the
        // additive `supports_multimodal` capability. see spaces.rs.
        let spaces = crate::spaces::open_space_sections(&view)?;
        let embedding_model = view.manifest.embedding_model.clone();
        let model_hash = view.manifest.model_hash.clone();
        let declared_index_type = view.manifest.index_type.clone();
        let declared_score_type = view.manifest.score_type.clone();
        let file_hash = view.file_hash_hex();
        let content_hash = view.content_hash_hex()?;
        drop(view);

        Ok(Self {
            _mmap: mmap,
            embedding_dim: dim,
            n_embeddings: n,
            dtype,
            embeddings_offset,
            embeddings_size,
            embeddings_fp,
            chunk_ids,
            spans,
            embedding_model,
            model_hash,
            file_hash,
            content_hash,
            ann_index,
            bm25_index,
            graph_index,
            blob_refs,
            blob_data,
            spaces,
            declared_index_type,
            declared_score_type,
        })
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
    pub fn n_embeddings(&self) -> usize {
        self.n_embeddings
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn file_hash(&self) -> &str {
        &self.file_hash
    }
    pub fn content_hash(&self) -> &str {
        &self.content_hash
    }
    pub fn simd_backend(&self) -> SimdBackend {
        simd::detect_backend()
    }
    pub fn declared_index_type(&self) -> &str {
        &self.declared_index_type
    }
    /// Manifest `model_hash` (the model fingerprint the corpus was built
    /// with). Callers embed the query with their own model and compare.
    pub fn model_hash(&self) -> &str {
        &self.model_hash
    }
    pub fn declared_score_type(&self) -> &str {
        &self.declared_score_type
    }
    pub fn has_ann(&self) -> bool {
        self.ann_index.is_some()
    }
    pub fn has_bm25(&self) -> bool {
        self.bm25_index.is_some()
    }
    pub fn has_graph(&self) -> bool {
        self.graph_index.is_some()
    }
    pub fn has_blobs(&self) -> bool {
        self.blob_refs.is_some()
    }
    /// The blob_refs (0x14) table, when the file declares `blobs_present`:
    /// content-hash references to the source media blobs, in table order
    /// (the overlay's `blob_ref_index` addresses this slice).
    pub fn blob_refs(&self) -> Option<&[BlobRefRecord]> {
        self.blob_refs.as_deref()
    }
    pub fn has_spaces(&self) -> bool {
        self.spaces.is_some()
    }
    /// Names of the multimodal spaces listed in the space_table (empty
    /// when the file has no `supports_multimodal` capability).
    pub fn space_names(&self) -> Vec<&str> {
        self.spaces
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .map(|s| s.entry.name.as_str())
            .collect()
    }
    /// One opened space by name, with its band slab sliced off the mmap.
    pub(crate) fn space(&self, name: &str) -> Option<(&crate::spaces::OpenSpace, &[u8])> {
        let s = self
            .spaces
            .as_deref()?
            .iter()
            .find(|s| s.entry.name == name)?;
        Some((s, &self._mmap[s.offset..s.offset + s.size]))
    }

    /// Re-run all reader-side validation. The file was already
    /// validated at `open()` time, but callers can invoke this
    /// explicitly to detect tampering after the fact (e.g. the mmap
    /// pages got swapped under the runtime).
    pub fn revalidate(&self) -> Result<(), RuntimeError> {
        let view = UrnaView::from_bytes(&self._mmap)?;
        view.validate_embeddings_values()?;
        let _ = view.search_contract()?;
        Ok(())
    }

    pub(crate) fn embeddings_bytes(&self) -> &[u8] {
        &self._mmap[self.embeddings_offset..self.embeddings_offset + self.embeddings_size]
    }

    /// The full-precision rerank slab + its dtype, if an `embeddings_fp`
    /// (0x09) section is present. The rerank handle prefers this over the
    /// stored dtype slab so a sub-int8 corpus still returns a real cosine.
    pub(crate) fn embeddings_fp_slab(&self) -> Option<(&[u8], DType)> {
        self.embeddings_fp
            .map(|fp| (&self._mmap[fp.offset..fp.offset + fp.size], fp.dtype))
    }
}
