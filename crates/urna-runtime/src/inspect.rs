//! `MmapUrnaFile::inspect_json`: the `urna inspect --json` document. Kept out
//! of `mmap_file.rs` so that file stays under the 300-line crate guard; this
//! is pure presentation (re-parse the mmap, dump header + section table +
//! manifest + hashes + simd backend), no search math.

use urna_format::UrnaError;
use urna_format::layout::SECTION_CHUNKS_CANONICAL;
use urna_format::reader::UrnaView;
use urna_format::sections::decode_chunks_canonical;

use crate::error::RuntimeError;
use crate::mmap_file::MmapUrnaFile;

impl MmapUrnaFile {
    /// Decode the stored canonical text for every chunk, in file order. this
    /// is the TIER-1 citation text, the same bytes `urna cite` returns, NOT
    /// the original source bytes. re-parses the mmap and decodes the
    /// chunks_canonical (0x02) section (handles zstd / txt_streams / dict /
    /// fsst transparently). used by the agent-native `retrieve()` to attach
    /// cited text to each hit.
    pub fn canonical_texts(&self) -> Result<Vec<String>, RuntimeError> {
        let view = UrnaView::from_bytes(&self._mmap)?;
        let n = view.header.n_chunks as usize;
        decode_chunks_canonical(&view.decoded_section(SECTION_CHUNKS_CANONICAL)?, n)
            .map_err(RuntimeError::Format)
    }

    /// The per-chunk ids in file order, parallel to `canonical_texts()`.
    /// lets a caller build a chunk_id -> canonical-text map without re-reading
    /// the file (the ids are already decoded and held at open time).
    pub fn chunk_ids(&self) -> &[String] {
        &self.chunk_ids
    }

    /// Re-parse the mmap and return a JSON document mirroring `urna
    /// inspect`: header fields, section table entries, manifest, hashes,
    /// and the runtime SIMD backend.
    pub fn inspect_json(&self) -> Result<String, RuntimeError> {
        let view = UrnaView::from_bytes(&self._mmap)?;
        let magic = std::str::from_utf8(&view.header.magic)
            .unwrap_or("")
            .to_string();
        let sections: Vec<serde_json::Value> = view
            .section_table
            .iter()
            .map(|e| {
                let name = urna_format::layout::section_name(e.section_id).unwrap_or("unknown");
                serde_json::json!({
                    "section_id": e.section_id,
                    "name": name,
                    "encoding": e.encoding,
                    "offset": e.offset,
                    "size": e.size,
                    "checksum": hex::encode(e.checksum),
                })
            })
            .collect();
        let doc = serde_json::json!({
            "magic": magic,
            "version_major": view.header.version_major,
            "version_minor": view.header.version_minor,
            "format_version": view.manifest.format_version,
            "schema_version": view.manifest.schema_version,
            "embedding_dim": view.header.embedding_dim,
            "n_chunks": view.header.n_chunks,
            "n_embeddings": view.header.n_embeddings,
            "file_size": view.header.file_size,
            "manifest": view.manifest,
            "sections": sections,
            "blobs": self.blob_refs_json(),
            "spaces": self.spaces_json(),
            "file_hash": view.file_hash_hex(),
            "content_hash": view.content_hash_hex()?,
            "simd_backend": self.simd_backend().name(),
        });
        serde_json::to_string(&doc).map_err(|e| RuntimeError::Format(UrnaError::Json(e)))
    }

    /// The space_table (0x15) as a JSON array (`null` when the file has no
    /// multimodal capability): one object per named space with its band
    /// geometry, so `inspect --json` consumers (stats, the model bench)
    /// see every queryable space without opening the bands.
    fn spaces_json(&self) -> serde_json::Value {
        match &self.spaces {
            None => serde_json::Value::Null,
            Some(spaces) => spaces
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "name": s.entry.name,
                        "space_index": s.entry.space_index,
                        "dim": s.entry.dim,
                        "dtype": s.entry.dtype_str(),
                        "model_hash": s.entry.model_hash,
                        "n_vectors": s.entry.n_vectors,
                        "band_bytes": s.size,
                    })
                })
                .collect(),
        }
    }

    /// The blob_refs (0x14) table as a JSON array (`null` when the file
    /// has no blob capability): one object per blob with its content hash
    /// as `sha256:<hex>`, the uri hint, original byte length, and whether
    /// the bytes are inlined in this .urna.
    fn blob_refs_json(&self) -> serde_json::Value {
        match self.blob_refs() {
            None => serde_json::Value::Null,
            Some(refs) => refs
                .iter()
                .map(|r| {
                    serde_json::json!({
                        "content_hash": format!("sha256:{}", hex::encode(r.content_hash)),
                        "original_uri": r.original_uri,
                        "byte_len": r.byte_len,
                        "inlined": r.inlined,
                    })
                })
                .collect(),
        }
    }
}
