//! `UrnaFileBuilder::build_bytes` — orchestrates manifest validation,
//! payload encoding, layout planning, buffer assembly, and final
//! checksums + file_hash. Result is byte-deterministic for identical
//! inputs (and `reproducible(true)`).

use super::REPRODUCIBLE_CREATED;
use super::SectionEncoding;
use super::UrnaFileBuilder;
use super::payload::{encode_embeddings_payload, maybe_zstd};
use super::text_codec;
use crate::chunk::{chunk_id, validate_chunk};
use crate::error::UrnaError;
use crate::layout::{
    REQUIRED_SECTIONS, SECTION_ALIGNMENT, SECTION_BLOB_DATA, SECTION_BLOB_REFS,
    SECTION_BLOB_SPAN_OVERLAY, SECTION_BM25_INDEX, SECTION_CHUNK_IDS, SECTION_CHUNKS_CANONICAL,
    SECTION_CHUNKS_ORIGINAL_SPANS, SECTION_EMBEDDINGS, SECTION_ENCODING_INTPACK,
    SECTION_ENCODING_RAW, SECTION_GRAPH_ADJACENCY, SECTION_HNSW_INDEX, SECTION_PROVENANCE,
    SECTION_SEARCH_CONTRACT, SECTION_SPACE_TABLE, SectionEntry, URNA_FOOTER_SIZE, URNA_HEADER_SIZE,
    URNA_SECTION_ENTRY_SIZE, UrnaFooter, UrnaHeader, align_up,
};
use crate::sections::{
    OriginalSpan, SearchContract, encode_chunk_ids, encode_chunk_ids_intpack,
    encode_chunks_canonical, encode_chunks_original_spans, encode_chunks_original_spans_intpack,
    encode_provenance, encode_search_contract,
};
use sha2::{Digest, Sha256};

impl UrnaFileBuilder {
    /// Build the file in memory. Pure computation — no I/O.
    pub fn build_bytes(mut self) -> crate::Result<Vec<u8>> {
        if self.reproducible {
            self.manifest.created = Some(REPRODUCIBLE_CREATED.into());
        }
        // 1. Validate manifest (rejects bad dtype/metric/etc up front).
        self.manifest.validate()?;

        // 2. Validate chunks against the manifest's embedding_dim and n_chunks.
        let embedding_dim = self.manifest.embedding_dim as usize;
        if self.chunks.len() as u64 != self.manifest.n_chunks {
            return Err(UrnaError::ManifestInvalid(format!(
                "n_chunks={} but builder has {} chunks",
                self.manifest.n_chunks,
                self.chunks.len()
            )));
        }
        for c in &self.chunks {
            validate_chunk(c, embedding_dim)?;
        }

        // 3. Derive section payloads.
        let chunk_ids: Vec<String> = self
            .chunks
            .iter()
            .map(|c| {
                chunk_id(
                    &c.canonical_text,
                    &c.source_uri,
                    c.byte_start,
                    c.byte_end,
                    &self.manifest.chunker_version,
                )
            })
            .collect();

        let canonical_texts: Vec<String> = self
            .chunks
            .iter()
            .map(|c| c.canonical_text.clone())
            .collect();
        let original_spans: Vec<OriginalSpan> = self
            .chunks
            .iter()
            .map(|c| OriginalSpan {
                source_uri: c.source_uri.clone(),
                byte_start: c.byte_start,
                byte_end: c.byte_end,
            })
            .collect();

        let embeddings_bytes = encode_embeddings_payload(self.dtype, &self.chunks, embedding_dim)?;

        let contract = SearchContract {
            metric: self.manifest.metric.clone(),
            score_type: self.manifest.score_type.clone(),
            normalize: self.manifest.normalize.clone(),
            index_type: self.manifest.index_type.clone(),
            rerank_policy: self.manifest.rerank_policy.clone(),
        };

        // (id, encoding, payload). embeddings get dtype-specific encoding;
        // text sections honor `text_encoding`. under a compressed (zstd-text)
        // preset, chunk_ids and spans get the `intpack` repack (encoding 4):
        // chunk_ids to 32 raw digest bytes (always a win for high-entropy
        // sha-256), spans to a deduped uri pool + bitpacked offsets. the span
        // repack only beats zstd when source_uris repeat, so spans takes the
        // SMALLER of intpack/zstd per corpus. all of these decode BYTE-
        // IDENTICALLY to the raw payload, so content_hash is unchanged. raw-
        // text presets (and the golden) keep raw/zstd, so they stay
        // byte-identical.
        let text_enc = self.text_encoding;
        let compressed = matches!(text_enc, SectionEncoding::Zstd);
        let mut sections: Vec<(u32, u32, Vec<u8>)> = Vec::with_capacity(8);

        let chunk_ids_section = match compressed.then(|| encode_chunk_ids_intpack(&chunk_ids)) {
            Some(Some(packed)) => (SECTION_CHUNK_IDS, SECTION_ENCODING_INTPACK, packed),
            _ => (
                SECTION_CHUNK_IDS,
                SECTION_ENCODING_RAW,
                encode_chunk_ids(&chunk_ids)?,
            ),
        };
        sections.push(chunk_ids_section);

        // chunks_canonical: under a compressed (zstd-text) preset, the text
        // codec chooser takes the SMALLEST of single-frame zstd, txt_streams
        // cold, txt_streams+trained-dict (0x0A), txt_streams+fsst, and
        // dedup+zstd (0x0B), so the build never regresses (single-frame zstd
        // is always in the race). every candidate decodes BYTE-IDENTICALLY to
        // the raw payload, so content_hash is unchanged; the dict/dedup aux
        // sections are excluded from content_hash. raw-text presets (and the
        // golden) keep raw bytes and stay byte-identical.
        if compressed {
            let choice = text_codec::choose(&canonical_texts)?;
            sections.push(choice.canonical);
            sections.extend(choice.aux);
        } else {
            let canonical_raw = encode_chunks_canonical(&canonical_texts)?;
            sections.push(maybe_zstd(
                SECTION_CHUNKS_CANONICAL,
                text_enc,
                canonical_raw,
            )?);
        }

        let spans_raw = encode_chunks_original_spans(&original_spans)?;
        let spans_section = if compressed {
            let zst = maybe_zstd(SECTION_CHUNKS_ORIGINAL_SPANS, text_enc, spans_raw)?;
            let packed = encode_chunks_original_spans_intpack(&original_spans);
            if packed.len() < zst.2.len() {
                (
                    SECTION_CHUNKS_ORIGINAL_SPANS,
                    SECTION_ENCODING_INTPACK,
                    packed,
                )
            } else {
                zst
            }
        } else {
            maybe_zstd(SECTION_CHUNKS_ORIGINAL_SPANS, text_enc, spans_raw)?
        };
        sections.push(spans_section);
        sections.push((SECTION_EMBEDDINGS, self.dtype.encoding(), embeddings_bytes));
        sections.push(maybe_zstd(
            SECTION_PROVENANCE,
            text_enc,
            encode_provenance(&self.provenance)?,
        )?);
        sections.push(maybe_zstd(
            SECTION_SEARCH_CONTRACT,
            text_enc,
            encode_search_contract(&contract)?,
        )?);

        if let Some(payload) = self.hnsw_index.take() {
            // HNSW is binary, mostly random — zstd would barely help and
            // would defeat mmap-friendly reads. Always raw.
            sections.push((SECTION_HNSW_INDEX, SECTION_ENCODING_RAW, payload));
        }
        if let Some(payload) = self.bm25_index.take() {
            // BM25 posting lists are integer-heavy; zstd usually halves
            // them. Honor text_encoding here too.
            sections.push(maybe_zstd(SECTION_BM25_INDEX, text_enc, payload)?);
        }
        if let Some(payload) = self.graph_adjacency.take() {
            // graph_adjacency (0x0C) is a self-describing csr that already
            // bitpacks its integer columns with intpack internally; like hnsw
            // it stays RAW so the runtime mmaps it directly. it is OPTIONAL and
            // EXCLUDED from content_hash (not in CANONICAL_SECTIONS).
            sections.push((SECTION_GRAPH_ADJACENCY, SECTION_ENCODING_RAW, payload));
        }
        if let Some(payload) = self.blob_refs.take() {
            // blob_refs (0x14) is a small content-hash reference table the
            // runtime decodes eagerly at open; RAW like hnsw/graph. OPTIONAL
            // and EXCLUDED from content_hash.
            sections.push((SECTION_BLOB_REFS, SECTION_ENCODING_RAW, payload));
        }
        if let Some(payload) = self.blob_data.take() {
            // blob_data (0x17) carries the inlined media bytes; they are
            // already codec-compressed, so ALWAYS raw — the runtime slices
            // individual blobs lazily off the mmap. OPTIONAL and EXCLUDED
            // from content_hash, like its 0x14 table.
            sections.push((SECTION_BLOB_DATA, SECTION_ENCODING_RAW, payload));
        }
        if let Some(payload) = self.blob_span_overlay.take() {
            // blob_span_overlay (0x16) is the per-chunk blob-relative span
            // overlay; RAW like its 0x14 table. OPTIONAL and EXCLUDED from
            // content_hash, so the overlay never invalidates a citation.
            sections.push((SECTION_BLOB_SPAN_OVERLAY, SECTION_ENCODING_RAW, payload));
        }
        if let Some(payload) = self.space_table.take() {
            // space_table (0x15) is the small multimodal directory the
            // runtime decodes eagerly at open; RAW like blob_refs. OPTIONAL
            // and EXCLUDED from content_hash.
            sections.push((SECTION_SPACE_TABLE, SECTION_ENCODING_RAW, payload));
        }
        for (band_id, encoding, payload) in std::mem::take(&mut self.space_bands) {
            // space bands are fixed-stride vector slabs scored by the simd
            // kernels straight off mmap; they carry their dtype encoding
            // (raw f32 / float16 / int8 / int4), NEVER zstd. OPTIONAL and
            // EXCLUDED from content_hash.
            sections.push((band_id, encoding, payload));
        }

        // Sanity: every required section is present (writer never drops one).
        debug_assert!(
            REQUIRED_SECTIONS
                .iter()
                .all(|(id, _)| sections.iter().any(|s| s.0 == *id))
        );

        // 4. Manifest JSON (canonical).
        let manifest_json = self.manifest.to_canonical_json()?;

        // 5. Layout offsets. Each section starts at SECTION_ALIGNMENT.
        sections.sort_by_key(|s| s.0);
        let section_table_count = sections.len() as u64;
        let section_table_size = section_table_count * URNA_SECTION_ENTRY_SIZE as u64;
        let header_size = URNA_HEADER_SIZE as u64;
        let manifest_offset = header_size + section_table_size;
        let manifest_size = manifest_json.len() as u64;

        let mut section_entries: Vec<SectionEntry> = Vec::with_capacity(sections.len());
        let mut current_offset = align_up(manifest_offset + manifest_size, SECTION_ALIGNMENT);
        for (id, encoding, data) in &sections {
            let mut entry = SectionEntry::new(*id, current_offset, data.len() as u64);
            entry.encoding = *encoding;
            section_entries.push(entry);
            let after = current_offset + data.len() as u64;
            current_offset = align_up(after, SECTION_ALIGNMENT);
        }

        // After the last section we want the footer immediately, so use
        // the unaligned end of the last section's data.
        let last_section_end = match (section_entries.last(), sections.last()) {
            (Some(entry), Some((_, _, data))) => entry.offset + data.len() as u64,
            _ => manifest_offset + manifest_size,
        };
        let data_end = last_section_end as usize;
        let file_size = data_end + URNA_FOOTER_SIZE;
        let mut buf = vec![0u8; file_size];

        // 6. Write header (placeholder; checksum recomputed at end).
        let mut header = UrnaHeader::new(
            self.manifest.embedding_dim,
            self.manifest.n_chunks,
            self.chunks.len() as u64,
            file_size as u64,
            header_size,
            section_table_count,
            manifest_offset,
            manifest_size,
        );
        buf[..URNA_HEADER_SIZE].copy_from_slice(header.as_bytes());

        // 7. Write section table (placeholder checksums; filled below).
        for (i, entry) in section_entries.iter().enumerate() {
            let off = URNA_HEADER_SIZE + i * URNA_SECTION_ENTRY_SIZE;
            buf[off..off + URNA_SECTION_ENTRY_SIZE].copy_from_slice(entry.as_bytes());
        }

        // 8. Write manifest.
        let manifest_off = manifest_offset as usize;
        buf[manifest_off..manifest_off + manifest_json.len()].copy_from_slice(&manifest_json);

        // 9. Write section data at its declared (aligned) offset and
        //    compute checksums over data only — padding stays zero and
        //    is not hashed. Section checksum hashes the **physical** bytes
        //    on disk (so for zstd sections it's over the compressed bytes).
        for (i, (_, _, data)) in sections.iter().enumerate() {
            let entry_off = URNA_HEADER_SIZE + i * URNA_SECTION_ENTRY_SIZE;
            let data_off = section_entries[i].offset as usize;
            buf[data_off..data_off + data.len()].copy_from_slice(data);
            let hash = Sha256::digest(data);
            buf[entry_off + 24..entry_off + 32].copy_from_slice(&hash[..8]);
        }

        // 10. Footer hash (covers everything before footer).
        let footer_hash = UrnaFooter::compute_file_hash(&buf[..data_end]);
        buf[data_end..data_end + 8].copy_from_slice(&URNA_FOOTER_SIZE.to_le_bytes());
        buf[data_end + 8..file_size].copy_from_slice(&footer_hash);

        // 11. Recompute header checksum (file_size already final).
        header.compute_checksum();
        buf[..URNA_HEADER_SIZE].copy_from_slice(header.as_bytes());

        Ok(buf)
    }
}
