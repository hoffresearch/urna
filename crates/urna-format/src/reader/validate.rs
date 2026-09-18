//! Validation hooks called during `from_bytes` after structural
//! parsing succeeds. Each function focuses on one invariant; callers
//! get a typed error pointing at the offending section / dtype.

use super::UrnaView;
use crate::encoding::{Int4EmbeddingsView, Int8EmbeddingsView, expected_embeddings_size};
use crate::error::UrnaError;
use crate::layout::{
    REQUIRED_SECTIONS, SECTION_EMBEDDINGS, SECTION_ENCODING_FLOAT16, SECTION_ENCODING_FSST,
    SECTION_ENCODING_INT4, SECTION_ENCODING_INT8, SECTION_ENCODING_INTPACK, SECTION_ENCODING_RAW,
    SECTION_ENCODING_TXT_STREAMS, SECTION_ENCODING_ZSTD, SECTION_ENCODING_ZSTD_DICT,
    SECTION_SEARCH_CONTRACT,
};
use crate::sections::decode_search_contract;

impl UrnaView<'_> {
    pub(super) fn check_required_sections(&self) -> crate::Result<()> {
        for (id, name) in REQUIRED_SECTIONS {
            if !self.section_table.iter().any(|e| e.section_id == *id) {
                return Err(UrnaError::MissingRequiredSection(name));
            }
        }
        Ok(())
    }

    pub(super) fn validate_embeddings_layout(&self) -> crate::Result<()> {
        let entry = self.entry(SECTION_EMBEDDINGS)?;
        let dim = self.header.embedding_dim as usize;
        let n = self.header.n_embeddings as usize;
        let dtype = self.manifest.dtype.as_str();

        // Encoding/dtype consistency: float16 dtype implies float16 encoding,
        // int8 dtype implies int8 encoding, int4 dtype implies int4 encoding,
        // float32 dtype implies raw or zstd (zstd on embeddings is rejected
        // separately by validate_encoding_for_section).
        let valid_combo = matches!(
            (dtype, entry.encoding),
            ("float32", SECTION_ENCODING_RAW)
                | ("float16", SECTION_ENCODING_FLOAT16)
                | ("int8", SECTION_ENCODING_INT8)
                | ("int4", SECTION_ENCODING_INT4)
        );
        if !valid_combo {
            return Err(UrnaError::ManifestInvalid(format!(
                "embeddings section encoding={} does not match dtype={}",
                entry.encoding, dtype
            )));
        }

        let want = expected_embeddings_size(dtype, n, dim).ok_or_else(|| {
            UrnaError::UnsupportedDType(format!(
                "unknown embeddings dtype {dtype}, or n={n} x dim={dim} overflows"
            ))
        })?;
        let got = entry.size as usize;
        if got != want {
            return Err(UrnaError::EmbeddingSizeMismatch {
                expected: want,
                got,
            });
        }
        Ok(())
    }

    /// When a space_table (0x15) is present, every listed space must have
    /// its band section (0x20 + space_index) present with exactly the size
    /// its (n_vectors, dim, dtype) imply, and n_vectors must match the
    /// corpus chunk count (the bands are parallel per-chunk embeddings).
    /// runs after the structural parse, so the band encoding is already
    /// known legal (dtype encodings only, never zstd).
    pub(super) fn validate_space_bands(&self) -> crate::Result<()> {
        use crate::layout::{SECTION_SPACE_EMBEDDINGS_BASE, SECTION_SPACE_TABLE};
        use crate::sections::decode_space_table;
        if self.entry(SECTION_SPACE_TABLE).is_err() {
            return Ok(());
        }
        let entries = decode_space_table(&self.decoded_section(SECTION_SPACE_TABLE)?)?;
        let n_chunks = self.header.n_chunks;
        for e in &entries {
            if e.n_vectors != n_chunks {
                return Err(UrnaError::ManifestInvalid(format!(
                    "space {} lists {} vectors but the corpus has {} chunks",
                    e.name, e.n_vectors, n_chunks
                )));
            }
            let band_id = SECTION_SPACE_EMBEDDINGS_BASE + e.space_index as u32;
            let band = self.entry(band_id).map_err(|_| {
                UrnaError::ManifestInvalid(format!(
                    "space {} lists band {:#x} but the section is missing",
                    e.name, band_id
                ))
            })?;
            let want =
                expected_embeddings_size(e.dtype_str(), e.n_vectors as usize, e.dim as usize)
                    .ok_or_else(|| {
                        UrnaError::UnsupportedDType(format!(
                            "space {} dtype code {}",
                            e.name, e.dtype
                        ))
                    })?;
            if band.size as usize != want {
                return Err(UrnaError::EmbeddingSizeMismatch {
                    expected: want,
                    got: band.size as usize,
                });
            }
        }
        Ok(())
    }

    pub(super) fn validate_search_contract(&self) -> crate::Result<()> {
        let bytes = self.decoded_section(SECTION_SEARCH_CONTRACT)?;
        let contract = decode_search_contract(&bytes)?;
        if contract.metric != self.manifest.metric {
            return Err(UrnaError::UnsupportedMetric(format!(
                "section says {} but manifest says {}",
                contract.metric, self.manifest.metric
            )));
        }
        if contract.score_type != self.manifest.score_type {
            return Err(UrnaError::UnsupportedScoreType(format!(
                "section says {} but manifest says {}",
                contract.score_type, self.manifest.score_type
            )));
        }
        if contract.normalize != self.manifest.normalize {
            return Err(UrnaError::UnsupportedNormalize(format!(
                "section says {} but manifest says {}",
                contract.normalize, self.manifest.normalize
            )));
        }
        if contract.index_type != self.manifest.index_type {
            return Err(UrnaError::UnsupportedIndexType(format!(
                "section says {} but manifest says {}",
                contract.index_type, self.manifest.index_type
            )));
        }
        if contract.rerank_policy != self.manifest.rerank_policy {
            return Err(UrnaError::UnsupportedRerankPolicy(format!(
                "section says {} but manifest says {}",
                contract.rerank_policy, self.manifest.rerank_policy
            )));
        }
        Ok(())
    }

    /// Walk the embeddings section and reject any NaN/Inf value. Works
    /// for all supported dtypes.
    pub fn validate_embeddings_values(&self) -> crate::Result<()> {
        self.validate_embeddings_layout()?;
        let entry = self.entry(SECTION_EMBEDDINGS)?;
        let data = self.get_section_data(SECTION_EMBEDDINGS)?;
        let n = self.header.n_embeddings as usize;
        let dim = self.header.embedding_dim as usize;
        validate_slab_values(entry.encoding, data, n, dim)
    }
}

/// Reject any NaN/Inf in a fixed-stride vector slab of the given section
/// encoding (raw f32, float16, int8, int4). The canonical embeddings, the
/// `embeddings_fp` rerank slab and every multimodal band go through this
/// before a kernel ever scores them: a NaN lane would turn a cosine into a
/// NaN score, and a NaN score is what breaks a ranking (found by the
/// mutation fuzz harness through the space bands).
pub fn validate_slab_values(encoding: u32, data: &[u8], n: usize, dim: usize) -> crate::Result<()> {
    match encoding {
        SECTION_ENCODING_RAW => {
            for chunk in data.chunks_exact(4) {
                let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                if !v.is_finite() {
                    return Err(UrnaError::InvalidEmbeddingValue);
                }
            }
        }
        SECTION_ENCODING_FLOAT16 => {
            for chunk in data.chunks_exact(2) {
                let v = half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
                if !v.is_finite() {
                    return Err(UrnaError::InvalidEmbeddingValue);
                }
            }
        }
        SECTION_ENCODING_INT8 => {
            // i8 cannot encode NaN/Inf; only the per-vector scales could.
            let view = Int8EmbeddingsView::parse(data, n, dim)?;
            if (0..view.n).any(|i| !view.scale(i).is_finite()) {
                return Err(UrnaError::InvalidEmbeddingValue);
            }
        }
        SECTION_ENCODING_INT4 => {
            // 4-bit codes cannot encode NaN/Inf; only the per-group f16
            // absmax scales could.
            let view = Int4EmbeddingsView::parse(data, n, dim)?;
            for i in 0..view.n {
                if (0..view.blocks).any(|g| !view.group_scale(i, g).is_finite()) {
                    return Err(UrnaError::InvalidEmbeddingValue);
                }
            }
        }
        other => {
            return Err(UrnaError::UnsupportedSectionEncoding {
                section_id: SECTION_EMBEDDINGS,
                encoding: other,
            });
        }
    }
    Ok(())
}

/// Encoding rules: the embeddings section gets dtype-specific encodings
/// (float16, int8, int4) and rejects zstd (we want SIMD-friendly mmap
/// reads). the per-space vector bands (0x20-0x2F and the fp rerank band
/// 0x30-0x3F) follow the SAME rule: fixed-stride slabs scored by the simd
/// kernels, NEVER zstd. All other sections accept raw, zstd, intpack (the
/// content_hash-preserving repack for chunk_ids / spans), txt_streams (the
/// per-chunk-streams repack for chunks_canonical), zstd_dict (the
/// trained-dictionary per-chunk-streams variant, decoded against section
/// 0x0A), or fsst (the static-symbol-table per-chunk-streams variant). every
/// non-embedding codec decodes byte-identically to the raw payload, so
/// content_hash stays stable.
pub(super) fn validate_encoding_for_section(section_id: u32, encoding: u32) -> crate::Result<()> {
    use crate::layout::{
        SECTION_SPACE_EMBEDDINGS_BASE, SECTION_SPACE_EMBEDDINGS_FP_BASE, SPACE_BAND_LEN,
    };
    let is_space_band = (SECTION_SPACE_EMBEDDINGS_BASE
        ..SECTION_SPACE_EMBEDDINGS_BASE + SPACE_BAND_LEN)
        .contains(&section_id)
        || (SECTION_SPACE_EMBEDDINGS_FP_BASE..SECTION_SPACE_EMBEDDINGS_FP_BASE + SPACE_BAND_LEN)
            .contains(&section_id);
    let allowed = if section_id == SECTION_EMBEDDINGS || is_space_band {
        matches!(
            encoding,
            SECTION_ENCODING_RAW
                | SECTION_ENCODING_FLOAT16
                | SECTION_ENCODING_INT8
                | SECTION_ENCODING_INT4
        )
    } else {
        matches!(
            encoding,
            SECTION_ENCODING_RAW
                | SECTION_ENCODING_ZSTD
                | SECTION_ENCODING_INTPACK
                | SECTION_ENCODING_TXT_STREAMS
                | SECTION_ENCODING_ZSTD_DICT
                | SECTION_ENCODING_FSST
        )
    };
    if !allowed {
        return Err(UrnaError::UnsupportedSectionEncoding {
            section_id,
            encoding,
        });
    }
    Ok(())
}
