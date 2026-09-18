//! `SearchHitPy`: the Python-visible search hit. One field per
//! `urna_runtime::SearchHit` field, all read-only from Python; `score` is
//! the exact-cosine rerank value and `citation_id` the stable
//! `urna://content_hash/chunk_id` reference.

use pyo3::prelude::*;

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct SearchHitPy {
    #[pyo3(get)]
    pub chunk_id: String,
    #[pyo3(get)]
    pub score: f32,
    #[pyo3(get)]
    pub score_type: String,
    #[pyo3(get)]
    pub source_uri: String,
    #[pyo3(get)]
    pub offset_start: u64,
    #[pyo3(get)]
    pub offset_end: u64,
    #[pyo3(get)]
    pub embedding_model: String,
    #[pyo3(get)]
    pub index_type: String,
    #[pyo3(get)]
    pub reranked: bool,
    #[pyo3(get)]
    pub file_hash: String,
    #[pyo3(get)]
    pub content_hash: String,
    #[pyo3(get)]
    pub citation_id: String,
}

impl From<urna_runtime::SearchHit> for SearchHitPy {
    fn from(h: urna_runtime::SearchHit) -> Self {
        Self {
            chunk_id: h.chunk_id,
            score: h.score,
            score_type: h.score_type.to_string(),
            source_uri: h.source_uri,
            offset_start: h.offset_start,
            offset_end: h.offset_end,
            embedding_model: h.embedding_model,
            index_type: h.index_type.to_string(),
            reranked: h.reranked,
            file_hash: h.file_hash,
            content_hash: h.content_hash,
            citation_id: h.citation_id,
        }
    }
}
