//! `UrnaFile.retrieve(query, k, ...)`: the agent-native flagship on the
//! python surface. returns cited spans where each hit's `score` IS the
//! exact-cosine rerank value (the same gate rerank_contract.rs enforces),
//! with the tier-1 stored canonical `text` and the verifying hashes
//! attached so an agent gets a citeable answer in one call.
//!
//! the query is a pre-embedded vector (the python convenience
//! `forge/retrieve.py` does the offline potion embed first, keeping
//! sentence-transformers off the path). routing mirrors the manifest
//! capability: hnsw/hybrid/graph as the file advertises, else exact.
//!
//! TIER-1 ONLY: `text` is the stored canonical text, the same bytes `cite`
//! returns; this never claims an original-byte reopen. typed errors map to
//! `PyValueError`, never a panic.

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use urna_runtime::{MmapUrnaFile, SearchResult};

/// one cited span from `retrieve()`. mirrors `SearchHitPy` and adds the
/// tier-1 `text` plus the `rerank_source` precision disclosure, so an agent
/// has the citeable answer and the honesty marker without a second call.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct RetrieveHitPy {
    #[pyo3(get)]
    pub chunk_id: String,
    /// the exact-cosine rerank value (NOT a candidate-generator proxy).
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
    pub citation_id: String,
    /// tier-1 stored canonical text (the same bytes `cite` returns).
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub file_hash: String,
    #[pyo3(get)]
    pub content_hash: String,
    /// "full_precision" | "stored_precision": the precision the rerank read.
    #[pyo3(get)]
    pub rerank_source: String,
}

/// Route by manifest capability, run search, then attach tier-1 canonical
/// text to each hit. the score on every returned hit IS the exact rerank
/// value `search_*` produced. shared by the `UrnaFile.retrieve` method.
pub fn retrieve(
    rt: &MmapUrnaFile,
    query: &Bound<PyAny>,
    k: i32,
    candidates: Option<usize>,
    hops: usize,
    ef: usize,
    expected_model_hash: Option<String>,
) -> PyResult<Vec<RetrieveHitPy>> {
    // honesty gate: when the caller passes the model_hash of the embedder it
    // used for `query`, reject a corpus built with a different model. A bare
    // query vector carries no model identity, so the runtime cannot gate
    // unconditionally the way the CLI does — the caller opts in by passing the
    // hash (forge/retrieve.py does so by default). Without it, behaviour is
    // unchanged, but a mismatch would silently return cosine-valid, wrong hits.
    if let Some(expected) = &expected_model_hash {
        let actual = rt.model_hash();
        if expected != actual {
            return Err(PyValueError::new_err(format!(
                "model_hash mismatch: the query was embedded with {expected}, but the corpus \
                 was built with {actual}. Results would be cosine-valid but semantically wrong. \
                 Pass expected_model_hash=None to bypass this check."
            )));
        }
    }

    let qvec: Vec<f32> = query
        .extract()
        .map_err(|e| PyValueError::new_err(format!("invalid query vector: {e}")))?;

    let cand = candidates.unwrap_or(((k as usize) * 4).max(64));
    let result: SearchResult = match rt.declared_index_type() {
        "hnsw" => rt.search_ann(&qvec, k, ef.max(cand)),
        "hybrid" => rt.search_hybrid(&qvec, "", k, cand),
        "graph" => rt.search_graph(&qvec, k, hops, ef),
        _ => rt.search(&qvec, k),
    }
    .map_err(|e| PyValueError::new_err(format!("{e}")))?;

    // tier-1 canonical text, decoded once, mapped by chunk_id (file order).
    let texts = rt
        .canonical_texts()
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let ids = rt.chunk_ids();
    let by_id: HashMap<&str, &str> = ids
        .iter()
        .map(String::as_str)
        .zip(texts.iter().map(String::as_str))
        .collect();

    let rerank_source = result.explain.rerank_source.as_str().to_string();
    Ok(result
        .hits
        .into_iter()
        .map(|h| {
            let text = by_id
                .get(h.chunk_id.as_str())
                .copied()
                .unwrap_or("")
                .to_string();
            RetrieveHitPy {
                chunk_id: h.chunk_id,
                score: h.score,
                score_type: h.score_type.to_string(),
                source_uri: h.source_uri,
                offset_start: h.offset_start,
                offset_end: h.offset_end,
                citation_id: h.citation_id,
                text,
                file_hash: h.file_hash,
                content_hash: h.content_hash,
                rerank_source: rerank_source.clone(),
            }
        })
        .collect())
}
