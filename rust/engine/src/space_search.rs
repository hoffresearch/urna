//! Per-space exact search over the multimodal vector bands (0x20-0x2F).
//!
//! This is the ONLY path that reads a non-text band, and the text paths
//! (exact/ann/graph/hybrid in `search.rs`) are the only paths that read
//! the canonical 0x04 slab: the two never mix, so a text query can never
//! be scored against the vision band by accident. every score is the real
//! cosine recomputed by the shared `RerankSource` (the same kernel the
//! text exact path uses), never an ann proxy.

use crate::dtype::DType;
use crate::error::RuntimeError;
use crate::mmap_file::MmapUrnaFile;
use crate::rerank::RerankSource;
use crate::search::sort_scores_desc;
use crate::{RerankSourceKind, SearchExplain, SearchResult};

impl MmapUrnaFile {
    /// Exact flat search over one named multimodal space (e.g. "vision").
    /// the query must be embedded with the model the space's
    /// `model_hash` fingerprints and have the space's dim; both are
    /// checked up front (the per-space honesty gate), so a text-tower
    /// query fails loudly instead of silently scoring the vision band.
    /// recall is 1.0: this is the exact ground truth over that space.
    pub fn search_space(
        &self,
        name: &str,
        query: &[f32],
        k: i32,
        expected_model_hash: Option<&str>,
    ) -> Result<SearchResult, RuntimeError> {
        let t0 = std::time::Instant::now();
        let (space, band) = self
            .space(name)
            .ok_or_else(|| RuntimeError::SpaceNotFound(name.to_string()))?;
        let entry = &space.entry;

        if let Some(expected) = expected_model_hash {
            if expected != entry.model_hash {
                return Err(RuntimeError::SpaceModelMismatch {
                    space: name.to_string(),
                    expected: expected.to_string(),
                    actual: entry.model_hash.clone(),
                });
            }
        }
        if k <= 0 {
            return Err(RuntimeError::InvalidK(k));
        }
        if query.is_empty() {
            return Err(RuntimeError::EmptyQuery);
        }
        if query.len() != entry.dim as usize {
            return Err(RuntimeError::DimensionMismatch {
                expected: entry.dim as usize,
                got: query.len(),
            });
        }
        for &v in query {
            if v.is_nan() || v.is_infinite() {
                return Err(RuntimeError::InvalidQueryValue);
            }
        }
        let mut qnorm = query.to_vec();
        let norm: f32 = qnorm.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            return Err(RuntimeError::ZeroNormQuery);
        }
        for x in &mut qnorm {
            *x /= norm;
        }

        let dtype = DType::from_str(entry.dtype_str())?;
        let n = entry.n_vectors as usize;
        let mut src = RerankSource::new(dtype, band, n, entry.dim as usize)?;
        let mut scores: Vec<(usize, f32)> = Vec::with_capacity(n);
        for i in 0..n {
            scores.push((i, src.score(&qnorm, i)));
        }
        sort_scores_desc(&mut scores);
        let k_usize = k as usize;
        let truncated = k_usize < n;
        let top = &scores[..k_usize.min(n)];
        let hits = self.materialize_hits(top, "space", false);
        Ok(SearchResult {
            hits: hits.clone(),
            query_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
            index_type: "space",
            recall: 1.0,
            truncated,
            k_requested: k,
            k_returned: hits.len(),
            explain: SearchExplain::exact(n, RerankSourceKind::from_dtype(dtype)),
        })
    }
}
