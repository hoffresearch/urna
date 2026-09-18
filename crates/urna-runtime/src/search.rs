//! Search entry points: exact, ANN, hybrid. All paths return `SearchResult`
//! with the real cosine score (ANN/hybrid rerank candidates with the exact
//! dot product before returning).

use crate::bm25;
use crate::error::RuntimeError;
use crate::mmap_file::MmapUrnaFile;
use crate::rerank::RerankSource;
use crate::{RerankSourceKind, SearchExplain, SearchHit, SearchResult};

impl MmapUrnaFile {
    /// the honesty marker: the precision the rerank reads from, from the
    /// `embeddings_fp` (0x09) slab dtype when present, else the stored dtype.
    fn rerank_source_kind(&self) -> RerankSourceKind {
        let dtype = self
            .embeddings_fp_slab()
            .map(|(_, d)| d)
            .unwrap_or(self.dtype);
        RerankSourceKind::from_dtype(dtype)
    }

    /// Validate query, L2-normalize, return the normalized vector.
    fn validate_query(&self, query: &[f32], k: i32) -> Result<Vec<f32>, RuntimeError> {
        if k <= 0 {
            return Err(RuntimeError::InvalidK(k));
        }
        if query.is_empty() {
            return Err(RuntimeError::EmptyQuery);
        }
        if query.len() != self.embedding_dim {
            return Err(RuntimeError::DimensionMismatch {
                expected: self.embedding_dim,
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
        Ok(qnorm)
    }

    /// The single rerank source the exact-cosine recompute reads from:
    /// the full-precision `embeddings_fp` (0x09) slab when present, else
    /// the stored dtype slab. Both `score_all` and `score_subset` route
    /// through this so every path's returned score is the SAME real
    /// cosine, and a future per-space path just hands a different slab.
    fn rerank_source(&self) -> Result<RerankSource<'_>, RuntimeError> {
        let (dtype, bytes) = match self.embeddings_fp_slab() {
            Some((bytes, dtype)) => (dtype, bytes),
            None => (self.dtype, self.embeddings_bytes()),
        };
        RerankSource::new(dtype, bytes, self.n_embeddings, self.embedding_dim)
    }

    /// Score every chunk against `qnorm` via the rerank source. Returns
    /// `(idx, score)` pairs in the natural index order.
    fn score_all(&self, qnorm: &[f32]) -> Result<Vec<(usize, f32)>, RuntimeError> {
        let n = self.n_embeddings;
        let mut src = self.rerank_source()?;
        let mut scores: Vec<(usize, f32)> = Vec::with_capacity(n);
        for i in 0..n {
            scores.push((i, src.score(qnorm, i)));
        }
        Ok(scores)
    }

    /// Score a sliced subset of indices (used by ANN/BM25 rerank). The
    /// returned vector mirrors `idxs.len()` in order. This IS the exact
    /// rerank every candidate-generating path must end in.
    fn score_subset(
        &self,
        qnorm: &[f32],
        idxs: &[usize],
    ) -> Result<Vec<(usize, f32)>, RuntimeError> {
        let mut src = self.rerank_source()?;
        let mut out: Vec<(usize, f32)> = Vec::with_capacity(idxs.len());
        for &i in idxs {
            out.push((i, src.score(qnorm, i)));
        }
        Ok(out)
    }

    /// Exact flat search. The recall=1.0 ground truth.
    pub fn search(&self, query: &[f32], k: i32) -> Result<SearchResult, RuntimeError> {
        let t0 = std::time::Instant::now();
        let qnorm = self.validate_query(query, k)?;
        let mut scores = self.score_all(&qnorm)?;
        sort_scores_desc(&mut scores);
        let k_usize = k as usize;
        let truncated = k_usize < self.n_embeddings;
        let top = &scores[..k_usize.min(self.n_embeddings)];
        let hits = self.materialize_hits(top, "exact", false);
        Ok(SearchResult {
            hits: hits.clone(),
            query_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
            index_type: "exact",
            recall: 1.0,
            truncated,
            k_requested: k,
            k_returned: hits.len(),
            explain: SearchExplain::exact(self.n_embeddings, self.rerank_source_kind()),
        })
    }

    /// ANN search. Pulls `ef_search` candidates from HNSW, reranks with
    /// the exact dot product, returns top-k. Falls back to `search()` if
    /// no ANN section is present.
    pub fn search_ann(
        &self,
        query: &[f32],
        k: i32,
        ef_search: usize,
    ) -> Result<SearchResult, RuntimeError> {
        let t0 = std::time::Instant::now();
        let Some(idx) = self.ann_index.as_ref() else {
            return self.search(query, k);
        };
        let qnorm = self.validate_query(query, k)?;
        let candidates = idx.search(&qnorm, ef_search.max(k as usize));
        let ann_candidates = candidates.len();
        let mut reranked = self.score_subset(&qnorm, &candidates)?;
        sort_scores_desc(&mut reranked);
        let k_usize = k as usize;
        let truncated = k_usize < self.n_embeddings;
        let top = &reranked[..k_usize.min(reranked.len())];
        let hits = self.materialize_hits(top, "hnsw", true);
        Ok(SearchResult {
            hits: hits.clone(),
            query_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
            index_type: "hnsw",
            // Recall is candidate-set dependent; runtime caller can
            // measure it against `search()` directly. We don't lie here.
            recall: f32::NAN,
            truncated,
            k_requested: k,
            k_returned: hits.len(),
            explain: SearchExplain::hnsw(ann_candidates, self.rerank_source_kind()),
        })
    }

    /// Graph search: seed from the exact-cosine top-`ef`, expand a bounded
    /// bfs over the chunk-to-chunk csr, union seed + frontier, then run the
    /// SAME mandatory exact rerank on the union. the graph ONLY generates
    /// candidates; the returned score is real cosine, identical contract to
    /// `search_ann`. falls back to `search()` when no graph section is
    /// present. recall stays `f32::NAN` (we never lie).
    pub fn search_graph(
        &self,
        query: &[f32],
        k: i32,
        hops: usize,
        ef: usize,
    ) -> Result<SearchResult, RuntimeError> {
        let t0 = std::time::Instant::now();
        let Some(graph) = self.graph_index.as_ref() else {
            return self.search(query, k);
        };
        let qnorm = self.validate_query(query, k)?;

        // exact-cosine top-ef seed (reuse score_all + sort), like the exact
        // path but truncated to the seed budget.
        let mut seed_scores = self.score_all(&qnorm)?;
        sort_scores_desc(&mut seed_scores);
        let seed_budget = ef.max(k as usize).min(seed_scores.len());
        let seeds: Vec<usize> = seed_scores[..seed_budget].iter().map(|p| p.0).collect();

        // bounded bfs expands the frontier over the csr; the union of seeds +
        // frontier is the candidate set. cap the frontier so a dense graph
        // cannot blow up the rerank cost.
        let max_frontier = (seed_budget.saturating_mul(8)).max(seed_budget);
        let mut traversal = crate::graph::Traversal::new(graph.n_nodes());
        let union = traversal.bounded_bfs(graph, &seeds, hops, max_frontier);
        let graph_candidates = union.len();

        let mut reranked = self.score_subset(&qnorm, &union)?;
        sort_scores_desc(&mut reranked);
        let k_usize = k as usize;
        let truncated = k_usize < self.n_embeddings;
        let top = &reranked[..k_usize.min(reranked.len())];
        let hits = self.materialize_hits(top, "graph", true);
        Ok(SearchResult {
            hits: hits.clone(),
            query_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
            index_type: "graph",
            recall: f32::NAN,
            truncated,
            k_requested: k,
            k_returned: hits.len(),
            explain: SearchExplain::graph(seed_budget, graph_candidates, self.rerank_source_kind()),
        })
    }

    /// Hybrid search: BM25 candidates ∪ ANN (or exact) candidates,
    /// reciprocal-rank fusion, then exact cosine rerank on the union.
    /// Final score is the real cosine.
    pub fn search_hybrid(
        &self,
        query_vec: &[f32],
        query_text: &str,
        k: i32,
        candidates_per_path: usize,
    ) -> Result<SearchResult, RuntimeError> {
        let t0 = std::time::Instant::now();
        let qnorm = self.validate_query(query_vec, k)?;
        let src = self.rerank_source_kind();

        // Vector path: ANN if available, otherwise top-`candidates`.
        let has_ann = self.ann_index.is_some();
        let vec_candidates: Vec<usize> = if let Some(idx) = self.ann_index.as_ref() {
            idx.search(&qnorm, candidates_per_path.max(k as usize))
        } else {
            let mut all = self.score_all(&qnorm)?;
            sort_scores_desc(&mut all);
            all.iter().take(candidates_per_path).map(|p| p.0).collect()
        };

        // Lexical path.
        let lex_candidates: Vec<usize> = if let Some(bm) = self.bm25_index.as_ref() {
            bm.search(query_text, candidates_per_path)
                .into_iter()
                .map(|(idx, _score)| idx)
                .collect()
        } else {
            Vec::new()
        };

        // attribute the vector count to the generator that ran (ann vs
        // exact-flat shortlist) so the explain line is honest.
        let nvec = vec_candidates.len();
        let (ann_candidates, exact_candidates) = if has_ann { (nvec, 0) } else { (0, nvec) };
        let bm25_candidates = lex_candidates.len();

        // Reciprocal-rank fusion to pick a union, then exact rerank.
        let union = bm25::rrf_union(&vec_candidates, &lex_candidates);
        let mut reranked = self.score_subset(&qnorm, &union)?;
        sort_scores_desc(&mut reranked);
        let k_usize = k as usize;
        let truncated = k_usize < self.n_embeddings;
        let top = &reranked[..k_usize.min(reranked.len())];
        let hits = self.materialize_hits(top, "hybrid", true);
        Ok(SearchResult {
            hits: hits.clone(),
            query_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
            index_type: "hybrid",
            recall: f32::NAN,
            truncated,
            k_requested: k,
            k_returned: hits.len(),
            explain: SearchExplain::hybrid(ann_candidates, exact_candidates, bm25_candidates, src),
        })
    }

    pub(crate) fn materialize_hits(
        &self,
        scored: &[(usize, f32)],
        index_type: &'static str,
        reranked: bool,
    ) -> Vec<SearchHit> {
        scored
            .iter()
            .map(|(idx, score)| {
                let span = &self.spans[*idx];
                let id = &self.chunk_ids[*idx];
                SearchHit {
                    chunk_id: id.clone(),
                    score: *score,
                    score_type: "cosine",
                    source_uri: span.source_uri.clone(),
                    offset_start: span.byte_start,
                    offset_end: span.byte_end,
                    embedding_model: self.embedding_model.clone(),
                    index_type,
                    reranked,
                    file_hash: self.file_hash.clone(),
                    content_hash: self.content_hash.clone(),
                    citation_id: format!("urna://{}/{}", self.content_hash, id),
                }
            })
            .collect()
    }
}

#[inline]
pub(crate) fn sort_scores_desc(scores: &mut [(usize, f32)]) {
    scores.sort_by(|a, b| crate::order::cmp_score_desc(*a, *b));
}
