//! The search paths under a filter: the bitset of kept chunks is applied
//! BEFORE scoring. The exact scan skips dropped chunks; BM25 drops their
//! postings; HNSW widens its beam and keeps only candidates inside, and
//! when the filter keeps few chunks the search becomes an exact scan over
//! them, so the top-k is exact and never short while kept chunks remain.
//! Every path still ends in the same exact-cosine rerank.

use crate::bm25;
use crate::error::RuntimeError;
use crate::mmap_file::MmapUrnaFile;
use crate::scalars::Bitset;
use crate::search::sort_scores_desc;
use crate::{SearchExplain, SearchResult};

/// below this many kept chunks the ann beam is not worth it: an exact
/// scan over the kept set is cheaper and exact.
const EXACT_BELOW: usize = 4096;

impl MmapUrnaFile {
    /// Exact search over the kept chunks only.
    pub fn search_where(
        &self,
        query: &[f32],
        k: i32,
        keep: &Bitset,
    ) -> Result<SearchResult, RuntimeError> {
        let t0 = std::time::Instant::now();
        let qnorm = self.validate_query(query, k)?;
        let kept: Vec<usize> = keep.ones().collect();
        let mut scores = self.score_subset(&qnorm, &kept)?;
        sort_scores_desc(&mut scores);
        let k_usize = k as usize;
        let top = &scores[..k_usize.min(scores.len())];
        let hits = self.materialize_hits(top, "exact", false);
        let mut explain = SearchExplain::exact(kept.len(), self.rerank_source_kind());
        explain.filter_kept = Some(kept.len());
        Ok(SearchResult {
            hits: hits.clone(),
            query_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
            index_type: "exact",
            recall: 1.0,
            truncated: k_usize < kept.len(),
            k_requested: k,
            k_returned: hits.len(),
            explain,
        })
    }

    /// ANN search over the kept chunks: the beam widens by the kept
    /// fraction so enough candidates survive the filter; a small kept set
    /// goes to `search_where` instead.
    pub fn search_ann_where(
        &self,
        query: &[f32],
        k: i32,
        ef_search: usize,
        keep: &Bitset,
    ) -> Result<SearchResult, RuntimeError> {
        let kept = keep.count();
        let Some(idx) = self.ann_index.as_ref() else {
            return self.search_where(query, k, keep);
        };
        if kept <= EXACT_BELOW {
            return self.search_where(query, k, keep);
        }
        let t0 = std::time::Instant::now();
        let qnorm = self.validate_query(query, k)?;
        let ef = widened(ef_search.max(k as usize), kept, self.n_embeddings);
        let candidates: Vec<usize> = idx
            .search(&qnorm, ef)
            .into_iter()
            .filter(|&i| keep.get(i))
            .collect();
        // a filter that keeps a small fraction of a large corpus can leave
        // the capped beam with fewer than k kept candidates; the exact kept
        // scan keeps the promise of never coming back short.
        if candidates.len() < (k as usize).min(kept) {
            return self.search_where(query, k, keep);
        }
        let ann_candidates = candidates.len();
        let mut reranked = self.score_subset(&qnorm, &candidates)?;
        sort_scores_desc(&mut reranked);
        let k_usize = k as usize;
        let top = &reranked[..k_usize.min(reranked.len())];
        let hits = self.materialize_hits(top, "hnsw", true);
        let mut explain = SearchExplain::hnsw(ann_candidates, self.rerank_source_kind());
        explain.filter_kept = Some(kept);
        Ok(SearchResult {
            hits: hits.clone(),
            query_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
            index_type: "hnsw",
            recall: f32::NAN,
            truncated: k_usize < kept,
            k_requested: k,
            k_returned: hits.len(),
            explain,
        })
    }

    /// Hybrid search over the kept chunks: BM25 drops dropped postings,
    /// the vector path is `search_ann_where`'s shortlist (or the exact
    /// kept scan), then rrf and the exact rerank as always.
    pub fn search_hybrid_where(
        &self,
        query_vec: &[f32],
        query_text: &str,
        k: i32,
        candidates_per_path: usize,
        keep: &Bitset,
    ) -> Result<SearchResult, RuntimeError> {
        let t0 = std::time::Instant::now();
        let qnorm = self.validate_query(query_vec, k)?;
        let src = self.rerank_source_kind();
        let kept = keep.count();
        let want = candidates_per_path.max(k as usize);

        let ann: Option<Vec<usize>> = self
            .ann_index
            .as_ref()
            .filter(|_| kept > EXACT_BELOW)
            .map(|idx| {
                let ef = widened(want, kept, self.n_embeddings);
                idx.search(&qnorm, ef)
                    .into_iter()
                    .filter(|&i| keep.get(i))
                    .take(want)
                    .collect::<Vec<usize>>()
            })
            // as in `search_ann_where`: a beam left short goes to the exact kept scan.
            .filter(|c| c.len() >= want.min(kept));
        let has_ann = ann.is_some();
        let vec_candidates: Vec<usize> = match ann {
            Some(c) => c,
            None => {
                let all: Vec<usize> = keep.ones().collect();
                let mut scored = self.score_subset(&qnorm, &all)?;
                sort_scores_desc(&mut scored);
                scored.iter().take(want).map(|p| p.0).collect()
            }
        };
        let lex_candidates: Vec<usize> = match self.bm25_index.as_ref() {
            Some(bm) if !query_text.trim().is_empty() => bm
                .search_where(query_text, want, &|i| keep.get(i))
                .into_iter()
                .map(|(i, _)| i)
                .collect(),
            _ => Vec::new(),
        };
        let nvec = vec_candidates.len();
        let (ann_candidates, exact_candidates) = if has_ann { (nvec, 0) } else { (0, nvec) };
        let bm25_candidates = lex_candidates.len();
        let union = bm25::rrf_union(&vec_candidates, &lex_candidates);
        let mut reranked = self.score_subset(&qnorm, &union)?;
        sort_scores_desc(&mut reranked);
        let k_usize = k as usize;
        let top = &reranked[..k_usize.min(reranked.len())];
        let hits = self.materialize_hits(top, "hybrid", true);
        let mut explain =
            SearchExplain::hybrid(ann_candidates, exact_candidates, bm25_candidates, src);
        explain.filter_kept = Some(kept);
        Ok(SearchResult {
            hits: hits.clone(),
            query_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
            index_type: "hybrid",
            recall: f32::NAN,
            truncated: k_usize < kept,
            k_requested: k,
            k_returned: hits.len(),
            explain,
        })
    }

    /// `search_routed` under a filter: the same route choice, each path's
    /// `_where` form. `None` is the unfiltered route.
    pub fn search_routed_where(
        &self,
        query: &[f32],
        query_text: Option<&str>,
        k: i32,
        candidates: usize,
        keep: Option<&Bitset>,
    ) -> Result<SearchResult, RuntimeError> {
        let Some(keep) = keep else {
            return self.search_routed(query, query_text, k, candidates);
        };
        match query_text {
            Some(text) if self.bm25_index.is_some() && !text.trim().is_empty() => {
                self.search_hybrid_where(query, text, k, candidates, keep)
            }
            _ if self.ann_index.is_some() => self.search_ann_where(query, k, candidates, keep),
            _ => self.search_where(query, k, keep),
        }
    }
}

/// the beam a filter needs: `ef` over the kept fraction, capped so a
/// filter that keeps a sliver never asks the graph for the whole corpus
/// (that case goes to the exact scan).
fn widened(ef: usize, kept: usize, n: usize) -> usize {
    if kept == 0 || n == 0 {
        return ef;
    }
    let scaled = (ef as f64 * n as f64 / kept as f64).ceil() as usize;
    scaled.clamp(ef, ef.saturating_mul(16).max(ef))
}
