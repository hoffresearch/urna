//! `urna ask <file> "<query>"` - the flagship user-facing verb. text query
//! in, cited hits out. embeds the query OFFLINE with the default potion
//! static table (NEVER sentence-transformers, so it works offline-by-
//! construction), validates the embedder's model_hash against the manifest
//! like `search-text` does, routes by manifest capability, and renders one
//! low-cognitive-load answer: the cited canonical text + a urna:// citation.
//!
//! `--disclose answer` (default): cited text + urna:// citation only, no
//! field-wall (the newcomer and the agent both want this). `--disclose
//! explain` ALSO prints the rerank-source honesty line ("real cosine" vs
//! "real cosine at stored precision") plus the route + candidate counts, so
//! a sub-int8 corpus can never present a stored-precision score as full.
//!
//! cite stays TIER-1: the printed text is the stored canonical text, the
//! same bytes `urna cite` returns; this verb never claims original-byte
//! reopen (that is net-new tier-2 catalog, post-gate).

use anyhow::Result;
use std::path::PathBuf;

use super::super::embed_gate::embed_and_search;

/// disclosure level for `ask`. mirrors the lens design's progressive
/// disclosure dial, scoped to what `ask` needs pre-gate (answer | explain).
#[derive(Clone, Copy, clap::ValueEnum)]
pub enum Disclose {
    /// cited text + urna:// citation only (default).
    Answer,
    /// answer plus the rerank-source honesty line, route, and candidate counts.
    Explain,
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    file: PathBuf,
    query: String,
    k: i32,
    disclose: Disclose,
    embedder: Option<PathBuf>,
    candidates: Option<usize>,
    model_path: Option<PathBuf>,
) -> Result<()> {
    let runtime = urna_runtime::MmapUrnaFile::open(&file)?;
    let result = embed_and_search(&runtime, &query, k, candidates, embedder, model_path)?;

    // tier-1 canonical text for the returned chunk_ids, the same bytes cite
    // returns. resolved once from the file; never the original source bytes.
    let texts = super::retrieve::canonical_texts(&file)?;
    let by_id: std::collections::HashMap<&str, &str> = texts
        .iter()
        .map(|(id, t)| (id.as_str(), t.as_str()))
        .collect();

    if matches!(disclose, Disclose::Explain) {
        let e = &result.explain;
        println!("route:         {}", e.route);
        println!(
            "candidates:    exact={} ann={} bm25={} graph={} fusion={}",
            e.exact_candidates,
            e.ann_candidates,
            e.bm25_candidates,
            e.graph_candidates,
            e.fusion_mode
        );
        // the honesty backbone: state whether the score is full-precision.
        println!("rerank_source: {}", e.rerank_source.disclosure());
        if e.recall_estimate.is_nan() {
            println!("recall:        (not computed; rerank guarantees real cosine)");
        } else {
            println!("recall:        {}", e.recall_estimate);
        }
        println!();
    }

    // the answer: one low-cognitive-load block per hit, sources beneath, no
    // field-wall. the citation is the stable urna://content_hash/chunk_id.
    if result.hits.is_empty() {
        println!("no hits.");
        return Ok(());
    }
    for hit in &result.hits {
        let text = by_id.get(hit.chunk_id.as_str()).copied().unwrap_or("");
        println!("{text}");
        println!("  -- {} ({})", hit.citation_id, hit.source_uri);
        println!();
    }
    Ok(())
}
