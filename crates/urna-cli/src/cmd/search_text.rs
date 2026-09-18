//! `urna search-text <file> "query" -k K` — embed the query via
//! `python/embed_query.py`, validate model_hash against the manifest
//! (the shared three-layer gate in `embed_gate`), route to the declared
//! `index_type`. Keeps `--skip-model-hash-check` for legacy placeholder
//! corpora.

use anyhow::Result;
use std::path::PathBuf;

use super::embed_gate::{default_embedder_path, spawn_embedder, validate_gate};
use super::util::print_result;

#[allow(clippy::too_many_arguments)]
pub fn run(
    file: PathBuf,
    query: String,
    k: i32,
    embedder: Option<PathBuf>,
    candidates: Option<usize>,
    model_path: Option<PathBuf>,
    skip_model_hash_check: bool,
) -> Result<()> {
    let runtime = urna_runtime::MmapUrnaFile::open(&file)?;
    let info: serde_json::Value = serde_json::from_str(&runtime.inspect_json()?)?;
    let model = info["manifest"]["embedding_model"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("manifest.embedding_model missing"))?
        .to_string();
    let declared_dim = info["manifest"]["embedding_dim"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("manifest.embedding_dim missing"))?
        as usize;
    let declared_model_hash = info["manifest"]["model_hash"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("manifest.model_hash missing"))?
        .to_string();

    let embedder = embedder.unwrap_or_else(default_embedder_path);
    eprintln!(
        "[urna] embedding query with {} via {}{}",
        model,
        embedder.display(),
        match &model_path {
            Some(p) => format!(" (--model-path {})", p.display()),
            None => String::new(),
        }
    );
    let payload = spawn_embedder(&embedder, model_path.as_ref(), &[], &model, &query)?;
    validate_gate(
        &payload,
        &model,
        declared_dim,
        &declared_model_hash,
        skip_model_hash_check,
    )?;

    let cand = candidates.unwrap_or(((k as usize) * 4).max(64));
    let result = match runtime.declared_index_type() {
        "hnsw" => runtime.search_ann(&payload.vector, k, cand)?,
        "hybrid" => runtime.search_hybrid(&payload.vector, &query, k, cand)?,
        _ => runtime.search(&payload.vector, k)?,
    };
    print_result(&result);
    Ok(())
}
