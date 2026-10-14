//! `urna search-text <file> "query" -k K` - embed the query via
//! `rust/bridge/python/urna/embed/searchtxt.py`, validate model_hash against the manifest
//! (the shared three-layer gate in `embed_gate`), route by capability
//! (bm25 -> hybrid, hnsw -> ann, else exact). Keeps
//! `--skip-model-hash-check` for legacy placeholder corpora.

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
    let runtime = urna_engine::MmapUrnaFile::open(&file)?;
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
    // a corpus whose default space was truncated at build time (full_dim
    // recorded) is queried at the manifest dim: the embedder slices and
    // renormalizes, exactly like the ask/retrieve path does.
    let mut extra: Vec<String> = Vec::new();
    if info["manifest"]["full_dim"].as_u64().is_some() {
        extra.push("--mrl-dim".into());
        extra.push(declared_dim.to_string());
    }
    let payload = spawn_embedder(&embedder, model_path.as_ref(), &extra, &model, &query)?;
    validate_gate(
        &payload,
        &model,
        declared_dim,
        &declared_model_hash,
        skip_model_hash_check,
    )?;

    let cand = candidates.unwrap_or(((k as usize) * 4).max(64));
    let result = runtime.search_routed(&payload.vector, Some(&query), k, cand)?;
    print_result(&result);
    Ok(())
}
