//! `urna search-text <file> "query" -k K`: the query embedded and gated
//! by `embed_gate::embed_and_route` (a potion corpus in process, any other
//! model through its query embedder), then routed by capability (bm25 ->
//! hybrid, hnsw -> ann, else exact). Keeps `--skip-model-hash-check` for
//! legacy placeholder corpora.

use anyhow::Result;
use std::path::PathBuf;

use super::embed_gate::embed_and_route;
use super::util::print_result;

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
    let result = embed_and_route(
        &runtime,
        &query,
        k,
        candidates,
        embedder,
        model_path,
        None,
        skip_model_hash_check,
    )?;
    print_result(&result);
    Ok(())
}
