//! `urna search-graph <file> <query-json> -k K --hops H --ef N`: seed from
//! the exact-cosine top-`ef`, expand a bounded bfs over the chunk-to-chunk
//! csr, then exact-rerank the union. the graph only generates candidates;
//! the returned score is real cosine. falls back to exact if the file has no
//! graph_adjacency section.

use anyhow::Result;
use std::path::PathBuf;

use super::util::print_result;

pub fn run(file: PathBuf, query: String, k: i32, hops: usize, ef: usize) -> Result<()> {
    let runtime = urna_runtime::MmapUrnaFile::open(&file)?;
    let qvec: Vec<f32> =
        serde_json::from_str(&query).map_err(|e| anyhow::anyhow!("invalid query JSON: {}", e))?;
    let result = runtime.search_graph(&qvec, k, hops, ef)?;
    print_result(&result);
    Ok(())
}
