//! `urna search-ann <file> <query-json> -k K --ef N` — force the HNSW
//! path. Falls back to exact if the file has no HNSW section.

use anyhow::Result;
use std::path::PathBuf;

use super::util::print_result;

pub fn run(file: PathBuf, query: String, k: i32, ef: usize) -> Result<()> {
    let runtime = urna_runtime::MmapUrnaFile::open(&file)?;
    let qvec: Vec<f32> =
        serde_json::from_str(&query).map_err(|e| anyhow::anyhow!("invalid query JSON: {}", e))?;
    let result = runtime.search_ann(&qvec, k, ef)?;
    print_result(&result);
    Ok(())
}
