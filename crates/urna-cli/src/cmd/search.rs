//! `urna search <file> <query-as-json> -k K` — exact path with a JSON
//! array query vector.

use anyhow::Result;
use std::path::PathBuf;

use super::util::print_result;

pub fn run(file: PathBuf, query: String, k: i32) -> Result<()> {
    let runtime = urna_runtime::MmapUrnaFile::open(&file)?;
    let qvec: Vec<f32> =
        serde_json::from_str(&query).map_err(|e| anyhow::anyhow!("invalid query JSON: {}", e))?;
    let result = runtime.search(&qvec, k)?;
    print_result(&result);
    Ok(())
}
