//! `urna search-space <file> <qvec> --space NAME -k K` — exact search over
//! one named multimodal space (0x15 + its band). The query vector must be
//! embedded with the model the space's model_hash fingerprints and have the
//! space's dim; the runtime's typed errors (SpaceNotFound, dim mismatch,
//! SpaceModelMismatch via --expect-model-hash) fail loudly, never fall back
//! to the text path.

use anyhow::Result;
use std::path::PathBuf;

use super::util::print_result;

pub fn run(
    file: PathBuf,
    query: String,
    space: String,
    k: i32,
    expect_model_hash: Option<String>,
) -> Result<()> {
    let qvec: Vec<f32> = serde_json::from_str(&query)
        .map_err(|e| anyhow::anyhow!("query must be a JSON array of f32: {}", e))?;
    let runtime = urna_runtime::MmapUrnaFile::open(&file)?;
    let result = runtime.search_space(&space, &qvec, k, expect_model_hash.as_deref())?;
    println!("space:        {}", space);
    print_result(&result);
    Ok(())
}
