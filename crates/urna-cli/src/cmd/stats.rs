//! `urna stats <file>` — size, dim, dtype, hashes, per-section sizes,
//! SIMD backend.

use anyhow::Result;
use std::path::PathBuf;

use super::util::encoding_name;

pub fn run(file: PathBuf) -> Result<()> {
    let data = std::fs::read(&file)?;
    let view = urna_format::UrnaView::from_bytes(&data)?;
    let size = std::fs::metadata(&file)?.len();
    println!("file:         {}", file.display());
    println!("size:         {} bytes ({:.2} MB)", size, size as f64 / 1e6);
    println!("chunks:       {}", view.header.n_chunks);
    println!("embeddings:   {}", view.header.n_embeddings);
    println!("dim:          {}", view.header.embedding_dim);
    // matryoshka disclosure: when the file was built with prefix truncation,
    // surface the effective prefix dim and the full source dim so the
    // size/recall tradeoff is visible. content_hash is over the truncated
    // embeddings, so a citation is tied to this mrl_dim.
    if let (Some(mrl), Some(full)) = (view.manifest.mrl_dim, view.manifest.full_dim) {
        println!("mrl_dim:      {} (prefix of {})", mrl, full);
        println!("full_dim:     {}", full);
    }
    println!("dtype:        {}", view.manifest.dtype);
    println!("metric:       {}", view.manifest.metric);
    println!("score_type:   {}", view.manifest.score_type);
    println!("normalize:    {}", view.manifest.normalize);
    println!("index_type:   {}", view.manifest.index_type);
    println!("rerank:       {}", view.manifest.rerank_policy);
    println!("model:        {}", view.manifest.embedding_model);
    println!("model_hash:   {}", view.manifest.model_hash);
    println!("chunker:      {}", view.manifest.chunker_version);
    println!("supports_ann: {}", view.manifest.capabilities.supports_ann);
    println!("supports_bm25:{}", view.manifest.capabilities.supports_bm25);
    println!(
        "simd_backend: {}",
        urna_runtime::simd::detect_backend().name()
    );
    println!("sections:     {}", view.section_table.len());
    for entry in &view.section_table {
        let name = urna_format::layout::section_name(entry.section_id).unwrap_or("unknown");
        let enc = encoding_name(entry.encoding);
        println!(
            "  0x{:02x} {:<24} encoding={:<8} {} bytes",
            entry.section_id, name, enc, entry.size
        );
    }
    // named multimodal spaces (0x15): each is a separately queryable vector
    // band with its own model_hash gate; surfaced here so an operator can see
    // what `search-space` can address without `inspect --json`.
    if let Ok(entry) = view.entry(urna_format::layout::SECTION_SPACE_TABLE) {
        let _ = entry;
        if let Ok(payload) = view.decoded_section(urna_format::layout::SECTION_SPACE_TABLE) {
            if let Ok(spaces) = urna_format::sections::decode_space_table(&payload) {
                println!("spaces:       {}", spaces.len());
                for s in &spaces {
                    println!(
                        "  {:<24} dim={:<5} dtype={:<8} n={:<7} model_hash={}",
                        s.name,
                        s.dim,
                        s.dtype_str(),
                        s.n_vectors,
                        s.model_hash
                    );
                }
            }
        }
    }
    println!("file_hash:    {}", view.file_hash_hex());
    println!("content_hash: {}", view.content_hash_hex()?);
    Ok(())
}
