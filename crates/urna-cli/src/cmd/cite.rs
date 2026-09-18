//! `urna cite <file> urna://...` — resolve a citation URI into the
//! canonical text, source span, and verifying hashes.

use anyhow::Result;
use std::path::PathBuf;

pub fn run(file: PathBuf, citation: String) -> Result<()> {
    let rest = citation
        .strip_prefix("urna://")
        .ok_or_else(|| anyhow::anyhow!("citation must start with 'urna://': {}", citation))?;
    let (content_hash_part, chunk_id_part) = rest
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("citation missing chunk_id: {}", citation))?;

    let chunk_id = match chunk_id_part.split_once('/') {
        Some((first, _)) => first,
        None => chunk_id_part,
    };
    let expected_content_hash =
        format!("sha256:{}", content_hash_part.trim_start_matches("sha256:"));

    let data = std::fs::read(&file)?;
    let view = urna_format::UrnaView::from_bytes(&data)?;
    let actual_content_hash = view.content_hash_hex()?;
    if actual_content_hash != expected_content_hash {
        anyhow::bail!(
            "content_hash mismatch: citation says {} but file is {}",
            expected_content_hash,
            actual_content_hash
        );
    }

    let n = view.header.n_chunks as usize;
    let ids = urna_format::sections::decode_chunk_ids(
        &view.decoded_section(urna_format::layout::SECTION_CHUNK_IDS)?,
        n,
    )?;
    let idx = ids
        .iter()
        .position(|id| id == chunk_id)
        .ok_or_else(|| anyhow::anyhow!("chunk_id {} not found in file", chunk_id))?;

    let texts = urna_format::sections::decode_chunks_canonical(
        &view.decoded_section(urna_format::layout::SECTION_CHUNKS_CANONICAL)?,
        n,
    )?;
    let spans = urna_format::sections::decode_chunks_original_spans(
        &view.decoded_section(urna_format::layout::SECTION_CHUNKS_ORIGINAL_SPANS)?,
        n,
    )?;

    let span = &spans[idx];
    println!("citation_id:  {}", citation);
    println!("file:         {}", file.display());
    println!("file_hash:    {}", view.file_hash_hex());
    println!("content_hash: {}", actual_content_hash);
    println!("chunk_id:     {}", chunk_id);
    println!("source_uri:   {}", span.source_uri);
    println!("byte_start:   {}", span.byte_start);
    println!("byte_end:     {}", span.byte_end);
    println!("text:");
    println!("{}", texts[idx]);
    Ok(())
}
