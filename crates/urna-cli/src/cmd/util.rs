//! Shared CLI pretty-printers. Embedder discovery and the model gate live
//! in `embed_gate` (the one copy of the spawn protocol + three-layer gate).

pub fn print_result(result: &urna_runtime::SearchResult) {
    println!("index_type:   {}", result.index_type);
    if !result.recall.is_nan() {
        println!("recall:       {}", result.recall);
    } else {
        println!("recall:       (not computed; rerank guarantees real cosine)");
    }
    println!("truncated:    {}", result.truncated);
    println!("k_requested:  {}", result.k_requested);
    println!("k_returned:   {}", result.k_returned);
    println!("query_time:   {:.3} ms", result.query_time_ms);
    println!("hits:");
    for (i, hit) in result.hits.iter().enumerate() {
        println!(
            "  [{:3}] chunk_id={} score={:.6} score_type={} source_uri={} \
             offset={}-{} model={} index_type={} reranked={} file_hash={} \
             content_hash={} citation_id={}",
            i + 1,
            hit.chunk_id,
            hit.score,
            hit.score_type,
            hit.source_uri,
            hit.offset_start,
            hit.offset_end,
            hit.embedding_model,
            hit.index_type,
            hit.reranked,
            hit.file_hash,
            hit.content_hash,
            hit.citation_id,
        );
    }
}

pub fn encoding_name(e: u32) -> &'static str {
    match e {
        urna_format::layout::SECTION_ENCODING_RAW => "raw",
        urna_format::layout::SECTION_ENCODING_ZSTD => "zstd",
        urna_format::layout::SECTION_ENCODING_FLOAT16 => "float16",
        urna_format::layout::SECTION_ENCODING_INT8 => "int8",
        urna_format::layout::SECTION_ENCODING_INT4 => "int4",
        urna_format::layout::SECTION_ENCODING_INTPACK => "intpack",
        _ => "unknown",
    }
}
