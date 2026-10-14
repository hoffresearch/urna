#![no_main]
//! `UrnaView::from_bytes` plus every section decoder and both hashes.

#[path = "reseal.rs"]
mod reseal;

use libfuzzer_sys::fuzz_target;
use urna_format::layout::*;
use urna_format::sections::{
    decode_chunk_ids, decode_chunks_canonical, decode_chunks_original_spans, decode_provenance,
    decode_search_contract,
};
use urna_format::{
    Int4EmbeddingsView, Int8EmbeddingsView, UrnaView, decode_blob_data_table, decode_blob_refs,
    decode_blob_span_overlay, decode_graph_adjacency, decode_space_table,
};

fuzz_target!(|data: &[u8]| {
    let Some(bytes) = reseal::split(data) else {
        return;
    };
    let Ok(view) = UrnaView::from_bytes(&bytes) else {
        return;
    };
    let n = view.header.n_chunks as usize;
    let dim = view.header.embedding_dim as usize;
    let _ = view.validate_embeddings_values();
    let _ = view.content_hash_hex();
    let _ = view.file_hash_hex();
    let _ = view.search_contract();
    for entry in view.section_table.clone() {
        let id = entry.section_id;
        let Ok(payload) = view.decoded_section(id) else {
            continue;
        };
        let p: &[u8] = &payload;
        match id {
            SECTION_CHUNK_IDS => drop(decode_chunk_ids(p, n)),
            SECTION_CHUNKS_CANONICAL => drop(decode_chunks_canonical(p, n)),
            SECTION_CHUNKS_ORIGINAL_SPANS => drop(decode_chunks_original_spans(p, n)),
            SECTION_PROVENANCE => drop(decode_provenance(p)),
            SECTION_SEARCH_CONTRACT => drop(decode_search_contract(p)),
            SECTION_EMBEDDINGS => {
                if let Ok(v) = Int8EmbeddingsView::parse(p, n, dim) {
                    for i in 0..v.n {
                        let _ = (v.row(i), v.scale(i));
                    }
                }
                if let Ok(v) = Int4EmbeddingsView::parse(p, n, dim) {
                    let mut scales = vec![0.0f32; v.blocks];
                    for i in 0..v.n {
                        v.row_scales_into(i, &mut scales);
                        let _ = v.row_codes(i);
                    }
                }
            }
            SECTION_GRAPH_ADJACENCY => drop(decode_graph_adjacency(p)),
            SECTION_BLOB_REFS => drop(decode_blob_refs(p)),
            SECTION_BLOB_SPAN_OVERLAY => drop(decode_blob_span_overlay(p)),
            SECTION_BLOB_DATA => drop(decode_blob_data_table(p)),
            SECTION_SPACE_TABLE => drop(decode_space_table(p)),
            _ => {}
        }
    }
});
