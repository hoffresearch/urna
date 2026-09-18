#![no_main]
//! One section / payload codec per input, selected by the first byte, fed
//! the remaining bytes directly (no container around it), so each decoder
//! gets the fuzzer's full attention instead of sharing it with the header.

use libfuzzer_sys::fuzz_target;
use urna_format::encoding::{
    IntpackReader, decode_dedup_map, decode_fsst_payload, decode_payload,
    decode_txt_streams_payload, decode_zstd_dict_payload,
};
use urna_format::sections::{
    decode_chunk_ids, decode_chunks_canonical, decode_chunks_original_spans,
    decode_intpack_repack, decode_provenance, decode_search_contract, decode_txt_streams,
};
use urna_format::{
    Int4EmbeddingsView, Int8EmbeddingsView, decode_blob_data_table, decode_blob_refs,
    decode_blob_span_overlay, decode_graph_adjacency, decode_space_table,
};

fuzz_target!(|data: &[u8]| {
    let Some((&sel, rest)) = data.split_first() else {
        return;
    };
    // a small claimed count / dim so the "expected_count" style decoders
    // and the embeddings views get exercised with plausible shapes.
    let n = (sel >> 4) as usize;
    let dim = 64 * (1 + (sel & 3) as usize);
    match sel % 20 {
        0 => drop(decode_chunk_ids(rest, n)),
        1 => drop(decode_chunks_canonical(rest, n)),
        2 => drop(decode_chunks_original_spans(rest, n)),
        3 => drop(decode_provenance(rest)),
        4 => drop(decode_search_contract(rest)),
        5 => drop(decode_graph_adjacency(rest)),
        6 => drop(decode_blob_refs(rest)),
        7 => drop(decode_blob_span_overlay(rest)),
        8 => drop(decode_blob_data_table(rest)),
        9 => drop(decode_space_table(rest)),
        10 => {
            if let Ok(r) = IntpackReader::parse(rest) {
                for i in 0..r.len().min(4096) {
                    let _ = r.get(i);
                }
            }
        }
        11 => drop(decode_txt_streams(rest)),
        12 => drop(decode_intpack_repack(rest)),
        13 => drop(decode_fsst_payload(rest)),
        14 => {
            let (dict, body) = rest.split_at(rest.len() / 2);
            let _ = decode_zstd_dict_payload(body, dict);
        }
        15 => drop(decode_txt_streams_payload(rest)),
        16 => drop(decode_dedup_map(rest)),
        17 => {
            if let Ok(v) = Int8EmbeddingsView::parse(rest, n, dim) {
                for i in 0..v.n {
                    let _ = (v.row(i), v.scale(i));
                }
            }
        }
        18 => {
            if let Ok(v) = Int4EmbeddingsView::parse(rest, n, dim) {
                let mut scales = vec![0.0f32; v.blocks];
                for i in 0..v.n {
                    v.row_scales_into(i, &mut scales);
                }
            }
        }
        _ => {
            for enc in 0..=10u32 {
                let _ = decode_payload(enc, rest);
            }
        }
    }
});
