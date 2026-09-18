#![no_main]
//! The runtime-owned index codecs: HNSW, BM25 and the graph CSR, each fed
//! raw bytes with a small claimed corpus shape.

use libfuzzer_sys::fuzz_target;
use urna_runtime::ann::HnswIndex;
use urna_runtime::bm25::Bm25Index;
use urna_runtime::graph::CsrIndex;

fuzz_target!(|data: &[u8]| {
    let Some((&sel, rest)) = data.split_first() else {
        return;
    };
    let n = 1 + (sel >> 2) as usize;
    let dim = 4 * (1 + (sel & 3) as usize);
    match sel % 3 {
        0 => {
            if let Ok(mut idx) = HnswIndex::from_bytes(rest, n, dim) {
                idx.attach_vectors(vec![0.5; n * dim]);
                let q = vec![0.25f32; dim];
                let _ = idx.search(&q, 16);
            }
        }
        1 => {
            if let Ok(idx) = Bm25Index::from_bytes(rest) {
                let _ = idx.search("alpha beta term1", 8);
            }
        }
        _ => {
            if let Ok(csr) = CsrIndex::from_bytes(rest, n) {
                for node in 0..csr.n_nodes() {
                    let _ = csr.neighbors(node);
                    let _ = csr.edge_type(node, 0);
                }
            }
        }
    }
});
