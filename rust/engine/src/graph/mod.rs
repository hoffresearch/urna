//! chunk-to-chunk graph runtime (G1). mirrors `ann`: a flat csr is parsed
//! off the graph_adjacency (0x0C) section at open time and walked with zero
//! allocation. `bounded_bfs` GENERATES candidate chunk ordinals seeded from
//! the exact-cosine top-k; the candidates are handed to the SAME mandatory
//! exact rerank (`score_subset`) so a graph edge can never leak into a
//! returned score. recall stays `f32::NAN` (we never lie).

mod csr;
mod traverse;

pub use csr::CsrIndex;
pub use traverse::Traversal;
