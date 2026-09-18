//! tiny BM25 inverted index for the hybrid search path.
//!
//! Tokenization is intentionally simple — lowercase, split on
//! Unicode-aware whitespace + punctuation, drop tokens with `len < 2`.
//! Multilingual corpora (PT-BR, EN, ES) get reasonable behavior without
//! a stemmer or stop list. A future version can pluggable-ize the
//! tokenizer if needed.
//!
//! The index is small enough that we keep it fully decoded in memory at
//! open time. On-disk layout (section `0x08`, payload version 1):
//!
//! ```text
//!   u32 LE  payload_version = 1
//!   f32 LE  k1                (BM25 hyperparameter)
//!   f32 LE  b                 (BM25 hyperparameter)
//!   f32 LE  avgdl             (average doc length)
//!   u32 LE  n_docs
//!   u32 LE  n_terms
//!   for d in 0..n_docs:
//!       u32 LE  dl_d          (document length in tokens)
//!   for t in 0..n_terms:
//!       u32 LE  len_t
//!       bytes   token_t       (UTF-8)
//!       u32 LE  df_t          (document frequency)
//!       for posting in postings_t:
//!           u32 LE  doc_id
//!           u32 LE  tf
//! ```

mod codec;
mod fusion;
mod index;
mod tokenize;

#[cfg(test)]
mod tests;

pub use fusion::rrf_union;
pub use index::{BM25_PAYLOAD_VERSION, Bm25Index, DEFAULT_B, DEFAULT_K1};
#[cfg(test)]
pub(crate) use tokenize::tokenize;
