//! mmap-backed runtime for `.urna` files.
//!
//! Owns the mmap so `MmapUrnaFile` is `'static`. Section metadata is parsed
//! once at open time using `UrnaView`, then the mmap is moved into `Self`
//! and embeddings are read directly from `&self._mmap[offset..]`.
//!
//! Supports float32 / float16 / int8 dtypes with a SIMD dispatcher
//! (AVX2 / NEON / scalar). Optional ANN (`hnsw`) and lexical (`bm25`)
//! sections rerank into the exact cosine path so the final score is
//! always the real cosine value.

pub mod ann;
mod blobs;
pub mod bm25;
mod dtype;
pub mod error;
pub mod graph;
mod inspect;
mod materialize;
mod mmap_cold;
mod mmap_file;
mod order;
mod rerank;
mod search;
pub mod simd;
mod space_search;
mod spaces;

pub use dtype::DType;
pub use error::RuntimeError;
pub use mmap_file::MmapUrnaFile;
pub use simd::SimdBackend;

#[derive(Clone, Debug, PartialEq)]
pub struct SearchHit {
    pub chunk_id: String,
    pub score: f32,
    pub score_type: &'static str,
    pub source_uri: String,
    pub offset_start: u64,
    pub offset_end: u64,
    pub embedding_model: String,
    pub index_type: &'static str,
    pub reranked: bool,
    pub file_hash: String,
    pub content_hash: String,
    pub citation_id: String,
}

/// What precision the mandatory exact-cosine rerank read its vectors from.
/// the honesty backbone of `--disclose explain` and `retrieve()`: every
/// returned `score` IS a real cosine, but at WHICH precision depends on
/// whether a full-precision `embeddings_fp` (0x09) slab is present.
///
/// - `FullPrecision`: a 0x09 fp slab is present (or the stored dtype is
///   float32), so the rerank read full-precision vectors. "real cosine".
/// - `StoredPrecision`: no fp slab and the stored dtype is lossy
///   (float16/int8/int4), so the rerank read the stored quantized slab.
///   "real cosine at stored precision". still a real recomputed cosine,
///   never a candidate-generator proxy, but at the on-disk precision.
///
/// this enum is computed in the runtime from
/// `embeddings_fp_slab().is_some()` + the stored dtype and never lies: a
/// sub-int8 corpus without an fp source reports `StoredPrecision`, so a
/// newcomer is never shown a stored-precision number as full-precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RerankSourceKind {
    /// score recomputed against a full-precision source (0x09 fp slab, or
    /// the stored slab when the stored dtype is already float32).
    FullPrecision,
    /// score recomputed against the lossy stored slab (float16/int8/int4),
    /// no fp source present. disclosed as "real cosine at stored precision".
    StoredPrecision,
}

impl RerankSourceKind {
    /// classify the precision the rerank reads from by its effective dtype
    /// (the 0x09 fp slab's dtype when present, else the stored dtype). only
    /// float32 is full-precision; float16/int8/int4 are stored-precision.
    /// this is the single source of the honesty marker.
    pub fn from_dtype(dtype: DType) -> Self {
        match dtype {
            DType::Float32 => Self::FullPrecision,
            DType::Float16 | DType::Int8 | DType::Int4 => Self::StoredPrecision,
        }
    }

    /// the one-line honesty marker for `--disclose explain` / `retrieve()`.
    pub fn disclosure(self) -> &'static str {
        match self {
            Self::FullPrecision => "real cosine",
            Self::StoredPrecision => "real cosine at stored precision",
        }
    }
    /// stable machine token for the json answer-pack / SearchExplain.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FullPrecision => "full_precision",
            Self::StoredPrecision => "stored_precision",
        }
    }
}

/// ADDITIVE per-search provenance, attached to every `SearchResult`. zero
/// format bytes, no `URNA_FORMAT_VERSION` bump: it is computed entirely in
/// the runtime from the search path and the open file. feeds the lens
/// `--disclose explain` honesty line and the agent-native `retrieve()`
/// answer-pack. the load-bearing field is `rerank_source`.
///
/// candidate counts are the per-path candidate-set sizes the rerank saw
/// (0 when that path did not run). `recall_estimate` is `1.0` only on the
/// exact path (the recall=1.0 ground truth) and `NaN` on every candidate-
/// generating path (ann/hybrid/graph), mirroring `SearchResult::recall`:
/// we never claim a recall we did not measure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchExplain {
    /// the route taken: "exact" | "hnsw" | "graph" | "hybrid".
    pub route: &'static str,
    /// exact-flat candidates considered (the whole corpus on the exact path
    /// and the graph seed path; 0 when the path used an ann shortlist).
    pub exact_candidates: usize,
    /// ann shortlist size (hnsw / hybrid vector path); 0 otherwise.
    pub ann_candidates: usize,
    /// bm25 lexical shortlist size (hybrid path); 0 otherwise.
    pub bm25_candidates: usize,
    /// graph bfs frontier union size (graph path); 0 otherwise.
    pub graph_candidates: usize,
    /// how the candidate lists were combined: "none" | "rrf".
    pub fusion_mode: &'static str,
    /// the precision the mandatory exact rerank read from. the honesty
    /// backbone: see `RerankSourceKind`.
    pub rerank_source: RerankSourceKind,
    /// `1.0` on the exact path, `NaN` on every candidate-generating path
    /// (we never claim a recall we did not measure).
    pub recall_estimate: f32,
}

impl SearchExplain {
    /// the exact-flat path: the whole corpus scored, recall=1.0 ground
    /// truth, no fusion.
    pub(crate) fn exact(n: usize, src: RerankSourceKind) -> Self {
        Self {
            route: "exact",
            exact_candidates: n,
            ann_candidates: 0,
            bm25_candidates: 0,
            graph_candidates: 0,
            fusion_mode: "none",
            rerank_source: src,
            recall_estimate: 1.0,
        }
    }

    /// the hnsw path: an ann shortlist into the exact rerank, recall=NaN.
    pub(crate) fn hnsw(ann: usize, src: RerankSourceKind) -> Self {
        Self {
            route: "hnsw",
            exact_candidates: 0,
            ann_candidates: ann,
            bm25_candidates: 0,
            graph_candidates: 0,
            fusion_mode: "none",
            rerank_source: src,
            recall_estimate: f32::NAN,
        }
    }

    /// the graph path: exact-cosine top-ef seed, bfs frontier union into
    /// the exact rerank, recall=NaN.
    pub(crate) fn graph(seed: usize, frontier: usize, src: RerankSourceKind) -> Self {
        Self {
            route: "graph",
            exact_candidates: seed,
            ann_candidates: 0,
            bm25_candidates: 0,
            graph_candidates: frontier,
            fusion_mode: "none",
            rerank_source: src,
            recall_estimate: f32::NAN,
        }
    }

    /// the hybrid path: vector (ann or exact) ∪ bm25, rrf fusion into the
    /// exact rerank, recall=NaN.
    pub(crate) fn hybrid(ann: usize, exact: usize, bm25: usize, src: RerankSourceKind) -> Self {
        Self {
            route: "hybrid",
            exact_candidates: exact,
            ann_candidates: ann,
            bm25_candidates: bm25,
            graph_candidates: 0,
            fusion_mode: "rrf",
            rerank_source: src,
            recall_estimate: f32::NAN,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    pub hits: Vec<SearchHit>,
    pub query_time_ms: f64,
    pub index_type: &'static str,
    pub recall: f32,
    pub truncated: bool,
    pub k_requested: i32,
    pub k_returned: usize,
    /// ADDITIVE per-search provenance (route, candidate counts, fusion
    /// mode, rerank-source honesty marker, recall estimate). no format
    /// bytes; computed in the runtime. see `SearchExplain`.
    pub explain: SearchExplain,
}
