//! The clap surface: `Cli` + `Commands`. Split from `main.rs` (which keeps
//! the dispatch) so both stay under the 300-line crate guard.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

use crate::cmd;

/// Two products share one binary and one engine. The ENGINE verbs take a
/// `.urna` file and (where relevant) a query VECTOR; they never run python.
/// The AGENT verbs (`ask`, `retrieve`, `build`) take TEXT or a build spec,
/// shell out to the offline python embedder / forge, and speak in cited
/// answers. `--help` lists the engine first, the agent verbs last, and
/// tags each group in its summary line; the implementations mirror the
/// split (`cmd/*` vs `cmd/agent/*`).
#[derive(Parser)]
#[command(name = "urna")]
#[command(version)]
#[command(
    about = "urna: single-file, memory-mapped, hash-verified vector database with stable citations",
    long_about = None,
    after_help = "start here (the five verbs that cover the loop):\n  build     creates the base       rows + embedding model in, one .urna out     urna build --spec corpus.toml\n  ask       queries it             text in, one cited answer out                urna ask corpus.urna \"question\"\n  retrieve  results for a program  json/jsonl of cited spans, exact score       urna retrieve corpus.urna \"question\" --format jsonl\n  cite      resolves the source    a urna:// citation back to its stored text   urna cite corpus.urna 'urna://...'\n  validate  proves the file        every checksum, every hash, the contract     urna validate corpus.urna\n\nverb groups:\n  engine  inspect, validate, stats, media, search, search-ann, search-graph,\n          search-space, search-text, benchmark, cite, doctor  (file + vector in, hits out; no python)\n  agent   ask, retrieve, build  (text or spec in, cited answers out; shells out to the offline python embedder / forge)\n\na corpus to try: examples/quickstart/ in the repo (urna build --spec examples/quickstart/corpus.toml)"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// [engine] Inspect file metadata, manifest, and section table.
    #[command(display_order = 1)]
    Inspect {
        file: PathBuf,
        /// Emit as JSON instead of the human-readable layout. Schema:
        /// `{magic, version_major, version_minor, format_version,
        /// schema_version, embedding_dim, n_chunks, n_embeddings,
        /// file_size, manifest, sections[], blobs, spaces[], file_hash,
        /// content_hash, simd_backend}`.
        #[arg(long)]
        json: bool,
    },
    /// [engine] Validate file integrity (magic, checksums, hashes, manifest, contract).
    #[command(display_order = 2)]
    Validate { file: PathBuf },
    /// [engine] List the media blobs a corpus references; --export writes the
    /// inlined (0x17) ones back to standalone files, hash-verified.
    #[command(display_order = 4)]
    Media {
        file: PathBuf,
        #[arg(long)]
        export: Option<PathBuf>,
    },
    /// [engine] Search a `.urna` file with a JSON-array query vector (exact path).
    #[command(display_order = 5)]
    Search {
        file: PathBuf,
        query: String,
        #[arg(short, long, default_value = "10")]
        k: i32,
    },
    /// [engine] Search by raw text — embeds the query with the model declared in
    /// the manifest, then runs the appropriate vector path. Honors the
    /// declared `index_type` (exact / hnsw / hybrid). Validates the
    /// embedder's model_hash against the manifest before running search;
    /// a mismatch fails with a typed error rather than returning
    /// silently-bad results.
    #[command(display_order = 6)]
    SearchText {
        file: PathBuf,
        query: String,
        #[arg(short, long, default_value = "10")]
        k: i32,
        /// Override the embedder script. Default: `python/embed_query.py`.
        #[arg(long)]
        embedder: Option<PathBuf>,
        /// `ef` (HNSW) / candidates-per-path (hybrid). Default: 4*k or 64.
        #[arg(long)]
        candidates: Option<usize>,
        /// Local path to the model snapshot dir. Use this for fully
        /// offline operation: copy the model dir alongside the .urna,
        /// pass --model-path at every search. Without this, the
        /// embedder resolves the model from the sentence-transformers
        /// cache (requires network on first use).
        #[arg(long)]
        model_path: Option<PathBuf>,
        /// Skip model_hash validation. ONLY use when intentionally
        /// running search-text against a corpus whose `model_hash`
        /// is the legacy zero-placeholder (pre-Phase-3 builds). In
        /// that case the search is still cosine-valid IF the user
        /// genuinely uses the same embedding model — but there is
        /// no guarantee. Prefer rebuilding the corpus.
        #[arg(long)]
        skip_model_hash_check: bool,
    },
    /// [engine] Force the ANN (HNSW) path. Falls back to exact if the file has
    /// no HNSW section.
    #[command(display_order = 7)]
    SearchAnn {
        file: PathBuf,
        query: String,
        #[arg(short, long, default_value = "10")]
        k: i32,
        #[arg(long, default_value = "100")]
        ef: usize,
    },
    /// [engine] Graph search: seed from the exact-cosine top-`ef`, expand a bounded
    /// bfs over the chunk-to-chunk graph, then exact-rerank the union. The
    /// graph only generates candidates; the score is real cosine. Falls back
    /// to exact if the file has no graph_adjacency section.
    #[command(display_order = 8)]
    SearchGraph {
        file: PathBuf,
        query: String,
        #[arg(short, long, default_value = "10")]
        k: i32,
        #[arg(long, default_value = "1")]
        hops: usize,
        #[arg(long, default_value = "100")]
        ef: usize,
    },
    /// [engine] Exact search over one NAMED multimodal space (0x15 band). The query
    /// vector must be embedded with the space's model and have the space's
    /// dim; mismatches are typed errors, never a silent text-path fallback.
    #[command(display_order = 9)]
    SearchSpace {
        file: PathBuf,
        /// JSON array of f32 at the space's dim.
        query: String,
        /// Space name as listed by `stats` / `inspect --json` (e.g. "wemm-2b@256").
        #[arg(long)]
        space: String,
        #[arg(short, long, default_value = "10")]
        k: i32,
        /// Also assert the space's model_hash equals this value.
        #[arg(long)]
        expect_model_hash: Option<String>,
    },
    /// [agent] Declarative corpus build from a TOML/JSON spec (launcher over
    /// python/tools/urna_forge.py; the build is officially a python
    /// frontend). Streams the tool's output and propagates its exit code.
    #[command(display_order = 20)]
    Build {
        /// Build spec path (.toml or .json).
        #[arg(long)]
        spec: PathBuf,
        /// Evenly-spaced row subset for pilots.
        #[arg(long)]
        sample: Option<usize>,
        /// Comma-separated preset subset (e.g. "potion,wemm-2b").
        #[arg(long)]
        models: Option<String>,
        /// Override the spec's [output].dir.
        #[arg(long)]
        out_dir: Option<PathBuf>,
        /// Override the spec's [output].cache_dir (the shared embed cache
        /// root; else URNA_CACHE_DIR, else ${XDG_CACHE_HOME:-~/.cache}/urna).
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        /// Resume from per-stage state after an interrupted build.
        #[arg(long)]
        resume: bool,
        /// Re-emit byte-identically from cached vectors (L3 check).
        #[arg(long)]
        rebuild_only: bool,
        /// Resolve the plan + dependency status without loading models.
        #[arg(long)]
        dry_run: bool,
        /// Allow presets flagged too heavy for this machine (wemm-4b/9b).
        #[arg(long)]
        allow_heavy: bool,
    },
    /// [engine] Benchmark exact flat search latency.
    #[command(display_order = 10)]
    Benchmark {
        file: PathBuf,
        #[arg(short, long, default_value = "100")]
        queries: usize,
        #[arg(short, long, default_value = "10")]
        k: i32,
        /// If set, also benchmark `search_ann` with the given ef.
        #[arg(long)]
        ann: Option<usize>,
        /// Force a "madvise-cold" cache between queries by calling
        /// posix_madvise(MADV_DONTNEED) on the mmap. Approximates the
        /// first hit pos-boot — but it's a hint, not a guarantee.
        /// See MmapUrnaFile::madvise_cold for caveats.
        #[arg(long)]
        madvise_cold: bool,
        /// Benchmark the named multimodal space instead of the default path.
        #[arg(long)]
        space: Option<String>,
    },
    /// [engine] Show file stats.
    #[command(display_order = 3)]
    Stats { file: PathBuf },
    /// [engine] Resolve a `urna://content_hash/chunk_id` citation into the
    /// canonical text and original span for the chunk.
    #[command(display_order = 11)]
    Cite {
        file: PathBuf,
        /// `urna://<content_hash>/<chunk_id>` URI.
        citation: String,
    },
    /// [agent] Flagship verb: text query in, cited answer out. embeds the query
    /// OFFLINE (potion for potion corpora; the registry embedder for any
    /// other manifest model), validates model_hash against the manifest,
    /// routes by manifest capability, and prints the cited canonical text
    /// with a urna:// citation. `--disclose explain` adds the rerank-source
    /// honesty line (real cosine vs real cosine at stored precision). cite is
    /// tier-1: the printed text is the stored canonical text, never an
    /// original-byte reopen.
    #[command(display_order = 21)]
    Ask {
        file: PathBuf,
        query: String,
        #[arg(short, long, default_value = "10")]
        k: i32,
        /// Disclosure level: `answer` (cited text + urna:// only, default)
        /// or `explain` (also the rerank-source honesty line + route).
        #[arg(long, value_enum, default_value = "answer")]
        disclose: cmd::agent::ask::Disclose,
        /// Override the offline embedder. default: routed by manifest model.
        #[arg(long)]
        embedder: Option<PathBuf>,
        /// `ef` (HNSW) / candidates-per-path (hybrid). Default: 4*k or 64.
        #[arg(long)]
        candidates: Option<usize>,
        /// Local path to the model dir (fully offline).
        #[arg(long)]
        model_path: Option<PathBuf>,
    },
    /// [agent] Agent-shaped flagship: text query in, a json/jsonl answer-pack of
    /// cited spans out. each hit's `score` IS the exact-cosine rerank value.
    /// embeds OFFLINE with the same routed embedder + model_hash gate as
    /// `ask`. `text` is the stored canonical text (TIER-1), the citation_id
    /// round-trips through `cite`; never an original-byte reopen.
    #[command(display_order = 22)]
    Retrieve {
        file: PathBuf,
        query: String,
        #[arg(short, long, default_value = "10")]
        k: i32,
        /// Output format: `jsonl` (one object per line, default) or `json`.
        #[arg(long, value_enum, default_value = "jsonl")]
        format: cmd::agent::retrieve::Format,
        /// Override the offline embedder. default: routed by manifest model.
        #[arg(long)]
        embedder: Option<PathBuf>,
        #[arg(long)]
        candidates: Option<usize>,
        /// Local path to the model dir (fully offline).
        #[arg(long)]
        model_path: Option<PathBuf>,
    },
    /// [engine] Post-install health check: versions, simd backend, python deps, and
    /// one real offline potion embed. exits with a typed code (0 ok, 2 python
    /// missing, 3 python deps missing, 4 embedder missing, 5 potion table
    /// missing, 6 embedder run failed).
    #[command(display_order = 12)]
    Doctor,
}
