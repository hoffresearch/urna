//! The clap surface: `Cli` + `Commands`; `main.rs` keeps the dispatch.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

use crate::cmd;

/// The help footer every build shares; the terminal-ui lines are added only
/// when the `tui` feature (`setup`, `tui`) is compiled in.
macro_rules! help_footer {
    () => {
        "start here (the five verbs that cover the loop):\n  build     creates the base       rows + embedding model in, one .urna out     urna build --spec corpus.toml\n  ask       queries it             text in, one cited answer out                urna ask corpus.urna \"question\"\n  retrieve  results for a program  json/jsonl of cited spans, exact score       urna retrieve corpus.urna \"question\" --format jsonl\n  cite      resolves the source    a urna:// citation back to its stored text   urna cite corpus.urna 'urna://...'\n  validate  proves the file        every checksum, every hash, the contract     urna validate corpus.urna\n\nverb groups:\n  engine  inspect, validate, stats, media, search, search-ann, search-graph,\n          search-space, search-text, benchmark, cite, doctor  (file + vector in, hits out; python only in search-text and doctor)\n  agent   ask, retrieve, build  (text or spec in, cited answers out; shells out to the offline python embedder / forge)\n"
    };
}

#[cfg(feature = "tui")]
const AFTER_HELP: &str = concat!(
    help_footer!(),
    "  setup   setup, tui  (the installer every channel ends in, and the terminal explorer)\n\n",
    "first run: urna setup (a bare `urna` on a terminal opens the explorer)\n",
    "a corpus to try: demo/starter/ in the repo (urna build --spec demo/starter/corpus.toml)"
);

#[cfg(not(feature = "tui"))]
const AFTER_HELP: &str = concat!(
    help_footer!(),
    "\nthis build has no terminal ui (`setup`, `tui`): it was compiled without the `tui` feature\n",
    "a corpus to try: demo/starter/ in the repo (urna build --spec demo/starter/corpus.toml)"
);

/// Two products share one binary and one engine. The ENGINE verbs take a
/// `.urna` file and (where relevant) a query VECTOR; none of them needs
/// python except `search-text` (the sentence-transformers embedder) and
/// `doctor` (it probes the python env). The AGENT verbs (`ask`,
/// `retrieve`, `build`) take TEXT or a build spec,
/// shell out to the offline python embedder / forge, and speak in cited
/// answers. `--help` lists the engine first, the agent verbs last, and
/// tags each group in its summary line; the implementations mirror the
/// split (`cmd/*` vs `cmd/agent/*`).
#[derive(Parser)]
#[command(name = "urna")]
#[command(version = cmd::buildid::NAME)]
#[command(
    about = "urna: single-file, memory-mapped, hash-verified vector database with stable citations",
    long_about = None,
    after_help = AFTER_HELP
)]
pub struct Cli {
    /// none on a terminal opens `urna tui`; none in a pipe prints this help.
    #[command(subcommand)]
    pub command: Option<Commands>,
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
    /// [engine] Search by raw text - embeds the query with the model declared in
    /// the manifest, then routes by what the file carries: hybrid when it
    /// has a bm25 section, hnsw when it has an hnsw section, else exact.
    /// One of the two engine verbs that run python (the other is
    /// `doctor`). Validates the
    /// embedder's model_hash against the manifest before running search;
    /// a mismatch fails with a typed error rather than returning
    /// silently-bad results.
    #[command(display_order = 6)]
    SearchText {
        file: PathBuf,
        query: String,
        #[arg(short, long, default_value = "10")]
        k: i32,
        /// Override the embedder script. Default: `rust/bridge/python/urna/embed/searchtxt.py` in a
        /// checkout, else the installed payload's `<data root>/urna/python/urna/embed/searchtxt.py`.
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
        /// Accept a corpus whose `model_hash` is the legacy
        /// zero-placeholder (pre-Phase-3 builds), which has no
        /// fingerprint to compare against. The search is then
        /// cosine-valid only IF you genuinely use the same embedding
        /// model; nothing checks it. A real fingerprint that disagrees
        /// with the embedder is never skipped. Prefer rebuilding.
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
        /// HNSW beam width. The beam that runs is max(ef, k, the file's
        /// ef_construction): values below that floor (400 for python
        /// builds) change nothing; the `candidates:` line shows the beam.
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
    /// rust/bridge/python/urna/entry/specbuild.py; the build is officially a python
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
        /// If set, also benchmark `search_ann` with the given ef. The beam
        /// that runs is max(ef, k, the file's ef_construction) and is
        /// printed next to the ef.
        #[arg(long)]
        ann: Option<usize>,
        /// Force a "madvise-cold" cache between queries by calling
        /// posix_madvise(MADV_DONTNEED) on the mmap. Approximates the
        /// first hit post-boot - but it's a hint, not a guarantee.
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
    /// [setup] Interactive installer: the offline embedder payload and a
    /// python env (numpy + tokenizers), proven by the doctor checks. plain
    /// output with --yes or without a terminal; --model adds catalog models.
    /// exit: 0 ready, 2..=6 a doctor check, 10 download, 11 checksum, 12
    /// unpack, 13 python env, 14 blocked, 15 a model install.
    #[cfg(feature = "tui")]
    #[command(display_order = 30)]
    Setup {
        /// Run the default plan with no questions (ci, scripts).
        #[arg(long, short)]
        yes: bool,
        /// Release tag to fetch the payload from (default: this binary's).
        #[arg(long)]
        version: Option<String>,
        /// Reinstall the payload even when one is present.
        #[arg(long)]
        force: bool,
        /// Skip the embedder payload step.
        #[arg(long)]
        no_payload: bool,
        /// Skip the python env step (bring your own via URNA_PYTHON).
        #[arg(long)]
        no_python: bool,
        /// Install a model from the payload's catalog: its packages into the
        /// managed venv, its pinned weights into the hugging face cache.
        /// repeatable; `all` picks every offered model. naming one is the
        /// consent to download it.
        #[arg(long = "model", value_name = "NAME")]
        models: Vec<String>,
        /// Allow a chosen model's repo code to run (a separate consent from
        /// the download). repeatable.
        #[arg(long, value_name = "NAME")]
        allow_remote_code: Vec<String>,
        /// Remove the payload and the env setup created (never the binary;
        /// the hugging face cache is shared and stays).
        #[arg(long, conflicts_with_all = ["force", "no_payload", "no_python", "version", "models"])]
        uninstall: bool,
    },
    /// [setup] Terminal explorer: open a .urna, read its sections, ask it,
    /// check the install. what a bare `urna` opens on a terminal.
    #[cfg(feature = "tui")]
    #[command(display_order = 31)]
    Tui { file: Option<PathBuf> },
}

#[cfg(test)]
mod tests {
    use super::Cli;
    use clap::CommandFactory;

    #[test]
    fn the_help_footer_names_the_terminal_ui_only_when_it_is_built() {
        let help = Cli::command().render_long_help().to_string();
        assert!(help.contains("start here"), "{help}");
        assert_eq!(
            help.contains("first run: urna setup"),
            cfg!(feature = "tui"),
            "{help}"
        );
        assert_eq!(
            help.contains("has no terminal ui"),
            !cfg!(feature = "tui"),
            "{help}"
        );
    }
}
