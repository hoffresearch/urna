//! Dispatch only: the clap surface lives in `cli.rs`, implementations in
//! `cmd/*` (one module per subcommand).

use anyhow::Result;
use clap::Parser;

mod cli;
mod cmd;

use cli::{Cli, Commands};

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Inspect { file, json } => cmd::inspect::run(file, json),
        Commands::Validate { file } => cmd::validate::run(file),
        Commands::Media { file, export } => cmd::media::run(file, export),
        Commands::Search { file, query, k } => cmd::search::run(file, query, k),
        Commands::SearchText {
            file,
            query,
            k,
            embedder,
            candidates,
            model_path,
            skip_model_hash_check,
        } => cmd::search_text::run(
            file,
            query,
            k,
            embedder,
            candidates,
            model_path,
            skip_model_hash_check,
        ),
        Commands::SearchAnn { file, query, k, ef } => cmd::search_ann::run(file, query, k, ef),
        Commands::SearchGraph {
            file,
            query,
            k,
            hops,
            ef,
        } => cmd::search_graph::run(file, query, k, hops, ef),
        Commands::SearchSpace {
            file,
            query,
            space,
            k,
            expect_model_hash,
        } => cmd::search_space::run(file, query, space, k, expect_model_hash),
        Commands::Build {
            spec,
            sample,
            models,
            out_dir,
            cache_dir,
            resume,
            rebuild_only,
            dry_run,
            allow_heavy,
        } => cmd::agent::build::run(
            spec,
            sample,
            models,
            out_dir,
            cache_dir,
            resume,
            rebuild_only,
            dry_run,
            allow_heavy,
        ),
        Commands::Benchmark {
            file,
            queries,
            k,
            ann,
            madvise_cold,
            space,
        } => cmd::benchmark::run(file, queries, k, ann, madvise_cold, space),
        Commands::Stats { file } => cmd::stats::run(file),
        Commands::Cite { file, citation } => cmd::cite::run(file, citation),
        Commands::Ask {
            file,
            query,
            k,
            disclose,
            embedder,
            candidates,
            model_path,
        } => cmd::agent::ask::run(file, query, k, disclose, embedder, candidates, model_path),
        Commands::Retrieve {
            file,
            query,
            k,
            format,
            embedder,
            candidates,
            model_path,
        } => cmd::agent::retrieve::run(file, query, k, format, embedder, candidates, model_path),
        Commands::Doctor => cmd::doctor::run(),
    }
}
