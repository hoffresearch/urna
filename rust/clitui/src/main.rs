//! Dispatch only: the clap surface lives in `cli.rs`, implementations in
//! `cmd/*` (one module per subcommand).

use anyhow::Result;
use clap::Parser;

mod cli;
mod cmd;
#[cfg(feature = "tui")]
mod tui;

use cli::{Cli, Commands};

fn main() -> Result<()> {
    let cli = Cli::parse();
    let Some(command) = cli.command else {
        return no_command();
    };
    match command {
        Commands::Inspect { file, json } => cmd::inspect::run(file, json),
        Commands::Validate { file } => cmd::validate::run(file),
        Commands::Media { file, export } => cmd::media::run(file, export),
        Commands::Scalars {
            file,
            rows,
            schema,
            out,
            dataset,
            media,
        } => cmd::scalars::run(file, rows, schema, out, dataset, media),
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
        Commands::Doctor { build } => cmd::doctor::run(build),
        #[cfg(feature = "tui")]
        Commands::Setup {
            yes,
            version,
            force,
            no_payload,
            python,
            no_python,
            no_sample,
            models,
            allow_remote_code,
            uninstall,
        } => {
            let code = if uninstall {
                tui::setup::uninstall()?
            } else {
                tui::setup::run(tui::setup::Opts {
                    yes,
                    version,
                    force,
                    no_payload,
                    python,
                    no_python,
                    no_sample,
                    models,
                    allow_remote_code,
                })?
            };
            std::process::exit(code)
        }
        #[cfg(feature = "tui")]
        Commands::Tui {
            file,
            ask,
            snapshot,
            eval,
        } => {
            let code = match (file, snapshot, eval) {
                (Some(f), Some(q), _) => tui::app::headless::snapshot(&f, &q)?,
                (Some(f), _, Some(n)) => tui::app::headless::eval(&f, n)?,
                (file, _, _) => tui::app::run(file, ask)?,
            };
            std::process::exit(code)
        }
    }
}

/// A bare `urna`: the explorer when a person is at a terminal, the help
/// (exit 2, as clap does for a missing subcommand) for scripts and pipes.
fn no_command() -> Result<()> {
    #[cfg(feature = "tui")]
    if std::io::IsTerminal::is_terminal(&std::io::stdin())
        && std::io::IsTerminal::is_terminal(&std::io::stdout())
    {
        std::process::exit(tui::app::run(None, None)?);
    }
    let _ = <Cli as clap::CommandFactory>::command().print_help();
    std::process::exit(2);
}
