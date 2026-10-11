//! `urna setup`: the installer every channel ends in. brew, npm, crates.io
//! and the tarballs all ship the bare binary; this lays down what the
//! binary cannot carry (the offline embedder payload and a python with
//! numpy + tokenizers), installs the catalog models the user chooses, and
//! proves the result with the doctor checks.
//!
//! with a terminal on both ends it runs inline (the screen stays in the
//! scrollback when it ends): splash, scan, plan, install, verify. with
//! `--yes` or without a terminal it prints plain lines (`text`). both read
//! the same worker (`job`).

pub mod job;
pub mod models;
pub mod net;
pub mod plan;
pub mod scan;
mod screen;
pub mod text;
mod ui;
pub mod unpack;
pub mod venv;

use std::io::IsTerminal;
use std::time::Duration;

use crossterm::event::{Event, KeyEventKind};

use crate::tui::fx::Fx;
use crate::tui::term;

pub use plan::Opts;

/// Runs setup and returns the process exit code: 0 ready, 2..=6 a doctor
/// failure after the steps ran, 10..=15 a step failure (`job::codes`).
pub fn run(opts: Opts) -> anyhow::Result<i32> {
    let tty = !opts.yes && std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    if tty {
        interactive(opts)
    } else {
        Ok(text::run(&opts))
    }
}

/// `urna setup --uninstall`: removes what setup laid down (payload + venv),
/// never the binary (its package manager owns it).
pub fn uninstall() -> anyhow::Result<i32> {
    let Some(home) = crate::cmd::paths::urna_home() else {
        anyhow::bail!("no data dir (set URNA_DATA_DIR)");
    };
    for sub in [crate::cmd::payload::DIR, "venv"] {
        let p = home.join(sub);
        if p.exists() {
            std::fs::remove_dir_all(&p)?;
            println!("removed {}", p.display());
        }
    }
    for name in unpack::TOP_LEVEL {
        let p = home.join(name);
        if p.is_file() {
            std::fs::remove_file(&p)?;
            println!("removed {}", p.display());
        }
    }
    let (removed, kept) = unpack::remove_legacy(&home);
    for p in removed {
        println!("removed {}", p.display());
    }
    if !kept.is_empty() {
        let kept: Vec<String> = kept.iter().map(|p| p.display().to_string()).collect();
        anyhow::bail!("could not remove {}", kept.join(", "));
    }
    println!(
        "urna setup: uninstalled the payload and the env; the binary stays with its package manager"
    );
    Ok(0)
}

/// Runs the interactive installer; returns the exit code.
fn interactive(opts: Opts) -> anyhow::Result<i32> {
    let (_, rows) = crossterm::terminal::size().unwrap_or((80, 24));
    let height = rows.saturating_sub(2).clamp(12, 22);
    let mut terminal = term::inline(height)?;
    let mut ui = ui::Ui::new(opts);
    let res = (|| -> anyhow::Result<i32> {
        loop {
            ui.pump();
            let dt = ui.clock.tick();
            ui.spinner.tick(dt);
            terminal.draw(|f| screen::draw(f, &mut ui, dt))?;
            if let Some(code) = ui.quit {
                return Ok(code);
            }
            if let Some(Event::Key(k)) = term::next_event()?
                && k.kind == KeyEventKind::Press
            {
                ui.key(k.code, k.modifiers);
            }
        }
    })();
    // last frame without effects, so the scrollback keeps a clean record.
    ui.fx = Fx::default();
    ui.entered = false;
    let _ = terminal.draw(|f| screen::draw(f, &mut ui, Duration::ZERO));
    term::leave_inline(&mut terminal)?;
    let code = res?;
    if code == 130 {
        println!("urna setup: cancelled");
    }
    Ok(code)
}
