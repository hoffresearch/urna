//! `urna doctor` - post-install health check (issue #75).
//!
//! validates the whole install surface a user/agent depends on and exits
//! with a TYPED code so installers, CI, and support can branch on the
//! failure class instead of parsing text:
//!
//!   0  ok (scalar simd still exits 0, printed as a warning)
//!   2  python interpreter missing or not runnable
//!   3  python deps missing (numpy / tokenizers, needed by the embedder)
//!   4  potion embedder script not found (repo layout or data dir)
//!   5  potion model table missing or still a git-lfs pointer
//!   6  embedder run failed (non-zero exit or invalid json contract)
//!
//! checks: urna/format versions, simd backend, python interpreter, python
//! deps, potion embedder presence, potion table presence, and one real
//! offline embed of a fixed probe string. the embedder opens no socket, so
//! doctor itself stays offline-by-construction. the checks live in
//! `health` (shared with the terminal ui); this module only prints them,
//! colored on a terminal and plain when piped.

use super::health::{self, Level};
use super::tone::{self, Tone};

pub mod codes {
    pub const OK: i32 = 0;
    pub const PYTHON_MISSING: i32 = 2;
    pub const PYTHON_DEPS_MISSING: i32 = 3;
    pub const EMBEDDER_MISSING: i32 = 4;
    pub const POTION_TABLE_MISSING: i32 = 5;
    pub const EMBEDDER_FAILED: i32 = 6;
}

pub fn run() -> anyhow::Result<()> {
    let t = Tone::stdout();
    println!("{}", t.bold("urna doctor"));
    let rows = health::collect();
    for row in &rows {
        if let Some(err) = &row.stderr {
            eprintln!("  embedder stderr: {err}");
        }
        let tag = match row.level {
            Level::Ok => t.fg("ok      ", tone::HIGH),
            Level::Warn => t.fg("warn    ", tone::MID),
            Level::Fail(code) => t.fg(&format!("fail({code})"), tone::LOW),
        };
        println!("  {tag} {}: {}", row.what, t.fg(&row.detail, tone::FAINT));
    }
    let code = health::exit_code(&rows);
    if code == codes::OK {
        println!("urna doctor: {}", t.fg("ok", tone::HIGH));
    } else {
        println!("urna doctor: {}", t.fg("failed", tone::LOW));
        if code != codes::EMBEDDER_FAILED {
            println!(
                "  {}",
                t.fg(
                    "next: `urna setup` lays down the embedder and a python env",
                    tone::TEAL
                )
            );
        }
    }
    std::process::exit(code);
}
