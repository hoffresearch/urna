//! `urna doctor` - post-install health check (issue #75).
//!
//! validates the surface `ask` depends on and exits with a TYPED code so
//! installers, CI, and support can branch on the failure class instead of
//! parsing text:
//!
//!   0  ok (scalar simd still exits 0, printed as a warning)
//!   4  potion embedder payload not found (repo layout or data dir)
//!   5  potion model table missing or still a git-lfs pointer
//!   6  the in-process embed failed
//!
//! python is the corpus-building toolchain, not the query's: the doctor
//! reports it as a warning, and `urna doctor --build` turns a missing
//! interpreter or missing deps into the typed failures 2 and 3 and runs the
//! embedder script under it.
//!
//! checks: urna/format versions, simd backend, the potion payload and
//! table, one real embed in process (urna's own runtime, no interpreter),
//! then python and its deps. nothing here opens a socket. the checks live
//! in `health` (shared with the terminal ui); this module only prints
//! them, colored on a terminal and plain when piped.

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

pub fn run(build: bool) -> anyhow::Result<()> {
    let t = Tone::stdout();
    println!("{}", t.bold("urna doctor"));
    let rows = if build {
        health::collect_build()
    } else {
        health::collect()
    };
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
        let next = match code {
            codes::PYTHON_MISSING | codes::PYTHON_DEPS_MISSING => {
                Some("next: only corpus builds need python; `urna setup --python` builds its env")
            }
            codes::EMBEDDER_FAILED => None,
            _ => Some("next: `urna setup` lays down the embedder payload"),
        };
        if let Some(next) = next {
            println!("  {}", t.fg(next, tone::TEAL));
        }
    }
    std::process::exit(code);
}
