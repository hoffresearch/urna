//! Resolving the python interpreter the offline embedder runs under.
//!
//! `search-text`, `ask`, and `retrieve` shell out to a python embedder that
//! needs numpy + tokenizers (the potion table). this picks an interpreter that
//! has them with no manual setup, while never touching the network: the
//! embedder opens no socket regardless of which interpreter runs it.
//!
//! trust note: with `URNA_PYTHON` unset, discovery EXECUTES the first
//! `.venv/bin/python` found walking up from the cwd, so the choice is made by
//! filesystem proximity. running from inside a tree whose ancestor carries an
//! untrusted `.venv` (a world-writable parent, or a repo from an untrusted
//! source) would run that binary before any socket is involved. set
//! `URNA_PYTHON` to pin the interpreter explicitly there; the resolved
//! interpreter is always logged to stderr so the choice is never silent.
//!
//! `urna setup` creates a managed venv (`<data root>/urna/venv`, see
//! `paths`) that carries the deps by construction; it ranks right after
//! `URNA_PYTHON`, ahead of the walk-up heuristic, so an installed urna run
//! from inside some other project's `.venv` still embeds with its own.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

/// Set by the terminal ui: its screen owns stderr, so the interpreter note
/// is kept out of it (the ui shows the interpreter in its own doctor view).
static QUIET: AtomicBool = AtomicBool::new(false);

#[cfg_attr(not(feature = "tui"), allow(dead_code))]
pub fn set_quiet(quiet: bool) {
    QUIET.store(quiet, Ordering::Relaxed);
}

/// The python interpreter the embedder scripts run under. `URNA_PYTHON` wins;
/// then the venv `urna setup` created; otherwise prefer the repo's `.venv`
/// (it carries numpy + tokenizers for the potion table) so `search-text`,
/// `ask`, and `retrieve` work without extra setup; else fall back to
/// `python3` on PATH.
pub fn resolve_interpreter() -> String {
    let explicit = std::env::var("URNA_PYTHON").ok().filter(|s| !s.is_empty());
    let managed = super::paths::setup_python().map(|p| p.to_string_lossy().into_owned());
    let interp = resolve_interpreter_from(explicit.or(managed), std::env::current_dir().ok());
    // surface the choice: discovery can execute a `.venv` found by filesystem
    // proximity, so the selected interpreter must never be silent.
    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("[urna] embedder interpreter: {interp}");
    }
    interp
}

/// Testable core of [`resolve_interpreter`]: an explicit interpreter (the
/// `URNA_PYTHON` value, else the managed setup venv) wins, then the nearest
/// `.venv/bin/python` in `start` or up to three of its ancestors, then
/// `python3`.
fn resolve_interpreter_from(urna_python: Option<String>, start: Option<PathBuf>) -> String {
    if let Some(p) = urna_python {
        return p;
    }
    if let Some(mut dir) = start {
        for _ in 0..4 {
            let venv = dir.join(".venv").join("bin").join("python");
            if venv.exists() {
                return venv.to_string_lossy().into_owned();
            }
            if !dir.pop() {
                break;
            }
        }
    }
    "python3".into()
}

#[cfg(test)]
mod tests {
    use super::resolve_interpreter_from;
    use std::path::PathBuf;

    #[test]
    fn urna_python_overrides_everything() {
        let got = resolve_interpreter_from(Some("/opt/py/bin/python".into()), None);
        assert_eq!(got, "/opt/py/bin/python");
    }

    #[test]
    fn falls_back_to_python3_without_a_venv() {
        // a deep path whose ancestors carry no `.venv`; the four-step walk finds
        // nothing and resolves to the PATH `python3`.
        let start = Some(PathBuf::from("/urna-no-venv-here/a/b/c"));
        assert_eq!(resolve_interpreter_from(None, start), "python3");
    }

    #[test]
    fn discovers_a_venv_in_an_ancestor() {
        // lay down <base>/.venv/bin/python and start two levels below it.
        let base = std::env::temp_dir().join(format!("urna_interp_{}", std::process::id()));
        let venv_python = base.join(".venv").join("bin").join("python");
        std::fs::create_dir_all(venv_python.parent().unwrap()).unwrap();
        std::fs::write(&venv_python, b"").unwrap();
        let start = base.join("nested").join("deeper");
        std::fs::create_dir_all(&start).unwrap();

        let got = resolve_interpreter_from(None, Some(start));
        assert_eq!(got, venv_python.to_string_lossy());

        let _ = std::fs::remove_dir_all(&base);
    }
}
