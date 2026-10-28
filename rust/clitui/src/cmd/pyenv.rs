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

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

/// macOS's `python3` before the Command Line Tools are installed: a stub
/// that opens Apple's installer dialog when anything runs it.
const MACOS_STUB: &str = "/usr/bin/python3";

/// Whether `interp` (a name looked up on PATH, or a path) may run here.
/// false only on macOS, for the `/usr/bin/python3` stub while
/// `xcode-select -p` fails: running it, even for `--version`, opens a
/// system dialog. setup's scan, `urna doctor` and the embedder fallback all
/// ask this first, so none of them ever runs the stub.
pub fn may_run(interp: &str) -> bool {
    if !cfg!(target_os = "macos") {
        return true;
    }
    let p = Path::new(interp);
    let resolved = if p.components().count() == 1 {
        super::paths::which(interp)
    } else {
        Some(p.to_path_buf())
    };
    !is_stub(resolved.as_deref()) || command_line_tools()
}

/// `path` is the stub, directly or through a symlink.
fn is_stub(path: Option<&Path>) -> bool {
    let stub = Path::new(MACOS_STUB);
    path.is_some_and(|p| p == stub || p.canonicalize().is_ok_and(|c| c == stub))
}

/// `xcode-select -p` succeeds: the tools are installed and the stub runs
/// their python3. it prints a path and opens no dialog; asked once.
fn command_line_tools() -> bool {
    static ONCE: OnceLock<bool> = OnceLock::new();
    *ONCE.get_or_init(|| {
        std::process::Command::new("xcode-select")
            .arg("-p")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    })
}

/// On macOS without the Command Line Tools: every developer tool in
/// `/usr/bin` (`python3`, `install_name_tool`, `git`, …) is one shim that
/// opens Apple's installer dialog when anything runs it.
#[cfg_attr(not(feature = "tui"), allow(dead_code))]
pub fn shims_open_dialogs() -> bool {
    cfg!(target_os = "macos") && !command_line_tools()
}

/// What a caller says instead of running the stub.
pub fn stub_hint(interp: &str) -> String {
    format!(
        "{interp} is macOS's stub, which opens Apple's Command Line Tools installer; \
         run `urna setup` (it builds its own python env) or set URNA_PYTHON"
    )
}

/// Set by the terminal ui: its screen owns stderr, so the interpreter note
/// is kept out of it (the ui shows the interpreter in its own doctor view).
static QUIET: AtomicBool = AtomicBool::new(false);

pub fn set_quiet(quiet: bool) {
    QUIET.store(quiet, Ordering::Relaxed);
}

pub fn is_quiet() -> bool {
    QUIET.load(Ordering::Relaxed)
}

/// The python interpreter the embedder scripts run under. `URNA_PYTHON` wins;
/// then the venv `urna setup` created; otherwise prefer the repo's `.venv`
/// (it carries numpy + tokenizers for the potion table) so `search-text`,
/// `ask`, and `retrieve` work without extra setup; else fall back to
/// `python3` on PATH.
pub fn resolve_interpreter() -> String {
    let explicit = std::env::var("URNA_PYTHON").ok().filter(|s| !s.is_empty());
    let managed = super::paths::setup_python().map(|p| p.to_string_lossy().into_owned());
    let chosen = explicit.or(managed);
    let searched = chosen.is_none();
    let interp = resolve_interpreter_from(chosen, std::env::current_dir().ok());
    // a `.venv` found by walking up the folders runs by filesystem proximity,
    // so that choice is never silent; `URNA_PYTHON` and setup's env were
    // chosen on purpose, and naming them on every query is noise.
    if searched && interp != FALLBACK && !QUIET.load(Ordering::Relaxed) {
        eprintln!("[urna] embedder interpreter: {interp}");
    }
    interp
}

/// The interpreter when nothing names one and no `.venv` is near.
const FALLBACK: &str = "python3";

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
    FALLBACK.into()
}

#[cfg(test)]
mod tests {
    use super::{is_stub, may_run, resolve_interpreter_from};
    use std::path::{Path, PathBuf};

    #[test]
    fn only_the_macos_stub_path_counts_as_the_stub() {
        assert!(is_stub(Some(Path::new("/usr/bin/python3"))));
        assert!(!is_stub(Some(Path::new("/opt/homebrew/bin/python3"))));
        assert!(!is_stub(None));
    }

    #[test]
    fn anything_but_the_stub_may_run() {
        assert!(may_run("/opt/py/bin/python"));
        assert!(may_run("urna-definitely-not-a-binary-xyz"));
        if !cfg!(target_os = "macos") {
            assert!(may_run("/usr/bin/python3"));
        }
    }

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
