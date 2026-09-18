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

use std::path::PathBuf;

/// The python interpreter the embedder scripts run under. `URNA_PYTHON` wins;
/// otherwise prefer the repo's `.venv` (it carries numpy + tokenizers for the
/// potion table) so `search-text`, `ask`, and `retrieve` work without extra
/// setup; else fall back to `python3` on PATH.
pub fn resolve_interpreter() -> String {
    let interp = resolve_interpreter_from(
        std::env::var("URNA_PYTHON").ok(),
        std::env::current_dir().ok(),
    );
    // surface the choice: discovery can execute a `.venv` found by filesystem
    // proximity, so the selected interpreter must never be silent.
    eprintln!("[urna] embedder interpreter: {interp}");
    interp
}

/// Testable core of [`resolve_interpreter`]: an explicit `URNA_PYTHON` wins,
/// then the nearest `.venv/bin/python` walking up to four ancestors of `start`,
/// then `python3`.
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
