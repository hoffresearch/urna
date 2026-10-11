//! Where an installed urna keeps its data: the embedder payload
//! (`<root>/urna/python/urna/...`) and the python env `urna setup` creates
//! (`<root>/urna/venv`). One ladder for every reader and for the installer,
//! so what `urna setup` writes is exactly what `doctor`, `ask` and
//! `retrieve` find.
//!
//! order: `URNA_DATA_DIR` (the same override the one-liner installers
//! take), `XDG_DATA_HOME`, `~/.local/share`, `%LOCALAPPDATA%` (where
//! `installer.ps1` lays the payload down), then `<exe>/../share` (a tarball
//! unpacked next to its own `share/`).

use std::path::PathBuf;

/// Every candidate data root, most specific first. Each root holds an
/// `urna/` directory.
pub fn data_roots() -> Vec<PathBuf> {
    let env = |k: &str| {
        std::env::var_os(k)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    let mut roots: Vec<PathBuf> = Vec::new();
    roots.extend(env("URNA_DATA_DIR"));
    roots.extend(env("XDG_DATA_HOME"));
    roots.extend(env("HOME").map(|h| h.join(".local").join("share")));
    roots.extend(env("LOCALAPPDATA"));
    roots.extend(
        std::env::current_exe()
            .ok()
            .and_then(|e| e.parent().map(|p| p.join("..").join("share"))),
    );
    roots.dedup();
    roots
}

/// The root `urna setup` writes into: the first candidate that is not the
/// exe-relative `share/` (a package manager owns that one). On windows the
/// default is `%LOCALAPPDATA%`, matching `installer.ps1`.
#[cfg_attr(not(feature = "tui"), allow(dead_code))]
pub fn install_root() -> Option<PathBuf> {
    let env = |k: &str| {
        std::env::var_os(k)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    if let Some(p) = env("URNA_DATA_DIR") {
        return Some(p);
    }
    if cfg!(windows)
        && let Some(p) = env("LOCALAPPDATA")
    {
        return Some(p);
    }
    env("XDG_DATA_HOME").or_else(|| env("HOME").map(|h| h.join(".local").join("share")))
}

/// `<install root>/urna`, the directory `urna setup` owns.
#[cfg_attr(not(feature = "tui"), allow(dead_code))]
pub fn urna_home() -> Option<PathBuf> {
    install_root().map(|r| r.join("urna"))
}

/// The interpreter inside the venv `urna setup` creates, when it exists.
pub fn setup_python() -> Option<PathBuf> {
    data_roots()
        .into_iter()
        .map(|r| venv_python(&r.join("urna").join("venv")))
        .find(|p| p.is_file())
}

/// `bin/python` on unix, `Scripts\python.exe` on windows.
pub fn venv_python(venv: &std::path::Path) -> PathBuf {
    if cfg!(windows) {
        venv.join("Scripts").join("python.exe")
    } else {
        venv.join("bin").join("python")
    }
}

#[cfg(test)]
mod tests {
    use super::venv_python;
    use std::path::Path;

    #[test]
    fn venv_python_follows_the_platform_layout() {
        let p = venv_python(Path::new("/x/venv"));
        if cfg!(windows) {
            assert!(p.ends_with("Scripts/python.exe"));
        } else {
            assert!(p.ends_with("bin/python"));
        }
    }

    #[test]
    fn data_roots_is_never_empty_on_a_normal_host() {
        // the exe-relative share/ is always a candidate.
        assert!(!super::data_roots().is_empty());
    }
}
