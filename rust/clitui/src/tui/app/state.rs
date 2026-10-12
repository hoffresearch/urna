//! What the explorer remembers between runs: the corpora it opened, most
//! recent first, in `<data>/urna/tui.json`. the file is the explorer's
//! own; a missing or unreadable one is a first run, never an error, and a
//! failed write only loses the memory, not the session.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// How many corpora the file keeps.
const KEEP: usize = 9;

#[derive(Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct State {
    #[serde(default)]
    pub recent: Vec<PathBuf>,
}

/// `<data>/urna/tui.json`, when there is a data dir.
pub fn file() -> Option<PathBuf> {
    crate::cmd::paths::urna_home().map(|h| h.join("tui.json"))
}

impl State {
    pub fn read(file: &Path) -> State {
        std::fs::read_to_string(file)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default()
    }

    /// The corpus opened last, while it is still where it was.
    pub fn last(&self) -> Option<&Path> {
        self.recent
            .first()
            .map(PathBuf::as_path)
            .filter(|p| p.is_file())
    }

    /// Puts `corpus` first, resolved (no `..`, no symlink), so the same
    /// file opened from another directory is one entry and a path given
    /// relative to a folder deleted since still opens; the ones gone from
    /// disk drop out.
    pub fn opened(&mut self, corpus: &Path) {
        let corpus = corpus
            .canonicalize()
            .or_else(|_| std::path::absolute(corpus))
            .unwrap_or_else(|_| corpus.to_path_buf());
        self.recent.retain(|p| *p != corpus && p.is_file());
        self.recent.insert(0, corpus);
        self.recent.truncate(KEEP);
    }

    /// Writes the file whole or not at all (a temp file renamed over it).
    pub fn write(&self, file: &Path) -> std::io::Result<()> {
        if let Some(dir) = file.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = file.with_extension(format!("json.{}", std::process::id()));
        let text = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(&tmp, text + "\n")?;
        std::fs::rename(&tmp, file).inspect_err(|_| {
            let _ = std::fs::remove_file(&tmp);
        })
    }
}

/// Records `corpus` as the one opened last.
pub fn remember(corpus: &Path) {
    if let Some(f) = file() {
        let mut s = State::read(&f);
        s.opened(corpus);
        let _ = s.write(&f);
    }
}

