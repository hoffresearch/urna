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
        let corpus = resolve(corpus);
        self.recent.retain(|p| *p != corpus && p.is_file());
        self.recent.insert(0, corpus);
        self.recent.truncate(KEEP);
    }

    /// Drops `corpus`, resolved the same way; the others keep their order.
    pub fn forgot(&mut self, corpus: &Path) {
        let corpus = resolve(corpus);
        self.recent.retain(|p| *p != corpus);
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

/// The path as the file keeps it: canonical when the file is there,
/// absolute otherwise, as given when not even that works.
fn resolve(corpus: &Path) -> PathBuf {
    corpus
        .canonicalize()
        .or_else(|_| std::path::absolute(corpus))
        .unwrap_or_else(|_| corpus.to_path_buf())
}

/// Records `corpus` in `file` as the one opened last.
pub fn remember(file: &Path, corpus: &Path) {
    update(file, |s| s.opened(corpus));
}

/// Drops `corpus` from `file`, so a corpus that would not open is not
/// retried on the next launch.
pub fn forget(file: &Path, corpus: &Path) {
    update(file, |s| s.forgot(corpus));
}

fn update(file: &Path, change: impl FnOnce(&mut State)) {
    let mut s = State::read(file);
    change(&mut s);
    let _ = s.write(file);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("urna_state_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d.canonicalize().unwrap()
    }

    #[test]
    fn a_missing_or_broken_file_is_a_first_run() {
        let d = dir("broken");
        assert_eq!(State::read(&d.join("tui.json")), State::default());
        std::fs::write(d.join("tui.json"), b"{not json").unwrap();
        assert_eq!(State::read(&d.join("tui.json")), State::default());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn the_last_corpus_comes_first_once_and_resolved() {
        let d = dir("recent");
        std::fs::create_dir_all(d.join("sub")).unwrap();
        for n in ["a.urna", "b.urna"] {
            std::fs::write(d.join(n), b"x").unwrap();
        }
        let mut s = State::default();
        s.opened(&d.join("a.urna"));
        s.opened(&d.join("b.urna"));
        // the same file through another path is one entry, moved first.
        s.opened(&d.join("sub/../a.urna"));
        assert_eq!(s.recent, [d.join("a.urna"), d.join("b.urna")]);
        assert_eq!(s.last(), Some(d.join("a.urna").as_path()));
        let f = d.join("tui.json");
        s.write(&f).unwrap();
        assert_eq!(State::read(&f), s);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_moved_last_corpus_is_none_and_gone_files_drop_out() {
        let d = dir("moved");
        for n in ["a.urna", "b.urna"] {
            std::fs::write(d.join(n), b"x").unwrap();
        }
        let mut s = State::default();
        s.opened(&d.join("a.urna"));
        s.opened(&d.join("b.urna"));
        std::fs::rename(d.join("b.urna"), d.join("c.urna")).unwrap();
        assert_eq!(s.last(), None);
        s.opened(&d.join("c.urna"));
        assert_eq!(s.recent, [d.join("c.urna"), d.join("a.urna")]);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_forgotten_corpus_goes_and_the_others_keep_their_order() {
        let d = dir("forget");
        std::fs::create_dir_all(d.join("sub")).unwrap();
        for n in ["a.urna", "b.urna", "c.urna"] {
            std::fs::write(d.join(n), b"x").unwrap();
        }
        let mut s = State::default();
        for n in ["a.urna", "b.urna", "c.urna"] {
            s.opened(&d.join(n));
        }
        // through another path, like `opened` would see it.
        s.forgot(&d.join("sub/../b.urna"));
        assert_eq!(s.recent, [d.join("c.urna"), d.join("a.urna")]);
        s.forgot(&d.join("nowhere.urna"));
        assert_eq!(s.recent.len(), 2);
        // the file round trip: the entry is gone from disk too.
        let f = d.join("tui.json");
        remember(&f, &d.join("b.urna"));
        remember(&f, &d.join("a.urna"));
        forget(&f, &d.join("b.urna"));
        assert_eq!(State::read(&f).recent, [d.join("a.urna")]);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn it_keeps_at_most_nine() {
        let d = dir("nine");
        let mut s = State::default();
        for i in 0..12 {
            let p = d.join(format!("{i}.urna"));
            std::fs::write(&p, b"x").unwrap();
            s.opened(&p);
        }
        assert_eq!(s.recent.len(), KEEP);
        assert_eq!(s.recent[0], d.join("11.urna"));
        std::fs::remove_dir_all(&d).unwrap();
    }
}
