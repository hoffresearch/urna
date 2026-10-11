//! The payload of a development build: the checkout it was compiled from
//! (`cmd::buildid::DEV_TREE`), staged in the release layout
//! (`cmd::payload::REQUIRED`) and laid down like a downloaded one. A
//! development build never fetches a release payload by the version it
//! reports: that release ships the scripts of another tree.
//!
//! The potion table is the checkout's own file when git lfs fetched it. A
//! bare lfs pointer (`cargo install --git` fetches no lfs object) is replaced
//! by the upstream file at the pinned revision, accepted only when its
//! sha256 is the pointer's oid, as `tool/tasks/getpotion.sh` does for the
//! release builds.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::cmd::payload::REQUIRED;

/// The potion table inside a payload home.
pub const TABLE: &str = "python/urna/model/potionb8m/model.safetensors";

/// The upstream revision of the potion table, the same as `REV` in
/// `tool/tasks/getpotion.sh` (a test holds the two together).
pub const POTION_REV: &str = "bf8b056651a2c21b8d2565580b8569da283cab23";

pub fn potion_url() -> String {
    format!(
        "https://huggingface.co/minishlab/potion-base-8M/resolve/{POTION_REV}/model.safetensors"
    )
}

/// The sha256 an lfs pointer names, when `path` is one.
pub fn lfs_oid(path: &Path) -> Option<String> {
    let head = std::fs::read(path).ok().filter(|b| b.len() < 1024)?;
    let text = String::from_utf8(head).ok()?;
    if !text.starts_with("version https://git-lfs") {
        return None;
    }
    text.lines()
        .find_map(|l| l.strip_prefix("oid sha256:"))
        .map(|h| h.trim().to_string())
}

/// The checkout's copy of a payload file (`REQUIRED` names them relative to
/// the payload home, under `python/`).
fn source(tree: &Path, rel: &str) -> PathBuf {
    tree.join(rel.strip_prefix("python/").unwrap_or(rel))
}

/// The oid of the table when the checkout holds only its lfs pointer: the
/// table then comes from upstream.
pub fn table_oid(tree: &Path) -> Option<String> {
    lfs_oid(&source(tree, TABLE))
}

/// Why `tree` cannot serve a payload: the checkout was moved or deleted
/// after the build. `CARGO_MANIFEST_DIR` is fixed at build time.
pub fn gone(tree: &Path) -> Option<String> {
    (!tree.join("urna").is_dir()).then(|| {
        format!(
            "the checkout this binary was built from is gone ({}); build it again from the checkout: \
             `cargo install --path rust/clitui`, or `cargo install --git https://github.com/hoffresearch/urna urna`",
            tree.display()
        )
    })
}

/// Copies every required file but `VERSION` and the table from `tree` into
/// `<staging>/urna`, and the table too when the checkout holds the real
/// file. Returns where the table goes when it still has to be fetched.
pub fn stage(tree: &Path, staging: &Path) -> Result<Option<PathBuf>> {
    if let Some(why) = gone(tree) {
        bail!(why);
    }
    let home = staging.join("urna");
    let fetch = table_oid(tree).map(|_| home.join(TABLE));
    for rel in REQUIRED.iter().filter(|r| **r != "VERSION") {
        let to = home.join(rel);
        if let Some(dir) = to.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        }
        if *rel == TABLE && fetch.is_some() {
            continue;
        }
        let from = source(tree, rel);
        std::fs::copy(&from, &to)
            .with_context(|| format!("copy {} from the checkout", from.display()))?;
    }
    Ok(fetch)
}

/// Writes the stamp last, as the release payload carries it: the build's
/// name, so a build from another commit replaces this payload.
pub fn stamp(staging: &Path, name: &str) -> Result<()> {
    let to = staging.join("urna").join("VERSION");
    std::fs::write(&to, format!("{name}\n")).with_context(|| format!("write {}", to.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const POINTER: &str =
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc123\nsize 30236760\n";

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("urna_tree_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A checkout's `rust/bridge/python` holding every required file, the
    /// table as `table`.
    fn checkout(at: &Path, table: &[u8]) -> PathBuf {
        let tree = at.join("python");
        for rel in REQUIRED.iter().filter(|r| **r != "VERSION") {
            let p = source(&tree, rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            let body: &[u8] = if *rel == TABLE { table } else { rel.as_bytes() };
            std::fs::write(p, body).unwrap();
        }
        tree
    }

    #[test]
    fn the_potion_revision_is_the_one_getpotion_fetches() {
        let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tool/tasks/getpotion.sh");
        let text = std::fs::read_to_string(&script).unwrap();
        assert!(
            text.contains(&format!("REV=\"{POTION_REV}\"")),
            "{} pins another revision",
            script.display()
        );
    }

    #[test]
    fn reads_the_oid_of_an_lfs_pointer_only() {
        let d = tmp("oid");
        std::fs::write(d.join("ptr"), POINTER).unwrap();
        assert_eq!(lfs_oid(&d.join("ptr")).as_deref(), Some("abc123"));
        std::fs::write(d.join("real"), b"\x50\x00\x00\x00{\"embeddings\"").unwrap();
        assert_eq!(lfs_oid(&d.join("real")), None);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_missing_checkout_names_its_path_and_the_fix() {
        let why = gone(Path::new("/nowhere/rust/bridge/python")).unwrap();
        assert!(why.contains("/nowhere/rust/bridge/python") && why.contains("cargo install"));
        let d = tmp("gone");
        assert!(stage(&d.join("python"), &d.join("staging")).is_err());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn stages_a_checkout_with_the_real_table_completely() {
        let d = tmp("real");
        let tree = checkout(&d, b"table bytes");
        let staging = d.join("staging");
        assert_eq!(stage(&tree, &staging).unwrap(), None);
        stamp(&staging, "0.5.4+dev.abc").unwrap();
        let home = staging.join("urna");
        assert!(crate::cmd::payload::missing(&home).is_empty());
        assert_eq!(std::fs::read(home.join(TABLE)).unwrap(), b"table bytes");
        assert_eq!(
            std::fs::read_to_string(home.join("VERSION")).unwrap(),
            "0.5.4+dev.abc\n"
        );
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn leaves_a_pointer_table_to_fetch() {
        let d = tmp("pointer");
        let tree = checkout(&d, POINTER.as_bytes());
        let staging = d.join("staging");
        let fetch = stage(&tree, &staging).unwrap();
        let home = staging.join("urna");
        assert_eq!(fetch, Some(home.join(TABLE)));
        assert!(!home.join(TABLE).exists(), "the pointer is never installed");
        assert_eq!(table_oid(&tree).as_deref(), Some("abc123"));
        std::fs::remove_dir_all(&d).unwrap();
    }
}
