//! Lays a verified payload tarball down under the data root. The archive is
//! unpacked into a staging dir first (entries that would escape it, `..`
//! or absolute paths, abort the install) and checked: every file of
//! `REQUIRED` present, nothing but `python/` and plain files at the top (so
//! the managed venv beside them can never be overwritten). only then is it
//! laid down, as one restorable step: the previous payload moves aside into
//! the staging dir, the new top-level files go in, then `python/`, then
//! `VERSION` last, so a stamp only ever names a payload that is complete. a
//! failure at any point removes what was placed and puts the previous
//! payload back.

use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;

/// The files a payload lays down beside `python/` (`embedpack.py`
/// writes them); `--uninstall` removes exactly these.
pub const TOP_LEVEL: [&str; 1] = ["VERSION"];

/// What a payload laid down beside `VERSION` before the move to `python/`:
/// an install of the new payload and `--uninstall` remove it, so an upgrade
/// leaves no stale copy behind.
// layout until 0.5.4; remove in the release after next
// `__pycache__` holds the bytecode python wrote beside embed_query.py.
pub const LEGACY: [&str; 4] = [
    "forge",
    "embed_query.py",
    "model_fingerprint.py",
    "__pycache__",
];

/// Removes the pre-0.5.5 payload from `home`; returns the paths it removed
/// and the ones still there (a removal that failed, on permissions say).
pub fn remove_legacy(home: &Path) -> (Vec<PathBuf>, Vec<PathBuf>) {
    LEGACY
        .iter()
        .map(|n| home.join(n))
        .filter(|p| p.exists())
        .partition(|p| {
            remove(p);
            !p.exists()
        })
}

use crate::cmd::payload::{DIR, PACKAGE, missing};

/// Unpacks `tar_gz` into `root` (the parent of `urna/`), calling
/// `on_entry(n)` after each entry; returns the installed package dir.
pub fn install(tar_gz: &Path, root: &Path, mut on_entry: impl FnMut(usize)) -> Result<PathBuf> {
    let staging = staging(root)?;
    let result = unpack_into(tar_gz, &staging, &mut on_entry)
        .and_then(|_| swap(&staging, root, Fault::default()));
    finish(&staging);
    result
}

/// An empty staging dir in `root` (`.urna-setup-<pid>`, which `job::sweep`
/// removes after an interrupted run).
pub fn staging(root: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(root).with_context(|| format!("create {}", root.display()))?;
    let staging = root.join(format!(".urna-setup-{}", std::process::id()));
    if staging.exists() {
        std::fs::remove_dir_all(&staging)?;
    }
    std::fs::create_dir_all(&staging)?;
    Ok(staging)
}

/// Lays down a payload already staged as `<staging>/urna` (a development
/// build stages its checkout there), with the same checks and restorable
/// swap as an unpacked tarball; removes the staging dir either way.
pub fn install_staged(staging: &Path, root: &Path) -> Result<PathBuf> {
    let result = swap(staging, root, Fault::default());
    finish(staging);
    result
}

/// Removes a staging dir a step gave up on before laying it down.
pub fn discard(staging: &Path) {
    finish(staging);
}

/// Removes the staging dir, unless it still holds the previous payload: a
/// restore that failed and could not move the copy out leaves it there,
/// and the error names that path.
fn finish(staging: &Path) {
    let holds_previous = std::fs::read_dir(staging.join("previous"))
        .map(|mut d| d.next().is_some())
        .unwrap_or(false);
    if !holds_previous {
        let _ = std::fs::remove_dir_all(staging);
    }
}

/// Where an install is made to fail, for the tests: `place` stops before
/// laying that entry down, `restore` makes putting that entry back fail,
/// `keep` makes moving the previous copy out of staging fail. setup passes
/// the default, which never fails.
#[derive(Clone, Copy, Default)]
struct Fault<'a> {
    place: Option<&'a str>,
    restore: Option<&'a str>,
    keep: bool,
}

fn unpack_into(tar_gz: &Path, staging: &Path, on_entry: &mut impl FnMut(usize)) -> Result<()> {
    let file = File::open(tar_gz).with_context(|| format!("open {}", tar_gz.display()))?;
    let mut archive = tar::Archive::new(GzDecoder::new(file));
    archive.set_preserve_permissions(false);
    let mut n = 0usize;
    for entry in archive.entries().context("read payload archive")? {
        let mut entry = entry.context("read payload entry")?;
        let path = entry.path()?.into_owned();
        // unpack_in refuses (returns false) anything that would land outside
        // the staging dir; an honest payload never does that, so it is fatal.
        if !entry.unpack_in(staging)? {
            bail!("payload entry escapes the install dir: {}", path.display());
        }
        n += 1;
        on_entry(n);
    }
    Ok(())
}

/// The files at the top of the staged `urna/`, after checking that nothing
/// else but `python/` sits there.
fn top_level_files(new_home: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(new_home).context("read the staged payload")? {
        let entry = entry?;
        let name = entry.file_name();
        if name == DIR {
            continue;
        }
        if !entry.file_type()?.is_file() {
            bail!(
                "payload carries urna/{}, which is not a file; refusing to install it",
                name.to_string_lossy()
            );
        }
        files.push(entry.path());
    }
    Ok(files)
}

/// Moves `from` to `to`, replacing a file there (windows refuses a rename
/// onto an existing file).
fn put(from: &Path, to: &Path) -> Result<()> {
    if to.is_file() {
        std::fs::remove_file(to).with_context(|| format!("remove {}", to.display()))?;
    }
    std::fs::rename(from, to)
        .with_context(|| format!("move {} to {}", from.display(), to.display()))
}

fn remove(p: &Path) {
    let _ = if p.is_dir() {
        std::fs::remove_dir_all(p)
    } else {
        std::fs::remove_file(p)
    };
}

/// Lays the staged payload down over `<root>/urna`, restoring the previous
/// one on any failure; see `Fault` for the failures the tests inject.
fn swap(staging: &Path, root: &Path, fault: Fault<'_>) -> Result<PathBuf> {
    let new_home = staging.join("urna");
    let gone: Vec<&str> = missing(&new_home);
    if !gone.is_empty() {
        bail!(
            "payload is incomplete, missing urna/{}",
            gone.join(", urna/")
        );
    }
    let mut names: Vec<String> = top_level_files(&new_home)?
        .iter()
        .filter_map(|f| f.file_name().map(|n| n.to_string_lossy().into_owned()))
        .filter(|n| n != "VERSION")
        .collect();
    // the package after the plain files, the stamp after everything.
    names.push(DIR.into());
    names.push("VERSION".into());
    let home = root.join("urna");
    std::fs::create_dir_all(&home)?;
    let previous = staging.join("previous");
    std::fs::create_dir_all(&previous)?;
    let mut aside: Vec<String> = Vec::new();
    let mut placed: Vec<String> = Vec::new();
    let laid = lay_down(
        &names,
        &new_home,
        &home,
        &previous,
        fault,
        &mut aside,
        &mut placed,
    );
    let Err(e) = laid else {
        // the previous payload is replaced; it goes with the staging dir.
        let _ = std::fs::remove_dir_all(&previous);
        remove_legacy(&home);
        return Ok(home.join(PACKAGE));
    };
    let lost = restore(&home, &previous, &aside, &placed, fault);
    if lost.is_empty() {
        return Err(
            e.context("installing the payload failed; the previous payload is back in place")
        );
    }
    let kept = keep_previous(&previous, root, fault);
    bail!(
        "installing the payload failed ({e:#}), and putting back the previous {} failed too; \
         the previous files are kept at {}: move them into {} by hand, or run urna setup again",
        lost.join(", "),
        kept.display(),
        home.display()
    )
}

/// Moves the previous payload's entries aside, then the new ones in, in
/// `names` order; records both so a failure can be undone.
#[allow(clippy::too_many_arguments, reason = "the undo log is two out-params")]
fn lay_down(
    names: &[String],
    new_home: &Path,
    home: &Path,
    previous: &Path,
    fault: Fault<'_>,
    aside: &mut Vec<String>,
    placed: &mut Vec<String>,
) -> Result<()> {
    for n in names {
        if home.join(n).exists() {
            std::fs::rename(home.join(n), previous.join(n))
                .with_context(|| format!("move the previous {n} aside"))?;
            aside.push(n.clone());
        }
    }
    for n in names {
        if fault.place == Some(n.as_str()) {
            bail!("stopped before {n}");
        }
        put(&new_home.join(n), &home.join(n))?;
        placed.push(n.clone());
    }
    Ok(())
}

/// Removes what was placed and puts the previous entries back; returns the
/// entries that could not be put back (still in `previous`).
fn restore(
    home: &Path,
    previous: &Path,
    aside: &[String],
    placed: &[String],
    fault: Fault<'_>,
) -> Vec<String> {
    for n in placed.iter().rev() {
        remove(&home.join(n));
    }
    let mut lost = Vec::new();
    for n in aside.iter().rev() {
        let back = fault.restore != Some(n.as_str())
            && std::fs::rename(previous.join(n), home.join(n)).is_ok();
        if !back {
            lost.push(n.clone());
        }
    }
    lost
}

/// Moves what is left of the previous payload out of the staging dir, so
/// cleaning the staging dir can never delete it; returns where it now is
/// (the staging copy itself when even that move fails, and `finish` then
/// keeps the staging dir).
fn keep_previous(previous: &Path, root: &Path, fault: Fault<'_>) -> PathBuf {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let out = root.join(format!(".urna-previous-{}-{secs}", std::process::id()));
    if !fault.keep && std::fs::rename(previous, &out).is_ok() {
        out
    } else {
        previous.to_path_buf()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cmd::payload::REQUIRED;
    use flate2::Compression;
    use flate2::write::GzEncoder;

    fn tarball(dir: &Path, entries: &[(&str, &[u8])]) -> PathBuf {
        let out = dir.join("p.tar.gz");
        let gz = GzEncoder::new(File::create(&out).unwrap(), Compression::fast());
        let mut b = tar::Builder::new(gz);
        for (name, data) in entries {
            let mut h = tar::Header::new_gnu();
            h.set_size(data.len() as u64);
            h.set_mode(0o644);
            // set_path refuses `..`; write the raw name to model a hostile archive.
            let raw = &mut h.as_old_mut().name;
            raw[..name.len()].copy_from_slice(name.as_bytes());
            h.set_cksum();
            b.append(&h, *data).unwrap();
        }
        b.into_inner().unwrap().finish().unwrap();
        out
    }

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("urna_unpack_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A complete payload stamped `version`, every file's body naming the
    /// release (`<rel>@<version>`), plus `extra` entries.
    fn payload(version: &str, extra: &[(&str, &[u8])]) -> Vec<(String, Vec<u8>)> {
        let mut v: Vec<(String, Vec<u8>)> = REQUIRED
            .iter()
            .map(|rel| {
                let body = if *rel == "VERSION" {
                    format!("{version}\n")
                } else {
                    format!("{rel}@{version}")
                };
                (format!("urna/{rel}"), body.into_bytes())
            })
            .collect();
        v.extend(extra.iter().map(|(n, b)| (n.to_string(), b.to_vec())));
        v
    }

    fn pack(dir: &Path, entries: &[(String, Vec<u8>)]) -> PathBuf {
        let refs: Vec<(&str, &[u8])> = entries
            .iter()
            .map(|(n, b)| (n.as_str(), b.as_slice()))
            .collect();
        tarball(dir, &refs)
    }

    const STAGING: &str = ".urna-setup-test";

    /// What `install` does with a fault injected: unpack `entries` into a
    /// staging dir under `root`, swap it in, clean the staging dir up.
    fn swap_faulty(
        d: &Path,
        root: &Path,
        entries: &[(String, Vec<u8>)],
        fault: Fault<'_>,
    ) -> Result<PathBuf> {
        let tgz = pack(d, entries);
        let staging = root.join(STAGING);
        let _ = std::fs::remove_dir_all(&staging);
        std::fs::create_dir_all(&staging).unwrap();
        unpack_into(&tgz, &staging, &mut |_| {}).unwrap();
        let r = swap(&staging, root, fault);
        finish(&staging);
        r
    }

    fn swap_staged(
        d: &Path,
        root: &Path,
        entries: &[(String, Vec<u8>)],
        place: Option<&str>,
    ) -> Result<PathBuf> {
        let fault = Fault {
            place,
            ..Fault::default()
        };
        swap_faulty(d, root, entries, fault)
    }

    /// The `.urna-previous-*` dirs a failed restore left under `root`.
    fn kept_copies(root: &Path) -> Vec<PathBuf> {
        std::fs::read_dir(root)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with(".urna-previous-"))
            })
            .collect()
    }

    fn read(root: &Path, rel: &str) -> String {
        String::from_utf8(std::fs::read(root.join("urna").join(rel)).unwrap()).unwrap()
    }

    #[test]
    fn installs_a_payload_and_keeps_the_venv() {
        let d = tmp("ok");
        let root = d.join("root");
        std::fs::create_dir_all(root.join("urna/venv/bin")).unwrap();
        std::fs::write(root.join("urna/venv/bin/python"), b"venv").unwrap();
        let tgz = pack(&d, &payload("0.5.2", &[]));
        let mut seen = 0;
        let package = install(&tgz, &root, |n| seen = n).unwrap();
        assert!(package.join("embed/potionqry.py").is_file());
        assert_eq!(seen, REQUIRED.len());
        assert!(missing(&root.join("urna")).is_empty());
        assert_eq!(read(&root, "VERSION"), "0.5.2\n");
        assert_eq!(read(&root, "venv/bin/python"), "venv");
        assert!(!root.join(".urna-setup-test").exists());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn an_upgrade_replaces_every_file_of_the_previous_release() {
        let d = tmp("upgrade");
        let root = d.join("root");
        swap_staged(&d, &root, &payload("0.5.1", &[]), None).unwrap();
        swap_staged(&d, &root, &payload("0.5.2", &[]), None).unwrap();
        for rel in REQUIRED.iter().filter(|r| **r != "VERSION") {
            assert_eq!(read(&root, rel), format!("{rel}@0.5.2"));
        }
        assert_eq!(read(&root, "VERSION"), "0.5.2\n");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_failed_upgrade_puts_the_previous_payload_back() {
        // stopped after the top-level files went in, and again after the
        // package: either way every file is 0.5.1's again and the stamp says
        // 0.5.1.
        for stop in ["python", "VERSION"] {
            let d = tmp(&format!("restore_{}", stop.to_lowercase()));
            let root = d.join("root");
            std::fs::create_dir_all(root.join("urna/venv")).unwrap();
            std::fs::write(root.join("urna/venv/marker"), b"keep").unwrap();
            swap_staged(&d, &root, &payload("0.5.1", &[]), None).unwrap();
            let err = swap_staged(&d, &root, &payload("0.5.2", &[]), Some(stop)).unwrap_err();
            assert!(
                format!("{err:#}").contains("previous payload is back"),
                "{err:#}"
            );
            for rel in REQUIRED.iter().filter(|r| **r != "VERSION") {
                assert_eq!(read(&root, rel), format!("{rel}@0.5.1"), "{stop}: {rel}");
            }
            assert_eq!(read(&root, "VERSION"), "0.5.1\n", "{stop}");
            assert_eq!(read(&root, "venv/marker"), "keep");
            std::fs::remove_dir_all(&d).unwrap();
        }
    }

    #[test]
    fn a_failed_restore_keeps_the_previous_files_and_names_where() {
        // the upgrade stops before the stamp, and putting the previous python/
        // back fails too: python/ must survive outside the staging dir, the
        // error must say so and where, the rest must be back in place.
        let d = tmp("restore_fails");
        let root = d.join("root");
        swap_staged(&d, &root, &payload("0.5.1", &[]), None).unwrap();
        let fault = Fault {
            place: Some("VERSION"),
            restore: Some("python"),
            keep: false,
        };
        let err = swap_faulty(&d, &root, &payload("0.5.2", &[]), fault).unwrap_err();
        let msg = format!("{err:#}");
        assert!(!msg.contains("is back in place"), "{msg}");
        let kept = kept_copies(&root);
        assert_eq!(kept.len(), 1, "{kept:?}");
        assert!(
            msg.contains("previous python failed") && msg.contains(&kept[0].display().to_string()),
            "{msg}"
        );
        let potion = kept[0].join("python/urna/embed/potionqry.py");
        assert_eq!(
            std::fs::read_to_string(potion).unwrap(),
            "python/urna/embed/potionqry.py@0.5.1"
        );
        assert_eq!(read(&root, "VERSION"), "0.5.1\n");
        assert!(
            !root.join("urna/python").exists(),
            "the new package must not stay half-installed"
        );
        assert!(!root.join(STAGING).exists());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_failed_restore_that_cannot_move_the_copy_keeps_the_staging_dir() {
        let d = tmp("keep_staging");
        let root = d.join("root");
        swap_staged(&d, &root, &payload("0.5.1", &[]), None).unwrap();
        let fault = Fault {
            place: Some("VERSION"),
            restore: Some("python"),
            keep: true,
        };
        let err = swap_faulty(&d, &root, &payload("0.5.2", &[]), fault).unwrap_err();
        let previous = root.join(STAGING).join("previous");
        assert!(
            format!("{err:#}").contains(&previous.display().to_string()),
            "{err:#}"
        );
        let potion = previous.join("python/urna/embed/potionqry.py");
        assert_eq!(
            std::fs::read_to_string(potion).unwrap(),
            "python/urna/embed/potionqry.py@0.5.1"
        );
        assert!(kept_copies(&root).is_empty());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_failed_first_install_leaves_no_stamp_and_no_half_payload() {
        let d = tmp("first");
        let root = d.join("root");
        assert!(swap_staged(&d, &root, &payload("0.5.2", &[]), Some("VERSION")).is_err());
        assert!(!root.join("urna/VERSION").exists());
        assert!(!root.join("urna/python").exists());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn refuses_an_incomplete_payload_and_moves_nothing() {
        let d = tmp("incomplete");
        let root = d.join("root");
        swap_staged(&d, &root, &payload("0.5.1", &[]), None).unwrap();
        let mut half = payload("0.5.2", &[]);
        half.retain(|(n, _)| n != "urna/python/urna/embed/presetqry.py");
        let err = swap_staged(&d, &root, &half, None).unwrap_err();
        assert!(
            err.to_string()
                .contains("urna/python/urna/embed/presetqry.py"),
            "{err}"
        );
        assert_eq!(read(&root, "VERSION"), "0.5.1\n");
        assert_eq!(
            read(&root, "python/urna/embed/searchtxt.py"),
            "python/urna/embed/searchtxt.py@0.5.1"
        );
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn missing_names_what_an_installed_payload_lacks() {
        // the two the review removed by hand, each of which breaks a query:
        // the module the potion route imports and the table's tokenizer.
        for rel in [
            "python/urna/embed/potiontab.py",
            "python/urna/model/potionb8m/tokenizer.json",
        ] {
            let d = tmp(&format!("missing_{}", rel.len()));
            let root = d.join("root");
            swap_staged(&d, &root, &payload("0.5.2", &[]), None).unwrap();
            assert!(missing(&root.join("urna")).is_empty());
            std::fs::remove_file(root.join("urna").join(rel)).unwrap();
            assert_eq!(missing(&root.join("urna")), vec![rel]);
            std::fs::remove_dir_all(&d).unwrap();
        }
    }

    #[test]
    fn refuses_a_payload_that_carries_a_venv_and_moves_nothing() {
        let d = tmp("venv");
        let root = d.join("root");
        swap_staged(&d, &root, &payload("0.5.1", &[]), None).unwrap();
        let hostile = payload("0.5.2", &[("urna/venv/bin/python", b"divergent")]);
        let err = swap_staged(&d, &root, &hostile, None).unwrap_err();
        assert!(err.to_string().contains("urna/venv"), "{err}");
        assert_eq!(
            read(&root, "python/urna/model/modelhash.py"),
            "python/urna/model/modelhash.py@0.5.1"
        );
        assert!(!root.join("urna/venv").exists());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn rejects_path_traversal_and_leaves_the_old_payload() {
        let d = tmp("evil");
        let root = d.join("root");
        std::fs::create_dir_all(root.join("urna/python")).unwrap();
        std::fs::write(root.join("urna/python/keep"), b"old").unwrap();
        let tgz = tarball(&d, &[("../../escaped", b"x")]);
        assert!(install(&tgz, &root, |_| {}).is_err());
        assert!(!d.join("escaped").exists());
        assert!(root.join("urna/python/keep").is_file());
        std::fs::remove_dir_all(&d).unwrap();
    }
}
