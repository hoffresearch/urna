//! The uv setup downloads when the machine has none: the release the
//! payload's catalogue names, one archive per platform, checked against
//! the sha256 the catalogue holds before anything is unpacked. the
//! catalogue is generated (`tool/tasks/upstreams.py`) and checked in ci,
//! so a user only ever gets a uv urna's ci has run; nothing here asks
//! github which uv is the latest.
//!
//! the binary lands at `<home>/bin/uv` (`paths::managed_uv`); the archive
//! is a `.tar.gz` holding `uv-<target>/uv`, or on windows a `.zip` holding
//! `uv.exe`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use super::models::UvRelease;
use super::net;

/// The archive of `release` for `target` (`<arch>-<os>`, `Scan::target`).
pub fn archive<'a>(release: &'a UvRelease, target: &str) -> Option<&'a super::models::Archive> {
    release.archives.get(target)
}

/// Downloads the uv of `release` for `target` into `home`, checks its
/// sha256 and unpacks the binary; returns its path. a mismatch removes the
/// download and installs nothing.
pub fn install(
    curl: &Path,
    release: &UvRelease,
    target: &str,
    home: &Path,
    mut progress: impl FnMut(u64),
) -> Result<PathBuf> {
    let Some(a) = archive(release, target) else {
        bail!(
            "the catalogue names no uv {} archive for {target}",
            release.version
        );
    };
    let dest = crate::cmd::paths::managed_uv(home);
    let dir = dest.parent().context("uv path has no parent")?;
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let tmp = dir.join(format!(".uv-{}.download", std::process::id()));
    let got = net::download(curl, &a.url, &tmp, &mut progress);
    let got = match got {
        Ok(g) => g,
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
    };
    if got != a.sha256 {
        let _ = std::fs::remove_file(&tmp);
        bail!(
            "uv {} does not match the catalogue's sha256 (got {}…, want {}…); nothing was installed",
            release.version,
            &got[..12],
            &a.sha256[..a.sha256.len().min(12)]
        );
    }
    let res = if a.url.ends_with(".zip") {
        unzip(&tmp, &dest)
    } else {
        untar(&tmp, &dest)
    };
    let _ = std::fs::remove_file(&tmp);
    res?;
    Ok(dest)
}

/// The name of the uv binary inside an archive.
fn binary_name() -> &'static str {
    if cfg!(windows) { "uv.exe" } else { "uv" }
}

/// Writes `bytes` to `dest` through a sibling temp file, executable on
/// unix, so a half-written uv is never at `dest`.
fn place(mut src: impl std::io::Read, dest: &Path) -> Result<()> {
    let part = dest.with_extension("part");
    {
        let mut out =
            std::fs::File::create(&part).with_context(|| format!("create {}", part.display()))?;
        std::io::copy(&mut src, &mut out).context("unpack uv")?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&part, std::fs::Permissions::from_mode(0o755))?;
    }
    std::fs::rename(&part, dest).with_context(|| format!("move uv to {}", dest.display()))?;
    Ok(())
}

/// The `uv` entry of a `.tar.gz` (any folder depth), and nothing else.
fn untar(archive: &Path, dest: &Path) -> Result<()> {
    let file = std::fs::File::open(archive)?;
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(file));
    for entry in tar.entries().context("read the uv archive")? {
        let entry = entry?;
        let is_uv = entry.header().entry_type().is_file()
            && entry
                .path()?
                .file_name()
                .is_some_and(|n| n == binary_name());
        if is_uv {
            return place(entry, dest);
        }
    }
    bail!("the uv archive holds no {}", binary_name())
}

#[cfg(windows)]
fn unzip(archive: &Path, dest: &Path) -> Result<()> {
    let file = std::fs::File::open(archive)?;
    let mut zip = zip::ZipArchive::new(file).context("read the uv archive")?;
    for i in 0..zip.len() {
        let entry = zip.by_index(i)?;
        let is_uv = entry.is_file()
            && entry
                .enclosed_name()
                .and_then(|p| p.file_name().map(|n| n == binary_name()))
                .unwrap_or(false);
        if is_uv {
            return place(entry, dest);
        }
    }
    bail!("the uv archive holds no {}", binary_name())
}

#[cfg(not(windows))]
fn unzip(_archive: &Path, _dest: &Path) -> Result<()> {
    bail!("a .zip uv archive is the windows build; this platform takes the .tar.gz")
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use sha2::{Digest, Sha256};

    fn url(p: &Path) -> String {
        let s = p.display().to_string().replace('\\', "/");
        if s.starts_with('/') {
            format!("file://{s}")
        } else {
            format!("file:///{s}")
        }
    }

    /// a `.tar.gz` laid out like uv's: `uv-x/uvx` then `uv-x/uv`.
    fn uv_tarball(path: &Path, body: &[u8]) {
        let gz = GzEncoder::new(std::fs::File::create(path).unwrap(), Compression::fast());
        let mut tar = tar::Builder::new(gz);
        for (name, data) in [("uv-x/uvx", &b"uvx"[..]), ("uv-x/uv", body)] {
            let mut h = tar::Header::new_gnu();
            h.set_size(data.len() as u64);
            h.set_mode(0o755);
            h.set_cksum();
            tar.append_data(&mut h, name, data).unwrap();
        }
        tar.into_inner().unwrap().finish().unwrap();
    }

    fn release(url: String, sha256: String) -> UvRelease {
        let mut archives = std::collections::BTreeMap::new();
        archives.insert(
            "t-os".to_string(),
            super::super::models::Archive { url, sha256 },
        );
        UvRelease {
            version: "0.0.1".into(),
            archives,
        }
    }

    #[test]
    fn installs_the_checked_archive_and_refuses_other_bytes() {
        if cfg!(windows) {
            return;
        }
        let Some(curl) = crate::cmd::paths::which("curl") else {
            return;
        };
        let d = std::env::temp_dir().join(format!("urna_uv_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let tgz = d.join("uv.tar.gz");
        uv_tarball(&tgz, b"#!/bin/sh\necho uv 0.0.1\n");
        let sha = hex::encode(Sha256::digest(std::fs::read(&tgz).unwrap()));
        let home = d.join("home");

        let bad = release(url(&tgz), "0".repeat(64));
        let err = install(&curl, &bad, "t-os", &home, |_| {}).unwrap_err();
        assert!(err.to_string().contains("sha256"), "{err}");
        assert!(!crate::cmd::paths::managed_uv(&home).exists());

        let good = release(url(&tgz), sha);
        let missing = install(&curl, &good, "other-os", &home, |_| {}).unwrap_err();
        assert!(
            missing
                .to_string()
                .contains("no uv 0.0.1 archive for other-os")
        );

        let uv = install(&curl, &good, "t-os", &home, |_| {}).unwrap();
        assert_eq!(uv, crate::cmd::paths::managed_uv(&home));
        assert_eq!(
            std::fs::read(&uv).unwrap(),
            b"#!/bin/sh\necho uv 0.0.1\n".to_vec()
        );
        // retried while busy: a child another test forks during the write
        // holds the descriptor until its exec (ETXTBSY, os error 26), the
        // race setup_e2e's `output` retries too.
        let run = || std::process::Command::new(&uv).output();
        let mut out = run();
        for _ in 0..100 {
            match &out {
                Err(e) if e.raw_os_error() == Some(26) => {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    out = run();
                }
                _ => break,
            }
        }
        let out = out.unwrap();
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "uv 0.0.1");
        let leftovers: Vec<_> = std::fs::read_dir(uv.parent().unwrap())
            .unwrap()
            .flatten()
            .map(|e| e.file_name())
            .collect();
        assert_eq!(leftovers, [std::ffi::OsString::from("uv")]);
        std::fs::remove_dir_all(&d).unwrap();
    }
}
