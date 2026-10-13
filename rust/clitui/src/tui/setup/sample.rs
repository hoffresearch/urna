// SPDX-FileCopyrightText: 2026 Brenner Cruvinel, V. Hoff Research
//
// SPDX-License-Identifier: MIT

//! The sample corpus setup offers a first-time user: the release of the
//! fakenews-ptbr benchmark the payload's catalogue names, Brazilian
//! Portuguese news and fact-checked claims, true and fake side by side, so
//! there is a corpus to ask the moment setup ends. it is built with potion,
//! so it needs nothing beyond the payload.
//!
//! the catalogue pins the dataset's commit and the file's sha256
//! (`tool/tasks/upstreams.py`, checked in ci); nothing here asks the hub
//! which release is the newest. the file lands at `<home>/sample/<name>`,
//! only after its sha256 matched and `urna validate`'s checks passed.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use super::models::Sample;
use super::net;

/// How every place that offers the sample describes it: a fake-news
/// benchmark holds true and fake claims side by side, and a hit carries no
/// label saying which.
pub const ABOUT: &str = "Brazilian Portuguese news, true and fake, for testing search";

/// Where the sample of `sample` lives under `home`.
pub fn dest(home: &Path, sample: &Sample) -> PathBuf {
    let name = Path::new(&sample.path)
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_else(|| "sample.urna".into());
    home.join("sample").join(name)
}

/// The sample already in `home` with the catalogue's bytes, when it is.
pub fn present(home: &Path, sample: &Sample) -> Option<PathBuf> {
    let p = dest(home, sample);
    let meta = std::fs::metadata(&p).ok()?;
    if meta.len() != sample.size {
        return None;
    }
    (sha256_of(&p).ok()? == sample.sha256).then_some(p)
}

/// A sample setup laid down in `home`, whatever its release.
pub fn installed(home: &Path) -> Option<PathBuf> {
    let dir = home.join("sample");
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "urna") && p.is_file())
        .collect();
    found.sort();
    found.into_iter().next()
}

fn sha256_of(p: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut f = std::fs::File::open(p)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex::encode(h.finalize()))
}

/// `20 MB`.
pub fn size(sample: &Sample) -> String {
    format!("{:.0} MB", sample.size as f64 / 1e6)
}

/// Downloads `sample` into `home`, checks its sha256 against the
/// catalogue's and runs `urna validate`'s checks on it; returns its path.
/// a file that fails either is removed, and one already there stays.
pub fn install(
    curl: &Path,
    sample: &Sample,
    home: &Path,
    mut progress: impl FnMut(u64),
) -> Result<PathBuf> {
    let dest = dest(home, sample);
    let dir = dest.parent().context("sample path has no parent")?;
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let tmp = dir.join(format!(".sample-{}.download", std::process::id()));
    let res = fetch(curl, sample, &tmp, &mut progress).and_then(|()| {
        std::fs::rename(&tmp, &dest).with_context(|| format!("move to {}", dest.display()))
    });
    if res.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    res?;
    // a sample of another release, left by an older catalogue.
    for e in std::fs::read_dir(dir)?.flatten() {
        let p = e.path();
        if p != dest && p.extension().is_some_and(|x| x == "urna") {
            let _ = std::fs::remove_file(p);
        }
    }
    Ok(dest)
}

fn fetch(curl: &Path, sample: &Sample, tmp: &Path, progress: &mut impl FnMut(u64)) -> Result<()> {
    let got = net::download(curl, &sample.url, tmp, progress)?;
    if got != sample.sha256 {
        bail!(
            "the sample does not match the catalogue's sha256 (got {}…, want {}…); nothing was saved",
            &got[..12],
            &sample.sha256[..sample.sha256.len().min(12)]
        );
    }
    crate::cmd::validate::check(tmp)
        .map(|_| ())
        .context("the sample fails urna validate; nothing was saved")
}
