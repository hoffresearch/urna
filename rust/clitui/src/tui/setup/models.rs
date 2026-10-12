//! The one model install `urna setup` and the explorer share: the catalog
//! the payload ships (`urna/model/catalogue.json`, generated from the registry),
//! and the install of one entry. packages go into the managed venv (uv when
//! it is on PATH, else that env's pip); then the payload's
//! `urna/model/installer.py`, run by the same interpreter, fetches the pinned
//! revision into the hugging face cache and proves its model_hash. the
//! download needs `Consent::download`, a model that runs repo code also
//! `Consent::remote_code`; without them nothing is installed or fetched.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;

use super::venv;

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct Entry {
    pub name: String,
    pub embedding_model: String,
    pub repo: String,
    pub revision: String,
    pub bytes: u64,
    pub model_hash: String,
    pub packages: Vec<String>,
    pub remote_code: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct Excluded {
    pub name: String,
    pub reason: String,
}

/// One uv archive: where it is and the sha256 its release publishes.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct Archive {
    pub url: String,
    pub sha256: String,
}

/// The uv release setup downloads when the machine has none, one archive
/// per platform urna ships, keyed `<arch>-<os>` (`Scan::target`).
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
pub struct UvRelease {
    pub version: String,
    pub archives: std::collections::BTreeMap<String, Archive>,
}

/// The python range the env is built on: the cpython minors torch has
/// wheels for (`>=3.10,<3.15`), read off torch's wheels.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
pub struct PythonRange {
    pub requires: String,
    pub torch: String,
}

/// The sample corpus setup offers: one release of a dataset on hugging
/// face, pinned to a commit, with the size and sha256 read at that commit.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
pub struct Sample {
    pub repo: String,
    pub release: String,
    pub commit: String,
    pub path: String,
    pub url: String,
    pub size: u64,
    pub sha256: String,
}

/// A payload from 0.5.4 or older carries no `uv`, no `python` and no
/// `sample`: setup then uses the uv on PATH, lets it pick the python, as it
/// did, and offers no sample.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct Catalog {
    #[serde(default)]
    pub uv: Option<UvRelease>,
    #[serde(default)]
    pub python: Option<PythonRange>,
    #[serde(default)]
    pub sample: Option<Sample>,
    pub models: Vec<Entry>,
    pub excluded: Vec<Excluded>,
}

impl Catalog {
    pub fn parse(text: &str) -> anyhow::Result<Catalog> {
        Ok(serde_json::from_str(text)?)
    }

    /// The offered entry for a catalog name or a manifest model name; an
    /// excluded or unknown name is an error that says why.
    pub fn find(&self, name: &str) -> Result<&Entry, String> {
        if let Some(e) = self
            .models
            .iter()
            .find(|e| e.name == name || e.embedding_model == name)
        {
            return Ok(e);
        }
        if let Some(x) = self.excluded.iter().find(|x| x.name == name) {
            return Err(format!("{} is not offered: {}", x.name, x.reason));
        }
        let offered: Vec<&str> = self.models.iter().map(|e| e.name.as_str()).collect();
        Err(format!(
            "{name} is not in the model catalog (offered: {})",
            if offered.is_empty() {
                "none".to_string()
            } else {
                offered.join(", ")
            }
        ))
    }

    /// `--model` values to entries, in catalog order without repeats; `all`
    /// selects every offered model.
    pub fn select(&self, names: &[String]) -> Result<Vec<Entry>, String> {
        if names.iter().any(|n| n == "all") {
            return Ok(self.models.clone());
        }
        let mut picked = Vec::new();
        for n in names {
            let e = self.find(n)?;
            if !picked.contains(e) {
                picked.push(e.clone());
            }
        }
        picked.sort_by_key(|e| self.models.iter().position(|m| m == e));
        Ok(picked)
    }
}

/// The installer a payload carries: `urna/model/installer.py` and the
/// catalog beside it.
#[derive(Clone, Debug)]
pub struct Kit {
    pub script: PathBuf,
    pub catalog: Catalog,
}

impl Kit {
    /// The kit in a package's `model/` dir, when both files are there.
    pub fn at(dir: &Path) -> Option<Kit> {
        let text = std::fs::read_to_string(dir.join("catalogue.json")).ok()?;
        let script = dir.join("installer.py");
        let catalog = Catalog::parse(&text).ok()?;
        script.is_file().then_some(Kit { script, catalog })
    }

    /// The kit the query embedders resolve to (the checkout's, then each
    /// data root's payload): the one the explorer offers from.
    pub fn resolve() -> Option<Kit> {
        let script = crate::cmd::embed_gate::installed_script_in(&["model", "installer.py"]);
        Kit::at(script.parent()?)
    }

    /// The kit of the payload laid down in `home` (`<root>/urna`).
    pub fn in_home(home: &Path) -> Option<Kit> {
        Kit::at(&home.join(crate::cmd::payload::PACKAGE).join("model"))
    }

    /// The python range the catalogue names, when it does.
    pub fn range(&self) -> Option<super::venv::Range> {
        self.catalog
            .python
            .as_ref()
            .and_then(|p| super::venv::Range::parse(&p.requires))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Consent {
    /// the user confirmed fetching the weights from huggingface.co.
    pub download: bool,
    /// the user separately allowed this model's repo code to run.
    pub remote_code: bool,
}

/// What an install still has to do, read off `installer.py plan`.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
pub struct Plan {
    pub missing_bytes: u64,
    pub cached: bool,
    pub packages_missing: Vec<String>,
    pub remote_code: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Progress {
    /// a step started: packages, fetch, verify.
    Step(String),
    /// bytes fetched so far, of the model's total, and the file at hand.
    Bytes(u64, u64, String),
    Log(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// queries do not run the managed venv (URNA_PYTHON pins another, or
    /// setup has not built it): urna installs packages nowhere else.
    Unmanaged { python: String, detail: String },
    /// the weights are not cached and the download was not confirmed.
    NeedsDownload { bytes: u64 },
    /// the model runs repo code and that was not separately allowed.
    NeedsRemoteCode { name: String },
    /// a step ran and failed; the detail is its error, one line or the
    /// lines of uv's own error.
    Failed { step: &'static str, detail: String },
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::Unmanaged { python, detail } => write!(
                f,
                "{detail}; urna installs packages only into the venv `urna setup` manages (queries run {python})"
            ),
            Refusal::NeedsDownload { bytes } => write!(
                f,
                "the weights ({}) are not in the local cache and the download was not confirmed",
                mb(*bytes)
            ),
            Refusal::NeedsRemoteCode { name } => write!(
                f,
                "{name} runs code from its model repo; that needs its own consent (--allow-remote-code {name})"
            ),
            Refusal::Failed { step, detail } => write!(f, "{step} failed: {detail}"),
        }
    }
}

pub fn mb(bytes: u64) -> String {
    if bytes >= 1_000_000 {
        format!("{:.0} MB", bytes as f64 / 1e6)
    } else {
        format!("{:.1} MB", bytes as f64 / 1e6)
    }
}

/// The interpreter packages may go into: the managed venv, and only when
/// queries will run it (an `URNA_PYTHON` pin wins over it on the ladder).
/// pure: the caller reads the env and the data dir.
pub fn managed_python(pinned: Option<&str>, venv_py: Option<&Path>) -> Result<PathBuf, Refusal> {
    match (pinned, venv_py) {
        (Some(p), Some(v)) if Path::new(p) == v => Ok(v.to_path_buf()),
        (Some(p), _) => Err(Refusal::Unmanaged {
            python: p.to_string(),
            detail: format!("URNA_PYTHON pins {p}"),
        }),
        (None, Some(v)) if v.is_file() => Ok(v.to_path_buf()),
        (None, v) => Err(Refusal::Unmanaged {
            python: v.map(|v| v.display().to_string()).unwrap_or_default(),
            detail: "the managed venv is not built yet: run `urna setup`".into(),
        }),
    }
}

fn pinned() -> Option<String> {
    std::env::var("URNA_PYTHON").ok().filter(|s| !s.is_empty())
}

/// The venv this setup manages: the one in its own data dir, never another
/// data root's (that belongs to another install). `URNA_PYTHON` read here.
pub fn setup_python() -> Result<PathBuf, Refusal> {
    let venv =
        crate::cmd::paths::urna_home().map(|h| crate::cmd::paths::venv_python(&h.join("venv")));
    managed_python(pinned().as_deref(), venv.as_deref())
}

/// The venv queries run, when `urna setup` made it: the explorer installs
/// there so the retried query finds what it installed.
pub fn query_python() -> Result<PathBuf, Refusal> {
    managed_python(
        pinned().as_deref(),
        crate::cmd::paths::setup_python().as_deref(),
    )
}

fn script_cmd(py: &Path, kit: &Kit, action: &str, entry: &Entry, expect: Option<&str>) -> Command {
    let mut c = Command::new(py);
    c.arg(&kit.script).args([action, entry.name.as_str()]);
    if let Some(h) = expect {
        c.args(["--expect-hash", h]);
    }
    c
}

/// What installing `entry` still needs, with no network: missing packages,
/// whether the pinned revision is cached, remote code.
pub fn plan(kit: &Kit, entry: &Entry, py: &Path, expect: Option<&str>) -> Result<Plan, Refusal> {
    let out = script_cmd(py, kit, "plan", entry, expect)
        .output()
        .map_err(|e| failed("plan", format!("{}: {e}", py.display())))?;
    if !out.status.success() {
        return Err(failed(
            "plan",
            last_error(&String::from_utf8_lossy(&out.stderr)),
        ));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    serde_json::from_str(text.trim()).map_err(|e| failed("plan", e.to_string()))
}

fn failed(step: &'static str, detail: String) -> Refusal {
    Refusal::Failed { step, detail }
}

/// The last `error: ...` line of a run, else its last non-empty line.
fn last_error(text: &str) -> String {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    lines
        .iter()
        .rev()
        .find_map(|l| l.strip_prefix("error:").map(str::trim))
        .or_else(|| lines.last().copied())
        .unwrap_or("no output")
        .to_string()
}

/// The error of a failed package install, whole: from uv's first `error:`
/// line, or the `×` that opens a resolver failure, to the end (its causes
/// and hints sit below it), else the last lines. the command echo
/// (`$ ...`) and blank lines are left out; at most 24 lines.
fn install_error(lines: &[String]) -> String {
    const LAST: usize = 6;
    const MOST: usize = 24;
    let kept: Vec<&str> = lines
        .iter()
        .map(|l| l.trim_end())
        .filter(|l| !l.trim().is_empty() && !l.starts_with("$ "))
        .collect();
    let start = kept
        .iter()
        .position(|l| {
            let t = l.trim_start();
            t.starts_with("error:") || t.starts_with('×') || t.starts_with("ERROR:")
        })
        .unwrap_or(kept.len().saturating_sub(LAST));
    let tail = &kept[start..];
    let tail = &tail[..tail.len().min(MOST)];
    if tail.is_empty() {
        "no output".into()
    } else {
        tail.join("\n")
    }
}

/// `urna-progress: <done> <total> <file>` to `Progress::Bytes`.
fn progress_line(line: &str) -> Option<Progress> {
    let rest = line.strip_prefix("urna-progress:")?.trim();
    let mut it = rest.splitn(3, ' ');
    let done = it.next()?.parse().ok()?;
    let total = it.next()?.parse().ok()?;
    Some(Progress::Bytes(
        done,
        total,
        it.next().unwrap_or("").to_string(),
    ))
}

/// Installs `entry` into the managed venv `py` and the hf cache: the plan,
/// the missing packages, the fetch, the hash check. refuses before any step
/// that `consent` does not cover.
pub fn install(
    kit: &Kit,
    entry: &Entry,
    py: &Path,
    uv: Option<&Path>,
    consent: Consent,
    expect: Option<&str>,
    log: &mut dyn FnMut(Progress),
) -> Result<String, Refusal> {
    let p = plan(kit, entry, py, expect)?;
    if p.remote_code && !consent.remote_code {
        return Err(Refusal::NeedsRemoteCode {
            name: entry.name.clone(),
        });
    }
    if !p.cached && !consent.download {
        return Err(Refusal::NeedsDownload {
            bytes: p.missing_bytes,
        });
    }
    if !p.packages_missing.is_empty() {
        log(Progress::Step(format!(
            "packages: {}",
            p.packages_missing.join(" ")
        )));
        let cmd = venv::add_command(uv, py, &p.packages_missing);
        let mut lines = Vec::new();
        let status = venv::stream_status(cmd, &mut |l| {
            lines.push(l.clone());
            log(Progress::Log(l));
        })
        .map_err(|e| failed("packages", format!("{e:#}")))?;
        if !status.success() {
            return Err(failed("packages", install_error(&lines)));
        }
    }
    log(Progress::Step(if p.cached {
        format!("verify {} at {}", entry.repo, short(&entry.revision))
    } else {
        format!(
            "fetch {} at {} ({})",
            entry.repo,
            short(&entry.revision),
            mb(p.missing_bytes)
        )
    }));
    let mut cmd = script_cmd(py, kit, "fetch", entry, expect);
    if consent.download {
        cmd.env("URNA_ALLOW_DOWNLOAD", "1");
    }
    if consent.remote_code {
        cmd.arg("--allow-remote-code");
    }
    let mut errors = String::new();
    let status = venv::stream_status(cmd, &mut |l| match progress_line(&l) {
        Some(b) => log(b),
        None => {
            errors.push_str(&l);
            errors.push('\n');
            log(Progress::Log(l));
        }
    })
    .map_err(|e| failed("fetch", format!("{e:#}")))?;
    match status.code() {
        Some(0) => Ok(format!(
            "{} at {} · {} · model_hash {}…",
            entry.name,
            short(&entry.revision),
            mb(entry.bytes),
            &entry.model_hash[..entry.model_hash.len().min(19)]
        )),
        Some(3) => Err(Refusal::NeedsDownload {
            bytes: p.missing_bytes,
        }),
        Some(5) => Err(Refusal::NeedsRemoteCode {
            name: entry.name.clone(),
        }),
        _ => Err(failed("fetch", last_error(&errors))),
    }
}

fn short(rev: &str) -> &str {
    &rev[..rev.len().min(12)]
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// a two-model catalog (`b` runs repo code) with one exclusion.
    pub fn sample() -> Catalog {
        Catalog::parse(CATALOG).unwrap()
    }

    const CATALOG: &str = r#"{"schema": 1,
      "models": [
        {"name": "a", "embedding_model": "org/a", "repo": "org/a", "revision": "aaaa",
         "bytes": 5, "model_hash": "sha256:a", "packages": ["p"], "remote_code": false,
         "files": [], "imports": ["p"], "remote_code_hashes": [], "dim": 4, "modalities": ["text"]},
        {"name": "b", "embedding_model": "org/b", "repo": "org/b", "revision": "bbbb",
         "bytes": 7, "model_hash": "sha256:b", "packages": [], "remote_code": true,
         "files": [], "imports": [], "remote_code_hashes": [], "dim": 4, "modalities": ["text"]}],
      "excluded": [{"name": "big", "embedding_model": "org/big", "reason": "flagged too heavy"}]}"#;

    #[test]
    fn select_resolves_names_models_and_all_in_catalog_order() {
        let c = Catalog::parse(CATALOG).unwrap();
        let names = |v: Vec<Entry>| v.into_iter().map(|e| e.name).collect::<Vec<_>>();
        assert_eq!(names(c.select(&["all".into()]).unwrap()), ["a", "b"]);
        let picked = c.select(&["org/b".into(), "a".into(), "b".into()]).unwrap();
        assert_eq!(names(picked), ["a", "b"]);
        let err = c.select(&["big".into()]).unwrap_err();
        assert!(
            err.contains("not offered") && err.contains("too heavy"),
            "{err}"
        );
        let err = c.select(&["nope".into()]).unwrap_err();
        assert!(err.contains("offered: a, b"), "{err}");
    }

    #[test]
    fn packages_go_only_into_the_venv_queries_run() {
        let dir = std::env::temp_dir().join(format!("urna_models_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let v = dir.join("python");
        std::fs::write(&v, b"").unwrap();
        assert_eq!(managed_python(None, Some(&v)), Ok(v.clone()));
        let same = v.to_str().unwrap();
        assert_eq!(managed_python(Some(same), Some(&v)), Ok(v.clone()));
        let pinned = managed_python(Some("/usr/bin/python3"), Some(&v)).unwrap_err();
        assert!(
            pinned
                .to_string()
                .contains("URNA_PYTHON pins /usr/bin/python3")
        );
        assert!(pinned.to_string().contains("only into the venv"));
        let none = managed_python(None, Some(&dir.join("absent"))).unwrap_err();
        assert!(none.to_string().contains("run `urna setup`"), "{none}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn progress_lines_and_errors_are_read_off_the_fetch() {
        assert_eq!(
            progress_line("urna-progress: 10 470641600 model.safetensors"),
            Some(Progress::Bytes(10, 470641600, "model.safetensors".into()))
        );
        assert_eq!(progress_line("urna-progress: x"), None);
        assert_eq!(progress_line("Downloading"), None);
        assert_eq!(
            last_error("noise\nerror: the hash does not match\nurna-fetch: x\n"),
            "the hash does not match"
        );
        assert_eq!(last_error("Segmentation fault\n"), "Segmentation fault");
    }

    #[test]
    fn a_failed_install_keeps_uvs_whole_error() {
        let s = |v: &[&str]| v.iter().map(|l| l.to_string()).collect::<Vec<_>>();
        // the case #529 names: the hint alone said nothing.
        let resolver = s(&[
            "$ uv pip install --python /h/venv/bin/python sentence-transformers>=3",
            "Resolved 30 packages in 1.2s",
            "  × No solution found when resolving dependencies:",
            "  ╰─▶ Because torch==2.14.1 has no wheels with a matching Python ABI tag and",
            "      sentence-transformers>=3 depends on torch, we can conclude that",
            "      sentence-transformers>=3 cannot be used.",
            "",
            "      hint: You require CPython 3.15 (`cp315`), but we only found wheels for",
            "      `torch` (v2.14.1) with the following Python ABI tags: `cp310` .. `cp314`",
        ]);
        let got = install_error(&resolver);
        assert!(got.starts_with("  × No solution found"), "{got}");
        assert!(got.contains("no wheels with a matching Python ABI") && got.contains("hint:"));
        assert!(!got.contains("Resolved 30") && !got.contains("$ uv"));
        let plain = s(&[
            "Downloading x",
            "error: Failed to fetch: `https://pypi.org/simple/torch/`",
            "  Caused by: dns error",
        ]);
        assert_eq!(
            install_error(&plain),
            "error: Failed to fetch: `https://pypi.org/simple/torch/`\n  Caused by: dns error"
        );
        let none = s(&["a", "b", "c", "d", "e", "f", "g", "h"]);
        assert_eq!(install_error(&none), "c\nd\ne\nf\ng\nh");
        assert_eq!(install_error(&[]), "no output");
    }

    #[test]
    fn sizes_and_refusals_read_plainly() {
        assert_eq!(mb(479_729_010), "480 MB");
        assert_eq!(
            Refusal::NeedsDownload { bytes: 479_729_010 }.to_string(),
            "the weights (480 MB) are not in the local cache and the download was not confirmed"
        );
    }
}
