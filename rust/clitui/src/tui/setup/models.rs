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

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Catalog {
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
    /// a step ran and failed; the detail is its last error line.
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
        let mut tail = String::new();
        let status = venv::stream_status(cmd, &mut |l| {
            tail = l.clone();
            log(Progress::Log(l));
        })
        .map_err(|e| failed("packages", format!("{e:#}")))?;
        if !status.success() {
            return Err(failed("packages", tail));
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
    fn sizes_and_refusals_read_plainly() {
        assert_eq!(mb(479_729_010), "480 MB");
        assert_eq!(
            Refusal::NeedsDownload { bytes: 479_729_010 }.to_string(),
            "the weights (480 MB) are not in the local cache and the download was not confirmed"
        );
    }
}
