//! The one model install `urna setup` and the explorer share: the catalog
//! the payload ships (`urna/model/catalogue.json`, generated from the registry),
//! and the install of one entry. packages go into the managed venv (uv when
//! it is on PATH, else that env's pip); then the payload's
//! `urna/model/installer.py`, run by the same interpreter, fetches the pinned
//! revision into the hugging face cache and proves its model_hash; last,
//! the payload's query embedder loads the model and asks it once, offline,
//! the proof a query will work. the download needs `Consent::download`, a
//! model that runs repo code also `Consent::remote_code`; without them
//! nothing is installed or fetched.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;

use super::venv;

pub use super::catalog::*;

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

    /// What a package install is resolved against: the lock beside the
    /// catalogue and torch's cpu index, when the payload carries them.
    pub fn pins(&self) -> venv::Pins {
        let lock = self
            .catalog
            .lock
            .as_ref()
            .map(|l| self.script.with_file_name(&l.file));
        venv::Pins {
            lock: lock.filter(|p| p.is_file()),
            torch_index: self.catalog.torch.as_ref().map(|t| t.index.clone()),
        }
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
                "{name} runs code from its model repo, reviewed and pinned; that needs its own consent (--allow-remote-code {name})"
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
        let cmd = venv::add_command(uv, py, &p.packages_missing, &kit.pins());
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
        Some(0) => {}
        Some(3) => {
            return Err(Refusal::NeedsDownload {
                bytes: p.missing_bytes,
            });
        }
        Some(5) => {
            return Err(Refusal::NeedsRemoteCode {
                name: entry.name.clone(),
            });
        }
        _ => return Err(failed("fetch", last_error(&errors))),
    }
    log(Progress::Step(format!(
        "query: load {} and ask it once, offline",
        entry.name
    )));
    let asked = query(kit, entry, py, consent)?;
    if let Some(want) = expect
        && asked.model_hash != want
    {
        return Err(failed(
            "query",
            format!(
                "the corpus was built with {want}, but {} loads here as {}: on this machine it would not answer this corpus",
                entry.name, asked.model_hash
            ),
        ));
    }
    Ok(format!(
        "{} at {} · {} · model_hash {}… · answered a query in {:.1} s",
        entry.name,
        short(&entry.revision),
        mb(entry.bytes),
        &entry.model_hash[..entry.model_hash.len().min(19)],
        asked.secs
    ))
}

/// The text setup asks a model it just installed.
const PROBE: &str = "urna setup: does this model answer offline?";

/// What the query step read off the query embedder.
#[derive(Debug, Deserialize)]
pub struct Asked {
    pub model_hash: String,
    pub embedding_dim: u64,
    #[serde(skip)]
    pub secs: f64,
}

/// Loads `entry` through the payload's query embedder (`presetqry.py`, what
/// `ask` runs) and embeds one query, with downloads off whatever the fetch
/// was allowed: a model that answers here answers `ask` with no network.
pub fn query(kit: &Kit, entry: &Entry, py: &Path, consent: Consent) -> Result<Asked, Refusal> {
    let script = kit
        .script
        .parent()
        .and_then(Path::parent)
        .map(|pkg| pkg.join("embed").join("presetqry.py"))
        .ok_or_else(|| failed("query", "the payload has no embed/presetqry.py".into()))?;
    let mut c = Command::new(py);
    c.arg(&script)
        .args(["--preset", &entry.name, &entry.embedding_model, PROBE])
        .env_remove("URNA_ALLOW_DOWNLOAD");
    if consent.remote_code {
        c.env("URNA_ALLOW_REMOTE_CODE", &entry.name);
    }
    let t0 = std::time::Instant::now();
    let out = c
        .output()
        .map_err(|e| failed("query", format!("{}: {e}", py.display())))?;
    if !out.status.success() {
        return Err(failed(
            "query",
            last_error(&String::from_utf8_lossy(&out.stderr)),
        ));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut asked: Asked =
        serde_json::from_str(text.trim()).map_err(|e| failed("query", e.to_string()))?;
    if asked.embedding_dim == 0 {
        return Err(failed("query", "the model returned an empty vector".into()));
    }
    asked.secs = t0.elapsed().as_secs_f64();
    Ok(asked)
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

    /// A fake interpreter for `install`: answers `installer.py plan` (cached,
    /// nothing to install), `installer.py fetch` (ok) and `presetqry.py`
    /// with `hash`, so the query step runs with no python and no network.
    #[cfg(unix)]
    fn fake_python(dir: &Path, hash: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let py = dir.join("python");
        let script = format!(
            r#"#!/bin/sh
case "$1" in
  *installer.py) [ "$2" = plan ] && echo '{{"missing_bytes": 0, "cached": true, "packages_missing": [], "remote_code": false}}'; exit 0 ;;
  *presetqry.py) [ -z "$URNA_ALLOW_DOWNLOAD" ] || exit 9; echo '{{"model_hash": "{hash}", "embedding_dim": 4}}' ;;
esac
"#
        );
        std::fs::write(&py, script).unwrap();
        std::fs::set_permissions(&py, std::fs::Permissions::from_mode(0o755)).unwrap();
        py
    }

    #[cfg(unix)]
    #[test]
    fn an_install_ends_with_an_offline_query_that_must_match_the_corpus() {
        let dir = std::env::temp_dir().join(format!("urna_query_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let kit = Kit {
            script: dir.join("urna/model/installer.py"),
            catalog: sample(),
        };
        let a = kit.catalog.models[0].clone();
        let consent = Consent {
            download: true,
            remote_code: false,
        };
        let py = fake_python(&dir, "sha256:corpus");
        let mut steps = Vec::new();
        let ok = install(
            &kit,
            &a,
            &py,
            None,
            consent,
            Some("sha256:corpus"),
            &mut |p| {
                if let Progress::Step(s) = p {
                    steps.push(s)
                }
            },
        )
        .unwrap();
        assert!(ok.contains("answered a query"), "{ok}");
        assert!(
            steps.iter().any(|s| s.contains("ask it once, offline")),
            "{steps:?}"
        );
        // the query loads the model as another hash: this corpus would not answer.
        let py = fake_python(&dir, "sha256:other");
        let err = install(
            &kit,
            &a,
            &py,
            None,
            consent,
            Some("sha256:corpus"),
            &mut |_| {},
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.starts_with("query failed") && err.contains("would not answer this corpus"),
            "{err}"
        );
        // no corpus to match: any hash the model loads as is an install.
        assert!(install(&kit, &a, &py, None, consent, None, &mut |_| {}).is_ok());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
