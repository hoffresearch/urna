// SPDX-FileCopyrightText: 2026 Brenner Cruvinel, V. Hoff Research
//
// SPDX-License-Identifier: MIT

//! The worker: runs the planned steps in order on its own thread and
//! reports every move as an `Ev`. The interactive screen and the plain
//! `--yes` printer read the same stream, so both modes do exactly the same
//! work. a failed step does not stop the next one: verify always runs, so
//! the user ends on the real state of the install.

use std::sync::mpsc::{Receiver, Sender, channel};

use super::models::{self, Consent, Kit, Progress};
use super::net;
use super::plan::{Opts, Task};
use super::scan::{Scan, tilde};
use super::unpack;
use super::uv;
use super::venv::{self, Built, Range, Tool};
use crate::cmd::health::{self, Row};
use crate::cmd::{paths, pyenv};

/// A step whose failure leaves urna ready: the models and the sample are
/// optional, the potion embedder works without them. the run still exits
/// non-zero (`codes::MODEL`, `codes::SAMPLE`), so a script sees what it
/// asked for is missing.
pub fn optional(task: Task) -> bool {
    matches!(task, Task::Models | Task::Sample)
}

/// setup's own exit codes, above doctor's 2..=6 so the two never collide.
pub mod codes {
    pub const DOWNLOAD: i32 = 10;
    pub const CHECKSUM: i32 = 11;
    pub const UNPACK: i32 = 12;
    pub const PYTHON: i32 = 13;
    pub const BLOCKED: i32 = 14;
    pub const MODEL: i32 = 15;
    pub const SAMPLE: i32 = 16;
}

pub type Failure = (i32, String);

#[derive(Debug)]
pub enum Ev {
    Start(Task),
    Bytes(Task, u64, Option<u64>),
    Note(Task, String),
    Log(String),
    Done(Task, Result<String, Failure>),
    Health(Vec<Row>),
    End,
}

/// Runs `tasks` in order; `opts` carries the release and the model choice.
pub fn spawn(scan: Scan, tasks: Vec<Task>, opts: Opts) -> Receiver<Ev> {
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        pyenv::set_quiet(true);
        let version = super::plan::wanted_version(&scan, &opts);
        let tree = super::plan::from_tree(&scan, &opts);
        for task in tasks {
            let _ = tx.send(Ev::Start(task));
            let res = match task {
                Task::Payload => match tree {
                    Some(t) => payload_tree(&scan, t, &version, &tx),
                    None => payload(&scan, &version, &tx),
                },
                Task::Python => python(&scan, &tx),
                Task::Models => install_models(&opts, &tx),
                Task::Verify => verify(&tx),
                Task::Sample => sample(&scan, &tx),
            };
            let _ = tx.send(Ev::Done(task, res));
        }
        let _ = tx.send(Ev::End);
    });
    rx
}

fn fail(code: i32) -> impl Fn(anyhow::Error) -> Failure {
    move |e| (code, format!("{e:#}"))
}

fn payload(scan: &Scan, version: &str, tx: &Sender<Ev>) -> Result<String, Failure> {
    let t = Task::Payload;
    let curl = scan
        .curl
        .as_deref()
        .ok_or((codes::BLOCKED, "curl is not on PATH".into()))?;
    let root = paths::install_root().ok_or((codes::BLOCKED, "no data dir".into()))?;
    sweep(&root);
    let url = format!("{}/{}", net::release_base(version), net::PAYLOAD);
    let _ = tx.send(Ev::Log(format!("source {url}")));
    let _ = tx.send(Ev::Note(t, "checksum".into()));
    let want = net::fetch_text(curl, &format!("{url}.sha256"))
        .and_then(|s| net::parse_sha256(&s))
        .map_err(fail(codes::DOWNLOAD))?;
    let total = net::content_length(curl, &url);
    let _ = tx.send(Ev::Note(t, "downloading".into()));
    std::fs::create_dir_all(&root)
        .map_err(|e| (codes::UNPACK, format!("{}: {e}", root.display())))?;
    let tmp = root.join(format!(".urna-payload-{}.tar.gz", std::process::id()));
    let got = net::download(curl, &url, &tmp, |n| {
        let _ = tx.send(Ev::Bytes(t, n, total));
    });
    let got = match got {
        Ok(g) => g,
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            return Err(fail(codes::DOWNLOAD)(e));
        }
    };
    let size = std::fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);
    let _ = tx.send(Ev::Note(t, "verifying sha256".into()));
    if got != want {
        let _ = std::fs::remove_file(&tmp);
        let _ = tx.send(Ev::Log(format!("sha256 got  {got}")));
        let _ = tx.send(Ev::Log(format!("sha256 want {want}")));
        return Err((
            codes::CHECKSUM,
            format!(
                "the download does not match the release checksum (got {}…, want {}…); nothing was installed",
                &got[..12],
                &want[..12]
            ),
        ));
    }
    let _ = tx.send(Ev::Log(format!("sha256 ok {got}")));
    let _ = tx.send(Ev::Note(t, "unpacking".into()));
    let package = unpack::install(&tmp, &root, |n| {
        if n % 8 == 0 {
            let _ = tx.send(Ev::Note(t, format!("unpacking · {n} files")));
        }
    });
    let _ = std::fs::remove_file(&tmp);
    let package = package.map_err(fail(codes::UNPACK))?;
    let _ = tx.send(Ev::Log(format!("unpacked into {}", package.display())));
    Ok(format!(
        "{:.1} MB · sha256 {}… · {}",
        size as f64 / 1e6,
        &got[..12],
        tilde(&package)
    ))
}

/// A development build's payload: its checkout staged in the release
/// layout, the potion table fetched from upstream when the checkout holds
/// only its lfs pointer, then laid down like a downloaded payload.
fn payload_tree(
    scan: &Scan,
    tree: &std::path::Path,
    name: &str,
    tx: &Sender<Ev>,
) -> Result<String, Failure> {
    let t = Task::Payload;
    if let Some(why) = super::tree::gone(tree) {
        return Err((codes::BLOCKED, why));
    }
    let root = paths::install_root().ok_or((codes::BLOCKED, "no data dir".into()))?;
    sweep(&root);
    let _ = tx.send(Ev::Log(format!("source {}", tree.display())));
    let _ = tx.send(Ev::Note(t, "copying the checkout".into()));
    let staging = unpack::staging(&root).map_err(fail(codes::UNPACK))?;
    let staged = super::tree::stage(tree, &staging).map_err(fail(codes::UNPACK));
    let fetched = staged.and_then(|table| match table {
        Some(to) => fetch_table(scan, tree, &to, tx).map(Some),
        None => Ok(None),
    });
    let laid = fetched.and_then(|sha| {
        super::tree::stamp(&staging, name).map_err(fail(codes::UNPACK))?;
        Ok(sha)
    });
    let sha = match laid {
        Ok(sha) => sha,
        Err(e) => {
            unpack::discard(&staging);
            return Err(e);
        }
    };
    let _ = tx.send(Ev::Note(t, "installing".into()));
    let package = unpack::install_staged(&staging, &root).map_err(fail(codes::UNPACK))?;
    let _ = tx.send(Ev::Log(format!("installed into {}", package.display())));
    let table = match sha {
        Some(sha) => format!("potion table sha256 {}… (lfs pointer)", &sha[..12]),
        None => "potion table from the checkout".into(),
    };
    Ok(format!("{name} · {table} · {}", tilde(&package)))
}

/// Downloads the potion table into `to`, accepted only when its sha256 is
/// the oid of the checkout's lfs pointer; returns that sha256.
fn fetch_table(
    scan: &Scan,
    tree: &std::path::Path,
    to: &std::path::Path,
    tx: &Sender<Ev>,
) -> Result<String, Failure> {
    let t = Task::Payload;
    let curl = scan
        .curl
        .as_deref()
        .ok_or((codes::BLOCKED, "curl is not on PATH".into()))?;
    let want = super::tree::table_oid(tree).ok_or((
        codes::UNPACK,
        "the checkout's potion table is no lfs pointer".into(),
    ))?;
    let url = super::tree::potion_url();
    let _ = tx.send(Ev::Log(format!("potion table {url}")));
    let _ = tx.send(Ev::Note(t, "downloading the potion table".into()));
    let total = net::content_length(curl, &url);
    let got = net::download(curl, &url, to, |n| {
        let _ = tx.send(Ev::Bytes(t, n, total));
    })
    .map_err(fail(codes::DOWNLOAD))?;
    if got != want {
        let _ = tx.send(Ev::Log(format!("sha256 got  {got}")));
        let _ = tx.send(Ev::Log(format!("sha256 want {want}")));
        return Err((
            codes::CHECKSUM,
            format!(
                "the potion table does not match its lfs pointer (got {}…, want {}…); nothing was installed",
                &got[..12],
                &want[..12]
            ),
        ));
    }
    let _ = tx.send(Ev::Log(format!("sha256 ok {got} (lfs pointer)")));
    Ok(got)
}

/// Removes what an interrupted run left in the data root: a partial
/// download (`.urna-payload-<pid>.tar.gz`) or a staging dir
/// (`.urna-setup-<pid>`). only those two prefixes, only at the top level.
pub fn sweep(root: &std::path::Path) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let p = e.path();
        if name.starts_with(".urna-payload-") && p.is_file() {
            let _ = std::fs::remove_file(&p);
        } else if name.starts_with(".urna-setup-") && p.is_dir() {
            let _ = std::fs::remove_dir_all(&p);
        }
    }
}

/// The tool that builds the env: the uv on PATH (or one setup downloaded
/// before), else the uv the catalogue names, downloaded and checked now,
/// else pip from the interpreter the scan resolved inside the range (found
/// again when the payload step just changed the range). the catalogue is
/// read from the payload the step before may just have laid down.
fn env_tool(
    scan: &Scan,
    kit: Option<&Kit>,
    range: Option<Range>,
    home: &std::path::Path,
    tx: &Sender<Ev>,
) -> Result<Tool, Failure> {
    let t = Task::Python;
    if let Some(uv) = paths::uv() {
        return Ok(Tool::Uv(uv));
    }
    let release = kit.and_then(|k| k.catalog.uv.as_ref());
    let why = match (release, scan.curl.as_deref()) {
        (Some(rel), Some(curl)) => {
            let _ = tx.send(Ev::Note(t, format!("downloading uv {}", rel.version)));
            let total =
                uv::archive(rel, &scan.target).and_then(|a| net::content_length(curl, &a.url));
            match uv::install(curl, rel, &scan.target, home, |n| {
                let _ = tx.send(Ev::Bytes(t, n, total));
            }) {
                Ok(p) => {
                    let _ = tx.send(Ev::Log(format!(
                        "uv {} · sha256 ok · {}",
                        rel.version,
                        tilde(&p)
                    )));
                    return Ok(Tool::Uv(p));
                }
                Err(e) => format!("{e:#}"),
            }
        }
        (Some(_), None) => "curl is not on PATH, so uv cannot be downloaded".into(),
        (None, _) => "the installed payload names no uv to download".into(),
    };
    let base = if range == scan.range() {
        scan.base_python.clone()
    } else {
        venv::find_python(range)
    };
    if let Some(py) = base {
        let _ = tx.send(Ev::Log(format!(
            "no uv ({why}); pip from {}",
            tilde(std::path::Path::new(&py))
        )));
        return Ok(Tool::Pip(py));
    }
    Err((
        codes::BLOCKED,
        format!(
            "{}, then run `urna setup` again",
            venv::no_tool(&why, range)
        ),
    ))
}

fn python(scan: &Scan, tx: &Sender<Ev>) -> Result<String, Failure> {
    let home = paths::urna_home().ok_or((codes::BLOCKED, "no data dir".into()))?;
    let kit = Kit::in_home(&home);
    let range = kit.as_ref().and_then(Kit::range);
    let tool = env_tool(scan, kit.as_ref(), range, &home, tx)?;
    let dir = home.join("venv");
    let _ = tx.send(Ev::Note(
        Task::Python,
        match range {
            Some(r) => format!("building with {} on {r}", tool.name()),
            None => format!("building with {}", tool.name()),
        },
    ));
    let mut last = String::new();
    let (py, built) = venv::build(&tool, &dir, range, &mut |line| {
        if line.starts_with("numpy ") {
            last = line.clone();
        }
        let _ = tx.send(Ev::Log(line));
    })
    .map_err(fail(codes::PYTHON))?;
    let mut msg = format!("{} · {}", tilde(&py), last);
    if built == Built::Rebuilt
        && let Some(kit) = &kit
    {
        msg.push_str(&reinstall_cached(kit, &py, &tool, tx));
    }
    Ok(msg)
}

/// After a rebuild the env holds none of the models' packages; the models
/// whose weights are in the cache get theirs back (no download: the fetch
/// step only re-proves the cached files). one that runs repo code is named,
/// not reinstalled: that needs its own consent.
fn reinstall_cached(kit: &Kit, py: &std::path::Path, tool: &Tool, tx: &Sender<Ev>) -> String {
    let t = Task::Python;
    let uv = match tool {
        Tool::Uv(p) => Some(p.as_path()),
        Tool::Pip(_) => None,
    };
    let (mut back, mut left) = (Vec::new(), Vec::new());
    for e in &kit.catalog.models {
        let cached = models::plan(kit, e, py, None).is_ok_and(|p| p.cached);
        if !cached {
            continue;
        }
        if e.remote_code {
            left.push(format!(
                "{0} (urna setup --model {0} --allow-remote-code {0})",
                e.name
            ));
            continue;
        }
        let _ = tx.send(Ev::Note(
            t,
            format!("{}: packages back into the new env", e.name),
        ));
        let consent = Consent {
            download: false,
            remote_code: false,
        };
        match models::install(kit, e, py, uv, consent, None, &mut |p| {
            if let Progress::Log(l) = p {
                let _ = tx.send(Ev::Log(l));
            }
        }) {
            Ok(_) => back.push(e.name.clone()),
            Err(why) => left.push(format!("{}: {why}", e.name)),
        }
    }
    let mut msg = String::new();
    if !back.is_empty() {
        msg.push_str(&format!(
            " · rebuilt, packages back for {}",
            back.join(", ")
        ));
    }
    if !left.is_empty() {
        let _ = tx.send(Ev::Log(format!("not reinstalled: {}", left.join("; "))));
        msg.push_str(&format!(" · not reinstalled: {}", left.join("; ")));
    }
    msg
}

/// The chosen catalog models, one after the other through the install the
/// explorer shares. the catalog is read now, from the payload the earlier
/// step may just have laid down; choosing a model on the plan (or naming it
/// with --model) is the consent to download it, its repo code needs
/// --allow-remote-code. a failed model does not stop the next.
fn install_models(opts: &Opts, tx: &Sender<Ev>) -> Result<String, Failure> {
    let t = Task::Models;
    let home = paths::urna_home().ok_or((codes::BLOCKED, "no data dir".into()))?;
    let kit = Kit::in_home(&home).ok_or((
        codes::MODEL,
        "the installed payload carries no model catalog; install the payload first".into(),
    ))?;
    let entries = kit
        .catalog
        .select(&opts.models)
        .map_err(|e| (codes::BLOCKED, e))?;
    if let Some(why) = super::plan::needs_remote_code(&entries, &opts.allow_remote_code) {
        return Err((codes::BLOCKED, why));
    }
    let py = models::setup_python().map_err(|e| (codes::BLOCKED, e.to_string()))?;
    let (mut done, mut failed) = (Vec::new(), Vec::new());
    for e in &entries {
        let _ = tx.send(Ev::Note(t, format!("{}: planning", e.name)));
        let consent = Consent {
            download: true,
            remote_code: opts.allow_remote_code.contains(&e.name),
        };
        let uv = paths::uv();
        let res = models::install(&kit, e, &py, uv.as_deref(), consent, None, &mut |p| {
            let _ = tx.send(match p {
                Progress::Step(s) => Ev::Note(t, format!("{}: {s}", e.name)),
                Progress::Bytes(n, total, _) => Ev::Bytes(t, n, Some(total)),
                Progress::Log(l) => Ev::Log(l),
            });
        });
        match res {
            Ok(m) => {
                let _ = tx.send(Ev::Log(format!("installed {m}")));
                done.push(m);
            }
            Err(why) => failed.push(format!("{}: {why}", e.name)),
        }
    }
    if failed.is_empty() {
        Ok(done.join(" · "))
    } else {
        Err((codes::MODEL, failed.join("\n")))
    }
}

fn verify(tx: &Sender<Ev>) -> Result<String, Failure> {
    let _ = tx.send(Ev::Note(Task::Verify, "running the doctor checks".into()));
    // the payload setup manages, not whatever a checkout nearby resolves to.
    let rows = match paths::urna_home() {
        Some(home) => health::collect_installed(&home),
        None => health::collect(),
    };
    let code = health::exit_code(&rows);
    let first_fail = rows
        .iter()
        .find(|r| matches!(r.level, health::Level::Fail(_)))
        .map(|r| format!("{}: {}", r.what, r.detail));
    let _ = tx.send(Ev::Health(rows));
    match first_fail {
        None => Ok("every check passes, offline".into()),
        Some(msg) => Err((code, msg)),
    }
}

/// The sample corpus the installed payload's catalogue names, read now:
/// the payload step may just have laid it down.
fn sample(scan: &Scan, tx: &Sender<Ev>) -> Result<String, Failure> {
    let t = Task::Sample;
    let fail = |why: String| (codes::SAMPLE, why);
    let curl = scan
        .curl
        .as_deref()
        .ok_or_else(|| fail("curl is not on PATH".into()))?;
    let home = paths::urna_home().ok_or_else(|| fail("no data dir".into()))?;
    let kit = Kit::in_home(&home)
        .ok_or_else(|| fail("no payload is installed, so no catalogue names a sample".into()))?;
    let s = kit
        .catalog
        .sample
        .ok_or_else(|| fail("the installed payload names no sample corpus".into()))?;
    if let Some(p) = super::sample::present(&home, &s) {
        return Ok(format!(
            "{} {} already here · {}",
            s.repo,
            s.release,
            tilde(&p)
        ));
    }
    let _ = tx.send(Ev::Log(format!("source {}", s.url)));
    let _ = tx.send(Ev::Note(
        t,
        format!("downloading {}", super::sample::size(&s)),
    ));
    let p = super::sample::install(curl, &s, &home, |n| {
        let _ = tx.send(Ev::Bytes(t, n, Some(s.size)));
    })
    .map_err(|e| fail(format!("{e:#}")))?;
    let _ = tx.send(Ev::Log(format!(
        "sha256 ok {} · urna validate ok",
        s.sha256
    )));
    Ok(format!(
        "{} {} · {} · sha256 {}… · {}",
        s.repo,
        s.release,
        super::sample::size(&s),
        &s.sha256[..12],
        tilde(&p)
    ))
}

#[cfg(test)]
mod tests {
    use super::sweep;

    #[test]
    fn sweep_removes_only_interrupted_leftovers() {
        let d = std::env::temp_dir().join(format!("urna_sweep_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join(".urna-setup-42/urna")).unwrap();
        std::fs::create_dir_all(d.join("urna/python")).unwrap();
        std::fs::write(d.join(".urna-payload-42.tar.gz"), b"partial").unwrap();
        std::fs::write(d.join("keep.txt"), b"x").unwrap();
        sweep(&d);
        assert!(!d.join(".urna-setup-42").exists());
        assert!(!d.join(".urna-payload-42.tar.gz").exists());
        assert!(d.join("urna/python").is_dir() && d.join("keep.txt").is_file());
        std::fs::remove_dir_all(&d).unwrap();
    }
}
