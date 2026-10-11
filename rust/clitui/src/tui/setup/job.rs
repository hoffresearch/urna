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

/// A step whose failure leaves urna ready: the models are optional, the
/// potion embedder works without them. the run still exits non-zero
/// (`codes::MODEL`), so a script sees the model it asked for is missing.
pub fn optional(task: Task) -> bool {
    task == Task::Models
}

/// setup's own exit codes, above doctor's 2..=6 so the two never collide.
pub mod codes {
    pub const DOWNLOAD: i32 = 10;
    pub const CHECKSUM: i32 = 11;
    pub const UNPACK: i32 = 12;
    pub const PYTHON: i32 = 13;
    pub const BLOCKED: i32 = 14;
    pub const MODEL: i32 = 15;
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
        for task in tasks {
            let _ = tx.send(Ev::Start(task));
            let res = match task {
                Task::Payload => payload(&scan, &version, &tx),
                Task::Python => python(&scan, &tx),
                Task::Models => install_models(&opts, &tx),
                Task::Verify => verify(&tx),
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
/// else pip from an interpreter inside the range. the catalogue is read
/// from the payload the step before may just have laid down.
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
    let mut why = match (release, scan.curl.as_deref()) {
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
    let base = match range {
        Some(r) => venv::find_python(r),
        None => scan.base_python.clone(),
    };
    if let Some(py) = base {
        let _ = tx.send(Ev::Log(format!(
            "no uv ({why}); pip from {}",
            tilde(std::path::Path::new(&py))
        )));
        return Ok(Tool::Pip(py));
    }
    if let Some(r) = range {
        why = format!("{why}, and no {r} is on PATH");
    }
    Err((
        codes::BLOCKED,
        format!(
            "{why}: install uv (https://docs.astral.sh/uv/) or a python inside the range, then run `urna setup` again"
        ),
    ))
}

fn python(scan: &Scan, tx: &Sender<Ev>) -> Result<String, Failure> {
    let home = paths::urna_home().ok_or((codes::BLOCKED, "no data dir".into()))?;
    let kit = Kit::in_home(&home);
    let range = kit
        .as_ref()
        .and_then(|k| k.catalog.python.as_ref())
        .and_then(|p| Range::parse(&p.requires));
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
    let rows = health::collect();
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
