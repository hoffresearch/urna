//! The managed python env: `<data root>/urna/venv` with numpy and
//! tokenizers, the two packages the potion embedder imports. uv builds it
//! (the one on PATH, else the one setup downloads), on the python range the
//! catalogue names, the minors torch has wheels for: uv takes a matching
//! interpreter on the machine and fetches one when there is none. without
//! uv, the stock `python3 -m venv` + pip, from an interpreter inside that
//! range. every line either tool prints is streamed to `log`, so the
//! installer shows real output.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};

use crate::cmd::paths;

/// the same floors as `pkgs/wheel/pyproject.toml` (`urna[embed]`).
pub const DEPS: [&str; 2] = ["numpy>=1.26", "tokenizers>=0.20"];

#[derive(Clone, Debug)]
pub enum Tool {
    Uv(PathBuf),
    Pip(String),
}

impl Tool {
    pub fn name(&self) -> &'static str {
        match self {
            Tool::Uv(_) => "uv",
            Tool::Pip(_) => "pip",
        }
    }
}

/// A `>=3.A,<3.B` range of cpython minors: `lo` included, `hi` excluded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Range {
    pub lo: u32,
    pub hi: u32,
}

impl Range {
    /// `>=3.10,<3.15`, the only shape the catalogue writes.
    pub fn parse(spec: &str) -> Option<Range> {
        let (a, b) = spec.split_once(',')?;
        let minor = |s: &str, op: &str| -> Option<u32> {
            s.trim()
                .strip_prefix(op)?
                .trim()
                .strip_prefix("3.")?
                .parse()
                .ok()
        };
        let r = Range {
            lo: minor(a, ">=")?,
            hi: minor(b, "<")?,
        };
        (r.lo < r.hi).then_some(r)
    }

    pub fn holds(&self, (major, minor): (u32, u32)) -> bool {
        major == 3 && (self.lo..self.hi).contains(&minor)
    }

    /// what uv's `--python` takes.
    pub fn spec(&self) -> String {
        format!(">=3.{},<3.{}", self.lo, self.hi)
    }
}

impl std::fmt::Display for Range {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "python 3.{} to 3.{}", self.lo, self.hi - 1)
    }
}

/// `Python 3.12.4` to `(3, 12)`.
pub fn parse_version(text: &str) -> Option<(u32, u32)> {
    let v = text.trim().strip_prefix("Python ")?;
    let mut it = v.split('.');
    let major = it.next()?.parse().ok()?;
    let minor = it
        .next()?
        .trim_end_matches(|c: char| !c.is_ascii_digit())
        .parse()
        .ok()?;
    Some((major, minor))
}

/// `py --version`, never run on macOS's stub (`pyenv::may_run`).
pub fn version_of(py: &str) -> Option<(u32, u32)> {
    if !crate::cmd::pyenv::may_run(py) {
        return None;
    }
    let out = Command::new(py).arg("--version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text =
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    parse_version(&text)
}

/// For pip, when there is no uv: the newest interpreter on PATH inside
/// `range` that can create a venv (`python3.14`, …, then `python3`,
/// `python`).
pub fn find_python(range: Range) -> Option<String> {
    let mut names: Vec<String> = (range.lo..range.hi)
        .rev()
        .map(|m| format!("python3.{m}"))
        .collect();
    names.extend(["python3".to_string(), "python".to_string()]);
    names.iter().find_map(|n| {
        let p = crate::cmd::paths::which(n)?.to_string_lossy().into_owned();
        let ok = version_of(&p).is_some_and(|v| range.holds(v))
            && Command::new(&p)
                .args(["-c", "import venv, ensurepip"])
                .output()
                .is_ok_and(|o| o.status.success());
        ok.then_some(p)
    })
}

/// Runs `cmd`, forwarding stdout and stderr line by line to `log`; a
/// non-zero exit is an error naming the command.
pub fn stream(cmd: Command, log: &mut dyn FnMut(String)) -> Result<()> {
    let shown = format!("{cmd:?}").replace('"', "");
    let status = stream_status(cmd, log)?;
    if !status.success() {
        bail!("{shown} exited with {status}");
    }
    Ok(())
}

/// Like `stream`, but hands back the exit status for the caller to read
/// (the model fetch names its refusals by exit code). a child that prints
/// progress with carriage returns (pip, uv) has each `\r` segment logged
/// as its own line, so a log never shows a half-redrawn bar.
pub fn stream_status(
    mut cmd: Command,
    log: &mut dyn FnMut(String),
) -> Result<std::process::ExitStatus> {
    let shown = format!("{cmd:?}").replace('"', "");
    log(format!("$ {shown}"));
    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawn {shown}"))?;
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    let err = child.stderr.take().map(|e| {
        let tx = tx.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(e).lines().map_while(Result::ok) {
                send_segments(&tx, &line);
            }
        })
    });
    if let Some(out) = child.stdout.take() {
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            send_segments(&tx, &line);
            while let Ok(l) = rx.try_recv() {
                log(l);
            }
        }
    }
    if let Some(h) = err {
        let _ = h.join();
    }
    drop(tx);
    for l in rx {
        log(l);
    }
    Ok(child.wait()?)
}

/// One line as printed, split on the carriage returns a progress bar
/// redraws with; empty segments are dropped.
fn send_segments(tx: &std::sync::mpsc::Sender<String>, line: &str) {
    for seg in line
        .split('\r')
        .map(str::trim_end)
        .filter(|s| !s.is_empty())
    {
        let _ = tx.send(seg.to_string());
    }
}

/// `/usr/bin` left out of a PATH.
fn without_usr_bin(path: &std::ffi::OsStr) -> Option<std::ffi::OsString> {
    let kept: Vec<PathBuf> = std::env::split_paths(path)
        .filter(|p| p != Path::new("/usr/bin"))
        .collect();
    std::env::join_paths(kept).ok()
}

/// A uv command. on a Mac without the Command Line Tools, uv runs with
/// `/usr/bin` off its PATH: there `python3` and `install_name_tool` are
/// Apple's shim, which opens the installer dialog. uv's interpreter
/// discovery runs every `python3` it finds, and after it fetches a python
/// it runs `install_name_tool` to patch the dylib (uv-python-managed,
/// `macos_dylib.rs`); without one it warns and goes on, and the patch only
/// matters to native extensions built against libpython, which setup
/// never builds.
pub fn uv_command(uv: &Path) -> Command {
    let mut c = Command::new(uv);
    if crate::cmd::pyenv::shims_open_dialogs()
        && let Some(path) = std::env::var_os("PATH").and_then(|p| without_usr_bin(&p))
    {
        c.env("PATH", path);
    }
    c
}

/// What a model's packages are resolved against: the payload's lock (exact
/// versions, torch's cpu build) and the index that build is on. both empty
/// for a payload from before the lock.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Pins {
    pub lock: Option<PathBuf>,
    pub torch_index: Option<String>,
}

/// The command that adds `pkgs` to the env whose interpreter is `py`: uv
/// when it is on PATH, else that interpreter's own pip. with a lock, only
/// `pkgs` and what they need install, at the lock's versions (`-c`), and
/// torch comes as its cpu build: uv finds that index itself with
/// `--torch-backend cpu`, pip is pointed at it.
pub fn add_command(uv: Option<&Path>, py: &Path, pkgs: &[String], pins: &Pins) -> Command {
    match uv {
        Some(uv) => {
            let mut c = uv_command(uv);
            c.args(["pip", "install", "--python"]).arg(py);
            if let Some(lock) = &pins.lock {
                c.args(["--torch-backend", "cpu", "-c"]).arg(lock);
            }
            c.args(pkgs);
            c
        }
        None => {
            let mut c = Command::new(py);
            c.args(["-m", "pip", "install", "--disable-pip-version-check"])
                .args(["--progress-bar", "off"]);
            if let Some(lock) = &pins.lock {
                c.arg("-c").arg(lock);
                if let Some(index) = &pins.torch_index {
                    c.args(["--extra-index-url", index]);
                }
            }
            c.args(pkgs);
            c
        }
    }
}

/// What `build` did with the env already at its dir.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Built {
    /// there was none.
    Fresh,
    /// it was inside the range and was kept.
    Kept,
    /// its python was outside the range: deleted and created again, so the
    /// packages it held are gone.
    Rebuilt,
}

/// Creates (or refreshes) the venv at `dir` on `range` (when the catalogue
/// names one) and installs the deps; returns its interpreter once
/// `import numpy, tokenizers` succeeds there. an env whose python is
/// outside the range is deleted first, never reused.
pub fn build(
    tool: &Tool,
    dir: &Path,
    range: Option<Range>,
    log: &mut dyn FnMut(String),
) -> Result<(PathBuf, Built)> {
    if let Some(parent) = dir.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let py = paths::venv_python(dir);
    let mut built = if py.is_file() {
        Built::Kept
    } else {
        Built::Fresh
    };
    if let (Some(r), Some(v)) = (range, version_of(&py.to_string_lossy()))
        && !r.holds(v)
    {
        log(format!(
            "the env runs python {}.{}, outside {r}: deleting it to build it again",
            v.0, v.1
        ));
        std::fs::remove_dir_all(dir).with_context(|| format!("remove {}", dir.display()))?;
        built = Built::Rebuilt;
    }
    match tool {
        Tool::Uv(uv) => {
            let mut venv = uv_command(uv);
            venv.args(["venv", "--allow-existing", "--quiet"]);
            if let Some(r) = range {
                venv.arg("--python").arg(r.spec());
            }
            venv.arg(dir);
            stream(venv, log)?;
            let mut pip = uv_command(uv);
            pip.args(["pip", "install", "--python"]).arg(&py).args(DEPS);
            stream(pip, log)?;
        }
        Tool::Pip(base) => {
            if !py.is_file() {
                let mut venv = Command::new(base);
                venv.args(["-m", "venv"]).arg(dir);
                stream(venv, log)?;
            }
            let mut pip = Command::new(&py);
            pip.args([
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--progress-bar",
                "off",
            ])
            .args(DEPS);
            stream(pip, log)?;
        }
    }
    let mut check = Command::new(&py);
    check.args(["-c", "import numpy, tokenizers; print('numpy', numpy.__version__, '· tokenizers', tokenizers.__version__)"]);
    stream(check, log).context("the new env cannot import numpy + tokenizers")?;
    Ok((py, built))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usr_bin_is_left_out_of_uvs_path_and_nothing_else() {
        if !cfg!(unix) {
            return;
        }
        let got = without_usr_bin(std::ffi::OsStr::new(
            "/opt/homebrew/bin:/usr/bin:/bin:/usr/bin/extra",
        ))
        .unwrap();
        assert_eq!(got, "/opt/homebrew/bin:/bin:/usr/bin/extra");
    }

    #[test]
    fn the_catalogue_range_reads_and_holds_minors() {
        let r = Range::parse(">=3.10,<3.15").unwrap();
        assert_eq!(r, Range { lo: 10, hi: 15 });
        assert_eq!(r.spec(), ">=3.10,<3.15");
        assert_eq!(r.to_string(), "python 3.10 to 3.14");
        assert!(r.holds((3, 10)) && r.holds((3, 14)));
        assert!(!r.holds((3, 15)) && !r.holds((3, 9)) && !r.holds((2, 12)));
        assert_eq!(
            Range::parse(">= 3.11 , < 3.16"),
            Some(Range { lo: 11, hi: 16 })
        );
        assert!(Range::parse(">=3.15,<3.10").is_none());
        assert!(Range::parse("3.12").is_none());
        assert!(Range::parse(">=4.0,<4.1").is_none());
    }

    #[test]
    fn python_versions_are_read_off_the_banner() {
        assert_eq!(parse_version("Python 3.15.0\n"), Some((3, 15)));
        assert_eq!(parse_version("Python 3.14.0rc2"), Some((3, 14)));
        assert_eq!(parse_version("Python 3.13t"), Some((3, 13)));
        assert_eq!(parse_version("pypy"), None);
    }

    #[cfg(unix)]
    #[test]
    fn an_env_outside_the_range_is_rebuilt() {
        let d = std::env::temp_dir().join(format!("urna_venv_range_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let py = paths::venv_python(&d);
        std::fs::create_dir_all(py.parent().unwrap()).unwrap();
        std::fs::write(&py, "#!/bin/sh\necho Python 3.15.0\n").unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&py, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::write(d.join("marker"), b"old package").unwrap();
        let mut lines = Vec::new();
        // a tool that cannot run: build fails after the delete, which is
        // what this test reads.
        let tool = Tool::Uv(PathBuf::from("/urna/no/such/uv"));
        let r = build(&tool, &d, Range::parse(">=3.10,<3.15"), &mut |l| {
            lines.push(l)
        });
        assert!(r.is_err());
        assert!(!d.join("marker").exists(), "the old env must be gone");
        assert!(
            lines[0].contains("python 3.15, outside python 3.10 to 3.14"),
            "{lines:?}"
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn stream_forwards_both_pipes_and_reports_failure() {
        if !cfg!(unix) {
            return;
        }
        let mut lines = Vec::new();
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "echo out; echo err 1>&2"]);
        stream(cmd, &mut |l| lines.push(l)).unwrap();
        assert!(lines.iter().any(|l| l == "out"));
        assert!(lines.iter().any(|l| l == "err"));
        let mut fail = Command::new("sh");
        fail.args(["-c", "exit 3"]);
        assert!(stream(fail, &mut |_| {}).is_err());
        let mut code = Command::new("sh");
        code.args(["-c", "exit 5"]);
        let status = stream_status(code, &mut |_| {}).unwrap();
        assert_eq!(status.code(), Some(5));
    }

    #[test]
    fn a_redrawn_progress_bar_logs_each_state_once() {
        if !cfg!(unix) {
            return;
        }
        let mut lines = Vec::new();
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "printf '10%%\\r50%%\\r100%%\\ndone\\n'"]);
        stream(cmd, &mut |l| lines.push(l)).unwrap();
        assert_eq!(lines[1..], ["10%", "50%", "100%", "done"]);
    }

    #[test]
    fn packages_go_through_uv_or_the_envs_own_pip() {
        let py = Path::new("/h/urna/venv/bin/python");
        let pkgs = vec!["sentence-transformers>=3".to_string()];
        let uv = format!(
            "{:?}",
            add_command(Some(Path::new("/u/uv")), py, &pkgs, &Pins::default())
        );
        assert!(
            uv.contains("--python") && uv.contains("/h/urna/venv/bin/python"),
            "{uv}"
        );
        let pip = format!("{:?}", add_command(None, py, &pkgs, &Pins::default()));
        assert!(
            pip.starts_with("\"/h/urna/venv/bin/python\"") && pip.contains("pip"),
            "{pip}"
        );
    }

}
