//! The managed python env: `<data root>/urna/venv` with numpy and
//! tokenizers, the two packages the potion embedder imports. uv builds it
//! when it is on PATH (fast, and it can fetch a python when the machine has
//! none); otherwise the stock `python3 -m venv` + pip. every line either
//! tool prints is streamed to `log`, so the installer shows real output.

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
    pub fn pick(uv: Option<&Path>, base_python: Option<&str>) -> Option<Tool> {
        uv.map(|p| Tool::Uv(p.to_path_buf()))
            .or_else(|| base_python.map(|p| Tool::Pip(p.to_string())))
    }

    pub fn name(&self) -> &'static str {
        match self {
            Tool::Uv(_) => "uv",
            Tool::Pip(_) => "pip",
        }
    }
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

/// The command that adds `pkgs` to the env whose interpreter is `py`: uv
/// when it is on PATH, else that interpreter's own pip.
pub fn add_command(uv: Option<&Path>, py: &Path, pkgs: &[String]) -> Command {
    match uv {
        Some(uv) => {
            let mut c = Command::new(uv);
            c.args(["pip", "install", "--python"]).arg(py).args(pkgs);
            c
        }
        None => {
            let mut c = Command::new(py);
            c.args(["-m", "pip", "install", "--disable-pip-version-check"])
                .args(["--progress-bar", "off"])
                .args(pkgs);
            c
        }
    }
}

/// Creates (or refreshes) the venv at `dir` and installs the deps; returns
/// its interpreter once `import numpy, tokenizers` succeeds there.
pub fn build(tool: &Tool, dir: &Path, log: &mut dyn FnMut(String)) -> Result<PathBuf> {
    if let Some(parent) = dir.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let py = paths::venv_python(dir);
    match tool {
        Tool::Uv(uv) => {
            let mut venv = Command::new(uv);
            venv.args(["venv", "--allow-existing", "--quiet"]).arg(dir);
            stream(venv, log)?;
            let mut pip = Command::new(uv);
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
    Ok(py)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uv_wins_over_pip_and_nothing_means_none() {
        let t = Tool::pick(Some(Path::new("/x/uv")), Some("python3")).unwrap();
        assert_eq!(t.name(), "uv");
        assert_eq!(Tool::pick(None, Some("python3")).unwrap().name(), "pip");
        assert!(Tool::pick(None, None).is_none());
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
        let uv = format!("{:?}", add_command(Some(Path::new("/u/uv")), py, &pkgs));
        assert!(
            uv.contains("--python") && uv.contains("/h/urna/venv/bin/python"),
            "{uv}"
        );
        let pip = format!("{:?}", add_command(None, py, &pkgs));
        assert!(
            pip.starts_with("\"/h/urna/venv/bin/python\"") && pip.contains("pip"),
            "{pip}"
        );
    }
}
