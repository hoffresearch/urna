//! What this machine already has, probed once before the plan is shown:
//! how urna got here (the channel), where its data lives, whether the
//! embedder payload and a python with numpy + tokenizers are in place, and
//! which tools the install steps can use (curl, uv, a base python).

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::cmd::{embed_gate, paths, pyenv};
use crate::tui::hud::Badge;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    Homebrew,
    Npm,
    Cargo,
    OneLiner,
    Dev,
    Other,
}

impl Channel {
    pub fn name(self) -> &'static str {
        match self {
            Channel::Homebrew => "homebrew",
            Channel::Npm => "npm",
            Channel::Cargo => "cargo",
            Channel::OneLiner => "install script",
            Channel::Dev => "dev build",
            Channel::Other => "manual",
        }
    }

    /// How the binary was installed, read off its own path.
    pub fn of(exe: &Path) -> Channel {
        let s = exe.to_string_lossy().replace('\\', "/");
        if s.contains("/Cellar/") || s.contains("/homebrew/") || s.contains("/linuxbrew/") {
            Channel::Homebrew
        } else if s.contains("/node_modules/") {
            Channel::Npm
        } else if s.contains("/.cargo/bin/") {
            Channel::Cargo
        } else if s.contains("/target/debug/") || s.contains("/target/release/") {
            Channel::Dev
        } else if s.contains("/.local/bin/") {
            Channel::OneLiner
        } else {
            Channel::Other
        }
    }
}

/// The payload setup manages (`<home>/python`), the release it came from,
/// read from `<home>/VERSION` (`None` for a payload laid down before the
/// stamp existed, 0.5.1 and older), and the required files it lacks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Payload {
    pub version: Option<String>,
    /// required files (`cmd::payload::REQUIRED`) not on disk, the stamp aside.
    pub missing: Vec<String>,
}

impl Payload {
    pub fn read(home: &Path) -> Option<Payload> {
        if !home.join(crate::cmd::payload::DIR).is_dir() {
            return None;
        }
        let version = std::fs::read_to_string(home.join("VERSION"))
            .ok()
            .map(|v| v.trim().trim_start_matches('v').to_string())
            .filter(|v| !v.is_empty());
        let missing = crate::cmd::payload::missing(home)
            .into_iter()
            .filter(|rel| *rel != "VERSION")
            .map(String::from)
            .collect();
        Some(Payload { version, missing })
    }

    /// `v0.5.2`, or what an unstamped payload is.
    pub fn label(&self) -> String {
        match &self.version {
            Some(v) => format!("v{v}"),
            None => "unstamped (0.5.1 or older)".into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Scan {
    pub version: &'static str,
    pub target: String,
    pub exe: PathBuf,
    pub channel: Channel,
    /// `<data root>/urna`, where setup writes.
    pub home: Option<PathBuf>,
    /// the embedder script, when one resolves.
    pub embedder: Option<PathBuf>,
    /// the payload in `home`, when setup laid one down there.
    pub payload: Option<Payload>,
    /// interpreter + `--version` of the one the embedder would run under.
    pub python: Option<(String, String)>,
    /// `URNA_PYTHON` when set: it wins over any venv setup builds, so the
    /// plan has to say so. read here, once, so `plan` stays a pure function
    /// of the scan (its tests run under whatever env the gate exports).
    pub pinned_python: Option<String>,
    pub deps: bool,
    /// the venv setup manages exists in `home` (model packages go only there).
    pub venv: bool,
    /// a python able to create a venv (`python3 -m venv`), when present.
    pub base_python: Option<String>,
    pub uv: Option<PathBuf>,
    pub curl: Option<PathBuf>,
    pub simd: &'static str,
    /// the model catalog and installer of the payload in `home`.
    pub kit: Option<super::models::Kit>,
}

/// First match for `name` on PATH (with the PATHEXT suffixes on windows).
pub fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    let exts: Vec<String> = if cfg!(windows) {
        std::env::var("PATHEXT")
            .unwrap_or_else(|_| ".EXE;.CMD;.BAT".into())
            .split(';')
            .map(|e| e.to_ascii_lowercase())
            .chain(std::iter::once(String::new()))
            .collect()
    } else {
        vec![String::new()]
    };
    std::env::split_paths(&path).find_map(|dir| {
        exts.iter()
            .map(|e| dir.join(format!("{name}{e}")))
            .find(|p| p.is_file())
    })
}

fn output(cmd: &mut Command) -> Option<String> {
    let o = cmd.output().ok().filter(|o| o.status.success())?;
    let mut s = String::from_utf8_lossy(&o.stdout).trim().to_string();
    if s.is_empty() {
        s = String::from_utf8_lossy(&o.stderr).trim().to_string();
    }
    Some(s)
}

fn base_python() -> Option<String> {
    let names: &[&str] = if cfg!(windows) {
        &["python", "py"]
    } else {
        &["python3", "python"]
    };
    names.iter().find_map(|n| {
        let p = which(n)?;
        let s = p.to_string_lossy().into_owned();
        output(Command::new(&s).args(["-c", "import venv, ensurepip"])).map(|_| s)
    })
}

impl Scan {
    /// Runs every probe. Blocking (it spawns python a few times, well
    /// under a second on a warm machine); the ui calls it off-thread.
    pub fn probe() -> Scan {
        pyenv::set_quiet(true);
        let exe = std::env::current_exe()
            .ok()
            .and_then(|p| p.canonicalize().ok())
            .unwrap_or_default();
        let embedder = Some(embed_gate::default_potion_embedder_path()).filter(|p| p.is_file());
        let interp = pyenv::resolve_interpreter();
        let python = output(Command::new(&interp).arg("--version")).map(|v| (interp.clone(), v));
        let deps = python.is_some()
            && output(Command::new(&interp).args(["-c", "import numpy, tokenizers"])).is_some();
        Scan {
            version: env!("CARGO_PKG_VERSION"),
            target: format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
            channel: Channel::of(&exe),
            exe,
            payload: paths::urna_home().and_then(|h| Payload::read(&h)),
            home: paths::urna_home(),
            embedder,
            python,
            pinned_python: std::env::var("URNA_PYTHON").ok().filter(|s| !s.is_empty()),
            deps,
            venv: paths::urna_home().is_some_and(|h| paths::venv_python(&h.join("venv")).is_file()),
            base_python: base_python(),
            uv: which("uv"),
            curl: which("curl"),
            simd: urna_engine::simd::detect_backend().name(),
            kit: paths::urna_home().and_then(|h| super::models::Kit::in_home(&h)),
        }
    }

    /// The rows the scan screen reveals, in order: label, value, badge.
    pub fn facts(&self) -> Vec<(&'static str, String, Badge)> {
        let tilde = |p: &Path| tilde(p);
        let mut v = vec![
            (
                "urna",
                format!("{} · {}", self.version, self.target),
                Badge::Ok,
            ),
            (
                "channel",
                format!("{} · {}", self.channel.name(), tilde(&self.exe)),
                Badge::Ok,
            ),
            ("simd", self.simd.to_string(), Badge::Ok),
        ];
        v.push(match &self.home {
            Some(h) => ("data dir", tilde(h), Badge::Ok),
            None => ("data dir", "no HOME, set URNA_DATA_DIR".into(), Badge::Fail),
        });
        v.push(match (&self.embedder, &self.payload) {
            (Some(p), Some(pl)) if !pl.missing.is_empty() => (
                "embedder",
                format!(
                    "{} · missing {}, setup repairs it",
                    tilde(p),
                    pl.missing.join(" ")
                ),
                Badge::Warn,
            ),
            (Some(p), Some(pl)) if pl.version.as_deref() != Some(self.version) => (
                "embedder",
                format!("{} · {}, setup replaces it", tilde(p), pl.label()),
                Badge::Warn,
            ),
            (Some(p), Some(pl)) => (
                "embedder",
                format!("{} · {}", tilde(p), pl.label()),
                Badge::Ok,
            ),
            (Some(p), None) => ("embedder", tilde(p), Badge::Ok),
            (None, _) => (
                "embedder",
                "missing, setup downloads it".into(),
                Badge::Warn,
            ),
        });
        v.push(match &self.python {
            Some((i, ver)) => (
                "python",
                format!("{ver} · {}", tilde(Path::new(i))),
                Badge::Ok,
            ),
            None => ("python", "not found".into(), Badge::Warn),
        });
        v.push(if self.deps {
            ("numpy, tokenizers", "present".into(), Badge::Ok)
        } else {
            (
                "numpy, tokenizers",
                "missing, setup builds a venv".into(),
                Badge::Warn,
            )
        });
        let tool = match (&self.uv, &self.base_python) {
            (Some(uv), _) => (format!("uv · {}", tilde(uv)), Badge::Ok),
            (None, Some(py)) => (format!("pip · {}", tilde(Path::new(py))), Badge::Ok),
            (None, None) => ("no uv, no python3".into(), Badge::Fail),
        };
        v.push(("env tool", tool.0, tool.1));
        v.push(match &self.curl {
            Some(c) => ("curl", tilde(c), Badge::Ok),
            None => ("curl", "missing, downloads disabled".into(), Badge::Fail),
        });
        v
    }
}

/// `$HOME/x` as `~/x`, for display only.
pub fn tilde(p: &Path) -> String {
    crate::tui::txt::home(&p.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_reads_the_install_path() {
        let c = |s: &str| Channel::of(Path::new(s));
        assert_eq!(
            c("/opt/homebrew/Cellar/urna/0.5.0/bin/urna"),
            Channel::Homebrew
        );
        assert_eq!(
            c("/usr/lib/node_modules/urna/node_modules/.bin_real/urna"),
            Channel::Npm
        );
        assert_eq!(c("/home/u/.cargo/bin/urna"), Channel::Cargo);
        assert_eq!(c("/home/u/.local/bin/urna"), Channel::OneLiner);
        assert_eq!(c("/src/urna/target/release/urna"), Channel::Dev);
        assert_eq!(c("/usr/bin/urna"), Channel::Other);
    }

    #[test]
    fn which_finds_a_shell_and_misses_nonsense() {
        if cfg!(unix) {
            assert!(which("sh").is_some());
        }
        assert!(which("urna-definitely-not-a-binary-xyz").is_none());
    }

    #[test]
    fn probe_reports_the_running_binary() {
        let s = Scan::probe();
        assert_eq!(s.version, env!("CARGO_PKG_VERSION"));
        assert_eq!(s.facts()[0].0, "urna");
        assert!(s.facts().len() >= 8);
    }
}
