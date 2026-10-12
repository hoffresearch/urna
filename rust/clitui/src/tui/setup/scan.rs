//! What this machine already has, probed once before the plan is shown:
//! how urna got here (the channel), where its data lives, whether the
//! embedder payload and a python with numpy + tokenizers are in place, and
//! which tools the install steps can use (curl, uv, a base python).

use std::path::{Path, PathBuf};
use std::process::Command;

use super::models::Kit;
use crate::cmd::{buildid, embed_gate, paths, pyenv};
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
    /// the crate version: the release a release build fetches.
    pub version: &'static str,
    /// what the binary reports (`cmd::buildid::NAME`): the version for a
    /// release, `<version>+dev.<sha>` for a development build.
    pub name: &'static str,
    /// the checkout a development build installs its payload from; `None`
    /// for a release, and when `URNA_RELEASE_BASE` names where to fetch one.
    pub tree: Option<PathBuf>,
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
    /// the python minor that venv runs, `(3, 12)`.
    pub venv_version: Option<(u32, u32)>,
    /// a python able to create a venv (`python3 -m venv`), inside the
    /// range the installed payload's catalogue names when it names one:
    /// the one pip would build from, resolved the way the worker does.
    pub base_python: Option<String>,
    pub uv: Option<PathBuf>,
    pub curl: Option<PathBuf>,
    pub simd: &'static str,
    /// the model catalog and installer of the payload in `home`.
    pub kit: Option<super::models::Kit>,
    /// the sample corpus in `home`, when it holds the bytes the kit's
    /// catalogue names.
    pub sample: Option<PathBuf>,
}

pub use crate::cmd::paths::which;

fn output(cmd: &mut Command) -> Option<String> {
    let o = cmd.output().ok().filter(|o| o.status.success())?;
    let mut s = String::from_utf8_lossy(&o.stdout).trim().to_string();
    if s.is_empty() {
        s = String::from_utf8_lossy(&o.stderr).trim().to_string();
    }
    Some(s)
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
        let python = pyenv::may_run(&interp)
            .then(|| output(Command::new(&interp).arg("--version")))
            .flatten()
            .map(|v| (interp.clone(), v));
        let deps = python.is_some()
            && output(Command::new(&interp).args(["-c", "import numpy, tokenizers"])).is_some();
        let home = paths::urna_home();
        let kit = home.as_deref().and_then(super::models::Kit::in_home);
        let sample = match (&home, kit.as_ref().and_then(|k| k.catalog.sample.as_ref())) {
            (Some(h), Some(s)) => super::sample::present(h, s),
            _ => None,
        };
        let base_python = super::venv::find_python(kit.as_ref().and_then(Kit::range));
        Scan {
            version: env!("CARGO_PKG_VERSION"),
            name: buildid::NAME,
            tree: buildid::dev_tree()
                .filter(|_| std::env::var_os("URNA_RELEASE_BASE").is_none_or(|b| b.is_empty()))
                .map(Path::to_path_buf),
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
            venv_version: paths::urna_home()
                .map(|h| paths::venv_python(&h.join("venv")))
                .filter(|p| p.is_file())
                .and_then(|p| super::venv::version_of(&p.to_string_lossy())),
            base_python,
            uv: paths::uv(),
            curl: which("curl"),
            simd: urna_engine::simd::detect_backend().name(),
            kit,
            sample,
        }
    }

    /// The payload stamp this binary wants: its own name when it installs
    /// from its checkout, else the release of its version.
    pub fn wants(&self) -> &str {
        if self.tree.is_some() {
            self.name
        } else {
            self.version
        }
    }

    /// The python range the installed payload's catalogue names, when it
    /// does; before a payload is installed there is none.
    pub fn range(&self) -> Option<super::venv::Range> {
        self.kit.as_ref().and_then(Kit::range)
    }

    /// The rows the scan screen reveals, in order: label, value, badge.
    pub fn facts(&self) -> Vec<(&'static str, String, Badge)> {
        let tilde = |p: &Path| tilde(p);
        let mut v = vec![
            (
                "urna",
                format!("{} · {}", self.name, self.target),
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
            (Some(p), Some(pl)) if pl.version.as_deref() != Some(self.wants()) => (
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
        let tool = match (&self.uv, &self.curl, &self.base_python) {
            (Some(uv), _, _) => (format!("uv · {}", tilde(uv)), Badge::Ok),
            (None, Some(_), _) => ("uv · setup downloads it".into(), Badge::Ok),
            (None, None, Some(py)) => (format!("pip · {}", tilde(Path::new(py))), Badge::Ok),
            (None, None, None) => (
                match self.range() {
                    Some(r) => format!("no uv, no curl to fetch it, no {r}"),
                    None => "no uv, no curl to fetch it, no python3".into(),
                },
                Badge::Fail,
            ),
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
    fn probe_reports_the_running_binary() {
        let s = Scan::probe();
        assert_eq!(s.version, env!("CARGO_PKG_VERSION"));
        assert_eq!(s.facts()[0].0, "urna");
        assert!(s.facts().len() >= 8);
    }
}
