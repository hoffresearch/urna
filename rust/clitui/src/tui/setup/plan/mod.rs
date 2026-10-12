//! The plan: which steps run, derived from the scan and the flags. A step
//! that is already satisfied starts unticked (ticking it reinstalls); a
//! step the machine cannot run is blocked with the reason, never hidden.

use super::models::{self, Entry};
use super::scan::{Scan, tilde};
use super::venv;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Task {
    Payload,
    Python,
    Models,
    Verify,
}

impl Task {
    pub fn title(self) -> &'static str {
        match self {
            Task::Payload => "embedder payload",
            Task::Python => "python env",
            Task::Models => "models",
            Task::Verify => "verify",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Opts {
    /// no questions: run the default plan and print plain progress.
    pub yes: bool,
    /// release tag to fetch the payload from (default: this binary's).
    pub version: Option<String>,
    /// reinstall the payload even when one is present.
    pub force: bool,
    pub no_payload: bool,
    pub no_python: bool,
    /// catalog models to install (`--model`, repeatable; `all` is every
    /// offered one). naming one is the consent to download it.
    pub models: Vec<String>,
    /// models whose repo code may run (`--allow-remote-code`), a consent
    /// separate from the download.
    pub allow_remote_code: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Item {
    pub task: Task,
    pub on: bool,
    /// runs no matter what (verify).
    pub locked: bool,
    /// why the step cannot run here.
    pub blocked: Option<String>,
    pub detail: String,
}

impl Item {
    pub fn runs(&self) -> bool {
        self.on && self.blocked.is_none()
    }
}

/// 14 when a step the machine needs is blocked here (it wanted to run and
/// cannot), else 0. the run starts from this code; a later failure only
/// replaces a 0.
pub fn blocked_code(items: &[Item]) -> i32 {
    if items.iter().any(|i| i.on && i.blocked.is_some()) {
        super::job::codes::BLOCKED
    } else {
        0
    }
}

/// The release the payload comes from: `--version` when given, else this
/// binary's own (a leading `v` is accepted either way).
pub fn wanted_version(scan: &Scan, opts: &Opts) -> String {
    opts.version
        .as_deref()
        .unwrap_or(scan.version)
        .trim_start_matches('v')
        .to_string()
}

pub fn plan(scan: &Scan, opts: &Opts) -> Vec<Item> {
    let home = scan
        .home
        .as_deref()
        .map(tilde)
        .unwrap_or_else(|| "?".into());
    let payload = {
        let blocked = if scan.home.is_none() {
            Some("no data dir: set URNA_DATA_DIR".to_string())
        } else if scan.curl.is_none() {
            Some("curl is not on PATH".to_string())
        } else {
            None
        };
        let want = wanted_version(scan, opts);
        // a payload from another release (or from before the stamp) is
        // replaced, so upgrading the binary upgrades the scripts it runs; one
        // missing a required file is repaired the same way.
        let broken = scan.payload.as_ref().filter(|p| !p.missing.is_empty());
        let stale = scan
            .payload
            .as_ref()
            .filter(|p| p.version.as_deref() != Some(want.as_str()));
        // setup manages the payload in its own data dir: one found in
        // another data root (or a checkout) does not count as installed.
        let missing = scan.payload.is_none();
        let fresh = format!(
            "download ~30 MB from the v{want} release, verify sha256, unpack to {home}/python"
        );
        let detail = match (&scan.embedder, stale, missing) {
            _ if broken.is_some() => format!(
                "installed payload is missing {}: download v{want}, verify sha256, replace {home}/python (the venv stays)",
                broken.map(|p| p.missing.join(" ")).unwrap_or_default()
            ),
            (_, Some(p), _) => format!(
                "installed payload is {}, v{want} is wanted: download it, verify sha256, replace {home}/python (the venv stays)",
                p.label()
            ),
            (Some(p), None, true) => format!("{fresh}; {} resolves until then", tilde(p)),
            (Some(p), None, false) => format!("present at {}; tick to reinstall", tilde(p)),
            (None, None, _) => fresh,
        };
        Item {
            task: Task::Payload,
            on: (missing || broken.is_some() || stale.is_some() || opts.force) && !opts.no_payload,
            locked: false,
            blocked,
            detail,
        }
    };
    let python = {
        let tool = env_tool(scan);
        let via = tool.as_ref().ok();
        let pinned = scan.pinned_python.clone();
        // chosen models need the venv setup manages (their packages go
        // nowhere else), even when another python already has the base deps.
        let for_models = !opts.models.is_empty() && !scan.venv && pinned.is_none();
        let range = scan.range();
        // an env on a python outside the range cannot install torch: it is
        // deleted and built again, not reused.
        let stale = match (range, scan.venv_version) {
            (Some(r), Some(v)) if !r.holds(v) && pinned.is_none() => Some((r, v)),
            _ => None,
        };
        let on = range.map(|r| format!(" on {r}")).unwrap_or_default();
        let detail = match (&scan.python, scan.deps, via, stale) {
            (_, _, Some(t), Some((r, v))) => format!(
                "the env at {home}/venv runs python {}.{}, outside {r}: it is deleted and built again via {t}; the models whose weights are cached get their packages back",
                v.0, v.1
            ),
            (_, true, Some(t), _) if for_models => format!(
                "venv at {home}/venv with numpy + tokenizers, via {t}{on}: the chosen models' packages go only there"
            ),
            (_, false, _, _) if pinned.is_some() => format!(
                "URNA_PYTHON={} lacks numpy + tokenizers and wins over any env setup builds; unset it or install the deps there",
                pinned.clone().unwrap_or_default()
            ),
            (Some((i, _)), true, _, _) => format!(
                "{} already imports numpy + tokenizers",
                tilde(std::path::Path::new(i))
            ),
            (_, _, Some(t), _) => {
                format!("venv at {home}/venv with numpy + tokenizers, via {t}{on}")
            }
            _ => match range {
                Some(r) => format!("needs uv (or curl to download it) or {r} with venv"),
                None => "needs uv (or curl to download it) or python3 with venv".into(),
            },
        };
        Item {
            task: Task::Python,
            on: (!scan.deps || for_models || stale.is_some()) && !opts.no_python,
            locked: false,
            blocked: match tool {
                Err(why) => Some(why),
                Ok(_) if pinned.is_some() && !scan.deps => {
                    Some("URNA_PYTHON pins an interpreter without the deps".to_string())
                }
                Ok(_) => None,
            },
            detail,
        }
    };
    let verify = Item {
        task: Task::Verify,
        on: true,
        locked: true,
        blocked: None,
        detail: "urna doctor: interpreter, deps, table, one offline embed".into(),
    };
    vec![payload, python, models_item(scan, opts), verify]
}

/// How the env will be built, as the plan says it: the uv already here,
/// the uv the catalogue names (downloaded; before a payload is installed,
/// the one its catalogue will name), else pip from the base python the
/// scan resolved inside the catalogue's range. with none of those, the
/// block names the fix the way the worker does.
fn env_tool(scan: &Scan) -> Result<String, String> {
    if scan.uv.is_some() {
        return Ok("uv".into());
    }
    let release = scan.kit.as_ref().map(|k| k.catalog.uv.as_ref());
    let why = match (release, scan.curl.is_some()) {
        (Some(Some(r)), true) if r.archives.contains_key(&scan.target) => {
            return Ok(format!("uv {} (downloaded, sha256 checked)", r.version));
        }
        (None, true) => return Ok("uv (downloaded after the payload, sha256 checked)".into()),
        (_, false) => "no uv and no curl to download it",
        (Some(Some(_)), true) => "no uv and the catalogue names none for this platform",
        (Some(None), true) => "no uv and the catalogue names none to download",
    };
    match &scan.base_python {
        Some(_) => Ok("pip".into()),
        None => Err(venv::no_tool(why, scan.range())),
    }
}

/// Re-derives the steps a model choice decides, after the picker changed
/// it: the models step, and the python step (chosen models need the managed
/// venv, even when another python has the base deps). the other steps keep
/// the ticks the user gave them.
pub fn reselect(items: &mut [Item], scan: &Scan, opts: &Opts) {
    let fresh = plan(scan, opts);
    for item in items
        .iter_mut()
        .filter(|i| matches!(i.task, Task::Python | Task::Models))
    {
        if let Some(f) = fresh.iter().find(|f| f.task == item.task) {
            *item = f.clone();
        }
    }
}

/// The models step: what the chosen catalog models will install and fetch,
/// or why they cannot. the catalog comes with the payload, so before one is
/// installed the names are checked when the step runs.
fn models_item(scan: &Scan, opts: &Opts) -> Item {
    let offered = scan.kit.as_ref().map(|k| k.catalog.models.len());
    let mut item = Item {
        task: Task::Models,
        on: !opts.models.is_empty(),
        locked: false,
        blocked: None,
        detail: match offered {
            Some(n) => format!("none chosen; {n} offered (m picks, or --model NAME|all)"),
            None => "none chosen; the catalog comes with the payload (--model NAME|all)".into(),
        },
    };
    if opts.models.is_empty() {
        return item;
    }
    if let Some(p) = scan.pinned_python.as_deref() {
        let ours = scan
            .home
            .as_ref()
            .map(|h| crate::cmd::paths::venv_python(&h.join("venv")));
        if ours.as_deref() != Some(std::path::Path::new(p)) {
            item.blocked = Some(format!(
                "URNA_PYTHON pins {p}: urna installs model packages only into its own venv"
            ));
            return item;
        }
    }
    let Some(kit) = &scan.kit else {
        item.detail = format!("after the payload: {}", opts.models.join(", "));
        return item;
    };
    match kit.catalog.select(&opts.models) {
        Ok(entries) => {
            item.blocked = needs_remote_code(&entries, &opts.allow_remote_code);
            item.detail = entries.iter().map(describe).collect::<Vec<_>>().join("; ");
        }
        Err(why) => item.blocked = Some(why),
    }
    item
}

/// The first chosen model whose repo code was not allowed, as a blocker.
pub fn needs_remote_code(entries: &[Entry], allowed: &[String]) -> Option<String> {
    entries
        .iter()
        .find(|e| e.remote_code && !allowed.contains(&e.name))
        .map(|e| {
            format!(
                "{} runs code from its model repo: allow it separately (--allow-remote-code {0}, r in the picker)",
                e.name
            )
        })
}

/// `name: packages + N MB from huggingface.co (repo@rev)`.
pub fn describe(e: &Entry) -> String {
    let pkgs = if e.packages.is_empty() {
        String::new()
    } else {
        format!("{} + ", e.packages.join(" "))
    };
    format!(
        "{}: {pkgs}{} from huggingface.co ({}@{})",
        e.name,
        models::mb(e.bytes),
        e.repo,
        &e.revision[..e.revision.len().min(8)]
    )
}

#[cfg(test)]
mod tests;
