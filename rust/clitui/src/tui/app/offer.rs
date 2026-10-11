//! The install panel over the ask tab: when a query fails because the
//! corpus's model needs packages or weights this machine lacks, and the
//! payload's catalog offers that model, the panel says what installing it
//! takes (packages into the managed venv, megabytes from huggingface.co at a
//! pinned revision, the model_hash it must come out as) and runs the same
//! install `urna setup --model` runs. `y` is the consent to download, `r`
//! the separate one for a model's repo code, `n` or esc declines; the
//! query runs again once the install succeeds.

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, channel};

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::widgets::{Clear, Widget};

use crate::tui::setup::models::{self, Consent, Entry, Kit, Plan, Progress, Refusal};
use crate::tui::setup::scan::{self, tilde};
use crate::tui::{hud, pal};

/// The corpus model a failed query needs, from the manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Need {
    pub model: String,
    pub corpus_hash: String,
}

/// What the panel knows, in the order it learns it.
pub enum State {
    /// the plan is running (python, no network).
    Planning(Receiver<Result<Ready, Refusal>>),
    /// what the install takes, waiting for `y`.
    Ready(Ready),
    /// queries do not run the managed venv: the panel shows the command.
    Unmanaged(String),
    Running(Ready, Receiver<Ev>),
    Failed(Ready, String),
}

#[derive(Clone, Debug)]
pub struct Ready {
    pub kit: Kit,
    pub entry: Entry,
    pub plan: Plan,
    pub python: PathBuf,
}

pub enum Ev {
    Progress(Progress),
    Done(Result<String, String>),
}

pub struct Offer {
    pub need: Need,
    pub state: State,
    pub remote_code: bool,
    pub log: Vec<String>,
    pub bytes: Option<(u64, u64)>,
    pub step: String,
}

/// What `poll` reports to the app.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    Installed(String),
}

impl Offer {
    /// The panel for `need` when the catalog offers its model; `None` when
    /// nothing here can install it (no catalog, or a model it leaves out:
    /// the query's own message stands then).
    pub fn new(need: Need) -> Option<Offer> {
        let kit = Kit::resolve()?;
        let entry = kit.catalog.find(&need.model).ok()?.clone();
        let hash = need.corpus_hash.clone();
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let res = models::query_python().and_then(|python| {
                let plan = models::plan(&kit, &entry, &python, Some(&hash))?;
                Ok(Ready {
                    kit,
                    entry,
                    plan,
                    python,
                })
            });
            let _ = tx.send(res);
        });
        Some(Offer {
            need,
            state: State::Planning(rx),
            remote_code: false,
            log: Vec::new(),
            bytes: None,
            step: String::new(),
        })
    }

    /// an install is under way (planning can still be closed).
    pub fn running(&self) -> bool {
        matches!(self.state, State::Running(..))
    }

    /// `y`: runs the install with the download consent (and the repo-code
    /// one when `r` gave it).
    pub fn accept(&mut self) {
        let ready = match &self.state {
            State::Ready(r) | State::Failed(r, _) => r.clone(),
            _ => return,
        };
        if ready.plan.remote_code && !self.remote_code {
            return;
        }
        let consent = Consent {
            download: true,
            remote_code: self.remote_code,
        };
        let (tx, rx) = channel();
        let r = ready.clone();
        let hash = self.need.corpus_hash.clone();
        std::thread::spawn(move || {
            let uv = scan::which("uv");
            let line = tx.clone();
            let res = models::install(
                &r.kit,
                &r.entry,
                &r.python,
                uv.as_deref(),
                consent,
                Some(&hash),
                &mut |p| {
                    let _ = line.send(Ev::Progress(p));
                },
            );
            let _ = tx.send(Ev::Done(res.map_err(|e| e.to_string())));
        });
        self.log.clear();
        self.bytes = None;
        self.step = "starting".into();
        self.state = State::Running(ready, rx);
    }

    /// `r`: the separate consent for a model that runs its repo's code.
    pub fn toggle_remote_code(&mut self) {
        if let State::Ready(r) | State::Failed(r, _) = &self.state
            && r.plan.remote_code
        {
            self.remote_code = !self.remote_code;
        }
    }

    /// Drains the workers; `Some(Installed)` once the install succeeded.
    pub fn poll(&mut self) -> Option<Outcome> {
        if let State::Planning(rx) = &self.state
            && let Ok(res) = rx.try_recv()
        {
            self.state = match res {
                Ok(r) => State::Ready(r),
                Err(e) => State::Unmanaged(e.to_string()),
            };
        }
        let State::Running(ready, rx) = &self.state else {
            return None;
        };
        let mut done = None;
        while let Ok(ev) = rx.try_recv() {
            match ev {
                Ev::Progress(Progress::Step(s)) => {
                    self.log.push(s.clone());
                    self.step = s;
                }
                Ev::Progress(Progress::Bytes(n, total, _)) => self.bytes = Some((n, total)),
                Ev::Progress(Progress::Log(l)) => self.log.push(crate::tui::txt::home(&l)),
                Ev::Done(r) => done = Some(r),
            }
        }
        let ready = ready.clone();
        match done? {
            Ok(msg) => Some(Outcome::Installed(msg)),
            Err(e) => {
                self.state = State::Failed(ready, e);
                None
            }
        }
    }

    /// The lines that describe the install, before any key.
    pub fn facts(&self, r: &Ready) -> Vec<(String, Style)> {
        let e = &r.entry;
        // the venv dir, not its interpreter: `<venv>/bin/python`.
        let venv = r
            .python
            .parent()
            .and_then(|bin| bin.parent())
            .map_or_else(|| tilde(&r.python), tilde);
        let mut v = vec![(
            format!(
                "{} is the model this corpus was built with ({})",
                e.name, e.embedding_model
            ),
            pal::title(),
        )];
        v.push(match r.plan.packages_missing.as_slice() {
            [] => (format!("packages: already in {venv}"), pal::dim()),
            p => (format!("packages: {} into {venv}", p.join(" ")), pal::dim()),
        });
        let fetch = if r.plan.cached {
            format!(
                "weights: cached at the pinned revision {}",
                short(&e.revision)
            )
        } else {
            format!(
                "download: {} from huggingface.co, {}@{}",
                models::mb(r.plan.missing_bytes),
                e.repo,
                short(&e.revision)
            )
        };
        v.push((fetch, pal::dim()));
        v.push((
            format!(
                "verify: model_hash {} = this corpus's",
                short_hash(&e.model_hash)
            ),
            pal::dim(),
        ));
        if r.plan.remote_code {
            v.push(if self.remote_code {
                (
                    "repo code: allowed for this model (r withdraws)".into(),
                    pal::ok(),
                )
            } else {
                (
                    "repo code: this model runs code from its repo; r allows it, separately".into(),
                    pal::warn(),
                )
            });
        }
        v
    }
}

fn short(rev: &str) -> &str {
    &rev[..rev.len().min(8)]
}

fn short_hash(h: &str) -> String {
    format!("{}…", &h[..h.len().min(19)])
}

/// The panel, centered over the ask tab, the screen behind it dimmed.
pub fn render(buf: &mut Buffer, area: Rect, offer: &Offer, spinner: &str) {
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            if let Some(c) = buf.cell_mut((x, y)) {
                c.fg = pal::mix(c.fg, pal::BG, 0.65);
            }
        }
    }
    let (mut lines, keys) = body(offer, spinner);
    let w = area.width.saturating_sub(6).min(100);
    let inner_w = w.saturating_sub(4) as usize;
    lines = lines
        .into_iter()
        .flat_map(|(s, st)| {
            hud::wrap(&s, inner_w)
                .into_iter()
                .map(move |l| (l, st))
                .collect::<Vec<_>>()
        })
        .collect();
    let want = lines.len() as u16 + 4;
    let h = want.clamp(8, area.height.saturating_sub(2));
    let r = Rect::new(area.x + (area.width - w) / 2, area.y + 1, w, h);
    Clear.render(r, buf);
    buf.set_style(r, Style::new().bg(pal::BG));
    let title = format!("install {}", offer.need.model);
    let inner = hud::panel(buf, r, &title, "", true);
    let foot = inner.bottom().saturating_sub(1);
    let room = foot.saturating_sub(inner.y) as usize;
    // a long log keeps its tail; while an install runs, its first line (the
    // step under way) stays on top of the tail.
    let pinned = usize::from(matches!(offer.state, State::Running(..)) && room > 1);
    let skip = lines.len().saturating_sub(room);
    let shown = lines[..pinned]
        .iter()
        .chain(lines.iter().skip(skip.max(pinned) + pinned.min(skip)));
    let w = inner.width.saturating_sub(2);
    for (i, (s, st)) in shown.enumerate() {
        hud::put(buf, inner.x + 1, inner.y + i as u16, s, *st, w);
    }
    if let (State::Running(..), Some((n, total))) = (&offer.state, offer.bytes)
        && total > 0
    {
        let ratio = n as f64 / total as f64;
        let label = format!(" {} / {}", models::mb(n), models::mb(total));
        let gw = w.saturating_sub(label.len() as u16 + 1);
        hud::gauge(
            buf,
            inner.x + 1,
            foot.saturating_sub(1),
            gw,
            ratio,
            pal::GLOW,
        );
        hud::put(
            buf,
            inner.x + 1 + gw,
            foot.saturating_sub(1),
            &label,
            pal::dim(),
            w,
        );
    }
    hud::put(buf, inner.x + 1, foot, keys, pal::key(), w);
}

/// The panel's lines and its key hint for the current state.
fn body(offer: &Offer, spinner: &str) -> (Vec<(String, Style)>, &'static str) {
    match &offer.state {
        State::Planning(_) => (
            vec![(
                format!("{spinner} reading the catalog and the cache"),
                pal::accent(),
            )],
            "n closes",
        ),
        State::Unmanaged(why) => {
            let line = format!("urna setup --model {}", offer.need.model);
            (
                vec![
                    (why.clone(), pal::warn()),
                    (format!("from a shell: {line}"), pal::dim()),
                ],
                "n closes",
            )
        }
        State::Ready(r) => {
            let keys = if r.plan.remote_code && !offer.remote_code {
                "r allows its repo code first · n not now"
            } else if r.plan.cached && r.plan.packages_missing.is_empty() {
                "y checks it and asks again · n not now"
            } else {
                "y installs and downloads, then asks again · n not now"
            };
            (offer.facts(r), keys)
        }
        State::Running(..) => {
            let mut v = vec![(format!("{spinner} {}", offer.step), pal::accent())];
            v.extend(offer.log.iter().map(|l| (l.clone(), pal::faint())));
            v.push((String::new(), pal::faint()));
            (v, "the query runs again when this ends")
        }
        State::Failed(r, e) => {
            let mut v = offer.facts(r);
            v.push((format!("failed: {e}"), pal::err()));
            (v, "y tries again · n closes")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ready(cached: bool, remote: bool) -> Ready {
        let catalog = crate::tui::setup::models::tests::sample();
        let entry = catalog.models[usize::from(remote)].clone();
        Ready {
            kit: Kit {
                script: "/d/urna/python/urna/model/installer.py".into(),
                catalog,
            },
            entry,
            plan: Plan {
                missing_bytes: if cached { 0 } else { 479_729_010 },
                cached,
                packages_missing: vec!["sentence-transformers>=3".into()],
                remote_code: remote,
            },
            python: "/d/urna/venv/bin/python".into(),
        }
    }

    fn offer(state: State) -> Offer {
        Offer {
            need: Need {
                model: "org/a".into(),
                corpus_hash: "sha256:a".into(),
            },
            state,
            remote_code: false,
            log: Vec::new(),
            bytes: None,
            step: String::new(),
        }
    }

    fn screen(o: &Offer) -> String {
        let area = Rect::new(0, 0, 110, 24);
        let mut buf = Buffer::empty(area);
        render(&mut buf, area, o, "·");
        (0..24)
            .map(|y| {
                (0..110)
                    .map(|x| buf[(x, y)].symbol().to_string())
                    .collect::<String>()
                    + "\n"
            })
            .collect()
    }

    #[test]
    fn the_panel_says_what_the_install_takes_before_any_key() {
        let s = screen(&offer(State::Ready(ready(false, false))));
        assert!(
            s.contains("packages: sentence-transformers>=3 into /d/urna/venv "),
            "{s}"
        );
        assert!(
            s.contains("download: 480 MB from huggingface.co, org/a@aaaa"),
            "{s}"
        );
        assert!(s.contains("model_hash sha256:a"), "{s}");
        assert!(s.contains("y installs and downloads"), "{s}");
        let cached = screen(&offer(State::Ready(ready(true, false))));
        assert!(cached.contains("cached at the pinned revision"), "{cached}");
    }

    #[test]
    fn repo_code_waits_for_its_own_consent() {
        let mut o = offer(State::Ready(ready(false, true)));
        assert!(screen(&o).contains("r allows its repo code first"));
        o.accept();
        assert!(
            matches!(o.state, State::Ready(_)),
            "y alone must not start it"
        );
        o.toggle_remote_code();
        assert!(o.remote_code && screen(&o).contains("repo code: allowed"));
    }

    #[test]
    fn an_env_urna_does_not_manage_gets_the_command_not_an_install() {
        let mut o = offer(State::Unmanaged(
            "URNA_PYTHON pins /usr/bin/python3; urna installs packages only into the venv".into(),
        ));
        o.accept();
        assert!(matches!(o.state, State::Unmanaged(_)));
        let s = screen(&o);
        assert!(
            s.contains("urna setup --model org/a") && s.contains("n closes"),
            "{s}"
        );
    }

    #[test]
    fn a_running_install_keeps_its_step_above_the_log_tail() {
        let (_tx, rx) = channel();
        let mut o = offer(State::Running(ready(false, false), rx));
        o.step = "fetch org/a at aaaa (480 MB)".into();
        o.log = (0..60).map(|i| format!("log line {i}")).collect();
        o.bytes = Some((240_000_000, 479_729_010));
        let s = screen(&o);
        assert!(s.contains("fetch org/a at aaaa (480 MB)"), "{s}");
        assert!(
            s.contains("log line 59") && !s.contains("log line 0 "),
            "{s}"
        );
        assert!(s.contains("240 MB / 480 MB"), "{s}");
    }

    #[test]
    fn a_failure_keeps_the_facts_and_offers_a_retry() {
        let o = offer(State::Failed(
            ready(false, false),
            "fetch failed: connection refused".into(),
        ));
        let s = screen(&o);
        assert!(
            s.contains("failed: fetch failed: connection refused"),
            "{s}"
        );
        assert!(s.contains("y tries again"), "{s}");
    }
}
