//! The interactive installer: state, input and the event loop. A splash,
//! then four steps in an inline viewport: scan, plan, install, verify. The
//! scan starts on launch, so it is usually done by the time the splash is
//! dismissed; the install runs on the `job` worker and this loop only
//! folds its events into state. drawing lives in `screen`.

use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::{Duration, Instant};

use crossterm::event::{KeyCode, KeyModifiers};
use ratatui_cheese::spinner::{SpinnerState, SpinnerType};

use super::job::{self, Ev};
use super::plan::{self, Item, Opts, Task};
use super::scan::Scan;
use crate::cmd::health;
use crate::cmd::tone::{self, Depth};
use crate::tui::fx::Fx;
use crate::tui::hud::Badge;
use crate::tui::term::Clock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Splash,
    Scan,
    Plan,
    /// the model picker, opened from the plan or, after a fresh install,
    /// from the done screen (the catalog comes with the payload).
    Models,
    Run,
    Done,
}

pub struct RunRow {
    pub task: Task,
    pub badge: Option<Badge>,
    pub note: String,
    pub bytes: u64,
    pub total: Option<u64>,
    pub result: Option<String>,
}

pub struct Ui {
    pub step: Step,
    pub since: Instant,
    pub opts: Opts,
    pub scan: Option<Scan>,
    scan_rx: Option<Receiver<Scan>>,
    /// facts revealed so far on the scan screen (they land one by one).
    pub shown: usize,
    /// facts already drawn (the ones past it get the landing effect).
    pub drawn: usize,
    pub plan: Vec<Item>,
    pub cursor: usize,
    /// the picker's cursor and the step it returns to.
    pub pick: usize,
    pick_from: Step,
    pub rows: Vec<RunRow>,
    pub log: Vec<String>,
    /// `None` follows the tail of the log.
    pub scroll: Option<usize>,
    pub health: Vec<health::Row>,
    pub code: i32,
    job: Option<Receiver<Ev>>,
    pub finished: bool,
    /// the view adds the entry effects the first frame after a step change.
    pub entered: bool,
    /// tasks that just finished; the view flashes their rows.
    pub flashes: Vec<(Task, bool)>,
    /// the run row the traveling glint is on.
    pub glint_on: Option<usize>,
    pub spinner: SpinnerState,
    pub fx: Fx,
    pub clock: Clock,
    pub depth: Depth,
    pub(super) quit: Option<i32>,
}

impl Ui {
    pub(super) fn new(opts: Opts) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(Scan::probe());
        });
        Ui {
            step: Step::Splash,
            since: Instant::now(),
            opts,
            scan: None,
            scan_rx: Some(rx),
            shown: 0,
            drawn: 0,
            plan: Vec::new(),
            cursor: 0,
            pick: 0,
            pick_from: Step::Plan,
            rows: Vec::new(),
            log: Vec::new(),
            scroll: None,
            health: Vec::new(),
            code: 0,
            job: None,
            finished: false,
            entered: true,
            flashes: Vec::new(),
            glint_on: None,
            spinner: SpinnerState::new(SpinnerType::Dot),
            fx: Fx::default(),
            clock: Clock::new(),
            depth: tone::depth(),
            quit: None,
        }
    }

    fn go(&mut self, step: Step) {
        self.step = step;
        self.since = Instant::now();
        self.entered = true;
    }

    pub fn in_step(&self) -> Duration {
        self.since.elapsed()
    }

    /// Pulls whatever the scan and the worker produced since last frame.
    pub(super) fn pump(&mut self) {
        if let Some(rx) = &self.scan_rx
            && let Ok(scan) = rx.try_recv()
        {
            self.plan = plan::plan(&scan, &self.opts);
            self.scan = Some(scan);
            self.scan_rx = None;
        }
        if self.step == Step::Scan
            && let Some(scan) = &self.scan
        {
            let due = (self.in_step().as_millis() / 140) as usize;
            self.shown = due.min(scan.facts().len());
            if self.shown == scan.facts().len()
                && self.in_step() > Duration::from_millis(140 * self.shown as u64 + 900)
            {
                self.go(Step::Plan);
            }
        }
        loop {
            let ev = match self.job.as_ref().map(|rx| rx.try_recv()) {
                Some(Ok(ev)) => ev,
                Some(Err(TryRecvError::Disconnected)) => {
                    self.job = None;
                    break;
                }
                _ => break,
            };
            self.apply(ev);
        }
        if self.finished && self.step == Step::Run && self.in_step() > Duration::from_millis(400) {
            let since_end = self.rows.iter().all(|r| r.result.is_some());
            if since_end {
                self.go(Step::Done);
            }
        }
    }

    fn row(&mut self, task: Task) -> Option<&mut RunRow> {
        self.rows.iter_mut().find(|r| r.task == task)
    }

    fn apply(&mut self, ev: Ev) {
        match ev {
            Ev::Start(t) => {
                if let Some(r) = self.row(t) {
                    r.badge = Some(Badge::Busy);
                }
            }
            Ev::Bytes(t, n, total) => {
                if let Some(r) = self.row(t) {
                    r.bytes = n;
                    r.total = total;
                }
            }
            Ev::Note(t, note) => {
                if let Some(r) = self.row(t) {
                    r.note = note;
                }
            }
            Ev::Log(line) => self.log.push(crate::tui::txt::home(&line)),
            Ev::Health(rows) => self.health = rows,
            Ev::Done(t, res) => {
                let ok = res.is_ok();
                if let Err((c, _)) = &res
                    && self.code == 0
                {
                    self.code = *c;
                }
                if let Some(r) = self.row(t) {
                    r.badge = Some(if ok { Badge::Ok } else { Badge::Fail });
                    r.result = Some(match res {
                        Ok(m) => m,
                        Err((_, m)) => m,
                    });
                }
                self.flashes.push((t, ok));
            }
            Ev::End => {
                self.finished = true;
                self.since = Instant::now();
            }
        }
    }

    fn start(&mut self) {
        self.code = plan::blocked_code(&self.plan);
        let tasks: Vec<Task> = self
            .plan
            .iter()
            .filter(|i| i.runs())
            .map(|i| i.task)
            .collect();
        self.run(tasks);
    }

    fn run(&mut self, tasks: Vec<Task>) {
        let Some(scan) = self.scan.clone() else {
            return;
        };
        self.finished = false;
        self.log.clear();
        self.scroll = None;
        self.rows = tasks
            .iter()
            .map(|&task| RunRow {
                task,
                badge: None,
                note: "queued".into(),
                bytes: 0,
                total: None,
                result: None,
            })
            .collect();
        self.job = Some(job::spawn(scan, tasks, self.opts.clone()));
        self.go(Step::Run);
    }

    /// The offered models, from the scan's payload kit.
    pub fn offered(&self) -> Vec<super::models::Entry> {
        self.scan
            .as_ref()
            .and_then(|s| s.kit.as_ref())
            .map(|k| k.catalog.models.clone())
            .unwrap_or_default()
    }

    pub fn chosen(&self, name: &str) -> bool {
        self.opts.models.iter().any(|m| m == name || m == "all")
    }

    /// Opens the picker when a catalog is at hand; after a run the payload
    /// may just have arrived, so the kit is read again first.
    fn open_picker(&mut self, from: Step) {
        if from == Step::Done
            && let Some(scan) = self.scan.as_mut()
        {
            scan.kit = scan
                .home
                .as_ref()
                .and_then(|h| super::models::Kit::in_home(h));
        }
        if self.offered().is_empty() {
            return;
        }
        self.pick = 0;
        self.pick_from = from;
        self.go(Step::Models);
    }

    fn pick_key(&mut self, code: KeyCode) {
        let offered = self.offered();
        let Some(here) = offered.get(self.pick).cloned() else {
            return;
        };
        // `all` is spelled out so one model can be dropped from it.
        if self.opts.models.iter().any(|m| m == "all") {
            self.opts.models = offered.iter().map(|e| e.name.clone()).collect();
        }
        let toggle = |list: &mut Vec<String>, name: &str| {
            if let Some(i) = list.iter().position(|m| m == name) {
                list.remove(i);
            } else {
                list.push(name.to_string());
            }
        };
        match code {
            KeyCode::Up | KeyCode::Char('k') => self.pick = self.pick.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => {
                self.pick = (self.pick + 1).min(offered.len().saturating_sub(1))
            }
            KeyCode::Char(' ') | KeyCode::Char('x') => toggle(&mut self.opts.models, &here.name),
            KeyCode::Char('a') => {
                let every = offered.iter().all(|e| self.chosen(&e.name));
                self.opts.models = if every {
                    Vec::new()
                } else {
                    offered.iter().map(|e| e.name.clone()).collect()
                };
            }
            KeyCode::Char('r') if here.remote_code => {
                toggle(&mut self.opts.allow_remote_code, &here.name)
            }
            KeyCode::Enter | KeyCode::Esc | KeyCode::Char('q') => self.close_picker(code),
            _ => {}
        }
    }

    fn close_picker(&mut self, code: KeyCode) {
        if let Some(scan) = &self.scan {
            plan::reselect(&mut self.plan, scan, &self.opts);
        }
        let ready = self.plan.iter().any(|i| i.task == Task::Models && i.runs());
        if self.pick_from == Step::Done && code == KeyCode::Enter && ready {
            // the run after a fresh install: the managed venv when models
            // need one it lacks, the models, then verify again.
            if let Some(scan) = self.scan.as_mut() {
                scan.venv = scan
                    .home
                    .as_ref()
                    .is_some_and(|h| crate::cmd::paths::venv_python(&h.join("venv")).is_file());
            }
            let tasks: Vec<Task> = match &self.scan {
                Some(scan) => plan::plan(scan, &self.opts)
                    .into_iter()
                    .filter(|i| matches!(i.task, Task::Python | Task::Models) && i.runs())
                    .map(|i| i.task)
                    .chain([Task::Verify])
                    .collect(),
                None => return,
            };
            self.code = 0;
            self.run(tasks);
        } else {
            let back = self.pick_from;
            self.go(back);
        }
    }

    pub(super) fn key(&mut self, code: KeyCode, mods: KeyModifiers) {
        if code == KeyCode::Char('c') && mods.contains(KeyModifiers::CONTROL) {
            self.quit = Some(130);
            return;
        }
        let quit = matches!(code, KeyCode::Char('q') | KeyCode::Esc);
        match self.step {
            Step::Splash => match code {
                KeyCode::Enter | KeyCode::Char(' ') | KeyCode::Right => self.go(Step::Scan),
                _ if quit => self.quit = Some(130),
                _ => {}
            },
            Step::Scan => match code {
                KeyCode::Enter if self.scan.is_some() => self.go(Step::Plan),
                _ if quit => self.quit = Some(130),
                _ => {}
            },
            Step::Plan => match code {
                KeyCode::Up | KeyCode::Char('k') => self.cursor = self.cursor.saturating_sub(1),
                KeyCode::Down | KeyCode::Char('j') => {
                    self.cursor = (self.cursor + 1).min(self.plan.len().saturating_sub(1))
                }
                // the models step is ticked by choosing models, in the picker.
                KeyCode::Char(' ') | KeyCode::Char('x')
                    if self.plan.get(self.cursor).map(|i| i.task) == Some(Task::Models) =>
                {
                    self.open_picker(Step::Plan)
                }
                KeyCode::Char(' ') | KeyCode::Char('x') => {
                    if let Some(i) = self.plan.get_mut(self.cursor)
                        && !i.locked
                        && i.blocked.is_none()
                    {
                        i.on = !i.on;
                    }
                }
                KeyCode::Char('m') => self.open_picker(Step::Plan),
                KeyCode::Enter => self.start(),
                _ if quit => self.quit = Some(130),
                _ => {}
            },
            Step::Models => self.pick_key(code),
            Step::Run => match code {
                KeyCode::Up | KeyCode::Char('k') => {
                    let top = self.scroll.unwrap_or(self.log.len());
                    self.scroll = Some(top.saturating_sub(1));
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    self.scroll = self.scroll.map(|s| s + 1).filter(|s| *s < self.log.len());
                }
                KeyCode::End => self.scroll = None,
                _ => {}
            },
            Step::Done => {
                if code == KeyCode::Char('m') {
                    self.open_picker(Step::Done);
                } else if quit || matches!(code, KeyCode::Enter) {
                    self.quit = Some(self.code);
                }
            }
        }
    }
}
