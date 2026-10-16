//! `urna tui`: the corpus explorer, full screen. four tabs (home, corpus,
//! ask, health), a file picker overlay, toasts, and an effect on every
//! change of screen. slow work (opening a corpus, asking it, the doctor
//! checks) runs on worker threads; this loop only draws and routes input.
//! `s` hands the terminal to `urna setup` and comes back when it ends.
//!
//! it opens the file it is given, else the corpus it opened last (on the
//! ask tab, ready for a question), else the first-run screen; back from
//! setup with the sample corpus just downloaded, it opens the sample. a
//! corpus that will not open lands on the first-run screen too, and is
//! forgotten, so the next launch does not try it again.

pub mod ask;
mod blocks;
mod card;
mod chrome;
pub mod corpus;
mod draw;
pub mod headless;
mod health;
mod hits;
mod home;
mod keys;
mod labels;
mod media;
mod offer;
mod pick;
pub mod session;
mod state;

use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

use ratatui::DefaultTerminal;
use ratatui::layout::Rect;

use crate::cmd::pyenv;
use crate::cmd::tone::{self, Depth};
use crate::tui::fx::{self, Fx};
use crate::tui::setup;
use crate::tui::term::{self, Clock};
use crate::tui::toast::{Kind, Toasts};
use crate::tui::widgets::{Picker, Spinner};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tab {
    Home,
    Corpus,
    Ask,
    Health,
}

enum Exit {
    Quit,
    Setup,
}

/// What the explorer opens on.
#[derive(Debug, PartialEq, Eq)]
enum Launch {
    /// `urna tui file.urna`: that file, on the corpus tab.
    File(PathBuf),
    /// the corpus opened last, or the sample setup just downloaded, on the
    /// ask tab.
    Ask(PathBuf),
    /// nothing to reopen: the first-run screen.
    First,
}

impl Launch {
    /// A file given wins over the saved last corpus; a last corpus deleted
    /// or moved since is no corpus.
    fn pick(file: Option<PathBuf>, last: Option<&Path>) -> Launch {
        match (file, last) {
            (Some(f), _) => Launch::File(f),
            (None, Some(l)) => Launch::Ask(l.to_path_buf()),
            (None, None) => Launch::First,
        }
    }
}

type Loaded = Result<corpus::Corpus, String>;

struct App {
    tab: Tab,
    entered: bool,
    corpus: Option<corpus::Corpus>,
    /// the corpus being opened and the tab it lands on.
    loading: Option<(Receiver<Loaded>, PathBuf, Tab)>,
    /// no corpus to reopen, or the one to reopen would not open: home is
    /// the first-run screen until one opens.
    first_run: bool,
    /// where the corpora opened are remembered, when there is a data dir.
    state_file: Option<PathBuf>,
    /// the sample corpus setup laid down, for the first-run screen.
    sample: Option<PathBuf>,
    sec_scroll: usize,
    ask: ask::Ask,
    health: health::Health,
    picker: Option<Picker>,
    picker_new: bool,
    files: Vec<(PathBuf, u64)>,
    cwd: PathBuf,
    tab_hits: Vec<(Tab, Rect)>,
    toasts: Toasts,
    fx: Fx,
    clock: Clock,
    spinner: Spinner,
    depth: Depth,
    booted: bool,
    area: Rect,
    leaving: Option<(Instant, Exit)>,
    /// `urna tui FILE --ask QUERY`: asked as soon as the corpus is open.
    ask_on_open: Option<String>,
}

impl App {
    fn new(launch: Launch, sample: Option<PathBuf>) -> Self {
        let cwd = std::env::current_dir().unwrap_or_default();
        let mut app = App {
            tab: Tab::Home,
            entered: true,
            corpus: None,
            loading: None,
            first_run: launch == Launch::First,
            state_file: state::file(),
            sample,
            sec_scroll: 0,
            ask: ask::Ask::default(),
            health: health::Health::default(),
            picker: None,
            picker_new: false,
            files: home::recent(&cwd),
            cwd,
            tab_hits: Vec::new(),
            toasts: Toasts::default(),
            fx: Fx::default(),
            clock: Clock::new(),
            spinner: Spinner::dots(),
            depth: tone::depth(),
            booted: false,
            area: Rect::default(),
            leaving: None,
            ask_on_open: None,
        };
        app.health.refresh();
        match launch {
            Launch::File(f) => app.open(f, Tab::Corpus),
            Launch::Ask(f) => app.open(f, Tab::Ask),
            Launch::First => {}
        }
        app
    }

    fn go(&mut self, tab: Tab) {
        if tab != self.tab {
            self.tab = tab;
            self.entered = true;
            if tab == Tab::Health && !self.health.busy() {
                self.health.refresh();
            }
        }
    }

    fn open(&mut self, path: PathBuf, land: Tab) {
        let (tx, rx) = channel();
        let p = path.clone();
        std::thread::spawn(move || {
            let _ = tx.send(corpus::load(&p).map_err(|e| format!("{e:#}")));
        });
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        self.toasts.push(format!("opening {name}"), Kind::Info);
        self.loading = Some((rx, path, land));
    }

    fn leave(&mut self, how: Exit) {
        if self.leaving.is_none() {
            self.leaving = Some((Instant::now(), how));
            self.fx.add_unique_effect("leave", fx::leave(self.area));
        }
    }

    fn pump(&mut self) {
        if let Some((rx, path, land)) = &self.loading
            && let Ok(res) = rx.try_recv()
        {
            let land = *land;
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            match res {
                Ok(c) => {
                    let kind = if c.valid.is_ok() {
                        Kind::Ok
                    } else {
                        Kind::Warn
                    };
                    self.toasts.push(
                        format!(
                            "{name}: {} chunks · {}",
                            c.facts[0].1,
                            corpus::human(c.size)
                        ),
                        kind,
                    );
                    if let Some(f) = &self.state_file {
                        state::remember(f, &c.path);
                    }
                    self.corpus = Some(c);
                    self.first_run = false;
                    self.sec_scroll = 0;
                    // a fresh ask for the new corpus, keeping what the
                    // terminal answered it can draw (probed once, at start).
                    let picker = self.ask.media.take_picker();
                    let glyphs = self.ask.media.glyphs();
                    self.ask = ask::Ask::default();
                    if let Some(p) = picker {
                        self.ask.media.set_picker(p, glyphs);
                    }
                    self.go(land);
                    if let Some(q) = self.ask_on_open.take()
                        && let Some(c) = &self.corpus
                    {
                        self.ask.input.set_value(q);
                        self.ask.input.end();
                        self.ask.submit(c.session.clone());
                    }
                }
                Err(e) => {
                    self.toasts.push(format!("{name}: {e}"), Kind::Err);
                    if let Some(f) = &self.state_file {
                        state::forget(f, path);
                    }
                    // with nothing open, home goes back to the first-run
                    // screen rather than the ordinary one.
                    if self.corpus.is_none() {
                        self.first_run = true;
                    }
                }
            }
            self.loading = None;
        }
        match self.ask.poll() {
            Some(Ok(0)) => self.toasts.push("no hits", Kind::Warn),
            // a model the catalog offers opens the install panel instead of
            // a toast that would sit on top of it.
            Some(Err(e)) => match e.need.and_then(offer::Offer::new) {
                Some(o) => self.ask.offer = Some(o),
                None => self.toasts.push(e.text, Kind::Err),
            },
            _ => {}
        }
        let installed = self.ask.offer.as_mut().and_then(offer::Offer::poll);
        if let Some(offer::Outcome::Installed(msg)) = installed {
            self.ask.offer = None;
            self.toasts
                .push(format!("installed {msg}; asking again"), Kind::Ok);
            if let Some(c) = &self.corpus {
                self.ask.retry(c.session.clone());
            }
        }
        if self.health.poll() && self.tab == Tab::Health {
            let n = self
                .health
                .rows
                .iter()
                .filter(|r| matches!(r.level, crate::cmd::health::Level::Fail(_)))
                .count();
            if n > 0 {
                self.toasts
                    .push(format!("{n} check(s) failing: s runs setup"), Kind::Warn);
            }
        }
    }

    fn run(mut self, terminal: &mut DefaultTerminal) -> anyhow::Result<Exit> {
        loop {
            self.pump();
            let dt = self.clock.tick();
            self.spinner.tick(dt);
            self.draw(terminal, dt)?;
            if let Some((t0, _)) = &self.leaving
                && t0.elapsed() > Duration::from_millis(380)
            {
                return Ok(self.leaving.take().map(|(_, h)| h).unwrap_or(Exit::Quit));
            }
            if let Some(ev) = term::next_event()? {
                keys::route(&mut self, ev);
            }
        }
    }
}

/// Runs the explorer; returns the exit code.
pub fn run(file: Option<PathBuf>, mut ask: Option<String>) -> anyhow::Result<i32> {
    pyenv::set_quiet(true);
    let home = crate::cmd::paths::urna_home();
    let sample = || home.as_deref().and_then(setup::sample::installed);
    let saved = state::file().map(|f| state::State::read(&f));
    let mut launch = Launch::pick(file, saved.as_ref().and_then(state::State::last));
    loop {
        let mut terminal = term::fullscreen()?;
        // a query to ask lands the file on the ask tab instead of corpus;
        // it is asked once, not again after a trip through setup.
        let ask_now = ask.take();
        let launch_now = match (launch, &ask_now) {
            (Launch::File(f), Some(_)) => Launch::Ask(f),
            (l, _) => l,
        };
        let mut app = App::new(launch_now, sample());
        app.ask_on_open = ask_now;
        app.ask.media.set_picker(
            crate::tui::graphics::picker(),
            crate::tui::graphics::glyph_set(),
        );
        // a terminal that does not know a query may have printed some of
        // it: the first frame starts from a clean screen. `CSI 2J`, not
        // `Terminal::clear`, which reads the cursor position (one more
        // query a terminal may not answer); ratatui's buffer is still empty.
        crossterm::execute!(
            std::io::stdout(),
            crossterm::terminal::Clear(crossterm::terminal::ClearType::All)
        )?;
        let res = app.run(&mut terminal);
        term::leave_fullscreen()?;
        match res? {
            Exit::Quit => return Ok(0),
            Exit::Setup => {
                let before = sample().map(|p| stamp(&p));
                setup::run(setup::Opts::default())?;
                let fresh = sample().filter(|p| Some(stamp(p)) != before);
                let saved = state::file().map(|f| state::State::read(&f));
                launch = match fresh {
                    Some(p) => Launch::Ask(p),
                    None => Launch::pick(None, saved.as_ref().and_then(state::State::last)),
                };
            }
        }
    }
}

/// A file's path and modification time: a sample setup downloads again
/// lands under the same name with a new time.
fn stamp(p: &Path) -> (PathBuf, Option<std::time::SystemTime>) {
    let t = std::fs::metadata(p).and_then(|m| m.modified()).ok();
    (p.to_path_buf(), t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_given_wins_then_the_last_corpus_then_the_first_run() {
        let f = PathBuf::from("/x/given.urna");
        let last = Path::new("/x/last.urna");
        assert_eq!(
            Launch::pick(Some(f.clone()), Some(last)),
            Launch::File(f.clone())
        );
        assert_eq!(Launch::pick(Some(f.clone()), None), Launch::File(f));
        assert_eq!(
            Launch::pick(None, Some(last)),
            Launch::Ask(last.to_path_buf())
        );
        assert_eq!(Launch::pick(None, None), Launch::First);
    }

    #[test]
    fn the_first_run_screen_shows_until_a_corpus_opens() {
        let app = App::new(Launch::First, None);
        assert!(app.first_run && app.tab == Tab::Home && app.loading.is_none());
        let app = App::new(Launch::Ask(PathBuf::from("/nowhere/x.urna")), None);
        assert!(!app.first_run);
        assert!(matches!(&app.loading, Some((_, _, Tab::Ask))));
        let app = App::new(Launch::File(PathBuf::from("/nowhere/x.urna")), None);
        assert!(matches!(&app.loading, Some((_, _, Tab::Corpus))));
    }

    #[test]
    fn a_corpus_that_will_not_open_lands_on_the_first_run_screen_forgotten() {
        let d = std::env::temp_dir().join(format!("urna_broken_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let broken = d.join("broken.urna");
        let other = d.join("other.urna");
        for p in [&broken, &other] {
            std::fs::write(p, b"not a corpus").unwrap();
        }
        let f = d.join("tui.json");
        state::remember(&f, &other);
        state::remember(&f, &broken);
        let mut app = App::new(Launch::First, None);
        app.state_file = Some(f.clone());
        app.first_run = false;
        let (tx, rx) = channel();
        app.loading = Some((rx, broken.clone(), Tab::Ask));
        tx.send(Err("broken".into())).unwrap();
        app.pump();
        assert!(app.first_run && app.corpus.is_none() && app.loading.is_none());
        assert_eq!(app.tab, Tab::Home);
        assert_eq!(
            state::State::read(&f).recent,
            [other.canonicalize().unwrap()]
        );
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_sample_setup_downloads_again_changes_its_stamp() {
        let d = std::env::temp_dir().join(format!("urna_stamp_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("fakenews.urna");
        std::fs::write(&p, b"v1").unwrap();
        let before = stamp(&p);
        std::thread::sleep(Duration::from_millis(20));
        std::fs::write(d.join(".tmp"), b"v2").unwrap();
        std::fs::rename(d.join(".tmp"), &p).unwrap();
        assert_ne!(stamp(&p), before);
        std::fs::remove_dir_all(&d).unwrap();
    }
}
