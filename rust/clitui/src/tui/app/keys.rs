//! Input routing for the explorer. the picker overlay takes everything
//! while open; the ask tab owns the letters (they are the query), so there
//! quitting is ctrl+q and the picker is ctrl+o. everywhere else the keys
//! are single letters, and the mouse picks tabs and scrolls.

use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers, MouseEventKind};

use super::{App, Exit, Tab, chrome, pick};
use crate::tui::toast::Kind;

pub(super) fn route(app: &mut App, ev: Event) {
    if app.leaving.is_some() {
        return;
    }
    if let Some(fe) = app.picker.as_mut() {
        match pick::handle(fe, &ev) {
            pick::Pick::Stay => {}
            pick::Pick::Close => app.picker = None,
            pick::Pick::Open(p) => {
                app.picker = None;
                app.open(p, Tab::Corpus);
            }
        }
        return;
    }
    match ev {
        Event::Key(k) if k.kind != KeyEventKind::Release => key(app, k.code, k.modifiers),
        Event::Mouse(m) => match m.kind {
            MouseEventKind::Down(_) => {
                let hit = app
                    .tab_hits
                    .iter()
                    .find(|(_, r)| r.contains((m.column, m.row).into()));
                if let Some((tab, _)) = hit.copied() {
                    app.go(tab);
                }
            }
            MouseEventKind::ScrollDown => scroll(app, 3, true),
            MouseEventKind::ScrollUp => scroll(app, 3, false),
            _ => {}
        },
        _ => {}
    }
}

fn scroll(app: &mut App, n: usize, down: bool) {
    match app.tab {
        Tab::Corpus => {
            app.sec_scroll = if down {
                app.sec_scroll + n
            } else {
                app.sec_scroll.saturating_sub(n)
            };
        }
        Tab::Ask => {
            let s = app.ask.scroll as usize;
            app.ask.scroll = if down { s + n } else { s.saturating_sub(n) } as u16;
        }
        _ => {}
    }
}

fn open_picker(app: &mut App) {
    match pick::new(&app.cwd) {
        Ok(fe) => {
            app.picker = Some(fe);
            app.picker_new = true;
        }
        Err(e) => app
            .toasts
            .push(format!("cannot list {}: {e}", app.cwd.display()), Kind::Err),
    }
}

fn cycle(app: &mut App, back: bool) {
    let i = chrome::TABS
        .iter()
        .position(|(t, _)| *t == app.tab)
        .unwrap_or(0);
    let n = chrome::TABS.len();
    let next = if back { (i + n - 1) % n } else { (i + 1) % n };
    app.go(chrome::TABS[next].0);
}

fn key(app: &mut App, code: KeyCode, mods: KeyModifiers) {
    let ctrl = mods.contains(KeyModifiers::CONTROL);
    match code {
        KeyCode::Char('c') | KeyCode::Char('q') if ctrl => return app.leave(Exit::Quit),
        KeyCode::Char('o') if ctrl => return open_picker(app),
        KeyCode::Tab => return cycle(app, false),
        KeyCode::BackTab => return cycle(app, true),
        _ => {}
    }
    if app.tab == Tab::Ask && app.ask.offer.is_some() {
        return offer_key(app, code);
    }
    if app.tab == Tab::Ask {
        return ask_key(app, code);
    }
    match code {
        KeyCode::Char('q') => app.leave(Exit::Quit),
        KeyCode::Esc if app.tab != Tab::Home => app.go(Tab::Home),
        KeyCode::Esc => app.leave(Exit::Quit),
        KeyCode::Char('o') => open_picker(app),
        KeyCode::Char('s') => app.leave(Exit::Setup),
        KeyCode::Char('h') => app.go(Tab::Health),
        KeyCode::Char('c') => app.go(Tab::Corpus),
        KeyCode::Char('a') => app.go(Tab::Ask),
        KeyCode::Char('r') if app.tab == Tab::Health => app.health.refresh(),
        KeyCode::Char(d @ '1'..='9') if app.tab == Tab::Home => {
            let i = d as usize - '1' as usize;
            if let Some((p, _)) = app.files.get(i).cloned() {
                app.open(p, Tab::Corpus);
            }
        }
        // the first-run screen's first action: the sample, ready to ask.
        KeyCode::Enter if app.tab == Tab::Home && app.first_run => {
            if let Some(p) = app.sample.clone() {
                app.open(p, Tab::Ask);
            }
        }
        KeyCode::Down | KeyCode::Char('j') => scroll(app, 1, true),
        KeyCode::Up | KeyCode::Char('k') => scroll(app, 1, false),
        _ => {}
    }
}

/// The install panel takes the keys while it is open: y installs (the
/// download consent), r is the separate consent for repo code, n or esc
/// declines. a running install is not cancelled from here.
fn offer_key(app: &mut App, code: KeyCode) {
    let Some(o) = app.ask.offer.as_mut() else {
        return;
    };
    match code {
        KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => o.accept(),
        KeyCode::Char('r') | KeyCode::Char('R') => o.toggle_remote_code(),
        KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc if !o.running() => {
            app.ask.offer = None;
            app.toasts
                .push("not installed; the query needs it", Kind::Warn);
        }
        _ => {}
    }
}

fn ask_key(app: &mut App, code: KeyCode) {
    let input = &mut app.ask.input;
    match code {
        KeyCode::Enter => match &app.corpus {
            Some(c) => app.ask.submit(c.path.clone()),
            None => app.toasts.push("open a corpus first: ctrl+o", Kind::Warn),
        },
        KeyCode::Esc if input.value().is_empty() => app.go(Tab::Home),
        KeyCode::Esc => input.set_value(String::new()),
        KeyCode::Char(c) => input.insert_char(c),
        KeyCode::Backspace => input.delete_before(),
        KeyCode::Delete => input.delete_at(),
        KeyCode::Left => input.move_left(),
        KeyCode::Right => input.move_right(),
        KeyCode::Home => input.home(),
        KeyCode::End => input.end(),
        KeyCode::Up => {
            app.ask.sel = app.ask.sel.saturating_sub(1);
            app.ask.scroll = 0;
        }
        KeyCode::Down => {
            app.ask.sel = (app.ask.sel + 1).min(app.ask.answers.len().saturating_sub(1));
            app.ask.scroll = 0;
        }
        KeyCode::PageDown => scroll(app, 8, true),
        KeyCode::PageUp => scroll(app, 8, false),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::app::Launch;
    use crossterm::event::{KeyEvent, MouseButton, MouseEvent};
    use ratatui::layout::Rect;

    fn press(app: &mut App, code: KeyCode) {
        route(app, Event::Key(KeyEvent::new(code, KeyModifiers::NONE)));
    }

    #[test]
    fn tab_cycles_and_letters_jump() {
        let mut app = App::new(Launch::First, None);
        press(&mut app, KeyCode::Tab);
        assert_eq!(app.tab, Tab::Corpus);
        press(&mut app, KeyCode::BackTab);
        assert_eq!(app.tab, Tab::Home);
        press(&mut app, KeyCode::Char('h'));
        assert_eq!(app.tab, Tab::Health);
    }

    #[test]
    fn the_ask_tab_keeps_letters_as_text() {
        let mut app = App::new(Launch::First, None);
        press(&mut app, KeyCode::Char('a'));
        for c in "quit".chars() {
            press(&mut app, KeyCode::Char(c));
        }
        assert_eq!(app.ask.input.value(), "quit");
        assert!(app.leaving.is_none());
        press(&mut app, KeyCode::Esc);
        assert_eq!(app.ask.input.value(), "");
        press(&mut app, KeyCode::Esc);
        assert_eq!(app.tab, Tab::Home);
    }

    #[test]
    fn clicking_a_tab_switches_to_it() {
        let mut app = App::new(Launch::First, None);
        app.tab_hits = vec![(Tab::Health, Rect::new(40, 0, 10, 1))];
        let click = MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 44,
            row: 0,
            modifiers: KeyModifiers::NONE,
        };
        route(&mut app, Event::Mouse(click));
        assert_eq!(app.tab, Tab::Health);
    }

    #[test]
    fn q_leaves_outside_the_ask_tab() {
        let mut app = App::new(Launch::First, None);
        press(&mut app, KeyCode::Char('q'));
        assert!(app.leaving.is_some());
    }
}
