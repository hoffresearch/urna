//! Drawing the installer: the frame (header with the stepper, footer with
//! the keys) around one screen per step. every frame ends the same way:
//! effects over the finished buffer, then the palette folded to the
//! terminal's color depth.

mod done;
mod models;
mod plan;
mod run;
mod scan;
mod splash;

use std::time::Duration;

use ratatui::Frame;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::widgets::Widget;
use ratatui_cheese::help::{Binding, Help, HelpStyles};

use super::ui::{Step, Ui};
use crate::tui::hud::{self};
use crate::tui::{art, fx, pal};

pub fn draw(f: &mut Frame<'_>, ui: &mut Ui, dt: Duration) {
    let area = f.area();
    let buf = f.buffer_mut();
    pal::ground(buf, area);
    if area.height < 6 || area.width < 30 {
        hud::put(
            buf,
            area.x,
            area.y,
            "urna setup: terminal too small",
            pal::err(),
            area.width,
        );
        return;
    }
    let mark = header(buf, area, ui);
    let top = header_rows(area);
    let body = Rect::new(area.x, area.y + top, area.width, area.height - top - 1);
    footer(
        buf,
        Rect::new(area.x, area.bottom() - 1, area.width, 1),
        ui.step,
    );
    match ui.step {
        Step::Splash => splash::draw(buf, body, ui),
        Step::Scan => scan::draw(buf, body, ui),
        Step::Plan => plan::draw(buf, body, ui),
        Step::Models => models::draw(buf, body, ui),
        Step::Run => run::draw(buf, body, ui),
        Step::Done => done::draw(buf, body, ui),
    }
    if ui.entered {
        ui.entered = false;
        if let Some(r) = mark {
            ui.fx.add_unique_effect("mark", fx::shimmer(r));
        }
        if ui.step != Step::Splash {
            ui.fx.add_unique_effect("page", fx::page_in(body));
        }
        // the symbol's breathing belongs to the screens that draw it.
        if matches!(ui.step, Step::Plan | Step::Models | Step::Run) {
            ui.fx.cancel_unique_effect("shimmer");
        }
    }
    ui.fx.process_effects(dt.into(), buf, area);
    pal::fit(buf, area, ui.depth);
}

/// Rows the header takes: the three-row mark when the viewport can spare
/// them, a single plain row on a short terminal.
fn header_rows(area: Rect) -> u16 {
    if area.height >= 20 { 3 } else { 1 }
}

/// The header: the mark, the title and the version, the stepper, a rule.
/// Returns the mark's rect (the frame keeps it breathing).
fn header(buf: &mut Buffer, area: Rect, ui: &Ui) -> Option<Rect> {
    let v = format!("v{} ", crate::cmd::buildid::NAME);
    let vx = area.right().saturating_sub(v.len() as u16);
    if header_rows(area) == 1 {
        let x = hud::put(
            buf,
            area.x + 1,
            area.y,
            "urna setup",
            pal::title(),
            area.width,
        );
        let end = stepper(buf, x + 4, area.y, area.right(), ui.step);
        if vx > end + 1 {
            hud::put(buf, vx, area.y, &v, pal::faint(), v.len() as u16);
        }
        return None;
    }
    let mark = art::mark(buf, area.x + 2, area.y);
    let tx = mark.right() + 2;
    hud::put(buf, tx, area.y, "urna setup", pal::title(), area.width);
    hud::put(buf, vx, area.y, &v, pal::faint(), v.len() as u16);
    stepper(buf, tx, area.y + 1, area.right(), ui.step);
    for x in tx..area.right().saturating_sub(1) {
        hud::put(buf, x, area.y + 2, "─", Style::new().fg(pal::SURFACE), 1);
    }
    Some(mark)
}

/// `■ scan ── ● plan ── ○ install ── ○ verify` from x; returns where it ends.
fn stepper(buf: &mut Buffer, x: u16, y: u16, right: u16, step: Step) -> u16 {
    // the picker is part of planning.
    let step = if step == Step::Models {
        Step::Plan
    } else {
        step
    };
    let steps = [
        ("scan", Step::Scan),
        ("plan", Step::Plan),
        ("install", Step::Run),
        ("verify", Step::Done),
    ];
    let at = steps.iter().position(|(_, s)| *s == step);
    let mut cx = x;
    for (i, (name, _)) in steps.iter().enumerate() {
        let (dot, st) = match at {
            Some(a) if i < a => ("■ ", pal::ok()),
            Some(a) if i == a => ("● ", pal::key()),
            _ => ("○ ", pal::faint()),
        };
        let label = if Some(i) == at {
            pal::key()
        } else {
            pal::faint()
        };
        cx = hud::spans(
            buf,
            cx,
            y,
            &[(dot, st), (name, label)],
            right.saturating_sub(cx),
        );
        if i + 1 < steps.len() {
            cx = hud::put(buf, cx + 1, y, "──", Style::new().fg(pal::LINE), 2) + 1;
        }
    }
    cx
}

fn footer(buf: &mut Buffer, area: Rect, step: Step) {
    let b = |k: &str, d: &str| Binding::new(k, d);
    let keys = match step {
        Step::Splash => vec![b("enter", "begin"), b("q", "quit")],
        Step::Scan => vec![b("enter", "skip ahead"), b("q", "quit")],
        Step::Plan => vec![
            b("↑↓", "move"),
            b("space", "toggle"),
            b("m", "models"),
            b("enter", "install"),
            b("q", "quit"),
        ],
        Step::Models => vec![
            b("↑↓", "move"),
            b("space", "toggle"),
            b("a", "all"),
            b("r", "allow repo code"),
            b("enter", "done"),
        ],
        Step::Run => vec![
            b("↑↓", "scroll log"),
            b("end", "follow"),
            b("ctrl+c", "abort"),
        ],
        Step::Done => vec![b("enter", "finish"), b("m", "add models")],
    };
    let styles = HelpStyles {
        ellipsis: pal::faint(),
        short_key: pal::key(),
        short_desc: pal::faint(),
        short_separator: Style::new().fg(pal::LINE),
        full_key: pal::key(),
        full_desc: pal::faint(),
        full_separator: Style::new().fg(pal::LINE),
    };
    let help = Help::default()
        .bindings(keys)
        .styles(styles)
        .short_separator(" · ");
    (&help).render(
        Rect::new(area.x + 1, area.y, area.width.saturating_sub(2), 1),
        buf,
    );
}
