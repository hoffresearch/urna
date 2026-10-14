//! The marked rows: the checkbox (symbol set and label-right layout of
//! tui-checkbox 0.4, sorinirimies, MIT, carried here because it is still on
//! ratatui 0.29), noble's boot badges, and the doctor rows built from both.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};

use super::hud::{leader, put, spans};
use super::pal;

/// The state of one checkbox row.
#[derive(Clone, Copy, Debug, Default)]
pub struct Check {
    pub checked: bool,
    /// a box the user cannot change (a step that always runs).
    pub locked: bool,
    /// the row under the cursor.
    pub cursor: bool,
}

/// A checkbox row: `[x] label` with the tui-checkbox symbols, drawn on the
/// first line of `row`.
pub fn checkbox(buf: &mut Buffer, row: Rect, label: &str, state: Check) {
    let Check {
        checked,
        locked,
        cursor,
    } = state;
    let (x, y, w) = (row.x, row.y, row.width);
    let mark = match (checked, locked) {
        (_, true) => "[*]",
        (true, false) => "[x]",
        (false, false) => "[ ]",
    };
    let box_st = if locked {
        pal::faint()
    } else if checked {
        pal::ok().add_modifier(Modifier::BOLD)
    } else {
        pal::dim()
    };
    let label_st = if cursor {
        Style::new().fg(pal::INK_HI).add_modifier(Modifier::BOLD)
    } else {
        pal::text()
    };
    let pointer = if cursor { "›" } else { " " };
    spans(
        buf,
        x,
        y,
        &[
            (pointer, pal::key()),
            (" ", label_st),
            (mark, box_st),
            (" ", label_st),
            (label, label_st),
        ],
        w,
    );
    if cursor {
        for i in 0..w {
            if let Some(c) = buf.cell_mut((x + i, y)) {
                c.set_bg(pal::SURFACE);
            }
        }
    }
}

/// The doctor checks as badge + leader rows, one per line inside `inner`.
pub fn health_rows(buf: &mut Buffer, inner: Rect, rows: &[crate::cmd::health::Row]) {
    use crate::cmd::health::Level;
    for (i, r) in rows.iter().enumerate() {
        let y = inner.y + i as u16;
        if y >= inner.bottom() {
            break;
        }
        let (badge, vst) = match r.level {
            Level::Ok => (Badge::Ok, pal::dim()),
            Level::Warn => (Badge::Warn, pal::warn()),
            Level::Fail(_) => (Badge::Fail, pal::err()),
        };
        let (b, bst) = self::badge(badge);
        put(buf, inner.x, y, b, bst, 6);
        leader(
            buf,
            inner.x + 7,
            y,
            inner.width.saturating_sub(7),
            r.what,
            &super::txt::home(&r.detail),
            vst,
        );
    }
}

/// Badge in the noble boot style: `[ ok ]`, `[ .. ]`, `[ !! ]`.
pub fn badge(state: Badge) -> (&'static str, Style) {
    match state {
        Badge::Ok => ("[ ok ]", pal::ok().add_modifier(Modifier::BOLD)),
        Badge::Busy => ("[ .. ]", pal::accent()),
        Badge::Warn => ("[ -- ]", pal::warn()),
        Badge::Fail => ("[ !! ]", pal::err().add_modifier(Modifier::BOLD)),
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Badge {
    Ok,
    Busy,
    Warn,
    Fail,
}
