// SPDX-FileCopyrightText: 2026 Brenner Cruvinel, V. Hoff Research
//
// SPDX-License-Identifier: MIT

//! The explorer's frame: the header (the mark, the tabs as pills, the open
//! corpus) and the footer (the keys for the current tab, ratatui-cheese's
//! help). the pill shape is two half blocks around a raised label, the
//! same shape comfy-tabs draws, done with plain glyphs so no font needs
//! powerline symbols.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::widgets::Widget;
use ratatui_cheese::help::{Binding, Help, HelpStyles};

use super::Tab;
use crate::tui::{art, hud, pal};

pub const TABS: [(Tab, &str); 4] = [
    (Tab::Home, "home"),
    (Tab::Corpus, "corpus"),
    (Tab::Ask, "ask"),
    (Tab::Health, "health"),
];

/// Rows the header takes (the three-row mark).
pub const HEADER_ROWS: u16 = 3;

/// Draws the header: the mark, `urna` and the open corpus on the first
/// row, the tabs as pills on the second, a rule on the third. Returns the
/// rect of each tab (mouse hit-testing) and the mark's rect.
pub fn header(buf: &mut Buffer, area: Rect, active: Tab, right: &str) -> (Vec<(Tab, Rect)>, Rect) {
    let mark = art::mark(buf, area.x + 2, area.y);
    let tx = mark.right() + 2;
    let title_end = hud::put(buf, tx, area.y, "urna", pal::title(), area.width);
    let rw = right.chars().count() as u16;
    if area.right() > title_end + rw + 3 {
        hud::put(buf, area.right() - rw - 2, area.y, right, pal::dim(), rw);
    }
    let y = area.y + 1;
    let mut x = tx.saturating_sub(1);
    let mut hits = Vec::new();
    for (tab, name) in TABS.iter() {
        let on = *tab == active;
        let label = format!(" {name} ");
        let w = label.chars().count() as u16 + 2;
        if x + w >= area.right() {
            break;
        }
        let rect = Rect::new(x, y, w, 1);
        if on {
            let pill = Style::new()
                .fg(pal::INK_HI)
                .bg(pal::RAISED)
                .add_modifier(Modifier::BOLD);
            hud::put(buf, x, y, "▐", Style::new().fg(pal::RAISED), 1);
            hud::put(buf, x + 1, y, &label, pill, w);
            hud::put(buf, x + w - 1, y, "▌", Style::new().fg(pal::RAISED), 1);
        } else {
            hud::put(buf, x + 1, y, &label, pal::faint(), w);
        }
        hits.push((*tab, rect));
        x += w + 1;
    }
    for cx in tx..area.right().saturating_sub(1) {
        hud::put(buf, cx, area.y + 2, "─", Style::new().fg(pal::SURFACE), 1);
    }
    (hits, mark)
}

/// `first_run` is `Some(sample is installed)` while home is the first-run
/// screen.
pub fn footer(
    buf: &mut Buffer,
    area: Rect,
    tab: Tab,
    has_corpus: bool,
    picking: bool,
    first_run: Option<bool>,
) {
    let b = |k: &str, d: &str| Binding::new(k, d);
    let mut keys = match tab {
        _ if picking => vec![
            b("↑↓", "move"),
            b("enter", "open"),
            b("←", "up a dir"),
            b("ctrl+h", "hidden"),
            b("esc", "close"),
        ],
        Tab::Home => match first_run {
            Some(true) => vec![b("enter", "the sample"), b("o", "open"), b("s", "setup")],
            Some(false) => vec![b("s", "get the sample"), b("o", "open")],
            None => vec![b("o", "open"), b("1-9", "recent"), b("s", "setup")],
        },
        Tab::Corpus => vec![b("o", "open"), b("↑↓", "scroll"), b("a", "ask it")],
        Tab::Ask => vec![
            b("enter", "ask"),
            b("↑↓", "hit"),
            b("pgup/pgdn", "text"),
            b("esc", "clear"),
        ],
        Tab::Health => vec![b("r", "re-run"), b("s", "setup")],
    };
    if !picking {
        keys.push(b("tab", "next tab"));
        keys.push(if tab == Tab::Ask {
            b("ctrl+q", "quit")
        } else {
            b("q", "quit")
        });
        if !has_corpus && tab == Tab::Ask {
            keys.insert(0, b("ctrl+o", "open a corpus first"));
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_reports_a_hit_rect_per_visible_tab() {
        let area = Rect::new(0, 0, 100, 3);
        let mut buf = Buffer::empty(area);
        let (hits, mark) = header(&mut buf, area, Tab::Ask, "corpus.urna · 1.3 KB");
        assert_eq!(hits.len(), 4);
        assert_eq!(mark.height, HEADER_ROWS);
        assert!(hits.iter().all(|(_, r)| r.y == 1 && r.x > mark.right()));
        // a narrow terminal drops tabs instead of overflowing.
        let narrow = Rect::new(0, 0, 30, 3);
        let mut nb = Buffer::empty(narrow);
        assert!(header(&mut nb, narrow, Tab::Home, "x").0.len() < 4);
    }
}
