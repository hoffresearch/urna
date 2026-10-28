//! A clickable link (OSC 8) that keeps the row aligned: the whole escape
//! sequence goes in the first cell, the cells under the rest of the label
//! are skipped, and the first cell's width is forced to the label's so
//! ratatui's diff does not count the escape bytes. Old windows consoles
//! print OSC 8 raw, so there the label is plain text.

use ratatui::buffer::{Buffer, CellDiffOption};
use ratatui::style::{Modifier, Style};

use crate::tui::{hud, pal};

pub fn link(buf: &mut Buffer, x: u16, y: u16, label: &str, url: &str, max: u16) -> u16 {
    let w = (label.chars().count() as u16).min(max);
    if w == 0 || !buf.area.contains((x, y).into()) {
        return x;
    }
    let style = Style::new()
        .fg(pal::GLOW)
        .add_modifier(Modifier::UNDERLINED);
    let supported = !cfg!(windows) || std::env::var_os("WT_SESSION").is_some();
    if !supported {
        return hud::put(buf, x, y, label, style, max);
    }
    let shown: String = label.chars().take(usize::from(w)).collect();
    let end = (x + w).min(buf.area.right());
    let Some(width) = std::num::NonZeroU16::new(end.saturating_sub(x)) else {
        return x;
    };
    if let Some(cell) = buf.cell_mut((x, y)) {
        cell.set_symbol(&format!("\x1b]8;;{url}\x1b\\{shown}\x1b]8;;\x1b\\"));
        cell.set_style(style);
        cell.set_diff_option(CellDiffOption::ForcedWidth(width));
    }
    for cx in x + 1..end {
        if let Some(cell) = buf.cell_mut((cx, y)) {
            cell.set_diff_option(CellDiffOption::Skip);
        }
    }
    end
}
