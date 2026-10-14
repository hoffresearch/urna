//! Drawing primitives shared by the installer and the explorer: the
//! rounded panel, the thin gauge, the dotted leader row and the checkbox.
//! The panel, the gauge and the leader are adapted from noble's hud
//! (MertSoylu/noble, src/ui/hud.rs, MIT); the checkbox keeps the symbol
//! set and the label-right layout of tui-checkbox 0.4 (sorinirimies, MIT),
//! which is still on ratatui 0.29 and so is carried here instead of pulled
//! in. every primitive is bounds-safe: nothing panics in a small window.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};

use super::pal;

pub use super::row::{Badge, Check, badge, checkbox, health_rows};
pub use super::txt::{tail, wrap};

/// Writes `s` at (x, y), at most `max` columns; returns the x after it.
pub fn put(buf: &mut Buffer, x: u16, y: u16, s: &str, style: Style, max: u16) -> u16 {
    let a = buf.area;
    if y < a.top() || y >= a.bottom() || x < a.left() || x >= a.right() || max == 0 {
        return x;
    }
    let limit = max.min(a.right() - x);
    buf.set_stringn(x, y, s, limit as usize, style).0
}

/// Writes spans one after another inside `max` columns.
pub fn spans(buf: &mut Buffer, x: u16, y: u16, parts: &[(&str, Style)], max: u16) -> u16 {
    let end = x.saturating_add(max);
    let mut cx = x;
    for (s, st) in parts {
        if cx >= end {
            break;
        }
        cx = put(buf, cx, y, s, *st, end - cx);
    }
    cx
}

/// Rounded panel with a title on the left and a tag on the right; returns
/// the inner area. `focused` lights the border in the glow color.
pub fn panel(buf: &mut Buffer, area: Rect, title: &str, tag: &str, focused: bool) -> Rect {
    if area.width < 4 || area.height < 2 {
        return Rect::new(area.x, area.y, 0, 0);
    }
    let line = Style::new().fg(if focused { pal::GLOW } else { pal::LINE });
    let (l, r, t, b) = (area.left(), area.right() - 1, area.top(), area.bottom() - 1);
    for x in l + 1..r {
        put(buf, x, t, "─", line, 1);
        put(buf, x, b, "─", line, 1);
    }
    for y in t + 1..b {
        put(buf, l, y, "│", line, 1);
        put(buf, r, y, "│", line, 1);
    }
    put(buf, l, t, "╭", line, 1);
    put(buf, r, t, "╮", line, 1);
    put(buf, l, b, "╰", line, 1);
    put(buf, r, b, "╯", line, 1);
    let room = area.width.saturating_sub(6);
    if !title.is_empty() && room > 2 {
        let st = if focused {
            pal::title()
        } else {
            Style::new().fg(pal::INK).add_modifier(Modifier::BOLD)
        };
        let x = put(buf, l + 2, t, &format!(" {title} "), st, room);
        let used = x - l;
        if !tag.is_empty() && area.width > used + tag.chars().count() as u16 + 5 {
            let w = tag.chars().count() as u16 + 2;
            put(buf, r - 1 - w, t, &format!(" {tag} "), pal::faint(), w);
        }
    }
    Rect::new(
        area.x + 2,
        area.y + 1,
        area.width.saturating_sub(4),
        area.height.saturating_sub(2),
    )
}

/// Thin gauge: filled `━` in `color` (half cells as `╸`), the rest of the
/// track in the line color.
pub fn gauge(buf: &mut Buffer, x: u16, y: u16, w: u16, ratio: f64, color: Color) {
    let exact = ratio.clamp(0.0, 1.0) * w as f64;
    let full = exact.floor() as u16;
    let half = exact - full as f64 >= 0.5;
    for i in 0..w {
        let (ch, st) = if i < full {
            ("━", Style::new().fg(color))
        } else if i == full && half {
            ("╸", Style::new().fg(color))
        } else {
            ("━", Style::new().fg(pal::SURFACE))
        };
        put(buf, x + i, y, ch, st, 1);
    }
}

/// An indeterminate gauge: a lit segment that travels the track.
pub fn pulse(buf: &mut Buffer, x: u16, y: u16, w: u16, t: f64, color: Color) {
    let seg = (w / 5).max(3);
    let span = (w + seg) as f64;
    let head = ((t * 0.9).fract() * span) as i32 - seg as i32;
    for i in 0..w {
        let d = i as i32 - head;
        let lit = (0..seg as i32).contains(&d);
        let st = if lit {
            Style::new().fg(pal::mix(
                pal::SURFACE,
                color,
                1.0 - d as f32 / seg as f32 * 0.6,
            ))
        } else {
            Style::new().fg(pal::SURFACE)
        };
        put(buf, x + i, y, "━", st, 1);
    }
}

/// `label ......... value`, noble's boot checklist row, inside `w` columns.
pub fn leader(buf: &mut Buffer, x: u16, y: u16, w: u16, label: &str, value: &str, vstyle: Style) {
    let lw = label.chars().count() as u16;
    let vw = (value.chars().count() as u16).min(w.saturating_sub(lw + 3));
    let after = put(buf, x, y, label, pal::text(), w);
    let dots = w.saturating_sub(lw + vw + 2);
    put(
        buf,
        after + 1,
        y,
        &".".repeat(dots as usize),
        Style::new().fg(pal::LINE),
        dots,
    );
    let vx = x + w.saturating_sub(vw);
    let shown = tail(value, vw as usize);
    put(buf, vx, y, &shown, vstyle, vw);
}

/// A clickable link (OSC 8, via hyperrat) that keeps the row aligned.
/// hyperrat puts the whole escape sequence in the first cell and leaves its
/// diff width to ratatui, which measures the escape bytes and then skips
/// that many cells; forcing the width to the label's fixes the row. old
/// windows consoles print OSC 8 raw, so there it degrades to plain text.
pub fn link(buf: &mut Buffer, x: u16, y: u16, label: &str, url: &str, max: u16) -> u16 {
    use ratatui::buffer::CellDiffOption;
    use ratatui::widgets::Widget;
    let w = (label.chars().count() as u16).min(max);
    if w == 0 {
        return x;
    }
    let supported = !cfg!(windows) || std::env::var_os("WT_SESSION").is_some();
    hyperrat::Link::new(label, url)
        .style(
            Style::new()
                .fg(pal::GLOW)
                .add_modifier(Modifier::UNDERLINED),
        )
        .enabled(supported)
        .render(Rect::new(x, y, w, 1).intersection(buf.area), buf);
    if supported
        && let Some(width) = std::num::NonZeroU16::new(w)
        && let Some(cell) = buf.cell_mut((x, y))
    {
        cell.set_diff_option(CellDiffOption::ForcedWidth(width));
    }
    x + w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitives_survive_tiny_and_offscreen_areas() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 3, 1));
        panel(&mut buf, Rect::new(0, 0, 3, 1), "t", "tag", true);
        gauge(&mut buf, 0, 0, 10, 0.5, pal::HIGH);
        pulse(&mut buf, 0, 0, 10, 0.3, pal::GLOW);
        leader(&mut buf, 0, 0, 3, "label", "value", pal::dim());
        let on = Check {
            checked: true,
            locked: false,
            cursor: true,
        };
        checkbox(&mut buf, Rect::new(0, 5, 10, 1), "x", on);
        put(&mut buf, 9, 9, "far", pal::dim(), 3);
    }

    #[test]
    fn gauge_fills_proportionally() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 10, 1));
        gauge(&mut buf, 0, 0, 10, 0.3, pal::HIGH);
        assert_eq!(buf[(2, 0)].fg, pal::HIGH);
        assert_eq!(buf[(3, 0)].fg, pal::SURFACE);
    }

    #[test]
    fn panel_returns_the_padded_inner_area() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 20, 6));
        let inner = panel(&mut buf, Rect::new(0, 0, 20, 6), "plan", "3 steps", false);
        assert_eq!(inner, Rect::new(2, 1, 16, 4));
        assert_eq!(buf[(0, 0)].symbol(), "╭");
    }
}
