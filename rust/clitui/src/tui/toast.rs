//! Toast notifications, stacked at the top right. Adapted from
//! ratatui-toaster 0.1.4 (JayanAXHF, MIT OR Unlicense): the same look, a
//! padded message between quadrant bars in the kind's color. changes: the
//! site palette instead of the terminal's named colors, a stack instead of
//! a single slot, and expiry by wall clock (the original needs tokio for
//! timing). the ui slides each new toast in with `fx::toast_in`.

use std::time::{Duration, Instant};

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::symbols;
use ratatui::widgets::{Block, Borders, Clear, Padding, Paragraph, Widget, Wrap};

use super::pal;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Info,
    Ok,
    Warn,
    Err,
}

impl Kind {
    fn color(self) -> Color {
        match self {
            Kind::Info => pal::GLOW,
            Kind::Ok => pal::HIGH,
            Kind::Warn => pal::MID,
            Kind::Err => pal::LOW,
        }
    }
}

struct Toast {
    msg: String,
    kind: Kind,
    born: Instant,
    shown: bool,
}

pub struct Toasts {
    list: Vec<Toast>,
    ttl: Duration,
}

impl Default for Toasts {
    fn default() -> Self {
        Self {
            list: Vec::new(),
            ttl: Duration::from_millis(3200),
        }
    }
}

impl Toasts {
    pub fn push(&mut self, msg: impl Into<String>, kind: Kind) {
        self.list.push(Toast {
            msg: msg.into(),
            kind,
            born: Instant::now(),
            shown: false,
        });
        // errors linger longer; never more than four on screen.
        if self.list.len() > 4 {
            self.list.remove(0);
        }
    }

    /// Draws the stack in the top right of `area`; returns the rects of the
    /// toasts drawn for the first time (the caller animates those).
    pub fn render(&mut self, buf: &mut Buffer, area: Rect) -> Vec<Rect> {
        let ttl = self.ttl;
        self.list.retain(|t| {
            let life = if t.kind == Kind::Err { ttl * 2 } else { ttl };
            t.born.elapsed() < life
        });
        let width = (area.width / 2).clamp(24.min(area.width), 52);
        let mut y = area.y + 1;
        let mut fresh = Vec::new();
        for t in &mut self.list {
            let inner_w = width.saturating_sub(4).max(1) as usize;
            let lines = super::txt::wrap(&t.msg, inner_w).len().max(1) as u16;
            let h = lines + 2;
            if y + h > area.bottom() {
                break;
            }
            let rect = Rect::new(area.right().saturating_sub(width + 1), y, width, h);
            Clear.render(rect, buf);
            Paragraph::new(t.msg.as_str())
                .wrap(Wrap { trim: true })
                .style(Style::new().fg(pal::INK).bg(pal::RAISED))
                .block(
                    Block::default()
                        .borders(Borders::LEFT | Borders::RIGHT)
                        .border_set(symbols::border::QUADRANT_OUTSIDE)
                        .border_style(Style::new().fg(t.kind.color()).bg(pal::BG))
                        .padding(Padding::uniform(1)),
                )
                .render(rect, buf);
            if !t.shown {
                t.shown = true;
                fresh.push(rect);
            }
            y += h + 1;
        }
        fresh
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_toast_is_reported_once() {
        let area = Rect::new(0, 0, 80, 20);
        let mut buf = Buffer::empty(area);
        let mut t = Toasts::default();
        t.push("opened corpus.urna", Kind::Ok);
        assert_eq!(t.render(&mut buf, area).len(), 1);
        assert!(t.render(&mut buf, area).is_empty());
    }

    #[test]
    fn the_stack_is_capped_and_clipped() {
        let area = Rect::new(0, 0, 40, 6);
        let mut buf = Buffer::empty(area);
        let mut t = Toasts::default();
        for i in 0..9 {
            t.push(format!("toast {i}"), Kind::Info);
        }
        // four kept, and only what fits in six rows is drawn.
        assert!(t.render(&mut buf, area).len() <= 2);
    }
}
