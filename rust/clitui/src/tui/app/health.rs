//! The health tab: the doctor checks, run off the ui thread on open and on
//! `r`, drawn with the same rows the installer ends on.

use std::sync::mpsc::{Receiver, channel};

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Modifier;

use crate::cmd::health::{self, Row};
use crate::tui::{hud, pal};

#[derive(Default)]
pub struct Health {
    pub rows: Vec<Row>,
    rx: Option<Receiver<Vec<Row>>>,
}

impl Health {
    pub fn busy(&self) -> bool {
        self.rx.is_some()
    }

    pub fn refresh(&mut self) {
        if self.busy() {
            return;
        }
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let _ = tx.send(health::collect());
        });
        self.rx = Some(rx);
    }

    /// True when a run just finished.
    pub fn poll(&mut self) -> bool {
        let Some(rows) = self.rx.as_ref().and_then(|rx| rx.try_recv().ok()) else {
            return false;
        };
        self.rows = rows;
        self.rx = None;
        true
    }
}

pub fn render(buf: &mut Buffer, body: Rect, h: &Health, spinner: &str) {
    let w = body.width.saturating_sub(4).min(110);
    let x = body.x + (body.width - w) / 2;
    let rows = h.rows.len().max(1) as u16;
    let prect = Rect::new(
        x,
        body.y + 1,
        w,
        (rows + 2).min(body.height.saturating_sub(4)),
    );
    let code = health::exit_code(&h.rows);
    let tag = if h.busy() {
        format!("{spinner} checking")
    } else if h.rows.is_empty() {
        String::new()
    } else if code == 0 {
        "every check passes".into()
    } else {
        format!("exit {code}")
    };
    let inner = hud::panel(buf, prect, "doctor", &tag, true);
    if h.rows.is_empty() {
        hud::put(
            buf,
            inner.x,
            inner.y,
            &format!("{spinner} running the checks"),
            pal::dim(),
            inner.width,
        );
        return;
    }
    hud::health_rows(buf, inner, &h.rows);
    let y = prect.bottom() + 1;
    let (msg, st) = if code == 0 {
        (
            "the offline embedder answers; ask and retrieve will work",
            pal::ok(),
        )
    } else {
        (
            "press s to run setup: it fixes the payload and the python env",
            pal::warn(),
        )
    };
    hud::put(buf, x + 1, y, msg, st.add_modifier(Modifier::BOLD), w);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refresh_runs_the_checks_once_and_reports_them() {
        let mut h = Health::default();
        h.refresh();
        h.refresh();
        for _ in 0..600 {
            if h.poll() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(!h.busy());
        assert_eq!(h.rows[0].what, "version");
        let area = Rect::new(0, 0, 90, 20);
        let mut buf = Buffer::empty(area);
        render(&mut buf, area, &h, "·");
    }
}
