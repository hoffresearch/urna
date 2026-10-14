//! The install: a row per step (spinner or badge, the live note, bytes
//! for the download) over a gauge, and the log of what the tools print.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::widgets::Widget;
use tui_scrollbar::{ScrollBar, ScrollLengths};

use crate::tui::hud::{self, Badge};
use crate::tui::setup::ui::Ui;
use crate::tui::{fx, pal};

fn row_rect(inner: Rect, i: usize) -> Rect {
    Rect::new(inner.x, inner.y + i as u16 * 2, inner.width, 2)
}

pub fn draw(buf: &mut Buffer, body: Rect, ui: &mut Ui) {
    let h = (ui.rows.len() as u16 * 2 + 2).min(body.height.saturating_sub(4));
    let top = Rect::new(body.x + 1, body.y, body.width.saturating_sub(2), h);
    let done = ui.rows.iter().filter(|r| r.result.is_some()).count();
    let inner = hud::panel(
        buf,
        top,
        "installing",
        &format!("{done}/{}", ui.rows.len()),
        true,
    );
    let t = ui.clock.secs();
    for (i, r) in ui.rows.iter().enumerate() {
        let rr = row_rect(inner, i);
        if rr.y + 1 > inner.bottom() {
            break;
        }
        let (mark, mst) = match r.badge {
            Some(Badge::Busy) => (ui.spinner.frame_str().to_string(), pal::accent()),
            Some(b) => (
                hud::badge(b)
                    .0
                    .trim_matches(|c| c == '[' || c == ']' || c == ' ')
                    .to_string(),
                hud::badge(b).1,
            ),
            None => ("·".into(), pal::faint()),
        };
        let x = hud::spans(
            buf,
            rr.x,
            rr.y,
            &[
                (&mark, mst),
                (" ", pal::text()),
                (
                    r.task.title(),
                    Style::new().fg(pal::INK).add_modifier(Modifier::BOLD),
                ),
            ],
            24,
        );
        let note = r.result.as_deref().unwrap_or(&r.note);
        let nst = match r.badge {
            Some(Badge::Fail) => pal::err(),
            Some(Badge::Ok) => pal::dim(),
            _ => pal::faint(),
        };
        let right = match (r.total, r.bytes) {
            (Some(tot), n) if tot > 0 && r.result.is_none() => format!(
                "{:.1} / {:.1} MB {:>3}%",
                n as f64 / 1e6,
                tot as f64 / 1e6,
                n * 100 / tot
            ),
            (None, n) if n > 0 && r.result.is_none() => format!("{:.1} MB", n as f64 / 1e6),
            _ => String::new(),
        };
        let rw = right.chars().count() as u16;
        hud::put(
            buf,
            x.max(rr.x + 20) + 1,
            rr.y,
            note,
            nst,
            inner.right().saturating_sub(x.max(rr.x + 20) + rw + 3),
        );
        hud::put(
            buf,
            inner.right().saturating_sub(rw),
            rr.y,
            &right,
            pal::dim(),
            rw,
        );
        let gy = rr.y + 1;
        match (r.badge, r.total) {
            (Some(Badge::Ok), _) => hud::gauge(buf, rr.x, gy, rr.width, 1.0, pal::HIGH),
            (Some(Badge::Fail), _) => hud::gauge(buf, rr.x, gy, rr.width, 1.0, pal::LOW),
            (Some(Badge::Busy), Some(tot)) if tot > 0 => hud::gauge(
                buf,
                rr.x,
                gy,
                rr.width,
                r.bytes as f64 / tot as f64,
                pal::GLOW,
            ),
            (Some(Badge::Busy), _) => hud::pulse(buf, rr.x, gy, rr.width, t, pal::GLOW),
            _ => hud::gauge(buf, rr.x, gy, rr.width, 0.0, pal::GLOW),
        }
    }
    match ui.rows.iter().position(|r| r.badge == Some(Badge::Busy)) {
        Some(i) => {
            let rr = row_rect(inner, i);
            if ui.glint_on != Some(i) {
                ui.glint_on = Some(i);
                ui.fx
                    .add_unique_effect("glint", fx::glint(Rect::new(rr.x, rr.y + 1, rr.width, 1)));
            }
        }
        None if ui.glint_on.is_some() => {
            ui.glint_on = None;
            ui.fx.cancel_unique_effect("glint");
        }
        None => {}
    }
    for (task, ok) in std::mem::take(&mut ui.flashes) {
        if let Some(i) = ui.rows.iter().position(|r| r.task == task) {
            ui.fx.add_effect(fx::flash(
                row_rect(inner, i),
                if ok { pal::HIGH } else { pal::LOW },
            ));
        }
    }
    let log = Rect::new(
        body.x + 1,
        top.bottom(),
        body.width.saturating_sub(2),
        body.bottom().saturating_sub(top.bottom()),
    );
    log_panel(buf, log, ui);
}

fn log_panel(buf: &mut Buffer, area: Rect, ui: &Ui) {
    let follow = if ui.scroll.is_none() {
        "following"
    } else {
        "scrolled"
    };
    let inner = hud::panel(buf, area, "log", follow, false);
    if inner.height == 0 {
        return;
    }
    let vh = inner.height as usize;
    let len = ui.log.len();
    let end = ui
        .scroll
        .map(|s| (s + 1).min(len))
        .unwrap_or(len)
        .max(vh.min(len));
    let start = end.saturating_sub(vh);
    for (i, line) in ui.log[start..end].iter().enumerate() {
        let st = if line.starts_with("$ ") {
            pal::accent()
        } else {
            pal::faint()
        };
        hud::put(
            buf,
            inner.x,
            inner.y + i as u16,
            line,
            st,
            inner.width.saturating_sub(2),
        );
    }
    if len > vh {
        let bar = ScrollBar::vertical(ScrollLengths {
            content_len: len,
            viewport_len: vh,
        })
        .offset(start)
        .track_style(Style::new().fg(pal::SURFACE))
        .thumb_style(Style::new().fg(pal::GLOW));
        bar.render(Rect::new(inner.right(), inner.y, 1, inner.height), buf);
    }
}
