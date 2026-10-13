// SPDX-FileCopyrightText: 2026 Brenner Cruvinel, V. Hoff Research
//
// SPDX-License-Identifier: MIT

//! The plan: a checkbox per step with what it will do (or why it cannot),
//! and where every byte goes.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Style;

use crate::tui::hud::{self};
use crate::tui::pal;
use crate::tui::setup::scan::tilde;
use crate::tui::setup::ui::Ui;

pub fn draw(buf: &mut Buffer, body: Rect, ui: &mut Ui) {
    let split = body.width >= 90;
    let lw = if split {
        body.width * 3 / 5
    } else {
        body.width
    };
    let left = Rect::new(body.x + 1, body.y, lw.saturating_sub(2), body.height);
    let n_on = ui.plan.iter().filter(|i| i.runs()).count();
    let inner = hud::panel(
        buf,
        left,
        "plan",
        &format!("{n_on} of {} steps", ui.plan.len()),
        true,
    );
    // four rows a step (title, two of detail, a gap); without the gap when
    // that does not fit, and one row of detail on a short terminal, so the
    // last step still shows.
    let n = ui.plan.len();
    let per: u16 = match inner.height as usize {
        h if h >= n * 4 => 4,
        h if h >= n * 3 => 3,
        _ => 2,
    };
    let room = per.min(3) as usize - 1;
    for (i, item) in ui.plan.iter().enumerate() {
        let y = inner.y + i as u16 * per;
        if y + 1 >= inner.bottom() {
            break;
        }
        let blocked = item.blocked.is_some();
        let state = hud::Check {
            checked: item.runs(),
            locked: item.locked,
            cursor: i == ui.cursor,
        };
        hud::checkbox(
            buf,
            Rect::new(inner.x, y, inner.width, 1),
            item.task.title(),
            state,
        );
        let (note, st) = match &item.blocked {
            Some(why) => (why.as_str(), pal::err()),
            None => (item.detail.as_str(), pal::faint()),
        };
        let w = inner.width.saturating_sub(6);
        for (j, line) in hud::wrap(note, w as usize).iter().take(room).enumerate() {
            hud::put(buf, inner.x + 6, y + 1 + j as u16, line, st, w);
        }
        if blocked {
            hud::put(
                buf,
                inner.right().saturating_sub(8),
                y,
                "blocked",
                pal::err(),
                8,
            );
        }
    }
    if split {
        let right = Rect::new(body.x + lw, body.y, body.width - lw - 1, body.height);
        let inner = hud::panel(buf, right, "where it goes", "", false);
        let home = ui
            .scan
            .as_ref()
            .and_then(|s| s.home.clone())
            .map(|h| tilde(&h))
            .unwrap_or_default();
        let tree = ui
            .scan
            .as_ref()
            .and_then(|s| crate::tui::setup::plan::from_tree(s, &ui.opts));
        let source = match tree {
            Some(t) => format!("this build's checkout, {}", tilde(t)),
            None => format!(
                "github release v{}",
                ui.opts
                    .version
                    .as_deref()
                    .unwrap_or(env!("CARGO_PKG_VERSION"))
                    .trim_start_matches('v')
            ),
        };
        let rows = [
            ("payload", format!("{home}/python")),
            ("python env", format!("{home}/venv")),
            ("models", "the shared hugging face cache".into()),
            ("sample", format!("{home}/sample")),
            ("source", source),
            (
                "network",
                "curl; the hf hub for the sample and picked models".into(),
            ),
            ("binary", "left to its package manager".into()),
        ];
        for (i, (k, v)) in rows.iter().enumerate() {
            let y = inner.y + i as u16 * 2;
            if y >= inner.bottom() {
                break;
            }
            hud::put(buf, inner.x, y, k, pal::dim(), inner.width);
            let w = inner.width.saturating_sub(2);
            hud::put(
                buf,
                inner.x + 2,
                y + 1,
                &hud::tail(v, w as usize),
                Style::new().fg(pal::INK),
                w,
            );
        }
    }
}
