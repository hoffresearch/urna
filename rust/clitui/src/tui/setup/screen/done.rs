// SPDX-FileCopyrightText: 2026 Brenner Cruvinel, V. Hoff Research
//
// SPDX-License-Identifier: MIT

//! The verify screen: the doctor rows, the verdict, what to run next.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};

use crate::tui::hud::{self, Badge};
use crate::tui::setup::plan::{self, Task};
use crate::tui::setup::ui::Ui;
use crate::tui::{art, fx, pal};

pub fn draw(buf: &mut Buffer, body: Rect, ui: &mut Ui) {
    let wide = body.width >= 80 && body.height >= 12;
    let px = if wide {
        let sy = body.y + (body.height.saturating_sub(10)) / 2;
        art::paint(
            buf,
            body.x + 3,
            sy,
            &art::SYMBOL_M,
            pal::TITLE_FROM,
            pal::TITLE_TO,
        );
        if ui.entered {
            ui.fx
                .add_unique_effect("shimmer", fx::shimmer(Rect::new(body.x + 3, sy, 18, 10)));
        }
        body.x + 25
    } else {
        body.x + 1
    };
    let ok = ui.ready();
    let verified = ui
        .rows
        .iter()
        .any(|r| r.task == Task::Verify && r.badge == Some(Badge::Ok));
    let n = ui.health.len() as u16;
    let vrect = Rect::new(
        px,
        body.y,
        body.right().saturating_sub(px + 1),
        (n + 2).min(body.height.saturating_sub(5)),
    );
    let verdict = if verified {
        "every check passes"
    } else {
        "needs attention"
    };
    let inner = hud::panel(buf, vrect, "verify", verdict, true);
    hud::health_rows(buf, inner, &ui.health);
    let y = vrect.bottom() + 1;
    let (title, st) = if ok {
        ("urna is ready", pal::ok())
    } else {
        ("setup finished with problems", pal::err())
    };
    let status = Rect::new(px + 1, y, body.right().saturating_sub(px + 2), 1);
    hud::put(
        buf,
        status.x,
        y,
        title,
        st.add_modifier(Modifier::BOLD),
        status.width,
    );
    let mut yy = y + 2;
    // why, step by step: the worker's own error text, whole, each of its
    // lines wrapped. ready, only an optional step can have failed: its
    // error is a warning under the verdict, not a failed setup.
    let w = status.width.saturating_sub(20) as usize;
    let mut reasons: Vec<(&str, String)> = ui
        .rows
        .iter()
        .filter(|r| r.badge == Some(Badge::Fail) && r.task != Task::Verify)
        .map(|r| (r.task.title(), r.result.clone().unwrap_or_default()))
        .collect();
    reasons.extend(
        ui.plan
            .iter()
            .filter(|i| i.on && i.blocked.is_some())
            .map(|i| {
                (
                    i.task.title(),
                    format!("blocked: {}", i.blocked.as_deref().unwrap_or("")),
                )
            }),
    );
    let tone = if ok { pal::warn() } else { pal::err() };
    for (task, why) in &reasons {
        let lines = why.lines().flat_map(|l| hud::wrap(l, w));
        for (j, line) in lines.enumerate() {
            if yy >= body.bottom() {
                break;
            }
            let label = if j == 0 { *task } else { "" };
            hud::put(buf, status.x, yy, label, pal::dim(), 18);
            hud::put(buf, status.x + 19, yy, &line, tone, w as u16);
            yy += 1;
        }
    }
    if !reasons.is_empty() {
        yy += 1;
    }
    let next: Vec<(String, String)> = match (ok, &ui.sample) {
        (true, Some(p)) => next_with_sample(p),
        (true, None) => {
            let mut next = vec![
                ("urna tui".into(), "open the explorer".into()),
                ("urna ask corpus.urna \"...\"".into(), "ask a corpus".into()),
            ];
            if plan::sample_hint(&ui.plan, &ui.opts) {
                next.push((
                    "urna setup".into(),
                    "again, to download the sample corpus".into(),
                ));
            }
            next
        }
        (false, _) => vec![
            ("urna setup".into(), "run it again".into()),
            (
                "urna setup --yes > setup.log".into(),
                "the plain log, to share".into(),
            ),
            (
                "urna doctor".into(),
                "re-check without changing anything".into(),
            ),
        ],
    };
    if ok && ui.sample.is_some() && yy < body.bottom() {
        hud::put(buf, status.x, yy, "sample corpus", pal::dim(), 18);
        hud::put(
            buf,
            status.x + 19,
            yy,
            crate::tui::setup::sample::ABOUT,
            Style::new().fg(pal::INK),
            status.width.saturating_sub(19),
        );
        yy += 1;
    }
    for (cmd, what) in &next {
        if yy >= body.bottom() {
            break;
        }
        let x = hud::spans(
            buf,
            status.x,
            yy,
            &[("$ ", pal::faint()), (cmd.as_str(), pal::key())],
            status.width,
        );
        let dx = (x + 2).max(status.x + 34);
        hud::put(
            buf,
            dx,
            yy,
            what,
            pal::faint(),
            status.right().saturating_sub(dx),
        );
        yy += 1;
    }
    if yy + 1 < body.bottom() {
        let (label, url, tail) = if ok {
            (
                "urna.dev",
                "https://urna.dev",
                "· docs, the format spec, benchmarks",
            )
        } else {
            (
                "report it",
                "https://github.com/hoffresearch/urna/issues",
                "· github.com/hoffresearch/urna/issues",
            )
        };
        let x = hud::link(buf, status.x, yy + 1, label, url, status.width);
        hud::put(
            buf,
            x + 1,
            yy + 1,
            tail,
            pal::faint(),
            status.right().saturating_sub(x + 1),
        );
    }
    if ui.entered {
        ui.fx
            .add_effect(fx::flash(status, if ok { pal::HIGH } else { pal::LOW }));
    }
}

/// The sample's two commands, under the line that says what it holds:
/// open it in the explorer, ask it from a shell.
fn next_with_sample(p: &std::path::Path) -> Vec<(String, String)> {
    let at = crate::tui::setup::scan::tilde(p);
    vec![
        (format!("urna tui {at}"), String::new()),
        (format!("urna ask {at} \"...\""), String::new()),
    ]
}
