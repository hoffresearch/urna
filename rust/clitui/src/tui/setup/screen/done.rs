//! The verify screen: the doctor rows, the verdict, what to run next.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Modifier;

use crate::tui::hud::{self, Badge};
use crate::tui::setup::plan::Task;
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
    let ok = ui.code == 0;
    let n = ui.health.len() as u16;
    let vrect = Rect::new(
        px,
        body.y,
        body.right().saturating_sub(px + 1),
        (n + 2).min(body.height.saturating_sub(5)),
    );
    let verdict = if ok {
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
    if !ok {
        // why, step by step: the worker's own error text, wrapped.
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
        for (task, why) in reasons {
            for (j, line) in hud::wrap(&why, w).iter().take(2).enumerate() {
                if yy >= body.bottom() {
                    break;
                }
                let label = if j == 0 { task } else { "" };
                hud::put(buf, status.x, yy, label, pal::dim(), 18);
                hud::put(buf, status.x + 19, yy, line, pal::err(), w as u16);
                yy += 1;
            }
        }
        yy += 1;
    }
    let next: &[(&str, &str)] = if ok {
        &[
            ("urna tui", "open the explorer"),
            ("urna ask corpus.urna \"...\"", "ask a corpus"),
            (
                "urna build --spec corpus.toml",
                "build one, from a checkout of the repo",
            ),
        ]
    } else {
        &[
            ("urna setup", "run it again"),
            ("urna setup --yes > setup.log", "the plain log, to share"),
            ("urna doctor", "re-check without changing anything"),
        ]
    };
    for (cmd, what) in next {
        if yy >= body.bottom() {
            break;
        }
        let x = hud::spans(
            buf,
            status.x,
            yy,
            &[("$ ", pal::faint()), (cmd, pal::key())],
            status.width,
        );
        let dx = x.max(status.x + 34);
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
