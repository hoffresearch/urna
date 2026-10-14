//! The scan: the facts about this machine landing one by one next to
//! the symbol, with a gauge of how many are in.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;

use crate::tui::hud::{self, Badge};
use crate::tui::setup::ui::Ui;
use crate::tui::{art, fx, pal};

pub fn draw(buf: &mut Buffer, body: Rect, ui: &mut Ui) {
    let show_symbol = body.width >= 70 && body.height >= 12;
    let (sym_rect, panel_x) = if show_symbol {
        let sy = body.y + (body.height.saturating_sub(10)) / 2;
        art::paint(
            buf,
            body.x + 3,
            sy,
            &art::SYMBOL_M,
            pal::TITLE_FROM,
            pal::TITLE_TO,
        );
        (Some(Rect::new(body.x + 3, sy, 18, 10)), body.x + 25)
    } else {
        (None, body.x + 1)
    };
    let prect = Rect::new(
        panel_x,
        body.y,
        body.right().saturating_sub(panel_x + 1),
        body.height,
    );
    let facts = ui.scan.as_ref().map(|s| s.facts()).unwrap_or_default();
    let tag = if ui.scan.is_some() {
        format!("{}/{}", ui.shown, facts.len())
    } else {
        "probing".into()
    };
    let inner = hud::panel(buf, prect, "this machine", &tag, true);
    if ui.scan.is_none() {
        hud::spans(
            buf,
            inner.x,
            inner.y,
            &[
                (ui.spinner.frame_str(), pal::accent()),
                (" probing python, tools and paths", pal::dim()),
            ],
            inner.width,
        );
    }
    let prev = ui.drawn;
    for (i, (label, value, badge)) in facts.iter().take(ui.shown).enumerate() {
        let y = inner.y + i as u16;
        if y >= inner.bottom().saturating_sub(1) {
            break;
        }
        let (b, bst) = hud::badge(*badge);
        hud::put(buf, inner.x, y, b, bst, 6);
        let vst = match badge {
            Badge::Ok => pal::dim(),
            Badge::Fail => pal::err(),
            _ => pal::warn(),
        };
        hud::leader(
            buf,
            inner.x + 7,
            y,
            inner.width.saturating_sub(7),
            label,
            value,
            vst,
        );
        if i >= prev {
            ui.fx
                .add_effect(fx::row_in(Rect::new(inner.x, y, inner.width, 1), 0));
        }
    }
    ui.drawn = ui.shown;
    let ratio = if facts.is_empty() {
        0.0
    } else {
        ui.shown as f64 / facts.len() as f64
    };
    let gy = (inner.y + facts.len().max(1) as u16 + 1).min(inner.bottom().saturating_sub(1));
    hud::gauge(buf, inner.x, gy, inner.width, ratio, pal::GLOW);
    let note_y = inner.bottom().saturating_sub(1);
    if note_y > gy + 1 {
        let note = "read-only so far: nothing is written until you confirm the plan";
        hud::put(buf, inner.x, note_y, note, pal::faint(), inner.width);
    }
    if let Some(r) = sym_rect
        && ui.entered
    {
        ui.fx.add_unique_effect("shimmer", fx::shimmer(r));
    }
}
