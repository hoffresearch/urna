// SPDX-FileCopyrightText: 2026 Brenner Cruvinel, V. Hoff Research
//
// SPDX-License-Identifier: MIT

//! The splash: the symbol over the hash rain, the wordmark resolving out
//! of block glyphs, what setup is about to do, and the key to begin.

use std::time::Duration;

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Style;

use crate::tui::hud::{self};
use crate::tui::rain::Rain;
use crate::tui::setup::ui::Ui;
use crate::tui::{art, fx, pal};

pub fn draw(buf: &mut Buffer, body: Rect, ui: &mut Ui) {
    Rain::quiet(ui.clock.start.elapsed()).render(body, buf);
    let wide = body.width >= 74 && body.height >= 16;
    let symbol: &[&str] = if wide { &art::SYMBOL_L } else { &art::SYMBOL_M };
    let sw = art::width(symbol);
    let sy = body.y + body.height.saturating_sub(symbol.len() as u16) / 2;
    let sx = if wide {
        body.x + 3
    } else {
        body.x + (body.width.saturating_sub(sw)) / 2
    };
    let sym_rect = Rect::new(sx, sy, sw, symbol.len() as u16).intersection(body);
    let tx = if wide { sx + sw + 5 } else { body.x + 2 };
    let text_w = body.right().saturating_sub(tx + 1);
    // the text column sits on clean ground, the rain stays around it.
    let top = if wide { sy } else { sym_rect.bottom() + 1 };
    let text_rect =
        Rect::new(tx, top, text_w, body.bottom().saturating_sub(top)).intersection(body);
    if wide {
        buf.set_style(
            Rect::new(tx.saturating_sub(2), sy, text_w + 3, symbol.len() as u16).intersection(body),
            Style::new().bg(pal::BG),
        );
        for y in text_rect.top()..text_rect.bottom() {
            for x in text_rect.left()..text_rect.right() {
                if let Some(c) = buf.cell_mut((x, y)) {
                    c.set_char(' ');
                }
            }
        }
    }
    art::paint(buf, sx, sy, symbol, pal::TITLE_FROM, pal::TITLE_TO);
    let mut y = text_rect.y;
    let word_rect = if wide && text_w >= art::width(&art::WORD) {
        art::paint(buf, tx, y + 1, &art::WORD, pal::INK_HI, pal::TITLE_TO);
        let r = Rect::new(tx, y + 1, art::width(&art::WORD), 4);
        y += 6;
        Some(r)
    } else {
        hud::put(buf, tx, y, "urna", pal::title(), text_w);
        y += 2;
        None
    };
    let line = |buf: &mut Buffer, y: u16, s: &str, st: Style| hud::put(buf, tx, y, s, st, text_w);
    line(
        buf,
        y,
        "the single-file vector database with stable citations",
        Style::new().fg(pal::INK),
    );
    line(
        buf,
        y + 1,
        "offline by construction · memory-mapped · hash-verified",
        pal::dim(),
    );
    line(
        buf,
        y + 3,
        "this lays down the offline embedder and a python env,",
        pal::faint(),
    );
    line(
        buf,
        y + 4,
        "then proves the install with the doctor checks.",
        pal::faint(),
    );
    let whereami = match &ui.scan {
        Some(s) => format!("v{} · {} · {}", s.name, s.target, s.channel.name()),
        None => format!(
            "v{} · probing this machine {}",
            crate::cmd::buildid::NAME,
            ui.spinner.frame_str()
        ),
    };
    line(buf, y + 6, &whereami, pal::faint());
    if y + 7 < body.bottom() {
        hud::link(buf, tx, y + 7, "urna.dev", "https://urna.dev", text_w);
        hud::put(
            buf,
            tx + 9,
            y + 7,
            "· docs, benchmarks, the format spec",
            pal::faint(),
            text_w.saturating_sub(9),
        );
    }
    let blink = ((ui.clock.secs() * 1.6) as u64).is_multiple_of(2);
    if y + 9 < body.bottom() && ui.in_step() > Duration::from_millis(900) {
        let st = if blink {
            pal::key()
        } else {
            Style::new().fg(pal::TITLE_TO)
        };
        hud::spans(
            buf,
            tx,
            y + 9,
            &[
                ("press ", pal::faint()),
                ("enter", st),
                (" to begin", pal::faint()),
            ],
            text_w,
        );
    }
    if ui.entered {
        ui.fx.add_unique_effect("logo", fx::logo_in(sym_rect));
        ui.fx.add_unique_effect("shimmer", fx::shimmer(sym_rect));
        if let Some(r) = word_rect {
            ui.fx.add_unique_effect("word", fx::resolve(r, 900));
        }
        ui.fx.add_unique_effect(
            "text",
            fx::page_in(Rect::new(tx, y, text_w, 8).intersection(body)),
        );
    }
}
