// SPDX-FileCopyrightText: 2026 Brenner Cruvinel, V. Hoff Research
//
// SPDX-License-Identifier: MIT

//! The model picker: one line per model the payload's catalog offers, its
//! download (torch counted once), its estimated minimum ram against this
//! machine's and its license, in a list that scrolls; under it, the details
//! of the model at the cursor. potion heads the list, already installed:
//! it ships in the payload. nothing here downloads; the run does, after
//! enter.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{List, ListItem, StatefulWidget};

use crate::tui::hud;
use crate::tui::pal;
use crate::tui::setup::models::{self, Entry, Torch};
use crate::tui::setup::ui::Ui;

/// Rows the details of the model at the cursor take.
const DETAILS: u16 = 6;

/// What the screen says about one model on this machine.
struct Fit {
    download: Result<u64, String>,
    ram: Option<u64>,
    fits: Option<bool>,
}

fn fit(e: &Entry, torch: Option<&Torch>, target: &str, has_torch: bool, ram: Option<u64>) -> Fit {
    let need = e.min_ram(torch, target);
    Fit {
        download: e.download(torch, target, has_torch),
        ram: need,
        fits: need.zip(ram).map(|(n, have)| n <= have),
    }
}

/// `name · 0.54 GB · ~2 GB RAM ✓ · MIT`, the name padded to `pad`.
fn line(e: &Entry, f: &Fit, chosen: bool, pad: usize) -> Line<'static> {
    let mark = if chosen { "[x] " } else { "[ ] " };
    let mark_st = if chosen {
        pal::ok().add_modifier(Modifier::BOLD)
    } else {
        pal::dim()
    };
    let mut spans = vec![
        Span::styled(mark, mark_st),
        Span::styled(format!("{:<pad$}", e.name), pal::text()),
    ];
    match &f.download {
        Ok(n) => spans.push(Span::styled(format!("  {}", models::gb(*n)), pal::text())),
        Err(_) => spans.push(Span::styled("  no torch build here", pal::faint())),
    }
    if let Some(r) = f.ram {
        spans.push(Span::styled(
            format!(" · ~{} GB RAM", models::ram_gb(r)),
            pal::text(),
        ));
        match f.fits {
            Some(true) => spans.push(Span::styled(" ✓", pal::ok())),
            Some(false) => spans.push(Span::styled(" ✗", pal::err())),
            None => {}
        }
    }
    if !e.license.is_empty() {
        let st = if e.terms().is_some() {
            pal::warn()
        } else {
            pal::faint()
        };
        spans.push(Span::styled(format!(" · {}", e.license), st));
        if let Some(t) = e.terms() {
            spans.push(Span::styled(format!(" · {t}"), st));
        }
    }
    Line::from(spans)
}

/// The model at the cursor, in full: where it comes from, what the
/// download and the ram estimate are made of, its license and repo code.
fn details(
    e: &Entry,
    f: &Fit,
    torch: Option<&Torch>,
    target: &str,
    allowed: bool,
) -> Vec<(String, Style)> {
    let mut out = Vec::new();
    let mut head = format!("{}@{}", e.repo, &e.revision[..e.revision.len().min(8)]);
    if !e.modalities.is_empty() {
        head.push_str(&format!(" · {}", e.modalities.join(", ")));
    }
    if e.dim > 0 {
        head.push_str(&format!(" · {} dims", e.dim));
    }
    out.push((head, pal::text()));
    let weights = models::gb(e.bytes);
    out.push(
        match (&f.download, torch.and_then(|t| t.sizes.get(target))) {
            (Err(why), _) => (format!("cannot install here: {why}"), pal::err()),
            (Ok(n), Some(t)) if *n > e.bytes => (
                format!(
                    "download {}: weights {weights} + torch {}, downloaded once and shared by every model",
                    models::gb(*n),
                    models::gb(*t)
                ),
                pal::faint(),
            ),
            _ => (
                format!("download {weights}: the weights; torch is in the env already"),
                pal::faint(),
            ),
        },
    );
    if let Some(r) = f.ram {
        let per = e
            .dtype_bytes
            .get(models::device(target))
            .copied()
            .unwrap_or(0);
        let base = torch.and_then(|t| t.rss.get(target)).copied().unwrap_or(0);
        out.push((
            format!(
                "ram ~{} GB, an estimate: {:.0}M parameters × {per} bytes on {} + torch {:.1} GB",
                models::ram_gb(r),
                e.params as f64 / 1e6,
                models::device(target),
                base as f64 / (1u64 << 30) as f64
            ),
            pal::faint(),
        ));
    }
    if !e.license.is_empty() {
        let terms = e.terms().map(|t| format!(": {t}")).unwrap_or_default();
        let st = if e.terms().is_some() {
            pal::warn()
        } else {
            pal::faint()
        };
        out.push((format!("license {}{terms}", e.license), st));
    }
    if e.remote_code {
        let n = e.remote_code_hashes.len();
        out.push(if allowed {
            (
                format!("its repo code ({n} files, reviewed and pinned) may run: r withdraws"),
                pal::ok(),
            )
        } else {
            (
                format!("runs code from its repo, reviewed and pinned ({n} files): r allows it"),
                pal::err(),
            )
        });
    }
    out
}

pub fn draw(buf: &mut Buffer, body: Rect, ui: &mut Ui) {
    let offered = ui.offered();
    let (torch, target, has_torch, ram) = match &ui.scan {
        Some(s) => (
            s.kit.as_ref().and_then(|k| k.catalog.torch.clone()),
            s.target.clone(),
            s.torch,
            s.ram,
        ),
        None => (None, String::new(), false, None),
    };
    let n_on = offered.iter().filter(|e| ui.chosen(&e.name)).count();
    let area = Rect::new(
        body.x + 1,
        body.y,
        body.width.saturating_sub(2),
        body.height,
    );
    let tag = match ram {
        Some(r) => format!(
            "{n_on} of {} chosen · this machine {} GB · a picks all",
            offered.len(),
            models::ram_gb(r)
        ),
        None => format!("{n_on} of {} chosen · a picks all", offered.len()),
    };
    let inner = hud::panel(buf, area, "models", &tag, true);
    if inner.height < 4 {
        return;
    }
    let pad = offered
        .iter()
        .map(|e| e.name.len())
        .max()
        .unwrap_or(0)
        .max(6);
    let fits: Vec<Fit> = offered
        .iter()
        .map(|e| fit(e, torch.as_ref(), &target, has_torch, ram))
        .collect();
    hud::spans(
        buf,
        inner.x + 2,
        inner.y,
        &[
            ("[*] ", pal::faint()),
            (&format!("{:<pad$}", "potion"), pal::faint()),
            ("  installed: it ships in the payload · MIT", pal::faint()),
        ],
        inner.width.saturating_sub(2),
    );
    // as tall as the longest details, so the list does not jump as the
    // cursor moves.
    let allowed = |e: &Entry| ui.opts.allow_remote_code.contains(&e.name);
    let tallest = offered
        .iter()
        .zip(&fits)
        .map(|(e, f)| details(e, f, torch.as_ref(), &target, allowed(e)).len())
        .max()
        .unwrap_or(0) as u16;
    let details_h = tallest.min(DETAILS).min(inner.height.saturating_sub(3));
    let list_area = Rect::new(
        inner.x,
        inner.y + 1,
        inner.width,
        inner.height.saturating_sub(2 + details_h),
    );
    let items: Vec<ListItem<'_>> = offered
        .iter()
        .zip(&fits)
        .map(|(e, f)| ListItem::new(line(e, f, ui.chosen(&e.name), pad)))
        .collect();
    let list = List::new(items)
        .highlight_symbol("› ")
        .highlight_spacing(ratatui::widgets::HighlightSpacing::Always)
        .highlight_style(Style::new().fg(pal::INK_HI).add_modifier(Modifier::BOLD));
    ui.list.select(Some(ui.pick));
    StatefulWidget::render(list, list_area, buf, &mut ui.list);
    let Some((e, f)) = offered.get(ui.pick).zip(fits.get(ui.pick)) else {
        return;
    };
    let dy = list_area.bottom() + 1;
    let w = inner.width.saturating_sub(4);
    let lines = details(e, f, torch.as_ref(), &target, allowed(e));
    for (i, (text, st)) in lines.iter().take(details_h as usize).enumerate() {
        hud::put(buf, inner.x + 2, dy + i as u16, text, *st, w);
    }
}

/// Whether the model at `i` can be chosen here: torch ships a build for
/// this platform (or the env holds it already).
pub fn installable(ui: &Ui, e: &Entry) -> bool {
    match &ui.scan {
        Some(s) => {
            let torch = s.kit.as_ref().and_then(|k| k.catalog.torch.as_ref());
            e.download(torch, &s.target, s.torch).is_ok()
        }
        None => true,
    }
}
