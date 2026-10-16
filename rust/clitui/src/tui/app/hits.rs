//! Drawing the ask tab: the query line (ratatui-cheese input), the status,
//! the suggestions while typing or the hits with the exact-rerank score as a
//! thermometer bar, and the selected hit: its image when its source is one,
//! beside its stored text, citation and source.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::widgets::{Paragraph, StatefulWidget, Widget, Wrap};

use super::ask::Ask;
use crate::tui::widgets::{Input, InputStyles, ScrollBar};
use crate::tui::{hud, pal};

pub fn render(buf: &mut Buffer, body: Rect, ask: &mut Ask, spinner: &str) {
    let styles = InputStyles {
        title: pal::title(),
        description: pal::faint(),
        prompt: pal::key(),
        text: Style::new().fg(pal::INK_HI),
        placeholder: pal::faint(),
        cursor: Style::new().fg(pal::BG).bg(pal::TITLE_FROM),
    };
    let qrect = Rect::new(body.x + 2, body.y, body.width.saturating_sub(4), 3);
    let input = Input::new("ask")
        .description("text in, cited spans out · embeds offline, gated by the model hash")
        .placeholder("what does the corpus say about …")
        .prompt("› ")
        .styles(styles);
    StatefulWidget::render(&input, qrect, buf, &mut ask.input);
    let status_y = body.y + 3;
    let status = match (&ask.pending, &ask.asked) {
        (Some((_, t0)), _) => format!(
            "{spinner} embedding and searching · {:.1}s",
            t0.elapsed().as_secs_f64()
        ),
        (None, Some(_)) if ask.failed.is_some() => {
            format!("ask failed: {}", ask.failed.as_deref().unwrap_or(""))
        }
        (None, Some(_)) if ask.suggesting() => format!(
            "{} labels match · {:.1} ms · ↑↓ points · tab asks for it · enter asks the text",
            ask.suggestions.len(),
            ask.typed_ms
        ),
        (None, Some(_)) if ask.answers.first().is_some_and(|a| a.via == "browse") => format!(
            "{} most played under the filters · {:.0} ms{}",
            ask.answers.len(),
            ask.took_ms,
            ask.media
                .decode_ms()
                .map(|ms| format!(" · image {ms:.1} ms ({})", ask.media.mode()))
                .unwrap_or_default()
        ),
        (None, Some(q)) => format!(
            "{} hits for \"{q}\" · {:.0} ms{}",
            ask.answers.len(),
            ask.took_ms,
            ask.media
                .decode_ms()
                .map(|ms| format!(" · image {ms:.1} ms ({})", ask.media.mode()))
                .unwrap_or_default()
        ),
        (None, None) if ask.suggesting() => format!(
            "{} labels match · {:.1} ms · ↑↓ points · tab asks for it · enter asks the text",
            ask.suggestions.len(),
            ask.typed_ms
        ),
        (None, None) => "enter asks · ↑↓ picks a hit · pgup/pgdn scrolls the text".into(),
    };
    let st = if ask.failed.is_some() {
        pal::err()
    } else {
        pal::faint()
    };
    hud::put(buf, qrect.x, status_y, &status, st, qrect.width);
    // the facets of the kept set, one line, when the corpus declares any.
    let mut top = status_y + 1;
    if !ask.facets.is_empty() && body.height > 12 {
        let line = ask
            .facets
            .iter()
            .map(|f| {
                let key = f.prefix.as_deref().unwrap_or(f.label.as_str());
                let values: Vec<String> = f
                    .values
                    .iter()
                    .map(|(v, c)| format!("{} {}", v.replace('_', " "), short(*c)))
                    .collect();
                format!("{key}: {}", values.join(" · "))
            })
            .collect::<Vec<_>>()
            .join("   ");
        hud::put(buf, qrect.x, top, &line, pal::dim(), qrect.width);
        top += 1;
    }
    let rest = Rect::new(body.x, top, body.width, body.bottom().saturating_sub(top));
    if rest.height < 3 {
        return;
    }
    let lw = (rest.width * 2 / 5).max(28.min(rest.width));
    if ask.suggesting() {
        suggestions(buf, Rect::new(rest.x + 1, rest.y, lw, rest.height), ask);
    } else {
        hits(buf, Rect::new(rest.x + 1, rest.y, lw, rest.height), ask);
    }
    detail(buf, rest, lw, ask);
}

fn suggestions(buf: &mut Buffer, area: Rect, ask: &Ask) {
    let list = hud::panel(buf, area, "suggestions", "tab asks", true);
    let first = ask
        .sug_sel
        .saturating_sub(list.height.saturating_sub(1) as usize);
    for (i, label) in ask.suggestions.iter().enumerate().skip(first) {
        let y = list.y + (i - first) as u16;
        if y >= list.bottom() {
            break;
        }
        let sel = i == ask.sug_sel;
        let st = if sel {
            Style::new().fg(pal::INK_HI).add_modifier(Modifier::BOLD)
        } else {
            Style::new().fg(pal::INK)
        };
        hud::spans(
            buf,
            list.x,
            y,
            &[(if sel { "› " } else { "  " }, pal::key()), (label, st)],
            list.width,
        );
        if sel {
            buf.set_style(
                Rect::new(list.x, y, list.width, 1),
                Style::new().bg(pal::SURFACE),
            );
        }
    }
}

fn hits(buf: &mut Buffer, area: Rect, ask: &Ask) {
    let list = hud::panel(buf, area, "hits", "exact rerank", true);
    let top = ask
        .answers
        .iter()
        .filter(|a| !a.by_name)
        .map(|a| a.score)
        .fold(0.0f32, f32::max)
        .max(1e-6);
    // two rows per hit; the window scrolls so the selected hit stays in it.
    let fits = ((list.height.saturating_sub(1)) / 2).max(1) as usize;
    let first = ask.sel.saturating_sub(fits - 1);
    for (i, a) in ask.answers.iter().enumerate().skip(first) {
        let y = list.y + (i - first) as u16 * 2;
        // a hit is two rows (text + score bar); both must clear the border.
        if y + 1 >= list.bottom() {
            break;
        }
        let sel = i == ask.sel;
        let st = if sel {
            Style::new().fg(pal::INK_HI).add_modifier(Modifier::BOLD)
        } else {
            Style::new().fg(pal::INK)
        };
        let first = a.text.lines().next().unwrap_or("");
        let kind = ask
            .session
            .as_ref()
            .map(|s| s.tags(a.chunk).join(" "))
            .filter(|t| !t.is_empty())
            .map(|t| format!("  {}", t.replace('_', " ")))
            .unwrap_or_default();
        hud::spans(
            buf,
            list.x,
            y,
            &[
                (if sel { "› " } else { "  " }, pal::key()),
                // a hit found by its label or browsed has no cosine to show.
                (
                    &match a.via {
                        "name" => "name  ".to_string(),
                        "browse" => "most  ".to_string(),
                        _ => format!("{:.3} ", a.score),
                    },
                    pal::accent(),
                ),
                (first, st),
                (&kind, pal::faint()),
            ],
            list.width,
        );
        let ratio = if a.via != "hybrid" {
            0.0
        } else {
            (a.score / top).clamp(0.0, 1.0) as f64
        };
        hud::gauge(
            buf,
            list.x + 2,
            y + 1,
            list.width.saturating_sub(2),
            ratio,
            pal::thermo(ratio as f32),
        );
        if sel {
            buf.set_style(
                Rect::new(list.x, y, list.width, 1),
                Style::new().bg(pal::SURFACE),
            );
        }
    }
}

fn detail(buf: &mut Buffer, rest: Rect, lw: u16, ask: &mut Ask) {
    let drect = Rect::new(
        rest.x + lw + 2,
        rest.y,
        rest.width.saturating_sub(lw + 3),
        rest.height,
    );
    let Some(a) = ask.answers.get(ask.sel).cloned() else {
        let inner = hud::panel(buf, drect, "text", "", false);
        hud::put(
            buf,
            inner.x,
            inner.y,
            "the cited text of the selected hit shows here",
            pal::faint(),
            inner.width,
        );
        return;
    };
    let as_card = ask.session.as_ref().is_some_and(|s| super::card::knows(s));
    let (title, note) = if as_card {
        ("record", "laid out from the file's schema")
    } else {
        ("text", "tier-1 · stored canonical")
    };
    let inner = hud::panel(buf, drect, title, note, false);
    if inner.height < 3 {
        return;
    }
    // an image source draws beside the text: as wide as its aspect needs at
    // the pane's height, never more than half the pane.
    let mut tx = inner.x;
    if let (Some(blob), Some(session)) = (a.image, ask.session.clone()) {
        let title = a.text.lines().next().unwrap_or("").trim().to_string();
        ask.media.show(&session, blob, &title);
        let rows = inner.height.saturating_sub(3);
        let w = ask.media.cells_wide(rows).min(inner.width / 2);
        if w > 0 {
            ask.media.render(buf, Rect::new(inner.x, inner.y, w, rows));
            tx = inner.x + w + 2;
        } else if let Some(err) = ask.media.error() {
            hud::put(buf, inner.x, inner.y, err, pal::err(), inner.width);
            tx = inner.x;
        }
    }
    let text_rect = Rect::new(
        tx,
        inner.y,
        inner.right().saturating_sub(tx + 2),
        inner.height.saturating_sub(3),
    );
    let (para, total) = if let Some(session) = ask.session.as_ref().filter(|_| as_card) {
        let lines = super::card::lines(session, a.chunk, &a.text, text_rect.width);
        let plain: String = lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        (
            Paragraph::new(lines),
            wrapped_lines(&plain, text_rect.width),
        )
    } else {
        (
            Paragraph::new(a.text.as_str()).style(Style::new().fg(pal::INK)),
            wrapped_lines(&a.text, text_rect.width),
        )
    };
    let para = para.wrap(Wrap { trim: false });
    ask.scroll = ask.scroll.min(total.saturating_sub(text_rect.height));
    para.scroll((ask.scroll, 0)).render(text_rect, buf);
    if total > text_rect.height {
        ScrollBar::vertical(total as usize, text_rect.height as usize)
            .offset(ask.scroll as usize)
            .track_style(Style::new().fg(pal::SURFACE))
            .thumb_style(Style::new().fg(pal::GLOW))
            .render(Rect::new(inner.right(), inner.y, 1, text_rect.height), buf);
    }
    let cy = inner.bottom() - 2;
    let w = inner.width.saturating_sub(8) as usize;
    hud::spans(
        buf,
        inner.x,
        cy,
        &[
            ("cite    ", pal::faint()),
            (&hud::tail(&a.citation, w), pal::accent()),
        ],
        inner.width,
    );
    hud::spans(
        buf,
        inner.x,
        cy + 1,
        &[
            ("source  ", pal::faint()),
            (&hud::tail(&a.source, w), pal::dim()),
        ],
        inner.width,
    );
}

/// A count in at most four characters: `987`, `12k`, `1.2M`.
fn short(n: usize) -> String {
    match n {
        0..=999 => n.to_string(),
        1_000..=9_999 => format!("{:.1}k", n as f64 / 1e3),
        10_000..=999_999 => format!("{}k", n / 1_000),
        _ => format!("{:.1}M", n as f64 / 1e6),
    }
}

/// Rows `text` takes when word-wrapped at `width` (close enough to the
/// paragraph's own wrap for sizing a scrollbar).
fn wrapped_lines(text: &str, width: u16) -> u16 {
    let w = width.max(1) as usize;
    text.lines()
        .map(|l| l.chars().count().div_ceil(w).max(1))
        .sum::<usize>() as u16
}
