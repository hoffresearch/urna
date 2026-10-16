//! Drawing: the input on top, the list on the left, the card on the right
//! (image beside its text), the keys and timings at the bottom.

use pasta_core::Card;
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, List, ListItem, ListState, Paragraph, Wrap};
use ratatui_image::{FontSize, Resize, StatefulImage};

use crate::app::{App, Mode};

const ACCENT: Color = Color::Rgb(233, 196, 106);
const FAINT: Color = Color::Rgb(120, 120, 130);
const INK: Color = Color::Rgb(225, 225, 230);

/// Card scans are 488x680.
const CARD_W: u32 = 488;
const CARD_H: u32 = 680;

pub fn draw(f: &mut Frame, app: &mut App) {
    let [top, body, foot] =
        Layout::vertical([Constraint::Length(3), Constraint::Min(5), Constraint::Length(1)]).areas(f.area());
    draw_input(f, app, top);
    let [left, right] = Layout::horizontal([Constraint::Percentage(36), Constraint::Min(30)]).areas(body);
    draw_list(f, app, left);
    draw_card(f, app, right);
    draw_footer(f, app, foot);
}

fn frame_block(title: &str) -> Block<'_> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(FAINT))
        .title(Span::styled(format!(" {title} "), Style::new().fg(ACCENT).add_modifier(Modifier::BOLD)))
}

fn draw_input(f: &mut Frame, app: &App, area: Rect) {
    let hint = if app.input.is_empty() {
        Span::styled("nome da carta, texto de regra, ou | format=commander id=RG cmc<=3 type=dragon", Style::new().fg(FAINT))
    } else {
        Span::styled(app.input.clone(), Style::new().fg(INK))
    };
    let line = Line::from(vec![Span::styled("› ", Style::new().fg(ACCENT)), hint, Span::styled("▏", Style::new().fg(ACCENT))]);
    f.render_widget(Paragraph::new(line).block(frame_block("Pasta Cards")), area);
}

fn draw_list(f: &mut Frame, app: &App, area: Rect) {
    let title = match app.mode {
        Mode::Suggest => format!("sugestões · {}", app.rows.len()),
        Mode::Results => format!("resultados · {}", app.rows.len()),
        Mode::Browse => "mais jogadas".to_string(),
    };
    let cards = app.pasta.cards();
    let items: Vec<ListItem> = app
        .rows
        .iter()
        .map(|r| {
            let c = &cards[r.ordinal];
            let mut spans = vec![Span::styled(c.name.clone(), Style::new().fg(rarity_color(&c.rarity)))];
            if let Some(pt) = &c.pt_name
                && pt != &c.name
            {
                spans.push(Span::styled(format!("  {pt}"), Style::new().fg(FAINT)));
            }
            spans.push(Span::raw("  "));
            spans.extend(mana_spans(&c.mana_cost));
            if let Some(score) = r.score.filter(|s| *s < 1.0) {
                spans.push(Span::styled(format!("  {score:.2}"), Style::new().fg(FAINT)));
            }
            ListItem::new(Line::from(spans))
        })
        .collect();
    let list = List::new(items)
        .block(frame_block(&title))
        .highlight_style(Style::new().bg(Color::Rgb(45, 45, 58)).add_modifier(Modifier::BOLD))
        .highlight_symbol("▌");
    let mut state = ListState::default().with_selected(if app.rows.is_empty() { None } else { Some(app.sel) });
    f.render_stateful_widget(list, area, &mut state);
}

fn draw_card(f: &mut Frame, app: &mut App, area: Rect) {
    let block = frame_block("carta");
    let inner = block.inner(area);
    f.render_widget(block, area);
    let Some(ordinal) = app.selected() else {
        f.render_widget(Paragraph::new(Span::styled("nada encontrado", Style::new().fg(FAINT))), inner);
        return;
    };
    let font = app.picker.font_size();
    // the image takes the width its aspect needs at the full inner height,
    // capped so the text keeps at least half the pane.
    let img_w = cells_for_height(inner.height, font).min(inner.width / 2);
    let [img_area, _, text_area] =
        Layout::horizontal([Constraint::Length(img_w), Constraint::Length(2), Constraint::Min(10)]).areas(inner);

    if let Some(shown) = app.shown.as_mut().filter(|s| s.ordinal == ordinal) {
        if let Some(proto) = shown.image.as_mut() {
            let fit = fit_rect(img_area, font);
            f.render_stateful_widget(StatefulImage::default().resize(Resize::Scale(None)), fit, proto);
        } else if let Some(err) = &shown.error {
            f.render_widget(Paragraph::new(err.as_str()).wrap(Wrap { trim: true }), img_area);
        }
    }
    let card = &app.pasta.cards()[ordinal];
    f.render_widget(Paragraph::new(card_lines(card)).wrap(Wrap { trim: false }), text_area);
}

fn draw_footer(f: &mut Frame, app: &App, area: Rect) {
    let decode = app.shown.as_ref().map(|s| format!(" · imagem {:.1} ms", s.decode_ms)).unwrap_or_default();
    let line = Line::from(vec![
        Span::styled(" ↑↓ ", Style::new().fg(ACCENT)),
        Span::styled("navegar  ", Style::new().fg(FAINT)),
        Span::styled("⏎ ", Style::new().fg(ACCENT)),
        Span::styled("buscar  ", Style::new().fg(FAINT)),
        Span::styled("tab ", Style::new().fg(ACCENT)),
        Span::styled("completar  ", Style::new().fg(FAINT)),
        Span::styled("esc ", Style::new().fg(ACCENT)),
        Span::styled("limpar/sair", Style::new().fg(FAINT)),
        Span::styled(
            format!("   {:?} · lista {:.1} ms{decode}", app.picker.protocol_type(), app.last_ms),
            Style::new().fg(FAINT),
        ),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

fn card_lines(c: &Card) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    let mut title = vec![Span::styled(c.name.clone(), Style::new().fg(INK).add_modifier(Modifier::BOLD)), Span::raw("  ")];
    title.extend(mana_spans(&c.mana_cost));
    out.push(Line::from(title));
    if let Some(pt) = c.pt_name.as_ref().filter(|pt| *pt != &c.name) {
        out.push(Line::from(Span::styled(pt.clone(), Style::new().fg(FAINT).add_modifier(Modifier::ITALIC))));
    }
    out.push(Line::from(""));
    out.push(Line::from(Span::styled(c.type_line.clone(), Style::new().fg(INK))));
    out.push(Line::from(vec![
        Span::styled(c.rarity.clone(), Style::new().fg(rarity_color(&c.rarity))),
        Span::styled(format!(" · {}", c.set_code.to_uppercase()), Style::new().fg(FAINT)),
        Span::styled(c.cmc.map(|v| format!(" · cmc {v}")).unwrap_or_default(), Style::new().fg(FAINT)),
    ]));
    out.push(Line::from(""));
    for l in c.oracle_text.lines() {
        out.push(Line::from(Span::styled(l.to_string(), Style::new().fg(INK))));
    }
    let stats = match (&c.power, &c.toughness, &c.loyalty) {
        (Some(p), Some(t), _) => Some(format!("{p}/{t}")),
        (_, _, Some(l)) => Some(format!("lealdade {l}")),
        _ => None,
    };
    if let Some(s) = stats {
        out.push(Line::from(""));
        out.push(Line::from(Span::styled(s, Style::new().fg(ACCENT).add_modifier(Modifier::BOLD))));
    }
    out.push(Line::from(""));
    let mut legal = vec![Span::styled("formatos  ", Style::new().fg(FAINT))];
    for f in ["standard", "pioneer", "modern", "legacy", "vintage", "commander", "pauper"] {
        let (mark, color) = match c.legal.get(f).map(String::as_str) {
            Some("legal") => ("●", Color::Rgb(120, 200, 140)),
            Some("restricted") => ("◐", Color::Rgb(230, 190, 90)),
            Some("banned") => ("✕", Color::Rgb(220, 90, 90)),
            _ => ("○", FAINT),
        };
        legal.push(Span::styled(format!("{mark} {f}  "), Style::new().fg(color)));
    }
    out.push(Line::from(legal));
    if let Some(r) = c.edhrec_rank {
        out.push(Line::from(Span::styled(format!("edhrec #{r}"), Style::new().fg(FAINT))));
    }
    out
}

/// `{2}{U}{R}` as colored pips.
fn mana_spans(cost: &str) -> Vec<Span<'static>> {
    let mut out = Vec::new();
    for part in cost.split_inclusive('}') {
        let sym = part.trim().trim_start_matches('{').trim_end_matches('}');
        if part.trim() == "//" || part.contains("//") {
            out.push(Span::styled(" // ", Style::new().fg(FAINT)));
            continue;
        }
        if sym.is_empty() {
            continue;
        }
        let color = match sym {
            "W" => Color::Rgb(248, 231, 185),
            "U" => Color::Rgb(100, 160, 230),
            "B" => Color::Rgb(170, 150, 190),
            "R" => Color::Rgb(235, 105, 80),
            "G" => Color::Rgb(110, 190, 120),
            "C" => Color::Rgb(190, 190, 200),
            _ if sym.contains('/') => Color::Rgb(210, 180, 230),
            _ => Color::Rgb(200, 200, 205),
        };
        out.push(Span::styled(format!("({sym})"), Style::new().fg(color).add_modifier(Modifier::BOLD)));
    }
    out
}

fn rarity_color(r: &str) -> Color {
    match r {
        "mythic" => Color::Rgb(240, 130, 60),
        "rare" => Color::Rgb(230, 190, 90),
        "uncommon" => Color::Rgb(170, 190, 205),
        _ => INK,
    }
}

/// Cells wide a card needs to fill `rows` cells tall, from the font's pixel size.
fn cells_for_height(rows: u16, font: FontSize) -> u16 {
    let px_h = rows as u32 * font.height.max(1) as u32;
    let px_w = px_h * CARD_W / CARD_H;
    px_w.div_ceil(font.width.max(1) as u32) as u16
}

/// The card's aspect fitted inside `area`, centered.
fn fit_rect(area: Rect, font: FontSize) -> Rect {
    let (fw, fh) = (font.width.max(1) as f64, font.height.max(1) as f64);
    let scale = (area.width as f64 * fw / CARD_W as f64).min(area.height as f64 * fh / CARD_H as f64);
    let w = ((CARD_W as f64 * scale / fw).floor() as u16).clamp(1, area.width.max(1));
    let h = ((CARD_H as f64 * scale / fh).floor() as u16).clamp(1, area.height.max(1));
    Rect::new(area.x + (area.width - w) / 2, area.y + (area.height - h) / 2, w, h)
}
