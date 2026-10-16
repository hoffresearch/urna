//! Laying out a hit's stored text by the template that rendered it. The
//! manifest names that template (`chunker_version`); a known one is drawn as
//! what it describes, anything else stays plain wrapped text. Only the
//! layout changes: every character shown is the stored canonical text.
//!
//! `mtgdataset/1` renders a card as
//!
//! ```text
//! Name (Nome em português)
//! {cost}; Type line
//! [rarity, set]
//! rules text, any number of lines
//! ```

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use crate::tui::pal;

/// Whether `chunker` is a template this module lays out.
pub fn knows(chunker: &str) -> bool {
    chunker == "mtgdataset/1"
}

/// A row that stands in for a card without being one: an art-series card,
/// a full-art print with no name, cost or rules.
pub fn placeholder(text: &str) -> bool {
    matches!(parse(text).type_line, "Card // Card" | "Card")
}

struct Card<'a> {
    name: &'a str,
    alias: Option<&'a str>,
    cost: &'a str,
    type_line: &'a str,
    rarity: &'a str,
    set: &'a str,
    rules: Vec<&'a str>,
}

fn parse(text: &str) -> Card<'_> {
    let mut lines = text.lines();
    let head = lines.next().unwrap_or("").trim();
    let (name, alias) = match head
        .strip_suffix(')')
        .and_then(|h| h.rfind(" (").map(|i| (i, h)))
    {
        Some((i, h)) => (&h[..i], Some(&h[i + 2..])),
        None => (head, None),
    };
    let ct = lines.next().unwrap_or("");
    let (cost, type_line) = ct.split_once("; ").unwrap_or(("", ct));
    let tags = lines
        .next()
        .unwrap_or("")
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']');
    let (rarity, set) = tags.split_once(", ").unwrap_or((tags, ""));
    Card {
        name,
        alias,
        cost: cost.trim().trim_start_matches(';').trim(),
        type_line: type_line.trim().trim_start_matches(';').trim(),
        rarity: rarity.trim(),
        set: set.trim(),
        rules: lines.filter(|l| l.trim() != "//").collect(),
    }
}

fn pip_color(sym: &str) -> Color {
    match sym {
        "W" => Color::Rgb(0xf3, 0xe6, 0xb8),
        "U" => Color::Rgb(0x6c, 0xa8, 0xe6),
        "B" => Color::Rgb(0xb4, 0x9c, 0xc8),
        "R" => Color::Rgb(0xe8, 0x74, 0x5c),
        "G" => Color::Rgb(0x74, 0xbe, 0x84),
        "C" => Color::Rgb(0xc8, 0xc8, 0xd0),
        "T" | "Q" => pal::DIM,
        s if s.contains('/') => Color::Rgb(0xd2, 0xb4, 0xe6),
        _ => Color::Rgb(0xd8, 0xd8, 0xdc),
    }
}

/// `{2}{U}{R}` as pips: each symbol on a dark chip in its color.
fn pips(cost: &str) -> Vec<Span<'static>> {
    let mut out = Vec::new();
    for part in cost.split_inclusive('}') {
        match part
            .trim()
            .strip_prefix('{')
            .and_then(|p| p.strip_suffix('}'))
        {
            Some(sym) => out.push(Span::styled(
                format!(" {sym} "),
                Style::new()
                    .fg(pip_color(sym))
                    .bg(pal::RAISED)
                    .add_modifier(Modifier::BOLD),
            )),
            None if part.trim() == "//" => out.push(Span::styled("  //  ", pal::faint())),
            None if !part.trim().is_empty() => {
                out.push(Span::styled(part.trim().to_string(), pal::dim()))
            }
            None => {}
        }
        out.push(Span::raw(" "));
    }
    out
}

/// A rules line with its `{…}` symbols as pips and the rest as text.
fn rules_line(line: &str) -> Line<'static> {
    let mut spans = Vec::new();
    let mut rest = line;
    while let Some(open) = rest.find('{') {
        let Some(close) = rest[open..].find('}') else {
            break;
        };
        if open > 0 {
            spans.push(Span::styled(
                rest[..open].to_string(),
                Style::new().fg(pal::INK),
            ));
        }
        let sym = &rest[open + 1..open + close];
        spans.push(Span::styled(
            sym.to_string(),
            Style::new()
                .fg(pip_color(sym))
                .bg(pal::RAISED)
                .add_modifier(Modifier::BOLD),
        ));
        rest = &rest[open + close + 1..];
    }
    if !rest.is_empty() {
        spans.push(Span::styled(rest.to_string(), Style::new().fg(pal::INK)));
    }
    Line::from(spans)
}

fn rarity_color(r: &str) -> Color {
    match r {
        "mythic" => Color::Rgb(0xec, 0x86, 0x3c),
        "rare" => Color::Rgb(0xd8, 0xb4, 0x5c),
        "uncommon" => Color::Rgb(0xa8, 0xbe, 0xcc),
        "special" | "bonus" => Color::Rgb(0xc4, 0x8c, 0xd8),
        _ => pal::DIM,
    }
}

fn chip(text: &str, fg: Color) -> Span<'static> {
    Span::styled(format!(" {text} "), Style::new().fg(fg).bg(pal::RAISED))
}

/// The card's lines, `width` cells wide (for the rules divider).
pub fn lines(text: &str, width: u16) -> Vec<Line<'static>> {
    let c = parse(text);
    let art_series = c.type_line == "Card // Card" || c.type_line == "Card";
    let mut out = Vec::new();
    out.push(Line::from(Span::styled(
        c.name.to_string(),
        Style::new().fg(pal::INK_HI).add_modifier(Modifier::BOLD),
    )));
    if let Some(a) = c.alias.filter(|a| *a != c.name) {
        out.push(Line::from(Span::styled(
            a.to_string(),
            Style::new().fg(pal::DIM).add_modifier(Modifier::ITALIC),
        )));
    }
    out.push(Line::raw(""));
    if !c.cost.is_empty() && c.cost != "//" {
        out.push(Line::from(pips(c.cost)));
        out.push(Line::raw(""));
    }
    if !art_series {
        out.push(Line::from(Span::styled(
            c.type_line.to_string(),
            Style::new()
                .fg(pal::TITLE_FROM)
                .add_modifier(Modifier::BOLD),
        )));
    }
    let mut tags = Vec::new();
    if art_series {
        tags.push(chip("art series", Color::Rgb(0xc4, 0x8c, 0xd8)));
        tags.push(Span::raw(" "));
    }
    if !c.rarity.is_empty() {
        tags.push(chip(c.rarity, rarity_color(c.rarity)));
        tags.push(Span::raw(" "));
    }
    if !c.set.is_empty() {
        tags.push(chip(&c.set.to_uppercase(), pal::DIM));
    }
    out.push(Line::from(tags));
    if !c.rules.is_empty() {
        out.push(Line::raw(""));
        out.push(Line::from(Span::styled(
            "─".repeat(usize::from(width.min(48))),
            Style::new().fg(pal::LINE),
        )));
        out.push(Line::raw(""));
        for (i, l) in c.rules.iter().enumerate() {
            if i > 0 {
                out.push(Line::raw(""));
            }
            out.push(rules_line(l));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_card_splits_into_its_parts_and_keeps_every_character() {
        let c = parse(
            "Lightning Bolt (Raio)\n{R}; Instant\n[uncommon, msc]\nLightning Bolt deals 3 damage to any target.",
        );
        assert_eq!(
            (c.name, c.alias, c.cost, c.type_line),
            ("Lightning Bolt", Some("Raio"), "{R}", "Instant")
        );
        assert_eq!((c.rarity, c.set), ("uncommon", "msc"));
        assert_eq!(
            c.rules,
            vec!["Lightning Bolt deals 3 damage to any target."]
        );
        let art = parse("Sol Ring // Sol Ring\n// ; Card // Card\n[common, acmm]\n//");
        assert_eq!((art.type_line, art.rules.len()), ("Card // Card", 0));
        assert!(knows("mtgdataset/1") && !knows("other/1"));
    }
}
