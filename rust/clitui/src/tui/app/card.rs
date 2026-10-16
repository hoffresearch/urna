//! Laying out a hit's stored text by the template that rendered it. The
//! manifest names that template (`chunker_version`); a known one is drawn as
//! what it describes, anything else stays plain wrapped text. Only the
//! layout changes: every character shown is the stored canonical text, or
//! a field of the card when the corpus carries them (#625, until #208 puts
//! them in the file): cmc, popularity, stats and the formats it is legal in.
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

use super::fields::{self, Kind};
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

/// What a row that is not a playable card is, for a chip: by its kind when
/// the corpus carries fields, by the template's placeholder text otherwise.
pub fn stand_in(text: &str, card: Option<&fields::Card>) -> Option<&'static str> {
    match card {
        Some(c) => match c.kind {
            Kind::Card => None,
            Kind::ArtSeries => Some("art series"),
            Kind::Token => Some("token"),
            Kind::Emblem => Some("emblem"),
            Kind::Alchemy => Some("alchemy"),
            Kind::Unsanctioned => Some("unsanctioned"),
        },
        None => placeholder(text).then_some("art series"),
    }
}

/// The card's lines, `width` cells wide (for the rules divider); `card`
/// adds the fields the text does not carry.
pub fn lines(text: &str, width: u16, card: Option<&fields::Card>) -> Vec<Line<'static>> {
    let c = parse(text);
    let art_series = match card {
        Some(f) => f.kind == Kind::ArtSeries,
        None => c.type_line == "Card // Card" || c.type_line == "Card",
    };
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
    if let Some(kind) = stand_in(text, card) {
        tags.push(chip(kind, Color::Rgb(0xc4, 0x8c, 0xd8)));
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
    if let Some(f) = card {
        out.extend(facts(f, width));
    }
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

/// The fields as a line of facts and a line of formats: `cmc 1 · 3/3 ·
/// edhrec #12` and `legal  pauper, modern, …`, the latter cut to `width`.
fn facts(f: &fields::Card, width: u16) -> Vec<Line<'static>> {
    let mut parts = Vec::new();
    if let Some(v) = f.cmc {
        parts.push(format!("cmc {v}"));
    }
    match (&f.power, &f.toughness, &f.loyalty) {
        (Some(p), Some(t), _) => parts.push(format!("{p}/{t}")),
        (_, _, Some(l)) => parts.push(format!("loyalty {l}")),
        _ => {}
    }
    if let Some(r) = f.edhrec_rank {
        parts.push(format!("edhrec #{r}"));
    }
    let mut out = Vec::new();
    if !parts.is_empty() {
        out.push(Line::from(Span::styled(parts.join(" · "), pal::dim())));
    }
    let formats = f.formats().collect::<Vec<_>>().join(", ");
    if !formats.is_empty() {
        let room = usize::from(width).saturating_sub(7);
        let shown: String = if formats.chars().count() > room {
            formats
                .chars()
                .take(room.saturating_sub(1))
                .collect::<String>()
                + "…"
        } else {
            formats
        };
        out.push(Line::from(vec![
            Span::styled("legal  ", pal::faint()),
            Span::styled(shown, Style::new().fg(pal::HIGH)),
        ]));
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

    #[test]
    fn the_fields_add_facts_and_formats_and_name_the_stand_ins() {
        let json = r#"{"ordinal":0,"oracle_id":"o","name":"Grizzly Bears","pt_name":null,
            "mana_cost":"{1}{G}","type_line":"Creature — Bear","rarity":"common","set_code":"lea",
            "oracle_text":"","kind":"token","cmc":2.0,"power":"2","toughness":"2","loyalty":null,
            "layout":"token","legal":{"pauper":"legal","modern":"banned","legacy":"restricted"},
            "edhrec_rank":7}"#;
        let f: fields::Card = serde_json::from_str(json).expect("card");
        let text = "Grizzly Bears
{1}{G}; Creature — Bear
[common, lea]
";
        let plain: Vec<String> = lines(text, 40, Some(&f))
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        assert!(
            plain.iter().any(|l| l == "cmc 2 · 2/2 · edhrec #7"),
            "{plain:?}"
        );
        assert!(
            plain.iter().any(|l| l == "legal  legacy, pauper"),
            "{plain:?}"
        );
        assert!(plain.iter().any(|l| l.contains(" token ")), "{plain:?}");
        assert_eq!(stand_in(text, Some(&f)), Some("token"));
        assert_eq!(stand_in(text, None), None);
        assert_eq!(
            stand_in(
                "Sol Ring // Sol Ring
// ; Card // Card
[common, acmm]
//",
                None
            ),
            Some("art series")
        );
        // the cut keeps the line one row: `legal  ` and the ellipsis fit 12 cells.
        let narrow: Vec<String> = lines(text, 12, Some(&f))
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        assert!(narrow.iter().any(|l| l == "legal  lega…"), "{narrow:?}");
    }
}
