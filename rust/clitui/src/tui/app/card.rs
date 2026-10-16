//! Laying out a hit from the manifest schema: each field says how it is
//! drawn (`text`, `chip`, `symbols`, `number`), the label fields head the
//! view, a `chunk` field takes its lines from the stored canonical text,
//! and an `enum_map` group becomes one line of the keys that hold. No
//! field name lives here: another corpus with a schema gets its own view,
//! and a corpus without one stays plain wrapped text. Every character
//! shown comes from the file.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use urna_engine::scalars::Scalars;

use super::session::{Session, label_fields, values_at};
use crate::tui::pal;

/// Whether the session carries a schema this module can lay out.
pub fn knows(session: &Session) -> bool {
    session.schema().iter().any(|f| f.has_role("display"))
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

fn pip(sym: &str) -> Span<'static> {
    Span::styled(
        format!(" {sym} "),
        Style::new()
            .fg(pip_color(sym))
            .bg(pal::RAISED)
            .add_modifier(Modifier::BOLD),
    )
}

/// `{2}{U}{R}` as pips, or bare values (`R`, `G`) each as one pip.
fn symbols(values: &[String]) -> Vec<Span<'static>> {
    let mut out = Vec::new();
    for v in values {
        if v.contains('{') {
            for part in v.split_inclusive('}') {
                match part
                    .trim()
                    .strip_prefix('{')
                    .and_then(|p| p.strip_suffix('}'))
                {
                    Some(sym) => out.push(pip(sym)),
                    None if part.trim() == "//" => out.push(Span::styled("  //  ", pal::faint())),
                    None if !part.trim().is_empty() => {
                        out.push(Span::styled(part.trim().to_string(), pal::dim()))
                    }
                    None => {}
                }
                out.push(Span::raw(" "));
            }
        } else {
            out.push(pip(v));
            out.push(Span::raw(" "));
        }
    }
    out
}

/// A text line with its `{…}` symbols as pips and the rest as text.
fn rich_line(line: &str) -> Line<'static> {
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
        spans.push(pip(&rest[open + 1..open + close]));
        rest = &rest[open + close + 1..];
    }
    if !rest.is_empty() {
        spans.push(Span::styled(rest.to_string(), Style::new().fg(pal::INK)));
    }
    Line::from(spans)
}

fn chip_color(value: &str) -> Color {
    match value {
        "mythic" => Color::Rgb(0xec, 0x86, 0x3c),
        "rare" => Color::Rgb(0xd8, 0xb4, 0x5c),
        "uncommon" => Color::Rgb(0xa8, 0xbe, 0xcc),
        "special" | "bonus" | "art_series" | "token" | "emblem" => Color::Rgb(0xc4, 0x8c, 0xd8),
        _ => pal::DIM,
    }
}

fn chip(text: &str) -> Span<'static> {
    Span::styled(
        format!(" {} ", text.replace('_', " ")),
        Style::new().fg(chip_color(text)).bg(pal::RAISED),
    )
}

/// The keys of an `enum_map` group that hold for `row`: `legal.pauper`
/// with value `legal` or `restricted` yields `pauper`.
fn map_keys(s: &Scalars, group: &str, row: usize) -> Vec<String> {
    let prefix = format!("{group}.");
    s.table
        .columns
        .iter()
        .filter(|c| c.name.starts_with(&prefix) && c.present(row))
        .filter(|c| {
            values_at(s, &c.name, row)
                .first()
                .is_some_and(|v| v == "legal" || v == "restricted" || v == "true")
        })
        .map(|c| c.name[prefix.len()..].to_string())
        .collect()
}

fn cut(text: String, width: usize) -> String {
    if text.chars().count() > width {
        text.chars()
            .take(width.saturating_sub(1))
            .collect::<String>()
            + "…"
    } else {
        text
    }
}

/// The lines of chunk `row`: the schema's view when the corpus has one,
/// else the stored text. `width` bounds the long lines.
pub fn lines(session: &Session, row: usize, text: &str, width: u16) -> Vec<Line<'static>> {
    let Some(s) = session.scalars().filter(|_| knows(session)) else {
        return text
            .lines()
            .map(|l| Line::from(Span::styled(l.to_string(), Style::new().fg(pal::INK))))
            .collect();
    };
    let w = usize::from(width).max(8);
    let mut out = Vec::new();
    let labels = label_fields(&s.schema);
    for (i, f) in labels.iter().enumerate() {
        for v in values_at(s, &f.name, row) {
            out.push(Line::from(Span::styled(
                v,
                if i == 0 {
                    Style::new().fg(pal::INK_HI).add_modifier(Modifier::BOLD)
                } else {
                    pal::dim()
                },
            )));
        }
    }
    let mut chips: Vec<Span<'static>> = Vec::new();
    let mut facts: Vec<String> = Vec::new();
    let mut later: Vec<Line<'static>> = Vec::new();
    for f in s.schema.iter().filter(|f| f.has_role("display")) {
        if labels.iter().any(|l| l.name == f.name) {
            continue;
        }
        let name = f.label.clone().unwrap_or_else(|| f.name.clone());
        match (f.kind.as_str(), f.display.as_deref().unwrap_or("text")) {
            ("chunk", _) => {
                let from = f
                    .unit
                    .as_deref()
                    .and_then(|u| u.parse::<usize>().ok())
                    .unwrap_or(0);
                let body: Vec<&str> = text
                    .lines()
                    .skip(from)
                    .filter(|l| l.trim() != "//")
                    .collect();
                if !body.is_empty() {
                    later.push(Line::raw(""));
                    later.push(Line::from(Span::styled(
                        "─".repeat(w.min(60)),
                        pal::faint(),
                    )));
                    later.extend(body.into_iter().map(rich_line));
                }
            }
            ("enum_map", _) => {
                let keys = map_keys(s, &f.name, row);
                if !keys.is_empty() {
                    later.push(Line::from(vec![
                        Span::styled(format!("{}  ", name.to_lowercase()), pal::faint()),
                        Span::styled(
                            cut(keys.join(", "), w.saturating_sub(name.len() + 2)),
                            Style::new().fg(pal::HIGH),
                        ),
                    ]));
                }
            }
            (_, "symbols") => {
                let v = values_at(s, &f.name, row);
                if !v.is_empty() {
                    out.push(Line::from(symbols(&v)));
                }
            }
            (_, "chip") => {
                for v in values_at(s, &f.name, row) {
                    chips.push(chip(&v));
                    chips.push(Span::raw(" "));
                }
            }
            (_, "number") => {
                let v = values_at(s, &f.name, row);
                if !v.is_empty() {
                    facts.push(format!("{} {}", name.to_lowercase(), v.join("/")));
                }
            }
            (_, _) => {
                for v in values_at(s, &f.name, row) {
                    out.push(Line::from(Span::styled(v, Style::new().fg(pal::INK))));
                }
            }
        }
    }
    if !chips.is_empty() {
        out.push(Line::from(chips));
    }
    if !facts.is_empty() {
        out.push(Line::from(Span::styled(
            cut(facts.join(" · "), w),
            pal::dim(),
        )));
    }
    out.extend(later);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn without_a_schema_the_text_is_shown_as_stored() {
        let golden = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../format/tests/fixtures/golden_v1_minimal.urna");
        let session = Session::open(&golden).expect("golden opens");
        assert!(!knows(&session));
        let text = "first line\n{R} second";
        let plain: Vec<String> = lines(&session, 0, text, 40)
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        assert_eq!(plain, vec!["first line", "{R} second"]);
    }
}
