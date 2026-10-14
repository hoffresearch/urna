//! The corpus tab: what one `.urna` is, read the way `inspect` and
//! `validate` read it. Loading happens off the ui thread (a big corpus
//! takes a moment to hash); the tab renders the manifest as leader rows and
//! the section table with a size bar per section and a scrollbar.

use std::path::{Path, PathBuf};

use anyhow::Result;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::widgets::Widget;
use tui_scrollbar::{ScrollBar, ScrollLengths};

use crate::cmd::util::encoding_name;
use crate::tui::hud::{self, Badge};
use crate::tui::pal;

pub struct Section {
    pub id: u64,
    pub name: String,
    pub encoding: &'static str,
    pub size: u64,
}

pub struct Corpus {
    pub path: PathBuf,
    pub size: u64,
    /// label, value pairs in display order.
    pub facts: Vec<(&'static str, String)>,
    pub sections: Vec<Section>,
    pub content_hash: String,
    /// `Ok` when every checksum, hash and the contract hold.
    pub valid: Result<(), String>,
}

fn short(h: &str) -> String {
    let hex = h.trim_start_matches("sha256:");
    if hex.len() > 16 {
        format!("sha256:{}…{}", &hex[..10], &hex[hex.len() - 6..])
    } else {
        h.to_string()
    }
}

/// Opens, inspects and validates `path` (the same checks `urna validate`
/// runs on the header, sections, hashes and embedding values).
pub fn load(path: &Path) -> Result<Corpus> {
    let rt = urna_engine::MmapUrnaFile::open(path)?;
    let info: serde_json::Value = serde_json::from_str(&rt.inspect_json()?)?;
    let m = &info["manifest"];
    let s = |v: &serde_json::Value| {
        v.as_str()
            .map(str::to_string)
            .unwrap_or_else(|| v.to_string())
    };
    let size = info["file_size"].as_u64().unwrap_or(0);
    let facts = vec![
        ("chunks", s(&info["n_chunks"])),
        ("dim", s(&info["embedding_dim"])),
        ("dtype", s(&m["dtype"])),
        (
            "metric",
            format!("{} · {}", s(&m["metric"]), s(&m["score_type"])),
        ),
        (
            "index",
            format!(
                "{} · rerank {}",
                s(&m["index_type"]),
                s(&m["rerank_policy"])
            ),
        ),
        ("model", s(&m["embedding_model"])),
        ("model_hash", short(&s(&m["model_hash"]))),
        ("size", human(size)),
        ("file_hash", short(&s(&info["file_hash"]))),
        ("content_hash", short(&s(&info["content_hash"]))),
    ];
    let sections = info["sections"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|x| Section {
                    id: x["section_id"].as_u64().unwrap_or(0),
                    name: x["name"].as_str().unwrap_or("?").to_string(),
                    encoding: encoding_name(x["encoding"].as_u64().unwrap_or(0) as u32),
                    size: x["size"].as_u64().unwrap_or(0),
                })
                .collect()
        })
        .unwrap_or_default();
    let valid = (|| -> Result<()> {
        let data = std::fs::read(path)?;
        let view = urna_format::UrnaView::from_bytes(&data)?;
        view.validate_embeddings_values()?;
        view.search_contract()?;
        Ok(())
    })()
    .map_err(|e| format!("{e:#}"));
    Ok(Corpus {
        path: path.to_path_buf(),
        size,
        facts,
        sections,
        content_hash: s(&info["content_hash"]),
        valid,
    })
}

pub fn human(bytes: u64) -> String {
    match bytes {
        b if b >= 1 << 30 => format!("{:.2} GB", b as f64 / (1u64 << 30) as f64),
        b if b >= 1 << 20 => format!("{:.1} MB", b as f64 / (1u64 << 20) as f64),
        b if b >= 1 << 10 => format!("{:.1} KB", b as f64 / 1024.0),
        b => format!("{b} B"),
    }
}

pub fn render(buf: &mut Buffer, body: Rect, corpus: &Corpus, scroll: usize) {
    let lw = (body.width * 9 / 20).clamp(30.min(body.width), 56);
    let left = Rect::new(body.x + 1, body.y, lw, body.height);
    let name = corpus
        .path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let inner = hud::panel(buf, left, &name, "manifest", true);
    let (badge, verdict, vst) = match &corpus.valid {
        Ok(()) => (
            Badge::Ok,
            "valid: checksums, hashes, contract".to_string(),
            pal::ok(),
        ),
        Err(e) => (Badge::Fail, e.clone(), pal::err()),
    };
    let (b, bst) = hud::badge(badge);
    hud::put(buf, inner.x, inner.y, b, bst, 6);
    hud::put(
        buf,
        inner.x + 7,
        inner.y,
        &verdict,
        vst,
        inner.width.saturating_sub(7),
    );
    for (i, (k, v)) in corpus.facts.iter().enumerate() {
        let y = inner.y + 2 + i as u16;
        if y >= inner.bottom() {
            break;
        }
        hud::leader(
            buf,
            inner.x,
            y,
            inner.width,
            k,
            v,
            Style::new().fg(pal::INK),
        );
    }
    let cite_y = inner.y + 3 + corpus.facts.len() as u16;
    if cite_y + 1 < inner.bottom() {
        hud::put(buf, inner.x, cite_y, "citations", pal::dim(), inner.width);
        let uri = format!("urna://{}/<chunk_id>", short(&corpus.content_hash));
        hud::put(buf, inner.x, cite_y + 1, &uri, pal::accent(), inner.width);
    }
    let right = Rect::new(
        left.right() + 1,
        body.y,
        body.right().saturating_sub(left.right() + 2),
        body.height,
    );
    sections(buf, right, corpus, scroll);
}

fn sections(buf: &mut Buffer, area: Rect, corpus: &Corpus, scroll: usize) {
    let inner = hud::panel(
        buf,
        area,
        "sections",
        &format!("{} · {}", corpus.sections.len(), human(corpus.size)),
        false,
    );
    if inner.height < 2 {
        return;
    }
    let head = pal::faint();
    // the name column is as wide as the longest name, so narrow panels keep
    // room for the size bar.
    let name_w = corpus
        .sections
        .iter()
        .map(|s| s.name.len())
        .max()
        .unwrap_or(4)
        .clamp(4, 24)
        + 2;
    hud::spans(
        buf,
        inner.x,
        inner.y,
        &[
            ("id   ", head),
            (&format!("{:<name_w$}", "name"), head),
            ("encoding ", head),
            ("     size", head),
        ],
        inner.width,
    );
    let rows = inner.height as usize - 1;
    let max = corpus
        .sections
        .iter()
        .map(|s| s.size)
        .max()
        .unwrap_or(1)
        .max(1) as f64;
    let start = scroll.min(corpus.sections.len().saturating_sub(rows));
    let bar_x = inner.x + 5 + name_w as u16 + 9 + 9 + 2;
    let bar_w = inner.right().saturating_sub(bar_x + 2).min(24);
    for (i, s) in corpus.sections.iter().skip(start).take(rows).enumerate() {
        let y = inner.y + 1 + i as u16;
        let share = (s.size as f64).ln_1p() / max.ln_1p();
        hud::spans(
            buf,
            inner.x,
            y,
            &[
                (&format!("0x{:02x} ", s.id), pal::dim()),
                (&format!("{:<name_w$}", s.name), Style::new().fg(pal::INK)),
                (&format!("{:<9}", s.encoding), pal::accent()),
                (&format!("{:>9}", human(s.size)), pal::dim()),
            ],
            inner.width,
        );
        if bar_w >= 4 {
            hud::gauge(buf, bar_x, y, bar_w, share, pal::thermo(share as f32));
        }
    }
    if corpus.sections.len() > rows {
        ScrollBar::vertical(ScrollLengths {
            content_len: corpus.sections.len(),
            viewport_len: rows,
        })
        .offset(start)
        .track_style(Style::new().fg(pal::SURFACE))
        .thumb_style(Style::new().fg(pal::GLOW))
        .render(
            Rect::new(inner.right(), inner.y + 1, 1, inner.height - 1),
            buf,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn golden() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../format/tests/fixtures/golden_v1_minimal.urna")
    }

    #[test]
    fn loads_and_validates_the_golden_fixture() {
        let c = load(&golden()).unwrap();
        assert!(c.valid.is_ok());
        assert_eq!(c.size, 1366);
        assert!(!c.sections.is_empty());
        assert_eq!(c.facts[0].0, "chunks");
    }

    #[test]
    fn a_missing_file_is_an_error_not_a_panic() {
        assert!(load(Path::new("/urna/no/such.urna")).is_err());
    }

    #[test]
    fn render_fits_small_bodies() {
        let c = load(&golden()).unwrap();
        for (w, h) in [(40, 8), (120, 30), (12, 3)] {
            let area = Rect::new(0, 0, w, h);
            let mut buf = Buffer::empty(area);
            render(&mut buf, area, &c, 99);
        }
        assert_eq!(human(1366), "1.3 KB");
    }
}
