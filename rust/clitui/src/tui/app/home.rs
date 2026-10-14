//! The home tab: the symbol over the hash rain, the wordmark, and the
//! ways in: open a file, the `.urna` files found near the working
//! directory (1-9), setup and health. with no corpus to reopen it is the
//! first-run screen: open the sample corpus, or open a `.urna` file.

use std::path::{Path, PathBuf};
use std::time::Duration;

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Style;

use super::corpus::human;
use crate::tui::rain::Rain;
use crate::tui::{art, hud, pal};

/// `.urna` files in the working directory and one level below, largest
/// first, at most nine (one digit each).
pub fn recent(cwd: &Path) -> Vec<(PathBuf, u64)> {
    let mut found = Vec::new();
    let mut scan = |dir: &Path| {
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().is_some_and(|x| x == "urna")
                    && let Ok(m) = e.metadata()
                    && m.is_file()
                {
                    found.push((p, m.len()));
                }
            }
        }
    };
    scan(cwd);
    if let Ok(rd) = std::fs::read_dir(cwd) {
        for e in rd.flatten().filter(|e| e.path().is_dir()) {
            let name = e.file_name();
            if !name.to_string_lossy().starts_with('.') {
                scan(&e.path());
            }
        }
    }
    found.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    found.truncate(9);
    found
}

/// What the home tab offers under the wordmark.
pub enum Menu<'a> {
    /// the `.urna` files near the working directory.
    Near(&'a [(PathBuf, u64)], &'a Path),
    /// nothing to reopen: the sample corpus, when setup laid it down, and
    /// the picker.
    First(Option<&'a Path>),
}

impl Menu<'_> {
    fn rows(&self) -> u16 {
        match self {
            Menu::Near(files, _) => files.len().max(1) as u16,
            Menu::First(_) => 4,
        }
    }
}

/// Draws the home tab; returns the rect of the symbol (for the effects).
pub fn render(buf: &mut Buffer, body: Rect, elapsed: Duration, menu: &Menu<'_>) -> Rect {
    Rain::quiet(elapsed).render(body, buf);
    let tall = body.height >= 30 && body.width >= 70;
    let symbol: &[&str] = if tall { &art::SYMBOL_L } else { &art::SYMBOL_M };
    let block_h = symbol.len() as u16 + 6 + 2 + menu.rows() + 5;
    let top = body.y + body.height.saturating_sub(block_h) / 2;
    let sym = art::paint_centered(
        buf,
        Rect::new(body.x, top, body.width, symbol.len() as u16),
        symbol,
    );
    let wy = sym.bottom() + 1;
    let word = art::width(&art::WORD);
    let col_w = 58.min(body.width.saturating_sub(4));
    let cx = body.x + body.width.saturating_sub(col_w) / 2;
    buf.set_style(
        Rect::new(
            cx.saturating_sub(2),
            wy,
            col_w + 4,
            body.bottom().saturating_sub(wy),
        )
        .intersection(body),
        Style::new().bg(pal::BG),
    );
    for y in wy..body.bottom() {
        for x in cx.saturating_sub(2)..(cx + col_w + 2).min(body.right()) {
            if let Some(c) = buf.cell_mut((x, y)) {
                c.set_char(' ');
            }
        }
    }
    if body.width >= word + 4 && wy + 4 < body.bottom() {
        art::paint(
            buf,
            body.x + (body.width - word) / 2,
            wy,
            &art::WORD,
            pal::INK_HI,
            pal::TITLE_TO,
        );
    }
    let mut y = wy + 5;
    let center = |buf: &mut Buffer, y: u16, s: &str, st: Style| {
        let w = (s.chars().count() as u16).min(body.width);
        hud::put(buf, body.x + (body.width - w) / 2, y, s, st, w);
    };
    center(
        buf,
        y,
        "single-file · memory-mapped · hash-verified · offline",
        pal::dim(),
    );
    y += 2;
    let row = |buf: &mut Buffer, y: u16, k: &str, what: &str, detail: &str| {
        if y < body.bottom() {
            let x = hud::spans(
                buf,
                cx,
                y,
                &[
                    (k, pal::key()),
                    ("  ", pal::text()),
                    (what, Style::new().fg(pal::INK)),
                ],
                col_w,
            );
            hud::put(
                buf,
                x + 1,
                y,
                detail,
                pal::faint(),
                (cx + col_w).saturating_sub(x + 1),
            );
        }
    };
    let mut setup_row = true;
    match menu {
        Menu::First(sample) => {
            let (k, what, detail) = match sample {
                Some(_) => ("enter", "open the sample corpus", ""),
                None => ("s", "get the sample corpus", "setup downloads it"),
            };
            setup_row = sample.is_some();
            row(buf, y, k, what, detail);
            y += 1;
            // what the sample holds, under its action: two lines at most.
            let x = cx + k.chars().count() as u16 + 2;
            let w = (cx + col_w).saturating_sub(x);
            for line in hud::wrap(crate::tui::setup::sample::ABOUT, w as usize)
                .iter()
                .take(2)
            {
                if y < body.bottom() {
                    hud::put(buf, x, y, line, pal::faint(), w);
                }
                y += 1;
            }
            row(buf, y, "o", "open a .urna file", "");
            y += 1;
        }
        Menu::Near([], cwd) => {
            row(
                buf,
                y,
                "o",
                "open a .urna",
                &format!(
                    "no .urna under {}",
                    hud::tail(&cwd.display().to_string(), 30)
                ),
            );
            y += 1;
        }
        Menu::Near(files, cwd) => {
            for (i, (p, size)) in files.iter().enumerate() {
                let shown = p.strip_prefix(cwd).unwrap_or(p).display().to_string();
                row(
                    buf,
                    y,
                    &format!("{}", i + 1),
                    &hud::tail(&shown, 34),
                    &human(*size),
                );
                y += 1;
            }
            row(buf, y, "o", "open another file", "");
            y += 1;
        }
    }
    if setup_row {
        row(
            buf,
            y + 1,
            "s",
            "setup",
            "embedder payload, python env, doctor",
        );
    }
    let y = if setup_row { y + 1 } else { y };
    row(buf, y + 1, "h", "health", "the doctor checks, live");
    if y + 3 < body.bottom() {
        hud::link(buf, cx, y + 3, "urna.dev", "https://urna.dev", col_w);
    }
    sym
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recent_finds_urna_files_one_level_down() {
        let d = std::env::temp_dir().join(format!("urna_home_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("sub")).unwrap();
        std::fs::write(d.join("a.urna"), b"x").unwrap();
        std::fs::write(d.join("sub/b.urna"), b"xx").unwrap();
        std::fs::write(d.join("c.txt"), b"x").unwrap();
        let r = recent(&d);
        assert_eq!(r.len(), 2);
        assert!(r[0].0.ends_with("sub/b.urna"));
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn render_survives_any_size() {
        let sample = Path::new("/tmp/sample/fakenews.urna");
        let menus = [
            Menu::Near(&[], Path::new("/tmp")),
            Menu::First(None),
            Menu::First(Some(sample)),
        ];
        for (w, h) in [(20, 5), (80, 24), (160, 50)] {
            for menu in &menus {
                let area = Rect::new(0, 0, w, h);
                let mut buf = Buffer::empty(area);
                render(&mut buf, area, Duration::from_secs(2), menu);
            }
        }
    }
}
