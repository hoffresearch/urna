//! The urna symbol and wordmark as terminal art. Rasterized from the site's
//! vectors (apps/web/public/images/logo-symbol.svg and
//! apps/web/assets/type-urna.svg) with rsvg-convert, 2x4 braille dots per
//! cell so the pixels stay square; the blank braille cell is a plain space
//! so whatever is behind (rain, an effect) shows through the art.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};

use super::pal;

/// 16 rows x 30 cols: the splash.
pub const SYMBOL_L: [&str; 16] = [
    "     ⠈⠻⣿⣷⣦⡀      ⢀⣴⣾⣿⠟      ",
    "       ⠈⢻⣿⣿⡆    ⢰⣿⣿⡟⠁       ",
    "        ⣸⣿⣿⣿    ⣿⣿⣿⣇        ",
    "       ⣰⣿⣿⣿⡟    ⢿⣿⣿⣿⣆       ",
    "     ⣠⣾⣿⣿⣿⡿⠁    ⠈⢿⣿⣿⣿⣷⣄     ",
    "   ⢠⣾⣿⣿⣿⣿⡟⠁      ⠈⢻⣿⣿⣿⣿⣷⡀   ",
    " ⢀⣴⣿⣿⣿⣿⣿⠟          ⠻⣿⣿⣿⣿⣿⣦  ",
    "⢀⣾⣿⣿⣿⣿⣿⠏            ⢹⣿⣿⣿⣿⣿⣧ ",
    "⣼⣿⣿⣿⣿⣿⡟              ⢿⣿⣿⣿⣿⣿⣇",
    "⣿⣿⣿⣿⣿⣿⡇              ⢸⣿⣿⣿⣿⣿⣿",
    "⣿⣿⣿⣿⣿⣿⡇              ⢸⣿⣿⣿⣿⣿⣿",
    "⢹⣿⣿⣿⣿⣿⣿⡀            ⢠⣿⣿⣿⣿⣿⣿⡏",
    " ⢻⣿⣿⣿⣿⣿⣿⣄         ⢀⣴⣿⣿⣿⣿⣿⣿⡟ ",
    "  ⠹⣿⣿⣿⣿⣿⣿⣿⣶⣤⣤⣀⣀⣤⣤⣶⣿⣿⣿⣿⣿⣿⣿⠟  ",
    "   ⠈⠻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠁   ",
    "      ⠉⠛⠿⠿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠟⠛⠉      ",
];

/// 10 rows x 18 cols: short terminals.
pub const SYMBOL_M: [&str; 10] = [
    "   ⠈⠻⣷⣄   ⣠⣾⠟⠁   ",
    "     ⣿⣿  ⢠⣿⣿     ",
    "   ⢀⣼⣿⡿   ⢿⣿⣧⡀   ",
    " ⢀⣴⣿⣿⡿⠁   ⠈⢿⣿⣿⣦⡀ ",
    "⢠⣾⣿⣿⡟       ⢻⣿⣿⣷⡀",
    "⣾⣿⣿⣿⠁       ⠈⣿⣿⣿⣷",
    "⣿⣿⣿⣿         ⣿⣿⣿⣿",
    "⠹⣿⣿⣿⣦      ⢀⣼⣿⣿⣿⠏",
    " ⠙⣿⣿⣿⣿⣦⣤⣤⣤⣶⣿⣿⣿⣿⠋ ",
    "   ⠙⠻⠿⣿⣿⣿⣿⣿⠿⠟⠋   ",
];

/// 3 rows x 5 cols: the header mark. Too small for a straight raster (the
/// horns vanish at 12 dots), so it is drawn dot by dot on the coverage
/// the svg has at 10x12: the tips flaring out, the teardrop slit, the full
/// round base.
pub const MARK: [&str; 3] = [" ⣳ ⣞ ", "⣾⠃ ⠘⣷", "⠹⢷⣤⡾⠏"];

/// 4 rows x 34 cols: the wordmark.
pub const WORD: [&str; 4] = [
    "⣶⣶⣶⡆   ⢰⣶⣶⡆⢰⣶⣶⣶⣠⣶⣾⡇⣶⣶⣶⡆⣤⣶⣿⣿⣷⣦⡀  ⣶⣾⣿⣿⣿⣿⣷⣦⡀",
    "⣿⣿⣿⡇   ⢸⣿⣿⡇⢸⣿⣿⣿⡿⠛⠉⠁⣿⣿⣿⡿⠋⠉⠉⢹⣿⣿⣿  ⣋⣩⣥⣤⣬⣽⣿⣿⣧",
    "⣿⣿⣿⣇⣀⣀⣠⣿⣿⣿⡇⢸⣿⣿⣿    ⣿⣿⣿⡇   ⢸⣿⣿⣿⢠⣾⣿⣿⠋⠉⢉⣹⣿⣿⣿",
    "⠙⠿⣿⣿⣿⡿⠟⢹⣿⣿⡇⢸⣿⣿⡿    ⢿⣿⣿⡇   ⠸⣿⣿⡿⠈⠻⢿⣿⣿⣿⠿⠛⣿⣿⣿",
];

/// Display width of a row of art (every glyph used is one column).
pub fn width(art: &[&str]) -> u16 {
    art.iter().map(|l| l.chars().count()).max().unwrap_or(0) as u16
}

/// Paints art at (x, y) with the site's title gradient running top to
/// bottom. Spaces are skipped, so the background stays untouched.
pub fn paint(buf: &mut Buffer, x: u16, y: u16, art: &[&str], from: Color, to: Color) {
    let n = art.len().max(2) as f32 - 1.0;
    for (row, line) in art.iter().enumerate() {
        let style = Style::new()
            .fg(pal::mix(from, to, row as f32 / n))
            .add_modifier(Modifier::BOLD);
        for (col, ch) in line.chars().enumerate() {
            if ch == ' ' {
                continue;
            }
            let pos = (x + col as u16, y + row as u16);
            if let Some(cell) = buf.cell_mut(pos) {
                cell.set_char(ch).set_style(style);
            }
        }
    }
}

/// The header mark at (x, y) in the title gradient; returns its rect.
pub fn mark(buf: &mut Buffer, x: u16, y: u16) -> Rect {
    paint(buf, x, y, &MARK, pal::TITLE_FROM, pal::TITLE_TO);
    Rect::new(x, y, width(&MARK), MARK.len() as u16).intersection(buf.area)
}

/// Paints art centered horizontally in `area`, starting at `area.y`.
/// Returns the rect the art occupies (clipped to `area`).
pub fn paint_centered(buf: &mut Buffer, area: Rect, art: &[&str]) -> Rect {
    let w = width(art).min(area.width);
    let x = area.x + (area.width - w) / 2;
    paint(buf, x, area.y, art, pal::TITLE_FROM, pal::TITLE_TO);
    Rect::new(x, area.y, w, (art.len() as u16).min(area.height))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_row_of_an_art_has_the_same_width() {
        for art in [&SYMBOL_L[..], &SYMBOL_M[..], &MARK[..], &WORD[..]] {
            let w = art[0].chars().count();
            assert!(art.iter().all(|l| l.chars().count() == w));
        }
    }

    #[test]
    fn paint_leaves_blank_cells_alone() {
        let area = Rect::new(0, 0, 18, 10);
        let mut buf = Buffer::empty(area);
        buf[(0, 0)].set_char('x');
        paint(&mut buf, 0, 0, &SYMBOL_M, pal::TITLE_FROM, pal::TITLE_TO);
        assert_eq!(buf[(0, 0)].symbol(), "x");
        assert_eq!(buf[(3, 0)].symbol(), "⠈");
    }
}
