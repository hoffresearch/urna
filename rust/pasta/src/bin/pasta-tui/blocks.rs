//! An image drawn with characters, for terminals that draw no pixels
//! (Terminal.app, Zed, most others). Each cell is split in 2x2 quadrants
//! and gets the quadrant glyph and the two colors that reproduce its four
//! pixels best: twice the horizontal detail of halfblocks, with glyphs
//! (U+2580..U+259F, block elements) that every monospace font carries.

use image::RgbaImage;
use image::imageops::{self, FilterType};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;

/// The quadrant glyph for a mask: bit 0 top-left, 1 top-right, 2 bottom-left,
/// 3 bottom-right; a set bit takes the foreground color.
const GLYPH: [&str; 16] = [
    " ", "▘", "▝", "▀", "▖", "▌", "▞", "▛", "▗", "▚", "▐", "▜", "▄", "▙", "▟", "█",
];

type Rgb = [f32; 3];

/// One drawn cell: the quadrant glyph, its foreground and its background.
pub type Cell = (&'static str, Color, Color);

fn mean(px: &[Rgb], mask: u8, set: bool) -> Rgb {
    let (mut acc, mut n) = ([0f32; 3], 0f32);
    for (i, p) in px.iter().enumerate() {
        if ((mask >> i) & 1 == 1) == set {
            for c in 0..3 {
                acc[c] += p[c];
            }
            n += 1.0;
        }
    }
    if n == 0.0 {
        acc
    } else {
        [acc[0] / n, acc[1] / n, acc[2] / n]
    }
}

fn dist(a: Rgb, b: Rgb) -> f32 {
    // weighted toward green, as the eye is.
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    0.3 * d[0] * d[0] + 0.59 * d[1] * d[1] + 0.11 * d[2] * d[2]
}

/// The mask and the two colors that best reproduce four pixels.
fn best(px: &[Rgb; 4]) -> (u8, Rgb, Rgb) {
    let err = |mask: u8, fg: Rgb, bg: Rgb| -> f32 {
        px.iter()
            .enumerate()
            .map(|(i, p)| dist(*p, if (mask >> i) & 1 == 1 { fg } else { bg }))
            .sum()
    };
    // one flat color first, so a split has to do strictly better to win.
    let flat = mean(px, 15, true);
    let mut pick = (15u8, flat, flat, err(15, flat, flat));
    // masks 1..=7 cover every split once (8..=14 are their inverses).
    for mask in 1u8..=7 {
        let (fg, bg) = (mean(px, mask, true), mean(px, mask, false));
        let e = err(mask, fg, bg);
        if e < pick.3 {
            pick = (mask, fg, bg, e);
        }
    }
    (pick.0, pick.1, pick.2)
}

fn color(c: Rgb) -> Color {
    Color::Rgb(c[0] as u8, c[1] as u8, c[2] as u8)
}

/// The image's cells for `area`, one (glyph, fg, bg) per cell, row by row.
/// The image is resampled to 2 pixels per cell across and 2 down; the cell
/// is about twice as tall as wide, so a fitted `area` keeps the aspect.
pub fn cells(img: &RgbaImage, w: u16, h: u16) -> Vec<Cell> {
    let (pw, ph) = (u32::from(w) * 2, u32::from(h) * 2);
    if pw == 0 || ph == 0 {
        return Vec::new();
    }
    let small = imageops::resize(img, pw, ph, FilterType::Lanczos3);
    let at = |x: u32, y: u32| -> Rgb {
        let p = small.get_pixel(x, y).0;
        [f32::from(p[0]), f32::from(p[1]), f32::from(p[2])]
    };
    let mut out = Vec::with_capacity(usize::from(w) * usize::from(h));
    for cy in 0..u32::from(h) {
        for cx in 0..u32::from(w) {
            let (x, y) = (cx * 2, cy * 2);
            let px = [at(x, y), at(x + 1, y), at(x, y + 1), at(x + 1, y + 1)];
            let (mask, fg, bg) = best(&px);
            out.push((GLYPH[usize::from(mask)], color(fg), color(bg)));
        }
    }
    out
}

pub fn draw(buf: &mut Buffer, area: Rect, cells: &[Cell]) {
    for (i, (glyph, fg, bg)) in cells.iter().enumerate() {
        let (x, y) = (i as u16 % area.width.max(1), i as u16 / area.width.max(1));
        if y >= area.height {
            break;
        }
        if let Some(cell) = buf.cell_mut((area.x + x, area.y + y)) {
            cell.set_symbol(glyph).set_fg(*fg).set_bg(*bg);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_split_cell_takes_its_two_colors_and_a_flat_one_a_full_block() {
        let (r, b) = ([255.0, 0.0, 0.0], [0.0, 0.0, 255.0]);
        // left column red, right column blue: the left-half glyph.
        let (mask, fg, bg) = best(&[r, b, r, b]);
        assert_eq!(GLYPH[usize::from(mask)], "▌");
        assert_eq!((fg, bg), (r, b));
        let (mask, ..) = best(&[r, r, r, r]);
        assert_eq!(GLYPH[usize::from(mask)], "█");
    }
}
