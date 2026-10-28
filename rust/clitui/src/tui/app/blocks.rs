//! An image drawn with characters, for terminals that draw no pixels. Each
//! cell is one glyph in two colors; the image is resampled to 8 x 24
//! subpixels per cell and every glyph the terminal draws (`graphics::glyphs`)
//! is tried against them: the two colors of a glyph are the means of the
//! subpixels it covers and of the ones it leaves, and the glyph kept is the
//! one whose two means explain the most of the cell. The means and the
//! error are taken in Oklab, where a distance is a perceived difference.

use std::sync::LazyLock;

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;

use crate::tui::graphics::Rgba;
use crate::tui::graphics::glyphs::{self, GH, GW, Glyph, Set};

type Lab = [f32; 3];

/// Two splits closer than this fraction of the score are a tie, which the
/// earlier glyph wins: the summed-area differences round at about 1e-6 of
/// it, and a flat cell would otherwise draw whichever glyph rounded up.
const TIE: f32 = 1e-5;

/// One drawn cell: the glyph, its foreground and its background.
pub type Cell = (&'static str, Color, Color);

/// The image's cells for a `w` x `h` area, row by row, drawn with `set`.
pub fn cells(img: &Rgba, w: u16, h: u16, set: Set) -> Vec<Cell> {
    let (cols, rows) = (usize::from(w), usize::from(h));
    if cols == 0 || rows == 0 {
        return Vec::new();
    }
    let small = img.resize((cols * GW) as u32, (rows * GH) as u32);
    let width = cols * GW;
    let lab: Vec<Lab> = small.data.as_chunks::<4>().0.iter().map(oklab).collect();
    let glyphs = glyphs::of(set);
    let mut sums = vec![[0f32; 3]; (GW + 1) * (GH + 1)];
    let mut out = Vec::with_capacity(cols * rows);
    for cy in 0..rows {
        for cx in 0..cols {
            integrate(&lab, width, cx * GW, cy * GH, &mut sums);
            out.push(best(glyphs, &sums));
        }
    }
    out
}

/// The summed-area table of one cell's subpixels: `sums[y][x]` holds the
/// sum of every subpixel above and left of (x, y).
fn integrate(lab: &[Lab], width: usize, x0: usize, y0: usize, sums: &mut [Lab]) {
    let stride = GW + 1;
    for y in 0..GH {
        let mut row = [0f32; 3];
        for x in 0..GW {
            let p = lab[(y0 + y) * width + x0 + x];
            let above = sums[y * stride + x + 1];
            for c in 0..3 {
                row[c] += p[c];
            }
            sums[(y + 1) * stride + x + 1] =
                [above[0] + row[0], above[1] + row[1], above[2] + row[2]];
        }
    }
}

fn rect_sum(sums: &[Lab], (x0, y0, x1, y1): glyphs::Rect) -> Lab {
    let stride = GW + 1;
    let at = |x: u8, y: u8| sums[usize::from(y) * stride + usize::from(x)];
    let (a, b, c, d) = (at(x1, y1), at(x0, y1), at(x1, y0), at(x0, y0));
    [
        a[0] - b[0] - c[0] + d[0],
        a[1] - b[1] - c[1] + d[1],
        a[2] - b[2] - c[2] + d[2],
    ]
}

/// The glyph and colors that explain the most of the cell: for a split into
/// sets of n1 and n2 subpixels with sums s1 and s2, the squared error left
/// is the cell's own energy minus |s1|²/n1 + |s2|²/n2, so the best split is
/// the one with the largest of that term.
fn best(glyphs: &'static [Glyph], sums: &[Lab]) -> Cell {
    let n = (GW * GH) as f32;
    let total = sums[sums.len() - 1];
    let mut pick: Option<(&'static Glyph, Lab, f32)> = None;
    for g in glyphs {
        let mut fg = [0f32; 3];
        for r in &g.rects {
            let s = rect_sum(sums, *r);
            for c in 0..3 {
                fg[c] += s[c];
            }
        }
        let nf = g.area as f32;
        let nb = n - nf;
        let bg = [total[0] - fg[0], total[1] - fg[1], total[2] - fg[2]];
        let score = energy(fg, nf) + energy(bg, nb);
        if pick.is_none_or(|(_, _, s)| score > s * (1.0 + TIE)) {
            pick = Some((g, fg, score));
        }
    }
    let Some((g, fg, _)) = pick else {
        return (" ", Color::Reset, Color::Reset);
    };
    let nf = g.area as f32;
    let nb = n - nf;
    let bg = [total[0] - fg[0], total[1] - fg[1], total[2] - fg[2]];
    let mean = |s: Lab, k: f32| [s[0] / k, s[1] / k, s[2] / k];
    let (f, b) = match (nf > 0.0, nb > 0.0) {
        (true, true) => (mean(fg, nf), mean(bg, nb)),
        (true, false) => (mean(fg, nf), mean(fg, nf)),
        _ => (mean(bg, nb), mean(bg, nb)),
    };
    (g.ch.as_str(), color(f), color(b))
}

fn energy(s: Lab, k: f32) -> f32 {
    if k > 0.0 {
        (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]) / k
    } else {
        0.0
    }
}

/// sRGB to linear light, one entry per byte value: the pane's 192
/// subpixels per cell would otherwise take three `powf` each.
fn to_linear(c: u8) -> f32 {
    static TABLE: LazyLock<[f32; 256]> = LazyLock::new(|| {
        std::array::from_fn(|i| {
            let c = i as f32 / 255.0;
            if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        })
    });
    TABLE[usize::from(c)]
}

fn to_srgb(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

/// sRGB to Oklab (Björn Ottosson's matrices).
fn oklab(px: &[u8; 4]) -> Lab {
    let (r, g, b) = (to_linear(px[0]), to_linear(px[1]), to_linear(px[2]));
    let l = (0.412_221_47 * r + 0.536_332_55 * g + 0.051_445_995 * b).cbrt();
    let m = (0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b).cbrt();
    let s = (0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b).cbrt();
    [
        0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
        1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
        0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
    ]
}

fn color(lab: Lab) -> Color {
    let l = lab[0] + 0.396_337_78 * lab[1] + 0.215_803_76 * lab[2];
    let m = lab[0] - 0.105_561_346 * lab[1] - 0.063_854_17 * lab[2];
    let s = lab[0] - 0.089_484_18 * lab[1] - 1.291_485_5 * lab[2];
    let (l, m, s) = (l * l * l, m * m * m, s * s * s);
    Color::Rgb(
        to_srgb(4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s),
        to_srgb(-1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s),
        to_srgb(-0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s),
    )
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

    /// a `w` x `h` image, `f(x, y)` per pixel.
    fn image(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 3]) -> Rgba {
        let mut data = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let [r, g, b] = f(x, y);
                data.extend_from_slice(&[r, g, b, 255]);
            }
        }
        Rgba::new(w, h, data).expect("image")
    }

    #[test]
    fn a_split_takes_the_glyph_that_draws_it_and_its_two_colors() {
        let (red, blue) = ([255, 0, 0], [0, 0, 255]);
        // left quarter red: the left one quarter block in every set.
        let img = image(8, 24, |x, _| if x < 2 { red } else { blue });
        for set in [Set::Blocks, Set::Sextants, Set::Octants] {
            let c = cells(&img, 1, 1, set);
            assert_eq!(c[0].0, "▎", "{set:?}");
            assert_eq!(c[0].1, Color::Rgb(255, 0, 0));
            assert_eq!(c[0].2, Color::Rgb(0, 0, 255));
        }
        // the top third red: a sextant where the font has them, else the
        // nearest block (the upper quarter is not in the basic set).
        let img = image(8, 24, |_, y| if y < 8 { red } else { blue });
        assert_eq!(cells(&img, 1, 1, Set::Sextants)[0].0, "\u{1FB02}");
        assert_eq!(cells(&img, 1, 1, Set::Blocks)[0].0.chars().count(), 1);
        // flat: one color, both sides the same.
        let flat = cells(&image(8, 24, |_, _| [9, 90, 200]), 1, 1, Set::Blocks);
        assert_eq!(flat[0].1, flat[0].2);
    }

    #[test]
    fn a_flat_cell_is_the_first_glyph_whatever_the_rounding() {
        let steps = || (0..=255u8).step_by(15);
        for rgb in steps().flat_map(|r| steps().flat_map(move |g| steps().map(move |b| [r, g, b])))
        {
            let img = image(8, 24, |_, _| rgb);
            for set in [Set::Blocks, Set::Sextants, Set::Octants] {
                for (ch, fg, bg) in cells(&img, 1, 1, set) {
                    assert_eq!(ch, " ", "{rgb:?} {set:?}");
                    assert_eq!(fg, bg);
                }
            }
        }
    }

    #[test]
    fn the_table_is_the_formula() {
        for c in 0..=255u8 {
            let x = f32::from(c) / 255.0;
            let want = if x <= 0.04045 {
                x / 12.92
            } else {
                ((x + 0.055) / 1.055).powf(2.4)
            };
            assert_eq!(to_linear(c), want);
        }
    }

    #[test]
    fn more_glyphs_never_draw_worse() {
        // a diagonal edge, which no block draws exactly.
        let img = image(64, 96, |x, y| {
            if 3 * x > 2 * y {
                [240, 200, 20]
            } else {
                [20, 30, 90]
            }
        });
        let err = |set: Set| -> f32 {
            let c = cells(&img, 8, 4, set);
            let small = img.resize(8 * GW as u32, 4 * GH as u32);
            let lab: Vec<Lab> = small.data.as_chunks::<4>().0.iter().map(oklab).collect();
            let mut e = 0.0;
            for (i, (ch, fg, bg)) in c.iter().enumerate() {
                let g = glyphs::of(set).iter().find(|g| g.ch == *ch);
                let (cx, cy) = (i % 8, i / 8);
                for y in 0..GH {
                    for x in 0..GW {
                        let inside = g.is_some_and(|g| {
                            g.rects.iter().any(|&(x0, y0, x1, y1)| {
                                (usize::from(x0)..usize::from(x1)).contains(&x)
                                    && (usize::from(y0)..usize::from(y1)).contains(&y)
                            })
                        });
                        let drawn = if inside { *fg } else { *bg };
                        let Color::Rgb(r, gg, b) = drawn else {
                            continue;
                        };
                        let q = oklab(&[r, gg, b, 255]);
                        let p = lab[(cy * GH + y) * 8 * GW + cx * GW + x];
                        e += (0..3).map(|k| (p[k] - q[k]).powi(2)).sum::<f32>();
                    }
                }
            }
            e
        };
        let (b, s, o) = (err(Set::Blocks), err(Set::Sextants), err(Set::Octants));
        assert!(s <= b + 1e-3 && o <= s + 1e-3, "{b} {s} {o}");
        assert!(o < b, "{b} {o}");
    }
}
