//! The glyphs an image can be drawn with, each as the exact part of the
//! cell it covers on a grid of 8 x 24 subpixels (8 across for the eighths,
//! 24 down for the eighths, the sextants' thirds and the octants' quarters).
//! Three sets, by what the terminal's font is known to carry: the block
//! elements of U+2580..U+259F and the box-drawing lines of U+2500..U+257B,
//! in every monospace font (the symbols chafa and TerminalImageViewer draw
//! with everywhere); plus the sextants and the quarter blocks of Unicode
//! 13 (U+1FB00..U+1FB8B); plus the octants of Unicode 16. A font cannot be
//! asked which glyphs it has, so the set comes from the terminal
//! (`graphics::detect`), never from a guess at the font.

use std::sync::OnceLock;

/// Subpixels across and down one cell.
pub const GW: usize = 8;
pub const GH: usize = 24;

/// Which glyphs the terminal draws.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Set {
    #[default]
    Blocks,
    Sextants,
    Octants,
}

impl Set {
    pub fn name(self) -> &'static str {
        match self {
            Self::Blocks => "blocks",
            Self::Sextants => "sextants",
            Self::Octants => "octants",
        }
    }
}

/// A half-open rectangle of subpixels: x0, y0, x1, y1.
pub type Rect = (u8, u8, u8, u8);

/// A glyph and the rectangles it covers.
#[derive(Debug)]
pub struct Glyph {
    pub ch: String,
    pub rects: Vec<Rect>,
    /// covered subpixels.
    pub area: u32,
}

/// The glyphs of `set`, each coverage once.
pub fn of(set: Set) -> &'static [Glyph] {
    static BLOCKS: OnceLock<Vec<Glyph>> = OnceLock::new();
    static SEXTANTS: OnceLock<Vec<Glyph>> = OnceLock::new();
    static OCTANTS: OnceLock<Vec<Glyph>> = OnceLock::new();
    match set {
        Set::Blocks => BLOCKS.get_or_init(|| build(set)),
        Set::Sextants => SEXTANTS.get_or_init(|| build(set)),
        Set::Octants => OCTANTS.get_or_init(|| build(set)),
    }
}

fn build(set: Set) -> Vec<Glyph> {
    let mut out = Glyphs::default();
    let (w, h) = (GW as u8, GH as u8);
    out.add(' ', vec![]);
    out.add('█', vec![(0, 0, w, h)]);
    for k in 1..8u8 {
        // lower and left eighths, U+2581..U+2587 and U+2589..U+258F
        out.add_cp(0x2580 + u32::from(k), vec![(0, h - 3 * k, w, h)]);
        out.add_cp(0x2590 - u32::from(k), vec![(0, 0, k, h)]);
    }
    out.add('▀', vec![(0, 0, w, 12)]);
    out.add('▐', vec![(4, 0, w, h)]);
    out.add('▔', vec![(0, 0, w, 3)]);
    out.add('▕', vec![(7, 0, w, h)]);
    // quadrants: bit 0 top left, 1 top right, 2 bottom left, 3 bottom right
    const QUAD: [char; 16] = [
        ' ', '▘', '▝', '▀', '▖', '▌', '▞', '▛', '▗', '▚', '▐', '▜', '▄', '▙', '▟', '█',
    ];
    for (m, ch) in QUAD.iter().enumerate() {
        out.add(*ch, grid(m as u32, 2, 2));
    }
    lines(&mut out);
    if set == Set::Blocks {
        return out.0;
    }
    // sextants, U+1FB00..U+1FB3B: every 2 x 3 pattern but the empty, the
    // full and the two half blocks, in pattern order (bit 0 top left).
    let mut cp = 0x1FB00;
    for m in 1..63u32 {
        if m == 21 || m == 42 {
            continue;
        }
        out.add_cp(cp, grid(m, 2, 3));
        cp += 1;
    }
    for (i, k) in [2u8, 3, 5, 6, 7].into_iter().enumerate() {
        // upper and right quarter to seven eighths, U+1FB82..U+1FB8B
        out.add_cp(0x1FB82 + i as u32, vec![(0, 0, w, 3 * k)]);
        out.add_cp(0x1FB87 + i as u32, vec![(w - k, 0, w, h)]);
    }
    if set == Set::Sextants {
        return out.0;
    }
    for (m, cp) in OCTANTS.iter().enumerate() {
        out.add_cp(*cp, grid(m as u32, 2, 4));
    }
    out.0
}

/// The box-drawing lines, light and heavy: the four half-segments from the
/// cell's centre (left, right, up, down) and every glyph U+2500..U+257B
/// makes of them: lines, corners, tees, crosses and the lone halves.
fn lines(out: &mut Glyphs) {
    // (left, right, up, down) rectangles of one weight
    let light = [
        (0, 11, 4, 13),
        (4, 11, 8, 13),
        (4, 0, 5, 12),
        (4, 12, 5, 24),
    ];
    let heavy = [
        (0, 10, 4, 14),
        (4, 10, 8, 14),
        (3, 0, 5, 12),
        (3, 12, 5, 24),
    ];
    // (light codepoint, heavy codepoint, which segments)
    const L: u8 = 1;
    const R: u8 = 2;
    const U: u8 = 4;
    const D: u8 = 8;
    let shapes: [(u32, u32, u8); 15] = [
        (0x2500, 0x2501, L | R),
        (0x2502, 0x2503, U | D),
        (0x250C, 0x250F, R | D),
        (0x2510, 0x2513, L | D),
        (0x2514, 0x2517, R | U),
        (0x2518, 0x251B, L | U),
        (0x251C, 0x2523, U | D | R),
        (0x2524, 0x252B, U | D | L),
        (0x252C, 0x2533, L | R | D),
        (0x2534, 0x253B, L | R | U),
        (0x253C, 0x254B, L | R | U | D),
        (0x2574, 0x2578, L),
        (0x2575, 0x2579, U),
        (0x2576, 0x257A, R),
        (0x2577, 0x257B, D),
    ];
    for (light_cp, heavy_cp, segs) in shapes {
        for (weight, cp) in [(&light, light_cp), (&heavy, heavy_cp)] {
            let rects = (0..4)
                .filter(|i| segs >> i & 1 == 1)
                .map(|i| weight[i])
                .collect();
            out.add_cp(cp, rects);
        }
    }
}

/// The rectangles of pattern `m` on a `cols` x `rows` grid, bit 0 top
/// left, row by row.
fn grid(m: u32, cols: u8, rows: u8) -> Vec<Rect> {
    let (cw, rh) = (GW as u8 / cols, GH as u8 / rows);
    (0..cols * rows)
        .filter(|b| m >> b & 1 == 1)
        .map(|b| {
            let (c, r) = (b % cols, b / cols);
            (c * cw, r * rh, (c + 1) * cw, (r + 1) * rh)
        })
        .collect()
}

#[derive(Default)]
struct Glyphs(Vec<Glyph>, Vec<[u64; 3]>);

impl Glyphs {
    /// Adds the glyph covering `rects` (which may overlap: the coverage is
    /// their union, stored again as rows of runs, so sums over it count
    /// each subpixel once); a coverage already present is skipped.
    fn add(&mut self, ch: char, rects: Vec<Rect>) {
        let mut mask = [0u64; 3];
        for &(x0, y0, x1, y1) in &rects {
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = usize::from(y) * GW + usize::from(x);
                    mask[i / 64] |= 1 << (i % 64);
                }
            }
        }
        if self.1.contains(&mask) {
            return;
        }
        self.1.push(mask);
        let area = mask.iter().map(|w| w.count_ones()).sum();
        self.0.push(Glyph {
            ch: ch.to_string(),
            rects: runs(&mask),
            area,
        });
    }

    fn add_cp(&mut self, cp: u32, rects: Vec<Rect>) {
        if let Some(ch) = char::from_u32(cp) {
            self.add(ch, rects);
        }
    }
}

/// The coverage as rectangles that do not overlap: each row's runs of set
/// subpixels, a run merged with the same run on the row above.
fn runs(mask: &[u64; 3]) -> Vec<Rect> {
    let set = |x: usize, y: usize| {
        let i = y * GW + x;
        mask[i / 64] >> (i % 64) & 1 == 1
    };
    let mut rects: Vec<Rect> = Vec::new();
    for y in 0..GH {
        let mut x = 0;
        while x < GW {
            if !set(x, y) {
                x += 1;
                continue;
            }
            let start = x;
            while x < GW && set(x, y) {
                x += 1;
            }
            let (x0, x1, yy) = (start as u8, x as u8, y as u8);
            match rects
                .iter_mut()
                .find(|r| r.0 == x0 && r.2 == x1 && r.3 == yy)
            {
                Some(r) => r.3 = yy + 1,
                None => rects.push((x0, yy, x1, yy + 1)),
            }
        }
    }
    rects
}

/// The glyph of every 2 x 4 pattern (bit 0 top left, row by row): the
/// octants of U+1CD00..U+1CDE5 and, for the 26 patterns Unicode had drawn
/// before, those older glyphs; read from UnicodeData.txt 16.0.
const OCTANTS: [u32; 256] = [
    0x20, 0x1CEA8, 0x1CEAB, 0x1FB82, 0x1CD00, 0x2598, 0x1CD01, 0x1CD02, 0x1CD03, 0x1CD04, 0x259D,
    0x1CD05, 0x1CD06, 0x1CD07, 0x1CD08, 0x2580, 0x1CD09, 0x1CD0A, 0x1CD0B, 0x1CD0C, 0x1FBE6,
    0x1CD0D, 0x1CD0E, 0x1CD0F, 0x1CD10, 0x1CD11, 0x1CD12, 0x1CD13, 0x1CD14, 0x1CD15, 0x1CD16,
    0x1CD17, 0x1CD18, 0x1CD19, 0x1CD1A, 0x1CD1B, 0x1CD1C, 0x1CD1D, 0x1CD1E, 0x1CD1F, 0x1FBE7,
    0x1CD20, 0x1CD21, 0x1CD22, 0x1CD23, 0x1CD24, 0x1CD25, 0x1CD26, 0x1CD27, 0x1CD28, 0x1CD29,
    0x1CD2A, 0x1CD2B, 0x1CD2C, 0x1CD2D, 0x1CD2E, 0x1CD2F, 0x1CD30, 0x1CD31, 0x1CD32, 0x1CD33,
    0x1CD34, 0x1CD35, 0x1FB85, 0x1CEA3, 0x1CD36, 0x1CD37, 0x1CD38, 0x1CD39, 0x1CD3A, 0x1CD3B,
    0x1CD3C, 0x1CD3D, 0x1CD3E, 0x1CD3F, 0x1CD40, 0x1CD41, 0x1CD42, 0x1CD43, 0x1CD44, 0x2596,
    0x1CD45, 0x1CD46, 0x1CD47, 0x1CD48, 0x258C, 0x1CD49, 0x1CD4A, 0x1CD4B, 0x1CD4C, 0x259E,
    0x1CD4D, 0x1CD4E, 0x1CD4F, 0x1CD50, 0x259B, 0x1CD51, 0x1CD52, 0x1CD53, 0x1CD54, 0x1CD55,
    0x1CD56, 0x1CD57, 0x1CD58, 0x1CD59, 0x1CD5A, 0x1CD5B, 0x1CD5C, 0x1CD5D, 0x1CD5E, 0x1CD5F,
    0x1CD60, 0x1CD61, 0x1CD62, 0x1CD63, 0x1CD64, 0x1CD65, 0x1CD66, 0x1CD67, 0x1CD68, 0x1CD69,
    0x1CD6A, 0x1CD6B, 0x1CD6C, 0x1CD6D, 0x1CD6E, 0x1CD6F, 0x1CD70, 0x1CEA0, 0x1CD71, 0x1CD72,
    0x1CD73, 0x1CD74, 0x1CD75, 0x1CD76, 0x1CD77, 0x1CD78, 0x1CD79, 0x1CD7A, 0x1CD7B, 0x1CD7C,
    0x1CD7D, 0x1CD7E, 0x1CD7F, 0x1CD80, 0x1CD81, 0x1CD82, 0x1CD83, 0x1CD84, 0x1CD85, 0x1CD86,
    0x1CD87, 0x1CD88, 0x1CD89, 0x1CD8A, 0x1CD8B, 0x1CD8C, 0x1CD8D, 0x1CD8E, 0x1CD8F, 0x2597,
    0x1CD90, 0x1CD91, 0x1CD92, 0x1CD93, 0x259A, 0x1CD94, 0x1CD95, 0x1CD96, 0x1CD97, 0x2590,
    0x1CD98, 0x1CD99, 0x1CD9A, 0x1CD9B, 0x259C, 0x1CD9C, 0x1CD9D, 0x1CD9E, 0x1CD9F, 0x1CDA0,
    0x1CDA1, 0x1CDA2, 0x1CDA3, 0x1CDA4, 0x1CDA5, 0x1CDA6, 0x1CDA7, 0x1CDA8, 0x1CDA9, 0x1CDAA,
    0x1CDAB, 0x2582, 0x1CDAC, 0x1CDAD, 0x1CDAE, 0x1CDAF, 0x1CDB0, 0x1CDB1, 0x1CDB2, 0x1CDB3,
    0x1CDB4, 0x1CDB5, 0x1CDB6, 0x1CDB7, 0x1CDB8, 0x1CDB9, 0x1CDBA, 0x1CDBB, 0x1CDBC, 0x1CDBD,
    0x1CDBE, 0x1CDBF, 0x1CDC0, 0x1CDC1, 0x1CDC2, 0x1CDC3, 0x1CDC4, 0x1CDC5, 0x1CDC6, 0x1CDC7,
    0x1CDC8, 0x1CDC9, 0x1CDCA, 0x1CDCB, 0x1CDCC, 0x1CDCD, 0x1CDCE, 0x1CDCF, 0x1CDD0, 0x1CDD1,
    0x1CDD2, 0x1CDD3, 0x1CDD4, 0x1CDD5, 0x1CDD6, 0x1CDD7, 0x1CDD8, 0x1CDD9, 0x1CDDA, 0x2584,
    0x1CDDB, 0x1CDDC, 0x1CDDD, 0x1CDDE, 0x2599, 0x1CDDF, 0x1CDE0, 0x1CDE1, 0x1CDE2, 0x259F,
    0x1CDE3, 0x2586, 0x1CDE4, 0x1CDE5, 0x2588,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_set_holds_its_glyphs_once_and_grows_with_the_font() {
        let (b, s, o) = (of(Set::Blocks), of(Set::Sextants), of(Set::Octants));
        // 2 + 14 eighths + 4 halves and eighths + the 10 quadrants left,
        // then 15 light and 15 heavy lines
        assert_eq!(b.len(), 60);
        assert_eq!(s.len(), b.len() + 60 + 10);
        // a cross is four runs that never overlap: its area is one count
        for ch in ["┏", "╋", "╹", "┼", "╴"] {
            assert!(b.iter().any(|g| g.ch == ch), "{ch}");
        }
        assert!(!b.iter().any(|g| g.ch == "┍"), "no mixed weights");
        let cross = b.iter().find(|g| g.ch == "┼").expect("cross");
        let summed: u32 = cross
            .rects
            .iter()
            .map(|&(x0, y0, x1, y1)| u32::from(x1 - x0) * u32::from(y1 - y0))
            .sum();
        assert_eq!(summed, cross.area);
        assert_eq!(cross.area, 8 * 2 + 24 - 2);
        assert!(o.len() > 300);
        assert!(o.iter().any(|g| g.ch == "\u{1CD00}"));
        let full = b.iter().find(|g| g.ch == "█").map(|g| g.area);
        assert_eq!(full, Some((GW * GH) as u32));
        let lower = b.iter().find(|g| g.ch == "▄").map(|g| g.area);
        assert_eq!(lower, Some((GW * GH / 2) as u32));
    }
}
