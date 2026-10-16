//! Pixels for the terminal. The protocols (kitty, iTerm2, sixel) and the
//! question of which one the terminal speaks belong to `ratatui-image`,
//! made here by `picker`; `URNA_GRAPHICS` overrides its answer. Terminals
//! with none of them get the image in block glyphs (`app::blocks`), from
//! the set of `glyphs` every font carries. `Rgba` is the decoded image as
//! both paths take it.

pub mod glyphs;

use ratatui_image::FontSize;
use ratatui_image::picker::{Picker, ProtocolType};

/// 8-bit RGBA pixels, rows top to bottom, no padding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgba {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl Rgba {
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Option<Self> {
        (data.len() == (width as usize) * (height as usize) * 4).then_some(Self {
            width,
            height,
            data,
        })
    }

    pub fn from_avif(img: urna_images::Image) -> Option<Self> {
        Self::new(img.width, img.height, img.rgba)
    }

    #[inline]
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let i = ((y as usize) * (self.width as usize) + x as usize) * 4;
        [
            self.data[i],
            self.data[i + 1],
            self.data[i + 2],
            self.data[i + 3],
        ]
    }

    /// The image resampled to `w` x `h`: each target pixel is the mean of
    /// the source box it covers (a box filter), which is what a photo
    /// shrunk to a terminal needs; growing one bilinear-blends the box.
    pub fn resize(&self, w: u32, h: u32) -> Rgba {
        let (w, h) = (w.max(1), h.max(1));
        if (w, h) == (self.width, self.height) {
            return self.clone();
        }
        let mut out = Vec::with_capacity((w as usize) * (h as usize) * 4);
        let (sw, sh) = (f64::from(self.width), f64::from(self.height));
        for y in 0..h {
            let y0 = f64::from(y) * sh / f64::from(h);
            let y1 = (f64::from(y + 1) * sh / f64::from(h)).max(y0 + 1.0);
            for x in 0..w {
                let x0 = f64::from(x) * sw / f64::from(w);
                let x1 = (f64::from(x + 1) * sw / f64::from(w)).max(x0 + 1.0);
                let mut acc = [0f64; 4];
                let mut n = 0f64;
                let (ya, yb) = (y0.floor() as u32, (y1.ceil() as u32).min(self.height));
                let (xa, xb) = (x0.floor() as u32, (x1.ceil() as u32).min(self.width));
                for sy in ya..yb.max(ya + 1) {
                    for sx in xa..xb.max(xa + 1) {
                        let p = self.pixel(sx.min(self.width - 1), sy.min(self.height - 1));
                        for c in 0..4 {
                            acc[c] += f64::from(p[c]);
                        }
                        n += 1.0;
                    }
                }
                for v in acc {
                    out.push((v / n).round() as u8);
                }
            }
        }
        Rgba {
            width: w,
            height: h,
            data: out,
        }
    }
}

/// The terminal, asked by ratatui-image: call it with the alternate screen
/// up and before any key is read. `URNA_GRAPHICS` (`kitty`, `iterm`,
/// `sixel`, `blocks`) wins over the answer; a terminal that answers
/// nothing gets blocks at the cell size the kernel reports.
pub fn picker() -> Picker {
    let mut picker = Picker::from_query_stdio().unwrap_or_else(|_| {
        let cell = crossterm::terminal::window_size()
            .ok()
            .filter(|w| w.width > 0 && w.height > 0 && w.columns > 0 && w.rows > 0)
            .map_or((10, 20), |w| (w.width / w.columns, w.height / w.rows));
        // `halfblocks()` would fix the cell at 10x20; the kernel's size
        // keeps the aspect right where the terminal answered no query.
        #[allow(deprecated, reason = "the only constructor that takes a cell size")]
        Picker::from_fontsize(FontSize::new(cell.0, cell.1))
    });
    if let Some(p) = forced(&std::env::var("URNA_GRAPHICS").unwrap_or_default()) {
        picker.set_protocol_type(p);
    }
    picker
}

/// The glyphs to draw with where there is no protocol: the blocks every
/// font has, unless `URNA_GLYPHS` (`sextants`, `octants`) says the font
/// has more. A font cannot be asked, so nothing here guesses.
pub fn glyph_set() -> glyphs::Set {
    match std::env::var("URNA_GLYPHS").unwrap_or_default().as_str() {
        "sextants" => glyphs::Set::Sextants,
        "octants" => glyphs::Set::Octants,
        _ => glyphs::Set::Blocks,
    }
}

fn forced(v: &str) -> Option<ProtocolType> {
    match v {
        "kitty" => Some(ProtocolType::Kitty),
        "sixel" => Some(ProtocolType::Sixel),
        "iterm" | "iterm2" => Some(ProtocolType::Iterm2),
        "blocks" | "glyphs" | "halfblocks" | "quadrants" => Some(ProtocolType::Halfblocks),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_averages_the_box_and_keeps_a_flat_image_flat() {
        let img = Rgba::new(
            2,
            2,
            vec![
                255, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 255, 255, 0, 0, 255,
            ],
        )
        .expect("2x2");
        let one = img.resize(1, 1);
        assert_eq!(one.data, vec![128, 0, 128, 255]);
        let flat = Rgba::new(3, 3, vec![10; 36]).expect("3x3");
        assert_eq!(flat.resize(2, 5).data, vec![10; 40]);
        assert!(Rgba::new(2, 2, vec![0; 3]).is_none());
    }

    #[test]
    fn the_override_names_match() {
        assert_eq!(forced("quadrants"), Some(ProtocolType::Halfblocks));
        assert_eq!(forced("kitty"), Some(ProtocolType::Kitty));
        assert_eq!(forced(""), None);
    }
}
