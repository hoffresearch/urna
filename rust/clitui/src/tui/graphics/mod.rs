//! Pixels in the terminal, by the protocol the terminal speaks, written
//! here from the specifications: kitty's graphics protocol (kitty,
//! ghostty), iTerm2's inline images (iTerm2, WezTerm) and sixel. Each
//! takes an `Rgba` image sized for a cell rectangle and gives the escape
//! bytes that draw it at the cursor; the explorer writes them after each
//! frame at the image's anchor cell. Terminals with none of these get
//! quadrant characters (`app::blocks`). The protocol comes from the
//! variables terminals set, `URNA_GRAPHICS` overrides them; nothing here
//! queries the terminal, so no reply can land in the key stream.

mod base64;
mod iterm;
mod kitty;
mod png;
mod sixel;

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

    pub fn from_avif(img: urna_avif::Image) -> Option<Self> {
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

/// What draws the image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Protocol {
    Kitty,
    Iterm2,
    Sixel,
    /// quadrant characters: every terminal and font.
    Quadrants,
}

impl Protocol {
    pub fn name(self) -> &'static str {
        match self {
            Self::Kitty => "kitty",
            Self::Iterm2 => "iterm2",
            Self::Sixel => "sixel",
            Self::Quadrants => "quadrants",
        }
    }

    pub fn pixels(self) -> bool {
        self != Self::Quadrants
    }
}

/// The terminal as the explorer draws on it: its protocol and the pixel
/// size of one cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Terminal {
    pub protocol: Protocol,
    /// (width, height) of a cell in pixels.
    pub cell: (u16, u16),
}

impl Default for Terminal {
    fn default() -> Self {
        Self {
            protocol: Protocol::Quadrants,
            cell: (10, 20),
        }
    }
}

/// The terminal from its environment: `URNA_GRAPHICS` (`kitty`, `iterm`,
/// `sixel`, `halfblocks`/`quadrants`) wins; else kitty's protocol in kitty
/// and ghostty, iTerm2's in iTerm2 and WezTerm, characters everywhere
/// else and inside tmux. The cell size comes from the window size the
/// kernel reports, never from a query.
pub fn detect() -> Terminal {
    let cell = crossterm::terminal::window_size()
        .ok()
        .filter(|w| w.width > 0 && w.height > 0 && w.columns > 0 && w.rows > 0)
        .map(|w| (w.width / w.columns, w.height / w.rows))
        .unwrap_or((10, 20));
    Terminal {
        protocol: forced().unwrap_or_else(from_env),
        cell,
    }
}

fn forced() -> Option<Protocol> {
    match std::env::var("URNA_GRAPHICS").unwrap_or_default().as_str() {
        "kitty" => Some(Protocol::Kitty),
        "sixel" => Some(Protocol::Sixel),
        "iterm" | "iterm2" => Some(Protocol::Iterm2),
        "halfblocks" | "quadrants" => Some(Protocol::Quadrants),
        _ => None,
    }
}

fn from_env() -> Protocol {
    let var = |k: &str| std::env::var(k).unwrap_or_default();
    let (term, program) = (var("TERM"), var("TERM_PROGRAM"));
    if term.starts_with("tmux") || term.starts_with("screen") {
        Protocol::Quadrants
    } else if !var("KITTY_WINDOW_ID").is_empty() || term.contains("kitty") || program == "ghostty" {
        Protocol::Kitty
    } else if program == "iTerm.app" || program == "WezTerm" || var("LC_TERMINAL").contains("iTerm")
    {
        Protocol::Iterm2
    } else {
        Protocol::Quadrants
    }
}

/// The escape bytes that draw `img` at the cursor, spanning `cols` x
/// `rows` cells. `id` names the image to the protocols that keep images
/// (kitty), so the next draw replaces this one.
pub fn encode(protocol: Protocol, img: &Rgba, cols: u16, rows: u16, id: u32) -> Vec<u8> {
    match protocol {
        Protocol::Kitty => kitty::encode(img, cols, rows, id),
        Protocol::Iterm2 => iterm::encode(img, cols, rows),
        Protocol::Sixel => sixel::encode(img),
        Protocol::Quadrants => Vec::new(),
    }
}

/// The escape bytes that remove image `id` where the protocol keeps
/// images past the cells they covered (kitty); nothing elsewhere, where
/// redrawing the cells is the removal.
pub fn clear(protocol: Protocol, id: u32) -> Vec<u8> {
    match protocol {
        Protocol::Kitty => kitty::delete(id),
        _ => Vec::new(),
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
    fn the_override_wins_and_the_names_match() {
        assert_eq!(Protocol::Quadrants.name(), "quadrants");
        assert!(Protocol::Sixel.pixels() && !Protocol::Quadrants.pixels());
        assert!(clear(Protocol::Sixel, 1).is_empty());
        assert!(clear(Protocol::Kitty, 1).starts_with(b"\x1b_G"));
    }
}
