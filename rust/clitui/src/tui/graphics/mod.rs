//! Pixels for the terminal, by the protocol the terminal speaks, written
//! here from the specifications: kitty's graphics protocol (kitty,
//! ghostty), iTerm2's inline images (iTerm2, WezTerm) and sixel. Each
//! takes an `Rgba` image sized for a cell rectangle and gives the escape
//! bytes that draw it at the cursor; the explorer writes them after each
//! frame at the image's anchor cell. Which one the terminal speaks, and
//! its cell size, the terminal says itself (`probe`); `URNA_GRAPHICS`
//! overrides it. Terminals with none get the image in block glyphs
//! (`app::blocks`), from the set of `glyphs` the font carries.

mod base64;
pub mod glyphs;
mod iterm;
mod kitty;
mod png;
pub mod probe;
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

/// What draws the image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Protocol {
    Kitty,
    Iterm2,
    Sixel,
    /// block glyphs (`app::blocks`): every terminal and font.
    Glyphs,
}

impl Protocol {
    pub fn pixels(self) -> bool {
        self != Self::Glyphs
    }
}

/// The terminal as the explorer draws on it: its protocol, the pixel size
/// of one cell and the glyphs its font carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Terminal {
    pub protocol: Protocol,
    /// (width, height) of a cell in pixels.
    pub cell: (u16, u16),
    pub glyphs: glyphs::Set,
}

impl Default for Terminal {
    fn default() -> Self {
        Self {
            protocol: Protocol::Glyphs,
            cell: (10, 20),
            glyphs: glyphs::Set::Blocks,
        }
    }
}

/// The terminal, asked once per process with the alternate screen up and
/// before any key is read. `URNA_GRAPHICS` (`kitty`, `iterm`, `sixel`,
/// `blocks`) wins over the answer, and `blocks` asks nothing: the glyphs
/// need only the cell size the kernel reports.
pub fn detect() -> Terminal {
    let env = |k: &str| std::env::var(k).unwrap_or_default();
    let forced = forced(&env("URNA_GRAPHICS"));
    let answer = match forced {
        Some(Protocol::Glyphs) => None,
        _ => probe::ask(probe::DEADLINE),
    };
    let kernel = crossterm::terminal::window_size()
        .ok()
        .filter(|w| w.width > 0 && w.height > 0 && w.columns > 0 && w.rows > 0)
        .map(|w| (w.width / w.columns, w.height / w.rows));
    Terminal {
        protocol: forced.unwrap_or_else(|| choose(answer.as_ref(), &env)),
        cell: answer
            .as_ref()
            .and_then(|a| a.cell)
            .or(kernel)
            .unwrap_or((10, 20)),
        glyphs: glyph_set(&env("URNA_GLYPHS")),
    }
}

/// The protocol from what the terminal answered. Inside tmux only sixel,
/// which tmux draws itself: an image passed through to the terminal
/// outside would land where that terminal's cursor is, not in the pane.
/// iTerm2 and WezTerm by name take inline images, then kitty's `OK`, then
/// DA1's sixel; a terminal that answered nothing is read from the
/// variables it sets, as before there was a query.
fn choose(answer: Option<&probe::Answer>, env: &dyn Fn(&str) -> String) -> Protocol {
    let name = answer.and_then(|a| a.name.as_deref()).unwrap_or_default();
    let term = env("TERM");
    let tmux = !env("TMUX").is_empty()
        || term.starts_with("tmux")
        || term.starts_with("screen")
        || name.starts_with("tmux");
    match answer.filter(|a| a.done) {
        Some(a) if tmux => {
            if a.sixel {
                Protocol::Sixel
            } else {
                Protocol::Glyphs
            }
        }
        Some(_) if name.starts_with("iTerm2") || name.starts_with("WezTerm") => Protocol::Iterm2,
        Some(a) if a.kitty => Protocol::Kitty,
        Some(a) if a.sixel => Protocol::Sixel,
        Some(_) => Protocol::Glyphs,
        None if tmux => Protocol::Glyphs,
        None => from_env(env),
    }
}

fn from_env(env: &dyn Fn(&str) -> String) -> Protocol {
    let (term, program) = (env("TERM"), env("TERM_PROGRAM"));
    if !env("KITTY_WINDOW_ID").is_empty() || term.contains("kitty") || program == "ghostty" {
        Protocol::Kitty
    } else if program == "iTerm.app" || program == "WezTerm" || env("LC_TERMINAL").contains("iTerm")
    {
        Protocol::Iterm2
    } else {
        Protocol::Glyphs
    }
}

fn forced(v: &str) -> Option<Protocol> {
    match v {
        "kitty" => Some(Protocol::Kitty),
        "sixel" => Some(Protocol::Sixel),
        "iterm" | "iterm2" => Some(Protocol::Iterm2),
        "blocks" | "glyphs" | "halfblocks" | "quadrants" => Some(Protocol::Glyphs),
        _ => None,
    }
}

/// The glyphs to draw with where there is no protocol: the blocks every
/// font has, unless `URNA_GLYPHS` (`sextants`, `octants`) says the font
/// has more. A font cannot be asked, so nothing here guesses.
fn glyph_set(v: &str) -> glyphs::Set {
    match v {
        "sextants" => glyphs::Set::Sextants,
        "octants" => glyphs::Set::Octants,
        _ => glyphs::Set::Blocks,
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
        Protocol::Glyphs => Vec::new(),
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
    fn the_override_names_match_and_only_kitty_clears() {
        assert_eq!(forced("quadrants"), Some(Protocol::Glyphs));
        assert_eq!(forced("kitty"), Some(Protocol::Kitty));
        assert_eq!(forced(""), None);
        assert_eq!(glyph_set("octants"), glyphs::Set::Octants);
        assert!(clear(Protocol::Sixel, 1).is_empty());
        assert!(clear(Protocol::Kitty, 1).starts_with(b"\x1b_G"));
    }

    fn vars(pairs: &[(&str, &str)]) -> impl Fn(&str) -> String {
        let pairs: Vec<(String, String)> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        move |k: &str| {
            pairs
                .iter()
                .find(|(n, _)| n == k)
                .map(|(_, v)| v.clone())
                .unwrap_or_default()
        }
    }

    #[test]
    fn the_answer_picks_the_protocol() {
        let none = vars(&[]);
        let a = |bytes: &[u8]| probe::parse(bytes);
        let kitty = a(b"\x1b_Gi=31;OK\x1b\\\x1b[?62;c");
        assert_eq!(choose(Some(&kitty), &none), Protocol::Kitty);
        let wez = a(b"\x1b_Gi=31;OK\x1b\\\x1bP>|WezTerm 20240203\x1b\\\x1b[?65;4c");
        assert_eq!(choose(Some(&wez), &none), Protocol::Iterm2);
        let foot = a(b"\x1b[?62;4;22c");
        assert_eq!(choose(Some(&foot), &none), Protocol::Sixel);
        let apple = a(b"\x1b[?1;2c");
        assert_eq!(choose(Some(&apple), &none), Protocol::Glyphs);
    }

    #[test]
    fn inside_tmux_only_the_sixel_tmux_draws() {
        let tmux = vars(&[("TMUX", "/tmp/tmux-501/default,1,0")]);
        let with = probe::parse(b"\x1bP>|tmux 3.5a\x1b\\\x1b[?65;4c");
        assert_eq!(choose(Some(&with), &tmux), Protocol::Sixel);
        // kitty outside tmux answered through it: still not drawn there.
        let kitty = probe::parse(b"\x1b_Gi=31;OK\x1b\\\x1b[?62;c");
        assert_eq!(choose(Some(&kitty), &tmux), Protocol::Glyphs);
        assert_eq!(choose(None, &tmux), Protocol::Glyphs);
    }

    #[test]
    fn no_answer_falls_back_to_the_variables() {
        let ghostty = vars(&[("TERM_PROGRAM", "ghostty")]);
        assert_eq!(choose(None, &ghostty), Protocol::Kitty);
        let iterm = vars(&[("LC_TERMINAL", "iTerm2")]);
        assert_eq!(choose(None, &iterm), Protocol::Iterm2);
        // a reply cut before DA1 is no answer either.
        let cut = probe::parse(b"\x1b_Gi=31;OK\x1b\\");
        assert_eq!(choose(Some(&cut), &vars(&[])), Protocol::Glyphs);
    }
}
