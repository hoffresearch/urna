//! The ask tab's image pane. `ratatui-image` asks the terminal once, at
//! start, which protocol it speaks (kitty, sixel, iTerm2) and how big a
//! cell is, and draws the image through the frame; a terminal with no
//! protocol gets the block glyphs of `blocks`, which every font has. The
//! selected hit's image blob is sliced off the mmap, decoded by
//! `urna-images` (the AV1 decoder carried in the tree, pure rust) and
//! fitted to the pane with its aspect kept.

use std::time::Instant;

use anyhow::{Result, anyhow};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::widgets::StatefulWidget;
use ratatui_image::picker::{Picker, ProtocolType};
use ratatui_image::protocol::StatefulProtocol;
use ratatui_image::{FilterType, Resize, StatefulImage};

use super::session::Session;
use crate::tui::graphics::Rgba;
use crate::tui::graphics::glyphs::Set;

struct Shown {
    blob: usize,
    rgba: Option<Rgba>,
    /// the protocol's state for this image, encoded on render.
    protocol: Option<StatefulProtocol>,
    /// the character cells last drawn, and the rect they were drawn for.
    blocks: Option<(Rect, Vec<super::blocks::Cell>)>,
    size: (u32, u32),
    error: Option<String>,
    ms: f64,
}

#[derive(Default)]
pub struct Media {
    picker: Option<Picker>,
    /// the glyphs to draw with where the picker found no protocol.
    glyphs: Set,
    shown: Option<Shown>,
}

impl std::fmt::Debug for Media {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Media")
            .field("mode", &self.mode())
            .field("shown", &self.shown.as_ref().map(|s| s.blob))
            .finish()
    }
}

impl Media {
    /// The probed terminal, handed over to the next ask.
    pub fn take_picker(&mut self) -> Option<Picker> {
        self.picker.take()
    }

    pub fn set_picker(&mut self, picker: Picker, glyphs: Set) {
        self.picker = Some(picker);
        self.glyphs = glyphs;
        self.shown = None;
    }

    pub fn glyphs(&self) -> Set {
        self.glyphs
    }

    /// The last decode, for the status line.
    pub fn decode_ms(&self) -> Option<f64> {
        self.shown
            .as_ref()
            .filter(|s| s.rgba.is_some())
            .map(|s| s.ms)
    }

    /// Decode `blob` unless it is already the one shown.
    pub fn show(&mut self, session: &Session, blob: usize) {
        if self.shown.as_ref().is_some_and(|s| s.blob == blob) {
            return;
        }
        let t = Instant::now();
        let shown = match decode(session, blob) {
            Ok(img) => Shown {
                blob,
                size: (img.width, img.height),
                protocol: self.picker.as_ref().and_then(|p| protocol(p, &img)),
                rgba: Some(img),
                blocks: None,
                error: None,
                ms: t.elapsed().as_secs_f64() * 1e3,
            },
            Err(e) => Shown {
                blob,
                size: (1, 1),
                rgba: None,
                protocol: None,
                blocks: None,
                error: Some(format!("{e:#}")),
                ms: 0.0,
            },
        };
        self.shown = Some(shown);
    }

    /// Ctrl+v: the selected image, as stored, in the system's viewer. The
    /// blob goes to a temporary file and `open`, `xdg-open` or `start`
    /// takes it from there; what the viewer does with it is the viewer's.
    pub fn open_external(&self, session: &Session) -> Result<&'static str> {
        let Some(s) = &self.shown else {
            return Err(anyhow!("select a hit first"));
        };
        if s.rgba.is_none() {
            return Err(anyhow!("the selected hit has no image"));
        }
        let bytes = session.rt.blob_bytes(s.blob)?;
        let path = std::env::temp_dir().join(format!("urna-blob-{}.avif", s.blob));
        std::fs::write(&path, bytes)?;
        let mut cmd = if cfg!(target_os = "macos") {
            let mut c = std::process::Command::new("open");
            c.arg(&path);
            c
        } else if cfg!(windows) {
            let mut c = std::process::Command::new("cmd");
            c.args(["/C", "start", ""]).arg(&path);
            c
        } else {
            let mut c = std::process::Command::new("xdg-open");
            c.arg(&path);
            c
        };
        cmd.stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| anyhow!("open the image: {e}"))?;
        Ok("image opened in the system viewer")
    }

    /// The image's width in cells when it fills `rows` cells of height.
    pub fn cells_wide(&self, rows: u16) -> u16 {
        let (Some(p), Some(s)) = (&self.picker, &self.shown) else {
            return 0;
        };
        let font = p.font_size();
        let px_h = u32::from(rows) * u32::from(font.height.max(1));
        let px_w = px_h * s.size.0 / s.size.1.max(1);
        px_w.div_ceil(u32::from(font.width.max(1))) as u16
    }

    pub fn error(&self) -> Option<&str> {
        self.shown.as_ref().and_then(|s| s.error.as_deref())
    }

    /// Draw into the frame: the protocol's cells, or block glyphs.
    pub fn render(&mut self, buf: &mut Buffer, area: Rect) {
        let Some(p) = &self.picker else { return };
        let Some(s) = self.shown.as_mut() else { return };
        let Some(rgba) = &s.rgba else { return };
        let font = p.font_size();
        let fit = fit(area, s.size, (font.width, font.height));
        match s.protocol.as_mut() {
            Some(state) if p.protocol_type() != ProtocolType::Halfblocks => {
                StatefulImage::<StatefulProtocol>::default()
                    .resize(Resize::Fit(Some(FilterType::Triangle)))
                    .render(fit, buf, state);
                if let Some(Err(e)) = state.last_encoding_result() {
                    s.error = Some(format!("{e}"));
                }
            }
            _ => {
                // block glyphs, recomputed only when the rect moves.
                if s.blocks.as_ref().is_none_or(|(r, _)| *r != fit) {
                    s.blocks = Some((
                        fit,
                        super::blocks::cells(rgba, fit.width, fit.height, self.glyphs),
                    ));
                }
                if let Some((r, cells)) = &s.blocks {
                    super::blocks::draw(buf, *r, cells);
                }
            }
        }
    }

    /// What draws the image, for the status line.
    pub fn mode(&self) -> &'static str {
        match self.picker.as_ref().map(Picker::protocol_type) {
            Some(ProtocolType::Kitty) => "kitty",
            Some(ProtocolType::Sixel) => "sixel",
            Some(ProtocolType::Iterm2) => "iterm2",
            _ => self.glyphs.name(),
        }
    }
}

/// The protocol state for `img`: its pixels handed to ratatui-image.
fn protocol(picker: &Picker, img: &Rgba) -> Option<StatefulProtocol> {
    if picker.protocol_type() == ProtocolType::Halfblocks {
        return None;
    }
    let buffer = image::RgbaImage::from_raw(img.width, img.height, img.data.clone())?;
    Some(picker.new_resize_protocol(image::DynamicImage::ImageRgba8(buffer)))
}

/// `size` pixels fitted in `area` cells with the aspect kept, centered.
fn fit(area: Rect, size: (u32, u32), font: (u16, u16)) -> Rect {
    let (fw, fh) = (f64::from(font.0.max(1)), f64::from(font.1.max(1)));
    let (iw, ih) = (f64::from(size.0.max(1)), f64::from(size.1.max(1)));
    let scale = (f64::from(area.width) * fw / iw).min(f64::from(area.height) * fh / ih);
    let w = ((iw * scale / fw).floor() as u16).clamp(1, area.width.max(1));
    let h = ((ih * scale / fh).floor() as u16).clamp(1, area.height.max(1));
    Rect::new(
        area.x + (area.width - w) / 2,
        area.y + (area.height - h) / 2,
        w,
        h,
    )
}

fn decode(session: &Session, blob: usize) -> Result<Rgba> {
    let bytes = session.rt.blob_bytes(blob)?;
    let img = urna_images::decode(bytes).map_err(|e| anyhow!("{e}"))?;
    Rgba::from_avif(img).ok_or_else(|| anyhow!("avif: the pixels do not match the size"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_keeps_the_aspect_and_centers() {
        // a 2:1 image in a 20x20 cell pane with 10x20 cells: 20 wide, 5 tall.
        let r = fit(Rect::new(0, 0, 20, 20), (200, 100), (10, 20));
        assert_eq!((r.width, r.height), (20, 5));
        assert_eq!(r.y, 7);
        let r = fit(Rect::new(3, 4, 10, 10), (100, 400), (10, 20));
        assert_eq!((r.width, r.height), (5, 10));
        assert_eq!(r.x, 5);
    }

    #[test]
    fn without_a_protocol_the_pane_draws_blocks_and_says_so() {
        let mut m = Media::default();
        assert_eq!(m.mode(), "blocks");
        m.set_picker(Picker::halfblocks(), Set::Blocks);
        let rgba = Rgba::new(4, 4, vec![200; 64]).expect("4x4");
        m.shown = Some(Shown {
            blob: 1,
            size: (4, 4),
            protocol: None,
            rgba: Some(rgba),
            blocks: None,
            error: None,
            ms: 0.0,
        });
        assert_eq!(m.cells_wide(10), 20);
        let area = Rect::new(0, 0, 20, 10);
        let mut buf = Buffer::empty(area);
        m.render(&mut buf, area);
        assert!(m.shown.as_ref().is_some_and(|s| s.blocks.is_some()));
        assert_ne!(buf[(5, 5)].symbol(), " ");
    }
}
