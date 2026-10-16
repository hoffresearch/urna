//! The ask tab's image pane. The terminal says once, at start, what it
//! can draw (kitty, sixel, iterm2, from its environment); without a
//! protocol it gets quadrant characters, which every font has. The
//! selected hit's image blob is sliced off the mmap, decoded by `urna-images`
//! (the AV1 decoder carried in the tree, pure rust) and fitted to the pane
//! with its aspect kept. Pixels go out after the frame, as the escape
//! bytes of the protocol written at the image's anchor cell.

use std::io::Write;
use std::time::Instant;

use anyhow::{Result, anyhow};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;

use super::session::Session;
use crate::tui::graphics::{self, Protocol, Rgba, Terminal};

/// The id the protocols that keep images know this pane's image by.
const IMAGE_ID: u32 = 7;

struct Shown {
    blob: usize,
    rgba: Option<Rgba>,
    /// the character cells last drawn, and the rect they were drawn for.
    blocks: Option<(Rect, Vec<super::blocks::Cell>)>,
    /// the protocol bytes last encoded, and the rect they fit.
    encoded: Option<(Rect, Vec<u8>)>,
    size: (u32, u32),
    error: Option<String>,
    ms: f64,
}

/// Where the next flush writes the image: the anchor cell and the bytes.
struct Placement {
    x: u16,
    y: u16,
    bytes: Vec<u8>,
    /// the same image at the same cells as the last flush: nothing to send.
    unchanged: bool,
}

#[derive(Default)]
pub struct Media {
    terminal: Option<Terminal>,
    shown: Option<Shown>,
    pending: Option<Placement>,
    /// a protocol image is on screen from the last flush.
    on_screen: Option<(usize, Rect)>,
}

impl Media {
    /// The probed terminal, handed over to the next ask.
    pub fn take_terminal(&mut self) -> Option<Terminal> {
        self.terminal.take()
    }

    pub fn set_terminal(&mut self, terminal: Terminal) {
        self.terminal = Some(terminal);
        self.shown = None;
    }

    pub fn protocol(&self) -> Option<Protocol> {
        self.terminal.map(|t| t.protocol)
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
                rgba: Some(img),
                blocks: None,
                encoded: None,
                error: None,
                ms: t.elapsed().as_secs_f64() * 1e3,
            },
            Err(e) => Shown {
                blob,
                size: (1, 1),
                rgba: None,
                blocks: None,
                encoded: None,
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
        let (Some(t), Some(s)) = (&self.terminal, &self.shown) else {
            return 0;
        };
        let px_h = u32::from(rows) * u32::from(t.cell.1.max(1));
        let px_w = px_h * s.size.0 / s.size.1.max(1);
        px_w.div_ceil(u32::from(t.cell.0.max(1))) as u16
    }

    pub fn error(&self) -> Option<&str> {
        self.shown.as_ref().and_then(|s| s.error.as_deref())
    }

    /// Draw into the frame: quadrant cells, or blank cells under the
    /// pixels the next `flush` writes there.
    pub fn render(&mut self, buf: &mut Buffer, area: Rect) {
        let Some(t) = self.terminal else { return };
        let Some(s) = self.shown.as_mut() else { return };
        let Some(rgba) = &s.rgba else { return };
        let fit = fit(area, s.size, t.cell);
        if !t.protocol.pixels() {
            // no pixels: quadrant characters, recomputed only when the rect moves.
            if s.blocks.as_ref().is_none_or(|(r, _)| *r != fit) {
                s.blocks = Some((fit, super::blocks::cells(rgba, fit.width, fit.height)));
            }
            if let Some((r, cells)) = &s.blocks {
                super::blocks::draw(buf, *r, cells);
            }
            return;
        }
        // the terminal draws pixels: the cells under them stay blank so the
        // frame never paints text over the image.
        for y in fit.y..fit.bottom() {
            for x in fit.x..fit.right() {
                if let Some(cell) = buf.cell_mut((x, y)) {
                    cell.reset();
                }
            }
        }
        if s.encoded.as_ref().is_none_or(|(r, _)| *r != fit) {
            let px = rgba.resize(
                u32::from(fit.width) * u32::from(t.cell.0.max(1)),
                u32::from(fit.height) * u32::from(t.cell.1.max(1)),
            );
            s.encoded = Some((
                fit,
                graphics::encode(t.protocol, &px, fit.width, fit.height, IMAGE_ID),
            ));
        }
        if let Some((r, bytes)) = &s.encoded {
            self.pending = Some(Placement {
                x: r.x,
                y: r.y,
                bytes: bytes.clone(),
                unchanged: self.on_screen == Some((s.blob, *r)),
            });
        }
    }

    /// After the frame: write the pending pixels at their anchor, or
    /// remove the image the terminal keeps when nothing was rendered.
    pub fn flush(&mut self, out: &mut impl Write) -> std::io::Result<()> {
        let Some(t) = self.terminal else {
            return Ok(());
        };
        match self.pending.take() {
            Some(p) if p.unchanged => Ok(()),
            Some(p) => {
                // save the cursor, go to the anchor (1-based), draw, restore.
                out.write_all(b"\x1b7")?;
                out.write_all(format!("\x1b[{};{}H", p.y + 1, p.x + 1).as_bytes())?;
                out.write_all(&p.bytes)?;
                out.write_all(b"\x1b8")?;
                out.flush()?;
                self.on_screen = self
                    .shown
                    .as_ref()
                    .and_then(|s| s.encoded.as_ref().map(|(r, _)| (s.blob, *r)));
                Ok(())
            }
            None => {
                if self.on_screen.take().is_some() {
                    let bytes = graphics::clear(t.protocol, IMAGE_ID);
                    if !bytes.is_empty() {
                        out.write_all(&bytes)?;
                        out.flush()?;
                    }
                }
                Ok(())
            }
        }
    }

    /// What draws the image: the terminal's protocol, or characters.
    pub fn mode(&self) -> &'static str {
        self.protocol().unwrap_or(Protocol::Quadrants).name()
    }
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
    fn flush_writes_the_anchor_then_the_bytes_and_clears_once_gone() {
        let mut m = Media {
            terminal: Some(Terminal {
                protocol: Protocol::Kitty,
                cell: (10, 20),
            }),
            ..Default::default()
        };
        m.pending = Some(Placement {
            x: 4,
            y: 2,
            bytes: b"IMG".to_vec(),
            unchanged: false,
        });
        m.shown = Some(Shown {
            blob: 1,
            rgba: None,
            blocks: None,
            encoded: Some((Rect::new(4, 2, 1, 1), b"IMG".to_vec())),
            size: (1, 1),
            error: None,
            ms: 0.0,
        });
        let mut out = Vec::new();
        m.flush(&mut out).expect("writes");
        assert_eq!(out, b"\x1b7\x1b[3;5HIMG\x1b8");
        assert!(m.on_screen.is_some());
        let mut out = Vec::new();
        m.flush(&mut out).expect("clears");
        assert!(out.starts_with(b"\x1b_Ga=d"));
        assert!(m.on_screen.is_none());
    }
}
