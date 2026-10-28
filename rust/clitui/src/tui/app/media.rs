//! The ask tab's image pane. The terminal says once, at start, what it
//! draws (`graphics::detect`): kitty, sixel or iTerm2 pixels, or block
//! glyphs from the set its font carries. The selected hit's image blob is
//! sliced off the mmap, decoded by `urna-images` (the AV1 decoder carried
//! in the tree, pure rust) and fitted to the pane with its aspect kept.
//! Glyphs are drawn into the frame; pixels go out after it (`flush`), as
//! the protocol's escape bytes at the image's anchor cell, and the frame
//! is told (`settle`) which cells the pixels cover and which they left.

use std::io::Write;
use std::time::Instant;

use anyhow::{Result, anyhow};
use ratatui::buffer::{Buffer, CellDiffOption};
use ratatui::layout::Rect;

use super::session::Session;
use crate::tui::graphics::{self, Protocol, Rgba, Terminal};

/// The id the protocols that keep images know this pane's image by.
const IMAGE_ID: u32 = 7;

/// The image made for the rect it was last drawn at.
enum Drawn {
    Glyphs(Vec<super::blocks::Cell>),
    /// the protocol's escape bytes.
    Pixels(Vec<u8>),
}

struct Shown {
    blob: usize,
    /// `None` when the blob did not decode (`error` says why).
    rgba: Option<Rgba>,
    drawn: Option<(Rect, Drawn)>,
    size: (u32, u32),
    error: Option<String>,
    ms: f64,
}

#[derive(Default)]
pub struct Media {
    terminal: Option<Terminal>,
    shown: Option<Shown>,
    /// where this frame's pixels go: set by `render`, kept by `settle`
    /// when nothing covers them, written by `flush`.
    pending: Option<Rect>,
    /// the blob and the rect of the pixels on screen since the last flush.
    on_screen: Option<(usize, Rect)>,
    /// a resize cleared the screen since `on_screen` was drawn: its pixels
    /// go out again, or the image a terminal may keep is removed.
    stale: bool,
    /// the frame's size at the last settle: a resize clears the screen.
    frame: Rect,
}

impl Media {
    /// A new corpus: nothing is shown, but what is on screen and what the
    /// terminal draws stay known, so the next flush removes the old image.
    pub fn forget(&mut self) {
        self.shown = None;
        self.pending = None;
    }

    pub fn set_terminal(&mut self, terminal: Terminal) {
        self.terminal = Some(terminal);
        self.shown = None;
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
                drawn: None,
                error: None,
                ms: t.elapsed().as_secs_f64() * 1e3,
            },
            Err(e) => Shown {
                blob,
                size: (1, 1),
                rgba: None,
                drawn: None,
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

    /// Draw into the frame: the glyphs, or blank cells where the next
    /// `flush` writes the pixels. A pane the layout collapsed to nothing
    /// draws nothing.
    pub fn render(&mut self, buf: &mut Buffer, area: Rect) {
        if area.is_empty() {
            return;
        }
        let Some(t) = self.terminal else { return };
        let Some(s) = self.shown.as_mut() else { return };
        let Some(rgba) = &s.rgba else { return };
        let fit = fit(area, s.size, t.cell);
        // made again only when the rect moves.
        if s.drawn.as_ref().is_none_or(|(r, _)| *r != fit) {
            let drawn = if t.protocol.pixels() {
                let px = rgba.resize(
                    u32::from(fit.width) * u32::from(t.cell.0.max(1)),
                    u32::from(fit.height) * u32::from(t.cell.1.max(1)),
                );
                Drawn::Pixels(graphics::encode(
                    t.protocol, &px, fit.width, fit.height, IMAGE_ID,
                ))
            } else {
                Drawn::Glyphs(super::blocks::cells(rgba, fit.width, fit.height, t.glyphs))
            };
            s.drawn = Some((fit, drawn));
        }
        match &s.drawn {
            Some((r, Drawn::Glyphs(cells))) => super::blocks::draw(buf, *r, cells),
            Some((r, Drawn::Pixels(_))) => {
                for p in r.positions() {
                    if let Some(c) = buf.cell_mut(p) {
                        c.reset();
                    }
                }
                self.pending = Some(*r);
            }
            None => {}
        }
    }

    /// At the end of the frame, with every overlay drawn: pixels that
    /// something in `covers` overlaps are not drawn this frame. The cells
    /// under pixels already on screen are skipped, so no redraw of the
    /// frame paints over them; the cells pixels left are drawn again,
    /// since ratatui would not repaint a blank over a blank.
    pub fn settle(&mut self, buf: &mut Buffer, covers: &[Rect]) {
        if buf.area != self.frame {
            // a resize clears the screen; kitty may keep its image, so
            // the one on screen is still deleted if nothing replaces it.
            self.frame = buf.area;
            self.stale = self.on_screen.is_some();
        }
        let blob = self.shown.as_ref().map(|s| s.blob);
        self.pending = self
            .pending
            .filter(|r| !covers.iter().any(|c| c.intersects(*r)));
        if let Some((_, old)) = self.on_screen {
            let stays = self.pending;
            for p in old.intersection(buf.area).positions() {
                if !stays.is_some_and(|r| r.contains(p))
                    && let Some(c) = buf.cell_mut(p)
                {
                    c.set_diff_option(CellDiffOption::AlwaysUpdate);
                }
            }
        }
        // the first frame of a placement draws its blanks, so nothing the
        // pixels leave transparent shows the text that was there.
        if let (Some(r), Some(b)) = (self.pending, blob)
            && !self.stale
            && self.on_screen == Some((b, r))
        {
            for p in r.intersection(buf.area).positions() {
                if let Some(c) = buf.cell_mut(p) {
                    c.set_diff_option(CellDiffOption::Skip);
                }
            }
        }
    }

    /// After the frame: write the pending pixels at their anchor, or
    /// remove the image the terminal keeps when none were rendered.
    pub fn flush(&mut self, out: &mut impl Write) -> std::io::Result<()> {
        let Some(t) = self.terminal else {
            return Ok(());
        };
        let placed = self.pending.take().and_then(|r| {
            let s = self.shown.as_ref()?;
            match &s.drawn {
                Some((at, Drawn::Pixels(bytes))) if *at == r => Some((s.blob, r, bytes)),
                _ => None,
            }
        });
        match placed {
            Some((blob, r, _)) if !self.stale && self.on_screen == Some((blob, r)) => Ok(()),
            Some((blob, r, bytes)) => {
                // save the cursor, go to the anchor (1-based), draw, restore.
                out.write_all(b"\x1b7")?;
                out.write_all(format!("\x1b[{};{}H", r.y + 1, r.x + 1).as_bytes())?;
                out.write_all(bytes)?;
                out.write_all(b"\x1b8")?;
                out.flush()?;
                self.on_screen = Some((blob, r));
                self.stale = false;
                Ok(())
            }
            None => {
                self.stale = false;
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

    /// What draws the image, for the status line.
    pub fn mode(&self) -> &'static str {
        match self.terminal.map(|t| (t.protocol, t.glyphs)) {
            Some((Protocol::Kitty, _)) => "kitty",
            Some((Protocol::Sixel, _)) => "sixel",
            Some((Protocol::Iterm2, _)) => "iterm2",
            Some((Protocol::Glyphs, g)) => g.name(),
            None => graphics::glyphs::Set::default().name(),
        }
    }
}

/// `size` pixels fitted in `area` cells with the aspect kept, centered;
/// `area` is not empty (`render` returns before).
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
    use crate::tui::graphics::glyphs::Set;

    fn media(protocol: Protocol) -> Media {
        let mut m = Media::default();
        m.set_terminal(Terminal {
            protocol,
            cell: (10, 20),
            glyphs: Set::Blocks,
        });
        m.shown = Some(Shown {
            blob: 1,
            size: (4, 4),
            rgba: Rgba::new(4, 4, vec![200; 64]),
            drawn: None,
            error: None,
            ms: 0.0,
        });
        m
    }

    /// One frame: render the pane at `pane`, settle against `covers`,
    /// flush; the bytes written.
    fn frame(m: &mut Media, pane: Option<Rect>, covers: &[Rect]) -> (Buffer, Vec<u8>) {
        let mut buf = Buffer::empty(Rect::new(0, 0, 40, 20));
        if let Some(p) = pane {
            m.render(&mut buf, p);
        }
        m.settle(&mut buf, covers);
        let mut out = Vec::new();
        m.flush(&mut out).expect("writes");
        (buf, out)
    }

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
    fn without_pixels_the_pane_draws_glyphs_and_says_so() {
        let mut m = media(Protocol::Glyphs);
        assert_eq!(m.mode(), "blocks");
        assert_eq!(m.cells_wide(10), 20);
        let (buf, out) = frame(&mut m, Some(Rect::new(0, 0, 20, 10)), &[]);
        // a flat image: a blank in its color on both sides.
        let c = &buf[(5, 5)];
        assert_eq!(c.bg, ratatui::style::Color::Rgb(200, 200, 200));
        assert_eq!(c.fg, c.bg);
        assert!(out.is_empty());
    }

    #[test]
    fn a_collapsed_pane_draws_nothing_and_does_not_panic() {
        for protocol in [Protocol::Glyphs, Protocol::Kitty] {
            let mut m = media(protocol);
            for pane in [Rect::new(3, 3, 0, 5), Rect::new(3, 3, 5, 0)] {
                let (buf, out) = frame(&mut m, Some(pane), &[]);
                assert!(buf.content.iter().all(|c| c.symbol() == " "));
                assert!(out.is_empty());
            }
        }
    }

    #[test]
    fn pixels_go_out_once_at_the_anchor_and_leave_when_the_pane_goes() {
        let mut m = media(Protocol::Kitty);
        let pane = Rect::new(4, 2, 10, 5);
        let (_, out) = frame(&mut m, Some(pane), &[]);
        // a 4x4 image in 10x5 cells of 10x20 px fills them: anchor (4, 2).
        assert!(out.starts_with(b"\x1b7\x1b[3;5H\x1b_G"), "{:?}", &out[..12]);
        assert!(out.ends_with(b"\x1b8"));
        // the same image at the same cells: nothing to send, the cells skipped.
        let (buf, out) = frame(&mut m, Some(pane), &[]);
        assert!(out.is_empty());
        assert_eq!(buf[(7, 3)].diff_option, CellDiffOption::Skip);
        // off the tab: kitty deletes it, the cells it held are drawn again.
        let (buf, out) = frame(&mut m, None, &[]);
        assert!(out.starts_with(b"\x1b_Ga=d"));
        assert_eq!(buf[(7, 3)].diff_option, CellDiffOption::AlwaysUpdate);
        let (_, out) = frame(&mut m, None, &[]);
        assert!(out.is_empty());
    }

    #[test]
    fn an_overlay_hides_the_pixels_and_a_resize_sends_them_again() {
        let mut m = media(Protocol::Sixel);
        let pane = Rect::new(0, 0, 10, 5);
        let (_, out) = frame(&mut m, Some(pane), &[]);
        assert!(!out.is_empty());
        // a toast over the image: not drawn this frame, its cells repainted.
        let (buf, out) = frame(&mut m, Some(pane), &[Rect::new(2, 2, 3, 1)]);
        assert!(out.is_empty(), "sixel has nothing to delete");
        assert_eq!(buf[(3, 2)].diff_option, CellDiffOption::AlwaysUpdate);
        let (_, out) = frame(&mut m, Some(pane), &[]);
        assert!(out.starts_with(b"\x1b7"), "back once the toast is gone");
        // a resize clears the screen: the same placement goes out again.
        m.frame = Rect::new(0, 0, 80, 24);
        let (_, out) = frame(&mut m, Some(pane), &[]);
        assert!(out.starts_with(b"\x1b7"));
    }

    #[test]
    fn a_resize_that_hides_the_pane_still_deletes_the_kept_image() {
        let mut m = media(Protocol::Kitty);
        let (_, out) = frame(&mut m, Some(Rect::new(0, 0, 10, 5)), &[]);
        assert!(out.starts_with(b"\x1b7"));
        m.frame = Rect::new(0, 0, 80, 24);
        let (_, out) = frame(&mut m, None, &[]);
        assert!(out.starts_with(b"\x1b_Ga=d"), "{out:?}");
    }

    #[test]
    fn a_new_corpus_deletes_the_image_the_last_one_left() {
        let mut m = media(Protocol::Kitty);
        let pane = Rect::new(4, 2, 10, 5);
        frame(&mut m, Some(pane), &[]);
        m.forget();
        let (buf, out) = frame(&mut m, None, &[]);
        assert!(out.starts_with(b"\x1b_Ga=d"), "{out:?}");
        assert_eq!(buf[(7, 3)].diff_option, CellDiffOption::AlwaysUpdate);
        assert_eq!(m.mode(), "kitty");
    }
}
