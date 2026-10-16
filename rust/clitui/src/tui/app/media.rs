//! The ask tab's image pane. The terminal is asked once, at start, what it
//! can draw (kitty, sixel, iterm2); without an answer it gets truecolor
//! halfblocks, which every font has. The selected hit's image blob is sliced
//! off the mmap, decoded by rav1d (pure rust, no system library) and fitted
//! to the pane with its aspect kept.

use std::time::Instant;

use anyhow::{Result, anyhow};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::widgets::StatefulWidget;
use ratatui_image::picker::{Picker, ProtocolType};
use ratatui_image::protocol::StatefulProtocol;
use ratatui_image::{FilterType, Resize, StatefulImage};

use super::session::Session;

struct Shown {
    blob: usize,
    proto: Option<StatefulProtocol>,
    /// the decoded pixels: the character renderer draws from them and the
    /// native window shows them.
    rgba: Option<image::RgbaImage>,
    /// the character cells last drawn, and the rect they were drawn for.
    blocks: Option<(Rect, Vec<super::blocks::Cell>)>,
    #[cfg_attr(not(feature = "window"), allow(dead_code))]
    title: String,
    size: (u32, u32),
    error: Option<String>,
    ms: f64,
}

/// The native window process and the frame file it follows.
struct Viewer {
    child: std::process::Child,
    frame: std::path::PathBuf,
}

#[derive(Default)]
pub struct Media {
    picker: Option<Picker>,
    shown: Option<Shown>,
    viewer: Option<Viewer>,
}

impl Drop for Media {
    fn drop(&mut self) {
        self.close_viewer();
    }
}

/// Ask the terminal for its graphics protocol and cell size. Runs in raw
/// mode, before the event loop reads stdin (the replies arrive on stdin).
/// `URNA_GRAPHICS=kitty|sixel|iterm|halfblocks` overrides what it answered.
pub fn probe() -> Picker {
    let mut picker = Picker::from_query_stdio().unwrap_or_else(|_| Picker::halfblocks());
    let forced = match std::env::var("URNA_GRAPHICS").unwrap_or_default().as_str() {
        "kitty" => Some(ProtocolType::Kitty),
        "sixel" => Some(ProtocolType::Sixel),
        "iterm" => Some(ProtocolType::Iterm2),
        "halfblocks" => Some(ProtocolType::Halfblocks),
        _ => None,
    };
    if let Some(p) = forced {
        picker.set_protocol_type(p);
    }
    picker
}

impl Media {
    /// The probed picker, handed over to the next ask.
    pub fn take_picker(&mut self) -> Option<Picker> {
        self.picker.take()
    }

    pub fn set_picker(&mut self, picker: Picker) {
        self.picker = Some(picker);
        self.shown = None;
    }

    pub fn protocol(&self) -> Option<ProtocolType> {
        self.picker.as_ref().map(Picker::protocol_type)
    }

    /// The last decode, for the status line.
    pub fn decode_ms(&self) -> Option<f64> {
        self.shown
            .as_ref()
            .filter(|s| s.proto.is_some())
            .map(|s| s.ms)
    }

    /// Decode `blob` unless it is already the one shown; `title` names it in
    /// the native window.
    pub fn show(&mut self, session: &Session, blob: usize, title: &str) {
        if self.shown.as_ref().is_some_and(|s| s.blob == blob) {
            return;
        }
        let picker = self.picker.get_or_insert_with(Picker::halfblocks);
        let t = Instant::now();
        let shown = match decode(session, blob) {
            Ok(img) => Shown {
                blob,
                size: (img.width(), img.height()),
                rgba: Some(img.to_rgba8()),
                blocks: None,
                title: title.to_string(),
                proto: Some(picker.new_resize_protocol(img)),
                error: None,
                ms: t.elapsed().as_secs_f64() * 1e3,
            },
            Err(e) => Shown {
                blob,
                size: (1, 1),
                rgba: None,
                blocks: None,
                title: title.to_string(),
                proto: None,
                error: Some(format!("{e:#}")),
                ms: 0.0,
            },
        };
        self.shown = Some(shown);
        if self.viewer.is_some() {
            self.write_frame();
        }
    }

    /// Ctrl+v: open the selected image in a native window, or close it.
    /// Returns what happened, in words for a toast.
    pub fn toggle_viewer(&mut self) -> Result<&'static str> {
        if self.viewer.is_some() {
            self.close_viewer();
            return Ok("image window closed");
        }
        self.open_viewer()?;
        Ok("image window open · it follows the selected hit")
    }

    #[cfg(feature = "window")]
    fn open_viewer(&mut self) -> Result<()> {
        let Some(s) = &self.shown else {
            return Err(anyhow!("select a hit first"));
        };
        let Some(rgba) = &s.rgba else {
            return Err(anyhow!("the selected hit has no image"));
        };
        let frame = std::env::temp_dir().join(format!("urna-view-{}.frm", std::process::id()));
        // the window reads the frame as it starts: write it before spawning.
        crate::tui::view::frame::write(
            &frame,
            rgba.width(),
            rgba.height(),
            &s.title,
            rgba.as_raw(),
        )?;
        let child = std::process::Command::new(std::env::current_exe()?)
            .arg("view")
            .arg(&frame)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| anyhow!("image window: {e}"))?;
        self.viewer = Some(Viewer { child, frame });
        Ok(())
    }

    #[cfg(not(feature = "window"))]
    fn open_viewer(&mut self) -> Result<()> {
        Err(anyhow!(
            "this build has no image window: cargo build --features window"
        ))
    }

    fn write_frame(&mut self) {
        #[cfg(feature = "window")]
        if let (Some(v), Some(s)) = (&self.viewer, &self.shown)
            && let Some(rgba) = &s.rgba
        {
            let _ = crate::tui::view::frame::write(
                &v.frame,
                rgba.width(),
                rgba.height(),
                &s.title,
                rgba.as_raw(),
            );
        }
    }

    fn close_viewer(&mut self) {
        if let Some(mut v) = self.viewer.take() {
            let _ = std::fs::remove_file(&v.frame);
            let _ = v.child.kill();
            let _ = v.child.wait();
        }
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

    pub fn render(&mut self, buf: &mut Buffer, area: Rect) {
        let Some(p) = &self.picker else { return };
        let font = p.font_size();
        let pixels = p.protocol_type() != ProtocolType::Halfblocks;
        let Some(s) = self.shown.as_mut() else { return };
        let fit = fit(area, s.size, (font.width, font.height));
        if pixels {
            // the terminal draws pixels (kitty, sixel, iterm2): the image itself.
            if let Some(proto) = s.proto.as_mut() {
                StatefulImage::default()
                    .resize(Resize::Fit(Some(FilterType::Lanczos3)))
                    .render(fit, buf, proto);
            }
            return;
        }
        // no pixels: quadrant characters, recomputed only when the rect moves.
        let Some(rgba) = &s.rgba else { return };
        if s.blocks.as_ref().is_none_or(|(r, _)| *r != fit) {
            s.blocks = Some((fit, super::blocks::cells(rgba, fit.width, fit.height)));
        }
        if let Some((r, cells)) = &s.blocks {
            super::blocks::draw(buf, *r, cells);
        }
    }

    /// What draws the image: the terminal's protocol, or characters.
    pub fn mode(&self) -> &'static str {
        match self.protocol() {
            Some(ProtocolType::Kitty) => "kitty",
            Some(ProtocolType::Sixel) => "sixel",
            Some(ProtocolType::Iterm2) => "iterm2",
            _ => "quadrants",
        }
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

fn decode(session: &Session, blob: usize) -> Result<image::DynamicImage> {
    let bytes = session.rt.blob_bytes(blob)?;
    let img = avif_decode::Decoder::from_avif(bytes)
        .map_err(|e| anyhow!("avif: {e}"))?
        .to_image()
        .map_err(|e| anyhow!("avif: {e}"))?;
    let rgba = |w: usize, h: usize, px: Vec<[u8; 4]>| {
        image::RgbaImage::from_raw(w as u32, h as u32, px.concat())
            .map(image::DynamicImage::ImageRgba8)
            .ok_or_else(|| anyhow!("avif: {w}x{h} does not match its pixels"))
    };
    let to8 = |v: u16| (v >> 8) as u8;
    match img {
        avif_decode::Image::Rgb8(i) => rgba(
            i.width(),
            i.height(),
            i.pixels().map(|p| [p.r, p.g, p.b, 255]).collect(),
        ),
        avif_decode::Image::Rgba8(i) => rgba(
            i.width(),
            i.height(),
            i.pixels().map(|p| [p.r, p.g, p.b, p.a]).collect(),
        ),
        avif_decode::Image::Rgb16(i) => rgba(
            i.width(),
            i.height(),
            i.pixels()
                .map(|p| [to8(p.r), to8(p.g), to8(p.b), 255])
                .collect(),
        ),
        avif_decode::Image::Rgba16(i) => rgba(
            i.width(),
            i.height(),
            i.pixels()
                .map(|p| [to8(p.r), to8(p.g), to8(p.b), to8(p.a)])
                .collect(),
        ),
        avif_decode::Image::Gray8(i) => rgba(
            i.width(),
            i.height(),
            i.pixels()
                .map(|p| [p.value(), p.value(), p.value(), 255])
                .collect(),
        ),
        avif_decode::Image::Gray16(i) => rgba(
            i.width(),
            i.height(),
            i.pixels()
                .map(|p| {
                    let v = to8(p.value());
                    [v, v, v, 255]
                })
                .collect(),
        ),
    }
}
