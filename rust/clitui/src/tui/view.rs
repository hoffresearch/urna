//! `urna view FRAME` (feature `window`, hidden): a native window with the
//! image the ask tab has selected, at full resolution, for terminals that
//! draw no pixels. The explorer writes the frame file (`frame::write`) on
//! every selection; this process polls its mtime and redraws. SDL runs on
//! this process's main thread, which macOS requires of every window; the
//! explorer keeps the terminal. Esc, q or closing the window ends it.

use std::path::Path;
use std::time::{Duration, SystemTime};

use anyhow::{Result, anyhow};
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::{Color, PixelFormatEnum};
use sdl2::rect::Rect;

/// The frame file: `URNAFRM1`, width and height as u32 le, the title's
/// length as u32 le and its utf-8 bytes, then width * height * 4 rgba bytes.
pub mod frame {
    use std::path::Path;

    use anyhow::{Result, bail};

    const MAGIC: &[u8; 8] = b"URNAFRM1";

    pub struct Frame {
        pub width: u32,
        pub height: u32,
        pub title: String,
        pub rgba: Vec<u8>,
    }

    /// Writes beside then renames, so the reader never sees half a frame.
    pub fn write(path: &Path, width: u32, height: u32, title: &str, rgba: &[u8]) -> Result<()> {
        let mut out = Vec::with_capacity(20 + title.len() + rgba.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&width.to_le_bytes());
        out.extend_from_slice(&height.to_le_bytes());
        out.extend_from_slice(&(title.len() as u32).to_le_bytes());
        out.extend_from_slice(title.as_bytes());
        out.extend_from_slice(rgba);
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, out)?;
        std::fs::rename(tmp, path)?;
        Ok(())
    }

    pub fn read(path: &Path) -> Result<Frame> {
        let b = std::fs::read(path)?;
        let u32_at = |i: usize| -> Result<u32> {
            match b.get(i..i + 4) {
                Some(&[a, b2, c, d]) => Ok(u32::from_le_bytes([a, b2, c, d])),
                _ => bail!("frame truncated"),
            }
        };
        if b.get(..8) != Some(MAGIC.as_slice()) {
            bail!("not a frame file");
        }
        let (width, height, tlen) = (u32_at(8)?, u32_at(12)?, u32_at(16)? as usize);
        let title = String::from_utf8_lossy(b.get(20..20 + tlen).unwrap_or_default()).into_owned();
        let need = width as usize * height as usize * 4;
        let rgba = b
            .get(20 + tlen..20 + tlen + need)
            .map(<[u8]>::to_vec)
            .unwrap_or_default();
        if rgba.len() != need {
            bail!("frame truncated");
        }
        Ok(Frame {
            width,
            height,
            title,
            rgba,
        })
    }
}

fn mtime(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

pub fn run(path: &Path) -> Result<i32> {
    let first = frame::read(path)?;
    let sdl = sdl2::init().map_err(|e| anyhow!("sdl: {e}"))?;
    let video = sdl.video().map_err(|e| anyhow!("sdl video: {e}"))?;
    // bilinear when the window is larger or smaller than the image.
    sdl2::hint::set("SDL_RENDER_SCALE_QUALITY", "1");
    let bounds = video
        .display_usable_bounds(0)
        .map_err(|e| anyhow!("sdl display: {e}"))?;
    // as tall as 85% of the screen allows, the image's aspect kept.
    let h = (bounds.height() * 85 / 100).max(200);
    let w = (h as u64 * u64::from(first.width) / u64::from(first.height.max(1))) as u32;
    let window = video
        .window(&first.title, w, h)
        .position_centered()
        .resizable()
        .allow_highdpi()
        .build()?;
    let mut canvas = window.into_canvas().accelerated().present_vsync().build()?;
    let creator = canvas.texture_creator();
    let mut events = sdl.event_pump().map_err(|e| anyhow!("sdl events: {e}"))?;

    let mut current = first;
    let mut seen = mtime(path);
    let mut texture = None;
    let mut dirty = true;
    loop {
        for ev in events.poll_iter() {
            match ev {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape | Keycode::Q),
                    ..
                } => return Ok(0),
                Event::Window { .. } => dirty = true,
                _ => {}
            }
        }
        let now = mtime(path);
        if now.is_none() {
            // the explorer removed the frame: it closed the window or quit.
            return Ok(0);
        }
        if now != seen {
            seen = now;
            if let Ok(f) = frame::read(path) {
                current = f;
                texture = None;
                dirty = true;
            }
        }
        if dirty {
            if texture.is_none() {
                let mut t = creator.create_texture_static(
                    PixelFormatEnum::ABGR8888,
                    current.width,
                    current.height,
                )?;
                t.update(None, &current.rgba, current.width as usize * 4)?;
                texture = Some(t);
                canvas.window_mut().set_title(&current.title)?;
            }
            canvas.set_draw_color(Color::RGB(14, 22, 30));
            canvas.clear();
            let (ow, oh) = canvas
                .output_size()
                .map_err(|e| anyhow!("sdl output: {e}"))?;
            let scale = (f64::from(ow) / f64::from(current.width))
                .min(f64::from(oh) / f64::from(current.height));
            let (dw, dh) = (
                (f64::from(current.width) * scale) as u32,
                (f64::from(current.height) * scale) as u32,
            );
            let dst = Rect::new(((ow - dw) / 2) as i32, ((oh - dh) / 2) as i32, dw, dh);
            if let Some(t) = &texture {
                canvas
                    .copy(t, None, dst)
                    .map_err(|e| anyhow!("sdl copy: {e}"))?;
            }
            canvas.present();
            dirty = false;
        }
        std::thread::sleep(Duration::from_millis(30));
    }
}
