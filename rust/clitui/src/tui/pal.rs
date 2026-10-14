//! The urna.dev palette on the terminal. Same source as the site
//! (apps/web/constants/themes.ts, ADR 0029): the page was drawn in grey,
//! ground 32 and ink 248, and every grey is carried onto the ramp between
//! the two ends of the palette, dark `#002b36` and light `#dfe8ea`. accents
//! come apart: the title gradient, the glow, and the thermometer the
//! benchmark tables use (low / mid / high).
//!
//! everything renders in rgb; `fit` downsamples the finished frame once for
//! terminals without truecolor, so tachyonfx can interpolate in full color.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};

use crate::cmd::tone::{self, Depth};

const DARK: (i32, i32, i32) = (0x00, 0x2b, 0x36);
const LIGHT: (i32, i32, i32) = (0xdf, 0xe8, 0xea);

/// the grey `l` (0..=255) of the original drawing, carried onto the ramp
/// (themes.ts `tint`: anchors 32 -> dark, 248 -> light, clamped).
pub const fn tint(l: u8) -> Color {
    const fn ch(d: i32, w: i32, l: i32) -> u8 {
        let v = d + (w - d) * (l - 32) / 216;
        if v < 0 {
            0
        } else if v > 255 {
            255
        } else {
            v as u8
        }
    }
    let l = l as i32;
    Color::Rgb(
        ch(DARK.0, LIGHT.0, l),
        ch(DARK.1, LIGHT.1, l),
        ch(DARK.2, LIGHT.2, l),
    )
}

pub const BG: Color = tint(32);
pub const SURFACE: Color = tint(44);
pub const RAISED: Color = tint(58);
pub const LINE: Color = tint(84);
pub const FAINT: Color = tint(130);
pub const DIM: Color = tint(170);
pub const INK: Color = tint(248);
pub const INK_HI: Color = tint(255);
pub const TITLE_FROM: Color = Color::Rgb(0xa8, 0xdc, 0xdc);
pub const TITLE_TO: Color = Color::Rgb(0x85, 0xc8, 0xc8);
pub const GLOW: Color = Color::Rgb(0x85, 0xc8, 0xc8);
pub const LOW: Color = Color::Rgb(0xc4, 0x5c, 0x54);
pub const MID: Color = Color::Rgb(0xc9, 0xa8, 0x5c);
pub const HIGH: Color = Color::Rgb(0x6e, 0xb0, 0x7c);

fn rgb(c: Color) -> (f32, f32, f32) {
    match c {
        Color::Rgb(r, g, b) => (r as f32, g as f32, b as f32),
        _ => (0.0, 0.0, 0.0),
    }
}

/// Linear blend of two rgb colors, `t` in 0..=1.
pub fn mix(a: Color, b: Color, t: f32) -> Color {
    let t = t.clamp(0.0, 1.0);
    let (a, b) = (rgb(a), rgb(b));
    let f = |x: f32, y: f32| (x + (y - x) * t).round() as u8;
    Color::Rgb(f(a.0, b.0), f(a.1, b.1), f(a.2, b.2))
}

/// The thermometer: low (0) -> mid (0.5) -> high (1).
pub fn thermo(t: f32) -> Color {
    if t < 0.5 {
        mix(LOW, MID, t * 2.0)
    } else {
        mix(MID, HIGH, (t - 0.5) * 2.0)
    }
}

pub fn text() -> Style {
    Style::new().fg(INK).bg(BG)
}
pub fn dim() -> Style {
    Style::new().fg(DIM)
}
pub fn faint() -> Style {
    Style::new().fg(FAINT)
}
pub fn title() -> Style {
    Style::new().fg(TITLE_FROM).add_modifier(Modifier::BOLD)
}
pub fn accent() -> Style {
    Style::new().fg(GLOW)
}
pub fn key() -> Style {
    Style::new().fg(TITLE_FROM).add_modifier(Modifier::BOLD)
}
pub fn ok() -> Style {
    Style::new().fg(HIGH)
}
pub fn warn() -> Style {
    Style::new().fg(MID)
}
pub fn err() -> Style {
    Style::new().fg(LOW)
}

/// Paints the ground under `area` (the page background of the site).
pub fn ground(buf: &mut Buffer, area: Rect) {
    buf.set_style(area.intersection(buf.area), Style::new().bg(BG).fg(INK));
}

/// The final pass of every frame: fold rgb into what the terminal shows.
/// 256-color terminals get the nearest xterm index (coolor), `NO_COLOR`
/// terminals keep the modifiers and lose the color.
pub fn fit(buf: &mut Buffer, area: Rect, depth: Depth) {
    if depth == Depth::True {
        return;
    }
    let map = |c: Color| match (c, depth) {
        (Color::Rgb(..), Depth::None) => Color::Reset,
        (Color::Rgb(r, g, b), _) => Color::Indexed(tone::ansi256((r, g, b))),
        (other, _) => other,
    };
    let area = area.intersection(buf.area);
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            if let Some(cell) = buf.cell_mut((x, y)) {
                cell.fg = map(cell.fg);
                cell.bg = map(cell.bg);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tint_hits_the_site_anchors() {
        assert_eq!(tint(32), Color::Rgb(0x00, 0x2b, 0x36));
        assert_eq!(tint(248), Color::Rgb(0xdf, 0xe8, 0xea));
        // below the dark anchor the ramp clamps instead of wrapping.
        assert!(matches!(tint(0), Color::Rgb(0, _, _)));
    }

    #[test]
    fn thermo_runs_low_mid_high() {
        assert_eq!(thermo(0.0), LOW);
        assert_eq!(thermo(0.5), MID);
        assert_eq!(thermo(1.0), HIGH);
    }

    #[test]
    fn fit_downsamples_and_strips() {
        let area = Rect::new(0, 0, 2, 1);
        let mut buf = Buffer::empty(area);
        buf.set_style(area, Style::new().fg(GLOW).bg(BG));
        let mut b256 = buf.clone();
        fit(&mut b256, area, Depth::Ansi256);
        assert!(matches!(b256[(0, 0)].fg, Color::Indexed(_)));
        fit(&mut buf, area, Depth::None);
        assert_eq!(buf[(1, 0)].bg, Color::Reset);
    }
}
