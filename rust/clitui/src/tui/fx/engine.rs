//! The effect engine: a tree of atoms and combinators, each run by its
//! elapsed time over the cells of an area. Randomness is a hash of the
//! cell's place, so a frame is a pure function of time and the effect is
//! steady under redraws.

use std::time::Duration;

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};

use crate::tui::pal;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Motion {
    LeftToRight,
    RightToLeft,
    UpToDown,
}

#[derive(Clone, Copy, Debug)]
pub enum Ease {
    QuadIn,
    QuadOut,
    CubicOut,
    SineOut,
    SineInOut,
}

impl Ease {
    fn apply(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::QuadIn => t * t,
            Self::QuadOut => 1.0 - (1.0 - t) * (1.0 - t),
            Self::CubicOut => 1.0 - (1.0 - t).powi(3),
            Self::SineOut => (t * std::f32::consts::FRAC_PI_2).sin(),
            Self::SineInOut => 0.5 - 0.5 * (t * std::f32::consts::PI).cos(),
        }
    }
}

/// What one atom does to a cell at progress `p` in `0..=1`.
#[derive(Clone, Copy, Debug)]
pub enum Atom {
    /// cells ahead of a front moving in the motion are drawn in the color
    /// (hidden against the ground), with a soft band of `u16` cells.
    Sweep(Motion, u16, Color),
    /// the foreground comes from the color to its own.
    FadeFromFg(Color),
    /// the foreground goes to the color.
    FadeToFg(Color),
    /// glyphs settle from noise in a random order.
    Coalesce,
    /// glyphs settle from the center outward.
    CoalesceRadial,
    /// glyphs settle from quadrant blocks in the style, random order.
    Evolve(Style),
    /// glyphs vanish in a random order.
    Dissolve,
    /// a lightness swing of `f32` around the cell's own color.
    Breathe(f32),
    /// a band `u16` wide lightened by `f32`, travelling in the motion.
    Glint(Motion, u16, f32),
}

#[derive(Clone, Debug)]
enum Node {
    Atom { kind: Atom, ms: u32, ease: Ease },
    Sleep(u32),
    Parallel(Vec<Node>),
    Sequence(Vec<Node>),
    Delay(u32, Box<Node>),
    Repeat(Box<Node>),
    PingPong(Box<Node>),
}

impl Node {
    /// the length in ms; `None` for one that never ends.
    fn len(&self) -> Option<u32> {
        match self {
            Self::Atom { ms, .. } | Self::Sleep(ms) => Some(*ms),
            Self::Parallel(v) => v.iter().map(Node::len).try_fold(0, |m, l| Some(m.max(l?))),
            Self::Sequence(v) => v.iter().map(Node::len).try_fold(0, |s, l| Some(s + l?)),
            Self::Delay(ms, inner) => inner.len().map(|l| l + ms),
            Self::Repeat(_) => None,
            Self::PingPong(inner) => inner.len().map(|l| l * 2),
        }
    }

    fn run(&self, t: u32, buf: &mut Buffer, area: Rect) {
        match self {
            Self::Atom { kind, ms, ease } => {
                let p = if *ms == 0 { 1.0 } else { t as f32 / *ms as f32 };
                apply(*kind, ease.apply(p), buf, area);
            }
            Self::Sleep(_) => {}
            Self::Parallel(v) => {
                for n in v {
                    if n.len().is_none_or(|l| t < l) {
                        n.run(t, buf, area);
                    }
                }
            }
            Self::Sequence(v) => {
                let mut at = t;
                for n in v {
                    match n.len() {
                        Some(l) if at >= l => at -= l,
                        _ => {
                            n.run(at, buf, area);
                            return;
                        }
                    }
                }
            }
            Self::Delay(ms, inner) => {
                if t >= *ms {
                    inner.run(t - ms, buf, area);
                } else {
                    // before it starts the area is held hidden, so the
                    // staggered row does not flash before its turn.
                    apply(Atom::FadeToFg(pal::BG), 1.0, buf, area);
                }
            }
            Self::Repeat(inner) => match inner.len() {
                Some(l) if l > 0 => inner.run(t % l, buf, area),
                _ => inner.run(t, buf, area),
            },
            Self::PingPong(inner) => match inner.len() {
                Some(l) if l > 0 => {
                    let m = t % (2 * l);
                    inner.run(if m < l { m } else { 2 * l - m - 1 }, buf, area);
                }
                _ => inner.run(t, buf, area),
            },
        }
    }
}

/// One effect over an area.
#[derive(Clone, Debug)]
pub struct Effect {
    node: Node,
    area: Option<Rect>,
    elapsed: Duration,
}

impl Effect {
    pub fn with_area(mut self, area: Rect) -> Self {
        self.area = Some(area);
        self
    }

    fn done(&self) -> bool {
        self.node
            .len()
            .is_some_and(|l| self.elapsed.as_millis() >= u128::from(l))
    }
}

fn wrap(node: Node) -> Effect {
    Effect {
        node,
        area: None,
        elapsed: Duration::ZERO,
    }
}

pub(super) fn atom(kind: Atom, ms: u32, ease: Ease) -> Effect {
    wrap(Node::Atom { kind, ms, ease })
}

pub(super) fn sleep(ms: u32) -> Effect {
    wrap(Node::Sleep(ms))
}

pub(super) fn parallel(v: Vec<Effect>) -> Effect {
    wrap(Node::Parallel(v.into_iter().map(|e| e.node).collect()))
}

pub(super) fn sequence(v: Vec<Effect>) -> Effect {
    wrap(Node::Sequence(v.into_iter().map(|e| e.node).collect()))
}

pub(super) fn delay(ms: u32, e: Effect) -> Effect {
    wrap(Node::Delay(ms, Box::new(e.node)))
}

pub(super) fn repeat(e: Effect) -> Effect {
    wrap(Node::Repeat(Box::new(e.node)))
}

pub(super) fn ping_pong(e: Effect) -> Effect {
    wrap(Node::PingPong(Box::new(e.node)))
}

/// The effects of one screen; a key makes an effect unique (a new "page"
/// replaces the running one instead of stacking).
#[derive(Debug, Default)]
pub struct Fx {
    effects: Vec<(Option<&'static str>, Effect)>,
}

impl Fx {
    pub fn add_effect(&mut self, e: Effect) {
        self.effects.push((None, e));
    }

    pub fn add_unique_effect(&mut self, key: &'static str, e: Effect) {
        self.cancel_unique_effect(key);
        self.effects.push((Some(key), e));
    }

    pub fn cancel_unique_effect(&mut self, key: &'static str) {
        self.effects.retain(|(k, _)| *k != Some(key));
    }

    #[cfg(test)]
    pub fn is_running(&self) -> bool {
        !self.effects.is_empty()
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.effects.len()
    }

    /// Advance every effect by `dt` and apply it to `buf`; the ones that
    /// ended are dropped.
    pub fn process_effects(&mut self, dt: Duration, buf: &mut Buffer, screen: Rect) {
        for (_, e) in &mut self.effects {
            e.elapsed += dt;
            let area = e.area.unwrap_or(screen).intersection(buf.area);
            let t = u32::try_from(e.elapsed.as_millis()).unwrap_or(u32::MAX);
            e.node.run(t, buf, area);
        }
        self.effects.retain(|(_, e)| !e.done());
    }
}

/// a cell's own pseudo-random value in `0..1`, steady across frames.
fn noise(x: u16, y: u16) -> f32 {
    let mut h = (u32::from(x) << 16 | u32::from(y)).wrapping_mul(0x9E37_79B1);
    h ^= h >> 15;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 13;
    (h & 0xFFFF) as f32 / 65535.0
}

/// a cell's place along the motion, `0..1`.
fn along(m: Motion, x: u16, y: u16, area: Rect) -> f32 {
    let (i, n) = match m {
        Motion::LeftToRight => (x - area.x, area.width),
        Motion::RightToLeft => (area.right() - 1 - x, area.width),
        Motion::UpToDown => (y - area.y, area.height),
    };
    if n <= 1 {
        0.0
    } else {
        f32::from(i) / f32::from(n - 1)
    }
}

const NOISE_GLYPHS: [&str; 12] = ["░", "▒", "▓", "·", ":", "∙", "⋅", "˙", "¨", "⁚", "⁖", "⁘"];
const QUADRANTS: [&str; 14] = [
    "▘", "▝", "▖", "▗", "▚", "▞", "▛", "▜", "▙", "▟", "▀", "▄", "▌", "▐",
];

fn lighten(c: Color, amount: f32) -> Color {
    pal::mix(c, pal::INK_HI, amount.clamp(0.0, 1.0))
}

fn apply(kind: Atom, p: f32, buf: &mut Buffer, area: Rect) {
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            let Some(cell) = buf.cell_mut((x, y)) else {
                continue;
            };
            match kind {
                Atom::Sweep(m, band, color) => {
                    let pos = along(m, x, y, area);
                    let w = f32::from(band.max(1)) / f32::from(area.width.max(area.height).max(1));
                    let front = p * (1.0 + w);
                    if pos > front {
                        cell.set_fg(color);
                    } else if pos > front - w {
                        let k = (front - pos) / w;
                        cell.set_fg(pal::mix(color, cell.fg, k));
                    }
                }
                Atom::FadeFromFg(color) => {
                    cell.set_fg(pal::mix(color, cell.fg, p));
                }
                Atom::FadeToFg(color) => {
                    cell.set_fg(pal::mix(cell.fg, color, p));
                }
                Atom::Coalesce => {
                    if noise(x, y) > p && cell.symbol() != " " {
                        let i = (noise(y, x) * NOISE_GLYPHS.len() as f32) as usize;
                        cell.set_symbol(NOISE_GLYPHS[i.min(NOISE_GLYPHS.len() - 1)]);
                        cell.set_fg(pal::mix(cell.fg, pal::GLOW, 0.5));
                    }
                }
                Atom::CoalesceRadial => {
                    let cx = f32::from(area.x) + f32::from(area.width) / 2.0;
                    let cy = f32::from(area.y) + f32::from(area.height) / 2.0;
                    let dx = (f32::from(x) - cx) / f32::from(area.width.max(1));
                    let dy = (f32::from(y) - cy) / f32::from(area.height.max(1));
                    let d = (dx * dx + dy * dy).sqrt() * 1.4;
                    if d + noise(x, y) * 0.15 > p && cell.symbol() != " " {
                        let i = (noise(y, x) * NOISE_GLYPHS.len() as f32) as usize;
                        cell.set_symbol(NOISE_GLYPHS[i.min(NOISE_GLYPHS.len() - 1)]);
                    }
                }
                Atom::Evolve(style) => {
                    if noise(x, y) > p && cell.symbol() != " " {
                        let i = (noise(y, x) * QUADRANTS.len() as f32) as usize;
                        cell.set_symbol(QUADRANTS[i.min(QUADRANTS.len() - 1)]);
                        cell.set_style(style);
                    }
                }
                Atom::Dissolve => {
                    if noise(x, y) < p {
                        cell.set_symbol(" ");
                    }
                }
                Atom::Breathe(amount) => {
                    cell.set_fg(lighten(cell.fg, amount * p));
                }
                Atom::Glint(m, band, amount) => {
                    let pos = along(m, x, y, area);
                    let w = f32::from(band.max(1)) / f32::from(area.width.max(1));
                    let front = p * (1.0 + 2.0 * w) - w;
                    let d = (pos - front).abs();
                    if d < w {
                        cell.set_fg(lighten(cell.fg, amount * (1.0 - d / w)));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lengths_compose_and_a_sweep_hides_then_reveals() {
        let seq = sequence(vec![sleep(100), atom(Atom::Dissolve, 200, Ease::QuadIn)]);
        assert_eq!(seq.node.len(), Some(300));
        assert_eq!(repeat(sleep(10)).node.len(), None);
        assert_eq!(ping_pong(sleep(10)).node.len(), Some(20));
        assert_eq!(delay(50, sleep(10)).node.len(), Some(60));
        let area = Rect::new(0, 0, 10, 1);
        let mut buf = Buffer::empty(area);
        buf.set_string(0, 0, "0123456789", Style::new().fg(pal::INK));
        // halfway, the front has passed the first cells and not the last.
        apply(
            Atom::Sweep(Motion::LeftToRight, 1, pal::BG),
            0.5,
            &mut buf,
            area,
        );
        assert_eq!(buf[(9, 0)].fg, pal::BG);
        assert_eq!(buf[(0, 0)].fg, pal::INK);
        buf.set_string(0, 0, "0123456789", Style::new().fg(pal::INK));
        apply(
            Atom::Sweep(Motion::LeftToRight, 1, pal::BG),
            1.0,
            &mut buf,
            area,
        );
        assert_eq!(buf[(9, 0)].fg, pal::INK);
        apply(Atom::Dissolve, 1.0, &mut buf, area);
        assert!((0..10).all(|x| buf[(x, 0)].symbol() == " "));
    }
}
