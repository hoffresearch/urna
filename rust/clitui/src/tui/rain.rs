//! Hash rain: falling hex glyphs behind the logo, a nod to the
//! `sha256:` every urna file carries. Adapted from tui-rain 1.0.1
//! (levilutz, MIT): same stateless model, where every frame recomputes the
//! drops from a fixed seed and the elapsed time, so there is nothing to
//! update between frames. changes: the palette colors, a tiny splitmix rng
//! in place of rand_pcg, and the glyphs are written at the area offset (the
//! original wrote at buffer origin, which only worked for full-screen use).

use std::time::Duration;

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};

use super::pal;

const GLYPHS: &[char] = &[
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
];

/// splitmix64: deterministic, dependency-free, good enough for pixels.
struct Mix(u64);

impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

fn uniform(seed: u64, lo: f64, hi: f64) -> f64 {
    (seed as f64 / u64::MAX as f64) * (hi - lo) + lo
}

pub struct Rain {
    pub elapsed: Duration,
    /// one drop per `sparseness` cells.
    pub sparseness: usize,
    /// cells per second.
    pub speed: f64,
    pub tail: Duration,
    pub color: Color,
    pub head: Color,
}

impl Rain {
    /// The quiet default: sparse, slow, in the line color of the ramp.
    pub fn quiet(elapsed: Duration) -> Self {
        Self {
            elapsed,
            sparseness: 70,
            speed: 6.0,
            tail: Duration::from_millis(1600),
            color: pal::SURFACE,
            head: pal::LINE,
        }
    }

    fn drop_glyphs(
        &self,
        entropy: &[u64],
        area: Rect,
        out: &mut Vec<(u16, u16, f64, char, Style)>,
    ) {
        let Some(&first) = entropy.first() else {
            return;
        };
        let elapsed = self.elapsed.as_secs_f64();
        let track = entropy.len() as u16;
        let speed = uniform(first, self.speed * 0.5, self.speed * 1.5).max(1e-3);
        let cycle = entropy.len() as f64 / speed;
        let offset = uniform(first, 0.0, cycle);
        let head_y = (((elapsed + offset) % cycle) * speed) as u16;
        let len = ((speed * self.tail.as_secs_f64()) as u16).min(area.height);
        for dy in 0..len {
            let age = dy as f64 / speed;
            if age > elapsed {
                continue;
            }
            let cycle_n = ((elapsed + offset - age) / cycle) as usize;
            if cycle_n == 0 {
                continue;
            }
            let x = (entropy[cycle_n % entropy.len()] % area.width as u64) as u16;
            let y = (head_y + track - dy) % track;
            if y >= area.height {
                continue;
            }
            let t_off = uniform(entropy[y as usize], 0.0, 1.2 * GLYPHS.len() as f64);
            let ch = GLYPHS[(((t_off + elapsed) / 1.2) as usize) % GLYPHS.len()];
            let fade = dy as f32 / len.max(1) as f32;
            let mut style = Style::new().fg(if dy == 0 {
                self.head
            } else {
                pal::mix(self.color, pal::BG, fade * 0.8)
            });
            if dy == 0 {
                style = style.add_modifier(Modifier::BOLD);
            }
            out.push((area.x + x, area.y + y, age, ch, style));
        }
    }

    pub fn render(&self, area: Rect, buf: &mut Buffer) {
        if area.width == 0 || area.height == 0 || self.sparseness == 0 {
            return;
        }
        let mut rng = Mix(0x75_72_6e_61);
        let drops = (area.width as usize * area.height as usize) / self.sparseness * 2;
        let mut glyphs = Vec::new();
        for _ in 0..drops {
            let track = area.height as u64 * 3 / 2 + rng.next() % area.height as u64;
            let entropy: Vec<u64> = (0..track).map(|_| rng.next()).collect();
            self.drop_glyphs(&entropy, area, &mut glyphs);
        }
        // oldest first, so drop heads land on top.
        glyphs.sort_by(|a, b| b.2.total_cmp(&a.2));
        for (x, y, _, ch, style) in glyphs {
            if let Some(cell) = buf.cell_mut((x, y)) {
                cell.set_char(ch).set_style(style);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rain_stays_inside_an_offset_area() {
        let full = Rect::new(0, 0, 40, 20);
        let area = Rect::new(10, 5, 12, 8);
        let mut buf = Buffer::empty(full);
        Rain::quiet(Duration::from_secs(9)).render(area, &mut buf);
        let mut drew = 0;
        for y in 0..20 {
            for x in 0..40 {
                if buf[(x, y)].symbol() != " " {
                    assert!(area.contains((x, y).into()), "glyph at {x},{y}");
                    drew += 1;
                }
            }
        }
        assert!(drew > 0);
    }

    #[test]
    fn rain_is_deterministic_for_a_given_time() {
        let area = Rect::new(0, 0, 30, 10);
        let (mut a, mut b) = (Buffer::empty(area), Buffer::empty(area));
        Rain::quiet(Duration::from_millis(4200)).render(area, &mut a);
        Rain::quiet(Duration::from_millis(4200)).render(area, &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn empty_area_draws_nothing() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 4, 4));
        Rain::quiet(Duration::from_secs(3)).render(Rect::new(0, 0, 0, 0), &mut buf);
        assert_eq!(buf, Buffer::empty(Rect::new(0, 0, 4, 4)));
    }
}
