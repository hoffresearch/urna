//! Every animation the ui runs, as tachyonfx presets. Screens never build
//! effects inline: they ask for one of these by intent (a page coming in, a
//! row landing, a success), so the motion language stays the same across
//! the installer and the explorer. durations are short on purpose; the
//! effects decorate a change, they never gate input.

use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use tachyonfx::fx::{self, EvolveSymbolSet};
use tachyonfx::pattern::{RadialPattern, SweepPattern};
use tachyonfx::{Effect, EffectManager, Interpolation::*, Motion};

use super::pal;

/// One manager per screen; keys make an effect unique (a new "page"
/// replaces the running one instead of stacking).
pub type Fx = EffectManager<&'static str>;

/// The symbol materializing: a sweep from the top while the glyphs
/// coalesce, then the title gradient.
pub fn logo_in(area: Rect) -> Effect {
    fx::parallel(&[
        fx::sweep_in(Motion::UpToDown, 12, 3, pal::BG, (1100, QuadOut)),
        fx::coalesce((900, CubicOut)),
    ])
    .with_area(area)
}

/// The slow breathing of the symbol after it lands: a hue and lightness
/// swing that never completes.
pub fn shimmer(area: Rect) -> Effect {
    fx::repeating(fx::ping_pong(fx::hsl_shift_fg(
        [-12.0, 10.0, 9.0],
        (2200, SineInOut),
    )))
    .with_area(area)
}

/// Text resolving out of block glyphs (the wordmark, a title).
pub fn resolve(area: Rect, ms: u32) -> Effect {
    fx::evolve_from(
        (EvolveSymbolSet::Quadrants, Style::new().fg(pal::GLOW)),
        (ms, QuadOut),
    )
    .with_area(area)
}

/// A page (installer step, explorer tab) coming in from the left.
pub fn page_in(area: Rect) -> Effect {
    fx::parallel(&[
        fx::sweep_in(Motion::LeftToRight, 20, 6, pal::BG, (480, QuadOut)),
        fx::fade_from_fg(pal::BG, (420, SineOut)),
    ])
    .with_area(area)
}

/// A panel opening from its center (overlays, the file picker).
pub fn open(area: Rect) -> Effect {
    fx::coalesce((360, QuadOut))
        .with_pattern(RadialPattern::center().with_transition_width(6.0))
        .with_area(area)
}

/// One row landing, after `delay_ms` (staggered lists).
pub fn row_in(area: Rect, delay_ms: u32) -> Effect {
    fx::delay(
        delay_ms,
        fx::parallel(&[
            fx::slide_in(Motion::RightToLeft, 8, 0, pal::BG, (320, QuadOut)),
            fx::fade_from_fg(pal::GLOW, (500, SineOut)),
        ]),
    )
    .with_area(area)
}

/// A row that just finished: it lights up in `color` and settles back.
pub fn flash(area: Rect, color: Color) -> Effect {
    fx::fade_from_fg(color, (700, QuadOut)).with_area(area)
}

/// The glint that travels along a gauge while work is in flight.
pub fn glint(area: Rect) -> Effect {
    fx::repeating(fx::sequence(&[
        fx::lighten_fg(0.35, (900, SineInOut))
            .with_pattern(SweepPattern::new(Motion::LeftToRight, 6)),
        fx::sleep(300),
    ]))
    .with_area(area)
}

/// A toast sliding in from the right edge.
pub fn toast_in(area: Rect) -> Effect {
    fx::parallel(&[
        fx::slide_in(Motion::RightToLeft, 6, 0, pal::BG, (260, QuadOut)),
        fx::fade_from_fg(pal::BG, (260, QuadOut)),
    ])
    .with_area(area)
}

/// The whole screen coming up on start.
pub fn boot(area: Rect) -> Effect {
    fx::sequence(&[
        fx::fade_from(pal::BG, pal::BG, (220, QuadOut)),
        fx::coalesce((520, CubicOut)),
    ])
    .with_area(area)
}

/// Leaving: the screen dissolves into the ground.
pub fn leave(area: Rect) -> Effect {
    fx::parallel(&[
        fx::dissolve((360, QuadIn)),
        fx::fade_to_fg(pal::BG, (360, QuadIn)),
    ])
    .with_area(area)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::buffer::Buffer;
    use std::time::Duration;

    #[test]
    fn every_preset_runs_to_completion_or_loops_without_panicking() {
        let area = Rect::new(0, 0, 20, 4);
        let mut buf = Buffer::empty(area);
        buf.set_string(0, 0, "urna hash-verified", Style::new().fg(pal::INK));
        let mut fx = Fx::default();
        for e in [
            logo_in(area),
            shimmer(area),
            resolve(area, 300),
            page_in(area),
            open(area),
            row_in(area, 50),
            flash(area, pal::HIGH),
            glint(area),
            toast_in(area),
            boot(area),
            leave(area),
        ] {
            fx.add_effect(e);
        }
        for _ in 0..80 {
            fx.process_effects(Duration::from_millis(33).into(), &mut buf, area);
        }
        // the looping ones (shimmer, glint) are still alive; the rest ended.
        assert!(fx.is_running());
    }
}
