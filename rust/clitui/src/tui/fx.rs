//! Every animation the ui runs, as presets over the ui's own effect
//! engine. Screens never build effects inline: they ask for one of these
//! by intent (a page coming in, a row landing, a success), so the motion
//! language stays the same across the installer and the explorer.
//! Durations are short on purpose; the effects decorate a change, they
//! never gate input.
//!
//! An effect runs over a drawn frame: the screen draws, then
//! `Fx::process_effects` rewrites the cells of each live effect's area by
//! how far along it is. Atoms change colors, hide cells ahead of a sweep,
//! or replace glyphs; `parallel`, `sequence`, `delay`, `repeat` and
//! `ping_pong` compose them.

mod engine;

use ratatui::layout::Rect;
use ratatui::style::{Color, Style};

use engine::{Atom, Ease, atom, parallel, ping_pong, repeat, sequence, sleep};
pub use engine::{Effect, Fx, Motion};

use super::pal;

/// The symbol materializing: a sweep from the top while the glyphs
/// coalesce, then the title gradient.
pub fn logo_in(area: Rect) -> Effect {
    parallel(vec![
        atom(
            Atom::Sweep(Motion::UpToDown, 12, pal::BG),
            1100,
            Ease::QuadOut,
        ),
        atom(Atom::Coalesce, 900, Ease::CubicOut),
    ])
    .with_area(area)
}

/// The slow breathing of the symbol after it lands: a lightness swing
/// that never completes.
pub fn shimmer(area: Rect) -> Effect {
    repeat(ping_pong(atom(Atom::Breathe(0.22), 2200, Ease::SineInOut))).with_area(area)
}

/// Text resolving out of block glyphs (the wordmark, a title).
pub fn resolve(area: Rect, ms: u32) -> Effect {
    atom(Atom::Evolve(Style::new().fg(pal::GLOW)), ms, Ease::QuadOut).with_area(area)
}

/// A page (installer step, explorer tab) coming in from the left.
pub fn page_in(area: Rect) -> Effect {
    parallel(vec![
        atom(
            Atom::Sweep(Motion::LeftToRight, 20, pal::BG),
            480,
            Ease::QuadOut,
        ),
        atom(Atom::FadeFromFg(pal::BG), 420, Ease::SineOut),
    ])
    .with_area(area)
}

/// A panel opening from its center (overlays, the file picker).
pub fn open(area: Rect) -> Effect {
    atom(Atom::CoalesceRadial, 360, Ease::QuadOut).with_area(area)
}

/// One row landing, after `delay_ms` (staggered lists).
pub fn row_in(area: Rect, delay_ms: u32) -> Effect {
    engine::delay(
        delay_ms,
        parallel(vec![
            atom(
                Atom::Sweep(Motion::RightToLeft, 8, pal::BG),
                320,
                Ease::QuadOut,
            ),
            atom(Atom::FadeFromFg(pal::GLOW), 500, Ease::SineOut),
        ]),
    )
    .with_area(area)
}

/// A row that just finished: it lights up in `color` and settles back.
pub fn flash(area: Rect, color: Color) -> Effect {
    atom(Atom::FadeFromFg(color), 700, Ease::QuadOut).with_area(area)
}

/// The glint that travels along a gauge while work is in flight.
pub fn glint(area: Rect) -> Effect {
    repeat(sequence(vec![
        atom(
            Atom::Glint(Motion::LeftToRight, 6, 0.35),
            900,
            Ease::SineInOut,
        ),
        sleep(300),
    ]))
    .with_area(area)
}

/// A toast sliding in from the right edge.
pub fn toast_in(area: Rect) -> Effect {
    parallel(vec![
        atom(
            Atom::Sweep(Motion::RightToLeft, 6, pal::BG),
            260,
            Ease::QuadOut,
        ),
        atom(Atom::FadeFromFg(pal::BG), 260, Ease::QuadOut),
    ])
    .with_area(area)
}

/// The whole screen coming up on start.
pub fn boot(area: Rect) -> Effect {
    sequence(vec![
        atom(Atom::FadeFromFg(pal::BG), 220, Ease::QuadOut),
        atom(Atom::Coalesce, 520, Ease::CubicOut),
    ])
    .with_area(area)
}

/// Leaving: the screen dissolves into the ground.
pub fn leave(area: Rect) -> Effect {
    parallel(vec![
        atom(Atom::Dissolve, 360, Ease::QuadIn),
        atom(Atom::FadeToFg(pal::BG), 360, Ease::QuadIn),
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
            fx.process_effects(Duration::from_millis(33), &mut buf, area);
        }
        // the looping ones (shimmer, glint) are still alive; the rest ended.
        assert!(fx.is_running());
        assert_eq!(fx.len(), 2);
        let mut fx = Fx::default();
        fx.add_unique_effect("page", page_in(area));
        fx.add_unique_effect("page", page_in(area));
        assert_eq!(fx.len(), 1);
        fx.cancel_unique_effect("page");
        assert!(!fx.is_running());
    }
}
