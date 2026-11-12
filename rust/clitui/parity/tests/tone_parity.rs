//! The cli's xterm-256 fold (`rust/clitui/src/cmd/tone.rs`) against coolor's
//! `nearest_ansi_in_range(16, 255)`, the call it replaced, over a 16-level
//! grid of every channel (4096 colors) plus the palette the ui paints.
//!
//! coolor's table carries two greys off xterm's ramp: index 241 as 0x60
//! (xterm: 98) and 242 as 0x66 (xterm: 108). the fold uses xterm's values,
//! what the terminal shows, so the two answers may differ only where one of
//! them is 241 or 242; everywhere the fold equals coolor's own metric run
//! over coolor's table with those two entries corrected.

#[allow(dead_code)]
#[path = "../../src/cmd/tone.rs"]
mod tone;

use coolor::{AnsiColor, Rgb};

fn grid() -> impl Iterator<Item = (u8, u8, u8)> {
    let level = |i: u8| i * 17;
    (0..16).flat_map(move |r| {
        (0..16).flat_map(move |g| (0..16).map(move |b| (level(r), level(g), level(b))))
    })
}

fn coolor_fold(rgb: (u8, u8, u8)) -> u8 {
    Rgb::new(rgb.0, rgb.1, rgb.2)
        .nearest_ansi_in_range(16, 255)
        .code
}

/// coolor's metric and search, over coolor's palette with 241 and 242 put
/// back on xterm's grey ramp.
fn coolor_corrected(rgb: (u8, u8, u8)) -> u8 {
    let target = Rgb::new(rgb.0, rgb.1, rgb.2);
    let shown = |code: u8| match code {
        241 => Rgb::new(98, 98, 98),
        242 => Rgb::new(108, 108, 108),
        _ => AnsiColor::new(code).to_rgb(),
    };
    let mut best = 16;
    let mut best_distance = target.distance_to(shown(16));
    for code in 17..=255 {
        let d = target.distance_to(shown(code));
        if d < best_distance {
            best = code;
            best_distance = d;
        }
    }
    best
}

#[test]
fn the_palette_differs_from_coolor_only_at_241_and_242() {
    for code in 16..=255u8 {
        let theirs = AnsiColor::new(code).to_rgb();
        let ours = tone::xterm_rgb(code);
        if code == 241 || code == 242 {
            assert_ne!((theirs.r, theirs.g, theirs.b), ours, "{code}");
        } else {
            assert_eq!((theirs.r, theirs.g, theirs.b), ours, "{code}");
        }
    }
}

#[test]
fn the_metric_is_coolors_bit_for_bit() {
    for rgb in grid() {
        for code in 16..=255u8 {
            let p = tone::xterm_rgb(code);
            let theirs = Rgb::new(rgb.0, rgb.1, rgb.2).distance_to(Rgb::new(p.0, p.1, p.2));
            assert_eq!(
                tone::distance(rgb, p).to_bits(),
                theirs.to_bits(),
                "{rgb:?} to {code}"
            );
        }
    }
}

#[test]
fn the_fold_matches_coolor_on_the_4096_grid() {
    let mut differ = Vec::new();
    for rgb in grid() {
        let ours = tone::ansi256(rgb);
        assert_eq!(ours, coolor_corrected(rgb), "{rgb:?}");
        let theirs = coolor_fold(rgb);
        if ours != theirs {
            assert!(
                [ours, theirs].iter().any(|c| *c == 241 || *c == 242),
                "{rgb:?}: ours {ours}, coolor {theirs}"
            );
            differ.push((rgb, ours, theirs));
        }
    }
    // measured when the fold replaced coolor: two of 4096, both at the greys.
    eprintln!("differ from coolor: {differ:?}");
    assert!(differ.len() <= 2, "{differ:?}");
}

#[test]
fn the_ui_palette_folds_as_coolor_did() {
    for rgb in [
        tone::HIGH,
        tone::MID,
        tone::LOW,
        tone::TEAL,
        tone::FAINT,
        (0x00, 0x2b, 0x36),
        (0, 0, 0),
        (255, 255, 255),
    ] {
        assert_eq!(tone::ansi256(rgb), coolor_fold(rgb), "{rgb:?}");
    }
}
