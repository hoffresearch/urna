//! Color for the plain (non-ui) output: the urna.dev palette as ansi
//! escapes, and the one color-depth probe the terminal ui shares.
//!
//! color only reaches a terminal: piped or redirected output, `NO_COLOR`
//! (any value, no-color.org) or `URNA_COLOR=none` print plain text, so
//! scripts that parse `doctor` never see an escape.

use std::io::IsTerminal;

/// the site palette (urna.dev, constants/themes.ts): the thermometer the
/// benchmark and preset tables use, plus the title teal and the ink.
pub const HIGH: (u8, u8, u8) = (0x6e, 0xb0, 0x7c);
pub const MID: (u8, u8, u8) = (0xc9, 0xa8, 0x5c);
pub const LOW: (u8, u8, u8) = (0xc4, 0x5c, 0x54);
pub const TEAL: (u8, u8, u8) = (0x85, 0xc8, 0xc8);
pub const FAINT: (u8, u8, u8) = (0x8a, 0x9f, 0xa3);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Depth {
    /// 24-bit rgb escapes.
    True,
    /// the xterm 256-color cube (Terminal.app, tmux without `Tc`).
    Ansi256,
    /// no color at all.
    None,
}

/// What the terminal can show. `URNA_COLOR` (`truecolor`, `256`, `none`)
/// overrides the probe; `NO_COLOR` wins over everything.
pub fn depth() -> Depth {
    depth_from(|k| std::env::var(k).ok())
}

fn depth_from(env: impl Fn(&str) -> Option<String>) -> Depth {
    if env("NO_COLOR").is_some_and(|v| !v.is_empty()) {
        return Depth::None;
    }
    match env("URNA_COLOR").as_deref() {
        Some("none") => return Depth::None,
        Some("256") => return Depth::Ansi256,
        Some("truecolor" | "24bit") => return Depth::True,
        _ => {}
    }
    let has = |k: &str, needles: &[&str]| {
        env(k).is_some_and(|v| needles.iter().any(|n| v.to_ascii_lowercase().contains(n)))
    };
    if has("COLORTERM", &["truecolor", "24bit"])
        || has(
            "TERM_PROGRAM",
            &[
                "iterm", "wezterm", "vscode", "ghostty", "hyper", "warp", "tabby", "rio", "zed",
            ],
        )
        || has("TERM", &["kitty", "alacritty", "direct", "ghostty", "foot"])
        || env("WT_SESSION").is_some()
    {
        Depth::True
    } else {
        Depth::Ansi256
    }
}

/// The xterm-256 index nearest to an rgb triple: coolor's perceptual
/// search over the cube and the grey ramp (16..=255; 0..16 follow the
/// user's theme). its quick `to_ansi` buckets by the red channel and sends
/// the ground (#002b36, solarized base03) to navy; the full search lands on
/// the grey solarized itself falls back to. memoized: a frame repeats a
/// handful of colors thousands of times.
pub fn ansi256(rgb: (u8, u8, u8)) -> u8 {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    type Folds = Mutex<HashMap<(u8, u8, u8), u8>>;
    static CACHE: OnceLock<Folds> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(map) = cache.lock()
        && let Some(&hit) = map.get(&rgb)
    {
        return hit;
    }
    let code = coolor::Rgb::new(rgb.0, rgb.1, rgb.2)
        .nearest_ansi_in_range(16, 255)
        .code;
    if let Ok(mut map) = cache.lock() {
        map.insert(rgb, code);
    }
    code
}

/// A painter for one output stream, decided once.
#[derive(Clone, Copy)]
pub struct Tone {
    depth: Depth,
}

impl Tone {
    /// color when stdout is a terminal, plain text otherwise.
    pub fn stdout() -> Self {
        let depth = if std::io::stdout().is_terminal() {
            depth()
        } else {
            Depth::None
        };
        Self { depth }
    }

    pub fn fg(&self, text: &str, rgb: (u8, u8, u8)) -> String {
        match self.depth {
            Depth::None => text.to_string(),
            Depth::True => format!("\x1b[38;2;{};{};{}m{text}\x1b[0m", rgb.0, rgb.1, rgb.2),
            Depth::Ansi256 => format!("\x1b[38;5;{}m{text}\x1b[0m", ansi256(rgb)),
        }
    }

    pub fn bold(&self, text: &str) -> String {
        match self.depth {
            Depth::None => text.to_string(),
            _ => format!("\x1b[1m{text}\x1b[0m"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let v: Vec<(String, String)> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k| v.iter().find(|(a, _)| a == k).map(|(_, b)| b.clone())
    }

    #[test]
    fn no_color_beats_every_probe() {
        let d = depth_from(env(&[("NO_COLOR", "1"), ("COLORTERM", "truecolor")]));
        assert_eq!(d, Depth::None);
    }

    #[test]
    fn colorterm_truecolor_and_apple_terminal_fall_back() {
        assert_eq!(depth_from(env(&[("COLORTERM", "truecolor")])), Depth::True);
        let apple = env(&[
            ("TERM_PROGRAM", "Apple_Terminal"),
            ("TERM", "xterm-256color"),
        ]);
        assert_eq!(depth_from(apple), Depth::Ansi256);
        assert_eq!(
            depth_from(env(&[("URNA_COLOR", "256"), ("COLORTERM", "24bit")])),
            Depth::Ansi256
        );
    }

    #[test]
    fn ansi256_keeps_the_ground_grey_and_the_accents_teal() {
        assert_eq!(ansi256((0x00, 0x2b, 0x36)), 235);
        assert_eq!(ansi256(TEAL), 116);
        assert_eq!(ansi256(LOW), 167);
    }

    #[test]
    fn ansi256_stays_out_of_the_themeable_sixteen() {
        // 0..16 follow the user's terminal theme; the palette must not.
        for rgb in [
            (0, 0, 0),
            (255, 255, 255),
            HIGH,
            LOW,
            TEAL,
            (0x00, 0x2b, 0x36),
        ] {
            assert!(ansi256(rgb) >= 16, "{rgb:?}");
        }
    }

    #[test]
    fn plain_tone_emits_no_escape() {
        let t = Tone { depth: Depth::None };
        assert_eq!(t.fg("ok", HIGH), "ok");
        assert_eq!(t.bold("x"), "x");
    }
}
