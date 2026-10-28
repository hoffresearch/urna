//! Case and accent folding for the type-ahead: `Relâmpago Vivo` ->
//! `relampago vivo`, so a query typed without accents finds the label.
//! The table covers Latin-1 Supplement and Latin Extended-A, the letters
//! of Portuguese, Spanish, French, German, Italian and the Nordic and
//! Central European names a card list carries; anything outside those
//! blocks is lowercased and kept as it is.

/// Lowercase with the accents dropped.
pub fn fold(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match base(c) {
            Some(b) => out.push_str(b),
            None => out.extend(c.to_lowercase()),
        }
    }
    out
}

/// The unaccented lowercase letters of `c`, for the two Latin blocks the
/// table covers; `None` for every other character.
fn base(c: char) -> Option<&'static str> {
    let u = u32::from(c);
    let s = match u {
        0xC0..=0xFF => LATIN_1[(u - 0xC0) as usize],
        0x100..=0x17F => LATIN_EXT_A[(u - 0x100) as usize],
        _ => return None,
    };
    (!s.is_empty()).then_some(s)
}

/// U+00C0..U+00FF; an empty entry means "not a letter", kept as it is.
const LATIN_1: [&str; 64] = [
    "a", "a", "a", "a", "a", "a", "ae", "c", "e", "e", "e", "e", "i", "i", "i", "i", // C0
    "d", "n", "o", "o", "o", "o", "o", "", "o", "u", "u", "u", "u", "y", "th", "ss", // D0
    "a", "a", "a", "a", "a", "a", "ae", "c", "e", "e", "e", "e", "i", "i", "i", "i", // E0
    "d", "n", "o", "o", "o", "o", "o", "", "o", "u", "u", "u", "u", "y", "th", "y", // F0
];

/// U+0100..U+017F.
const LATIN_EXT_A: [&str; 128] = [
    "a", "a", "a", "a", "a", "a", "c", "c", "c", "c", "c", "c", "c", "c", "d", "d", // 100
    "d", "d", "e", "e", "e", "e", "e", "e", "e", "e", "e", "e", "g", "g", "g", "g", // 110
    "g", "g", "g", "g", "h", "h", "h", "h", "i", "i", "i", "i", "i", "i", "i", "i", // 120
    "i", "i", "ij", "ij", "j", "j", "k", "k", "k", "l", "l", "l", "l", "l", "l", "l", // 130
    "l", "l", "l", "n", "n", "n", "n", "n", "n", "n", "n", "n", "o", "o", "o", "o", // 140
    "o", "o", "oe", "oe", "r", "r", "r", "r", "r", "r", "s", "s", "s", "s", "s", "s", // 150
    "s", "s", "t", "t", "t", "t", "t", "t", "u", "u", "u", "u", "u", "u", "u", "u", // 160
    "u", "u", "u", "u", "w", "w", "y", "y", "y", "z", "z", "z", "z", "z", "z", "s", // 170
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_accents_and_case_and_keeps_the_rest() {
        assert_eq!(fold("Relâmpago Vivo"), "relampago vivo");
        assert_eq!(fold("Jötun Grunt"), "jotun grunt");
        assert_eq!(fold("Æther Vial"), "aether vial");
        assert_eq!(fold("Łukasz Żółć"), "lukasz zolc");
        assert_eq!(fold("Straße × 3"), "strasse × 3");
        assert_eq!(fold("Ōkami 狼"), "okami 狼");
        assert_eq!(fold(""), "");
    }
}
