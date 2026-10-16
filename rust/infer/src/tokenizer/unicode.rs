//! The Unicode facts the normalizer asks, from the tables `build.rs`
//! generates out of the vendored character database (Unicode 16.0.0, see
//! `ucd/UPSTREAM`): category lookups by binary search over ranges, and
//! NFD by full canonical decomposition, Hangul by the algorithm, then
//! canonical ordering of the marks. The reference tokenizer carries older
//! tables (its categories and its NFD predate several Unicode versions),
//! so a character assigned since then can normalize differently; the
//! parity tests cover the scripts the corpora use.

mod tables {
    include!(concat!(env!("OUT_DIR"), "/ucd_tables.rs"));
}

fn in_ranges(table: &[(u32, u32)], c: char) -> bool {
    let x = c as u32;
    table
        .binary_search_by(|&(a, b)| {
            if x < a {
                std::cmp::Ordering::Greater
            } else if x > b {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// Cc, Cf or Co: what BERT's clean_text drops (tab, newline and carriage
/// return are whitespace to it, handled by the caller).
pub fn is_other(c: char) -> bool {
    in_ranges(tables::OTHER, c)
}

/// Mn: the nonspacing marks strip_accents removes after NFD.
pub fn is_mark_nonspacing(c: char) -> bool {
    in_ranges(tables::MARK_NONSPACING, c)
}

/// Pc, Pd, Pe, Pf, Pi, Po or Ps.
pub fn is_punctuation(c: char) -> bool {
    in_ranges(tables::PUNCTUATION, c)
}

fn combining_class(c: char) -> u8 {
    let x = c as u32;
    tables::COMBINING_CLASS
        .binary_search_by_key(&x, |&(code, _)| code)
        .map_or(0, |i| tables::COMBINING_CLASS[i].1)
}

const HANGUL_BASE: u32 = 0xAC00;
const HANGUL_COUNT: u32 = 11172;
const JAMO_L: u32 = 0x1100;
const JAMO_V: u32 = 0x1161;
const JAMO_T: u32 = 0x11A7;
const JAMO_T_COUNT: u32 = 28;
const JAMO_N_COUNT: u32 = 21 * JAMO_T_COUNT;

fn decompose_into(c: char, out: &mut Vec<char>) {
    let x = c as u32;
    if (HANGUL_BASE..HANGUL_BASE + HANGUL_COUNT).contains(&x) {
        let s = x - HANGUL_BASE;
        let l = JAMO_L + s / JAMO_N_COUNT;
        let v = JAMO_V + (s % JAMO_N_COUNT) / JAMO_T_COUNT;
        let t = JAMO_T + s % JAMO_T_COUNT;
        out.extend([l, v].into_iter().filter_map(char::from_u32));
        if t != JAMO_T {
            out.extend(char::from_u32(t));
        }
        return;
    }
    match tables::DECOMPOSITION.binary_search_by_key(&x, |&(code, _)| code) {
        Ok(i) => out.extend(
            tables::DECOMPOSITION[i]
                .1
                .iter()
                .filter_map(|&p| char::from_u32(p)),
        ),
        Err(_) => out.push(c),
    }
}

/// Canonical decomposition with the marks in canonical order.
pub fn nfd(s: &str) -> String {
    let mut chars: Vec<char> = Vec::with_capacity(s.len());
    for c in s.chars() {
        decompose_into(c, &mut chars);
    }
    // canonical ordering: a stable sort of every run of nonzero combining
    // class by that class, starters untouched.
    let mut i = 0;
    while i < chars.len() {
        if combining_class(chars[i]) == 0 {
            i += 1;
            continue;
        }
        let start = i;
        while i < chars.len() && combining_class(chars[i]) != 0 {
            i += 1;
        }
        chars[start..i].sort_by_key(|&c| combining_class(c));
    }
    chars.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categories_and_decomposition_match_the_standard() {
        assert!(is_other('\u{0}') && is_other('\u{200B}') && is_other('\n') && !is_other('a'));
        assert!(is_mark_nonspacing('\u{301}') && !is_mark_nonspacing('a'));
        assert!(
            is_punctuation('!')
                && is_punctuation('«')
                && is_punctuation('。')
                && !is_punctuation('a')
        );
        assert!(
            !is_punctuation('$'),
            "a symbol, not punctuation; ascii handles it"
        );
        assert_eq!(nfd("é"), "e\u{301}");
        assert_eq!(nfd("ự"), "u\u{31B}\u{323}", "marks in canonical order");
        assert_eq!(nfd("한"), "\u{1112}\u{1161}\u{11AB}");
        assert_eq!(nfd("Ω"), "Ω");
        assert_eq!(nfd("plain"), "plain");
    }
}
