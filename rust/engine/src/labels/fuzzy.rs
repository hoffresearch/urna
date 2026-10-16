//! How well a folded label answers a folded query: the tier says what kind
//! of match it is (the whole label, a prefix, a word prefix, a subsequence)
//! and the score ranks the labels inside a tier, favouring letters at the
//! start of a word, next to the previous letter and early in the label.

/// The trailing `(…)` of a label: `lightning bolt (raio)` -> `raio`.
pub fn alias(label: &str) -> Option<&str> {
    label
        .strip_suffix(')')
        .and_then(|l| l.rfind(" (").map(|i| &l[i + 2..]))
        .filter(|a| !a.is_empty())
}

/// 3 the whole label, its alias or the name before the alias; 2 a prefix
/// of the label or of its alias; 1 a prefix of one of its words; 0 nothing
/// of the kind.
pub fn tier(label: &str, q: &str) -> u8 {
    let alias = alias(label).unwrap_or("");
    // the name without its alias: `lightning bolt (raio)` -> `lightning bolt`.
    let name = if alias.is_empty() {
        label
    } else {
        label.get(..label.len() - alias.len() - 3).unwrap_or(label)
    };
    if q.is_empty() {
        0
    } else if label == q || alias == q || name == q {
        3
    } else if label.starts_with(q) || (!alias.is_empty() && alias.starts_with(q)) {
        2
    } else if label.split(separator).any(|w| w.starts_with(q)) {
        1
    } else {
        0
    }
}

/// What ends a word in a label: a space, a comma, a hyphen, the `(` of an
/// alias and the `/` of a split name.
pub(super) fn separator(c: char) -> bool {
    matches!(c, ' ' | ',' | '-' | '(' | '/')
}

/// The score `score` gives `q` found whole at byte `at` of a label, at the
/// start of a word: every letter adjacent to the one before, the first
/// scored by how early it is, the ones after a separator in `q` as word
/// starts. Computed from `q` alone.
pub(super) fn score_contiguous(q: &str, at: usize) -> u32 {
    let mut total: i64 = 0;
    let mut prev: Option<char> = None;
    for c in q.chars() {
        total += 1;
        if prev.is_none_or(separator) {
            total += 8;
        }
        match prev {
            None => total += 16 - i64::try_from(at.min(16)).unwrap_or(16),
            Some(_) => total += 4,
        }
        prev = Some(c);
    }
    u32::try_from(total.max(1)).unwrap_or(1)
}

/// The score of `q` as a subsequence of `label`, `None` when a letter of
/// `q` is not in `label` after the previous one. Each letter counts 1,
/// plus 8 at the start of a word, 4 right after the previous letter, and
/// up to 16 for the first letter the earlier it is; a gap of skipped
/// letters costs up to 4. Never below 1.
pub fn score(label: &str, q: &str) -> Option<u32> {
    let mut total: i64 = 0;
    let mut prev: Option<usize> = None;
    let mut chars = label.char_indices().peekable();
    let mut before: Option<char> = None;
    for want in q.chars() {
        let mut found = None;
        for (i, c) in chars.by_ref() {
            if c == want {
                found = Some((i, before));
                before = Some(c);
                break;
            }
            before = Some(c);
        }
        let (i, left) = found?;
        total += 1;
        if left.is_none_or(separator) {
            total += 8;
        }
        match prev {
            None => total += 16 - i64::try_from(i.min(16)).unwrap_or(16),
            Some(p) if i == p + 1 => total += 4,
            Some(p) => total -= i64::try_from(i - p - 1).unwrap_or(4).min(4),
        }
        prev = Some(i);
    }
    Some(u32::try_from(total.max(1)).unwrap_or(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiers_tell_whole_prefix_word_and_subsequence_apart() {
        assert_eq!(tier("lightning bolt (raio)", "raio"), 3);
        assert_eq!(tier("lightning bolt (raio)", "lightning bolt"), 3);
        assert_eq!(tier("lightning bolt (raio)", "lightning"), 2);
        assert_eq!(tier("lightning bolt (raio)", "ra"), 2);
        assert_eq!(tier("lightning bolt (raio)", "bolt"), 1);
        assert_eq!(
            tier("lightning bolt // lightning bolt", "lightning bolt"),
            2
        );
        assert_eq!(tier("lightning bolt", "lb"), 0);
        assert_eq!(tier("lightning bolt", ""), 0);
        assert_eq!(alias("lightning bolt (raio)"), Some("raio"));
        assert_eq!(alias("lightning bolt ()"), None);
        assert_eq!(alias("lightning bolt"), None);
    }

    #[test]
    fn the_score_prefers_word_starts_adjacency_and_early_letters() {
        assert!(score("lightning bolt", "lb").is_some());
        assert_eq!(score("lightning bolt", "xb"), None);
        assert_eq!(score("bolt", "bolts"), None);
        let whole = score("bolt", "bolt").unwrap_or(0);
        let spread = score("boring old lute", "bolt").unwrap_or(0);
        let late = score("the firebolt", "bolt").unwrap_or(0);
        assert!(whole > spread && whole > late, "{whole} {spread} {late}");
        assert!(score("lightning bolt", "li").unwrap_or(0) > score("flight", "li").unwrap_or(0));
        assert_eq!(score("", "a"), None);
        assert_eq!(score("a", ""), Some(1));
        for (label, q, at) in [
            ("bolt", "bolt", 0),
            ("lightning bolt", "bolt", 10),
            ("bolt // ice", "ice", 8),
            ("lightning bolt (raio)", "lightning b", 0),
        ] {
            assert_eq!(
                Some(score_contiguous(q, at)),
                score(label, q),
                "{label} {q}"
            );
        }
    }
}
