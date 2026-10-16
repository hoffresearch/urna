//! Type-ahead for the ask tab over the corpus's own labels: the first line
//! of each chunk's stored text (a card name, a title, a diagnosis). Ranked
//! by tier first (the whole label, a prefix of it, a prefix of one of its
//! words, then a fuzzy match), then by the matcher's score, then shorter.

use std::collections::HashSet;
use std::sync::Mutex;

use nucleo_matcher::pattern::{CaseMatching, Normalization, Pattern};
use nucleo_matcher::{Config, Matcher, Utf32String};

struct Label {
    text: String,
    /// lowercase, accents dropped, for the tiers.
    folded: String,
    hay: Utf32String,
}

pub struct Labels {
    items: Vec<Label>,
    matcher: Mutex<Matcher>,
}

/// `Relâmpago Vivo` -> `relampago vivo`.
pub fn fold(s: &str) -> String {
    s.chars()
        .map(nucleo_matcher::chars::normalize)
        .flat_map(char::to_lowercase)
        .collect()
}

/// The trailing `(…)` of a label: `Lightning Bolt (Raio)` -> `Raio`.
pub fn alias(label: &str) -> Option<&str> {
    label
        .strip_suffix(')')
        .and_then(|l| l.rfind(" (").map(|i| &l[i + 2..]))
        .filter(|a| !a.is_empty())
}

/// How well a folded label answers a folded query: 3 the whole label or its
/// alias (a trailing `(…)`, like a translated name), 2 a prefix of either,
/// 1 a prefix of one of its words, 0 nothing of the kind.
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
    } else if label.split([' ', ',', '-', '(']).any(|w| w.starts_with(q)) {
        1
    } else {
        0
    }
}

impl Labels {
    pub fn new<'a>(texts: impl Iterator<Item = &'a str>) -> Self {
        let mut seen = HashSet::new();
        let mut items: Vec<Label> = texts
            .filter_map(|t| t.lines().next().map(str::trim))
            .filter(|l| !l.is_empty() && seen.insert(l.to_string()))
            .map(|l| Label {
                text: l.to_string(),
                folded: fold(l),
                hay: Utf32String::from(l),
            })
            .collect();
        items.sort_by(|a, b| a.text.cmp(&b.text));
        Self {
            items,
            matcher: Mutex::new(Matcher::new(Config::DEFAULT)),
        }
    }

    pub fn suggest(&self, query: &str, k: usize) -> Vec<String> {
        let q = fold(query.trim());
        if q.is_empty() || k == 0 {
            return Vec::new();
        }
        let pattern = Pattern::parse(&q, CaseMatching::Ignore, Normalization::Smart);
        let mut matcher = match self.matcher.lock() {
            Ok(m) => m,
            Err(poisoned) => poisoned.into_inner(),
        };
        let mut scored: Vec<((u8, u32, i64), &Label)> = self
            .items
            .iter()
            .filter_map(|l| {
                let score = pattern.score(l.hay.slice(..), &mut matcher)?;
                Some(((tier(&l.folded, &q), score, -(l.folded.len() as i64)), l))
            })
            .collect();
        scored.sort_by_key(|(k, _)| std::cmp::Reverse(*k));
        scored
            .into_iter()
            .take(k)
            .map(|(_, l)| l.text.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_label_beats_prefix_beats_fuzzy_and_accents_fold() {
        let texts = [
            "Lightning Bolt (Raio)\n{R}",
            "Lightning\nx",
            "Relâmpago Vivo\ny",
            "Bolt of Light\nz",
        ];
        let l = Labels::new(texts.into_iter());
        assert_eq!(l.suggest("lightning", 1), vec!["Lightning"]);
        assert_eq!(l.suggest("relampago", 1), vec!["Relâmpago Vivo"]);
        assert_eq!(l.suggest("raio", 1), vec!["Lightning Bolt (Raio)"]);
        assert_eq!(tier("lightning bolt (raio)", "raio"), 3);
        assert_eq!(tier("lightning bolt (raio)", "lightning"), 2);
        assert_eq!(tier("lightning bolt (raio)", "lightning bolt"), 3);
        assert_eq!(
            tier("lightning bolt // lightning bolt", "lightning bolt"),
            2
        );
        assert!(l.suggest("", 5).is_empty());
    }
}
