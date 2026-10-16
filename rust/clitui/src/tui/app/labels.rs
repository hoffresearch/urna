//! Type-ahead for the ask tab over the corpus's own labels: the first line
//! of each chunk's stored text (a card name, a title, a diagnosis), or the
//! card names, English and Portuguese, when the corpus carries fields.
//! Ranked by tier first (the whole label, a prefix of it, a prefix of one
//! of its words, then a fuzzy match), then by the matcher's score, then
//! shorter. With fields, the playable card leads its art-series twin, the
//! more played card (EDHREC rank) leads inside a name tier, and the many
//! tokens that share a name collapse into one line.

use std::collections::HashSet;
use std::sync::Mutex;

use nucleo_matcher::pattern::{CaseMatching, Normalization, Pattern};
use nucleo_matcher::{Config, Matcher, Utf32String};

use super::fields::{Card, Kind};

struct Label {
    text: String,
    /// lowercase, accents dropped, for the tiers.
    folded: String,
    hay: Utf32String,
    /// the chunk the label names, for the caller's filter.
    ordinal: usize,
    /// what one line stands for: the card's name, so its twins collapse.
    key: String,
    playable: bool,
    /// `u32::MAX - edhrec_rank`, so higher sorts first; 0 when unranked.
    popularity: u32,
}

/// (playable, tier, popularity in name tiers, score, -len): sorted descending.
type Rank = (bool, u8, u32, u32, i64);

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
    /// Labels from the stored text in chunk order: the first line of each
    /// chunk, one label per distinct line. A label's ordinal is its position
    /// in `texts`, so the caller passes them in the order of the chunks.
    pub fn new<'a>(texts: impl Iterator<Item = &'a str>) -> Self {
        let mut seen = HashSet::new();
        let items: Vec<Label> = texts
            .enumerate()
            .filter_map(|(i, t)| t.lines().next().map(|l| (i, l.trim())))
            .filter(|(_, l)| !l.is_empty() && seen.insert(l.to_string()))
            .map(|(i, l)| Label {
                text: l.to_string(),
                folded: fold(l),
                hay: Utf32String::from(l),
                ordinal: i,
                key: l.to_string(),
                playable: true,
                popularity: 0,
            })
            .collect();
        Self::build(items)
    }

    /// Labels from the card fields: the English name and the Portuguese one
    /// of every card that belongs in a name list (art-series repeat a real
    /// card's name and Alchemy cards are digital).
    pub fn from_cards(cards: &[Card]) -> Self {
        let mut items = Vec::with_capacity(cards.len() * 2);
        for c in cards
            .iter()
            .filter(|c| !matches!(c.kind, Kind::ArtSeries | Kind::Alchemy))
        {
            let shown = match &c.pt_name {
                Some(pt) => format!("{} ({pt})", c.name),
                None => c.name.clone(),
            };
            let names = std::iter::once(c.name.as_str()).chain(c.pt_name.as_deref());
            for name in names {
                items.push(Label {
                    text: shown.clone(),
                    folded: fold(name),
                    hay: Utf32String::from(name),
                    ordinal: c.ordinal,
                    key: c.name.clone(),
                    playable: c.playable(),
                    popularity: c.edhrec_rank.map_or(0, |r| u32::MAX - r),
                });
            }
        }
        Self::build(items)
    }

    fn build(mut items: Vec<Label>) -> Self {
        items.sort_by(|a, b| a.text.cmp(&b.text).then(a.ordinal.cmp(&b.ordinal)));
        Self {
            items,
            matcher: Mutex::new(Matcher::new(Config::DEFAULT)),
        }
    }

    #[cfg(test)]
    pub fn suggest(&self, query: &str, k: usize) -> Vec<String> {
        self.suggest_where(query, k, &|_| true)
    }

    /// The best `k` labels for `query` among the chunks `keep` accepts.
    pub fn suggest_where(
        &self,
        query: &str,
        k: usize,
        keep: &dyn Fn(usize) -> bool,
    ) -> Vec<String> {
        let q = fold(query.trim());
        if q.is_empty() || k == 0 {
            return Vec::new();
        }
        let pattern = Pattern::parse(&q, CaseMatching::Ignore, Normalization::Smart);
        let mut matcher = match self.matcher.lock() {
            Ok(m) => m,
            Err(poisoned) => poisoned.into_inner(),
        };
        let mut scored: Vec<(Rank, &Label)> = self
            .items
            .iter()
            .filter(|l| keep(l.ordinal))
            .filter_map(|l| {
                let score = pattern.score(l.hay.slice(..), &mut matcher)?;
                let t = tier(&l.folded, &q);
                let popularity = if t > 0 { l.popularity } else { 0 };
                Some((
                    (l.playable, t, popularity, score, -(l.folded.len() as i64)),
                    l,
                ))
            })
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.ordinal.cmp(&b.1.ordinal)));
        let mut seen: HashSet<&str> = HashSet::new();
        scored
            .into_iter()
            .filter(|(_, l)| seen.insert(l.key.as_str()))
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
        assert_eq!(
            l.suggest_where("lightning", 5, &|o| o != 1),
            vec!["Lightning Bolt (Raio)"]
        );
    }
}
