//! Type-ahead for the ask tab over the corpus's own labels: the first line
//! of each chunk's stored text (a card name, a title, a diagnosis), or the
//! card names, English and Portuguese, when the corpus carries fields.
//! The engine's label index folds, stores and matches them; this module
//! ranks the hits by tier first (the whole label, a prefix of it, a prefix
//! of one of its words, then a fuzzy match), then by the score, then
//! shorter. With fields, the playable card leads its art-series twin, the
//! more played card (EDHREC rank) leads inside a name tier, and the many
//! tokens that share a name collapse into one line.

use std::collections::HashSet;

use urna_engine::labels::Index;
pub use urna_engine::labels::{alias, fold, tier};

use super::fields::{Card, Kind};

struct Label {
    text: String,
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
    /// in the index's order: a hit's id is a position here.
    items: Vec<Label>,
    index: Index,
}

impl Labels {
    /// Labels from the stored text in chunk order: the first line of each
    /// chunk, one label per distinct line. A label's ordinal is its position
    /// in `texts`, so the caller passes them in the order of the chunks.
    pub fn new<'a>(texts: impl Iterator<Item = &'a str>) -> Self {
        let mut seen = HashSet::new();
        let items = texts
            .enumerate()
            .filter_map(|(i, t)| t.lines().next().map(|l| (i, l.trim())))
            .filter(|(_, l)| !l.is_empty() && seen.insert(l.to_string()))
            .map(|(i, l)| {
                (
                    fold(l),
                    Label {
                        text: l.to_string(),
                        ordinal: i,
                        key: l.to_string(),
                        playable: true,
                        popularity: 0,
                    },
                )
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
                items.push((
                    fold(name),
                    Label {
                        text: shown.clone(),
                        ordinal: c.ordinal,
                        key: c.name.clone(),
                        playable: c.playable(),
                        popularity: c.edhrec_rank.map_or(0, |r| u32::MAX - r),
                    },
                ));
            }
        }
        Self::build(items)
    }

    fn build(mut items: Vec<(String, Label)>) -> Self {
        items.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.ordinal.cmp(&b.1.ordinal)));
        // sorted just above, so the index cannot refuse it.
        let index = Index::from_sorted(items.iter().map(|(f, l)| (f.as_str(), l.ordinal)))
            .unwrap_or_default();
        Self {
            items: items.into_iter().map(|(_, l)| l).collect(),
            index,
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
        if k == 0 {
            return Vec::new();
        }
        let rank = |h: &urna_engine::labels::Hit| {
            let l = self.items.get(h.id)?;
            let popularity = if h.tier > 0 { l.popularity } else { 0 };
            let len = i64::try_from(h.len).unwrap_or(i64::MAX);
            Some(((l.playable, h.tier, popularity, h.score, -len), l))
        };
        let exact = self.index.exact(query, keep);
        let mut scored: Vec<(Rank, &Label)> = exact.iter().filter_map(rank).collect();
        // a playable label that q is a prefix of, or of one of its words,
        // outranks every subsequence match: with k of them the scan over
        // all the labels has nothing to add.
        let mut names: HashSet<&str> = HashSet::new();
        let enough = scored
            .iter()
            .filter(|(r, l)| r.0 && r.1 > 0 && names.insert(l.key.as_str()))
            .count()
            >= k;
        if !enough {
            let seen: HashSet<usize> = exact.iter().map(|h| h.id).collect();
            let fuzzy = self.index.fuzzy(query, keep);
            scored.extend(
                fuzzy
                    .iter()
                    .filter(|h| !seen.contains(&h.id))
                    .filter_map(rank),
            );
        }
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
        assert_eq!(l.suggest("bolt light", 1), vec!["Bolt of Light"]);
        assert_eq!(
            l.suggest("bolt", 2),
            vec!["Bolt of Light", "Lightning Bolt (Raio)"]
        );
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

    #[test]
    fn cards_offer_both_names_once_with_the_played_one_first() {
        let card = |ordinal: usize, name: &str, pt: Option<&str>, kind: &str, rank: Option<u32>| {
            let pt = pt.map_or("null".to_string(), |p| format!("\"{p}\""));
            let rank = rank.map_or("null".to_string(), |r| r.to_string());
            let json = format!(
                r#"{{"ordinal":{ordinal},"name":"{name}","pt_name":{pt},"type_line":"Instant",
                "kind":"{kind}","cmc":1.0,"power":null,"toughness":null,"loyalty":null,
                "legal":{{}},"edhrec_rank":{rank}}}"#
            );
            serde_json::from_str::<Card>(&json).ok()
        };
        let cards: Vec<Card> = [
            card(0, "Lightning Bolt", Some("Raio"), "card", Some(10)),
            card(1, "Lightning Bolt", None, "art_series", None),
            card(
                2,
                "Lightning Helix",
                Some("Hélice de Raios"),
                "card",
                Some(300),
            ),
            card(3, "Lightning Greaves", None, "card", Some(1)),
            card(4, "Lightning", None, "token", None),
        ]
        .into_iter()
        .flatten()
        .collect();
        let l = Labels::from_cards(&cards);
        assert_eq!(
            l.suggest("lightn", 5),
            vec![
                "Lightning Greaves",
                "Lightning Bolt (Raio)",
                "Lightning Helix (Hélice de Raios)",
                "Lightning",
            ]
        );
        assert_eq!(l.suggest("raio", 1), vec!["Lightning Bolt (Raio)"]);
        assert_eq!(
            l.suggest("helice", 1),
            vec!["Lightning Helix (Hélice de Raios)"]
        );
        assert_eq!(
            l.suggest_where("lightning bolt", 5, &|o| o != 0),
            Vec::<String>::new()
        );
    }
}
